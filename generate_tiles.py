#!/usr/bin/env python3
"""
generate_tiles.py

Read cached NDVI/NBR arrays from cache/<mine-slug>/ and write XYZ PNG tiles to
site/public/tiles/<mine-slug>/.  No network calls — reads only local .npy files.

Outputs
-------
  site/public/tiles/<mine-slug>/ndvi/{year}/{z}/{x}/{y}.png
  site/public/tiles/<mine-slug>/nbr/{year}/{z}/{x}/{y}.png
  site/public/tiles/<mine-slug>/manifest.json   ← bounds, years, zoom_levels

Usage
-----
  python generate_tiles.py --mine hobet-mine
  python generate_tiles.py --mine "Hobet Mine"
"""

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import mercantile
import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
CACHE_DIR    = HERE / "cache"
METADATA_CSV = HERE / "mine-metadata.csv"

# ── Tile / grid settings ──────────────────────────────────────────────────────
TILE_ZOOMS = [9, 10, 11, 12]
TILE_SIZE  = 256
OUT_RES    = 0.00027   # ~30 m in EPSG:4326
OUT_CRS    = CRS.from_epsg(4326)
EPSG3857   = CRS.from_epsg(3857)

# ── Colormaps (match mine_reclamation_timeseries.py) ─────────────────────────
NDVI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ndvi",
    [(0.0, "#8B5E3C"), (0.2, "#C8A96E"), (0.4, "#FFFFCC"),
     (0.6, "#78C679"), (1.0, "#006837")],
)
NBR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nbr",
    [(0.0, "#7B3F00"), (0.2, "#D4956A"), (0.4, "#F5F5DC"),
     (0.6, "#5DA35D"), (1.0, "#1A4731")],
)

NDVI_VMIN, NDVI_VMAX = -0.1, 0.8
NBR_VMIN,  NBR_VMAX  = -0.3, 0.8


# ── Mine metadata ─────────────────────────────────────────────────────────────

def mine_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_mines(csv_path: Path) -> dict:
    mines = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mines[mine_slug(row["mine_name"])] = row
    return mines


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_tile(
    array: np.ndarray,
    tile: mercantile.Tile,
    out_transform,
    out_crs,
) -> np.ndarray:
    """Reproject a EPSG:4326 float32 array into a 256×256 Web Mercator tile."""
    tb  = mercantile.xy_bounds(tile)
    dst = np.full((TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
    dst_transform = from_bounds(
        tb.left, tb.bottom, tb.right, tb.top, TILE_SIZE, TILE_SIZE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reproject(
            source=array,
            destination=dst,
            src_transform=out_transform,
            src_crs=out_crs,
            dst_transform=dst_transform,
            dst_crs=EPSG3857,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return dst


def save_tile_png(array: np.ndarray, cmap, vmin: float, vmax: float, path: Path) -> bool:
    """Colormap array → RGBA PNG. Returns False and skips if all pixels are NaN."""
    if np.all(np.isnan(array)):
        return False
    norm    = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba    = cmap(norm(np.nan_to_num(array, nan=vmin)))
    rgba_u8 = (rgba * 255).astype(np.uint8)
    rgba_u8[np.isnan(array), 3] = 0
    if rgba_u8[:, :, 3].max() == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba_u8, "RGBA").save(path)
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate XYZ map tiles from cached NDVI/NBR arrays."
    )
    parser.add_argument(
        "--mine", required=True, metavar="NAME_OR_SLUG",
        help="Mine slug or name to tile (see mine-metadata.csv)",
    )
    args = parser.parse_args()

    mines = load_mines(METADATA_CSV)
    slug  = mine_slug(args.mine)
    if slug not in mines:
        raise SystemExit(
            f"Mine '{args.mine}' not found. Available: {', '.join(mines)}"
        )

    mine   = mines[slug]
    bounds = (
        float(mine["bounds_west"]),
        float(mine["bounds_south"]),
        float(mine["bounds_east"]),
        float(mine["bounds_north"]),
    )
    out_width     = int(round((bounds[2] - bounds[0]) / OUT_RES))
    out_height    = int(round((bounds[3] - bounds[1]) / OUT_RES))
    out_transform = from_bounds(*bounds, out_width, out_height)

    cache_dir = CACHE_DIR / slug
    site_dir  = HERE / "site" / "public" / "tiles" / slug
    site_dir.mkdir(parents=True, exist_ok=True)

    ndvi_years = sorted(
        int(p.stem.split("_")[1]) for p in cache_dir.glob("ndvi_*.npy")
    )
    if not ndvi_years:
        raise SystemExit(f"No ndvi_*.npy files found in {cache_dir}")

    tile_list = list(mercantile.tiles(*bounds, zooms=TILE_ZOOMS))
    print(f"Mine:           {mine['mine_name']}")
    print(f"Tiles per year: {len(tile_list)} across zooms {TILE_ZOOMS}")
    print(f"Years with NDVI cache: {len(ndvi_years)}  ({ndvi_years[0]}–{ndvi_years[-1]})")
    print(f"Output → {site_dir}\n")

    manifest_years: list = []

    for year in tqdm(ndvi_years, desc="Years"):
        ndvi_path = cache_dir / f"ndvi_{year}.npy"
        nbr_path  = cache_dir / f"nbr_{year}.npy"

        ndvi = np.load(ndvi_path)
        nbr  = np.load(nbr_path) if nbr_path.exists() else None

        manifest_years.append(year)

        bands = [("ndvi", ndvi, NDVI_CMAP, NDVI_VMIN, NDVI_VMAX)]
        if nbr is not None:
            bands.append(("nbr", nbr, NBR_CMAP, NBR_VMIN, NBR_VMAX))

        saved = 0
        for tile in tile_list:
            for name, array, cmap, vmin, vmax in bands:
                out = (
                    site_dir / name / str(year) /
                    str(tile.z) / str(tile.x) / f"{tile.y}.png"
                )
                if out.exists():
                    continue
                data = render_tile(array, tile, out_transform, OUT_CRS)
                if save_tile_png(data, cmap, vmin, vmax, out):
                    saved += 1

        tqdm.write(f"  {year}: {saved} tiles written")

    manifest = {
        "mine":        mine["mine_name"],
        "bounds":      list(bounds),
        "years":       manifest_years,
        "zoom_levels": TILE_ZOOMS,
    }
    (site_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_tiles = sum(1 for _ in site_dir.rglob("*.png"))
    print(f"\nDone — {len(manifest_years)} years, {total_tiles} tiles total")
    print(f"Copy to your Next.js project:")
    print(f"  {site_dir}  →  <your-site>/public/tiles/{slug}/")


if __name__ == "__main__":
    main()
