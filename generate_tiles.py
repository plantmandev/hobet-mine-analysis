#!/usr/bin/env python3
"""
generate_tiles.py

Read cached NDVI/NBR arrays from cache/ and write XYZ PNG tiles to
site/public/tiles/.  No network calls — reads only local .npy files.

Outputs
-------
  site/public/tiles/ndvi/{year}/{z}/{x}/{y}.png
  site/public/tiles/nbr/{year}/{z}/{x}/{y}.png
  site/public/tiles/manifest.json   ← bounds, years, zoom_levels
  site/public/tiles/stats.json      ← per-year mean NDVI/NBR/pct_valid

Usage
-----
  python generate_tiles.py
"""

import json
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
HERE      = Path(__file__).parent
CACHE_DIR = HERE / "cache"
SITE_DIR  = HERE / "site" / "public" / "tiles"

# ── Study area (Hobet Mine, WV — WRS-2 path 19, row 34) ──────────────────────
BOUNDS = (-82.10, 37.85, -81.60, 38.15)   # west, south, east, north

OUT_RES       = 0.00027                   # ~30 m in EPSG:4326
OUT_WIDTH     = int(round((BOUNDS[2] - BOUNDS[0]) / OUT_RES))
OUT_HEIGHT    = int(round((BOUNDS[3] - BOUNDS[1]) / OUT_RES))
OUT_TRANSFORM = from_bounds(*BOUNDS, OUT_WIDTH, OUT_HEIGHT)
OUT_CRS       = CRS.from_epsg(4326)
EPSG3857      = CRS.from_epsg(3857)

# ── Tile settings ─────────────────────────────────────────────────────────────
TILE_ZOOMS = [9, 10, 11, 12]
TILE_SIZE  = 256

# ── Colormaps (match hobet_ndvi_timeseries.py) ────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_tile(array: np.ndarray, tile: mercantile.Tile) -> np.ndarray:
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
            src_transform=OUT_TRANSFORM,
            src_crs=OUT_CRS,
            dst_transform=dst_transform,
            dst_crs=EPSG3857,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return dst


def save_tile_png(
    array: np.ndarray,
    cmap,
    vmin: float,
    vmax: float,
    path: Path,
) -> bool:
    """Colormap array → RGBA PNG.  Returns False and skips if all pixels are NaN."""
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


def year_stats(ndvi: np.ndarray, nbr: np.ndarray | None) -> dict:
    v_ndvi = ndvi[~np.isnan(ndvi)]
    result: dict = {
        "mean_ndvi":  round(float(np.mean(v_ndvi)),   4) if len(v_ndvi) else None,
        "std_ndvi":   round(float(np.std(v_ndvi)),    4) if len(v_ndvi) else None,
        "pct_valid":  round(100 * len(v_ndvi) / ndvi.size, 2),
        "mean_nbr":   None,
    }
    if nbr is not None:
        v_nbr = nbr[~np.isnan(nbr)]
        result["mean_nbr"] = round(float(np.mean(v_nbr)), 4) if len(v_nbr) else None
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    # Years that actually have NDVI cache files
    ndvi_years = sorted(
        int(p.stem.split("_")[1]) for p in CACHE_DIR.glob("ndvi_*.npy")
    )
    if not ndvi_years:
        raise SystemExit(f"No ndvi_*.npy files found in {CACHE_DIR}")

    tile_list = list(mercantile.tiles(*BOUNDS, zooms=TILE_ZOOMS))
    print(f"Tiles per year: {len(tile_list)} across zooms {TILE_ZOOMS}")
    print(f"Years with NDVI cache: {len(ndvi_years)}  ({ndvi_years[0]}–{ndvi_years[-1]})")
    print(f"Output → {SITE_DIR}\n")

    stats: dict           = {}
    manifest_years: list  = []

    for year in tqdm(ndvi_years, desc="Years"):
        ndvi_path = CACHE_DIR / f"ndvi_{year}.npy"
        nbr_path  = CACHE_DIR / f"nbr_{year}.npy"

        ndvi = np.load(ndvi_path)
        nbr  = np.load(nbr_path) if nbr_path.exists() else None

        manifest_years.append(year)
        stats[str(year)] = year_stats(ndvi, nbr)

        bands = [("ndvi", ndvi, NDVI_CMAP, NDVI_VMIN, NDVI_VMAX)]
        if nbr is not None:
            bands.append(("nbr", nbr, NBR_CMAP, NBR_VMIN, NBR_VMAX))

        saved = 0
        for tile in tile_list:
            for name, array, cmap, vmin, vmax in bands:
                out = (
                    SITE_DIR / name / str(year) /
                    str(tile.z) / str(tile.x) / f"{tile.y}.png"
                )
                if out.exists():
                    continue
                data = render_tile(array, tile)
                if save_tile_png(data, cmap, vmin, vmax, out):
                    saved += 1

        tqdm.write(f"  {year}: {saved} tiles written")

    # ── Manifests ─────────────────────────────────────────────────────────────
    manifest = {
        "bounds":      list(BOUNDS),
        "years":       manifest_years,
        "zoom_levels": TILE_ZOOMS,
    }
    (SITE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (SITE_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    total_tiles = sum(
        1 for p in SITE_DIR.rglob("*.png")
    )
    print(f"\nDone — {len(manifest_years)} years, {total_tiles} tiles total")
    print("Copy site/ contents into your Next.js project root:")
    print("  site/public/tiles/  →  <your-site>/public/tiles/")
    print("  site/hobet-mine-analysis/  →  <your-site>/app/hobet-mine-analysis/")


if __name__ == "__main__":
    main()
