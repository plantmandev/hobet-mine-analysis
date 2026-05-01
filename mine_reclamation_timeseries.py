#!/usr/bin/env python3
"""
mine_reclamation_timeseries.py

Annual NDVI + NBR composites for mine reclamation study areas using
Landsat C2 L2 from Microsoft Planetary Computer (free, no account required).

Composites use May–September only (peak vegetation / full leaf-out).
Landsat 7 is excluded after 2003 due to the scan-line corrector failure
that causes permanent striped data gaps.

Outputs (per mine, relative to this file):
  outputs/<mine-slug>/ndvi/ndvi_YYYY.png   color-mapped NDVI RGBA PNG
  outputs/<mine-slug>/nbr/nbr_YYYY.png     color-mapped NBR  RGBA PNG
  cache/<mine-slug>/ndvi_YYYY.npy          raw NDVI float32 arrays (for reruns)
  cache/<mine-slug>/nbr_YYYY.npy           raw NBR  float32 arrays

Usage:
  python mine_reclamation_timeseries.py                          # all mines
  python mine_reclamation_timeseries.py --mine hobet-mine        # one mine by slug
  python mine_reclamation_timeseries.py --mine "Hobet Mine"      # or by name

Install:
  pip install pystac-client planetary-computer rasterio numpy matplotlib Pillow tqdm
"""

import argparse
import calendar
import csv
import logging
import re
import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
import pystac_client
import planetary_computer
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
OUTPUT_DIR   = HERE / "outputs"
CACHE_DIR    = HERE / "cache"
METADATA_CSV = HERE / "mine-metadata.csv"

# ── Output grid resolution (~30 m in EPSG:4326) ───────────────────────────────
OUT_RES = 0.00027

# ── Processing parameters ─────────────────────────────────────────────────────
YEARS          = list(range(1985, 2026))
CLOUD_THRESH   = 20    # scene-level pre-filter: skip scenes >20% cloudy (full footprint)
AOI_CLEAR_THRESH = 0.50  # per-AOI QA pre-screen: skip scenes where <50% of study area is clear
MAX_SCENES     = 40    # max scenes per year after screening, sorted least-cloudy first
MIN_CLEAR      = 2     # minimum clear pixel observations needed to keep a composite value
SEASON_START   = 5     # May  ─┐ peak vegetation / full leaf-out
SEASON_END     = 9     # Sep  ─┘

# ── Landsat band assets on Planetary Computer ─────────────────────────────────
SR_SCALE  = 0.0000275
SR_OFFSET = -0.2

PLATFORM_BANDS = {
    "landsat-5": {"red": "red", "nir": "nir08", "swir": "swir22", "qa": "qa_pixel"},
    "landsat-7": {"red": "red", "nir": "nir08", "swir": "swir22", "qa": "qa_pixel"},
    "landsat-8": {"red": "red", "nir": "nir08", "swir": "swir22", "qa": "qa_pixel"},
    "landsat-9": {"red": "red", "nir": "nir08", "swir": "swir22", "qa": "qa_pixel"},
}

# ── Colormaps ─────────────────────────────────────────────────────────────────
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


# ── Grid ──────────────────────────────────────────────────────────────────────

RasterGrid = namedtuple("RasterGrid", ["height", "width", "transform", "crs"])


def make_grid(bounds: tuple) -> RasterGrid:
    w = int(round((bounds[2] - bounds[0]) / OUT_RES))
    h = int(round((bounds[3] - bounds[1]) / OUT_RES))
    return RasterGrid(h, w, from_bounds(*bounds, w, h), CRS.from_epsg(4326))


# ── Mine metadata ─────────────────────────────────────────────────────────────

def mine_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_mines(csv_path: Path) -> dict:
    mines = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mines[mine_slug(row["mine_name"])] = row
    return mines


# ── Raster I/O ────────────────────────────────────────────────────────────────

def _reproject_band(href: str, dtype: type, nodata, grid: RasterGrid) -> np.ndarray:
    dst = np.full((grid.height, grid.width), nodata, dtype=dtype)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.bilinear if dtype == np.float32 else Resampling.nearest,
            src_nodata=0,
            dst_nodata=nodata,
        )
    return dst


def read_band_to_grid(href: str, grid: RasterGrid) -> np.ndarray:
    return _reproject_band(href, np.float32, np.nan, grid)


def read_qa_to_grid(href: str, grid: RasterGrid) -> np.ndarray:
    return _reproject_band(href, np.uint16, 0, grid)


def fetch_reflectance(hrefs: dict, grid: RasterGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download red, nir, swir in parallel (3 concurrent HTTP reads)."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_red  = pool.submit(read_band_to_grid, hrefs["red"],  grid)
        f_nir  = pool.submit(read_band_to_grid, hrefs["nir"],  grid)
        f_swir = pool.submit(read_band_to_grid, hrefs["swir"], grid)
    return f_red.result(), f_nir.result(), f_swir.result()


# ── Index computation ─────────────────────────────────────────────────────────

def cloud_mask(qa: np.ndarray) -> np.ndarray:
    """
    True for clear pixels. Landsat C2 QA_PIXEL bit flags (0-indexed):
      bit 1  — dilated cloud
      bit 2  — cirrus             (LS8/9; zero on LS5/7)
      bit 3  — cloud
      bit 4  — cloud shadow
      bit 5  — snow
      bits  8-9  — cloud confidence      (reject medium=2 or high=3)
      bits 10-11 — cloud shadow confidence (reject medium or high)
      bits 14-15 — cirrus confidence     (reject medium or high; zero on LS5/7)
    """
    cloud_conf  = (qa >>  8) & 3
    shadow_conf = (qa >> 10) & 3
    cirrus_conf = (qa >> 14) & 3
    return (
        ((qa & 2)  == 0) &     # dilated cloud
        ((qa & 4)  == 0) &     # cirrus
        ((qa & 8)  == 0) &     # cloud
        ((qa & 16) == 0) &     # cloud shadow
        ((qa & 32) == 0) &     # snow
        (cloud_conf  <= 1) &   # low or unset cloud confidence
        (shadow_conf <= 1) &   # low or unset shadow confidence
        (cirrus_conf <= 1)     # low or unset cirrus confidence
    )


def aoi_clear_fraction(qa: np.ndarray) -> float:
    """Fraction of non-fill AOI pixels that pass the cloud mask."""
    valid = qa != 0
    if not valid.any():
        return 0.0
    return float(cloud_mask(qa)[valid].sum()) / float(valid.sum())


def compute_indices(
    red_dn: np.ndarray, nir_dn: np.ndarray, swir_dn: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    NDVI = (NIR - red)  / (NIR + red)   — vegetation greenness
    NBR  = (NIR - SWIR) / (NIR + SWIR)  — burn / disturbance recovery
    """
    red  = red_dn  * SR_SCALE + SR_OFFSET
    nir  = nir_dn  * SR_SCALE + SR_OFFSET
    swir = swir_dn * SR_SCALE + SR_OFFSET
    nodata = (red_dn == 0) | (nir_dn == 0) | (swir_dn == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi = (nir - red)  / (nir + red)
        nbr  = (nir - swir) / (nir + swir)
    ndvi = np.where(nodata | (ndvi < -1) | (ndvi > 1), np.nan, ndvi)
    nbr  = np.where(nodata | (nbr  < -1) | (nbr  > 1), np.nan, nbr)
    return ndvi.astype(np.float32), nbr.astype(np.float32)


# ── Output writers ────────────────────────────────────────────────────────────

def _colormap_png(array: np.ndarray, cmap, vmin: float, vmax: float, path: Path) -> None:
    """RGBA PNG with transparency where NaN."""
    norm    = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba    = cmap(norm(np.nan_to_num(array, nan=vmin)))
    rgba_u8 = (rgba * 255).astype(np.uint8)
    rgba_u8[np.isnan(array), 3] = 0
    Image.fromarray(rgba_u8, mode="RGBA").save(path)


def save_outputs(
    ndvi: np.ndarray, nbr: np.ndarray, year: int,
    ndvi_dir: Path, nbr_dir: Path, cache_dir: Path,
) -> None:
    _colormap_png(ndvi, NDVI_CMAP, NDVI_VMIN, NDVI_VMAX, ndvi_dir / f"ndvi_{year}.png")
    _colormap_png(nbr,  NBR_CMAP,  NBR_VMIN,  NBR_VMAX,  nbr_dir  / f"nbr_{year}.png")
    np.save(cache_dir / f"ndvi_{year}.npy", ndvi)
    np.save(cache_dir / f"nbr_{year}.npy",  nbr)


# ── Core processing ───────────────────────────────────────────────────────────

def process_year(
    catalog, year: int, bounds: tuple, grid: RasterGrid
) -> tuple[np.ndarray | None, np.ndarray | None, int, list[str]]:
    """Returns (ndvi, nbr, n_scenes, platforms). Both arrays None if no scenes found."""
    last_day = calendar.monthrange(year, SEASON_END)[1]
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=list(bounds),
        datetime=f"{year}-{SEASON_START:02d}-01/{year}-{SEASON_END:02d}-{last_day:02d}",
    )
    # Scene-level pre-filter: exclude grossly cloudy scenes and broken LS7
    candidates = sorted(
        (
            i for i in search.items()
            if i.properties.get("eo:cloud_cover", 100) < CLOUD_THRESH
            # LS7 scan-line corrector failed May 2003 — stripes degrade composites
            and not (i.properties.get("platform") == "landsat-7" and year > 2003)
        ),
        key=lambda i: i.properties.get("eo:cloud_cover", 100),
    )

    if not candidates:
        return None, None, 0, []

    # Sign URLs and drop unknown platforms
    signed_candidates = []
    for item in candidates:
        signed = planetary_computer.sign(item)
        platform = signed.properties.get("platform", "")
        if platform in PLATFORM_BANDS:
            signed_candidates.append((signed, platform))

    # ── Phase 1: QA pre-screen ────────────────────────────────────────────────
    # eo:cloud_cover is measured over the full ~170×185 km Landsat footprint,
    # not the study area. Fetch QA for all candidates in parallel, then reject
    # any scene where fewer than AOI_CLEAR_THRESH of AOI pixels are actually clear.
    with ThreadPoolExecutor(max_workers=8) as pool:
        qa_futures = [
            (signed, platform, pool.submit(
                read_qa_to_grid,
                signed.assets[PLATFORM_BANDS[platform]["qa"]].href,
                grid,
            ))
            for signed, platform in signed_candidates
        ]
    # pool.__exit__ blocks until all QA downloads finish

    accepted: list[tuple] = []   # (signed, platform, qa_array)
    for signed, platform, future in qa_futures:
        try:
            qa = future.result()
        except Exception as exc:
            log.warning("QA fetch failed %s: %s", signed.id, exc)
            continue
        if aoi_clear_fraction(qa) >= AOI_CLEAR_THRESH:
            accepted.append((signed, platform, qa))
        if len(accepted) == MAX_SCENES:
            break

    if not accepted:
        return None, None, 0, []

    # ── Phase 2: Reflectance bands for accepted scenes ─────────────────────────
    ndvi_stack: list[np.ndarray] = []
    nbr_stack:  list[np.ndarray] = []
    platforms:  list[str]        = []

    for signed, platform, qa in accepted:
        bk = PLATFORM_BANDS[platform]
        try:
            red_dn, nir_dn, swir_dn = fetch_reflectance(
                {
                    "red":  signed.assets[bk["red"]].href,
                    "nir":  signed.assets[bk["nir"]].href,
                    "swir": signed.assets[bk["swir"]].href,
                },
                grid,
            )
        except Exception as exc:
            log.warning("Reflectance fetch failed %s: %s", signed.id, exc)
            continue

        ndvi, nbr = compute_indices(red_dn, nir_dn, swir_dn)
        clear = cloud_mask(qa)
        ndvi[~clear] = np.nan
        nbr[~clear]  = np.nan

        ndvi_stack.append(ndvi)
        nbr_stack.append(nbr)
        platforms.append(platform)

    if not ndvi_stack:
        return None, None, 0, []

    stack_ndvi = np.stack(ndvi_stack, axis=0)
    stack_nbr  = np.stack(nbr_stack,  axis=0)

    # pixels with fewer than MIN_CLEAR valid observations are discarded
    clear_count = np.sum(~np.isnan(stack_ndvi), axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi_out = np.nanmedian(stack_ndvi, axis=0).astype(np.float32)
        nbr_out  = np.nanmedian(stack_nbr,  axis=0).astype(np.float32)

    ndvi_out[clear_count < MIN_CLEAR] = np.nan
    nbr_out[clear_count  < MIN_CLEAR] = np.nan

    return ndvi_out, nbr_out, len(ndvi_stack), platforms


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate annual NDVI/NBR composites for mine reclamation sites."
    )
    parser.add_argument(
        "--mine", metavar="NAME_OR_SLUG",
        help="Process only this mine (default: all mines in mine-metadata.csv)",
    )
    args = parser.parse_args()

    mines = load_mines(METADATA_CSV)

    if args.mine:
        slug = mine_slug(args.mine)
        if slug not in mines:
            raise SystemExit(
                f"Mine '{args.mine}' not found. Available: {', '.join(mines)}"
            )
        selected = {slug: mines[slug]}
    else:
        selected = mines

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    for slug, mine in selected.items():
        bounds = (
            float(mine["bounds_west"]),
            float(mine["bounds_south"]),
            float(mine["bounds_east"]),
            float(mine["bounds_north"]),
        )
        grid = make_grid(bounds)

        ndvi_dir  = OUTPUT_DIR / slug / "ndvi"
        nbr_dir   = OUTPUT_DIR / slug / "nbr"
        cache_dir = CACHE_DIR  / slug

        ndvi_dir.mkdir(parents=True, exist_ok=True)
        nbr_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── {mine['mine_name']} ({mine['county']} Co., {mine['state']}) ──")
        print(f"   bounds: {bounds}  grid: {grid.width}×{grid.height} px")

        for year in tqdm(YEARS, desc=f"  {slug}"):
            ndvi_png = ndvi_dir  / f"ndvi_{year}.png"
            nbr_png  = nbr_dir   / f"nbr_{year}.png"
            ndvi_npy = cache_dir / f"ndvi_{year}.npy"
            nbr_npy  = cache_dir / f"nbr_{year}.npy"

            if ndvi_png.exists() and nbr_png.exists() and ndvi_npy.exists() and nbr_npy.exists():
                tqdm.write(f"  {year}: cached")
                continue

            ndvi, nbr, n_scenes, platforms = process_year(catalog, year, bounds, grid)
            if ndvi is None:
                tqdm.write(f"  {year}: no valid scenes — skipped")
                continue

            save_outputs(ndvi, nbr, year, ndvi_dir, nbr_dir, cache_dir)
            tqdm.write(
                f"  {year}: {n_scenes} scenes, {int((~np.isnan(ndvi)).sum()):,} valid px"
                + (f"  [{', '.join(sorted(set(platforms)))}]" if platforms else "")
            )

        print(f"   → outputs/{slug}/")


if __name__ == "__main__":
    main()
