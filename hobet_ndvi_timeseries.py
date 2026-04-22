#!/usr/bin/env python3
"""
hobet_ndvi_timeseries.py

Annual NDVI + NBR composites for the Hobet Mine study area (WV) using
Landsat C2 L2 from Microsoft Planetary Computer (free, no account required).

Composites use May–September only (peak vegetation / full leaf-out in WV
deciduous forest).  Landsat 7 is excluded after 2003 due to the scan-line
corrector failure that causes permanent striped data gaps.

Outputs (all relative to this file):
  outputs/ndvi_YYYY.png        color-mapped NDVI for deck.gl BitmapLayer
  outputs/nbr_YYYY.png         color-mapped NBR  for deck.gl BitmapLayer
  outputs/ndvi_diff.png        dNDVI: after (2014-2020) minus before (1985-2005)
  outputs/ndvi_timeseries.png  mean NDVI per year with ±1σ band
  outputs/stats.json           per-year statistics for the deck.gl chart
  outputs/manifest.json        bounds + file index for the frontend slider
  cache/ndvi_YYYY.npy          raw NDVI float32 arrays (used for reruns / diff)
  cache/nbr_YYYY.npy           raw NBR  float32 arrays

Install:
  pip install pystac-client planetary-computer rasterio numpy matplotlib Pillow tqdm
"""

import json
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
HERE       = Path(__file__).parent
OUTPUT_DIR = HERE / "outputs"
NDVI_DIR   = OUTPUT_DIR / "ndvi"
NBR_DIR    = OUTPUT_DIR / "nbr"
CACHE_DIR  = HERE / "cache"

# ── Study area ────────────────────────────────────────────────────────────────
# Hobet Mine, Lincoln/Boone County, West Virginia  (WRS-2 path 19, row 34)
BOUNDS = (-82.10, 37.85, -81.60, 38.15)   # west, south, east, north

# ── Output grid (~30 m in EPSG:4326) ─────────────────────────────────────────
OUT_RES       = 0.00027
OUT_WIDTH     = int(round((BOUNDS[2] - BOUNDS[0]) / OUT_RES))
OUT_HEIGHT    = int(round((BOUNDS[3] - BOUNDS[1]) / OUT_RES))
OUT_TRANSFORM = from_bounds(*BOUNDS, OUT_WIDTH, OUT_HEIGHT)
OUT_CRS       = CRS.from_epsg(4326)

# ── Processing parameters ─────────────────────────────────────────────────────
YEARS        = list(range(1985, 2026))
CLOUD_THRESH = 25
MAX_SCENES   = 15          # cap per year, sorted least-cloudy first
MIN_CLEAR    = 3           # minimum clear observations per pixel to keep composite value
SEASON_START = 5           # May  ─┐ peak vegetation / full leaf-out in WV
SEASON_END   = 9           # Sep  ─┘ deciduous forest

# Date ranges for the dNDVI summary (matches the original study design)
BEFORE_YEARS = list(range(1985, 2006))
AFTER_YEARS  = list(range(2014, 2021))

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
DIFF_CMAP = plt.get_cmap("RdYlGn")

NDVI_VMIN, NDVI_VMAX = -0.1,  0.8
NBR_VMIN,  NBR_VMAX  = -0.3,  0.8
DIFF_VMIN, DIFF_VMAX = -0.3,  0.3


# ── Raster I/O ────────────────────────────────────────────────────────────────

def _reproject_band(href: str, dtype: type, nodata) -> np.ndarray:
    dst = np.full((OUT_HEIGHT, OUT_WIDTH), nodata, dtype=dtype)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=OUT_TRANSFORM,
            dst_crs=OUT_CRS,
            resampling=Resampling.bilinear if dtype == np.float32 else Resampling.nearest,
            src_nodata=0,
            dst_nodata=nodata,
        )
    return dst


def read_band_to_grid(href: str) -> np.ndarray:
    return _reproject_band(href, np.float32, np.nan)


def read_qa_to_grid(href: str) -> np.ndarray:
    return _reproject_band(href, np.uint16, 0)


def fetch_scene_bands(hrefs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download red, nir, swir, qa concurrently (4 parallel HTTP reads)."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_red  = pool.submit(read_band_to_grid, hrefs["red"])
        f_nir  = pool.submit(read_band_to_grid, hrefs["nir"])
        f_swir = pool.submit(read_band_to_grid, hrefs["swir"])
        f_qa   = pool.submit(read_qa_to_grid,   hrefs["qa"])
    return f_red.result(), f_nir.result(), f_swir.result(), f_qa.result()


# ── Index computation ─────────────────────────────────────────────────────────

def cloud_mask(qa: np.ndarray) -> np.ndarray:
    """
    True for clear pixels. Masks:
      bit 1  — dilated cloud (cloud edges)
      bit 2  — cirrus (LS8/9; ignored as 0 on LS5/7)
      bit 3  — cloud shadow
      bit 4  — snow
      bit 5  — cloud
      bits 8-9 — cloud confidence medium/high (catches what the hard flag misses)
    """
    cloud_conf = (qa >> 8) & 3   # 0=not set, 1=low, 2=medium, 3=high
    return (
        ((qa & 2)   == 0) &      # dilated cloud
        ((qa & 4)   == 0) &      # cirrus
        ((qa & 8)   == 0) &      # cloud shadow
        ((qa & 16)  == 0) &      # snow
        ((qa & 32)  == 0) &      # cloud
        (cloud_conf <= 1)         # low or unset confidence only
    )


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

def _colormap_png(array: np.ndarray, cmap, vmin: float, vmax: float,
                  path: Path) -> None:
    """RGBA PNG with transparency where NaN. Used for deck.gl BitmapLayer."""
    norm    = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba    = cmap(norm(np.nan_to_num(array, nan=vmin)))
    rgba_u8 = (rgba * 255).astype(np.uint8)
    rgba_u8[np.isnan(array), 3] = 0
    Image.fromarray(rgba_u8, mode="RGBA").save(path)


def save_outputs(ndvi: np.ndarray, nbr: np.ndarray, year: int) -> None:
    _colormap_png(ndvi, NDVI_CMAP, NDVI_VMIN, NDVI_VMAX, NDVI_DIR / f"ndvi_{year}.png")
    _colormap_png(nbr,  NBR_CMAP,  NBR_VMIN,  NBR_VMAX,  NBR_DIR  / f"nbr_{year}.png")
    np.save(CACHE_DIR / f"ndvi_{year}.npy", ndvi)
    np.save(CACHE_DIR / f"nbr_{year}.npy",  nbr)


def save_timeseries(stats: dict) -> None:
    """Mean NDVI per year with ±1σ band across all processed years."""
    years = sorted(int(y) for y in stats if stats[y]["mean_ndvi"] is not None)
    means = np.array([stats[str(y)]["mean_ndvi"] for y in years], dtype=float)
    stds  = np.array([stats[str(y)]["std_ndvi"]  for y in years], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d0d0d")

    ax.axvspan(BEFORE_YEARS[0], BEFORE_YEARS[-1] + 0.9,
               alpha=0.08, color="#4A90D9", label="Before (1985–2005)")
    ax.axvspan(AFTER_YEARS[0],  AFTER_YEARS[-1]  + 0.9,
               alpha=0.08, color="#7ED321", label="After (2014–2020)")
    ax.fill_between(years, means - stds, means + stds,
                    alpha=0.2, color="#78C679", linewidth=0, label="±1σ")
    ax.plot(years, means, color="#78C679", linewidth=1.8,
            marker="o", markersize=4,
            markerfacecolor="white", markeredgecolor="#78C679", markeredgewidth=1.2)

    ax.set_xlabel("Year",                color="white", fontsize=10)
    ax.set_ylabel("Mean NDVI (May–Sep)", color="white", fontsize=10)
    ax.set_title("Hobet Mine — Annual Mean NDVI 1985–2025",
                 color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#333", linewidth=0.5, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white",
              facecolor="#222", loc="lower right")

    plt.tight_layout(pad=1.2)
    path = OUTPUT_DIR / "ndvi_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")


def save_diff() -> None:
    """Before vs after dNDVI map loaded from cached .npy arrays."""
    before_arrays = [np.load(CACHE_DIR / f"ndvi_{y}.npy")
                     for y in BEFORE_YEARS if (CACHE_DIR / f"ndvi_{y}.npy").exists()]
    after_arrays  = [np.load(CACHE_DIR / f"ndvi_{y}.npy")
                     for y in AFTER_YEARS  if (CACHE_DIR / f"ndvi_{y}.npy").exists()]

    if not before_arrays or not after_arrays:
        print("  → ndvi_diff.png skipped (before or after arrays missing from cache/)")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        before_mean = np.nanmean(np.stack(before_arrays), axis=0)
        after_mean  = np.nanmean(np.stack(after_arrays),  axis=0)
    diff = (after_mean - before_mean).astype(np.float32)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d0d0d")

    norm = mcolors.TwoSlopeNorm(vmin=DIFF_VMIN, vcenter=0.0, vmax=DIFF_VMAX)
    im   = ax.imshow(
        diff, cmap=DIFF_CMAP, norm=norm,
        extent=[BOUNDS[0], BOUNDS[2], BOUNDS[1], BOUNDS[3]],
        origin="upper", interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("dNDVI (after − before)", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_xlabel("Longitude", color="white", fontsize=8)
    ax.set_ylabel("Latitude",  color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_title(
        "dNDVI: after (2014–2020) minus before (1985–2005)\n"
        "green = vegetation gain   red = vegetation loss",
        color="white", fontsize=10, pad=8,
    )
    plt.tight_layout(pad=1.0)
    path = OUTPUT_DIR / "ndvi_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {path}")


# ── Core processing ───────────────────────────────────────────────────────────

def process_year(
    catalog, year: int
) -> tuple[np.ndarray | None, np.ndarray | None, int, list[str]]:
    """Returns (ndvi, nbr, n_scenes, platforms). Both arrays None if no scenes."""
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=list(BOUNDS),
        datetime=f"{year}-{SEASON_START:02d}-01/{year}-{SEASON_END:02d}-30",
    )
    items = sorted(
        (
            i for i in search.items()
            if i.properties.get("eo:cloud_cover", 100) < CLOUD_THRESH
            # LS7 scan-line corrector failed May 2003 — stripes degrade composites
            and not (i.properties.get("platform") == "landsat-7" and year > 2003)
        ),
        key=lambda i: i.properties.get("eo:cloud_cover", 100),
    )[:MAX_SCENES]

    if not items:
        return None, None, 0, []

    ndvi_stack: list[np.ndarray] = []
    nbr_stack:  list[np.ndarray] = []
    platforms:  list[str]        = []

    for item in items:
        signed   = planetary_computer.sign(item)
        platform = signed.properties.get("platform", "")
        if platform not in PLATFORM_BANDS:
            continue

        bk = PLATFORM_BANDS[platform]
        hrefs = {
            "red":  signed.assets[bk["red"]].href,
            "nir":  signed.assets[bk["nir"]].href,
            "swir": signed.assets[bk["swir"]].href,
            "qa":   signed.assets[bk["qa"]].href,
        }

        try:
            red_dn, nir_dn, swir_dn, qa = fetch_scene_bands(hrefs)
        except Exception as exc:
            log.warning("Scene %s failed: %s", item.id, exc)
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

    # pixels with fewer than MIN_CLEAR valid observations are likely cloud remnants
    clear_count = np.sum(~np.isnan(stack_ndvi), axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi_out = np.nanmedian(stack_ndvi, axis=0).astype(np.float32)
        nbr_out  = np.nanmedian(stack_nbr,  axis=0).astype(np.float32)

    ndvi_out[clear_count < MIN_CLEAR] = np.nan
    nbr_out[clear_count  < MIN_CLEAR] = np.nan

    return ndvi_out, nbr_out, len(ndvi_stack), platforms


def year_stats(ndvi: np.ndarray, nbr: np.ndarray) -> dict:
    valid_ndvi = ndvi[~np.isnan(ndvi)]
    valid_nbr  = nbr[~np.isnan(nbr)]
    return {
        "mean_ndvi":   round(float(np.mean(valid_ndvi)),   4) if len(valid_ndvi) else None,
        "median_ndvi": round(float(np.median(valid_ndvi)), 4) if len(valid_ndvi) else None,
        "std_ndvi":    round(float(np.std(valid_ndvi)),    4) if len(valid_ndvi) else None,
        "mean_nbr":    round(float(np.mean(valid_nbr)),    4) if len(valid_nbr)  else None,
        "pct_valid":   round(100 * len(valid_ndvi) / ndvi.size, 2),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    NDVI_DIR.mkdir(exist_ok=True)
    NBR_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    stats    = {}
    manifest = {
        "bounds":         list(BOUNDS),
        "resolution_deg": OUT_RES,
        "width_px":       OUT_WIDTH,
        "height_px":      OUT_HEIGHT,
        "season":         f"May–Sep (months {SEASON_START}–{SEASON_END})",
        "years":          [],
        "files":          {},
    }

    for year in tqdm(YEARS, desc="Processing years"):
        ndvi_png = NDVI_DIR  / f"ndvi_{year}.png"
        nbr_png  = NBR_DIR   / f"nbr_{year}.png"
        ndvi_npy = CACHE_DIR / f"ndvi_{year}.npy"
        nbr_npy  = CACHE_DIR / f"nbr_{year}.npy"

        cached = ndvi_png.exists() and nbr_png.exists() \
                 and ndvi_npy.exists() and nbr_npy.exists()

        if cached:
            ndvi = np.load(ndvi_npy)
            nbr  = np.load(nbr_npy)
            tqdm.write(f"  {year}: cached")
        else:
            ndvi, nbr, n_scenes, platforms = process_year(catalog, year)
            if ndvi is None:
                tqdm.write(f"  {year}: no valid scenes — skipped")
                continue
            save_outputs(ndvi, nbr, year)
            tqdm.write(
                f"  {year}: {n_scenes} scenes, {int((~np.isnan(ndvi)).sum()):,} valid px"
                + (f"  [{', '.join(sorted(set(platforms)))}]" if platforms else "")
            )

        stats[str(year)] = year_stats(ndvi, nbr)
        manifest["years"].append(year)
        manifest["files"][str(year)] = {"ndvi": str(ndvi_png), "nbr": str(nbr_png)}

    print("\nGenerating summary outputs...")

    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  → {OUTPUT_DIR / 'stats.json'}")

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  → {OUTPUT_DIR / 'manifest.json'}")

    if len(stats) >= 2:
        save_timeseries(stats)

    save_diff()

    print(f"\nDone. {len(manifest['years'])} years in {OUTPUT_DIR}/")
    print(f"deck.gl:  bounds={list(BOUNDS)}")
    print(f"          NDVI  → outputs/ndvi_{{year}}.png")
    print(f"          NBR   → outputs/nbr_{{year}}.png")
    print(f"          chart → outputs/ndvi_timeseries.png")


if __name__ == "__main__":
    main()
