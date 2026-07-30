"""
rasterize_crowns.py

Rasterise crown polygons to a soft training mask and (optionally)
batch-export band-selected, normalised, resampled MS images.

Steps:
  1. Load crown polygons, keep only son/soff
  2. Reproject polygons to raster CRS
  3. Rasterize to binary mask at target GSD
  4. Gaussian blur for soft crown boundaries
  5. Set noData=255 for pixels far from any crown
  6. Save mask
  7. (optional) Resample all OM tifs in --raster_dir to target GSD,
     select 4 MS bands, normalise to [0,1], save as float32

Usage (config-driven; sources replace the old numeric `bands:` list):
  python explore_and_process/rasterize_crowns.py --config configs/preprocess/preprocess.yaml

  # rasterize.sources (in configs/preprocess/preprocess.yaml):
  #   sources:
  #     - path: data/raster/20230824_..._OM_RGB.tif
  #       bands: [1, 2, 3]
  #       names: [red, green, blue]
  #     - path: data/raster/20230824_..._OM_MS.tif
  #       bands: [1, 2, 3, 4]
  #       names: [green_ms, red_ms, rededge, nir]
"""
# python explore_and_process/rasterize_crowns.py \\     --crowns  datafiles/crown_poly/2_crown_main_20260409_editLP.gpkg --reference datafiles/raster/20260313/20260313_Airport_Main_MAVICM3MFIXEDM3M_tile001_OM_shift.tif --out_mask  datafiles/process_out/crown_mask.tif --raster_dir    data/raster --out_image_dir explore_and_process/out/images --target_gsd 0.05
import argparse
import json
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from omegaconf import OmegaConf
from rasterio.enums import Resampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Only these crown categories map to class=1; everything else is excluded
INCLUDE_CATEGORIES = {"son", "soff"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def target_grid(src, gsd):
    """Return (height, width, transform) for a resampled grid at gsd metres."""
    h = int(round(src.height * src.res[0] / gsd))
    w = int(round(src.width  * src.res[1] / gsd))
    return h, w, from_bounds(*src.bounds, w, h)


def write_tif(path, data, transform, crs, nodata=None, descriptions=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis]
    profile = dict(
        driver="GTiff", dtype="float32",
        width=data.shape[2], height=data.shape[1],
        count=data.shape[0], crs=crs, transform=transform,
        nodata=nodata, compress="lzw", tiled=True,
        blockxsize=512, blockysize=512,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def build_mask(crowns_paths, src, h, w, transform, sigma, nodata_threshold):
    """Rasterize crowns → Gaussian blur → noData sentinel."""
    gdfs = [gpd.read_file(p) for p in crowns_paths]
    gdf = pd.concat(gdfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdfs[0].crs)
    gdf = gdf[gdf["crown_category"].isin(INCLUDE_CATEGORIES)].to_crs(src.crs)
    print(f"  {len(gdf)} crown polygons (son/soff) from {len(crowns_paths)} file(s)")

    shapes = [(geom, 1.0) for geom in gdf.geometry if geom is not None and geom.is_valid]
    # Burn crown polygons into a binary raster: crown pixels = 1.0, background = 0.0
    binary = rio_rasterize(shapes, out_shape=(h, w), transform=transform,
                           fill=0.0, dtype="float32")

    soft = gaussian_filter(binary, sigma=sigma)
    # Pixels outside all crowns that received no Gaussian bleed-over become noData (255)
    soft[(binary == 0) & (soft < nodata_threshold)] = 255.0

    n_crown  = int(np.sum((soft > 0) & (soft < 255)))
    soft_zero = int(np.sum(soft == 0.0))
    n_nodata = int(np.sum(soft == 255.0))
    print(f"  Crown: {n_crown:,}  Soft == 0.0: {soft_zero:,}  noData: {n_nodata:,}")
    return soft


def read_scaled_bands(path, bands, h, w):
    """Read selected bands, resample to (h, w), scale uint16-range to [0,1]."""
    with rasterio.open(path) as src:
        data = src.read(indexes=bands,
                        out_shape=(len(bands), h, w),
                        resampling=Resampling.bilinear).astype(np.float32)
    data /= 65535.0          # uint16-range → [0, 1]
    data = np.where(np.isnan(data), 0.0, data)
    # Sensor/calibration artifacts can produce physically impossible
    # reflectance (hot pixels up to ~1e23 in the raw mosaic); clip to [0,1]
    n_clipped = int(np.sum((data < 0.0) | (data > 1.0)))
    if n_clipped:
        print(f"  [WARN] clipped {n_clipped} out-of-range pixel value(s) to [0,1]")
    np.clip(data, 0.0, 1.0, out=data)
    return data


def validate_sources(sources, raster_dir=None):
    """Check a rasterize.sources config list; return the combined channel names."""
    if not sources:
        raise ValueError("rasterize.sources must list at least one source")
    if raster_dir and len(sources) > 1:
        raise ValueError("raster_dir batch mode requires exactly one source entry")
    names = []
    for s in sources:
        s_names = [str(n) for n in s.names]
        if len(list(s.bands)) != len(s_names):
            raise ValueError(f"{s.path}: bands/names length mismatch "
                             f"({list(s.bands)} vs {s_names})")
        names.extend(s_names)
    if "ndsm" in names:
        raise ValueError("channel name 'ndsm' is reserved for the DSM channel")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate channel names across sources: {names}")
    return names


def stack_sources(specs, h, w, transform, crs, out_path):
    """Resample each (path, bands, names) source to the target grid and stack
    all bands into one float32 [0,1] GeoTIFF with named band descriptions."""
    arrays, names = [], []
    for path, bands, band_names in specs:
        arrays.append(read_scaled_bands(path, bands, h, w))
        names.extend(band_names)
    data = np.concatenate(arrays, axis=0)
    write_tif(out_path, data, transform, crs, nodata=None, descriptions=names)
    print(f"  -> {os.path.basename(out_path)}  ({len(names)} ch: {', '.join(names)})")
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logger.info("Config:\n%s", OmegaConf.to_yaml(args))
    names = validate_sources(args.sources, args.raster_dir) if args.out_image_dir else None

    with rasterio.open(args.reference) as ref:
        crs = ref.crs
        h, w, transform = target_grid(ref, args.target_gsd)
        print(f"Target grid: {h} x {w} at {args.target_gsd * 100:.1f} cm GSD "
              f"(native {ref.res[0]*100:.2f} cm -> {args.target_gsd*100:.1f} cm)")

        print("\nBuilding crown mask...")
        mask = build_mask(args.crowns, ref, h, w, transform,
                          args.sigma, args.nodata_threshold)  # args.crowns is a list
        write_tif(args.out_mask, mask, transform, crs, nodata=255.0)
        print(f"Mask saved: {args.out_mask}")

    if args.out_image_dir:
        specs = [
            (str(s.path), [int(b) for b in s.bands], [str(n) for n in s.names])
            for s in args.sources
        ]
        if args.raster_dir:
            om_files = sorted(
                os.path.join(args.raster_dir, f)
                for f in os.listdir(args.raster_dir)
                if "_OM_" in f and f.endswith(".tif")
            )
            _, bands, band_names = specs[0]
            jobs = [([(f, bands, band_names)], f) for f in om_files]
        else:
            jobs = [(specs, str(args.reference))]

        print(f"\nStacking {len(jobs)} image(s) at {args.target_gsd*100:.1f} cm...")
        for job_specs, stem_src in jobs:
            stem = os.path.splitext(os.path.basename(stem_src))[0]
            out_path = os.path.join(args.out_image_dir, f"{stem}_stack.tif")
            stack_sources(job_specs, h, w, transform, crs, out_path)

        os.makedirs(args.out_image_dir, exist_ok=True)
        manifest = os.path.join(args.out_image_dir, "channels.json")
        with open(manifest, "w") as f:
            json.dump({"names": names}, f, indent=2)
        print(f"\nDone. {len(jobs)} stack(s) + channels.json written to {args.out_image_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Stage 1a: rasterize crown polygons to soft mask.")
    p.add_argument("--config", required=True, help="Path to preprocess.yaml")
    cfg = OmegaConf.load(p.parse_args().config)
    main(cfg.rasterize)
