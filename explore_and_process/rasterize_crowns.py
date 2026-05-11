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

Usage:
  python explore_and_process/rasterize_crowns.py \\
      --crowns  explore_and_process/crowns/Tree_Inventory_20260325_processed_crowns.gpkg \\
      --reference data/raster/20230824_Airport_Main_MAVICM3MFIXEDM3M_OM_coregReference.tif \\
      --out_mask  explore_and_process/out/crown_mask.tif \\
      --raster_dir    data/raster \\
      --out_image_dir explore_and_process/out/images \\
      [--target_gsd 0.05] [--sigma 10.0] [--nodata_threshold 0.05]
      [--bands 5 4 6 7]
"""
# python explore_and_process/rasterize_crowns.py \\     --crowns  datafiles/crown_poly/2_crown_main_20260409_editLP.gpkg --reference datafiles/raster/20260313/20260313_Airport_Main_MAVICM3MFIXEDM3M_tile001_OM_shift.tif --out_mask  datafiles/process_out/crown_mask.tif --raster_dir    data/raster --out_image_dir explore_and_process/out/images --target_gsd 0.05
import argparse
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

# Default 1-indexed bands from the 7-band Mavic M3M stack:
#   Band 4 = MS Green, 5 = MS Red, 6 = RE, 7 = NIR
# Stored in R,G,RE,NIR order to align with TorchGeo TCD pretrained encoder
MS_BANDS_DEFAULT = [5, 4, 6, 7]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def target_grid(src, gsd):
    """Return (height, width, transform) for a resampled grid at gsd metres."""
    h = int(round(src.height * src.res[0] / gsd))
    w = int(round(src.width  * src.res[1] / gsd))
    return h, w, from_bounds(*src.bounds, w, h)


def write_tif(path, data, transform, crs, nodata=None):
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
    binary = rio_rasterize(shapes, out_shape=(h, w), transform=transform,
                           fill=0.0, dtype="float32")

    soft = gaussian_filter(binary, sigma=sigma)
    soft[(binary == 0) & (soft < nodata_threshold)] = 255.0

    n_crown  = int(np.sum((soft > 0) & (soft < 255)))
    n_ground = int(np.sum(soft == 0.0))
    n_nodata = int(np.sum(soft == 255.0))
    print(f"  Crown: {n_crown:,}  Ground: {n_ground:,}  noData: {n_nodata:,}")
    return soft


def resample_image(om_path, bands, h, w, transform, crs, out_path):
    """Read bands, resample to target grid, normalise to [0,1], save."""
    with rasterio.open(om_path) as src:
        data = src.read(indexes=bands,
                        out_shape=(len(bands), h, w),
                        resampling=Resampling.bilinear).astype(np.float32)
    data /= 65535.0          # uint16-range → [0, 1]
    data = np.where(np.isnan(data), 0.0, data)
    write_tif(out_path, data, transform, crs, nodata=None)
    print(f"  -> {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logger.info("Config:\n%s", OmegaConf.to_yaml(args))

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
        if args.raster_dir:
            om_files = sorted(
                os.path.join(args.raster_dir, f)
                for f in os.listdir(args.raster_dir)
                if "_OM_" in f and f.endswith(".tif")
            )
        else:
            om_files = [args.reference]

        print(f"\nResampling {len(om_files)} OM image(s) to {args.target_gsd*100:.1f} cm...")
        for om_path in om_files:
            stem = os.path.splitext(os.path.basename(om_path))[0]
            out_path = os.path.join(args.out_image_dir, f"{stem}_ms4.tif")
            resample_image(om_path, args.bands, h, w, transform, crs, out_path)

        print(f"\nDone. {len(om_files)} image(s) written to {args.out_image_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Stage 1a: rasterize crown polygons to soft mask.")
    p.add_argument("--config", required=True, help="Path to preprocess.yaml")
    cfg = OmegaConf.load(p.parse_args().config)
    main(cfg.rasterize)
