"""
tile_patches.py

Tile full-resolution MS image, final mask, and nDSM into fixed-size patches
for training. Patches with a noData fraction above --nodata_thresh are skipped.

All three inputs must share the same grid (transform + shape). Run
rasterize_crowns.py and apply_dsm_mask.py first to ensure this.

Output naming per patch (zero-padded row/col index):
  <row>_<col>_ms4.tif       — 4-band MS image  (float32, [0,1])
  <row>_<col>_ms4_mask.tif  — 1-band soft mask  (float32, 0/0-1/255)
  <row>_<col>_ms4_dsm.tif   — 1-band nDSM       (float32, [0,1])

Usage:
  python explore_and_process/tile_patches.py \\
      --image explore_and_process/out/images/20260313_..._ms4.tif \\
      --mask  explore_and_process/out/crown_mask_final.tif \\
      --dsm   explore_and_process/out/dsm_ndsm.tif \\
      --out   data/crown_ms_patches \\
      [--size 512] [--nodata_thresh 0.9]
"""

import argparse
import logging
import os

import numpy as np
import rasterio
from omegaconf import OmegaConf
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


def tile_raster(src, row_off, col_off, size):
    """Read a size×size window from an open raster; pad with 0 at edges."""
    h, w = src.height, src.width
    read_h = min(size, h - row_off)
    read_w = min(size, w - col_off)
    data = src.read(window=rasterio.windows.Window(col_off, row_off, read_w, read_h))
    if read_h < size or read_w < size:
        pad = np.zeros((data.shape[0], size, size), dtype=data.dtype)
        pad[:, :read_h, :read_w] = data
        return pad
    return data


def patch_transform(src_transform, row_off, col_off, size):
    left   = src_transform.c + col_off * src_transform.a
    top    = src_transform.f + row_off * src_transform.e
    right  = left + size * src_transform.a
    bottom = top  + size * src_transform.e
    return from_bounds(left, bottom, right, top, size, size)


def write_patch(path, data, transform, crs, nodata=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis]
    profile = dict(
        driver="GTiff", dtype="float32",
        width=data.shape[2], height=data.shape[1],
        count=data.shape[0], crs=crs, transform=transform,
        nodata=nodata, compress="lzw",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def main(args):
    logger.info("Config:\n%s", OmegaConf.to_yaml(args))

    with rasterio.open(args.image) as img_src, \
         rasterio.open(args.mask)  as mask_src, \
         rasterio.open(args.dsm)   as dsm_src:

        h, w = img_src.height, img_src.width
        crs = img_src.crs
        transform = img_src.transform

        n_rows = (h + args.size - 1) // args.size
        n_cols = (w + args.size - 1) // args.size
        pad_rows = len(str(n_rows))
        pad_cols = len(str(n_cols))

        kept = skipped = 0
        for r in range(n_rows):
            for c in range(n_cols):
                row_off = r * args.size
                col_off = c * args.size

                mask_data = tile_raster(mask_src, row_off, col_off, args.size)
                nodata_frac = np.mean(mask_data[0] == 255.0)
                if nodata_frac > args.nodata_thresh:
                    skipped += 1
                    continue

                img_data  = tile_raster(img_src,  row_off, col_off, args.size)
                dsm_data  = tile_raster(dsm_src,  row_off, col_off, args.size)
                pt = patch_transform(transform, row_off, col_off, args.size)

                stem = f"{str(r).zfill(pad_rows)}_{str(c).zfill(pad_cols)}_ms4"
                write_patch(os.path.join(args.out, f"{stem}.tif"),
                            img_data, pt, crs, nodata=None)
                write_patch(os.path.join(args.out, f"{stem}_mask.tif"),
                            mask_data, pt, crs, nodata=255.0)
                write_patch(os.path.join(args.out, f"{stem}_dsm.tif"),
                            dsm_data, pt, crs, nodata=None)
                kept += 1

    print(f"Done. Kept {kept} patches, skipped {skipped} (>{args.nodata_thresh*100:.0f}% noData).")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Stage 2: tile full-res outputs into patches.")
    p.add_argument("--config", required=True, help="Path to preprocess.yaml")
    cfg = OmegaConf.load(p.parse_args().config)
    main(cfg.tiling)
