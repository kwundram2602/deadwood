"""Bring every time-series orthomosaic onto the reference grid.

The source scenes are ~2.2 GB each and sit 2.08 m off the reference extent, so
this reprojects band by band with rasterio.warp.reproject rather than reading a
scene into memory or rescaling its own extent onto the target shape.

Idempotent by design: the 2023-2026 upload is still in progress, so this is
expected to be re-run as dates arrive.
"""

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid
from deadwood_spectral.indices import BAND_NAMES

logger = logging.getLogger(__name__)

DATE_RE = re.compile(r"^(\d{8})_")
UINT16_MAX = 65535.0


def parse_date(path: str | Path) -> str:
    """Extract the leading YYYYMMDD from a scene filename."""
    match = DATE_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"{Path(path).name}: no leading YYYYMMDD_ date in filename")
    return match.group(1)


def align_scene(
    src_path: str | Path,
    grid: ReferenceGrid,
    out_path: str | Path,
    band_names: Sequence[str] = BAND_NAMES,
) -> Path:
    """Reproject one scene onto the reference grid as float32 [0,1].

    Reads and writes one band at a time: a full 7-band float32 scene on the
    reference grid is ~1.2 GB, and the native source is far larger.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = dict(
        driver="GTiff", dtype="float32",
        height=grid.height, width=grid.width, count=len(band_names),
        crs=grid.crs, transform=grid.transform, nodata=np.nan,
        compress="lzw", tiled=True, blockxsize=512, blockysize=512,
    )

    with rasterio.open(src_path) as src:
        if src.crs != grid.crs:
            raise ValueError(f"{src_path}: CRS {src.crs} != reference {grid.crs}")
        if src.count < len(band_names):
            raise ValueError(
                f"{src_path}: {src.count} bands, expected at least {len(band_names)}"
            )
        tmp_path = out_path.with_suffix(".tif.part")
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for out_idx, name in enumerate(band_names, start=1):
                buffer = np.full(grid.shape, np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, out_idx),
                    destination=buffer,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
                buffer /= UINT16_MAX
                # Calibration artifacts reach ~1e23 in the raw mosaics.
                n_bad = int(np.sum((buffer < 0.0) | (buffer > 1.0)))
                if n_bad:
                    logger.warning(
                        "%s band %s: clipped %d out-of-range value(s)",
                        Path(src_path).name, name, n_bad,
                    )
                np.clip(buffer, 0.0, 1.0, out=buffer)
                dst.write(buffer, out_idx)
                dst.set_band_description(out_idx, name)
    # Rename last so a crashed run never leaves a half-written stack that
    # is_aligned would accept.
    tmp_path.replace(out_path)
    return out_path


def is_aligned(out_path: str | Path, grid: ReferenceGrid) -> bool:
    """True if an output already exists and sits on the reference grid."""
    out_path = Path(out_path)
    if not out_path.exists():
        return False
    try:
        with rasterio.open(out_path) as src:
            assert_matches_grid(src, grid, str(out_path))
    except (ValueError, rasterio.errors.RasterioIOError):
        return False
    return True


def align_all(
    src_dir: str | Path,
    grid: ReferenceGrid,
    out_dir: str | Path,
    band_names: Sequence[str] = BAND_NAMES,
    force: bool = False,
) -> list[Path]:
    """Align every *.tif in src_dir; return the paths actually (re)written."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for src_path in sorted(src_dir.glob("*.tif")):
        date = parse_date(src_path)
        out_path = out_dir / f"{date}_stack.tif"
        if not force and is_aligned(out_path, grid):
            logger.info("skip %s (already aligned)", out_path.name)
            continue
        logger.info("align %s -> %s", src_path.name, out_path.name)
        align_scene(src_path, grid, out_path, band_names)
        written.append(out_path)

    (out_dir / "channels.json").write_text(
        json.dumps({"names": list(band_names)}, indent=2)
    )
    return written
