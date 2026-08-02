"""Read the sampled pixels out of every aligned stack into one table.

This is the last step that touches the rasters in stage B. Everything after it
works on a table that fits in memory, so exploration is fast and repeatable.

Reads run row-chunk by row-chunk rather than per pixel: the samples are spread
over 6459 x 6962 px, and a windowed read per pixel would be thousands of seeks
per date.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid
from deadwood_spectral.indices import BAND_NAMES, compute_indices

logger = logging.getLogger(__name__)

LABEL_COLUMNS = (
    "row", "col", "class_name", "class_code", "tree_id",
    "group_id", "species", "certaintyLP", "coverage", "quality_ok",
)


def feature_column(name: str, date: str) -> str:
    """Column name for one measurement on one date."""
    return f"{name}_{date}"


def available_dates(stack_dir: str | Path, exclude: Sequence[str] = ()) -> list[str]:
    """Dates with an aligned stack, minus the excluded ones."""
    excluded = set(exclude)
    dates = sorted(
        p.name[:8] for p in Path(stack_dir).glob("*_stack.tif") if p.name[:8] not in excluded
    )
    return dates


def _read_at(src, rows: np.ndarray, cols: np.ndarray, chunk_rows: int) -> np.ndarray:
    """Values at (rows, cols) for all bands, read in row chunks. -> (C, N)."""
    out = np.full((src.count, rows.size), np.nan, dtype=np.float32)
    order = np.argsort(rows, kind="stable")
    for start in range(0, src.height, chunk_rows):
        stop = min(start + chunk_rows, src.height)
        sel = order[(rows[order] >= start) & (rows[order] < stop)]
        if sel.size == 0:
            continue
        block = src.read(window=Window(0, start, src.width, stop - start)).astype(np.float32)
        out[:, sel] = block[:, rows[sel] - start, cols[sel]]
    return out


def extract_samples(
    samples: pd.DataFrame,
    stack_dir: str | Path,
    grid: ReferenceGrid,
    dates: Sequence[str] | None = None,
    exclude_dates: Sequence[str] = (),
    ndsm_path: str | Path | None = None,
    chunk_rows: int = 2048,
) -> pd.DataFrame:
    """Attach every band and index, for every date, to the sample table."""
    stack_dir = Path(stack_dir)
    dates = list(dates) if dates is not None else available_dates(stack_dir, exclude_dates)
    if not dates:
        raise ValueError(f"no aligned stacks in {stack_dir}")

    rows = samples["row"].to_numpy(dtype=np.int64)
    cols = samples["col"].to_numpy(dtype=np.int64)
    keep = [c for c in LABEL_COLUMNS if c in samples.columns]
    out = samples[keep].reset_index(drop=True).copy()

    for date in dates:
        path = stack_dir / f"{date}_stack.tif"
        with rasterio.open(path) as src:
            assert_matches_grid(src, grid, str(path))
            values = _read_at(src, rows, cols, chunk_rows)
            names = [d or n for d, n in zip(src.descriptions, BAND_NAMES)]
        for name, band in zip(names, values):
            out[feature_column(name, date)] = band
        # compute_indices wants (C, H, W); the sample vector is a 1-px-tall image.
        indices = compute_indices(values[:, None, :], names)
        for name, arr in indices.items():
            out[feature_column(name, date)] = arr[0]
        logger.info("extracted %s (%d samples)", date, rows.size)

    if ndsm_path is not None:
        with rasterio.open(ndsm_path) as src:
            assert_matches_grid(src, grid, str(ndsm_path))
            out["ndsm"] = _read_at(src, rows, cols, chunk_rows)[0]

    return out
