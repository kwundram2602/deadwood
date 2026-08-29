"""Does the photogrammetry actually reconstruct the soff crowns?

The spectral tables say what a dead crown looks like; this one says whether
there is a surface over it at all. The nDSM is NaN exactly where the DSM has a
hole, and a leafless crown over dark ground is precisely the situation in which
SfM finds no correspondences — so the share of non-NaN pixels per tree is the
number to read first, before any height is interpreted.

Single epoch by design: the nDSM is one static product, not a time series, and
it is read for the same fixed pixel sample the spectral curves use so that both
tables describe the same pixels.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid

logger = logging.getLogger(__name__)

# Not a tree_id: the pooled row over every soff pixel, appended last.
ALL_TREES = "all_soff"


def read_single_band(
    path: str | Path,
    grid: ReferenceGrid,
    rows: Sequence[int] | np.ndarray,
    cols: Sequence[int] | np.ndarray,
    chunk_rows: int = 512,
) -> np.ndarray:
    """One static raster sampled at a pixel set -> one value per pixel.

    Row-chunked like `read_values`, for the same reason: the nDSM is a
    full-survey raster at 5 cm, and the pixel set is scattered across all of it.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.shape != cols.shape:
        raise ValueError(f"rows/cols length mismatch: {rows.shape} vs {cols.shape}")

    out = np.full(rows.size, np.nan, dtype=np.float32)
    order = np.argsort(rows, kind="stable")
    with rasterio.open(path) as src:
        assert_matches_grid(src, grid, str(path))
        for start in range(0, grid.height, chunk_rows):
            stop = min(start + chunk_rows, grid.height)
            sel = order[(rows[order] >= start) & (rows[order] < stop)]
            if sel.size == 0:
                continue
            block = src.read(1, window=Window.from_slices((start, stop), (0, grid.width)))
            out[sel] = block.astype(np.float32)[rows[sel] - start, cols[sel]]
    return out


def _summarise(heights: np.ndarray) -> dict:
    """Coverage first, then the quartiles of whatever was reconstructed."""
    finite = np.isfinite(heights)
    n_valid = int(finite.sum())
    quartiles = (
        np.nanquantile(heights[finite], [0.25, 0.5, 0.75]) if n_valid else np.full(3, np.nan)
    )
    return {
        "n_px": int(heights.size),
        "n_valid_px": n_valid,
        # The photogrammetry indicator. A tree at 0.0 is not a short tree, it is
        # a tree the surface model never saw.
        "valid_frac": float(n_valid / heights.size) if heights.size else np.nan,
        "median_m": float(quartiles[1]),
        "q25_m": float(quartiles[0]),
        "q75_m": float(quartiles[2]),
        "iqr_m": float(quartiles[2] - quartiles[0]),
    }


def topography_table(values: np.ndarray, pixels: pd.DataFrame) -> pd.DataFrame:
    """One row per soff tree, plus a pooled `all_soff` row last.

    Trees whose every pixel is a hole stay in the table with valid_frac 0 and a
    NaN median: dropping them would hide exactly the failure the table is for.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size != len(pixels):
        raise ValueError(f"values/pixels length mismatch: {values.size} vs {len(pixels)}")

    # Kept as a Series: the column is nullable, and an object array holding
    # pd.NA raises on comparison rather than yielding False.
    tree_ids = pixels["tree_id"]
    records = []
    for tree_id in sorted(tree_ids.dropna().unique()):
        member = (tree_ids == tree_id).fillna(False).to_numpy()
        records.append({"tree_id": str(tree_id), **_summarise(values[member])})
    records.append({"tree_id": ALL_TREES, **_summarise(values[tree_ids.notna().to_numpy()])})

    table = pd.DataFrame.from_records(records)
    empty = table[(table["tree_id"] != ALL_TREES) & (table["n_valid_px"] == 0)]["tree_id"]
    if len(empty):
        logger.warning("no reconstructed nDSM pixel for %d tree(s): %s", len(empty), list(empty))
    return table
