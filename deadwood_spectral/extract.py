"""Read the sampled pixels out of every aligned stack into one table.

This is the last step that touches the rasters in stage B. Everything after it
works on a table that fits in memory, so exploration is fast and repeatable.

Reads run row-chunk by row-chunk rather than per pixel: the samples are spread
over 6459 x 6962 px, and a windowed read per pixel would be thousands of seeks
per date.
"""

import json
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
    "group_id", "certaintyLP", "coverage", "quality_ok",
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


NDSM_REFERENCE_FILE = "ndsm_reference.json"
_SIGNATURE_WINDOW_PX = 512


def ndsm_signature(path: str | Path) -> dict:
    """Cheap identity record for the nDSM raster used at a pipeline stage.

    Two nDSM variants exist on disk — metres and normalized — both on the
    correct reference grid, so `assert_matches_grid` cannot tell them apart.
    Training on one and running inference with the other produces silently
    wrong scene-wide predictions with no error anywhere.

    The signature is a GDAL checksum over a fixed central window plus the file
    size, not a hash of the whole file: it reads a few hundred kB instead of
    ~180 MB, and two rasters that differ in units or normalisation differ in
    the middle of the AOI with certainty. The absolute path is recorded too,
    but only for the message — a moved or renamed file with identical content
    is the same nDSM.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        height = min(_SIGNATURE_WINDOW_PX, src.height)
        width = min(_SIGNATURE_WINDOW_PX, src.width)
        window = Window(
            (src.width - width) // 2, (src.height - height) // 2, width, height
        )
        checksum = int(src.checksum(1, window=window))
        shape = [int(src.height), int(src.width)]
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "shape": shape,
        "window_px": [int(window.col_off), int(window.row_off), int(width), int(height)],
        "window_checksum": checksum,
    }


def samples_ndsm_reference_path(samples_path: str | Path) -> Path:
    """Sidecar path recording which nDSM went into a samples table."""
    samples_path = Path(samples_path)
    return samples_path.with_name(f"{samples_path.stem}_{NDSM_REFERENCE_FILE}")


def save_ndsm_reference(signature: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signature, indent=2))
    return path


def load_ndsm_reference(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def assert_same_ndsm(expected: dict, actual_path: str | Path) -> None:
    """Fail unless `actual_path` is the nDSM the model was trained against.

    `paths.ndsm` is declared independently in analysis.yaml (training samples)
    and classify.yaml (inference), with nothing pinning them together. This is
    that pin. Content decides; a differing path with identical content is only
    logged.
    """
    actual = ndsm_signature(actual_path)
    keys = ("size_bytes", "shape", "window_px", "window_checksum")
    if any(expected.get(k) != actual.get(k) for k in keys):
        raise ValueError(
            "nDSM mismatch: the model was trained against "
            f"{expected.get('path')!r} (size {expected.get('size_bytes')}, "
            f"window checksum {expected.get('window_checksum')}), but inference "
            f"was given {actual['path']!r} (size {actual['size_bytes']}, window "
            f"checksum {actual['window_checksum']}). Both nDSM variants sit on "
            "the reference grid, so nothing else would catch this — set "
            "paths.ndsm to the SAME file in configs/spectral/analysis.yaml and "
            "configs/spectral/classify.yaml, or retrain."
        )
    if expected.get("path") != actual["path"]:
        logger.info(
            "nDSM path differs from training (%s -> %s) but the content matches",
            expected.get("path"), actual["path"],
        )


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

    # Validate EVERY input before reading the first one. A real run reads a
    # dozen ~800 MB stacks; discovering a bad ndsm path afterwards throws away
    # tens of minutes of work and never writes samples.parquet.
    inputs = [stack_dir / f"{date}_stack.tif" for date in dates]
    if ndsm_path is not None:
        inputs.append(Path(ndsm_path))
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing input raster(s) before extraction started: " + ", ".join(missing)
        )
    for path in inputs:
        with rasterio.open(path) as src:
            assert_matches_grid(src, grid, str(path))

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
