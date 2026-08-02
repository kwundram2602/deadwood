"""Residual co-registration between aligned dates.

Alignment puts every date on the reference grid, but that only fixes the
georeference — it cannot fix an image that was georeferenced wrongly upstream.
This measures what is left over on stable, non-vegetated tiles.

The report is diagnosis; the consequence is exclusion. A shift correction is
deliberately not applied: correcting a badly estimated shift is worse than
dropping the date, and with 18 positives a corrupted date costs more than a
missing one.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from skimage.registration import phase_cross_correlation

from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid
from deadwood_spectral.indices import BAND_NAMES

logger = logging.getLogger(__name__)


def estimate_shift(
    reference: np.ndarray, moving: np.ndarray, upsample: int = 10
) -> tuple[float, float]:
    """(drow, dcol) in pixels that `moving` sits away from `reference`.

    NaNs are replaced by the tile mean: phase correlation needs a filled array,
    and a constant fill contributes no cross-correlation peak of its own.
    """

    def _fill(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32, copy=True)
        bad = ~np.isfinite(arr)
        if bad.all():
            return np.zeros_like(arr)
        arr[bad] = float(np.nanmean(arr))
        return arr

    shift, _, _ = phase_cross_correlation(
        _fill(reference), _fill(moving), upsample_factor=upsample
    )
    # skimage returns the shift to apply to `moving` to match `reference`;
    # negate so the sign reads as "where moving sits relative to reference".
    return (-float(shift[0]), -float(shift[1]))


def tile_windows(
    grid: ReferenceGrid, tiles_xy: Sequence[tuple[float, float]], tile_size_px: int
) -> list[Window]:
    """Square read windows centred on the given map coordinates."""
    half = tile_size_px / 2.0
    res_x, res_y = abs(grid.transform.a), abs(grid.transform.e)
    windows = []
    for x, y in tiles_xy:
        left, right = x - half * res_x, x + half * res_x
        bottom, top = y - half * res_y, y + half * res_y
        windows.append(
            from_bounds(left, bottom, right, top, transform=grid.transform)
            .round_offsets()
            .round_lengths()
        )
    return windows


def _read_tile(path: Path, window: Window, band_index: int) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(band_index, window=window).astype(np.float32)


def coreg_report(
    stack_dir: str | Path,
    grid: ReferenceGrid,
    tiles_xy: Sequence[tuple[float, float]],
    tile_size_px: int = 512,
    reference_date: str | None = None,
    max_shift_m: float = 0.15,
    band: str = "NIR",
) -> pd.DataFrame:
    """Per-date residual shift against the reference date, on stable tiles."""
    stack_dir = Path(stack_dir)
    stacks = {p.name[:8]: p for p in sorted(stack_dir.glob("*_stack.tif"))}
    if not stacks:
        raise ValueError(f"no *_stack.tif in {stack_dir}")
    if reference_date is None:
        reference_date = max(stacks)
    if reference_date not in stacks:
        raise ValueError(f"reference_date {reference_date} not among {sorted(stacks)}")
    if not tiles_xy:
        raise ValueError("coreg.tiles is empty; pick stable, non-vegetated tiles first")

    band_index = list(BAND_NAMES).index(band) + 1
    windows = tile_windows(grid, tiles_xy, tile_size_px)
    res_x, res_y = abs(grid.transform.a), abs(grid.transform.e)

    for path in stacks.values():
        with rasterio.open(path) as src:
            assert_matches_grid(src, grid, str(path))

    ref_tiles = [_read_tile(stacks[reference_date], w, band_index) for w in windows]

    rows = []
    for date, path in sorted(stacks.items()):
        shifts_m = []
        for ref_tile, window in zip(ref_tiles, windows):
            moving = _read_tile(path, window, band_index)
            drow, dcol = estimate_shift(ref_tile, moving)
            shifts_m.append((dcol * res_x, -drow * res_y))
        arr = np.asarray(shifts_m, dtype=np.float64)
        dx, dy = np.median(arr, axis=0)
        spread = float(np.max(np.std(arr, axis=0)))
        magnitude = float(np.hypot(dx, dy))
        rows.append(
            {
                "date": date,
                "dx_m": float(dx),
                "dy_m": float(dy),
                "spread_m": spread,
                "n_tiles": len(windows),
                "flagged": bool(magnitude > max_shift_m),
            }
        )
        logger.info(
            "%s: dx=%+.3f m dy=%+.3f m |d|=%.3f m spread=%.3f m %s",
            date, dx, dy, magnitude, spread, "FLAGGED" if rows[-1]["flagged"] else "",
        )
    return pd.DataFrame(rows)


def flagged_dates(report: pd.DataFrame) -> list[str]:
    """Dates whose residual shift exceeded the threshold."""
    return report.loc[report["flagged"], "date"].astype(str).tolist()
