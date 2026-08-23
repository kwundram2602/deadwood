"""Residual co-registration between aligned dates.

Alignment puts every date on the reference grid,
but that only fixes the georeference — it cannot fix an image that was georeferenced wrongly upstream.
This measures what is left over on stable, non-vegetated tiles.

The report is diagnosis; the consequence is exclusion.
A shift correction is
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


DEFAULT_MAX_TILE_NAN_FRAC = 0.02
DEFAULT_MIN_TILES = 3


def nan_fraction(arr: np.ndarray) -> float:
    """Share of non-finite pixels in a tile."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return 1.0
    return float((~np.isfinite(arr)).sum()) / float(arr.size)


def estimate_shift(
    reference: np.ndarray, moving: np.ndarray, upsample: int = 10
) -> tuple[float, float]:
    """(drow, dcol) in pixels that `moving` sits away from `reference`.

    NaNs are replaced by the tile mean: phase correlation needs a filled array.

    That argument only holds for a SMALL NaN share. A real aligned stack is
    ~45% NaN outside the source footprint, and a large filled region abutting
    real data is a step edge with a cross-correlation peak of its own, which
    can drag the estimate. Callers must therefore reject NaN-heavy tiles
    before calling this — `coreg_report` does, via `max_tile_nan_frac`.
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
    max_tile_nan_frac: float = DEFAULT_MAX_TILE_NAN_FRAC,
    min_tiles: int = DEFAULT_MIN_TILES,
) -> pd.DataFrame:
    """Per-date residual shift against the reference date, on stable tiles.

    Tiles whose NaN share exceeds `max_tile_nan_frac` on either the reference
    or the moving date are rejected: `estimate_shift` fills NaN with the tile
    mean, and on real stacks (~45% NaN outside the source footprint) a large
    filled region abutting real data biases the phase-correlation peak.

    A date with fewer than `min_tiles` usable tiles is not estimated at all —
    dx/dy/spread come back NaN and the date is flagged. `spread_m` is also NaN
    for a single tile, where it would otherwise be identically 0 and read as
    perfect agreement.

    The returned frame reports `n_tiles` (used), `n_tiles_total`,
    `n_tiles_rejected_nan` and a `status` string, so a skipped tile or date is
    visible in the report rather than only in the log.
    """
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
    ref_nan = [nan_fraction(t) for t in ref_tiles]
    for i, frac in enumerate(ref_nan):
        if frac > max_tile_nan_frac:
            logger.warning(
                "reference date %s: tile %d is %.1f%% NaN (limit %.1f%%) — it is "
                "rejected for every date; pick tiles fully inside the source footprint",
                reference_date, i, 100 * frac, 100 * max_tile_nan_frac,
            )

    rows = []
    for date, path in sorted(stacks.items()):
        shifts_m = []
        rejected = 0
        for i, (ref_tile, window) in enumerate(zip(ref_tiles, windows)):
            moving = _read_tile(path, window, band_index)
            moving_nan = nan_fraction(moving)
            if ref_nan[i] > max_tile_nan_frac or moving_nan > max_tile_nan_frac:
                rejected += 1
                logger.info(
                    "%s: tile %d rejected — NaN %.1f%% (reference %.1f%%), limit %.1f%%",
                    date, i, 100 * moving_nan, 100 * ref_nan[i], 100 * max_tile_nan_frac,
                )
                continue
            drow, dcol = estimate_shift(ref_tile, moving)
            shifts_m.append((dcol * res_x, -drow * res_y))

        n_used = len(shifts_m)
        if n_used < min_tiles:
            # No estimate at all rather than a confident-looking one from one
            # or two tiles. Flagged so the date is excluded downstream.
            rows.append(
                {
                    "date": date,
                    "dx_m": float("nan"),
                    "dy_m": float("nan"),
                    "spread_m": float("nan"),
                    "n_tiles": n_used,
                    "n_tiles_total": len(windows),
                    "n_tiles_rejected_nan": rejected,
                    "flagged": True,
                    "status": f"skipped: {n_used} usable tile(s) < min_tiles={min_tiles}",
                }
            )
            logger.warning(
                "%s: only %d of %d tile(s) usable (%d rejected as NaN-heavy) — "
                "no shift estimated, date flagged",
                date, n_used, len(windows), rejected,
            )
            continue

        arr = np.asarray(shifts_m, dtype=np.float64)
        dx, dy = np.median(arr, axis=0)
        # With one tile the standard deviation is identically 0, which reads
        # as perfect agreement between tiles that were never compared.
        spread = float(np.max(np.std(arr, axis=0))) if n_used > 1 else float("nan")
        magnitude = float(np.hypot(dx, dy))
        rows.append(
            {
                "date": date,
                "dx_m": float(dx),
                "dy_m": float(dy),
                "spread_m": spread,
                "n_tiles": n_used,
                "n_tiles_total": len(windows),
                "n_tiles_rejected_nan": rejected,
                "flagged": bool(magnitude > max_shift_m),
                "status": "ok" if rejected == 0 else f"{rejected} tile(s) rejected as NaN-heavy",
            }
        )
        logger.info(
            "%s: dx=%+.3f m dy=%+.3f m |d|=%.3f m spread=%.3f m tiles %d/%d %s",
            date, dx, dy, magnitude, spread, n_used, len(windows),
            "FLAGGED" if rows[-1]["flagged"] else "",
        )
    return pd.DataFrame(rows)


def flagged_dates(report: pd.DataFrame) -> list[str]:
    """Dates whose residual shift exceeded the threshold."""
    return report.loc[report["flagged"], "date"].astype(str).tolist()
