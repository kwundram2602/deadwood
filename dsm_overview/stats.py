"""Per-crown numbers to read alongside the 3D panels.

The panels show what happened; these say how much. The pair that decides the
question is `offset_aligned_m` (does bare ground beside the crown end up at
zero?) and `crown_above_ring_m` (does the crown stand above its surroundings in
the DSM at all, DTM not involved?).
"""

import logging

import numpy as np
import pandas as pd
from rasterio.features import rasterize as rio_rasterize
from rasterio.windows import transform as window_transform

from deadwood_spectral.grid import ReferenceGrid
from dsm_overview.surfaces import STAGES, Surfaces
from dsm_overview.window import Aoi, crop

logger = logging.getLogger(__name__)

# The ground level of a patch that also holds vegetation: low enough to sit
# under the canopy, high enough not to chase the single lowest pixel.
GROUND_QUANTILE = 10.0

COLUMNS = [
    "tree_id",
    "n_crown_px",
    "n_ring_px",
    "crown_ndsm_m",
    "crown_above_ring_m",
    "ring_method",
    *[f"offset_{stage}_m" for stage in STAGES],
    # How much of the move from offset_plane_m to offset_aligned_m the local
    # refinement is responsible for, and how much of it overshot. Without these
    # two a negative offset_aligned_m cannot be told apart from the fact that
    # `offset_*_m` is a low quantile of a half-vegetated ring, which sits below
    # the ground level by construction.
    "local_corr_m",
    "frac_dtm_above_dsm",
]

# Below this many ground candidates in the ring a plane's tilt is unconstrained
# (mirrors align_dtm_to_dsm's min_candidates, one scale down for a single AOI).
MIN_PLANE_PIXELS = 30


def crown_ring_masks(
    geometry, aoi: Aoi, grid: ReferenceGrid, ring_gap_m: float, ring_width_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """The crown itself, and an annulus of ground around it.

    The ring is the geometry buffered by `ring_gap_m` to `ring_gap_m +
    ring_width_m` — a fixed-width band, not "everything else in the AOI". It
    used to be `~grown`, i.e. the whole AOI outside the buffered crown, which
    made the statistics depend on `buffer_m` (a plot-framing knob) and, on
    sloping terrain, pulled in ground far enough from the crown to sit at a
    different elevation. The gap keeps the crown's own edge pixels — half
    canopy, half ground — out of the ring.

    If `ring_gap_m + ring_width_m` exceeds the AOI's `buffer_m`, the outer
    edge of the ring is clipped by the AOI boundary; the ring is then narrower
    than requested but still centred on the crown, which is acceptable.
    """
    shape = (aoi.window.height, aoi.window.width)
    affine = window_transform(aoi.window, grid.transform)
    crown = rio_rasterize([(geometry, 1)], out_shape=shape, transform=affine, dtype="int32") > 0
    inner = (
        rio_rasterize(
            [(geometry.buffer(ring_gap_m), 1)],
            out_shape=shape,
            transform=affine,
            dtype="int32",
        )
        > 0
    )
    outer = (
        rio_rasterize(
            [(geometry.buffer(ring_gap_m + ring_width_m), 1)],
            out_shape=shape,
            transform=affine,
            dtype="int32",
        )
        > 0
    )
    return crown, outer & ~inner


def _level(values: np.ndarray, quantile: float) -> float:
    """A robust level for a patch, NaN when nothing finite is left."""
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, quantile)) if finite.size else np.nan


def _ring_ground_level(
    dsm: np.ndarray, ring: np.ndarray, affine, centroid_xy: tuple[float, float]
) -> tuple[float, str]:
    """Ground level under the crown centroid, from a plane through the ring.

    A single quantile over the ring assumes the ring is flat. On the real
    survey's 6-9% sloping terrain the ring is wide enough (a ~30 m square at
    the default `buffer_m`) that its low quantile lands well below the ground
    actually under the crown. Fitting a plane and evaluating it at the crown
    centroid follows the slope instead of averaging across it.

    The re-fit loop mirrors `apply_dsm_mask.align_dtm_to_dsm`'s ground-plane
    estimator one scale down: MAD-trim against a plane over 2-3 iterations
    rather than a single quantile, so the package uses one idea at two
    scales. Falls back to `percentile(DSM[ring], GROUND_QUANTILE)` — today's
    behaviour — when the ring does not offer enough ground candidates to
    constrain a plane; the caller must surface that fallback, since a silent
    one reintroduces exactly the bias this function exists to remove.
    """
    h, w = ring.shape
    rows, cols = np.nonzero(ring)
    values = dsm[rows, cols]
    finite = np.isfinite(values)
    rows, cols, values = rows[finite], cols[finite], values[finite]

    # Ground candidates: ring pixels at or below their own median, the same
    # idea as apply_dsm_mask's per-block ground candidates one scale up.
    if values.size >= MIN_PLANE_PIXELS:
        ground = values <= np.median(values)
        rows, cols, values = rows[ground], cols[ground], values[ground]

    if values.size < MIN_PLANE_PIXELS:
        logger.warning(
            "ring has only %d ground candidate(s) (need %d) — falling back to "
            "percentile(DSM[ring], %.0f) for crown_above_ring_m",
            values.size,
            MIN_PLANE_PIXELS,
            GROUND_QUANTILE,
        )
        return _level(dsm[ring], GROUND_QUANTILE), "percentile"

    # normalised coordinates in [-1, 1] keep the least-squares well conditioned
    yy = rows.astype(np.float64) * (2.0 / max(h - 1, 1)) - 1.0
    xx = cols.astype(np.float64) * (2.0 / max(w - 1, 1)) - 1.0
    values = values.astype(np.float64)

    keep = np.ones(values.shape, dtype=bool)
    coef = np.array([0.0, 0.0, np.median(values)])
    for _ in range(3):
        design = np.column_stack([xx[keep], yy[keep], np.ones(int(keep.sum()))])
        coef, *_ = np.linalg.lstsq(design, values[keep], rcond=None)
        resid = values - (coef[0] * xx + coef[1] * yy + coef[2])
        centre = np.median(resid)
        mad = 1.4826 * np.median(np.abs(resid - centre))
        tol = max(2.5 * mad, 0.25)  # 0.25 m floor: real ground is rough
        new_keep = np.abs(resid - centre) <= tol
        if new_keep.sum() < MIN_PLANE_PIXELS or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    # crown centroid in the same normalised pixel-coordinate frame
    col, row = ~affine * centroid_xy
    cx = col * (2.0 / max(w - 1, 1)) - 1.0
    cy = row * (2.0 / max(h - 1, 1)) - 1.0
    return float(coef[0] * cx + coef[1] * cy + coef[2]), "plane"


def aoi_stats(
    surfaces: Surfaces,
    aoi: Aoi,
    geometry,
    ring_gap_m: float = 2.0,
    ring_width_m: float = 8.0,
) -> dict:
    """Every number for one crown, measured on the AOI cut-out."""
    crown, ring = crown_ring_masks(geometry, aoi, surfaces.grid, ring_gap_m, ring_width_m)
    dsm = crop(surfaces.dsm, aoi)
    affine = window_transform(aoi.window, surfaces.grid.transform)
    ground_level, ring_method = _ring_ground_level(
        dsm, ring, affine, (geometry.centroid.x, geometry.centroid.y)
    )

    row = {
        "tree_id": aoi.tree_id,
        "n_crown_px": int(crown.sum()),
        "n_ring_px": int(ring.sum()),
        # The production number: what topography_tree.csv reports.
        "crown_ndsm_m": _level(surfaces.ndsm_window("aligned", aoi)[crown], 50.0),
        # The control: no DTM in it at all.
        "crown_above_ring_m": _level(dsm[crown], 50.0) - ground_level,
        # Which of the two methods above produced crown_above_ring_m — the
        # fallback must be visible, not silent, or the bias comes back
        # unnoticed on the closed-canopy crowns.
        "ring_method": ring_method,
    }
    for stage in STAGES:
        # Ground beside the crown after this stage. Zero means the DTM sits on
        # the DSM's ground here; anything else lands in the nDSM unchanged.
        row[f"offset_{stage}_m"] = _level(surfaces.ndsm_window(stage, aoi)[ring], GROUND_QUANTILE)

    # What the local refinement added on top of the plane, right here.
    local = crop(surfaces.dtm["aligned"], aoi)[ring] - crop(surfaces.dtm["plane"], aoi)[ring]
    row["local_corr_m"] = float(np.nanmedian(local)) if local.size else np.nan
    # Ring pixels the aligned DTM ends up above — physically impossible, so a
    # non-zero fraction is a real bias and not a quantile artefact.
    aligned_ndsm = surfaces.ndsm_window("aligned", aoi)[ring]
    finite = aligned_ndsm[np.isfinite(aligned_ndsm)]
    row["frac_dtm_above_dsm"] = float((finite < 0).mean()) if finite.size else np.nan
    return row


def stats_table(rows: list[dict]) -> pd.DataFrame:
    """The per-crown rows in a fixed column order."""
    return pd.DataFrame.from_records(rows, columns=COLUMNS)
