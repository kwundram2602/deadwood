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
    *[f"offset_{stage}_m" for stage in STAGES],
]


def crown_ring_masks(
    geometry, aoi: Aoi, grid: ReferenceGrid, ring_gap_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """The crown itself, and the ground ring around it with a gap in between.

    The gap keeps the crown's own edge pixels — half canopy, half ground — out
    of the reference level. Without it a small crown would partly define the
    ground it is measured against.
    """
    shape = (aoi.window.height, aoi.window.width)
    affine = window_transform(aoi.window, grid.transform)
    crown = rio_rasterize([(geometry, 1)], out_shape=shape, transform=affine, dtype="int32") > 0
    grown = (
        rio_rasterize(
            [(geometry.buffer(ring_gap_m), 1)],
            out_shape=shape,
            transform=affine,
            dtype="int32",
        )
        > 0
    )
    return crown, ~grown


def _level(values: np.ndarray, quantile: float) -> float:
    """A robust level for a patch, NaN when nothing finite is left."""
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, quantile)) if finite.size else np.nan


def aoi_stats(surfaces: Surfaces, aoi: Aoi, geometry, ring_gap_m: float = 2.0) -> dict:
    """Every number for one crown, measured on the AOI cut-out."""
    crown, ring = crown_ring_masks(geometry, aoi, surfaces.grid, ring_gap_m)
    dsm = crop(surfaces.dsm, aoi)

    row = {
        "tree_id": aoi.tree_id,
        "n_crown_px": int(crown.sum()),
        "n_ring_px": int(ring.sum()),
        # The production number: what topography_tree.csv reports.
        "crown_ndsm_m": _level(crop(surfaces.ndsm("aligned"), aoi)[crown], 50.0),
        # The control: no DTM in it at all.
        "crown_above_ring_m": _level(dsm[crown], 50.0) - _level(dsm[ring], GROUND_QUANTILE),
    }
    for stage in STAGES:
        # Ground beside the crown after this stage. Zero means the DTM sits on
        # the DSM's ground here; anything else lands in the nDSM unchanged.
        row[f"offset_{stage}_m"] = _level(crop(surfaces.ndsm(stage), aoi)[ring], GROUND_QUANTILE)
    return row


def stats_table(rows: list[dict]) -> pd.DataFrame:
    """The per-crown rows in a fixed column order."""
    return pd.DataFrame.from_records(rows, columns=COLUMNS)
