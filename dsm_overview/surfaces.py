"""DSM and DTM on the reference grid, plus the DTM stages in between.

`apply_dsm_mask` lifts an external DTM onto the DSM in two steps — a global
plane through the scene's ground candidates, then a blockwise residual warp of
at most +-1 m — and subtracts the result. Both steps are estimated from the
whole scene, so both are reproduced here at full extent and only cropped
afterwards. Cropping first would fit a different plane and answer a different
question.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from deadwood_spectral.grid import (
    ReferenceGrid,
    assert_matches_grid,
    load_reference_grid,
)
from dsm_overview.window import Aoi, crop

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explore_and_process.apply_dsm_mask import (  # noqa: E402
    align_dtm_stages,
    resample_raster,
)

logger = logging.getLogger(__name__)

# raw: the DTM as delivered. plane: after the global levelling. aligned: after
# the local refinement — the surface the production nDSM is actually built on.
STAGES: tuple[str, ...] = ("raw", "plane", "aligned")


@dataclass(frozen=True)
class Surfaces:
    """One DSM and the three DTM stages, all on the reference grid."""

    grid: ReferenceGrid
    dsm: np.ndarray
    dtm: dict[str, np.ndarray]
    info: dict[str, dict]

    def ndsm(self, stage: str) -> np.ndarray:
        """DSM - DTM for one stage, NaN wherever either input is NaN."""
        if stage not in self.dtm:
            raise KeyError(f"unknown stage {stage!r}; have {list(self.dtm)}")
        out = (self.dsm - self.dtm[stage]).astype(np.float32)
        out[np.isnan(self.dsm) | np.isnan(self.dtm[stage])] = np.nan
        return out

    def ndsm_window(self, stage: str, aoi: Aoi) -> np.ndarray:
        """`ndsm(stage)` cropped to one AOI, without ever building the full array.

        `ndsm(stage)` allocates a full-scene 45-million-pixel array; every
        caller only wants an AOI-sized cut-out of it. Cropping the DSM and the
        DTM stage first, then subtracting, gets the same numbers at AOI size
        instead of scene size — at 18 crowns the difference is 90 full-scene
        allocations versus 90 AOI-sized ones.
        """
        if stage not in self.dtm:
            raise KeyError(f"unknown stage {stage!r}; have {list(self.dtm)}")
        dsm = crop(self.dsm, aoi)
        dtm = crop(self.dtm[stage], aoi)
        out = (dsm - dtm).astype(np.float32)
        out[np.isnan(dsm) | np.isnan(dtm)] = np.nan
        return out


def build_surfaces(
    dsm: np.ndarray,
    dtm: np.ndarray,
    grid: ReferenceGrid,
    local_blocks: int = 12,
    clamp_to_dsm: bool = False,
) -> Surfaces:
    """Run the co-registration once and keep both stages it produces.

    `align_dtm_stages` fits the plane once and returns plane and aligned from
    it, so the two surfaces differ by exactly the local correction and nothing
    else — which is the whole point of comparing them here.

    `clamp_to_dsm` defaults to False, the opposite of production: the
    diagnostic has to be able to see how far the refinement over-lifts the DTM,
    and a clamp would hide exactly that.
    """
    dsm = dsm.astype(np.float32)
    dtm = dtm.astype(np.float32)

    stages, infos = align_dtm_stages(dsm, dtm, local_blocks=local_blocks, clamp_to_dsm=clamp_to_dsm)
    logger.info(
        "plane: shift %+.2f m, tilt %.2f m | local: RMS %.2f m over %d block(s) "
        "| DTM above DSM: %d px (plane) / %d px (aligned)",
        infos["plane"]["mean_shift"],
        infos["plane"]["tilt"],
        infos["aligned"]["local_rms"],
        infos["aligned"]["local_blocks"],
        infos["plane"]["n_above_dsm"],
        infos["aligned"]["n_above_dsm"],
    )
    return Surfaces(
        grid=grid,
        dsm=dsm,
        dtm={"raw": dtm, **stages},
        info={"raw": {}, **infos},
    )


def load_surfaces(
    reference: str | Path,
    dsm_path: str | Path,
    dtm_path: str | Path,
    local_blocks: int = 12,
    clamp_to_dsm: bool = False,
) -> Surfaces:
    """Read both rasters onto the reference grid and build every stage.

    Bilinear onto the crown-mask grid, exactly as apply_dsm_mask does it — the
    3.67 cm DSM is downsampled, the 0.5 m DTM interpolated up by a factor of
    ten, and that interpolation is itself one of the suspects.
    """
    grid = load_reference_grid(reference)
    logger.info("reference grid %s from %s", grid.shape, reference)
    dsm = resample_raster(str(dsm_path), grid.height, grid.width, grid.transform, grid.crs)
    dtm = resample_raster(str(dtm_path), grid.height, grid.width, grid.transform, grid.crs)
    return build_surfaces(dsm, dtm, grid, local_blocks, clamp_to_dsm)


def _read_on_grid(path: str | Path, grid: ReferenceGrid, context: str) -> np.ndarray:
    """One band of a raster that is already on the reference grid.

    Deliberately not `resample_raster`: a stage raster that does not match the
    grid means the `dtm_coreg` run directory belongs to a different mask, and
    resampling it would quietly compare two unrelated scenes. `assert_matches_grid`
    turns that into a ValueError naming the stage.
    """
    with rasterio.open(path) as src:
        assert_matches_grid(src, grid, context)
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None and not np.isnan(nodata):
        array[array == nodata] = np.nan
    return array


def load_surfaces_from_stages(
    reference: str | Path,
    dsm_path: str | Path,
    dtm_path: str | Path,
    plane_path: str | Path,
    aligned_path: str | Path,
) -> Surfaces:
    """The same three stages, but read from disk instead of re-fitted.

    `apply_dsm_mask` already wrote `plane` and `aligned` to
    `process_out/dtm_coreg/<run_id>/`, on the reference grid, from the inputs
    and settings production actually used — including `clamp_to_dsm`, which
    this package's own fit deliberately switches off. Re-running the fit here
    would therefore answer a subtly different question than the one the mask
    was built on, so the stages are taken verbatim and only `raw` and the DSM,
    which live on their own grids, are resampled.

    `info` stays empty for every stage: no fit ran, so there is no `mean_shift`
    or `local_rms` to report, and inventing one would be worse than its absence.
    """
    grid = load_reference_grid(reference)
    logger.info("reference grid %s from %s", grid.shape, reference)
    dsm = resample_raster(str(dsm_path), grid.height, grid.width, grid.transform, grid.crs)
    raw = resample_raster(str(dtm_path), grid.height, grid.width, grid.transform, grid.crs)
    stages = {
        "raw": raw,
        "plane": _read_on_grid(plane_path, grid, "DTM stage plane"),
        "aligned": _read_on_grid(aligned_path, grid, "DTM stage aligned"),
    }
    for name, stage in stages.items():
        finite = stage[np.isfinite(stage)]
        logger.info(
            "stage %-7s median %.2f m over %d valid px",
            name,
            float(np.median(finite)) if finite.size else np.nan,
            finite.size,
        )
    return Surfaces(
        grid=grid,
        dsm=dsm.astype(np.float32),
        dtm=stages,
        info={stage: {} for stage in STAGES},
    )
