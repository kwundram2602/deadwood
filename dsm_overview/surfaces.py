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

from deadwood_spectral.grid import ReferenceGrid, load_reference_grid
from dsm_overview.window import Aoi, crop

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explore_and_process.apply_dsm_mask import (  # noqa: E402
    align_dtm_to_dsm,
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


def build_surfaces(dsm: np.ndarray, dtm: np.ndarray, grid: ReferenceGrid) -> Surfaces:
    """Run the co-registration twice: once plane-only, once with the refinement.

    Twice rather than once because the two stages are what is being compared.
    The plane-only pass is the same call with local_refine off, so the two
    surfaces differ by exactly the local correction and nothing else.
    """
    dsm = dsm.astype(np.float32)
    dtm = dtm.astype(np.float32)

    plane, plane_info = align_dtm_to_dsm(dsm, dtm, local_refine=False)
    aligned, aligned_info = align_dtm_to_dsm(dsm, dtm, local_refine=True)
    logger.info(
        "plane: shift %+.2f m, tilt %.2f m | local: RMS %.2f m over %d block(s)",
        plane_info["mean_shift"],
        plane_info["tilt"],
        aligned_info["local_rms"],
        aligned_info["local_blocks"],
    )
    return Surfaces(
        grid=grid,
        dsm=dsm,
        dtm={"raw": dtm, "plane": plane, "aligned": aligned},
        info={"raw": {}, "plane": plane_info, "aligned": aligned_info},
    )


def load_surfaces(reference: str | Path, dsm_path: str | Path, dtm_path: str | Path) -> Surfaces:
    """Read both rasters onto the reference grid and build every stage.

    Bilinear onto the crown-mask grid, exactly as apply_dsm_mask does it — the
    3.67 cm DSM is downsampled, the 0.5 m DTM interpolated up by a factor of
    ten, and that interpolation is itself one of the suspects.
    """
    grid = load_reference_grid(reference)
    logger.info("reference grid %s from %s", grid.shape, reference)
    dsm = resample_raster(str(dsm_path), grid.height, grid.width, grid.transform, grid.crs)
    dtm = resample_raster(str(dtm_path), grid.height, grid.width, grid.transform, grid.crs)
    return build_surfaces(dsm, dtm, grid)
