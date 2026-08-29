"""One crown, four 3D panels, read left to right.

The first three panels hold the DSM and one DTM stage in the same axes and the
same z range, so the vertical gap between them is the thing the eye measures.
The fourth is the product: DSM - DTM_aligned, with the ground threshold drawn
in as a plane. A crown that stands clear in panel 1 but sinks under the plane
in panel 4 was lost by the co-registration, not by the photogrammetry.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the HPC
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dsm_overview.surfaces import STAGES, Surfaces  # noqa: E402
from dsm_overview.window import Aoi, crop, decimate, patch_coordinates  # noqa: E402

logger = logging.getLogger(__name__)

STAGE_TITLES = {
    "raw": "DSM vs DTM as delivered",
    "plane": "DSM vs DTM after the global plane",
    "aligned": "DSM vs DTM after plane + local refine",
}
DSM_COLOR = "#c0392b"
DTM_COLOR = "#2c3e50"


def _surface(ax, x, y, z, color, alpha, label):
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=False, shade=True)
    # plot_surface produces no legend handle, so the label is carried by a
    # zero-length proxy line instead.
    ax.plot([], [], color=color, label=label)


def dem_figure(surfaces: Surfaces, aoi: Aoi, height_threshold: float = 0.5, max_side: int = 200):
    """The four panels for one crown."""
    dsm, step = decimate(crop(surfaces.dsm, aoi), max_side)
    x, y = patch_coordinates(aoi, surfaces.grid, step)
    stages = {stage: decimate(crop(surfaces.dtm[stage], aoi), max_side)[0] for stage in STAGES}
    ndsm = decimate(surfaces.ndsm_window("aligned", aoi), max_side)[0]

    stack = np.concatenate([dsm.ravel(), *[s.ravel() for s in stages.values()]])
    finite = stack[np.isfinite(stack)]
    zlim = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)

    fig = plt.figure(figsize=(18, 5))
    for index, stage in enumerate(STAGES, start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        _surface(ax, x, y, dsm, DSM_COLOR, 0.95, "DSM")
        _surface(ax, x, y, stages[stage], DTM_COLOR, 0.55, f"DTM ({stage})")
        ax.set_zlim(*zlim)
        ax.set_title(STAGE_TITLES[stage], fontsize=9)
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.set_zlabel("elevation [m]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    _surface(ax, x, y, ndsm, DSM_COLOR, 0.95, "nDSM")
    _surface(
        ax,
        x,
        y,
        np.full_like(ndsm, height_threshold),
        "#7f8c8d",
        0.35,
        f"ground threshold {height_threshold:g} m",
    )
    ax.set_title("product: nDSM = DSM - DTM (aligned)", fontsize=9)
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.set_zlabel("height above ground [m]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"tree {aoi.tree_id} — every {step}th pixel", fontsize=11)
    fig.tight_layout()
    return fig


def plot_dem_overview(
    surfaces: Surfaces,
    aoi: Aoi,
    path: str | Path,
    height_threshold: float = 0.5,
    max_side: int = 200,
) -> Path:
    """Build the figure for one crown and write it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = dem_figure(surfaces, aoi, height_threshold, max_side)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path
