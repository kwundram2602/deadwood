"""One crown, four 3D panels, read left to right.

The first three panels hold the DSM and one DTM stage in the same axes and the
same z range, so the vertical gap between them is the thing the eye measures.
The fourth is the product: DSM - DTM_aligned, with the ground threshold drawn
in as a plane. A crown that stands clear in panel 1 but sinks under the plane
in panel 4 was lost by the co-registration, not by the photogrammetry. The
fifth panel is the control the other four cannot give: DTM_aligned - DSM, where
anything above zero is a DTM lifted through the surface it was fitted to.

Where the crown polygon is known it is drawn onto the DSM and the nDSM — the
pixels inside it tinted, its outline traced as a line lying on the surface.
Without that mark a mound in the patch cannot be told apart from the neighbour
tree next to it, which is the one thing these panels have to answer.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the HPC
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, to_rgb  # noqa: E402

from dsm_overview.surfaces import STAGES, Surfaces  # noqa: E402
from dsm_overview.window import (  # noqa: E402
    Aoi,
    crop,
    decimate,
    geometry_pixels,
    patch_coordinates,
    rasterize_geometry,
)

logger = logging.getLogger(__name__)

STAGE_TITLES = {
    "raw": "DSM vs DTM as delivered",
    "plane": "DSM vs DTM after the global plane",
    "aligned": "DSM vs DTM after plane + local refine",
}
DSM_COLOR = "#c0392b"
DTM_COLOR = "#2c3e50"
CROWN_COLOR = "#f4d03f"
OVERSHOOT_COLOR = "#8e44ad"
OUTLINE_COLOR = "#1b2631"

# Steeper than matplotlib's default 30 deg: from higher up the crown reads as a
# patch on the ground plane, which is what the outline is there to show, while
# still oblique enough to keep the DSM-to-DTM gap visible as a gap.
DEFAULT_ELEV = 45.0
DEFAULT_AZIM = -60.0

# The outline is lifted off the surface it traces, or the surface's own faces
# hide it wherever the crown is convex towards the viewer.
OUTLINE_LIFT_FRAC = 0.02

# Explicit draw order, in force only because `computed_zorder` is switched off
# per axes (see `_new_axes`). Insertion order does not decide anything here.
Z_DTM = 1
Z_DSM = 2
Z_OUTLINE = 3

_LIGHT = LightSource(azdeg=315, altdeg=45)


PANELS = 5


def _new_axes(fig, index: int, elev: float, azim: float):
    """One 3D panel with depth sorting switched off.

    mplot3d sorts whole collections by their mean depth, and by that measure a
    DTM plane spanning the patch beats a DSM whose mean sits near the ground —
    so the plane is painted over the crown tint even where the crown clearly
    stands above it. `computed_zorder = False` replaces that with the explicit
    `Z_*` order, which is the only way to keep the mark readable.

    The cost is real and deliberate: the DSM now covers the DTM everywhere,
    including the spots where the aligned DTM has been lifted *above* the DSM.
    That overshoot is no longer visible in these panels; it is measured, not
    eyeballed, in `dem_offsets.csv` — `frac_dtm_above_dsm` and the gap between
    `offset_plane_m` and `offset_aligned_m`.
    """
    ax = fig.add_subplot(1, PANELS, index, projection="3d")
    ax.computed_zorder = False
    ax.view_init(elev=elev, azim=azim)
    return ax


def _shaded_colors(z: np.ndarray, base: str, mask: np.ndarray | None, highlight: str):
    """Per-face RGB for a surface, with `mask` pixels tinted.

    `plot_surface`'s own `shade` is unavailable once facecolors are given, so
    the relief is baked in here instead — otherwise a tinted surface arrives
    flat and unreadable.
    """
    rgb = np.empty(z.shape + (3,), dtype=float)
    rgb[...] = to_rgb(base)
    if mask is not None:
        rgb[mask] = to_rgb(highlight)
    # LightSource differentiates the elevation, so NaN holes would smear over
    # their neighbours; they are filled for the shading only, and the faces
    # themselves stay undrawn because Z still holds the NaN.
    finite = np.isfinite(z)
    filled = np.where(finite, z, np.nanmedian(z[finite]) if finite.any() else 0.0)
    return _LIGHT.shade_rgb(rgb, filled, blend_mode="soft", vert_exag=1.0)


def _surface(ax, x, y, z, color, alpha, label, mask=None, zorder=1, highlight=CROWN_COLOR):
    if mask is None:
        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=alpha,
            linewidth=0,
            antialiased=False,
            shade=True,
            zorder=zorder,
        )
    else:
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=_shaded_colors(z, color, mask, highlight),
            alpha=alpha,
            linewidth=0,
            antialiased=False,
            shade=False,
            zorder=zorder,
        )
    # plot_surface produces no legend handle, so the label is carried by a
    # zero-length proxy line instead.
    ax.plot([], [], color=color, label=label)


def _outline(ax, rings, surface: np.ndarray, res_x: float, res_y: float, lift: float, label=None):
    """Trace the crown boundary along the surface it sits on.

    The z of every vertex is read from the full-resolution surface rather than
    the strided one the faces are drawn from, so the line follows the canopy
    even where the plot itself is thinned by a factor of ten.
    """
    height, width = surface.shape
    for index, (cols, rows) in enumerate(rings):
        rr = np.clip(np.round(rows).astype(int), 0, height - 1)
        cc = np.clip(np.round(cols).astype(int), 0, width - 1)
        z = surface[rr, cc].astype(float)
        if not np.isfinite(z).any():
            continue
        z[~np.isfinite(z)] = np.nanmedian(z[np.isfinite(z)])
        ax.plot(
            cols * res_x,
            rows * res_y,
            z + lift,
            color=OUTLINE_COLOR,
            linewidth=2.0,
            zorder=Z_OUTLINE,
            label=label if index == 0 else None,
        )


def dem_figure(
    surfaces: Surfaces,
    aoi: Aoi,
    height_threshold: float = 0.5,
    max_side: int = 200,
    geometry=None,
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
):
    """The four panels for one crown.

    `geometry` is the crown polygon in the grid's CRS. It is optional so the
    figure still builds from a bare AOI, but every production call has it and
    without it the panels show a mound with no way to say which tree it is.
    """
    dsm_full = crop(surfaces.dsm, aoi)
    dsm, step = decimate(dsm_full, max_side)
    x, y = patch_coordinates(aoi, surfaces.grid, step)
    stages = {stage: decimate(crop(surfaces.dtm[stage], aoi), max_side)[0] for stage in STAGES}
    ndsm_full = surfaces.ndsm_window("aligned", aoi)
    ndsm = decimate(ndsm_full, max_side)[0]

    crown_mask = None
    rings: list = []
    if geometry is not None:
        crown_mask = rasterize_geometry(geometry, aoi, surfaces.grid)[::step, ::step]
        rings = geometry_pixels(geometry, aoi, surfaces.grid)
    res_x = abs(surfaces.grid.transform.a)
    res_y = abs(surfaces.grid.transform.e)

    stack = np.concatenate([dsm.ravel(), *[s.ravel() for s in stages.values()]])
    finite = stack[np.isfinite(stack)]
    zlim = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)
    lift = OUTLINE_LIFT_FRAC * max(zlim[1] - zlim[0], 1.0)

    fig = plt.figure(figsize=(22, 5))
    for index, stage in enumerate(STAGES, start=1):
        ax = _new_axes(fig, index, elev, azim)
        _surface(ax, x, y, dsm, DSM_COLOR, 0.95, "DSM", mask=crown_mask, zorder=Z_DSM)
        _surface(ax, x, y, stages[stage], DTM_COLOR, 0.55, f"DTM ({stage})", zorder=Z_DTM)
        _outline(ax, rings, dsm_full, res_x, res_y, lift, label="crown polygon")
        ax.set_zlim(*zlim)
        ax.set_title(STAGE_TITLES[stage], fontsize=9)
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.set_zlabel("elevation [m]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=7, loc="upper left")

    ax = _new_axes(fig, 4, elev, azim)
    _surface(ax, x, y, ndsm, DSM_COLOR, 0.95, "nDSM", mask=crown_mask, zorder=Z_DSM)
    _surface(
        ax,
        x,
        y,
        np.full_like(ndsm, height_threshold),
        "#7f8c8d",
        0.35,
        f"ground threshold {height_threshold:g} m",
        zorder=Z_DTM,
    )
    _outline(ax, rings, ndsm_full, res_x, res_y, lift, label="crown polygon")
    ax.set_title("product: nDSM = DSM - DTM (aligned)", fontsize=9)
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.set_zlabel("height above ground [m]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper left")

    # Panel 5 exists because panels 1-3 can no longer show this: with depth
    # sorting off the DSM covers the DTM everywhere, including where the
    # aligned DTM was lifted above it. Here that overshoot is the whole subject
    # and nothing hides it.
    overshoot = -ndsm
    ax = _new_axes(fig, 5, elev, azim)
    above = overshoot > 0
    _surface(
        ax,
        x,
        y,
        overshoot,
        DTM_COLOR,
        0.95,
        "DTM (aligned) - DSM",
        mask=above,
        zorder=Z_DSM,
        highlight=OVERSHOOT_COLOR,
    )
    _surface(ax, x, y, np.zeros_like(overshoot), "#7f8c8d", 0.35, "DSM surface", zorder=Z_DTM)
    _outline(ax, rings, -ndsm_full, res_x, res_y, lift, label="crown polygon")
    # Scaled to the overshoot, not to the data: the crown itself reaches tens of
    # metres below zero here and would flatten the half-metre that matters into
    # a line. Faces outside the range are clipped, which is the intent.
    positive = overshoot[np.isfinite(overshoot) & above]
    limit = max(0.5, float(np.percentile(positive, 99))) if positive.size else 0.5
    ax.set_zlim(-limit, limit)
    ax.set_title("control: DTM above DSM (purple = impossible)", fontsize=9)
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.set_zlabel("DTM - DSM [m]", fontsize=7)
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
    geometry=None,
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
) -> Path:
    """Build the figure for one crown and write it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = dem_figure(surfaces, aoi, height_threshold, max_side, geometry, elev, azim)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path
