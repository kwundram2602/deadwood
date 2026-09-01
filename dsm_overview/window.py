"""Cut-outs around a single crown, small enough to draw as a surface.

A 3D surface of the full survey is 45 million faces and says nothing. The
question is local anyway: whether the ground under *this* crown sits at zero
after the DTM was lifted onto the DSM.
"""

import logging
from dataclasses import dataclass

import numpy as np
from rasterio.features import rasterize as rio_rasterize
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from deadwood_spectral.grid import ReferenceGrid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Aoi:
    """One crown's neighbourhood on the reference grid. Integer pixels only."""

    tree_id: str
    window: Window


def aoi_from_bounds(
    bounds: tuple[float, float, float, float],
    grid: ReferenceGrid,
    buffer_m: float,
    tree_id: str = "",
) -> Aoi:
    """Map bounds plus a buffer to whole pixels, clipped to the grid.

    The buffer is what makes the cut-out readable: a window flush with the
    crown shows only crown, and the whole point is to see the ground it stands
    on next to it.
    """
    minx, miny, maxx, maxy = bounds
    inverse = ~grid.transform
    left, top = inverse * (minx - buffer_m, maxy + buffer_m)
    right, bottom = inverse * (maxx + buffer_m, miny - buffer_m)

    col_off = max(0, int(np.floor(left)))
    row_off = max(0, int(np.floor(top)))
    col_end = min(grid.width, int(np.ceil(right)))
    row_end = min(grid.height, int(np.ceil(bottom)))
    if col_end <= col_off or row_end <= row_off:
        raise ValueError(f"AOI {tree_id or bounds} lies outside the reference grid")

    window_args = {
        "col_off": col_off,
        "row_off": row_off,
        "width": col_end - col_off,
        "height": row_end - row_off,
    }
    # Built as a dict and splatted rather than passed as explicit keywords:
    # ty cannot model rasterio's attrs-generated Window, and explicit keywords
    # produce four unknown-argument errors where positional args produce one
    # too-many-positional-arguments error. This is the smaller workaround, not
    # an oversight — do not "clean it up" to explicit keywords.
    return Aoi(tree_id, Window(**window_args))


def aoi_transform(aoi: Aoi, grid: ReferenceGrid):
    """The affine of the AOI cut-out, i.e. the grid's transform shifted to it."""
    return window_transform(aoi.window, grid.transform)


def rasterize_geometry(geometry, aoi: Aoi, grid: ReferenceGrid) -> np.ndarray:
    """Boolean AOI-sized mask of one geometry, on the AOI's own pixel grid."""
    shape = (aoi.window.height, aoi.window.width)
    burnt = rio_rasterize(
        [(geometry, 1)], out_shape=shape, transform=aoi_transform(aoi, grid), dtype="int32"
    )
    return burnt > 0


def geometry_pixels(geometry, aoi: Aoi, grid: ReferenceGrid) -> list[tuple[np.ndarray, np.ndarray]]:
    """The geometry's outlines as fractional (col, row) rings on the AOI grid.

    One entry per ring, so a MultiPolygon draws as several closed curves rather
    than one line jumping between parts.
    """
    inverse = ~aoi_transform(aoi, grid)
    parts = getattr(geometry, "geoms", [geometry])
    rings = []
    for part in parts:
        exterior = getattr(part, "exterior", None)
        if exterior is None:
            continue
        xs, ys = np.asarray(exterior.coords).T
        cols, rows = inverse * (xs, ys)
        rings.append((np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)))
    return rings


def crop(array: np.ndarray, aoi: Aoi) -> np.ndarray:
    """The AOI's block of a full-scene array."""
    window = aoi.window
    return array[
        window.row_off : window.row_off + window.height,
        window.col_off : window.col_off + window.width,
    ]


def decimate(array: np.ndarray, max_side: int) -> tuple[np.ndarray, int]:
    """Thin by striding until no side exceeds `max_side`.

    Striding rather than averaging: a mean would smooth away exactly the sharp
    crown-to-ground step that has to be judged here.
    """
    step = max(1, int(np.ceil(max(array.shape) / max_side)))
    return array[::step, ::step], step


def patch_coordinates(aoi: Aoi, grid: ReferenceGrid, step: int) -> tuple[np.ndarray, np.ndarray]:
    """X/Y in metres from the AOI's first pixel centre, matching a strided patch."""
    res_x = abs(grid.transform.a)
    res_y = abs(grid.transform.e)
    rows = np.arange(0, aoi.window.height, step, dtype=np.float32) * res_y
    cols = np.arange(0, aoi.window.width, step, dtype=np.float32) * res_x
    return np.meshgrid(cols, rows)
