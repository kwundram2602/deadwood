"""Cut-outs around a single crown, small enough to draw as a surface.

A 3D surface of the full survey is 45 million faces and says nothing. The
question is local anyway: whether the ground under *this* crown sits at zero
after the DTM was lifted onto the DSM.
"""

import logging
from dataclasses import dataclass

import numpy as np
from rasterio.windows import Window

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

    return Aoi(tree_id, Window(col_off, row_off, col_end - col_off, row_end - row_off))


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
