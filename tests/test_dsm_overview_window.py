"""AOI cut-outs around a crown: a 3D surface plot cannot take 45 million pixels."""

import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402
from dsm_overview.window import (  # noqa: E402
    aoi_from_bounds,
    crop,
    decimate,
    patch_coordinates,
)

# 100 x 100 px at 0.5 m, upper-left corner at (1000, 2000).
TRANSFORM = from_origin(1000.0, 2000.0, 0.5, 0.5)


@pytest.fixture
def grid():
    return ReferenceGrid(100, 100, TRANSFORM, rasterio.crs.CRS.from_string("EPSG:32736"))


def test_the_window_covers_the_bounds_plus_the_buffer(grid):
    # x 1010..1015 is cols 20..30, y 1990..1985 is rows 20..30.
    aoi = aoi_from_bounds((1010.0, 1985.0, 1015.0, 1990.0), grid, buffer_m=2.5, tree_id="4157")

    assert aoi.tree_id == "4157"
    assert (aoi.window.col_off, aoi.window.row_off) == (15, 15)
    assert (aoi.window.width, aoi.window.height) == (20, 20)


def test_the_window_is_clipped_to_the_grid(grid):
    aoi = aoi_from_bounds((1000.0, 1999.0, 1002.0, 2000.0), grid, buffer_m=50.0)

    assert aoi.window.col_off == 0
    assert aoi.window.row_off == 0
    assert aoi.window.col_off + aoi.window.width <= grid.width
    assert aoi.window.row_off + aoi.window.height <= grid.height


def test_bounds_outside_the_grid_are_rejected(grid):
    with pytest.raises(ValueError, match="outside"):
        aoi_from_bounds((9000.0, 9000.0, 9001.0, 9001.0), grid, buffer_m=1.0)


def test_crop_returns_exactly_the_window(grid):
    array = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    aoi = aoi_from_bounds((1010.0, 1985.0, 1015.0, 1990.0), grid, buffer_m=2.5)

    cut = crop(array, aoi)

    assert cut.shape == (20, 20)
    np.testing.assert_array_equal(cut, array[15:35, 15:35])


def test_decimate_keeps_a_small_patch_untouched():
    array = np.zeros((30, 40), dtype=np.float32)
    thinned, step = decimate(array, max_side=200)
    assert step == 1
    assert thinned.shape == (30, 40)


def test_decimate_thins_a_large_patch_below_the_limit():
    array = np.zeros((1000, 600), dtype=np.float32)
    thinned, step = decimate(array, max_side=200)
    assert step == 5
    assert max(thinned.shape) <= 200


def test_the_coordinates_span_the_patch_in_metres(grid):
    aoi = aoi_from_bounds((1010.0, 1985.0, 1015.0, 1990.0), grid, buffer_m=2.5)

    x, y = patch_coordinates(aoi, grid, step=1)

    assert x.shape == (20, 20)
    # 20 px at 0.5 m, measured from the first pixel centre.
    assert x[0, -1] == pytest.approx(9.5)
    assert y[-1, 0] == pytest.approx(9.5)
