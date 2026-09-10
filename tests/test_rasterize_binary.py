import os
import sys

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.rasterize_crowns import rasterize_binary

TRANSFORM = from_origin(0, 20, 1, 1)


def test_polygon_interior_burns_to_one():
    binary = rasterize_binary([box(5, 5, 15, 15)], 20, 20, TRANSFORM)
    assert binary.dtype == np.float32
    assert binary[10, 10] == 1.0
    assert binary[0, 0] == 0.0


def test_empty_geoms_gives_all_zeros_instead_of_raising():
    # rasterio.features.rasterize raises on an empty shape list. The deadwood
    # mask blurs two classes and either one may legitimately be empty (a scene
    # with no background polygons digitised yet), so absence has to be a valid
    # input rather than a crash.
    binary = rasterize_binary([], 20, 20, TRANSFORM)
    assert binary.shape == (20, 20)
    assert not binary.any()


def test_invalid_and_none_geometries_are_skipped():
    binary = rasterize_binary([None, box(5, 5, 15, 15)], 20, 20, TRANSFORM)
    assert binary[10, 10] == 1.0
