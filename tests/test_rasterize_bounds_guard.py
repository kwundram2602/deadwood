import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from explore_and_process.rasterize_crowns import assert_same_bounds  # noqa: E402


def _write(path, left, top, res=1.0, size=4):
    profile = dict(
        driver="GTiff", dtype="uint16", width=size, height=size, count=1,
        crs="EPSG:32736", transform=from_origin(left, top, res, res),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ones((1, size, size), dtype="uint16"))


def test_matching_bounds_pass(tmp_path):
    ref = tmp_path / "ref.tif"
    same = tmp_path / "same.tif"
    _write(ref, 1000.0, 2000.0)
    _write(same, 1000.0, 2000.0)
    with rasterio.open(ref) as r, rasterio.open(same) as s:
        assert_same_bounds(s, r.bounds)  # must not raise


def test_shifted_bounds_raise(tmp_path):
    ref = tmp_path / "ref.tif"
    shifted = tmp_path / "shifted.tif"
    _write(ref, 1000.0, 2000.0)
    _write(shifted, 1002.08, 2001.26)
    with rasterio.open(ref) as r, rasterio.open(shifted) as s:
        with pytest.raises(ValueError, match="bounds"):
            assert_same_bounds(s, r.bounds)
