import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import (  # noqa: E402
    assert_matches_grid,
    load_reference_grid,
)


def _write(path, left=1000.0, top=2000.0, res=0.05, size=8, crs="EPSG:32736", count=1):
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=size,
        height=size,
        count=count,
        crs=crs,
        transform=from_origin(left, top, res, res),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((count, size, size), dtype="float32"))
    return path


def test_load_reference_grid_reads_shape_and_crs(tmp_path):
    grid = load_reference_grid(_write(tmp_path / "ref.tif", size=8))
    assert grid.shape == (8, 8)
    assert grid.crs.to_epsg() == 32736
    assert grid.transform.a == pytest.approx(0.05)


def test_identical_raster_matches(tmp_path):
    grid = load_reference_grid(_write(tmp_path / "ref.tif"))
    with rasterio.open(_write(tmp_path / "same.tif", count=7)) as src:
        assert_matches_grid(src, grid, "same.tif")  # must not raise


def test_shifted_transform_raises(tmp_path):
    grid = load_reference_grid(_write(tmp_path / "ref.tif"))
    with rasterio.open(_write(tmp_path / "shift.tif", left=1002.08)) as src:
        with pytest.raises(ValueError, match="transform"):
            assert_matches_grid(src, grid, "shift.tif")


def test_wrong_shape_raises(tmp_path):
    grid = load_reference_grid(_write(tmp_path / "ref.tif", size=8))
    with rasterio.open(_write(tmp_path / "big.tif", size=16)) as src:
        with pytest.raises(ValueError, match="shape"):
            assert_matches_grid(src, grid, "big.tif")


def test_wrong_crs_raises(tmp_path):
    grid = load_reference_grid(_write(tmp_path / "ref.tif"))
    with rasterio.open(_write(tmp_path / "wgs.tif", crs="EPSG:4326")) as src:
        with pytest.raises(ValueError, match="CRS"):
            assert_matches_grid(src, grid, "wgs.tif")
