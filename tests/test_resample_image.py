# tests/test_resample_image.py
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from explore_and_process.rasterize_crowns import resample_image  # noqa: E402

CRS = "EPSG:32736"
SIZE = 16
# uint16-range value that normalises to ~0.2 after /65535
VALID_RAW = 13107.0


def _write_source(path, data):
    transform = from_origin(357000.0, 7238000.0, 0.05, 0.05)
    profile = dict(
        driver="GTiff", dtype="float32",
        width=data.shape[2], height=data.shape[1],
        count=data.shape[0], crs=CRS, transform=transform,
        nodata=None,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return transform


def _run(tmp_path, data):
    src_path = tmp_path / "src.tif"
    out_path = tmp_path / "out_ms4.tif"
    transform = _write_source(str(src_path), data)
    resample_image(
        str(src_path), bands=list(range(1, data.shape[0] + 1)),
        h=SIZE, w=SIZE, transform=transform, crs=CRS, out_path=str(out_path),
    )
    with rasterio.open(out_path) as src:
        return src.read()


def test_hot_pixels_clipped_to_valid_range(tmp_path):
    data = np.full((2, SIZE, SIZE), VALID_RAW, dtype=np.float32)
    data[0, 3, 4] = 5.0e23  # corrupt reflectance blow-up
    data[1, 8, 8] = 1.2e9

    out = _run(tmp_path, data)

    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_negative_values_clipped_to_zero(tmp_path):
    data = np.full((1, SIZE, SIZE), VALID_RAW, dtype=np.float32)
    data[0, 2, 2] = -500.0

    out = _run(tmp_path, data)

    assert out.min() >= 0.0


def test_valid_values_unchanged(tmp_path):
    data = np.full((1, SIZE, SIZE), VALID_RAW, dtype=np.float32)
    data[0, 3, 4] = 5.0e23

    out = _run(tmp_path, data)

    # a pixel far away from the hot pixel keeps its normalised value
    assert out[0, 12, 12] == pytest.approx(VALID_RAW / 65535.0, abs=1e-6)


def test_nan_becomes_zero(tmp_path):
    data = np.full((1, SIZE, SIZE), VALID_RAW, dtype=np.float32)
    data[0, 5, 5] = np.nan

    out = _run(tmp_path, data)

    assert np.isfinite(out).all()
