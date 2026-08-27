# tests/test_channel_stats.py
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import compute_channel_stats  # noqa: E402

CRS = "EPSG:32636"
TRANSFORM = from_origin(500000, 5400000, 0.05, 0.05)


def _write(path, data):
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        crs=CRS,
        transform=TRANSFORM,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def _make_split(root, n_bands=2):
    (root / "images").mkdir(parents=True)
    (root / "dsm").mkdir()
    img = np.stack([np.full((4, 4), 0.2), np.full((4, 4), 0.6)])[:n_bands]
    _write(root / "images" / "0_0.tif", img)
    _write(root / "dsm" / "0_0_dsm.tif", np.full((1, 4, 4), 0.5))


def test_stats_names_and_values(tmp_path):
    _make_split(tmp_path)
    stats = compute_channel_stats(tmp_path, names=["rededge", "nir"])
    assert stats["names"] == ["rededge", "nir", "ndsm"]
    assert stats["mean"] == pytest.approx([0.2, 0.6, 0.5], abs=1e-6)
    assert len(stats["std"]) == 3


def test_stats_band_count_mismatch_raises(tmp_path):
    _make_split(tmp_path, n_bands=2)
    with pytest.raises(ValueError, match="band"):
        compute_channel_stats(tmp_path, names=["a", "b", "c"])
