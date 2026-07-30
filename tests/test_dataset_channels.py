# tests/test_dataset_channels.py
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.channels import ChannelSpec  # noqa: E402
from data.dataset import CrownDataset  # noqa: E402

CRS = "EPSG:32636"
TRANSFORM = from_origin(500000, 5400000, 0.05, 0.05)
STACK = ["red", "green", "blue", "rededge", "nir"]


def _write(path, data, nodata=None):
    profile = dict(
        driver="GTiff", dtype="float32", width=data.shape[2],
        height=data.shape[1], count=data.shape[0], crs=CRS,
        transform=TRANSFORM, nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def _make_split(root):
    for sub in ("images", "masks", "dsm"):
        (root / sub).mkdir(parents=True)
    # band b has constant value (b+1)/10 → identifies channels after selection
    img = np.stack([np.full((4, 4), (b + 1) / 10.0) for b in range(5)])
    _write(root / "images" / "0_0.tif", img)
    _write(root / "masks" / "0_0_mask.tif", np.zeros((1, 4, 4)), nodata=255.0)
    _write(root / "dsm" / "0_0_dsm.tif", np.full((1, 4, 4), 0.9))


def test_selected_channels_in_order_with_ndsm(tmp_path):
    _make_split(tmp_path)
    spec = ChannelSpec(STACK, ["nir", "red", "ndsm"])
    ds = CrownDataset(tmp_path, spec)
    img, mask = ds[0]
    assert img.shape == (3, 4, 4)
    assert torch.allclose(img[0], torch.full((4, 4), 0.5))   # nir = band 5
    assert torch.allclose(img[1], torch.full((4, 4), 0.1))   # red = band 1
    assert torch.allclose(img[2], torch.full((4, 4), 0.9))   # ndsm
    assert mask.shape == (1, 4, 4)


def test_without_ndsm_dsm_file_not_needed(tmp_path):
    _make_split(tmp_path)
    (tmp_path / "dsm" / "0_0_dsm.tif").unlink()   # prove it isn't read
    spec = ChannelSpec(STACK, ["green", "blue"])
    img, _ = CrownDataset(tmp_path, spec)[0]
    assert img.shape == (2, 4, 4)
    assert torch.allclose(img[0], torch.full((4, 4), 0.2))


def test_norm_stats_applied(tmp_path):
    _make_split(tmp_path)
    spec = ChannelSpec(STACK, ["red"])
    ds = CrownDataset(tmp_path, spec, norm_stats={"mean": [0.1], "std": [0.2]})
    img, _ = ds[0]
    assert torch.allclose(img[0], torch.zeros(4, 4), atol=1e-6)
