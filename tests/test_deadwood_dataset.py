import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.deadwood_dataset import IMAGENET_MEAN, IMAGENET_STD, DeadwoodPatchDataset
from utils.nodata import MASK_OUTSIDE, MASK_UNLABELLED


def _write(path, data, dtype, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[-2],
        width=data.shape[-1],
        count=data.shape[0],
        dtype=dtype,
        crs="EPSG:32736",
        transform=from_origin(0, 1, 0.05, 0.05),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def _split_dir(tmp_path: Path, size=8) -> Path:
    image = np.full((3, size, size), 128, dtype="uint8")
    mask = np.full((1, size, size), MASK_UNLABELLED, dtype="float32")
    mask[0, 2:4, 2:4] = 1.0
    mask[0, 4:6, 4:6] = 0.0
    mask[0, 0, 0] = MASK_OUTSIDE
    _write(tmp_path / "images" / "21.tif", image, "uint8", 0)
    _write(tmp_path / "masks" / "21_mask.tif", mask, "float32", MASK_OUTSIDE)
    return tmp_path


def test_image_is_imagenet_normalised(tmp_path):
    ds = DeadwoodPatchDataset(_split_dir(tmp_path))
    image, _ = ds[0]
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    expected = (128 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    for c in range(3):
        assert np.allclose(image[c].numpy(), expected[c], atol=1e-5)


def test_mask_sentinels_survive_untouched(tmp_path):
    # The loss decides validity from the sentinels; normalising or clipping the
    # mask would silently turn unlabelled pixels into confident background.
    ds = DeadwoodPatchDataset(_split_dir(tmp_path))
    _, mask = ds[0]
    assert mask.shape == (1, 8, 8)
    assert mask[0, 0, 0].item() == MASK_OUTSIDE
    assert mask[0, 1, 1].item() == MASK_UNLABELLED
    assert mask[0, 2, 2].item() == 1.0
    assert mask[0, 4, 4].item() == 0.0


def test_length_matches_the_patch_count(tmp_path):
    assert len(DeadwoodPatchDataset(_split_dir(tmp_path))) == 1


def test_augmentation_preserves_sentinels(tmp_path):
    from data.deadwood_dataset import get_deadwood_train_transform

    ds = DeadwoodPatchDataset(_split_dir(tmp_path), transform=get_deadwood_train_transform())
    for _ in range(10):
        _, mask = ds[0]
        values = set(np.unique(mask.numpy()).tolist())
        assert values <= {MASK_OUTSIDE, MASK_UNLABELLED, 0.0, 1.0}
        assert (mask.numpy() == 1.0).sum() == 4  # D4 moves the crown, never resizes it
