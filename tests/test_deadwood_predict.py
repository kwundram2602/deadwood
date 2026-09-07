import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.raw_predict_deadwood import predict_mask, predict_probs


class ConstantLogit(torch.nn.Module):
    """Stand-in for the real model: emits one fixed logit everywhere."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.logit)


def _scene(height=200, width=200, nodata_rows=0):
    # dataset_mask ORs the per-band masks, so a pixel only reads as noData when
    # every band is 0. Blanking one band would leave it valid.
    data = np.full((3, height, width), 128, dtype="uint8")
    data[:, :nodata_rows, :] = 0
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32736",
        transform=from_origin(0, height * 0.05, 0.05, 0.05),
        nodata=0,
    )
    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(data)
    return memfile


def _cfg():
    return OmegaConf.create({"tile_size": 128, "padding": 32, "batch_size": 2, "threshold": 0.5})


def test_predict_probs_returns_sigmoid_of_logits():
    with _scene().open() as src:
        probs, valid = predict_probs(src, ConstantLogit(0.0), torch.device("cpu"), _cfg())
    assert probs.shape == (200, 200)
    assert probs.dtype == np.float32
    assert np.allclose(probs, 0.5, atol=1e-5)
    assert valid.all()


def test_predict_mask_thresholds_the_probabilities():
    # logit 2.0 -> sigmoid 0.88, above the 0.5 threshold everywhere.
    with _scene().open() as src:
        mask = predict_mask(src, ConstantLogit(2.0), torch.device("cpu"), _cfg())
    assert mask.dtype == np.uint8
    assert mask.min() == 1 and mask.max() == 1


def test_predict_mask_suppresses_nodata_pixels():
    with _scene(nodata_rows=50).open() as src:
        mask = predict_mask(src, ConstantLogit(2.0), torch.device("cpu"), _cfg())
    assert mask[:50, :].max() == 0
    assert mask[60:, :].min() == 1
