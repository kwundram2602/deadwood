# tests/test_scene_errors.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.channels import ChannelSpec  # noqa: E402
from training.scene_errors import evaluate_scene_errors, scene_of  # noqa: E402

CRS = "EPSG:32636"
GSD = 0.05
N = 8
P = 0.8  # constant model probability
LOGIT = float(np.log(P / (1 - P)))


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), LOGIT)


def _write(path, data, col, row, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", width=N, height=N,
        count=data.shape[0], crs=CRS, nodata=nodata,
        transform=from_origin(500000 + col * N * GSD, 5400000 - row * N * GSD, GSD, GSD),
    ) as dst:
        dst.write(data.astype(np.float32))


def _tile(root, split, stem, col, row, mask):
    for sub in ("images", "masks", "dsm"):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    _write(root / split / "images" / f"{stem}.tif", np.full((2, N, N), 0.5), col, row)
    _write(root / split / "masks" / f"{stem}_mask.tif", mask[np.newaxis], col, row, nodata=-1.0)
    _write(root / split / "dsm" / f"{stem}_dsm.tif", np.zeros((1, N, N)), col, row)


def _masks():
    ones = np.ones((N, N), np.float32)
    zeros = np.zeros((N, N), np.float32)
    half = np.full((N, N), 0.5, np.float32)
    with_sentinels = zeros.copy()
    with_sentinels[:2] = 255.0  # unlabelled
    with_sentinels[2:4] = -1.0  # outside the footprint
    return ones, zeros, half, with_sentinels


@pytest.fixture
def split_root(tmp_path):
    """2 dates × train/val. Train: one all-crown and one all-ground tile."""
    ones, zeros, half, with_sentinels = _masks()
    root = tmp_path / "split"
    _tile(root, "train", "20260226_0_0", 0, 0, ones)
    _tile(root, "train", "20260226_0_1", 1, 0, zeros)
    _tile(root, "val", "20260226_1_0", 0, 1, half)
    _tile(root, "train", "20260313_0_0", 0, 0, ones)
    _tile(root, "val", "20260313_0_1", 1, 0, with_sentinels)
    return root


def _run(split_root, tmp_path):
    spec = ChannelSpec(["a", "b"], ["a", "b"])
    return evaluate_scene_errors(
        ConstantModel(), split_root, spec, None, torch.device("cpu"), tmp_path / "out",
        splits=["train", "val"], threshold=0.5, target_threshold=0.5, batch_size=2,
    )


def test_mse_binary_and_baselines_match_hand_values(split_root, tmp_path):
    res = _run(split_root, tmp_path)
    # train targets: 2 all-ones tiles + 1 all-zeros tile -> c = 2/3, majority = 1
    c = 2.0 / 3.0
    assert res["baseline"]["constant"] == pytest.approx(c)
    assert res["baseline"]["majority_class"] == 1

    a = res["scenes"]["20260226"]["train"]  # one ones tile, one zeros tile
    assert a["n_px"] == 2 * N * N
    assert a["mse"] == pytest.approx(((P - 1) ** 2 + P**2) / 2)
    assert a["mse_binary"] == pytest.approx(0.5)  # predicts crown everywhere
    assert a["mse_baseline"] == pytest.approx(((c - 1) ** 2 + c**2) / 2)
    assert a["mse_binary_baseline"] == pytest.approx(0.5)

    v = res["scenes"]["20260226"]["val"]  # all 0.5 -> binarised as ground (y > 0.5 is false)
    assert v["mse"] == pytest.approx((P - 0.5) ** 2)
    assert v["mse_binary"] == pytest.approx(1.0)


def test_sentinels_are_excluded(split_root, tmp_path):
    res = _run(split_root, tmp_path)
    v = res["scenes"]["20260313"]["val"]
    assert v["n_px"] == (N - 4) * N  # 2 rows unlabelled + 2 rows outside dropped
    assert v["mse"] == pytest.approx(P**2)


def test_error_map_places_tiles_and_leaves_gaps_nan(split_root, tmp_path):
    _run(split_root, tmp_path)
    with rasterio.open(tmp_path / "out" / "error_20260226.tif") as src:
        err = src.read(1)
        assert src.crs.to_string() == CRS
        assert (src.height, src.width) == (2 * N, 2 * N)
    assert np.allclose(err[:N, :N], (P - 1) ** 2)  # train ones tile, top-left
    assert np.allclose(err[:N, N:], P**2)  # train zeros tile, top-right
    assert np.allclose(err[N:, :N], (P - 0.5) ** 2)  # val tile, bottom-left
    assert np.isnan(err[N:, N:]).all()  # no tile there

    with rasterio.open(tmp_path / "out" / "error_20260313.tif") as src:
        err = src.read(1)
    assert np.isnan(err[:4, N:]).all()  # sentinel rows of the val tile


def test_outputs_per_scene_and_json(split_root, tmp_path):
    _run(split_root, tmp_path)
    out = tmp_path / "out"
    for scene in ("20260226", "20260313"):
        assert (out / f"error_{scene}.tif").exists()
        assert (out / f"error_{scene}.png").exists()
    data = json.loads((out / "scene_errors.json").read_text())
    assert set(data["scenes"]) == {"20260226", "20260313"}
    assert set(data["scenes"]["20260313"]) == {"train", "val"}
    assert data["overall"]["train"]["n_px"] == 3 * N * N


def test_scene_of():
    assert scene_of("20260313_04_12") == "20260313"
    assert scene_of("04_12") == "scene"
