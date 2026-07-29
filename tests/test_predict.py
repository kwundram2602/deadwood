# tests/test_predict.py
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.predict import (  # noqa: E402
    TileMerger,
    binarize,
    blend_weights,
    make_windows,
)


# ---------------------------------------------------------------- make_windows
def test_windows_cover_scene_exactly_divisible():
    offs = make_windows(height=1024, width=1536, tile_size=512, stride=512)
    assert (0, 0) in offs and (512, 1024) in offs
    covered = np.zeros((1024, 1536), dtype=bool)
    for r, c in offs:
        covered[r : r + 512, c : c + 512] = True
    assert covered.all()


def test_windows_last_offset_clamped_inside():
    offs = make_windows(height=700, width=700, tile_size=512, stride=256)
    rows = sorted({r for r, _ in offs})
    # last window must end exactly at the scene edge, not beyond
    assert rows[-1] == 700 - 512
    covered = np.zeros((700, 700), dtype=bool)
    for r, c in offs:
        covered[r : r + 512, c : c + 512] = True
    assert covered.all()


def test_windows_scene_smaller_than_tile():
    offs = make_windows(height=300, width=200, tile_size=512, stride=256)
    assert offs == [(0, 0)]


def test_windows_no_duplicates():
    offs = make_windows(height=1000, width=1000, tile_size=512, stride=256)
    assert len(offs) == len(set(offs))


# --------------------------------------------------------------- blend_weights
def test_blend_weights_shape_and_positive():
    w = blend_weights(512)
    assert w.shape == (512, 512)
    assert (w > 0).all()  # eps keeps borders non-zero → no div-by-zero
    assert w.max() <= 1.0 + 1e-3


def test_blend_weights_center_heavier_than_edge():
    w = blend_weights(64)
    assert w[32, 32] > w[0, 0] * 10


# ------------------------------------------------------------------ TileMerger
def test_merger_constant_tiles_give_constant_scene():
    merger = TileMerger(height=700, width=700, tile_size=512)
    for r, c in make_windows(700, 700, 512, 256):
        merger.add(np.full((512, 512), 0.7, dtype=np.float32), r, c)
    prob = merger.merge()
    assert prob.shape == (700, 700)
    np.testing.assert_allclose(prob, 0.7, atol=1e-5)


def test_merger_clips_padded_tile_at_scene_edge():
    # scene smaller than tile: padded area of the tile must be discarded
    merger = TileMerger(height=100, width=80, tile_size=512)
    tile = np.zeros((512, 512), dtype=np.float32)
    tile[:100, :80] = 0.3
    tile[100:, 80:] = 99.0  # garbage in the padded region
    merger.add(tile, 0, 0)
    prob = merger.merge()
    assert prob.shape == (100, 80)
    np.testing.assert_allclose(prob, 0.3, atol=1e-5)


def test_merger_overlap_averages_between_values():
    merger = TileMerger(height=64, width=96, tile_size=64)
    merger.add(np.full((64, 64), 0.2, dtype=np.float32), 0, 0)
    merger.add(np.full((64, 64), 0.8, dtype=np.float32), 0, 32)
    prob = merger.merge()
    overlap = prob[:, 32:64]
    assert (overlap >= 0.2 - 1e-6).all() and (overlap <= 0.8 + 1e-6).all()
    # non-overlapping parts keep their tile's value
    np.testing.assert_allclose(prob[:, :32], 0.2, atol=1e-5)
    np.testing.assert_allclose(prob[:, 64:], 0.8, atol=1e-5)


# -------------------------------------------------------------------- binarize
def test_binarize_threshold_and_nodata():
    prob = np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])
    out = binarize(prob, threshold=0.5, valid_mask=valid)
    assert out.dtype == np.uint8
    expected = np.array([[0, 1], [1, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("t,expected_ones", [(0.1, 3), (0.9, 0)])
def test_binarize_threshold_sweeps(t, expected_ones):
    prob = np.array([[0.2, 0.5], [0.8, 0.05]], dtype=np.float32)
    valid = np.ones_like(prob, dtype=bool)
    out = binarize(prob, threshold=t, valid_mask=valid)
    assert int((out == 1).sum()) == expected_ones


# ------------------------------------------------------------- predict config
def test_predict_config_has_required_keys():
    from omegaconf import OmegaConf

    root = Path(os.path.join(os.path.dirname(__file__), ".."))
    pcfg = OmegaConf.load(root / "configs/predict/predict.yaml")
    for key in ("image", "dsm", "weights", "stats", "threshold", "tile_size", "overlap"):
        assert key in pcfg, f"predict.yaml missing key: {key}"
    assert 0 <= float(pcfg.overlap) < 1
    assert 0 <= float(pcfg.threshold) <= 1


# -------------------------------------------------------------- grid validation
def _open_raster(tmp_path, name, h=8, w=8, origin=(357000.0, 7238000.0), res=0.05, crs="EPSG:32736"):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / name
    profile = dict(
        driver="GTiff", dtype="float32", width=w, height=h, count=1,
        crs=crs, transform=from_origin(*origin, res, res),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((1, h, w), dtype=np.float32))
    return rasterio.open(path)


def test_validate_grid_accepts_matching(tmp_path):
    from scripts.predict import validate_grid

    with _open_raster(tmp_path, "a.tif") as a, _open_raster(tmp_path, "b.tif") as b:
        validate_grid(a, b)  # no raise


def test_validate_grid_rejects_shifted_transform(tmp_path):
    from scripts.predict import validate_grid

    with _open_raster(tmp_path, "a.tif") as a, \
         _open_raster(tmp_path, "b.tif", origin=(357005.0, 7238000.0)) as b:
        with pytest.raises(ValueError, match="Geotransform mismatch"):
            validate_grid(a, b)


def test_validate_grid_rejects_shape_mismatch(tmp_path):
    from scripts.predict import validate_grid

    with _open_raster(tmp_path, "a.tif") as a, _open_raster(tmp_path, "b.tif", h=9) as b:
        with pytest.raises(ValueError, match="Grid mismatch"):
            validate_grid(a, b)


# ---------------------------------------------------------- nDSM normalisation
def test_normalize_ndsm_matches_training_procedure():
    from explore_and_process.apply_dsm_mask import normalize_ndsm

    ndsm = np.linspace(0.0, 30.0, 1000, dtype=np.float32).reshape(20, 50)
    dsm = np.ones_like(ndsm)
    out = normalize_ndsm(ndsm, dsm, max_ndsm_height=16.0)
    # ceiling = min(p95=28.5, 16.0) = 16.0
    assert out.min() == 0.0 and out.max() == 1.0
    np.testing.assert_allclose(out[0, 10], ndsm[0, 10] / 16.0, rtol=1e-5)
    assert (out[ndsm >= 16.0] == 1.0).all()


def test_normalize_ndsm_zeroes_dsm_nodata():
    from explore_and_process.apply_dsm_mask import normalize_ndsm

    ndsm = np.full((4, 4), 8.0, dtype=np.float32)
    dsm = np.ones_like(ndsm)
    dsm[0, 0] = np.nan
    out = normalize_ndsm(ndsm, dsm, max_ndsm_height=16.0)
    assert out[0, 0] == 0.0


# ------------------------------------------------------- architecture inference
def _fake_state(in_ch=5, classes=1, bottleneck=True):
    import torch

    state = {
        "encoder.conv1.weight": torch.zeros(64, in_ch, 7, 7),
        "segmentation_head.0.weight": torch.zeros(classes, 16, 3, 3),
    }
    if bottleneck:
        state["encoder.layer1.0.conv3.weight"] = torch.zeros(256, 64, 1, 1)
    else:
        state["encoder.layer3.2.conv1.weight"] = torch.zeros(256, 256, 3, 3)
    return state


def test_infer_architecture_resnet50():
    from scripts.predict import infer_architecture

    assert infer_architecture(_fake_state()) == ("resnet50", 5, 1)


def test_infer_architecture_resnet34():
    from scripts.predict import infer_architecture

    assert infer_architecture(_fake_state(in_ch=4, bottleneck=False)) == ("resnet34", 4, 1)


def test_infer_architecture_rejects_foreign_state():
    from scripts.predict import infer_architecture

    with pytest.raises(ValueError, match="missing expected UNet key"):
        infer_architecture({"some.other.model": None})


_REAL_CKPT = Path(
    "/home/kjell/projects/py_projects/InnoLabDL/deadwood/experiments/"
    "crown_ms_mask_and_hr0.9__OAM_RGB_RESNET50_TCD__bce0.3_dice0.6/ft_best.pt"
)


@pytest.mark.skipif(not _REAL_CKPT.exists(), reason="local checkpoint not present")
def test_build_model_from_real_checkpoint():
    import torch

    from scripts.predict import build_model_from_checkpoint

    model = build_model_from_checkpoint(_REAL_CKPT, torch.device("cpu"))
    with torch.no_grad():
        out = model(torch.zeros(1, 5, 64, 64))
    assert out.shape == (1, 1, 64, 64)
