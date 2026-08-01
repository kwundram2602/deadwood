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
ROOT = Path(os.path.join(os.path.dirname(__file__), ".."))
PREDICT_CONFIGS = sorted((ROOT / "configs/predict").glob("*.yaml"))


def _triples(sources):
    return [
        (str(s.path), [int(b) for b in s.bands], [str(n) for n in s.names])
        for s in sources
    ]


def test_predict_configs_exist():
    assert PREDICT_CONFIGS, "no predict configs found under configs/predict/"


@pytest.mark.parametrize("cfg_path", PREDICT_CONFIGS, ids=lambda p: p.name)
def test_predict_config_has_required_keys(cfg_path):
    from omegaconf import OmegaConf

    pcfg = OmegaConf.load(cfg_path)
    for key in (
        "weights", "channels", "dsm", "dtm", "stats", "threshold", "tile_size", "overlap"
    ):
        assert key in pcfg, f"{cfg_path.name} missing key: {key}"
    p = pcfg.preprocess
    assert p.enabled is True
    assert p.target_gsd > 0
    assert len(p.sources) >= 1
    for s in p.sources:
        assert len(list(s.bands)) == len(list(s.names))
    assert 0 <= float(pcfg.overlap) < 1
    assert 0 <= float(pcfg.threshold) <= 1


@pytest.mark.parametrize("cfg_path", PREDICT_CONFIGS, ids=lambda p: p.name)
def test_predict_sources_match_preprocess_sources(cfg_path):
    """Every predict source must be byte-identical to a training source, in the
    same relative order — a predict config may use fewer rasters (RGB only) but
    never different bands, names, or ordering than the data was built with."""
    from omegaconf import OmegaConf

    prep = OmegaConf.load(ROOT / "configs/preprocess/preprocess.yaml")
    train = _triples(prep.rasterize.sources)
    pred = _triples(OmegaConf.load(cfg_path).preprocess.sources)

    unknown = [t for t in pred if t not in train]
    assert not unknown, f"{cfg_path.name}: sources absent from preprocess.yaml: {unknown}"
    assert pred == [t for t in train if t in pred], f"{cfg_path.name}: source order differs"


def test_quicklook_band_indexes_true_rgb():
    from scripts.predict import quicklook_band_indexes

    names = ["green_ms", "red_ms", "red", "green", "blue", "nir"]
    assert quicklook_band_indexes(names) == [3, 4, 5]


def test_quicklook_band_indexes_fallback_first_three():
    from scripts.predict import quicklook_band_indexes

    assert quicklook_band_indexes(["green_ms", "red_ms", "nir"]) == [1, 2, 3]
    assert quicklook_band_indexes(["nir"]) == [1, 1, 1]


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


# ------------------------------------------------------------ main hard errors
def _run_main(cfg_text: str, tmp_path: Path) -> None:
    """Run scripts.predict.main() against an inline config rooted at tmp_path."""
    from scripts.predict import main

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(cfg_text)
    old_argv = sys.argv
    sys.argv = ["predict.py", "--config", str(cfg), "--working_dir", str(tmp_path)]
    try:
        main()
    finally:
        sys.argv = old_argv


def _write_stack(path: Path, names: list[str], h: int = 8, w: int = 8) -> None:
    """Minimal [0,1] scene stack with band descriptions, as rasterize writes it."""
    import rasterio
    from rasterio.transform import from_origin

    profile = dict(
        driver="GTiff", dtype="float32", width=w, height=h, count=len(names),
        crs="EPSG:32736", transform=from_origin(357000.0, 7238000.0, 0.05, 0.05),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((len(names), h, w), dtype=np.float32))
        for i, name in enumerate(names, 1):
            dst.set_band_description(i, name)


def test_main_rejects_missing_channel_manifest(tmp_path):
    (tmp_path / "exp").mkdir()
    with pytest.raises(ValueError, match="Channel manifest not found"):
        _run_main("weights: exp/best.pt\nchannels: null\n", tmp_path)


def test_main_rejects_stack_without_band_descriptions(tmp_path):
    import json

    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "channels.json").write_text(json.dumps({"names": ["red"]}))
    _open_raster(tmp_path, "scene.tif").close()  # 1 band, no descriptions

    cfg = (
        "weights: exp/best.pt\nchannels: null\nimage: scene.tif\n"
        "preprocess:\n  enabled: false\n"
    )
    with pytest.raises(ValueError, match="no band descriptions"):
        _run_main(cfg, tmp_path)


def test_main_rejects_checkpoint_channel_width_mismatch(tmp_path):
    import json

    import segmentation_models_pytorch as smp
    import torch

    exp = tmp_path / "exp"
    exp.mkdir()
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1
    )
    torch.save(model.state_dict(), exp / "best.pt")
    # manifest lists 4 channels but the checkpoint's first conv takes 3
    names = ["red", "green", "blue", "nir"]
    (exp / "channels.json").write_text(json.dumps({"names": names}))
    _write_stack(tmp_path / "scene.tif", names)

    cfg = (
        "weights: exp/best.pt\nchannels: null\nstats: null\nimage: scene.tif\n"
        "preprocess:\n  enabled: false\n"
    )
    with pytest.raises(ValueError, match="Checkpoint expects 3 input channels"):
        _run_main(cfg, tmp_path)


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


def test_normalize_ndsm_zeroes_dtm_nodata():
    from explore_and_process.apply_dsm_mask import normalize_ndsm

    # DSM is valid but the external DTM has a gap -> nDSM NaN; the channel must
    # never carry NaN into the tiles.
    ndsm = np.full((4, 4), 8.0, dtype=np.float32)
    ndsm[0, 0] = np.nan
    dsm = np.ones_like(ndsm)
    out = normalize_ndsm(ndsm, dsm, max_ndsm_height=16.0)
    assert out[0, 0] == 0.0
    assert np.isfinite(out).all()


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
