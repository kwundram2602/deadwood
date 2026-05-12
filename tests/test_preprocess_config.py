# tests/test_preprocess_config.py
import os
import sys
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "preprocess.yaml")


def test_config_rasterize_section():
    cfg = OmegaConf.load(CONFIG_PATH)
    r = cfg.rasterize
    assert len(list(r.crowns)) > 0
    assert all(isinstance(p, str) for p in r.crowns)
    assert r.target_gsd == pytest.approx(0.05)
    assert r.sigma == pytest.approx(10.0)
    assert r.nodata_threshold == pytest.approx(0.05)
    assert list(r.bands) == [5, 4, 6, 7]
    assert r.raster_dir is None


def test_config_dsm_mask_section():
    cfg = OmegaConf.load(CONFIG_PATH)
    d = cfg.dsm_mask
    assert len(list(d.windows)) > 0
    assert d.gradient_threshold is None
    assert d.height_threshold > 0
    assert 0.0 < d.nodata_resolve_threshold < 1.0
    assert d.method in ("local_min", "gradient", "both")


def test_config_tiling_section():
    cfg = OmegaConf.load(CONFIG_PATH)
    t = cfg.tiling
    assert t.size == 512
    assert t.nodata_thresh == pytest.approx(0.9)
    assert isinstance(t.out, str)


def test_omegaconf_null_becomes_none():
    """null in the real config must deserialize to None — critical for gradient_threshold / raster_dir."""
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg.dsm_mask.gradient_threshold is None
    assert cfg.rasterize.raster_dir is None


def test_omegaconf_list_is_iterable():
    """ListConfig must support list() conversion — used by rasterize (crowns) and dsm_mask (windows)."""
    cfg = OmegaConf.create({"crowns": ["a.gpkg", "b.gpkg"], "windows": [150, 350, 700]})
    assert list(cfg.crowns) == ["a.gpkg", "b.gpkg"]
    assert list(cfg.windows) == [150, 350, 700]


def test_run_propagates_dsm_paths_to_tiling(tmp_path):
    """After dsm_main mutates cfg.dsm_mask.out/_dsm via _embed_params, run() copies to cfg.tiling."""
    config_file = tmp_path / "preprocess.yaml"
    config_file.write_text(
        "rasterize:\n"
        "  crowns: []\n"
        "  reference: dummy.tif\n"
        "  out_mask: masks/crown_mask.tif\n"
        "  out_image_dir: images/\n"
        "  raster_dir: null\n"
        "  target_gsd: 0.05\n"
        "  sigma: 10.0\n"
        "  nodata_threshold: 0.05\n"
        "  bands: [5, 4, 6, 7]\n"
        "dsm_mask:\n"
        "  mask: masks/crown_mask.tif\n"
        "  dsm: dummy_dsm.tif\n"
        "  out: masks/crown_mask_final.tif\n"
        "  out_dsm: ndsm/dsm_ndsm.tif\n"
        "  method: local_min\n"
        "  combine: or\n"
        "  windows: [150, 350, 700]\n"
        "  height_threshold: 2.0\n"
        "  gradient_sigma: 3.0\n"
        "  gradient_threshold: null\n"
        "  max_ndsm_height: 50.0\n"
        "  nodata_resolve_threshold: 0.7\n"
        "tiling:\n"
        "  image: images/test_ms4.tif\n"
        "  mask: masks/old_mask.tif\n"
        "  dsm: ndsm/old_dsm.tif\n"
        "  out: out/patches\n"
        "  size: 512\n"
        "  nodata_thresh: 0.9\n"
    )

    def fake_dsm_main(args):
        args.out = "masks/crown_mask_final_lm_w150-350-700_ht2.0.tif"
        args.out_dsm = "ndsm/dsm_ndsm_w150-350-700.tif"

    with patch("scripts.preprocess.rasterize_main"), \
         patch("scripts.preprocess.dsm_main", side_effect=fake_dsm_main), \
         patch("scripts.preprocess.tile_main") as mock_tile:
        from scripts.preprocess import run
        run(str(config_file))
        tiling_args = mock_tile.call_args[0][0]
        assert tiling_args.mask == "masks/crown_mask_final_lm_w150-350-700_ht2.0.tif"
        assert tiling_args.dsm == "ndsm/dsm_ndsm_w150-350-700.tif"


def test_run_does_not_propagate_null_out_dsm(tmp_path):
    """If dsm_mask.out_dsm is null (or not mutated), cfg.tiling.dsm must keep its original value."""
    config_file = tmp_path / "preprocess.yaml"
    config_file.write_text(
        "rasterize:\n"
        "  crowns: []\n"
        "  reference: dummy.tif\n"
        "  out_mask: masks/crown_mask.tif\n"
        "  out_image_dir: images/\n"
        "  raster_dir: null\n"
        "  target_gsd: 0.05\n"
        "  sigma: 10.0\n"
        "  nodata_threshold: 0.05\n"
        "  bands: [5, 4, 6, 7]\n"
        "dsm_mask:\n"
        "  mask: masks/crown_mask.tif\n"
        "  dsm: dummy_dsm.tif\n"
        "  out: masks/crown_mask_final.tif\n"
        "  out_dsm: null\n"
        "  method: gradient\n"
        "  combine: or\n"
        "  windows: [150, 350, 700]\n"
        "  height_threshold: 2.0\n"
        "  gradient_sigma: 3.0\n"
        "  gradient_threshold: null\n"
        "  max_ndsm_height: 50.0\n"
        "  nodata_resolve_threshold: 0.7\n"
        "tiling:\n"
        "  image: images/test_ms4.tif\n"
        "  mask: masks/old_mask.tif\n"
        "  dsm: ndsm/explicit_dsm_path.tif\n"
        "  out: out/patches\n"
        "  size: 512\n"
        "  nodata_thresh: 0.9\n"
    )

    def fake_dsm_main(args):
        args.out = "masks/crown_mask_final_gr_s3.0.tif"
        # out_dsm stays None — gradient method does not produce nDSM

    with patch("scripts.preprocess.rasterize_main"), \
         patch("scripts.preprocess.dsm_main", side_effect=fake_dsm_main), \
         patch("scripts.preprocess.tile_main") as mock_tile:
        from scripts.preprocess import run
        run(str(config_file))
        tiling_args = mock_tile.call_args[0][0]
        assert tiling_args.mask == "masks/crown_mask_final_gr_s3.0.tif"
        assert tiling_args.dsm == "ndsm/explicit_dsm_path.tif"  # unchanged from config
