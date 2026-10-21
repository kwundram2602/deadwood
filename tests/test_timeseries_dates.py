# tests/test_timeseries_dates.py
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_and_process.timeseries import (  # noqa: E402
    date_config,
    select_dates,
    validate_label_cfg,
    validate_timeseries_cfg,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "preprocess"
    / "use_timeseries"
    / "all_dsm_dates_shuffle.yaml"
)


def _load_cfg():
    """Shipped time-series config with the split fields these tests rely on pinned."""
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.timeseries.dates = [20260226, 20260313]
    cfg.split.strategy = "by_date"
    cfg.split.dates = {"train": [20260226], "val": [20260313], "test": []}
    return cfg


def _files(root, om_dates, dsm_dates):
    (root / "om").mkdir()
    (root / "dsm").mkdir()
    for d in om_dates:
        (root / "om" / f"{d}_Airport_OM_coreg.tif").touch()
    for d in dsm_dates:
        (root / "dsm" / f"{d}_Airport_DSM_coreg.tif").touch()


def _ts(root, dates=None, max_dates=None):
    return OmegaConf.create(
        {
            "om_dir": str(root / "om"),
            "om_pattern": "*_OM_coreg.tif",
            "dsm_dir": str(root / "dsm"),
            "dsm_pattern": "*_DSM_coreg.tif",
            "dates": dates,
            "max_dates": max_dates,
            "work_dir": str(root / "work"),
        }
    )


def test_select_pairs_by_date_and_skips_unpaired(tmp_path):
    _files(tmp_path, ["20260101", "20260201", "20260301"], ["20260301", "20260101"])
    scenes = select_dates(_ts(tmp_path))
    assert [s.date for s in scenes] == ["20260101", "20260301"]
    assert scenes[0].om.name.startswith("20260101_")
    assert scenes[0].dsm.name.startswith("20260101_")


def test_max_dates_keeps_first_n(tmp_path):
    _files(tmp_path, ["20260101", "20260201", "20260301"], ["20260101", "20260201", "20260301"])
    scenes = select_dates(_ts(tmp_path, max_dates=2))
    assert [s.date for s in scenes] == ["20260101", "20260201"]


def test_explicit_dates_keep_order_and_accept_ints(tmp_path):
    _files(tmp_path, ["20260101", "20260201"], ["20260101", "20260201"])
    scenes = select_dates(_ts(tmp_path, dates=[20260201, 20260101]))
    assert [s.date for s in scenes] == ["20260201", "20260101"]


def test_explicit_date_without_dsm_raises(tmp_path):
    _files(tmp_path, ["20260101", "20260201"], ["20260101"])
    with pytest.raises(ValueError, match="20260201"):
        select_dates(_ts(tmp_path, dates=["20260101", "20260201"]))


def test_no_paired_date_raises(tmp_path):
    _files(tmp_path, ["20260101"], ["20260201"])
    with pytest.raises(ValueError, match="no date"):
        select_dates(_ts(tmp_path))


def test_date_config_sets_per_date_paths(tmp_path):
    _files(tmp_path, ["20260101"], ["20260101"])
    cfg = _load_cfg()
    cfg.timeseries = _ts(tmp_path)
    scene = select_dates(cfg.timeseries)[0]
    d = date_config(cfg, scene)

    work = Path(cfg.timeseries.work_dir) / "20260101"
    assert d.rasterize.reference == str(scene.om)
    assert d.rasterize.sources[0].path == str(scene.om)
    assert Path(d.rasterize.out_mask).parent == work / "masks"
    assert d.dsm_mask.mask == d.rasterize.out_mask
    assert d.dsm_mask.dsm == str(scene.dsm)
    assert d.dsm_mask.out is None
    assert Path(d.tiling.image) == work / "images" / "20260101_Airport_OM_coreg_stack.tif"
    assert Path(d.tiling.out) == Path(cfg.tiling.out) / "20260101"
    assert d.tiling.prefix == "20260101"
    # the base config stays untouched
    assert cfg.rasterize.reference is None
    assert cfg.tiling.get("prefix", None) is None


def test_shipped_config_validates():
    cfg = _load_cfg()
    validate_timeseries_cfg(cfg, [str(d) for d in cfg.timeseries.dates])


def test_without_polygons_crown_resolve_threshold_is_required():
    cfg = _load_cfg()
    cfg.dsm_mask.crown_resolve_threshold = None
    with pytest.raises(ValueError, match="crown_resolve_threshold"):
        validate_label_cfg(cfg)
    cfg.rasterize.use_polygons = True
    validate_label_cfg(cfg)


def test_split_input_must_be_tiles_root():
    cfg = _load_cfg()
    cfg.split.input = "out/elsewhere"
    with pytest.raises(ValueError, match="split.input"):
        validate_timeseries_cfg(cfg, [str(d) for d in cfg.timeseries.dates])


def test_explicit_split_dates_must_match_selection():
    cfg = _load_cfg()
    with pytest.raises(ValueError, match="missing"):
        validate_timeseries_cfg(cfg, ["20260226", "20260313", "20260126"])


def test_multiple_sources_rejected():
    cfg = _load_cfg()
    cfg.rasterize.sources.append({"path": None, "bands": [1], "names": ["x"]})
    with pytest.raises(ValueError, match="exactly one"):
        validate_timeseries_cfg(cfg, [str(d) for d in cfg.timeseries.dates])
