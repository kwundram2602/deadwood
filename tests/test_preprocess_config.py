# tests/test_preprocess_config.py
import json
import os
import sys
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "preprocess", "preprocess.yaml"
)
TIMESERIES_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "configs",
    "preprocess",
    "use_timeseries",
    "all_dsm_dates_shuffle.yaml",
)


def test_config_rasterize_section():
    cfg = OmegaConf.load(CONFIG_PATH)
    r = cfg.rasterize
    assert r.reference
    assert len(r.sources) >= 1
    for s in r.sources:
        assert len(list(s.bands)) == len(list(s.names))
    assert "ndsm" not in [n for s in r.sources for n in s.names]
    assert r.target_gsd > 0


def test_config_dsm_mask_section():
    cfg = OmegaConf.load(CONFIG_PATH)
    d = cfg.dsm_mask
    assert d.method == "dtm" or d.dtm_approx is not None
    assert d.height_threshold > 0
    assert 0.0 < d.nodata_resolve_threshold < 1.0
    assert d.method in ("dtm_approx", "dtm")
    if d.method == "dtm":
        assert d.dtm_real.path, "method: dtm requires dtm_real.path"


def test_config_tiling_section():
    cfg = OmegaConf.load(CONFIG_PATH)
    t = cfg.tiling
    assert t.size == 512
    assert t.nodata_thresh == pytest.approx(0.8)
    assert isinstance(t.out, str)


def test_omegaconf_null_becomes_none():
    """null in the real config must deserialize to None — critical for raster_dir."""
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg.rasterize.raster_dir is None


def test_omegaconf_list_is_iterable():
    """ListConfig must support list() conversion — used by rasterize (crowns)."""
    cfg = OmegaConf.create({"crowns": ["a.gpkg", "b.gpkg"]})
    assert list(cfg.crowns) == ["a.gpkg", "b.gpkg"]


def test_run_propagates_dsm_paths_to_tiling(tmp_path, monkeypatch):
    """After dsm_main mutates cfg.dsm_mask.out/_dsm via _embed_params, run() copies to cfg.tiling."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "split").mkdir(parents=True, exist_ok=True)
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
        "  method: dtm_approx\n"
        "  dtm_approx: {cell: 0.5, max_window: 30.0}\n"
        "  height_threshold: 2.0\n"
        "  max_ndsm_height: 50.0\n"
        "  nodata_resolve_threshold: 0.7\n"
        "tiling:\n"
        "  image: images/test_ms4.tif\n"
        "  mask: masks/old_mask.tif\n"
        "  dsm: ndsm/old_dsm.tif\n"
        "  out: out/patches\n"
        "  size: 512\n"
        "  nodata_thresh: 0.9\n"
        "split:\n"
        "  input: out/patches\n"
        "  output: out/split\n"
        "  train_ratio: 0.7\n"
        "  val_ratio: 0.2\n"
        "  seed: 42\n"
        "  mode: copy\n"
    )

    def fake_dsm_main(args):
        args.out = "masks/crown_mask_final_lm_w150-350-700_ht2.0.tif"
        args.out_dsm = "ndsm/dsm_ndsm_w150-350-700.tif"

    with (
        patch("scripts.preprocess.rasterize_main"),
        patch("scripts.preprocess.dsm_main", side_effect=fake_dsm_main),
        patch("scripts.preprocess.tile_main") as mock_tile,
        patch("scripts.preprocess.split_patches"),
        patch("scripts.preprocess.load_manifest", return_value=["red", "green"]),
        patch(
            "scripts.preprocess.compute_channel_stats",
            return_value={
                "names": ["red", "green", "ndsm"],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        ),
    ):
        from scripts.preprocess import run

        run(str(config_file))
        tiling_args = mock_tile.call_args[0][0]
        assert tiling_args.mask == "masks/crown_mask_final_lm_w150-350-700_ht2.0.tif"
        assert tiling_args.dsm == "ndsm/dsm_ndsm_w150-350-700.tif"


def test_run_does_not_propagate_null_out_dsm(tmp_path, monkeypatch):
    """If dsm_mask.out_dsm is null (or not mutated), cfg.tiling.dsm must keep its original value."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "split").mkdir(parents=True, exist_ok=True)
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
        "  method: dtm_approx\n"
        "  dtm_approx: {cell: 0.5, max_window: 30.0}\n"
        "  height_threshold: 2.0\n"
        "  max_ndsm_height: 50.0\n"
        "  nodata_resolve_threshold: 0.7\n"
        "tiling:\n"
        "  image: images/test_ms4.tif\n"
        "  mask: masks/old_mask.tif\n"
        "  dsm: ndsm/explicit_dsm_path.tif\n"
        "  out: out/patches\n"
        "  size: 512\n"
        "  nodata_thresh: 0.9\n"
        "split:\n"
        "  input: out/patches\n"
        "  output: out/split\n"
        "  train_ratio: 0.7\n"
        "  val_ratio: 0.2\n"
        "  seed: 42\n"
        "  mode: copy\n"
    )

    def fake_dsm_main(args):
        args.out = "masks/crown_mask_final_gr_s3.0.tif"
        # out_dsm stays None — gradient method does not produce nDSM

    with (
        patch("scripts.preprocess.rasterize_main"),
        patch("scripts.preprocess.dsm_main", side_effect=fake_dsm_main),
        patch("scripts.preprocess.tile_main") as mock_tile,
        patch("scripts.preprocess.split_patches"),
        patch("scripts.preprocess.load_manifest", return_value=["red", "green"]),
        patch(
            "scripts.preprocess.compute_channel_stats",
            return_value={
                "names": ["red", "green", "ndsm"],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        ),
    ):
        from scripts.preprocess import run

        run(str(config_file))
        tiling_args = mock_tile.call_args[0][0]
        assert tiling_args.mask == "masks/crown_mask_final_gr_s3.0.tif"
        assert tiling_args.dsm == "ndsm/explicit_dsm_path.tif"  # unchanged from config


def test_run_timeseries_runs_each_date_and_splits_by_date(tmp_path, monkeypatch):
    """Every selected date runs stages 1a–2 with its own paths; the split gets date -> split."""
    monkeypatch.chdir(tmp_path)
    for d in ("20260226", "20260313", "20260126"):
        (tmp_path / "om").mkdir(exist_ok=True)
        (tmp_path / "dsm").mkdir(exist_ok=True)
        (tmp_path / "om" / f"{d}_X_OM_coreg.tif").touch()
        (tmp_path / "dsm" / f"{d}_X_DSM_coreg.tif").touch()

    cfg = OmegaConf.load(TIMESERIES_CONFIG_PATH)
    cfg.timeseries.om_dir = "om"
    cfg.timeseries.dsm_dir = "dsm"
    cfg.timeseries.work_dir = "work"
    cfg.timeseries.dates = [20260226, 20260313]
    cfg.timeseries.max_dates = 2
    cfg.tiling.out = "out/crown_patches_ts"
    cfg.split.input = "out/crown_patches_ts"
    cfg.split.output = "out/crown_patches_ts_split"
    cfg.split.strategy = "by_date"
    cfg.split.dates = {"train": [20260226], "val": [20260313], "test": []}
    config_file = tmp_path / "timeseries.yaml"
    OmegaConf.save(cfg, config_file)

    seen = []

    def fake_dsm_main(args):
        args.out = args.mask.replace("intermediate", "final")
        return {"mse": 0.01, "rmse": 0.1, "bias": -0.02, "rmse_m": 0.3, "mae_m": 0.2, "bias_m": -0.1,
                "p95_abs_m": 0.6, "n_valid": 4}

    def fake_tile_main(args):
        seen.append(dict(prefix=args.prefix, image=args.image, mask=args.mask, out=args.out))
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "channels.json"), "w") as f:
            f.write('{"names": ["red", "green"]}')
        return 5

    with (
        patch("scripts.preprocess.rasterize_main"),
        patch("scripts.preprocess.dsm_main", side_effect=fake_dsm_main),
        patch("scripts.preprocess.tile_main", side_effect=fake_tile_main),
        patch("scripts.preprocess.split_patches") as mock_split,
        patch("scripts.preprocess.load_manifest", return_value=["red", "green"]),
        patch(
            "scripts.preprocess.compute_channel_stats",
            return_value={"names": ["red", "green", "ndsm"], "mean": [0.0] * 3, "std": [1.0] * 3},
        ),
    ):
        from scripts.preprocess import run

        (tmp_path / cfg.split.output).mkdir(parents=True)
        run(str(config_file))

    # dates: [20260226, 20260313] with max_dates 2 — 20260126 is not selected
    assert [s["prefix"] for s in seen] == ["20260226", "20260313"]
    assert seen[0]["mask"] == "work/20260226/masks/crown_mask_final.tif"
    assert seen[1]["out"] == "out/crown_patches_ts/20260313"
    kwargs = mock_split.call_args.kwargs
    assert kwargs["strategy"] == "by_date"
    assert kwargs["date_splits"] == {"20260226": "train", "20260313": "val"}
    records = json.loads((tmp_path / "out" / "crown_patches_ts" / "dates.json").read_text())
    assert [(r["date"], r["tiles"]) for r in records] == [("20260226", 5), ("20260313", 5)]
    assert (tmp_path / "out" / "crown_patches_ts" / "channels.json").exists()
    assert records[0]["ndsm_compare"]["mse"] == 0.01
    rows = (tmp_path / "out" / "crown_patches_ts" / "ndsm_compare.csv").read_text().splitlines()
    assert rows[0] == "date,mse,rmse,bias,rmse_m,mae_m,bias_m,p95_abs_m,n_valid"
    assert rows[1:] == ["20260226,0.01,0.1,-0.02,0.3,0.2,-0.1,0.6,4", "20260313,0.01,0.1,-0.02,0.3,0.2,-0.1,0.6,4"]


def test_ndsm_compare_only_rebuilds_comparison_and_runs_nothing_else(tmp_path, monkeypatch):
    """--ndsm_compare_only re-reads each date's raw nDSMs and refreshes the CSV and dates.json."""
    monkeypatch.chdir(tmp_path)
    for d in ("20260226", "20260313"):
        (tmp_path / "om").mkdir(exist_ok=True)
        (tmp_path / "dsm").mkdir(exist_ok=True)
        (tmp_path / "om" / f"{d}_X_OM_coreg.tif").touch()
        (tmp_path / "dsm" / f"{d}_X_DSM_coreg.tif").touch()
    cfg = OmegaConf.load(TIMESERIES_CONFIG_PATH)
    cfg.timeseries.om_dir, cfg.timeseries.dsm_dir, cfg.timeseries.work_dir = "om", "dsm", "work"
    cfg.timeseries.dates = None
    cfg.tiling.out = "out/ts"
    config_file = tmp_path / "timeseries.yaml"
    OmegaConf.save(cfg, config_file)
    (tmp_path / "out" / "ts").mkdir(parents=True)
    (tmp_path / "out" / "ts" / "dates.json").write_text(
        json.dumps([{"date": "20260226", "tiles": 5, "ndsm_compare": {"mse": 9.0}}])
    )

    def fake_recompare(process_out, max_h):
        return {"rmse": 0.1, "bias": 0.0, "rmse_m": 0.3, "mae_m": 0.2, "bias_m": -0.1, "p95_abs_m": 0.6,
                "dir": f"{process_out}/ndsm_compare/r"}

    with (
        patch("scripts.preprocess.recompare_ndsm", side_effect=fake_recompare) as mock_recompare,
        patch("scripts.preprocess.rasterize_main") as mock_rasterize,
        patch("scripts.preprocess.dsm_main") as mock_dsm,
        patch("scripts.preprocess.split_patches") as mock_split,
    ):
        from scripts.preprocess import run

        run(str(config_file), ndsm_compare_only=True)

    assert [c.args[0] for c in mock_recompare.call_args_list] == [
        os.path.join("work", "20260226"),
        os.path.join("work", "20260313"),
    ]
    assert not (mock_rasterize.called or mock_dsm.called or mock_split.called)
    rows = (tmp_path / "out" / "ts" / "ndsm_compare.csv").read_text().splitlines()
    assert rows[0] == "date,rmse,bias,rmse_m,mae_m,bias_m,p95_abs_m,dir" and len(rows) == 3
    record = json.loads((tmp_path / "out" / "ts" / "dates.json").read_text())[0]
    assert record["tiles"] == 5 and record["ndsm_compare"]["rmse_m"] == 0.3
