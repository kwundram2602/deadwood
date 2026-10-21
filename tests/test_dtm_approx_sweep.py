# tests/test_dtm_approx_sweep.py
import csv
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import pytest
import rasterio
from omegaconf import OmegaConf
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from explore_and_process.apply_dsm_mask import DtmApproxParams  # noqa: E402
from explore_and_process.dtm_approx_sweep import (  # noqa: E402
    load_scene,
    one_at_a_time,
    run_sweep,
    scene_inputs,
)

BASE = DtmApproxParams(cell=0.5, max_window=12.0)


def test_one_at_a_time_changes_one_parameter_per_run():
    runs = one_at_a_time(BASE, {"max_slope": [0.3, 0.5, 0.8], "dh0": [0.1, 0.1]})
    assert runs[0].name == "baseline" and runs[0].params == BASE
    # 0.5 is the baseline's max_slope and the repeated dh0 0.1 is dropped
    assert [r.name for r in runs[1:]] == ["max_slope=0.3", "max_slope=0.8", "dh0=0.1"]
    for run in runs[1:]:
        changed = [k for k, v in asdict(run.params).items() if v != getattr(BASE, k)]
        assert changed == [run.varied]


def test_one_at_a_time_rejects_unknown_and_invalid_values():
    with pytest.raises(ValueError, match="unknown dtm_approx parameters"):
        one_at_a_time(BASE, {"window": [1.0]})
    with pytest.raises(ValueError, match="percentile"):
        one_at_a_time(BASE, {"percentile": [150]})


def _write(path, arr, transform, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", width=arr.shape[1], height=arr.shape[0], count=1,
        crs="EPSG:32636", transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr[np.newaxis].astype(np.float32))


@pytest.fixture
def scene_files(tmp_path):
    """A 300 x 300 px scene at 0.1 m: sloped ground, one 10 m tree, reference nDSM from the true ground."""
    n, gsd = 300, 0.1
    transform = from_origin(500000, 5400000, gsd, gsd)
    rows, cols = np.meshgrid(np.arange(n) * gsd, np.arange(n) * gsd, indexing="ij")
    ground = (100.0 + 0.05 * cols).astype(np.float32)
    tree = (rows - 15.0) ** 2 + (cols - 15.0) ** 2 <= 3.0**2
    dsm = ground + np.where(tree, 10.0, 0.0).astype(np.float32)
    _write(tmp_path / "dsm.tif", dsm, transform)
    run_dir = tmp_path / "process_out" / "ndsm_in_m" / "20260101_000000"
    run_dir.mkdir(parents=True)
    _write(run_dir / "dsm_ndsm_dtm_raw_m.tif", dsm - ground, transform, nodata=float("nan"))
    return tmp_path


def test_load_scene_reads_reference_grid_and_crops(scene_files):
    scene = load_scene(str(scene_files / "dsm.tif"), scene_files / "process_out", "d")
    assert scene.dsm.shape == scene.reference.shape == (300, 300)
    assert scene.gsd == pytest.approx(0.1)
    small = load_scene(str(scene_files / "dsm.tif"), scene_files / "process_out", "d", window_px=100)
    assert small.dsm.shape == (100, 100)
    # the crop is centred: its DSM matches the full scene's centre
    np.testing.assert_allclose(small.dsm, scene.dsm[100:200, 100:200], atol=1e-3)


def test_load_scene_without_reference_raises(scene_files):
    with pytest.raises(FileNotFoundError, match="_dtm_raw_m.tif"):
        load_scene(str(scene_files / "dsm.tif"), scene_files / "process_out", "d", reference_run="nope")


def test_run_sweep_writes_ranked_run_folder(scene_files):
    scene = load_scene(str(scene_files / "dsm.tif"), scene_files / "process_out", "d")
    # max_window 2 m is narrower than the 6 m tree: it stays in the terrain, a large error
    runs = one_at_a_time(BASE, {"max_window": [2.0, 20.0]})
    out = scene_files / "sweep" / "run"
    ranked = run_sweep(scene, runs, 16.0, out, config={"sweep": {"date": "d"}})

    assert [r["rmse_m"] for r in ranked] == sorted(r["rmse_m"] for r in ranked)
    assert ranked[-1]["name"] == "max_window=2"
    with (out / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["name"] for r in rows] == [r["name"] for r in ranked]
    assert float(next(r for r in rows if r["name"] == "baseline")["delta_rmse_m"]) == 0.0
    best = OmegaConf.load(out / "best_params.yaml").dtm_approx
    assert DtmApproxParams.from_cfg(best) == next(r.params for r in runs if r.name == ranked[0]["name"])
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["baseline"]["name"] == "baseline" and metrics["best"]["name"] == ranked[0]["name"]
    assert len(list((out / "plots").glob("*.png"))) == 3
    assert (out / "sensitivity.png").is_file() and (out / "results_runs.csv").is_file()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_runs"] == 3 and "finished" in meta


def test_run_sweep_never_overwrites_a_run_folder(scene_files):
    scene = load_scene(str(scene_files / "dsm.tif"), scene_files / "process_out", "d", window_px=50)
    out = scene_files / "sweep" / "run"
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run_sweep(scene, one_at_a_time(BASE, {}), 16.0, out)


def test_scene_inputs_time_series_picks_the_date(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for d in ("20260210", "20260313"):
        for kind in ("OM", "DSM"):
            (tmp_path / kind.lower()).mkdir(exist_ok=True)
            (tmp_path / kind.lower() / f"{d}_X_{kind}_coreg.tif").touch()
    prep = OmegaConf.create({"timeseries": {
        "om_dir": "om", "om_pattern": "*_OM_coreg.tif", "dsm_dir": "dsm", "dsm_pattern": "*_DSM_coreg.tif",
        "dates": [20260210], "max_dates": 1, "work_dir": "work",
    }})
    dsm, process_out = scene_inputs(prep, "20260313")  # not limited by timeseries.dates
    assert dsm.endswith("20260313_X_DSM_coreg.tif")
    assert process_out == tmp_path.joinpath("work", "20260313").relative_to(tmp_path)


def test_zone_metrics_split_inside_and_rest():
    from explore_and_process.dtm_approx_sweep import zone_metrics

    err = np.array([[1.0, -1.0], [0.5, np.nan]], dtype=np.float32)
    wall = np.array([[True, True], [False, False]])
    m = zone_metrics(err, {"wall": wall})
    assert m["rmse_m_wall"] == pytest.approx(1.0) and m["bias_m_wall"] == pytest.approx(0.0)
    assert m["n_valid_rest"] == 1 and m["rmse_m_rest"] == pytest.approx(0.5)
    assert zone_metrics(err, {}) == {}


def test_run_sweep_reports_zone_metrics(scene_files):
    import geopandas as gpd
    from shapely.geometry import box

    # a zone over the tree (rows/cols 12-18 m from the origin 500000 / 5400000)
    zone = gpd.GeoDataFrame(geometry=[box(500012, 5399982, 500018, 5399988)], crs="EPSG:32636")
    zone.to_file(scene_files / "tree.gpkg")
    scene = load_scene(str(scene_files / "dsm.tif"), scene_files / "process_out", "d",
                       zones={"tree": str(scene_files / "tree.gpkg")})
    assert scene.zones["tree"].sum() == 60 * 60
    out = scene_files / "sweep" / "zones"
    ranked = run_sweep(scene, one_at_a_time(BASE, {"max_window": [2.0]}), 16.0, out, metric="rmse_m_rest")
    assert {"rmse_m_tree", "bias_m_rest", "n_valid_rest"} <= set(ranked[0])
    # the narrow window leaves the tree as terrain: the error is inside the zone
    narrow = next(r for r in ranked if r["name"] == "max_window=2")
    assert narrow["rmse_m_tree"] > 5.0 and narrow["rmse_m_tree"] > narrow["rmse_m_rest"]
