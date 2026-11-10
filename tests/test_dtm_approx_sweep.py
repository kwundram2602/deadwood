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
    inputs = scene_inputs(prep, "20260313")  # not limited by timeseries.dates
    assert inputs.dsm.endswith("20260313_X_DSM_coreg.tif")
    assert inputs.om.endswith("20260313_X_OM_coreg.tif")
    assert inputs.process_out == tmp_path.joinpath("work", "20260313").relative_to(tmp_path)


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


def _check(dsm_system, dtm_system, n=13.8):
    from explore_and_process.vertical_datum import DatumCheck, HeightSystem

    def hs(system):
        return HeightSystem(system, 0.5, 0.5, 0.5 - n, 1.0, 0.3)

    return DatumCheck(hs(dsm_system), hs(dtm_system), n, "EGM2008")


def test_load_scene_geoid_converts_the_dtm_without_fitting(scene_files, monkeypatch):
    """DSM orthometric, DTM ellipsoidal: reference = DSM - (DTM - N), any tilt stays."""
    import explore_and_process.dtm_approx_sweep as S
    from explore_and_process.dtm_approx_sweep import load_scene_geoid

    n, gsd = 300, 0.1
    transform = from_origin(500000, 5400000, gsd, gsd)
    with rasterio.open(scene_files / "dsm.tif") as src:
        dsm = src.read(1)
    rows = np.arange(n)[:, None] * np.ones((1, n))
    tilt = (0.001 * rows).astype(np.float32)  # 0.3 m across the scene, must survive
    _write(scene_files / "om.tif", np.zeros((n, n), np.float32), transform)
    # ground (dsm minus tree) shifted into ellipsoidal heights (+N) plus a tilt
    _write(scene_files / "dtm.tif", (dsm - np.where(dsm > 105, 10.0, 0.0) + 13.8 + tilt).astype(np.float32), transform)
    monkeypatch.setattr(S, "check_pair", lambda dsm, dtm, cache: _check("orthometric", "ellipsoidal"))
    monkeypatch.setattr(S, "undulation_grid", lambda shape, transform, crs: np.full(shape, 13.8, np.float32))

    scene = load_scene_geoid(str(scene_files / "dsm.tif"), str(scene_files / "om.tif"),
                             str(scene_files / "dtm.tif"), gsd, "d")
    assert scene.reference_mode == "geoid" and scene.datum["mixed"] is True
    ground = scene.reference < 5.0
    # no fit: the reference keeps exactly minus the tilt on the ground
    np.testing.assert_allclose(scene.reference[ground], -tilt[ground], atol=1e-3)


def test_to_dsm_system_directions_and_unknown():
    from explore_and_process.vertical_datum import to_dsm_system

    dtm = np.full((2, 2), 300.0, np.float32)
    n = np.full((2, 2), 13.8, np.float32)
    np.testing.assert_allclose(to_dsm_system(dtm, _check("orthometric", "ellipsoidal"), n), 286.2)
    np.testing.assert_allclose(to_dsm_system(dtm, _check("ellipsoidal", "orthometric"), n), 313.8)
    np.testing.assert_allclose(to_dsm_system(dtm, _check("ellipsoidal", "ellipsoidal"), n), 300.0)
    with pytest.raises(ValueError, match="cannot convert"):
        to_dsm_system(dtm, _check("unknown", "ellipsoidal"), n)


def test_height_system_classification(monkeypatch):
    import explore_and_process.vertical_datum as V

    def fake(ortho):
        return lambda path, res, block, n: {
            "as_orthometric": ortho, "as_ellipsoidal": ortho - n, "tilt": 1.0, "scatter": 0.5, "n": 80, "span": (1, 1)
        }

    monkeypatch.setattr(V, "verify_absolute", fake(-0.8))
    assert V.height_system("x", 13.8).system == "orthometric"
    monkeypatch.setattr(V, "verify_absolute", fake(12.0))
    assert V.height_system("x", 13.8).system == "ellipsoidal"
    monkeypatch.setattr(V, "verify_absolute", fake(47.3))  # neither agrees: broken survey
    assert V.height_system("x", 13.8).system == "unknown"


def test_check_pair_caches_per_raster(tmp_path, monkeypatch):
    import explore_and_process.vertical_datum as V

    calls = []
    monkeypatch.setattr(V, "_centre_lonlat", lambda path: (31.6, -24.96))
    monkeypatch.setattr(V, "geoid_undulation", lambda lon, lat, model: 13.8)
    monkeypatch.setattr(V, "verify_absolute", lambda path, res, block, n: calls.append(path) or {
        "as_orthometric": -0.8, "as_ellipsoidal": -14.6, "tilt": 1.0, "scatter": 0.5, "n": 80, "span": (1, 1)})
    for name in ("dsm.tif", "dtm.tif"):
        (tmp_path / name).write_bytes(b"x")
    cache = tmp_path / "cache.json"
    first = V.check_pair(tmp_path / "dsm.tif", tmp_path / "dtm.tif", cache)
    again = V.check_pair(tmp_path / "dsm.tif", tmp_path / "dtm.tif", cache)
    assert len(calls) == 2 and first == again and not first.mixed


def test_load_scene_geoid_plane_removes_offset_and_tilt(scene_files, monkeypatch):
    import explore_and_process.dtm_approx_sweep as S
    from explore_and_process.dtm_approx_sweep import load_scene_geoid

    n, gsd = 300, 0.1
    transform = from_origin(500000, 5400000, gsd, gsd)
    with rasterio.open(scene_files / "dsm.tif") as src:
        dsm = src.read(1)
    rows = np.arange(n)[:, None] * np.ones((1, n))
    survey = (0.7 + 0.003 * rows).astype(np.float32)  # offset + 0.9 m tilt between the surveys
    _write(scene_files / "om.tif", np.zeros((n, n), np.float32), transform)
    _write(scene_files / "dtm.tif", (dsm - np.where(dsm > 105, 10.0, 0.0) + 13.8 + survey).astype(np.float32),
           transform)
    monkeypatch.setattr(S, "check_pair", lambda dsm, dtm, cache: _check("orthometric", "ellipsoidal"))
    monkeypatch.setattr(S, "undulation_grid", lambda shape, transform, crs: np.full(shape, 13.8, np.float32))

    scene = load_scene_geoid(str(scene_files / "dsm.tif"), str(scene_files / "om.tif"),
                             str(scene_files / "dtm.tif"), gsd, "d", fit="plane")
    assert scene.reference_mode == "geoid_plane" and "plane_after_geoid" in scene.datum
    ground = scene.reference < 5.0
    assert np.abs(scene.reference[ground]).max() < 0.05  # offset and tilt gone
    assert np.median(scene.reference[~ground]) == pytest.approx(10.0, abs=0.05)  # tree height kept


def test_quadratic_surface_removes_doming_a_plane_leaves():
    from explore_and_process.apply_dsm_mask import align_dtm_stages
    from explore_and_process.dtm_approx_sweep import fit_quadratic_surface

    n = 200
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) * 2.0 - 1.0
    ground = np.full((n, n), 300.0, np.float32)
    tree = (yy - 0.3) ** 2 + (xx + 0.2) ** 2 <= 0.1**2
    wall = (np.abs(xx - 0.5) < 0.03).astype(np.float32) * 2.0  # local shape that must survive
    dsm = (ground + wall + np.where(tree, 8.0, 0.0)).astype(np.float32)
    survey = (0.9 + 0.4 * xx - 0.6 * (xx**2 + yy**2)).astype(np.float32)  # offset, tilt, bowl
    dtm = (ground + wall - survey).astype(np.float32)  # same terrain incl. the wall, other survey

    corrected, info = fit_quadratic_surface(dsm, dtm, blocks=16)
    bare = ~tree
    assert np.abs((dsm - corrected)[bare]).max() < 0.02
    assert info["dome"] == pytest.approx(1.2, abs=0.05)  # 0.6 * (2 - 0) over the extent
    plane = align_dtm_stages(dsm, dtm, blocks=16)[0]["plane"]
    assert np.abs((dsm - plane)[bare]).max() > 0.3  # a plane cannot remove the bowl


def test_check_pair_rejects_infinite_undulation_and_drops_void_cache(tmp_path, monkeypatch):
    import json as _json

    import explore_and_process.vertical_datum as V

    monkeypatch.setattr(V, "_centre_lonlat", lambda path: (31.6, -24.96))
    monkeypatch.setattr(V, "geoid_undulation", lambda lon, lat, model: float("inf"))
    for name in ("dsm.tif", "dtm.tif"):
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(RuntimeError, match="geoid grid"):
        V.check_pair(tmp_path / "dsm.tif", tmp_path / "dtm.tif")

    cache = tmp_path / "cache.json"
    void = {"system": "orthometric", "residual_m": 0.3, "as_orthometric_m": 0.3,
            "as_ellipsoidal_m": float("inf"), "tilt_m": 3.6, "scatter_m": 0.4}
    cache.write_text(_json.dumps({V._cache_key(tmp_path / "dsm.tif"): void}))
    calls = []
    monkeypatch.setattr(V, "geoid_undulation", lambda lon, lat, model: 13.8)
    monkeypatch.setattr(V, "verify_absolute", lambda path, res, block, n: calls.append(path) or {
        "as_orthometric": 0.3, "as_ellipsoidal": 0.3 - n, "tilt": 3.6, "scatter": 0.4, "n": 80, "span": (1, 1)})
    check = V.check_pair(tmp_path / "dsm.tif", tmp_path / "dtm.tif", cache)
    assert len(calls) == 2  # the void DSM entry was recomputed
    assert check.dsm.as_ellipsoidal_m == pytest.approx(0.3 - 13.8)


def test_survey_offset_below_half_n_keeps_the_height_system(monkeypatch):
    """20260313-like: ellipsoidal reading +5.57 m, orthometric +19.35 m -> ellipsoidal, flagged."""
    import explore_and_process.vertical_datum as V

    monkeypatch.setattr(V, "verify_absolute", lambda path, res, block, n: {
        "as_orthometric": 19.35, "as_ellipsoidal": 19.35 - n, "tilt": 3.8, "scatter": 0.5, "n": 80, "span": (1, 1)})
    h = V.height_system("x", 13.78)
    assert h.system == "ellipsoidal" and h.residual_m == pytest.approx(5.57)
    check = V.DatumCheck(h, h, 13.78, "EGM2008")
    assert "SURVEY OFFSET" in check.summary()


def test_harmonise_converts_only_a_known_mix(monkeypatch):
    import explore_and_process.vertical_datum as V

    terrain = np.full((3, 3), 300.0, np.float32)
    monkeypatch.setattr(V, "undulation_grid", lambda shape, transform, crs: np.full(shape, 13.8, np.float32))
    monkeypatch.setattr(V, "check_pair", lambda a, b, cache: _check("orthometric", "ellipsoidal"))
    out, info = V.harmonise(terrain, "dsm", "dtm", (3, 3), None, None)
    np.testing.assert_allclose(out, 286.2)
    assert info["converted"] is True
    monkeypatch.setattr(V, "check_pair", lambda a, b, cache: _check("unknown", "ellipsoidal"))
    out, info = V.harmonise(terrain, "dsm", "dtm", (3, 3), None, None)
    assert np.array_equal(out, terrain) and info["converted"] is False

    def boom(*a):
        raise OSError("offline")

    monkeypatch.setattr(V, "check_pair", boom)
    out, info = V.harmonise(terrain, "dsm", "dtm", (3, 3), None, None)
    assert np.array_equal(out, terrain) and info is None
