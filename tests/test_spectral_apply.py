import json
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.apply import aggregate_objects, predict_scene, window_table  # noqa: E402
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402

GRID = ReferenceGrid(16, 16, from_origin(1000.0, 2000.0, 1.0, 1.0), rasterio.crs.CRS.from_epsg(32736))
BANDS = ("R", "G", "B", "Green", "Red", "RedEdge", "NIR")
DATES = ["20250801", "20251115"]


def _stack_dir(tmp_path):
    d = tmp_path / "ts"
    d.mkdir(exist_ok=True)
    for i, date in enumerate(DATES):
        data = np.zeros((7, 16, 16), dtype="float32")
        data[0], data[1], data[2] = 0.1, 0.2, 0.3
        data[3], data[4], data[5] = 0.2, 0.1, 0.3
        data[6] = 0.5 + 0.1 * i
        # A patch that stays flat across dates — the "deadwood" corner.
        data[6, :4, :4] = 0.15
        profile = dict(
            driver="GTiff", dtype="float32", width=16, height=16, count=7,
            crs="EPSG:32736", transform=GRID.transform, nodata=np.nan,
        )
        with rasterio.open(d / f"{date}_stack.tif", "w", **profile) as dst:
            dst.write(data)
            for b, name in enumerate(BANDS, start=1):
                dst.set_band_description(b, name)
    (d / "channels.json").write_text(json.dumps({"names": list(BANDS)}))
    return d


def _ndsm(tmp_path):
    path = tmp_path / "ndsm.tif"
    profile = dict(
        driver="GTiff", dtype="float32", width=16, height=16, count=1,
        crs="EPSG:32736", transform=GRID.transform, nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((1, 16, 16), 4.0, dtype="float32"))
    return path


class _ConstantModel:
    """Predicts class 2 wherever ndsm > 1, else class 0. Deterministic."""

    classes_ = np.array([0, 1, 2])

    def __init__(self, ndsm_column: int):
        self.ndsm_column = ndsm_column

    def predict_proba(self, X):
        tall = X[:, self.ndsm_column] > 1.0
        out = np.zeros((X.shape[0], 3))
        out[tall, 2] = 1.0
        out[~tall, 0] = 1.0
        return out


def test_window_table_has_one_row_per_pixel(tmp_path):
    table = window_table(_stack_dir(tmp_path), DATES, GRID, 0, 0, 8, 8, _ndsm(tmp_path))
    assert len(table) == 64


def test_window_table_columns_feed_build_features(tmp_path):
    from deadwood_spectral.features import build_features, feature_names

    table = window_table(_stack_dir(tmp_path), DATES, GRID, 0, 0, 8, 8, _ndsm(tmp_path))
    matrix = build_features(table, DATES)
    assert list(matrix.columns) == feature_names(DATES)


def test_window_table_values_match_the_raster(tmp_path):
    table = window_table(_stack_dir(tmp_path), DATES, GRID, 0, 0, 8, 8, _ndsm(tmp_path))
    assert table["NIR_20250801"].iloc[0] == pytest.approx(0.15, abs=1e-6)
    assert table["ndsm"].iloc[0] == pytest.approx(4.0, abs=1e-6)


def test_predict_scene_shape_and_probability_sum(tmp_path):
    from deadwood_spectral.features import feature_names

    features = feature_names(DATES)
    model = _ConstantModel(features.index("ndsm"))
    proba = predict_scene(
        _stack_dir(tmp_path), DATES, GRID, model, features, _ndsm(tmp_path),
        {"per_date": True, "temporal": True, "static": True},
        tile_size=8, stride=8,
    )
    assert proba.shape == (3, 16, 16)
    assert np.allclose(proba.sum(axis=0), 1.0, atol=1e-5)


def test_blockwise_result_equals_wholearray_result(tmp_path):
    """Tiling must not change the answer."""
    from deadwood_spectral.features import feature_names

    features = feature_names(DATES)
    model = _ConstantModel(features.index("ndsm"))
    switches = {"per_date": True, "temporal": True, "static": True}
    tiled = predict_scene(_stack_dir(tmp_path), DATES, GRID, model, features,
                          _ndsm(tmp_path), switches, tile_size=8, stride=4)
    whole = predict_scene(_stack_dir(tmp_path), DATES, GRID, model, features,
                          _ndsm(tmp_path), switches, tile_size=16, stride=16)
    assert np.allclose(tiled, whole, atol=1e-5)


def test_predict_scene_rejects_mismatched_features(tmp_path):
    from deadwood_spectral.features import feature_names

    features = feature_names(DATES)
    model = _ConstantModel(features.index("ndsm"))
    with pytest.raises(ValueError, match="feature"):
        predict_scene(
            _stack_dir(tmp_path), DATES, GRID, model, features[:-1], _ndsm(tmp_path),
            {"per_date": True, "temporal": True, "static": True}, tile_size=8, stride=8,
        )


def test_aggregate_objects_finds_one_blob():
    class_raster = np.zeros(GRID.shape, dtype=np.uint8)
    class_raster[2:8, 2:8] = 2
    prob = np.zeros((3, *GRID.shape), dtype=np.float32)
    prob[2, 2:8, 2:8] = 0.9
    objects = aggregate_objects(class_raster, prob, GRID)
    assert len(objects) == 1
    assert objects["area_m2"].iloc[0] == pytest.approx(36.0, abs=1e-6)
    assert objects["mean_prob"].iloc[0] == pytest.approx(0.9, abs=1e-5)


def test_aggregate_objects_drops_blobs_below_the_area_floor():
    class_raster = np.zeros(GRID.shape, dtype=np.uint8)
    class_raster[2, 2] = 2                    # 1 m2
    class_raster[6:10, 6:10] = 2              # 16 m2
    prob = np.zeros((3, *GRID.shape), dtype=np.float32)
    prob[2] = 0.8
    objects = aggregate_objects(class_raster, prob, GRID, min_object_m2=4.0)
    assert len(objects) == 1
    assert objects["area_m2"].iloc[0] == pytest.approx(16.0, abs=1e-6)


def test_aggregate_objects_empty_input_returns_empty_frame():
    class_raster = np.zeros(GRID.shape, dtype=np.uint8)
    prob = np.zeros((3, *GRID.shape), dtype=np.float32)
    objects = aggregate_objects(class_raster, prob, GRID)
    assert len(objects) == 0
    assert {"area_m2", "mean_prob"} <= set(objects.columns)


def test_aggregate_objects_carries_mean_height():
    class_raster = np.zeros(GRID.shape, dtype=np.uint8)
    class_raster[2:8, 2:8] = 2
    prob = np.zeros((3, *GRID.shape), dtype=np.float32)
    prob[2, 2:8, 2:8] = 0.9
    ndsm = np.full(GRID.shape, 5.0, dtype=np.float32)
    objects = aggregate_objects(class_raster, prob, GRID, ndsm=ndsm)
    assert objects["mean_height_m"].iloc[0] == pytest.approx(5.0, abs=1e-6)
