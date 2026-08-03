import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.extract import (  # noqa: E402
    available_dates,
    extract_samples,
    feature_column,
)
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402

GRID = ReferenceGrid(8, 8, from_origin(1000.0, 2000.0, 1.0, 1.0), rasterio.crs.CRS.from_epsg(32736))
BANDS = ("R", "G", "B", "Green", "Red", "RedEdge", "NIR")


def _stack(path, nir=0.5, red=0.1, nan_pixel=None):
    data = np.zeros((7, 8, 8), dtype="float32")
    data[0], data[1], data[2] = 0.1, 0.2, 0.3     # R, G, B
    data[3] = 0.2                                  # Green
    data[4] = red
    data[5] = 0.3                                  # RedEdge
    data[6] = nir
    if nan_pixel is not None:
        data[:, nan_pixel[0], nan_pixel[1]] = np.nan
    profile = dict(
        driver="GTiff", dtype="float32", width=8, height=8, count=7,
        crs="EPSG:32736", transform=GRID.transform, nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(BANDS, start=1):
            dst.set_band_description(i, name)
    return path


def _stack_dir(tmp_path, dates=("20230824", "20230906")):
    d = tmp_path / "ts"
    d.mkdir()
    for i, date in enumerate(dates):
        _stack(d / f"{date}_stack.tif", nir=0.5 + 0.1 * i)
    (d / "channels.json").write_text(json.dumps({"names": list(BANDS)}))
    return d


def _samples():
    return pd.DataFrame(
        {
            "row": [0, 1, 2],
            "col": [0, 1, 2],
            "class_name": ["deadwood", "living", "background"],
            "class_code": [2, 1, 0],
            "tree_id": ["4170", pd.NA, pd.NA],
            "group_id": ["tree:4170", "block:1", "block:2"],
        }
    )


def test_feature_column_naming():
    assert feature_column("ndvi", "20230824") == "ndvi_20230824"


def test_available_dates_sorted(tmp_path):
    assert available_dates(_stack_dir(tmp_path)) == ["20230824", "20230906"]


def test_available_dates_honours_exclusions(tmp_path):
    d = _stack_dir(tmp_path)
    assert available_dates(d, exclude=["20230824"]) == ["20230906"]


def test_extract_adds_one_column_per_band_and_index_and_date(tmp_path):
    out = extract_samples(_samples(), _stack_dir(tmp_path), GRID)
    for date in ("20230824", "20230906"):
        for name in (*BANDS, "ndvi", "ndre", "brightness"):
            assert feature_column(name, date) in out.columns


def test_extract_preserves_label_columns_and_row_order(tmp_path):
    samples = _samples()
    out = extract_samples(samples, _stack_dir(tmp_path), GRID)
    assert out["group_id"].tolist() == samples["group_id"].tolist()
    assert out["class_code"].tolist() == samples["class_code"].tolist()


def test_extracted_ndvi_matches_the_band_values(tmp_path):
    out = extract_samples(_samples(), _stack_dir(tmp_path), GRID)
    assert out["ndvi_20230824"].iloc[0] == pytest.approx(0.4 / 0.6, abs=1e-5)
    assert out["NIR_20230824"].iloc[0] == pytest.approx(0.5, abs=1e-6)


def test_nodata_becomes_nan_not_zero(tmp_path):
    d = tmp_path / "ts"
    d.mkdir()
    _stack(d / "20230824_stack.tif", nan_pixel=(0, 0))
    (d / "channels.json").write_text(json.dumps({"names": list(BANDS)}))
    out = extract_samples(_samples(), d, GRID)
    assert np.isnan(out["NIR_20230824"].iloc[0])
    assert np.isnan(out["ndvi_20230824"].iloc[0])
    assert out["NIR_20230824"].iloc[1] == pytest.approx(0.5, abs=1e-6)


def test_ndsm_column_added_when_path_given(tmp_path):
    ndsm = tmp_path / "ndsm.tif"
    profile = dict(
        driver="GTiff", dtype="float32", width=8, height=8, count=1,
        crs="EPSG:32736", transform=GRID.transform, nodata=np.nan,
    )
    with rasterio.open(ndsm, "w", **profile) as dst:
        dst.write(np.full((1, 8, 8), 3.5, dtype="float32"))
    out = extract_samples(_samples(), _stack_dir(tmp_path), GRID, ndsm_path=ndsm)
    assert out["ndsm"].iloc[0] == pytest.approx(3.5, abs=1e-6)


def test_missing_stack_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="no aligned stacks"):
        extract_samples(_samples(), tmp_path / "empty", GRID)


def test_offgrid_stack_raises(tmp_path):
    d = tmp_path / "ts"
    d.mkdir()
    path = d / "20230824_stack.tif"
    profile = dict(
        driver="GTiff", dtype="float32", width=8, height=8, count=7,
        crs="EPSG:32736", transform=from_origin(1002.0, 2000.0, 1.0, 1.0), nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((7, 8, 8), dtype="float32"))
    with pytest.raises(ValueError, match="transform"):
        extract_samples(_samples(), d, GRID)


def _ndsm_raster(path, value=3.5):
    profile = dict(
        driver="GTiff", dtype="float32", width=8, height=8, count=1,
        crs="EPSG:32736", transform=GRID.transform, nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((1, 8, 8), value, dtype="float32"))
    return path


def test_bad_ndsm_path_fails_before_any_extraction_work(tmp_path, monkeypatch):
    """Validate every input up front, not after reading every date.

    On real data the stacks are ~800 MB each; validating the ndsm only after
    the date loop throws away tens of minutes and never writes
    samples.parquet. The guard is proven by making any raster read fatal: a
    correct implementation raises on the missing path before reading anything.
    """
    import deadwood_spectral.extract as extract_mod

    stack_dir = _stack_dir(tmp_path)

    def _explode(*args, **kwargs):
        raise AssertionError("extraction started before the inputs were validated")

    monkeypatch.setattr(extract_mod, "_read_at", _explode)

    with pytest.raises(FileNotFoundError, match="before extraction started"):
        extract_samples(
            _samples(), stack_dir, GRID, ndsm_path=tmp_path / "does-not-exist.tif"
        )


def test_missing_stack_file_is_reported_up_front(tmp_path):
    stack_dir = _stack_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="20991231_stack.tif"):
        extract_samples(_samples(), stack_dir, GRID, dates=["20230824", "20991231"])


def test_ndsm_signature_distinguishes_two_variants_on_the_same_grid(tmp_path):
    """The whole point: both nDSM variants pass assert_matches_grid."""
    from deadwood_spectral.extract import ndsm_signature

    metres = ndsm_signature(_ndsm_raster(tmp_path / "ndsm_m.tif", value=7.5))
    normalized = ndsm_signature(_ndsm_raster(tmp_path / "ndsm_norm.tif", value=0.3))
    assert metres["window_checksum"] != normalized["window_checksum"]
    # Same file read twice is the same signature.
    assert ndsm_signature(tmp_path / "ndsm_m.tif")["window_checksum"] == (
        metres["window_checksum"]
    )


def test_assert_same_ndsm_rejects_a_different_ndsm(tmp_path):
    from deadwood_spectral.extract import assert_same_ndsm, ndsm_signature

    trained_on = ndsm_signature(_ndsm_raster(tmp_path / "ndsm_m.tif", value=7.5))
    other = _ndsm_raster(tmp_path / "ndsm_norm.tif", value=0.3)

    assert_same_ndsm(trained_on, tmp_path / "ndsm_m.tif")  # must not raise
    with pytest.raises(ValueError, match="nDSM mismatch"):
        assert_same_ndsm(trained_on, other)


def test_ndsm_reference_round_trips_and_survives_a_rename(tmp_path):
    from deadwood_spectral.extract import (
        assert_same_ndsm,
        load_ndsm_reference,
        ndsm_signature,
        samples_ndsm_reference_path,
        save_ndsm_reference,
    )

    signature = ndsm_signature(_ndsm_raster(tmp_path / "ndsm_m.tif", value=7.5))
    path = samples_ndsm_reference_path(tmp_path / "samples.parquet")
    save_ndsm_reference(signature, path)
    assert path.name == "samples_ndsm_reference.json"
    assert load_ndsm_reference(path) == signature
    assert load_ndsm_reference(tmp_path / "nothing.json") is None

    # Content decides, not the path: a renamed but identical file is accepted.
    renamed = tmp_path / "ndsm_m_copy.tif"
    renamed.write_bytes((tmp_path / "ndsm_m.tif").read_bytes())
    assert_same_ndsm(load_ndsm_reference(path), renamed)
