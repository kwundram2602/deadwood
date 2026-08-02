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
