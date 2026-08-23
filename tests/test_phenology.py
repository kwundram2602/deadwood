import os
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402
from deadwood_spectral.phenology import (  # noqa: E402
    FEATURE_NAMES,
    aggregate_series,
    pixel_features,
    stack_paths,
    window_dates,
)

ALL_DATES = (
    "20231114",
    "20240220",  # zwei Zyklen früher
    "20250417",
    "20250907",
    "20251121",
    "20260226",
    "20260313",
    "20260401",  # nach dem Ankerdatum
)


def _stack_dir(tmp_path, dates=ALL_DATES):
    d = tmp_path / "ts"
    d.mkdir()
    for date in dates:
        (d / f"{date}_stack.tif").write_bytes(b"")
    (d / "channels.json").write_text("{}")
    return d


def test_window_dates_keeps_only_the_window(tmp_path):
    dates = window_dates(_stack_dir(tmp_path), "20260313", 12)
    assert dates == ["20250417", "20250907", "20251121", "20260226", "20260313"]


def test_window_dates_includes_the_label_date_and_excludes_the_window_start(tmp_path):
    d = _stack_dir(tmp_path, ("20250313", "20250314", "20260313"))
    assert window_dates(d, "20260313", 12) == ["20250314", "20260313"]


def test_window_dates_raises_when_the_window_is_empty(tmp_path):
    d = _stack_dir(tmp_path, ("20231114",))
    with pytest.raises(ValueError, match="no aligned stack"):
        window_dates(d, "20260313", 12)


def test_window_dates_handles_a_month_end_anchor(tmp_path):
    # 20260331 minus 1 Monat ist der 28.02.2026 — kein 31. Februar.
    d = _stack_dir(tmp_path, ("20260227", "20260301", "20260331"))
    assert window_dates(d, "20260331", 1) == ["20260301", "20260331"]


def test_feature_names_are_fixed_and_carry_no_date():
    assert len(FEATURE_NAMES) == 31
    assert FEATURE_NAMES[-1] == "ndsm"
    assert not any(c.isdigit() for name in FEATURE_NAMES for c in name)


def test_stack_paths_fails_loudly_on_a_missing_stack(tmp_path):
    d = _stack_dir(tmp_path, ("20260313",))
    with pytest.raises(FileNotFoundError, match="20250417"):
        stack_paths(d, ["20250417", "20260313"])


def _series(values_by_measure):
    return {m: np.asarray(v, dtype=np.float32) for m, v in values_by_measure.items()}


def _full_series(ndvi):
    """Alle Messgrößen auf denselben Verlauf setzen — Statistik ist je Größe gleich."""
    ndvi = np.asarray(ndvi, dtype=np.float32)
    return {m: ndvi.copy() for m in ("ndvi", "ndre", "NIR", "brightness", "green_red")}


def test_aggregate_computes_the_statistics_by_hand():
    out = aggregate_series(_full_series([[0.1, 0.5, 0.3]]), min_valid_dates=1)
    assert out.loc[0, "ndvi_max"] == pytest.approx(0.5, abs=1e-6)
    assert out.loc[0, "ndvi_min"] == pytest.approx(0.1, abs=1e-6)
    assert out.loc[0, "ndvi_amplitude"] == pytest.approx(0.4, abs=1e-6)
    assert out.loc[0, "ndvi_mean"] == pytest.approx(0.3, abs=1e-6)
    # Populationsstandardabweichung (ddof=0) von [0.1, 0.5, 0.3]
    assert out.loc[0, "ndvi_std"] == pytest.approx(0.16329932, abs=1e-6)


def test_greenup_slope_is_the_least_squares_slope_over_equal_steps():
    out = aggregate_series(_full_series([[0.0, 0.2, 0.4, 0.6]]), min_valid_dates=1)
    assert out.loc[0, "ndvi_greenup_slope"] == pytest.approx(0.2, abs=1e-6)


def test_greenup_slope_ignores_missing_dates():
    # Dieselbe Gerade, aber der dritte Termin fehlt: die Steigung bleibt 0.2.
    out = aggregate_series(_full_series([[0.0, 0.2, np.nan, 0.6]]), min_valid_dates=1)
    assert out.loc[0, "ndvi_greenup_slope"] == pytest.approx(0.2, abs=1e-6)


def test_statistics_ignore_missing_dates():
    out = aggregate_series(_full_series([[0.1, np.nan, 0.5]]), min_valid_dates=1)
    assert out.loc[0, "ndvi_mean"] == pytest.approx(0.3, abs=1e-6)
    assert out.loc[0, "ndvi_max"] == pytest.approx(0.5, abs=1e-6)


def test_pixels_below_min_valid_dates_become_nan():
    out = aggregate_series(_full_series([[0.1, 0.2, np.nan, np.nan]]), min_valid_dates=3)
    assert out.loc[0].isna().all()


def test_validity_is_shared_across_measures():
    # NIR fehlt am zweiten Termin -> der Termin zählt für KEINE Messgröße.
    series = _full_series([[0.1, 0.9, 0.3]])
    series["NIR"] = np.asarray([[0.1, np.nan, 0.3]], dtype=np.float32)
    out = aggregate_series(series, min_valid_dates=1)
    assert out.loc[0, "ndvi_max"] == pytest.approx(0.3, abs=1e-6)


def test_column_order_is_measures_times_stats():
    out = aggregate_series(_full_series([[0.1, 0.2]]), min_valid_dates=1)
    assert list(out.columns)[:3] == ["ndvi_max", "ndvi_min", "ndvi_amplitude"]
    assert len(out.columns) == 30


def test_a_single_date_yields_zero_amplitude_and_nan_slope():
    out = aggregate_series(_full_series([[0.4]]), min_valid_dates=1)
    assert out.loc[0, "ndvi_amplitude"] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(out.loc[0, "ndvi_greenup_slope"])


GRID = ReferenceGrid(8, 8, from_origin(1000.0, 2000.0, 1.0, 1.0), rasterio.crs.CRS.from_epsg(32736))
BANDS = ("R", "G", "B", "Green", "Red", "RedEdge", "NIR")


def _write_stack(path, nir, red=0.1, nan_pixel=None):
    data = np.zeros((7, 8, 8), dtype="float32")
    data[0], data[1], data[2] = 0.1, 0.2, 0.3  # R, G, B
    data[3] = 0.2  # Green
    data[4] = red
    data[5] = 0.3  # RedEdge
    data[6] = nir
    if nan_pixel is not None:
        data[:, nan_pixel[0], nan_pixel[1]] = np.nan
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=8,
        height=8,
        count=7,
        crs="EPSG:32736",
        transform=GRID.transform,
        nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(BANDS, start=1):
            dst.set_band_description(i, name)
    return path


def _write_ndsm(path, value=7.0):
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=8,
        height=8,
        count=1,
        crs="EPSG:32736",
        transform=GRID.transform,
        nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((1, 8, 8), value, dtype="float32"))
    return path


def _three_stacks(tmp_path, nan_pixel=None):
    d = tmp_path / "ts"
    d.mkdir()
    paths = []
    for date, nir in (("20250417", 0.3), ("20250907", 0.5), ("20260313", 0.7)):
        paths.append(_write_stack(d / f"{date}_stack.tif", nir=nir, nan_pixel=nan_pixel))
    return paths


def test_pixel_features_returns_the_fixed_columns_in_order(tmp_path):
    paths = _three_stacks(tmp_path)
    out = pixel_features(
        paths,
        GRID,
        np.array([0, 5]),
        np.array([1, 6]),
        ndsm_path=_write_ndsm(tmp_path / "ndsm.tif"),
        min_valid_dates=1,
    )
    assert list(out.columns) == list(FEATURE_NAMES)
    assert len(out) == 2


def test_pixel_features_computes_ndvi_statistics_from_the_stacks(tmp_path):
    # Red = 0.1 konstant, NIR = 0.3/0.5/0.7 -> NDVI = 0.5, 0.667, 0.75
    paths = _three_stacks(tmp_path)
    out = pixel_features(
        paths,
        GRID,
        np.array([2]),
        np.array([3]),
        ndsm_path=_write_ndsm(tmp_path / "ndsm.tif"),
        min_valid_dates=1,
    )
    assert out.loc[0, "ndvi_min"] == pytest.approx(0.5, abs=1e-5)
    assert out.loc[0, "ndvi_max"] == pytest.approx(0.75, abs=1e-5)
    assert out.loc[0, "ndsm"] == pytest.approx(7.0, abs=1e-6)


def test_pixel_features_preserves_input_order_across_chunks(tmp_path):
    paths = _three_stacks(tmp_path)
    ndsm = _write_ndsm(tmp_path / "ndsm.tif")
    rows = np.array([7, 0, 4])
    cols = np.array([0, 1, 2])
    # chunk_rows=2 zwingt die Pixel in verschiedene Chunks, in umgekehrter
    # Reihenfolge zur Eingabe.
    chunked = pixel_features(
        paths, GRID, rows, cols, ndsm_path=ndsm, chunk_rows=2, min_valid_dates=1
    )
    whole = pixel_features(
        paths, GRID, rows, cols, ndsm_path=ndsm, chunk_rows=64, min_valid_dates=1
    )
    pd.testing.assert_frame_equal(chunked, whole)


def test_pixel_features_marks_a_pixel_missing_on_every_date(tmp_path):
    paths = _three_stacks(tmp_path, nan_pixel=(3, 3))
    out = pixel_features(
        paths,
        GRID,
        np.array([3, 4]),
        np.array([3, 4]),
        ndsm_path=_write_ndsm(tmp_path / "ndsm.tif"),
        min_valid_dates=1,
    )
    assert out.loc[0, "ndvi_mean"] != out.loc[0, "ndvi_mean"]  # NaN
    assert np.isfinite(out.loc[1, "ndvi_mean"])


def test_pixel_features_rejects_a_stack_off_the_reference_grid(tmp_path):
    paths = _three_stacks(tmp_path)
    other = ReferenceGrid(
        9, 8, from_origin(1000.0, 2000.0, 1.0, 1.0), rasterio.crs.CRS.from_epsg(32736)
    )
    with pytest.raises(ValueError, match="shape"):
        pixel_features(paths, other, np.array([0]), np.array([0]), min_valid_dates=1)
