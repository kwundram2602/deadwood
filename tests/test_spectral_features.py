import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.features import (  # noqa: E402
    PER_DATE_MEASURES,
    TEMPORAL_MEASURES,
    TEMPORAL_STATS,
    assert_feature_names,
    build_features,
    feature_names,
    load_feature_names,
    save_feature_names,
)

DATES = ["20250801", "20251115", "20260212"]


def _table(n=6):
    data = {"class_name": ["deadwood"] * n, "ndsm": np.linspace(1.0, 6.0, n)}
    for i, date in enumerate(DATES):
        for measure in PER_DATE_MEASURES:
            data[f"{measure}_{date}"] = np.full(n, 0.1 * (i + 1))
    return pd.DataFrame(data)


def test_feature_names_length_full_set():
    names = feature_names(DATES)
    expected = len(PER_DATE_MEASURES) * len(DATES) + len(TEMPORAL_MEASURES) * len(TEMPORAL_STATS) + 1
    assert len(names) == expected


def test_feature_names_are_deterministic():
    assert feature_names(DATES) == feature_names(DATES)


def test_feature_names_reduced_set_is_temporal_plus_static():
    names = feature_names(DATES, per_date=False)
    assert len(names) == len(TEMPORAL_MEASURES) * len(TEMPORAL_STATS) + 1
    assert names[-1] == "ndsm"


def test_feature_names_static_only():
    assert feature_names(DATES, per_date=False, temporal=False) == ["ndsm"]


def test_build_features_columns_match_feature_names():
    matrix = build_features(_table(), DATES)
    assert list(matrix.columns) == feature_names(DATES)


def test_amplitude_is_max_minus_min():
    matrix = build_features(_table(), DATES)
    assert matrix["ndvi_amplitude"].iloc[0] == pytest.approx(0.3 - 0.1, abs=1e-6)


def test_max_and_min_are_taken_across_dates():
    matrix = build_features(_table(), DATES)
    assert matrix["ndvi_max"].iloc[0] == pytest.approx(0.3, abs=1e-6)
    assert matrix["ndvi_min"].iloc[0] == pytest.approx(0.1, abs=1e-6)


def test_greenup_slope_is_positive_for_a_rising_series():
    assert build_features(_table(), DATES)["ndvi_greenup_slope"].iloc[0] > 0


def test_greenup_slope_is_zero_for_a_flat_series():
    df = _table()
    for date in DATES:
        df[f"ndvi_{date}"] = 0.1
    assert build_features(df, DATES)["ndvi_greenup_slope"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_static_column_is_carried_through():
    matrix = build_features(_table(), DATES)
    assert matrix["ndsm"].iloc[-1] == pytest.approx(6.0, abs=1e-6)


def test_missing_date_column_raises():
    df = _table().drop(columns=["ndvi_20251115"])
    with pytest.raises(ValueError, match="20251115"):
        build_features(df, DATES)


def test_missing_ndsm_raises():
    df = _table().drop(columns=["ndsm"])
    with pytest.raises(ValueError, match="ndsm"):
        build_features(df, DATES)


def test_nan_in_one_date_propagates_to_the_aggregate():
    df = _table()
    df.loc[0, "ndvi_20251115"] = np.nan
    matrix = build_features(df, DATES)
    assert np.isnan(matrix.loc[0, "ndvi_mean"])


def test_save_and_load_feature_names_round_trip(tmp_path):
    path = tmp_path / "feature_names.json"
    save_feature_names(feature_names(DATES), path)
    assert load_feature_names(path) == feature_names(DATES)


def test_assert_feature_names_accepts_identical_lists():
    assert_feature_names(feature_names(DATES), feature_names(DATES))


def test_assert_feature_names_rejects_reordering():
    a = feature_names(DATES)
    with pytest.raises(ValueError, match="order"):
        assert_feature_names([a[1], a[0]] + a[2:], a)
