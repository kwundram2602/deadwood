import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.report import (  # noqa: E402
    best_date,
    class_auc,
    jeffries_matusita,
    run_report,
    seasonal_amplitude,
    separability_table,
)

DATES = ["20230824", "20231103", "20240212"]


def _table(seed=0):
    """Deadwood: flat ndvi ~0.1. Living: dry-season dip, wet-season peak."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(40):
        rows.append(
            {
                "class_name": "deadwood",
                "group_id": f"tree:{i % 4}",
                "ndvi_20230824": 0.10 + rng.normal(0, 0.01),
                "ndvi_20231103": 0.11 + rng.normal(0, 0.01),
                "ndvi_20240212": 0.10 + rng.normal(0, 0.01),
            }
        )
    for i in range(40):
        rows.append(
            {
                "class_name": "living",
                "group_id": f"block:{i % 8}",
                "ndvi_20230824": 0.12 + rng.normal(0, 0.01),
                "ndvi_20231103": 0.45 + rng.normal(0, 0.01),
                "ndvi_20240212": 0.75 + rng.normal(0, 0.01),
            }
        )
    for i in range(40):
        rows.append(
            {
                "class_name": "background",
                "group_id": f"block:{i % 8}",
                "ndvi_20230824": 0.05 + rng.normal(0, 0.01),
                "ndvi_20231103": 0.20 + rng.normal(0, 0.01),
                "ndvi_20240212": 0.30 + rng.normal(0, 0.01),
            }
        )
    return pd.DataFrame(rows)


def test_jm_of_identical_distributions_is_zero():
    a = np.random.default_rng(0).normal(0, 1, 500)
    assert jeffries_matusita(a, a.copy()) == pytest.approx(0.0, abs=1e-6)


def test_jm_of_far_apart_distributions_approaches_two():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, 500)
    b = rng.normal(10, 0.1, 500)
    assert jeffries_matusita(a, b) == pytest.approx(2.0, abs=1e-3)


def test_jm_is_bounded():
    rng = np.random.default_rng(1)
    value = jeffries_matusita(rng.normal(0, 1, 200), rng.normal(1, 2, 200))
    assert 0.0 <= value <= 2.0


def test_jm_ignores_nan():
    a = np.array([0.1, 0.2, np.nan, 0.15])
    b = np.array([0.9, np.nan, 0.85, 0.95])
    assert np.isfinite(jeffries_matusita(a, b))


def test_jm_with_too_few_finite_values_is_nan():
    assert np.isnan(jeffries_matusita(np.array([np.nan, 1.0]), np.array([np.nan])))


def test_auc_perfect_separation_is_one():
    a = np.array([3.0, 4.0, 5.0])
    b = np.array([0.0, 1.0, 2.0])
    assert class_auc(a, b) == pytest.approx(1.0)


def test_auc_identical_distributions_is_half():
    a = np.array([1.0, 2.0, 3.0])
    assert class_auc(a, a.copy()) == pytest.approx(0.5)


def test_separability_table_shape_and_columns():
    sep = separability_table(_table(), DATES, measures=["ndvi"])
    assert set(sep.columns) == {"date", "measure", "jm", "auc", "n_a", "n_b"}
    assert len(sep) == len(DATES)


def test_separability_is_worst_in_the_dry_season():
    """August is late dry season: living trees are leaf-off too, so ndvi
    separates deadwood from living far worse than in February."""
    sep = separability_table(_table(), DATES, measures=["ndvi"]).set_index("date")
    assert sep.loc["20230824", "jm"] < sep.loc["20240212", "jm"]


def test_best_date_picks_the_highest_auc():
    assert best_date(separability_table(_table(), DATES, measures=["ndvi"])) == "20240212"


def test_seasonal_amplitude_is_larger_for_living():
    df = _table()
    amp = seasonal_amplitude(df, DATES, measure="ndvi")
    df = df.assign(amplitude=amp)
    means = df.groupby("class_name")["amplitude"].mean()
    assert means["living"] > 3 * means["deadwood"]


def test_run_report_writes_summary_and_plots(tmp_path):
    out = run_report(_table(), DATES, tmp_path / "rep")
    assert (out / "summary.csv").exists()
    assert any(out.glob("*.png"))
    summary = pd.read_csv(out / "summary.csv")
    assert {"date", "measure", "jm", "auc"} <= set(summary.columns)
