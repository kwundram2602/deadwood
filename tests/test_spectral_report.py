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
    assert set(sep.columns) == {
        "date", "measure", "jm", "auc_sep", "auc_raw", "n_a", "n_b",
    }
    assert len(sep) == len(DATES)


def test_auc_sep_is_folded_and_auc_raw_is_directional():
    """The two AUC columns must mean different things, and keep meaning them.

    `auc_raw` is P(deadwood > living): deadwood ndvi sits BELOW living ndvi,
    so it is near 0 on a well-separated date. `auc_sep` is the folded
    magnitude, which is what "best separated" means. Publishing only the
    folded value under the plain name `auc` told the reader the opposite of
    the truth, so both must be present and must satisfy the fold exactly.
    """
    sep = separability_table(_table(), DATES, measures=["ndvi"]).set_index("date")

    # Deadwood is the darker class on every date here.
    assert (sep["auc_raw"] < 0.5).all()
    # ... yet separation is high in February, which only auc_sep shows.
    assert sep.loc["20240212", "auc_raw"] == pytest.approx(0.0, abs=0.05)
    assert sep.loc["20240212", "auc_sep"] == pytest.approx(1.0, abs=0.05)
    # The fold relationship, pinned: a revert to one ambiguous column fails here.
    folded = np.maximum(sep["auc_raw"], 1.0 - sep["auc_raw"])
    assert np.allclose(sep["auc_sep"].to_numpy(), folded.to_numpy())
    assert (sep["auc_sep"] >= 0.5).all()


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
    assert {"date", "measure", "jm", "auc_sep", "auc_raw"} <= set(summary.columns)
    # summary.csv is the artefact a human reads: the directional value has to
    # survive into it, not just the folded magnitude.
    folded = np.maximum(summary["auc_raw"], 1.0 - summary["auc_raw"])
    assert np.allclose(summary["auc_sep"].to_numpy(), folded.to_numpy())


def test_best_date_selects_on_the_folded_not_the_directional_auc():
    """Regression: an argmax over auc_raw picks the WORST-separated date."""
    sep = separability_table(_table(), DATES, measures=["ndvi"])
    assert best_date(sep) == "20240212"
    assert sep.set_index("date")["auc_raw"].idxmax() != "20240212"


def test_seasonal_amplitude_propagates_nan_like_the_classifier_feature():
    """Stage B and Stage C must compute the SAME statistic.

    report.seasonal_amplitude used pandas max/min (skipna=True) while
    features.build_features' `<measure>_amplitude` uses numpy (NaN
    propagating). For ndvi [0.1, NaN, 0.2] Stage B said 0.1 and Stage C said
    NaN. Real stacks are 45.7% NaN with a different footprint per date, so the
    descriptive plot silently mixed 12-date and 2-date amplitudes. This pins
    the two to one definition: NaN-propagating, i.e. "observed on every date".
    """
    from deadwood_spectral.features import build_features

    df = pd.DataFrame(
        {
            "class_name": ["deadwood", "deadwood"],
            "ndvi_20230824": [0.1, 0.1],
            "ndvi_20231103": [np.nan, 0.3],
            "ndvi_20240212": [0.2, 0.2],
        }
    )
    stage_b = seasonal_amplitude(df, DATES, measure="ndvi")

    # The same rows through the Stage C feature builder.
    full = df.assign(
        **{
            f"{m}_{d}": df["ndvi_20230824"] if m != "ndvi" else df[f"ndvi_{d}"]
            for m in ("ndvi", "ndre", "NIR", "Red", "brightness", "green_red")
            for d in DATES
        },
        ndsm=1.0,
    )
    stage_c = build_features(full, DATES)["ndvi_amplitude"]

    assert np.isnan(stage_b.iloc[0])              # was 0.1 before the fix
    assert stage_b.iloc[1] == pytest.approx(0.2)
    assert np.allclose(stage_b.to_numpy(), stage_c.to_numpy(), equal_nan=True)


def test_amplitude_population_counts_the_rows_the_statistic_used():
    """The descriptive path must state the population it summarises."""
    from deadwood_spectral.report import amplitude_population

    df = _table()
    df.loc[df.index[:10], "ndvi_20231103"] = np.nan   # 10 deadwood rows incomplete
    population = amplitude_population(df, DATES, measures=("ndvi",))
    deadwood = population.set_index("class_name").loc["deadwood"]

    assert deadwood["n_rows"] == 40
    assert deadwood["n_complete"] == 30
    assert deadwood["n_dates"] == 3
    assert deadwood["complete_frac"] == pytest.approx(0.75)


def test_run_report_writes_the_amplitude_population(tmp_path):
    df = _table()
    df.loc[df.index[:10], "ndvi_20231103"] = np.nan
    out = run_report(df, DATES, tmp_path / "rep")
    population = pd.read_csv(out / "amplitude_population.csv")
    assert {"measure", "class_name", "n_rows", "n_complete"} <= set(population.columns)
    row = population[(population["measure"] == "ndvi")
                     & (population["class_name"] == "deadwood")].iloc[0]
    assert (row["n_rows"], row["n_complete"]) == (40, 30)


def test_empty_group_plot_leaves_a_log_trace(tmp_path, caplog):
    """A vanished plot must not be silent — 11 of 18 trees can be filtered out."""
    import logging

    df = _table().assign(species=np.nan)
    with caplog.at_level(logging.WARNING, logger="deadwood_spectral.report"):
        run_report(df, DATES, tmp_path / "rep")
    assert any("deadwood_by_species.png" in r.getMessage() for r in caplog.records)
