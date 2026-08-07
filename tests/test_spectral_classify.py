import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.classify import (  # noqa: E402
    grouped_cv,
    leave_one_tree_out,
    load_model,
    make_model,
    save_model,
    train_model,
)
from deadwood_spectral.features import PER_DATE_MEASURES, feature_names  # noqa: E402

DATES = ["20250801", "20251115", "20260212"]


def _table(seed=0, n_per_class=60, n_dead_trees=6):
    """Separable by construction: living swings, deadwood is flat and tall,
    background is flat and low."""
    rng = np.random.default_rng(seed)
    frames = []
    profiles = {
        "deadwood": ([0.10, 0.11, 0.10], 4.0),
        "living": ([0.12, 0.45, 0.75], 5.0),
        "background": ([0.05, 0.06, 0.07], 0.2),
    }
    for class_name, (levels, height) in profiles.items():
        block = {"class_name": [class_name] * n_per_class}
        for measure in PER_DATE_MEASURES:
            for date, level in zip(DATES, levels):
                block[f"{measure}_{date}"] = level + rng.normal(0, 0.02, n_per_class)
        block["ndsm"] = height + rng.normal(0, 0.3, n_per_class)
        if class_name == "deadwood":
            block["group_id"] = [f"tree:{i % n_dead_trees}" for i in range(n_per_class)]
            block["tree_id"] = [str(i % n_dead_trees) for i in range(n_per_class)]
        else:
            block["group_id"] = [f"block:{i % 10}" for i in range(n_per_class)]
            block["tree_id"] = [None] * n_per_class
        frames.append(pd.DataFrame(block))
    df = pd.concat(frames, ignore_index=True)
    df["class_code"] = df["class_name"].map({"background": 0, "living": 1, "deadwood": 2})
    return df


def test_grouped_cv_never_shares_a_group_between_folds():
    """The guard the whole validation rests on."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = _table()
    y = df["class_code"].to_numpy()
    groups = df["group_id"].to_numpy()
    X = np.zeros((len(df), 2))
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    for train_idx, test_idx in splitter.split(X, y, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_grouped_cv_returns_probabilities_and_metrics():
    df = _table()
    from deadwood_spectral.features import build_features

    X = build_features(df, DATES).to_numpy()
    proba, metrics = grouped_cv(X, df["class_code"].to_numpy(), df["group_id"].to_numpy(), n_splits=3)
    assert proba.shape == (len(df), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert {"class_name", "precision", "recall", "f1", "support"} <= set(metrics.columns)


def test_grouped_cv_separates_the_synthetic_classes():
    df = _table()
    from deadwood_spectral.features import build_features

    X = build_features(df, DATES).to_numpy()
    _, metrics = grouped_cv(X, df["class_code"].to_numpy(), df["group_id"].to_numpy(), n_splits=3)
    deadwood = metrics.set_index("class_name").loc["deadwood"]
    assert deadwood["recall"] > 0.8


def test_leave_one_tree_out_covers_every_tree():
    df = _table()
    from deadwood_spectral.features import build_features

    X = build_features(df, DATES).to_numpy()
    result = leave_one_tree_out(X, df["class_code"].to_numpy(), df["group_id"].to_numpy())
    assert set(result["tree_id"]) == {f"tree:{i}" for i in range(6)}
    assert result["recall"].between(0.0, 1.0).all()


def test_train_model_returns_model_metrics_and_feature_list():
    result = train_model(_table(), DATES, seed=0)
    assert set(result) >= {"model", "features", "metrics", "loto", "n_features"}
    assert result["features"] == feature_names(DATES)
    assert result["n_features"] == len(result["features"])


def test_rows_with_nan_features_are_dropped_not_zero_filled():
    df = _table()
    df.loc[0, "ndvi_20251115"] = np.nan
    result = train_model(df, DATES)
    assert result["n_samples"] == len(df) - 1


def test_save_and_load_model_round_trip(tmp_path):
    result = train_model(_table(), DATES)
    path = save_model(result["model"], result["features"], tmp_path / "full")
    model, features = load_model(path)
    assert features == result["features"]
    assert model.n_classes_ == 3


def test_make_model_is_deterministic_for_a_seed():
    a, b = make_model(seed=3), make_model(seed=3)
    assert a.random_state == b.random_state == 3


def test_grouped_cv_clamps_splits_to_scarcest_class_group_count():
    """With only 3 deadwood trees, an unclamped 5-split StratifiedGroupKFold
    can leave a fold with zero deadwood *training* examples (the group holding
    all the pixels of a rare class becomes the test fold with nothing left to
    train on). grouped_cv must clamp n_splits so every fold still has some
    deadwood in training."""
    from deadwood_spectral.features import build_features

    df = _table(n_dead_trees=3)
    X = build_features(df, DATES).to_numpy()
    y = df["class_code"].to_numpy()
    groups = df["group_id"].to_numpy()

    # Should not raise, and should produce sane, non-degenerate metrics.
    proba, metrics = grouped_cv(X, y, groups, n_splits=5)
    assert proba.shape == (len(df), 3)
    deadwood = metrics.set_index("class_name").loc["deadwood"]
    assert deadwood["support"] == (y == 2).sum()
    assert deadwood["recall"] > 0.0


def test_train_model_threads_n_estimators_to_the_persisted_model():
    """n_estimators is a config knob (classify.n_estimators). If train_model
    ignores it and always falls back to make_model's own default, changing the
    config silently does nothing — this must fail if that regresses."""
    result = train_model(_table(), DATES, n_estimators=17)
    assert result["model"].n_estimators == 17


def test_train_model_threads_n_estimators_into_grouped_cv_and_loto():
    """The CV numbers must describe the same forest that gets shipped — not a
    different one built with a hardcoded default. Patch make_model in the
    classify module and check every call it receives during train_model
    carries the requested n_estimators."""
    import deadwood_spectral.classify as classify_mod

    seen_n_estimators = []
    real_make_model = classify_mod.make_model

    def spying_make_model(seed=0, n_estimators=400):
        seen_n_estimators.append(n_estimators)
        return real_make_model(seed=seed, n_estimators=n_estimators)

    original = classify_mod.make_model
    classify_mod.make_model = spying_make_model
    try:
        classify_mod.train_model(_table(), DATES, n_estimators=17)
    finally:
        classify_mod.make_model = original

    assert seen_n_estimators, "make_model was never called"
    assert set(seen_n_estimators) == {17}


def test_nan_drop_names_a_group_it_empties_and_reports_per_class(caplog):
    """A bare dropped-row count hides a change of ground truth.

    Real stacks are ~45% NaN with a different footprint per date, so the NaN
    drop can remove every pixel of one soff tree and take the deadwood group
    count from 6 to 5 with nothing in the log but a row count.
    """
    import logging

    df = _table(n_per_class=30)
    doomed = df["group_id"] == "tree:0"
    df.loc[doomed, f"ndvi_{DATES[1]}"] = np.nan
    living = df.index[df["class_name"] == "living"][:5]
    df.loc[living, f"ndvi_{DATES[1]}"] = np.nan

    with caplog.at_level(logging.WARNING, logger="deadwood_spectral.classify"):
        result = train_model(df, DATES, n_estimators=10)

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "tree:0" in messages
    assert "lost ALL their rows" in messages
    assert "deadwood" in messages and "living" in messages
    assert result["n_deadwood_groups"] == 5
    assert "tree:0" not in result["deadwood_groups"]
