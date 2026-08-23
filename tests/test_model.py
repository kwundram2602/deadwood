import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.model import (  # noqa: E402
    DEADWOOD_CODE,
    grouped_cv,
    leave_one_tree_out,
    load_model,
    n_splits_for,
    save_model,
    train,
)
from deadwood_spectral.phenology import FEATURE_NAMES  # noqa: E402


def _toy(n_per_group=20, n_groups_per_class=6, seed=0):
    """Trennbares Spielzeugproblem mit ehrlichen Gruppen: ein Baum = eine Gruppe."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for code, offset in ((0, 0.0), (1, 3.0), (2, 6.0)):
        for g in range(n_groups_per_class):
            values = rng.normal(offset, 0.3, size=(n_per_group, len(FEATURE_NAMES)))
            rows.append(pd.DataFrame(values, columns=list(FEATURE_NAMES)))
            labels.append(
                pd.DataFrame(
                    {
                        "class_code": code,
                        "group_id": f"c{code}_g{g}",
                        "tree_id": f"c{code}_g{g}" if code == DEADWOOD_CODE else pd.NA,
                    },
                    index=range(n_per_group),
                )
            )
    features = pd.concat(rows, ignore_index=True)
    meta = pd.concat(labels, ignore_index=True)
    return features, meta


def test_n_splits_for_never_exceeds_the_scarcest_class_group_count():
    y = np.array([2, 2, 1, 1, 1, 1, 0, 0, 0, 0])
    groups = np.array(["t1", "t2", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"])
    assert n_splits_for(y, groups, requested=5) == 2


def test_n_splits_for_respects_a_smaller_request():
    y = np.array([2] * 6 + [1] * 6)
    groups = np.array([f"g{i}" for i in range(12)])
    assert n_splits_for(y, groups, requested=3) == 3


def test_grouped_cv_returns_one_probability_row_per_sample():
    features, meta = _toy()
    proba, metrics = grouped_cv(
        features.to_numpy(),
        meta["class_code"].to_numpy(),
        meta["group_id"].to_numpy(),
        n_splits=3,
        n_estimators=20,
    )
    assert proba.shape == (len(features), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(metrics["class_name"]) == {"background", "living", "deadwood"}


def test_grouped_cv_separates_a_well_separated_problem():
    features, meta = _toy()
    _, metrics = grouped_cv(
        features.to_numpy(),
        meta["class_code"].to_numpy(),
        meta["group_id"].to_numpy(),
        n_splits=3,
        n_estimators=50,
    )
    recall = metrics.set_index("class_name").loc["deadwood", "recall"]
    assert recall > 0.9


def test_leave_one_tree_out_reports_one_row_per_deadwood_tree():
    features, meta = _toy(n_groups_per_class=4)
    loto = leave_one_tree_out(
        features.to_numpy(),
        meta["class_code"].to_numpy(),
        meta["group_id"].to_numpy(),
        n_estimators=20,
    )
    assert len(loto) == 4
    assert list(loto.columns) == ["tree_id", "n_pixels", "recall"]


def test_train_drops_rows_with_nan_features():
    features, meta = _toy(n_groups_per_class=4)
    features.iloc[0, 0] = np.nan
    result = train(features, meta, n_splits=3, n_estimators=20, permutation_repeats=1)
    assert result["n_samples"] == len(features) - 1


def test_train_reports_the_number_of_deadwood_groups():
    features, meta = _toy(n_groups_per_class=4)
    result = train(features, meta, n_splits=3, n_estimators=20, permutation_repeats=1)
    assert result["n_deadwood_groups"] == 4
    assert list(result["importances"].columns) == ["feature", "importance_mean", "importance_std"]


def test_save_and_load_model_round_trip(tmp_path):
    features, meta = _toy(n_groups_per_class=4)
    result = train(features, meta, n_splits=3, n_estimators=20, permutation_repeats=1)
    save_model(result, tmp_path / "model")
    loaded = load_model(tmp_path / "model")
    assert (tmp_path / "model" / "metrics.json").exists()
    assert (tmp_path / "model" / "importances.csv").exists()
    predicted = loaded.predict(features.iloc[:5].to_numpy())
    assert len(predicted) == 5
