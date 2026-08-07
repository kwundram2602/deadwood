"""Train the deadwood classifier and validate it honestly.

One RandomForest, the full feature set (every configured cycle date, per-date
values plus temporal aggregates plus nDSM), trained on every sampled row —
no quality-filter split. `certaintyLP`/`coverage`/`quality_ok` stay in the
sample table as informational columns; they no longer gate what the
classifier trains on.

All validation is grouped by tree. Pixels of one crown are near-duplicates, so
an ungrouped score would look excellent and mean nothing.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from deadwood_spectral.extract import NDSM_REFERENCE_FILE, save_ndsm_reference
from deadwood_spectral.features import (
    build_features,
    feature_names,
    load_feature_names,
    save_feature_names,
)
from deadwood_spectral.sampling import CLASS_CODES

logger = logging.getLogger(__name__)

CODE_TO_NAME = {code: name for name, code in CLASS_CODES.items()}
DEADWOOD_CODE = CLASS_CODES["deadwood"]


def make_model(seed: int = 0, n_estimators: int = 400) -> RandomForestClassifier:
    """Class-balanced forest. Balancing matters even after subsampling."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )


def _metric_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(CODE_TO_NAME)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "class_name": [CODE_TO_NAME[c] for c in labels],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def _safe_n_splits(y: np.ndarray, groups: np.ndarray, n_splits: int) -> int:
    """Clamp n_splits so no class is thinner than the fold count.

    StratifiedGroupKFold does not error when a class has fewer groups than
    n_splits — it silently packs that class's samples into whichever folds it
    can, which can leave a fold with *zero training examples* of that class
    (every group of the class lands in that fold's test split, so the model
    fitted on the remaining folds never sees it). That fold's held-out
    predictions for the class are then close to a coin flip, pulling the
    pooled recall down for reasons that have nothing to do with the features.

    Requiring n_splits <= (groups in the scarcest class) guarantees at least
    one group of every class remains in training for every fold, as long as
    that class has >= 2 groups. A class with only one group cannot be grouped
    cross-validated at all — one fold will always hold out its only group —
    and no clamp can fix that; it is a data limitation, not a code bug.
    """
    n_total_groups = len(np.unique(groups))
    per_class_groups = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    scarcest = min(per_class_groups) if per_class_groups else n_total_groups
    if scarcest < 2:
        logger.warning(
            "a class has only %d group(s) — grouped CV cannot guarantee that "
            "class is present in every fold's training set",
            scarcest,
        )
    return max(2, min(n_splits, n_total_groups, scarcest))


def grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
    n_splits: int = 5,
    n_estimators: int = 400,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold probabilities and per-class metrics, grouped by tree/block."""
    n_splits = _safe_n_splits(y, groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    proba = np.zeros((len(y), len(CODE_TO_NAME)), dtype=np.float64)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = make_model(seed, n_estimators=n_estimators)
        model.fit(X[train_idx], y[train_idx])
        fold_proba = model.predict_proba(X[test_idx])
        # A fold can miss a class entirely; map columns back by class label.
        for column, label in enumerate(model.classes_):
            proba[test_idx, int(label)] = fold_proba[:, column]

    return proba, _metric_frame(y, proba.argmax(axis=1))


def leave_one_tree_out(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0, n_estimators: int = 400
) -> pd.DataFrame:
    """Per-tree deadwood recall when that tree is held out entirely.

    With 18 positives, the spread across trees says more than any mean: one
    unusual tree moves a pooled score far more than it should.
    """
    tree_groups = sorted({g for g, label in zip(groups, y) if label == DEADWOOD_CODE})
    records = []
    for tree in tree_groups:
        held_out = groups == tree
        model = make_model(seed, n_estimators=n_estimators)
        model.fit(X[~held_out], y[~held_out])
        predicted = model.predict(X[held_out])
        records.append(
            {
                "tree_id": tree,
                "n_pixels": int(held_out.sum()),
                "recall": float(np.mean(predicted == DEADWOOD_CODE)),
            }
        )
    return pd.DataFrame(records)


def train_model(
    table: pd.DataFrame,
    dates: list[str],
    seed: int = 0,
    n_splits: int = 5,
    n_estimators: int = 400,
) -> dict:
    """Build features, run grouped CV and leave-one-tree-out, fit the final model."""
    matrix = build_features(table, dates, per_date=True, temporal=True, static=True)
    features = feature_names(dates, per_date=True, temporal=True, static=True)

    finite = matrix.notna().all(axis=1).to_numpy()
    all_y = table["class_code"].to_numpy()
    all_groups = table["group_id"].to_numpy()
    dropped = int((~finite).sum())
    if dropped:
        # A bare count hides the thing that actually matters. Real stacks are
        # ~45% NaN with a different footprint per date, so a 12-date cycle
        # can drop every pixel of one soff tree — a change of ground truth,
        # reported as a number of rows.
        per_class = {
            CODE_TO_NAME.get(int(c), str(c)): int(((all_y == c) & ~finite).sum())
            for c in np.unique(all_y)
        }
        kept_groups = set(all_groups[finite].tolist())
        touched = set(np.unique(all_groups[~finite]).tolist())
        emptied = sorted(g for g in touched if g not in kept_groups)
        logger.warning(
            "dropping %d row(s) with NaN features — per class %s; "
            "%d group(s) lost rows, %d group(s) lost ALL their rows",
            dropped, per_class, len(touched), len(emptied),
        )
        if emptied:
            logger.warning(
                "group(s) removed entirely by the NaN drop: %s — the "
                "population this model is fitted and scored on is no longer "
                "the full label set",
                emptied,
            )

    X = matrix.to_numpy(dtype=np.float64)[finite]
    y = all_y[finite]
    groups = all_groups[finite]
    deadwood_groups = sorted({g for g, label in zip(groups, y) if label == DEADWOOD_CODE})

    proba, metrics = grouped_cv(
        X, y, groups, seed=seed, n_splits=n_splits, n_estimators=n_estimators
    )
    loto = leave_one_tree_out(X, y, groups, seed=seed, n_estimators=n_estimators)

    model = make_model(seed, n_estimators=n_estimators)
    model.fit(X, y)

    logger.info(
        "%d features, %d samples, %d deadwood group(s), deadwood recall %.3f "
        "(grouped CV), per-tree recall median %.3f [%.3f-%.3f]",
        len(features), len(y), len(deadwood_groups),
        float(metrics.set_index("class_name").loc["deadwood", "recall"]),
        float(loto["recall"].median()) if len(loto) else float("nan"),
        float(loto["recall"].min()) if len(loto) else float("nan"),
        float(loto["recall"].max()) if len(loto) else float("nan"),
    )
    return {
        "model": model,
        "features": features,
        "n_features": len(features),
        "n_samples": int(len(y)),
        "n_groups": int(len(set(groups.tolist()))),
        "n_deadwood_groups": int(len(deadwood_groups)),
        "deadwood_groups": deadwood_groups,
        "metrics": metrics,
        "loto": loto,
        "oof_proba": proba,
        "y": y,
    }


def save_model(
    model, feature_list: list[str], out_dir: str | Path, ndsm_reference: dict | None = None
) -> Path:
    """Persist the model, its feature order, and the nDSM it was trained on.

    The nDSM identity travels with the model because nothing else can catch a
    mismatch: both nDSM variants on disk sit on the reference grid, so
    inference with the wrong one is silently wrong rather than an error.
    `deadwood_spectral.extract.assert_same_ndsm` checks it in apply.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    save_feature_names(feature_list, out_dir / "feature_names.json")
    if ndsm_reference is not None:
        save_ndsm_reference(ndsm_reference, out_dir / NDSM_REFERENCE_FILE)
    return out_dir


def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    features = load_feature_names(model_dir / "feature_names.json")
    return model, features
