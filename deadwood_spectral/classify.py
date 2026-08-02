"""Train the deadwood classifier and validate it honestly.

Three feature variants run side by side: the full feature set, the temporal
aggregates alone, and a single-date baseline. The baseline is the bar the time
series has to clear — if one flight does as well as twelve, that is the result
worth reporting, not a disappointment.

Two label sets run alongside them. The default field-label quality filter
(`certaintyLP >= 50` and `coverage == 'nc'`) is far more costly than expected
on the real data: it leaves only 7 of the 18 deadwood trees. `apply_label_set`
lets the caller compare `filtered` against `all` so it stays visible whether
the 11 uncertain trees carry signal or noise, rather than quietly assuming the
filter is free.

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

from deadwood_spectral.features import (
    build_features,
    feature_names,
    load_feature_names,
    save_feature_names,
)
from deadwood_spectral.sampling import CLASS_CODES

logger = logging.getLogger(__name__)

VARIANTS: tuple[str, ...] = ("full", "reduced", "baseline")
LABEL_SETS: tuple[str, ...] = ("filtered", "all")
CODE_TO_NAME = {code: name for name, code in CLASS_CODES.items()}
DEADWOOD_CODE = CLASS_CODES["deadwood"]


def variant_spec(
    name: str, dates: list[str], baseline_date: str
) -> tuple[list[str], dict[str, bool]]:
    """The dates and feature-group switches a variant uses."""
    if name == "full":
        return list(dates), {"per_date": True, "temporal": True, "static": True}
    if name == "reduced":
        return list(dates), {"per_date": False, "temporal": True, "static": True}
    if name == "baseline":
        if baseline_date not in dates:
            raise ValueError(f"baseline_date {baseline_date!r} is not in the cycle {dates}")
        # Temporal aggregates over a single date are degenerate by definition.
        return [baseline_date], {"per_date": True, "temporal": False, "static": True}
    raise ValueError(f"unknown variant {name!r}; expected one of {VARIANTS}")


def apply_label_set(table: pd.DataFrame, label_set: str) -> pd.DataFrame:
    """Select the rows a label set is allowed to train/validate on.

    `all` keeps every row. `filtered` keeps only rows with `quality_ok`. The
    negative classes (living, background) are `quality_ok=True` by
    construction — they never came through the field certainty/coverage
    filter — so a well-formed table's quality filter should only ever drop
    deadwood rows. That is checked here rather than assumed: if it does not
    hold, the label sets would silently compare different negative-class
    populations too, which would confound the comparison.
    """
    if label_set == "all":
        return table
    if label_set != "filtered":
        raise ValueError(f"unknown label_set {label_set!r}; expected one of {LABEL_SETS}")

    mask = table["quality_ok"].astype(bool)
    dropped = table.loc[~mask]
    if not dropped.empty and not (dropped["class_code"] == DEADWOOD_CODE).all():
        offenders = sorted(dropped.loc[dropped["class_code"] != DEADWOOD_CODE, "class_code"].unique())
        raise ValueError(
            "quality_ok filter dropped non-deadwood rows (class codes "
            f"{offenders}) — the negative classes are expected to be quality_ok "
            "by construction; check apply_quality_filter and the sampling pipeline"
        )
    return table.loc[mask].reset_index(drop=True)


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
) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold probabilities and per-class metrics, grouped by tree/block."""
    n_splits = _safe_n_splits(y, groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    proba = np.zeros((len(y), len(CODE_TO_NAME)), dtype=np.float64)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = make_model(seed)
        model.fit(X[train_idx], y[train_idx])
        fold_proba = model.predict_proba(X[test_idx])
        # A fold can miss a class entirely; map columns back by class label.
        for column, label in enumerate(model.classes_):
            proba[test_idx, int(label)] = fold_proba[:, column]

    return proba, _metric_frame(y, proba.argmax(axis=1))


def leave_one_tree_out(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0
) -> pd.DataFrame:
    """Per-tree deadwood recall when that tree is held out entirely.

    With 18 positives, the spread across trees says more than any mean: one
    unusual tree moves a pooled score far more than it should.
    """
    tree_groups = sorted({g for g, label in zip(groups, y) if label == DEADWOOD_CODE})
    records = []
    for tree in tree_groups:
        held_out = groups == tree
        model = make_model(seed)
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


def train_variant(
    table: pd.DataFrame,
    dates: list[str],
    variant: str,
    baseline_date: str,
    seed: int = 0,
    n_splits: int = 5,
) -> dict:
    """Build features, run grouped CV and leave-one-tree-out, fit the final model."""
    variant_dates, switches = variant_spec(variant, dates, baseline_date)
    matrix = build_features(table, variant_dates, **switches)
    features = feature_names(variant_dates, **switches)

    finite = matrix.notna().all(axis=1).to_numpy()
    dropped = int((~finite).sum())
    if dropped:
        logger.warning("%s: dropping %d row(s) with NaN features", variant, dropped)

    X = matrix.to_numpy(dtype=np.float64)[finite]
    y = table["class_code"].to_numpy()[finite]
    groups = table["group_id"].to_numpy()[finite]

    proba, metrics = grouped_cv(X, y, groups, seed=seed, n_splits=n_splits)
    loto = leave_one_tree_out(X, y, groups, seed=seed)

    model = make_model(seed)
    model.fit(X, y)

    logger.info(
        "%s: %d features, %d samples, deadwood recall %.3f (grouped CV), "
        "per-tree recall median %.3f [%.3f-%.3f]",
        variant, len(features), len(y),
        float(metrics.set_index("class_name").loc["deadwood", "recall"]),
        float(loto["recall"].median()) if len(loto) else float("nan"),
        float(loto["recall"].min()) if len(loto) else float("nan"),
        float(loto["recall"].max()) if len(loto) else float("nan"),
    )
    return {
        "variant": variant,
        "dates": variant_dates,
        "model": model,
        "features": features,
        "n_features": len(features),
        "n_samples": int(len(y)),
        "metrics": metrics,
        "loto": loto,
        "oof_proba": proba,
        "y": y,
    }


def save_model(model, feature_list: list[str], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    save_feature_names(feature_list, out_dir / "feature_names.json")
    return out_dir


def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    features = load_feature_names(model_dir / "feature_names.json")
    return model, features
