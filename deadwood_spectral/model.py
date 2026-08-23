"""Training, Validierung und Persistenz des Totholz-Klassifikators.

Eine RandomForest über drei Klassen. Jede Validierung ist nach Baum bzw. Block
gruppiert: Pixel einer Krone sind Beinahe-Duplikate, ein ungruppierter Score
sähe hervorragend aus und bedeutete nichts. Die effektive Stichprobe sind die
Totholz-Bäume, nicht die Pixel.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from deadwood_spectral.phenology import FEATURE_NAMES
from deadwood_spectral.samples import CLASS_CODES

logger = logging.getLogger(__name__)

CODE_TO_NAME: dict[int, str] = {code: name for name, code in CLASS_CODES.items()}
DEADWOOD_CODE: int = CLASS_CODES["deadwood"]


def make_model(seed: int = 0, n_estimators: int = 400) -> RandomForestClassifier:
    """Klassenbalancierter Forest — Balancierung zählt auch nach dem Subsampling."""
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


def n_splits_for(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    """Fold-Zahl, die keine Klasse dünner macht als die Fold-Zahl selbst.

    StratifiedGroupKFold meldet keinen Fehler, wenn eine Klasse weniger Gruppen
    hat als Folds — sie verteilt deren Samples still, und ein Fold kann ohne ein
    einziges Trainingsbeispiel dieser Klasse dastehen. Dessen Vorhersagen sind
    dann Münzwurf und ziehen die gepoolte Recall herunter, aus Gründen, die
    nichts mit den Features zu tun haben.
    """
    per_class = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    scarcest = min(per_class) if per_class else len(np.unique(groups))
    if scarcest < 2:
        logger.warning(
            "a class has only %d group(s) — grouped CV cannot keep it in every fold's training set",
            scarcest,
        )
    return max(2, min(int(requested), scarcest))


def grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
    n_splits: int = 5,
    n_estimators: int = 400,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold-Wahrscheinlichkeiten und Metriken aus den gepoolten Vorhersagen."""
    n_splits = n_splits_for(y, groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    proba = np.zeros((len(y), len(CODE_TO_NAME)), dtype=np.float64)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        logger.info(
            "grouped_cv: fold %d/%d (%d train, %d test)",
            fold,
            n_splits,
            len(train_idx),
            len(test_idx),
        )
        fitted = make_model(seed, n_estimators=n_estimators)
        fitted.fit(X[train_idx], y[train_idx])
        fold_proba = fitted.predict_proba(X[test_idx])
        # Ein Fold kann eine Klasse ganz verfehlen; Spalten über die Labels
        # zurückordnen statt über die Position.
        for column, label in enumerate(fitted.classes_):
            proba[test_idx, int(label)] = fold_proba[:, column]

    return proba, _metric_frame(y, proba.argmax(axis=1))


def leave_one_tree_out(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
    n_estimators: int = 400,
) -> pd.DataFrame:
    """Totholz-Recall je Baum, wenn genau dieser Baum ganz aus dem Training fällt.

    Bei 18 Positiven sagt die Streuung über die Bäume mehr als jeder gepoolte
    Wert: ein ungewöhnlicher Baum bewegt einen gepoolten Score weit stärker,
    als er sollte.
    """
    trees = sorted({g for g, label in zip(groups, y) if label == DEADWOOD_CODE})
    records = []
    for i, tree in enumerate(trees, start=1):
        held_out = groups == tree
        logger.info(
            "leave_one_tree_out: tree %d/%d (%s, %d px)",
            i,
            len(trees),
            tree,
            int(held_out.sum()),
        )
        fitted = make_model(seed, n_estimators=n_estimators)
        fitted.fit(X[~held_out], y[~held_out])
        predicted = fitted.predict(X[held_out])
        records.append(
            {
                "tree_id": tree,
                "n_pixels": int(held_out.sum()),
                "recall": float(np.mean(predicted == DEADWOOD_CODE)),
            }
        )
    return pd.DataFrame(records, columns=["tree_id", "n_pixels", "recall"])


def train(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    seed: int = 0,
    n_splits: int = 5,
    n_estimators: int = 400,
    permutation_repeats: int = 5,
) -> dict:
    """Gruppierte CV, Leave-one-tree-out, Permutation-Importance, finales Modell."""
    matrix = features[list(FEATURE_NAMES)]
    finite = matrix.notna().all(axis=1).to_numpy()
    all_y = labels["class_code"].to_numpy()
    all_groups = labels["group_id"].to_numpy()

    dropped = int((~finite).sum())
    if dropped:
        # Eine blosse Zahl verbirgt, worauf es ankommt: reale Stacks sind je
        # Datum unterschiedlich beschnitten, ein Fenster kann daher sämtliche
        # Pixel eines soff-Baums verlieren — das ist eine Änderung des Ground
        # Truth, gemeldet als Zeilenzahl.
        per_class = {
            CODE_TO_NAME.get(int(c), str(c)): int(((all_y == c) & ~finite).sum())
            for c in np.unique(all_y)
        }
        kept = set(all_groups[finite].tolist())
        emptied = sorted(g for g in np.unique(all_groups[~finite]) if g not in kept)
        logger.warning("dropping %d row(s) with NaN features — per class %s", dropped, per_class)
        if emptied:
            logger.warning(
                "group(s) removed ENTIRELY by the NaN drop: %s — the population "
                "this model is fitted and scored on is no longer the full label set",
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

    logger.info("fitting final model on all %d sample(s)", len(y))
    fitted = make_model(seed, n_estimators=n_estimators)
    fitted.fit(X, y)

    permutation = permutation_importance(
        fitted, X, y, n_repeats=int(permutation_repeats), random_state=seed, n_jobs=-1
    )
    importances = pd.DataFrame(
        {
            "feature": list(FEATURE_NAMES),
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    logger.info(
        "%d features, %d samples, %d deadwood group(s); deadwood recall %.3f "
        "(grouped CV), per-tree recall median %.3f [%.3f-%.3f]",
        len(FEATURE_NAMES),
        len(y),
        len(deadwood_groups),
        float(metrics.set_index("class_name").loc["deadwood", "recall"]),
        float(loto["recall"].median()) if len(loto) else float("nan"),
        float(loto["recall"].min()) if len(loto) else float("nan"),
        float(loto["recall"].max()) if len(loto) else float("nan"),
    )
    return {
        "model": fitted,
        "metrics": metrics,
        "loto": loto,
        "importances": importances,
        "oof_proba": proba,
        "y": y,
        "n_samples": int(len(y)),
        "n_deadwood_groups": int(len(deadwood_groups)),
    }


def save_model(result: dict, model_dir: str | Path) -> Path:
    """Modell und Metriken ablegen.

    Es wird bewusst KEINE Feature-Liste mitgeschrieben: die Spaltenmenge ist
    `phenology.FEATURE_NAMES`, eine Konstante im Code, und hängt nicht mehr an
    der Config. Ebenso entfällt die nDSM-Signatur — Training und Inferenz lesen
    denselben `paths.ndsm` aus derselben Config.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], model_dir / "model.joblib")
    result["importances"].to_csv(model_dir / "importances.csv", index=False)
    result["loto"].to_csv(model_dir / "leave_one_tree_out.csv", index=False)
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "n_samples": result["n_samples"],
                "n_deadwood_groups": result["n_deadwood_groups"],
                "per_class": result["metrics"].to_dict(orient="records"),
                "per_tree_recall_median": (
                    float(result["loto"]["recall"].median()) if len(result["loto"]) else None
                ),
            },
            indent=2,
        )
    )
    return model_dir


def load_model(model_dir: str | Path) -> RandomForestClassifier:
    return joblib.load(Path(model_dir) / "model.joblib")
