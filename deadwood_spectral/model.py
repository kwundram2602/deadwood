"""Training, Validierung und Persistenz des Totholz-Klassifikators.

Eine RandomForest über drei Klassen. Jede Validierung ist nach Baum bzw. Block
gruppiert: Pixel einer Krone sind Beinahe-Duplikate, ein ungruppierter Score
sähe hervorragend aus und bedeutete nichts. Die effektive Stichprobe sind die
Totholz-Bäume, nicht die Pixel.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape
from skimage.measure import label as cc_label
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from deadwood_spectral.grid import ReferenceGrid
from deadwood_spectral.phenology import FEATURE_NAMES, pixel_features
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


def predict_crown_pixels(
    fitted: RandomForestClassifier,
    stack_paths_: Sequence[Path],
    grid: ReferenceGrid,
    crown: np.ndarray,
    ndsm_path: str | Path | None = None,
    chunk_rows: int = 512,
    min_valid_dates: int = 4,
    pixel_batch: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Klassenwahrscheinlichkeiten für jedes Pixel der Kronen-Prediction.

    Vorhergesagt wird nur innerhalb der Kronen — das ist der Grund, drei Klassen
    zu führen: eine Krone, die überwiegend als `background` gelesen wird, ist
    eine falsch-positive Krone des Torch-Modells und wird als solche ausgewiesen.

    Die Pixel werden in Stapeln von `pixel_batch` verarbeitet: eine ganze
    5-cm-Szene hat einige Millionen Kronenpixel, und deren Zeitreihen über alle
    Aufnahmen passen sonst nicht in den Speicher.
    """
    rows, cols = np.nonzero(crown)
    proba = np.full((rows.size, len(CODE_TO_NAME)), np.nan, dtype=np.float64)
    if rows.size == 0:
        logger.warning("crown mask is empty — nothing to predict")
        return rows, cols, proba

    for start in range(0, rows.size, pixel_batch):
        stop = min(start + pixel_batch, rows.size)
        logger.info("predicting pixels %d-%d of %d", start, stop, rows.size)
        features = pixel_features(
            stack_paths_,
            grid,
            rows[start:stop],
            cols[start:stop],
            ndsm_path=ndsm_path,
            chunk_rows=chunk_rows,
            min_valid_dates=min_valid_dates,
        )
        matrix = features[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        usable = np.isfinite(matrix).all(axis=1)
        if not usable.any():
            continue
        batch = fitted.predict_proba(matrix[usable])
        target = np.full((int(usable.sum()), len(CODE_TO_NAME)), 0.0)
        for column, label in enumerate(fitted.classes_):
            target[:, int(label)] = batch[:, column]
        proba[np.arange(start, stop)[usable]] = target

    unusable = int(np.isnan(proba[:, 0]).sum())
    if unusable:
        logger.warning(
            "%d/%d crown pixel(s) had too few valid dates and stay unevaluated",
            unusable,
            rows.size,
        )
    return rows, cols, proba


def write_probability_raster(
    rows: np.ndarray,
    cols: np.ndarray,
    proba: np.ndarray,
    grid: ReferenceGrid,
    path: str | Path,
) -> Path:
    """p(deadwood) über die Kronenpixel; ausserhalb NaN.

    Das ist die Ebene, an der die Schwelle in QGIS tatsächlich eingestellt wird.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raster = np.full(grid.shape, np.nan, dtype=np.float32)
    raster[rows, cols] = proba[:, DEADWOOD_CODE].astype(np.float32)
    profile = dict(
        driver="GTiff",
        dtype="float32",
        height=grid.height,
        width=grid.width,
        count=1,
        crs=grid.crs,
        transform=grid.transform,
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, "p_deadwood")
    return path


CROWN_COLUMNS = [
    "crown_id",
    "n_px",
    "area_m2",
    "dead_frac",
    "living_frac",
    "background_frac",
    "p_dead_mean",
    "p_dead_median",
    "mean_height_m",
    "label",
    "geometry",
]


def aggregate_crowns(
    crown: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    proba: np.ndarray,
    grid: ReferenceGrid,
    ndsm: np.ndarray | None = None,
    dead_frac_threshold: float = 0.5,
) -> gpd.GeoDataFrame:
    """Zusammenhangskomponenten der Kronen-Prediction -> ein Datensatz je Krone.

    Zusammenhangskomponenten statt Watershed: eine bewusste Vereinfachung.
    Sich berührende Kronen verschmelzen zu einem Objekt, was ehrlich ist
    gegenüber dem, was 18 Trainingsbäume tragen.
    """
    labels = cc_label(crown, connectivity=2)
    pixel_area = abs(grid.transform.a) * abs(grid.transform.e)
    per_pixel_label = labels[rows, cols]
    usable = np.isfinite(proba).all(axis=1)
    predicted = np.full(rows.size, -1, dtype=np.int64)
    predicted[usable] = proba[usable].argmax(axis=1)

    records = []
    for crown_id in range(1, int(labels.max()) + 1):
        member = per_pixel_label == crown_id
        n_px = int(member.sum())
        if n_px == 0:
            continue
        evaluated = member & usable
        n_evaluated = int(evaluated.sum())

        record = {
            "crown_id": crown_id,
            "n_px": n_px,
            "area_m2": float(n_px) * pixel_area,
            "geometry": _component_polygon(labels, crown_id, grid),
        }
        if n_evaluated == 0:
            record.update(
                dead_frac=float("nan"),
                living_frac=float("nan"),
                background_frac=float("nan"),
                p_dead_mean=float("nan"),
                p_dead_median=float("nan"),
                label="unevaluated",
            )
        else:
            classes = predicted[evaluated]
            fractions = {
                name: float(np.mean(classes == code)) for code, name in CODE_TO_NAME.items()
            }
            dead_probability = proba[evaluated, DEADWOOD_CODE]
            record.update(
                dead_frac=fractions["deadwood"],
                living_frac=fractions["living"],
                background_frac=fractions["background"],
                p_dead_mean=float(dead_probability.mean()),
                p_dead_median=float(np.median(dead_probability)),
                label=_crown_label(fractions, dead_frac_threshold),
            )

        if ndsm is None:
            record["mean_height_m"] = float("nan")
        else:
            heights = ndsm[rows[member], cols[member]]
            heights = heights[np.isfinite(heights)]
            record["mean_height_m"] = float(heights.mean()) if heights.size else float("nan")
        records.append(record)

    logger.info(
        "aggregated %d crown(s): %s",
        len(records),
        dict(pd.Series([r["label"] for r in records]).value_counts()) if records else {},
    )
    if not records:
        return gpd.GeoDataFrame(
            {c: pd.Series(dtype="object") for c in CROWN_COLUMNS if c != "geometry"},
            geometry=gpd.GeoSeries([], crs=grid.crs),
            crs=grid.crs,
        )[CROWN_COLUMNS]
    return gpd.GeoDataFrame(records, geometry="geometry", crs=grid.crs)[CROWN_COLUMNS]


def _crown_label(fractions: dict[str, float], dead_frac_threshold: float) -> str:
    """`rejected` schlägt `deadwood`: eine überwiegend als Hintergrund gelesene
    Krone ist keine Krone, und ihr Totholz-Anteil ist dann bedeutungslos."""
    if fractions["background"] >= 0.5:
        return "rejected"
    if fractions["deadwood"] >= dead_frac_threshold:
        return "deadwood"
    return "living"


def _component_polygon(labels: np.ndarray, crown_id: int, grid: ReferenceGrid):
    """Polygon einer Komponente, gerechnet in ihrer Bounding Box.

    Eine szenenweite `labels == crown_id`-Maske je Komponente kostet auf dem
    realen Gitter ~59 ms; über einige Tausend Kronen ist das Minuten für nichts.
    """
    from rasterio.transform import Affine
    from scipy import ndimage

    boxes = ndimage.find_objects(labels, max_label=crown_id)
    box = boxes[crown_id - 1]
    mask = labels[box] == crown_id
    box_transform = grid.transform * Affine.translation(box[1].start, box[0].start)
    polygons = [
        shapely_shape(geom)
        for geom, _ in rio_shapes(
            mask.astype(np.uint8), mask=mask, transform=box_transform, connectivity=8
        )
    ]
    return polygons[0] if len(polygons) == 1 else max(polygons, key=lambda p: p.area)
