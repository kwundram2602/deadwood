"""Diagnostische Plots. Dünne Wrapper — die Aussage steckt in den Daten."""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # kein Display auf dem HPC
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from deadwood_spectral.model import CODE_TO_NAME, DEADWOOD_CODE  # noqa: E402

logger = logging.getLogger(__name__)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def plot_phenology(
    series: dict[str, np.ndarray],
    class_codes: np.ndarray,
    dates: Sequence[str],
    path: str | Path,
    measure: str = "ndvi",
) -> Path:
    """Median-Verlauf je Klasse — die Abbildung, die die Projekthypothese zeigt.

    Trägt Totholz tatsächlich keinen saisonalen Schwung, liegt seine Kurve flach,
    während die lebenden Kronen ausschlagen.
    """
    values = np.asarray(series[measure], dtype=np.float64)
    x = np.arange(len(dates))
    fig, ax = plt.subplots(figsize=(9, 4))
    for code, name in sorted(CODE_TO_NAME.items()):
        member = class_codes == code
        if not member.any():
            continue
        median = np.nanmedian(values[member], axis=0)
        ax.plot(x, median, marker="o", label=f"{name} (n={int(member.sum())})")
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"median {measure}")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_importances(importances: pd.DataFrame, path: str | Path, top_n: int = 20) -> Path:
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(top) + 1))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("permutation importance")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_precision_recall(y: np.ndarray, oof_proba: np.ndarray, path: str | Path) -> Path:
    """PR-Kurve für die Totholz-Klasse aus den Out-of-fold-Wahrscheinlichkeiten."""
    precision, recall, _ = precision_recall_curve(
        (y == DEADWOOD_CODE).astype(int), oof_proba[:, DEADWOOD_CODE]
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("deadwood, grouped out-of-fold")
    ax.grid(alpha=0.3)
    return _save(fig, path)
