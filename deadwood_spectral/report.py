"""Descriptive overview of the three classes across the full time series.

No fixed cycle here: every available date is one point on the x-axis. The
separability heatmap answers "which index, on which date" directly, and the
amplitude plot tests the project's central hypothesis — living deciduous trees
swing with the season, deadwood does not.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from deadwood_spectral.extract import feature_column  # noqa: E402
from deadwood_spectral.indices import BAND_NAMES, INDEX_NAMES  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MEASURES = tuple(INDEX_NAMES) + tuple(BAND_NAMES)
MIN_FINITE = 2


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def jeffries_matusita(a: np.ndarray, b: np.ndarray) -> float:
    """Univariate JM distance in [0, 2]; 2 means fully separable.

    Built on the Gaussian Bhattacharyya distance. Bounded, unlike a raw mean
    difference, so values are comparable across bands with different scales.
    """
    a, b = _finite(a), _finite(b)
    if a.size < MIN_FINITE or b.size < MIN_FINITE:
        return float("nan")
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    if v1 <= 0 and v2 <= 0:
        return 0.0 if m1 == m2 else 2.0
    pooled = (v1 + v2) / 2.0
    bhattacharyya = (m1 - m2) ** 2 / (8.0 * pooled)
    if v1 > 0 and v2 > 0:
        bhattacharyya += 0.5 * np.log(pooled / np.sqrt(v1 * v2))
    return float(2.0 * (1.0 - np.exp(-bhattacharyya)))


def class_auc(a: np.ndarray, b: np.ndarray) -> float:
    """AUC of `a` scoring above `b`. 0.5 means indistinguishable."""
    a, b = _finite(a), _finite(b)
    if a.size < 1 or b.size < 1:
        return float("nan")
    y = np.concatenate([np.ones(a.size), np.zeros(b.size)])
    return float(roc_auc_score(y, np.concatenate([a, b])))


def separability_table(
    df: pd.DataFrame,
    dates: list[str],
    measures: list[str] | None = None,
    class_a: str = "deadwood",
    class_b: str = "living",
) -> pd.DataFrame:
    """JM distance and AUC per date and measure, class_a vs. class_b."""
    measures = list(measures) if measures is not None else list(DEFAULT_MEASURES)
    a_rows = df["class_name"] == class_a
    b_rows = df["class_name"] == class_b

    records = []
    for measure in measures:
        for date in dates:
            column = feature_column(measure, date)
            if column not in df.columns:
                continue
            a = df.loc[a_rows, column].to_numpy()
            b = df.loc[b_rows, column].to_numpy()
            raw_auc = class_auc(a, b)
            # class_auc is directional (a scoring above b): whichever class happens to
            # have the lower mean on a given date would otherwise report auc < 0.5 even
            # when separation is perfect. Report the magnitude, like jm, so "highest
            # auc" means "best separated" regardless of which class sits on top.
            magnitude_auc = raw_auc if np.isnan(raw_auc) else max(raw_auc, 1.0 - raw_auc)
            records.append(
                {
                    "date": date,
                    "measure": measure,
                    "jm": jeffries_matusita(a, b),
                    "auc": magnitude_auc,
                    "n_a": int(np.isfinite(a).sum()),
                    "n_b": int(np.isfinite(b).sum()),
                }
            )
    return pd.DataFrame(records)


def seasonal_amplitude(
    df: pd.DataFrame, dates: list[str], measure: str = "ndvi"
) -> pd.Series:
    """max - min of one measure across the given dates, per row."""
    columns = [feature_column(measure, d) for d in dates if feature_column(measure, d) in df]
    if not columns:
        raise ValueError(f"no {measure} columns for dates {dates}")
    values = df[columns]
    return values.max(axis=1) - values.min(axis=1)


def best_date(sep: pd.DataFrame, measure: str = "ndvi") -> str:
    """Date with the highest AUC for one measure — the single-date baseline.

    Ties (e.g. two dates both at perfect separation) break toward the later
    date: sorting by (auc, date) ascending and taking the last row picks the
    max auc, and among equal auc values, the chronologically latest date.
    """
    subset = sep[sep["measure"] == measure].dropna(subset=["auc"])
    if subset.empty:
        raise ValueError(f"no finite AUC for measure {measure!r}")
    return str(subset.sort_values(["auc", "date"]).iloc[-1]["date"])


def _plot_trajectories(df, dates, measure, out_dir) -> None:
    columns = [feature_column(measure, d) for d in dates if feature_column(measure, d) in df]
    if not columns:
        return
    x = np.arange(len(columns))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for class_name, group in df.groupby("class_name"):
        values = group[columns]
        median = values.median()
        ax.plot(x, median, marker="o", label=class_name)
        ax.fill_between(x, values.quantile(0.25), values.quantile(0.75), alpha=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels([c.rsplit("_", 1)[1] for c in columns], rotation=60, ha="right")
    ax.set_ylabel(measure)
    ax.set_title(f"{measure} trajectory by class (median, IQR band)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"trajectory_{measure}.png", dpi=140)
    plt.close(fig)


def _plot_separability_heatmap(sep, out_dir) -> None:
    pivot = sep.pivot(index="measure", columns="date", values="jm")
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * pivot.shape[1], 0.4 * pivot.shape[0] + 2))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=2, cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Jeffries-Matusita: deadwood vs. living")
    fig.colorbar(image, ax=ax, label="JM (0 = identical, 2 = separable)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "separability_jm.png", dpi=140)
    plt.close(fig)


def _plot_amplitude(df, dates, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for measure in ("ndvi", "ndre"):
        try:
            amplitude = seasonal_amplitude(df, dates, measure)
        except ValueError:
            continue
        data = [amplitude[df["class_name"] == c].dropna() for c in ("deadwood", "living", "background")]
        positions = np.arange(3) + (0.0 if measure == "ndvi" else 0.35)
        ax.boxplot(data, positions=positions, widths=0.3, tick_labels=None, showfliers=False)
    ax.set_xticks(np.arange(3) + 0.175)
    ax.set_xticklabels(["deadwood", "living", "background"])
    ax.set_ylabel("seasonal amplitude (max - min)")
    ax.set_title("Seasonal amplitude — left box ndvi, right box ndre")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "seasonal_amplitude.png", dpi=140)
    plt.close(fig)


def _plot_by_group(df, dates, out_dir, column, filename, title) -> None:
    if column not in df.columns:
        return
    subset = df[df["class_name"] == "deadwood"].dropna(subset=[column])
    if subset.empty:
        return
    columns = [feature_column("ndvi", d) for d in dates if feature_column("ndvi", d) in df]
    x = np.arange(len(columns))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, group in subset.groupby(column):
        ax.plot(x, group[columns].median(), marker="o", label=f"{column}={key}")
    ax.set_xticks(x)
    ax.set_xticklabels([c.rsplit("_", 1)[1] for c in columns], rotation=60, ha="right")
    ax.set_ylabel("ndvi")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / filename, dpi=140)
    plt.close(fig)


def run_report(df: pd.DataFrame, dates: list[str], out_dir: str | Path) -> Path:
    """Write every plot and summary.csv; return the output directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = separability_table(df, dates)
    sep.to_csv(out_dir / "summary.csv", index=False)

    coverage = (
        df.melt(id_vars=["class_name"], value_vars=[feature_column("ndvi", d) for d in dates
                                                    if feature_column("ndvi", d) in df])
        .assign(date=lambda t: t["variable"].str.rsplit("_", n=1).str[1])
        .groupby(["date", "class_name"])["value"]
        .apply(lambda s: int(np.isfinite(s).sum()))
        .unstack(fill_value=0)
    )
    coverage.to_csv(out_dir / "coverage.csv")

    for measure in ("ndvi", "ndre", "NIR", "brightness"):
        _plot_trajectories(df, dates, measure, out_dir)
    if not sep.empty:
        _plot_separability_heatmap(sep, out_dir)
    _plot_amplitude(df, dates, out_dir)
    _plot_by_group(df, dates, out_dir, "species", "deadwood_by_species.png",
                   "Deadwood ndvi by species (median)")
    _plot_by_group(df, dates, out_dir, "quality_ok", "deadwood_by_quality.png",
                   "Deadwood ndvi by label-quality group (median)")

    logger.info("report written to %s", out_dir)
    return out_dir
