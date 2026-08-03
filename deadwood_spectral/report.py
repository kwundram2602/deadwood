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
    """JM distance and AUC per date and measure, class_a vs. class_b.

    Two AUC columns, deliberately:

    - `auc_raw` is directional — P(a > b), i.e. the probability that a random
      `class_a` pixel scores above a random `class_b` one. Deadwood ndvi sits
      BELOW living ndvi, so a well-separated date shows `auc_raw` near 0.
    - `auc_sep` is the folded magnitude, max(auc_raw, 1 - auc_raw): "how well
      separated", regardless of which class sits on top. `best_date` selects
      on this one.

    The folded value used to be published under the plain name `auc`, which
    reads as P(a > b) to anyone who knows the metric: a reader seeing
    `auc = 0.97` for deadwood-vs-living would conclude deadwood is the
    BRIGHTER class, the exact opposite of the truth. Both are reported now, so
    the direction is recoverable from `summary.csv`.
    """
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
            magnitude_auc = raw_auc if np.isnan(raw_auc) else max(raw_auc, 1.0 - raw_auc)
            records.append(
                {
                    "date": date,
                    "measure": measure,
                    "jm": jeffries_matusita(a, b),
                    "auc_sep": magnitude_auc,
                    "auc_raw": raw_auc,
                    "n_a": int(np.isfinite(a).sum()),
                    "n_b": int(np.isfinite(b).sum()),
                }
            )
    return pd.DataFrame(records)


def seasonal_amplitude(
    df: pd.DataFrame, dates: list[str], measure: str = "ndvi"
) -> pd.Series:
    """max - min of one measure across the given dates, per row. NaN-propagating.

    A row observed on only some of the dates gets NaN, not a partial
    amplitude. This deliberately matches `features.build_features`'
    `<measure>_amplitude`, which is plain numpy max/min over the date matrix
    and so propagates NaN; the two used to disagree (pandas skipna=True here),
    and on real stacks — 45.7% NaN, with a different footprint per date — that
    meant the descriptive plot summarised amplitudes computed over 12 dates
    and over 2 dates in the same box, i.e. a statistic driven by how often a
    pixel happened to be observed rather than by phenology.

    An amplitude over a partial, row-dependent set of dates is not comparable
    across rows, so it is not reported. The cost is that a pixel missing on a
    single date drops out entirely — `amplitude_population` counts exactly how
    many rows that leaves, and `run_report` writes those counts next to the
    plot so the population is never implicit.
    """
    columns = [feature_column(measure, d) for d in dates if feature_column(measure, d) in df]
    if not columns:
        raise ValueError(f"no {measure} columns for dates {dates}")
    values = df[columns].to_numpy(dtype=np.float64)
    return pd.Series(values.max(axis=1) - values.min(axis=1), index=df.index)


def amplitude_population(df: pd.DataFrame, dates: list[str], measures=("ndvi", "ndre")):
    """How many rows per class survive the "observed on every date" rule.

    `seasonal_amplitude` is only defined for rows with a finite value on all
    of `dates`. This reports, per measure and class, how many rows that is out
    of how many exist, so a reader of `seasonal_amplitude.png` can see whether
    a box summarises the whole class or a well-observed corner of it.
    """
    records = []
    for measure in measures:
        columns = [feature_column(measure, d) for d in dates if feature_column(measure, d) in df]
        if not columns:
            continue
        complete = np.isfinite(df[columns].to_numpy(dtype=np.float64)).all(axis=1)
        for class_name, index in df.groupby("class_name").groups.items():
            rows = df.index.get_indexer(index)
            records.append(
                {
                    "measure": measure,
                    "class_name": class_name,
                    "n_dates": len(columns),
                    "n_rows": int(len(rows)),
                    "n_complete": int(complete[rows].sum()),
                }
            )
    out = pd.DataFrame(records)
    if not out.empty:
        out["complete_frac"] = out["n_complete"] / out["n_rows"].where(out["n_rows"] > 0)
    return out


def best_date(sep: pd.DataFrame, measure: str = "ndvi") -> str:
    """Best-separated date for one measure — the single-date baseline.

    Selection is on `auc_sep`, the folded separation magnitude, NOT on the
    directional `auc_raw`: deadwood ndvi sits below living ndvi, so an argmax
    over the directional value would pick the WORST-separated date.

    Ties (e.g. two dates both at perfect separation) break toward the later
    date: sorting by (auc_sep, date) ascending and taking the last row picks
    the max, and among equal values, the chronologically latest date.
    """
    subset = sep[sep["measure"] == measure].dropna(subset=["auc_sep"])
    if subset.empty:
        raise ValueError(f"no finite AUC for measure {measure!r}")
    return str(subset.sort_values(["auc_sep", "date"]).iloc[-1]["date"])


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
    classes = ("deadwood", "living", "background")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    n_dates = 0
    kept = {c: 0 for c in classes}
    total = {c: int((df["class_name"] == c).sum()) for c in classes}
    for measure in ("ndvi", "ndre"):
        try:
            amplitude = seasonal_amplitude(df, dates, measure)
        except ValueError:
            continue
        n_dates = max(
            n_dates, len([d for d in dates if feature_column(measure, d) in df])
        )
        data = [amplitude[df["class_name"] == c].dropna() for c in classes]
        if measure == "ndvi":
            kept = {c: int(len(s)) for c, s in zip(classes, data)}
        positions = np.arange(3) + (0.0 if measure == "ndvi" else 0.35)
        ax.boxplot(data, positions=positions, widths=0.3, tick_labels=None, showfliers=False)
    ax.set_xticks(np.arange(3) + 0.175)
    # The population is on the axis, not just in a sidecar file: the amplitude
    # is only defined for pixels observed on ALL dates, and on real stacks
    # (45.7% NaN, different footprint per date) that can be a small and
    # class-dependent subset. A box over an unstated population is not a
    # statistic a reader can compare.
    ax.set_xticklabels([f"{c}\nn={kept.get(c, 0)}/{total.get(c, 0)}" for c in classes])
    ax.set_ylabel("seasonal amplitude (max - min)")
    ax.set_title(
        "Seasonal amplitude — left box ndvi, right box ndre\n"
        f"pixels observed on all {n_dates} date(s) only (NaN-propagating, "
        "same definition as the classifier feature)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "seasonal_amplitude.png", dpi=140)
    plt.close(fig)
    logger.info(
        "seasonal amplitude computed over pixels observed on all %d date(s): %s",
        n_dates,
        ", ".join(f"{c} {kept.get(c, 0)}/{total.get(c, 0)}" for c in classes),
    )


def _plot_by_group(df, dates, out_dir, column, filename, title) -> None:
    # A plot that never appears must leave a trace: only 7 of the 18 deadwood
    # trees pass the default quality filter, so an empty group here is a
    # plausible outcome, not an impossible one.
    if column not in df.columns:
        logger.warning("skipping %s: no %r column in the sample table", filename, column)
        return
    subset = df[df["class_name"] == "deadwood"].dropna(subset=[column])
    if subset.empty:
        logger.warning(
            "skipping %s: no deadwood rows with a non-null %r value", filename, column
        )
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

    population = amplitude_population(df, dates)
    population.to_csv(out_dir / "amplitude_population.csv", index=False)

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
