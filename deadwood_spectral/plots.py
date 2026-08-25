"""Diagnostic plots. Thin wrappers — the statement is in the data.

Both figures answer the project hypothesis directly: if deadwood really carries
no seasonal swing, its curve lies flat while the living crowns oscillate, and
its band signature stays put between the seasons while theirs moves.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the HPC
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_COLORS = {"deadwood": "#c0392b", "living": "#27ae60", "background": "#7f8c8d"}


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def plot_timeseries(
    class_df: pd.DataFrame, tree_df: pd.DataFrame, measure: str, path: str | Path
) -> Path:
    """Class medians with their interquartile band, single soff trees behind.

    The trees are drawn thin and unlabelled on purpose: the question they answer
    is not "which tree is which" but "does the class median stand for all
    eighteen, or is one tree dragging it".
    """
    selection = class_df[class_df["measure"] == measure]
    if selection.empty:
        raise ValueError(
            f"no rows for measure {measure!r}; computed: {sorted(class_df['measure'].unique())}"
        )

    dates = sorted(selection["date"].unique())
    positions = {date: i for i, date in enumerate(dates)}
    fig, ax = plt.subplots(figsize=(11, 4.5))

    trees = tree_df[tree_df["measure"] == measure] if len(tree_df) else tree_df
    for _, group in trees.groupby("tree_id", observed=True) if len(trees) else []:
        group = group.sort_values("date")
        ax.plot(
            [positions[d] for d in group["date"]],
            group["median"],
            color=CLASS_COLORS["deadwood"],
            alpha=0.25,
            linewidth=0.8,
            zorder=1,
        )

    for name, group in selection.groupby("class", observed=True):
        group = group.sort_values("date")
        x = [positions[d] for d in group["date"]]
        color = CLASS_COLORS.get(str(name), None)
        ax.fill_between(x, group["q25"], group["q75"], color=color, alpha=0.15, linewidth=0)
        ax.plot(
            x, group["median"], color=color, marker="o", markersize=3, label=str(name), zorder=3
        )

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(f"median {measure}")
    ax.set_title(f"{measure} over the aligned time series")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_signature(signature_df: pd.DataFrame, path: str | Path) -> Path:
    """Mean reflectance against band, one panel per season."""
    seasons = sorted(signature_df["season"].unique())
    bands = list(dict.fromkeys(signature_df["band"]))
    fig, axes = plt.subplots(
        1, len(seasons), figsize=(5 * len(seasons), 4), sharey=True, squeeze=False
    )

    for ax, season in zip(axes[0], seasons):
        panel = signature_df[signature_df["season"] == season]
        for name, group in panel.groupby("class", observed=True):
            group = group.set_index("band").reindex(bands).reset_index()
            ax.errorbar(
                range(len(bands)),
                group["mean"],
                yerr=group["std"],
                marker="o",
                capsize=3,
                color=CLASS_COLORS.get(str(name), None),
                label=str(name),
            )
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, rotation=45, ha="right")
        ax.set_title(season)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("mean reflectance")
    axes[0][-1].legend()
    return _save(fig, path)
