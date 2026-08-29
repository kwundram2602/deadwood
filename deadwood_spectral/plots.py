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
import numpy as np  # noqa: E402
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


def timeseries_figure(class_df: pd.DataFrame, measure: str):
    """The class curves for one measure: median line plus interquartile band.

    Nothing else goes on this axis. The per-tree curves that used to sit behind
    the medians made eighteen overlapping lines out of one statement, and the
    statement is the only reason for the figure. The per-tree numbers stay
    available in `overview_tree.csv` for anyone who wants to check whether a
    single tree drags a median.
    """
    selection = class_df[class_df["measure"] == measure]
    if selection.empty:
        raise ValueError(
            f"no rows for measure {measure!r}; computed: {sorted(class_df['measure'].unique())}"
        )

    dates = sorted(selection["date"].unique())
    positions = {date: i for i, date in enumerate(dates)}
    fig, ax = plt.subplots(figsize=(11, 4.5))

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
    return fig


def plot_timeseries(class_df: pd.DataFrame, measure: str, path: str | Path) -> Path:
    """Build the figure for one measure and write it."""
    return _save(timeseries_figure(class_df, measure), path)


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


def topography_figure(topo_df: pd.DataFrame):
    """Per-tree nDSM height with its IQR, over the share of reconstructed pixels.

    Two panels because they answer in sequence: the lower one says whether the
    photogrammetry saw the tree, the upper one says how tall what it saw was.
    Reading a height for a tree whose bar is near zero is reading noise, and
    stacking them on one x makes that impossible to miss.
    """
    from deadwood_spectral.topography import ALL_TREES

    trees = topo_df[topo_df["tree_id"] != ALL_TREES].copy()
    if trees.empty:
        raise ValueError("no per-tree rows to plot")
    # NaN medians last: a tree with no reconstructed pixel has no height to
    # sort by, and it belongs at the end of the row rather than at the start.
    trees = trees.sort_values("median_m", na_position="last").reset_index(drop=True)
    x = range(len(trees))

    fig, axes = plt.subplots(
        2, 1, figsize=(max(6, 0.5 * len(trees)), 6), sharex=True, height_ratios=[2, 1]
    )
    lower = (trees["median_m"] - trees["q25_m"]).to_numpy()
    upper = (trees["q75_m"] - trees["median_m"]).to_numpy()
    axes[0].errorbar(
        x,
        trees["median_m"],
        yerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=4,
        capsize=3,
        color=CLASS_COLORS["deadwood"],
    )
    pooled = topo_df.loc[topo_df["tree_id"] == ALL_TREES, "median_m"]
    if len(pooled) and pd.notna(pooled.iloc[0]):
        axes[0].axhline(
            float(pooled.iloc[0]), color="#7f8c8d", linestyle="--", linewidth=1, label="all soff"
        )
        axes[0].legend()
    axes[0].set_ylabel("nDSM height [m]")
    axes[0].set_title("soff crowns: nDSM height (median, IQR) and photogrammetric coverage")
    axes[0].grid(alpha=0.3)

    axes[1].bar(x, trees["valid_frac"], color="#2c3e50", width=0.6)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("valid nDSM px")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(trees["tree_id"].astype(str), rotation=60, ha="right", fontsize=7)
    axes[1].grid(alpha=0.3, axis="y")
    return fig


def plot_topography(topo_df: pd.DataFrame, path: str | Path) -> Path:
    """Build the per-tree topography figure and write it."""
    return _save(topography_figure(topo_df), path)
