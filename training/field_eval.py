"""Score a binary mask against field-surveyed polygons.

The field polygons are an incomplete picture of reality, so IoU or precision
would punish the model for real objects nobody mapped. What is measured instead
is recall-side: the share of each polygon's area the mask covers, and how many
polygons reach a given share.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

from training.crown_recall import crown_coverage, crown_pixel_index  # noqa: E402

# Categorical slots 1-4 of the reference palette, in fixed order; validated as adjacent pairs.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def evaluate_polygons(
    mask_path: str | Path, polygons_path: str | Path, layer: str | None = None
) -> pd.DataFrame:
    """Covered fraction of every polygon, indexed by fid.

    Polygons are reprojected to the mask's CRS: a mismatch would place them
    elsewhere on the grid and score as a silent total miss.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform, crs, nodata = src.transform, src.crs, src.nodata
    valid = mask != nodata if nodata is not None else None

    polygons = gpd.read_file(polygons_path, layer=layer, fid_as_index=True).to_crs(crs)
    index = crown_pixel_index(polygons, transform, mask.shape)
    df = crown_coverage(mask, index, valid=valid).drop(columns="hit")

    if "tree_id" in polygons.columns:
        df.insert(0, "tree_id", polygons["tree_id"].reindex(df.index).to_numpy())
    df.insert(1 if "tree_id" in df.columns else 0, "area_m2", polygons.area.reindex(df.index))
    return df


def summarize(per_polygon: pd.DataFrame, thresholds: Sequence[float]) -> pd.Series:
    """Aggregate covered fractions; polygons without valid pixels are counted but not scored."""
    scored = per_polygon.dropna(subset=["covered_fraction"])
    frac = scored["covered_fraction"]
    n = len(scored)
    out: dict[str, float | int] = {
        "n_polygons": len(per_polygon),
        "n_evaluated": n,
        "mean_covered_fraction": float(frac.mean()) if n else float("nan"),
        "median_covered_fraction": float(frac.median()) if n else float("nan"),
        "area_weighted_covered_fraction": (
            float(np.average(frac, weights=scored["area_m2"]))
            if n and scored["area_m2"].sum() > 0
            else float("nan")
        ),
    }
    for t in thresholds:
        detected = int((frac >= t).sum())
        out[f"detected_ge_{t:g}"] = detected
        out[f"detected_rate_ge_{t:g}"] = detected / n if n else float("nan")
    return pd.Series(out)


def plot_detection(
    results: Mapping[str, pd.DataFrame], thresholds: Sequence[float], path: str | Path
) -> Path:
    """Histogram of covered fractions and the share of polygons detected at each cut-off."""
    if len(results) > len(SERIES_COLORS):
        # zip() would silently drop the extra series from the figure.
        raise ValueError(f"at most {len(SERIES_COLORS)} series, got {len(results)}")
    path = Path(path)
    fig, (ax_hist, ax_cum) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    bins = np.linspace(0, 1, 11)
    x = np.linspace(0, 1, 201)

    for (name, df), color in zip(results.items(), SERIES_COLORS):
        frac = df["covered_fraction"].dropna().to_numpy()
        label = f"{name} (n={frac.size})"
        # Share rather than count: the two sets differ in size.
        weights = np.full(frac.size, 1 / frac.size) if frac.size else None
        ax_hist.hist(
            frac,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )
        share = [(frac >= xi).mean() if frac.size else np.nan for xi in x]
        ax_cum.plot(x, share, linewidth=2, color=color, label=label)
        for t in thresholds:
            rate = (frac >= t).mean() if frac.size else np.nan
            ax_cum.plot([t], [rate], "o", markersize=8, color=color, markeredgecolor="white")
            ax_cum.annotate(
                f"{rate:.0%}",
                (t, rate),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                color="#3b3b38",
            )

    for t in thresholds:
        ax_cum.axvline(t, color="#9a9990", linewidth=1, linestyle="--")

    ax_hist.set_xlabel("covered fraction of polygon area")
    ax_hist.set_ylabel("share of polygons")
    ax_hist.set_title("Distribution of covered fraction", loc="left")
    ax_cum.set_xlabel("minimum covered fraction")
    ax_cum.set_ylabel("share of polygons detected")
    ax_cum.set_title("Polygons detected vs. required coverage", loc="left")
    ax_cum.set_ylim(0, 1.02)
    for ax in (ax_hist, ax_cum):
        ax.set_xlim(0, 1)
        ax.grid(axis="y", color="#e4e3dc", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    # High coverage is where the histogram mass sits; the cumulative curve starts top-left.
    ax_hist.legend(frameon=False, loc="upper left")
    ax_cum.legend(frameon=False, loc="upper right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def threshold_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Rows of a summary whose target name ends in `_t<threshold>`, indexed by that threshold."""
    thr = summary.index.to_series().str.extract(r"_t(\d+(?:\.\d+)?)$", expand=False)
    rows = summary[thr.notna().to_numpy()].copy()
    rows.index = pd.Index(thr.dropna().astype(float).to_numpy(), name="threshold")
    return rows.sort_index()


def plot_threshold_comparison(
    summaries: Mapping[str, pd.DataFrame], detect_at: float, path: str | Path
) -> Path:
    """Detection rate and mean covered fraction against probability threshold, one line per model."""
    if len(summaries) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} series, got {len(summaries)}")
    path = Path(path)
    rate_col = f"detected_rate_ge_{detect_at:g}"
    panels = [
        (rate_col, f"share of polygons with ≥ {detect_at:.0%} covered"),
        ("mean_covered_fraction", "mean covered fraction of polygon area"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")

    for (name, summary), color in zip(summaries.items(), SERIES_COLORS):
        table = threshold_table(summary)
        for ax, (col, _) in zip(axes, panels):
            ax.plot(
                table.index,
                table[col],
                "-o",
                linewidth=2,
                markersize=8,
                color=color,
                markeredgecolor="white",
                label=name,
            )

    for ax, (_, ylabel) in zip(axes, panels):
        ax.set_xlabel("crown probability threshold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e4e3dc", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Polygons detected", loc="left")
    axes[1].set_title("Mean covered area", loc="left")
    axes[0].legend(frameon=False, loc="lower left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
