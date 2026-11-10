"""Box + violin plot of the squared-error distribution in scene error maps.

Reads one or more error_<scene>.tif written by training/scene_errors.py
(per-pixel (p − y)², NaN where there is no valid label) and draws one box and
one violin per scene, in two panels:

    linear    SE on its own scale, mean (= MSE of the scene) as a marker
    log10     only the pixels with SE > 0, box and violin computed on log10(SE)

SE is heavily right-skewed (most pixels ≈ 0), so the linear panel shows the
tail up to 1 and the log panel shows the shape of the non-zero errors. Under
each scene the pixel count and the share of SE = 0 are printed.

Box statistics (quartiles, 1.5 IQR whiskers) and the numbers in the JSON are
exact over all valid pixels; the violins (KDE) use a random subsample of
--sample pixels per scene.

Outputs (refuses to overwrite without --overwrite):
    <out>.png     the plot
    <out>.json    per-scene stats: n_px, frac_zero, mean, p25, median, p75, p90, p99, max

Usage:
    uv run python scripts/plot_error_distribution.py \\
        experiments/<run>/eval/scene_errors
    uv run python scripts/plot_error_distribution.py \\
        experiments/<run>/eval/scene_errors/error_20260226.tif --out /tmp/se.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.scene_errors import _INK, _INK_2  # noqa: E402

_FILL = "#3987e5"  # mid step of the error ramp in training/scene_errors.py
_GRID = "#e4e3df"
_SURFACE = "#fcfcfb"


def find_tifs(paths: list[Path]) -> list[Path]:
    """TIFs as given; a folder expands to its sorted error_*.tif."""
    tifs: list[Path] = []
    for p in paths:
        if p.is_dir():
            found = sorted(p.glob("error_*.tif"))
            if not found:
                raise FileNotFoundError(f"no error_*.tif in {p}")
            tifs.extend(found)
        elif p.is_file():
            tifs.append(p)
        else:
            raise FileNotFoundError(p)
    return tifs


def scene_label(path: Path) -> str:
    """error_20260226.tif -> 20260226; other names keep their stem."""
    return path.stem.removeprefix("error_")


def read_valid(path: Path) -> np.ndarray:
    """All finite pixel values of band 1, read block by block."""
    parts = []
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            block = src.read(1, window=window)
            parts.append(block[np.isfinite(block)])
    return np.concatenate(parts).astype(np.float32) if parts else np.empty(0, np.float32)


def se_stats(values: np.ndarray) -> dict:
    """Exact distribution statistics of the squared errors of one scene."""
    if values.size == 0:
        return {"n_px": 0}
    p25, p50, p75, p90, p99 = np.percentile(values, [25, 50, 75, 90, 99])
    return {
        "n_px": int(values.size),
        "frac_zero": float((values == 0).mean()),
        "mean": float(values.mean(dtype=np.float64)),
        "p25": float(p25),
        "median": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p99": float(p99),
        "max": float(values.max()),
    }


def box_stats(values: np.ndarray, label: str) -> dict:
    """Exact Tukey box for ax.bxp: quartiles, whiskers at the last point within 1.5 IQR."""
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "label": label,
        "q1": q1,
        "med": med,
        "q3": q3,
        "whislo": float(values[values >= lo].min()),
        "whishi": float(values[values <= hi].max()),
        "mean": float(values.mean(dtype=np.float64)),
        "fliers": [],
    }


def subsample(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= n:
        return values
    return rng.choice(values, size=n, replace=False)


def _draw_panel(ax, data: list[np.ndarray], samples: list[np.ndarray], show_mean: bool) -> None:
    """Violin (from samples) + exact box (from data) at positions 1..n; empty groups skipped."""
    pos = [i + 1 for i, d in enumerate(data) if d.size > 0]
    if not pos:
        ax.text(0.5, 0.5, "no values", transform=ax.transAxes, ha="center", color=_INK_2)
        return
    parts = ax.violinplot(
        [samples[i - 1] for i in pos], positions=pos, widths=0.8,
        showextrema=False, points=200,
    )
    for body in parts["bodies"]:
        body.set_facecolor(_FILL)
        body.set_edgecolor(_FILL)
        body.set_alpha(0.35)
    ax.bxp(
        [box_stats(data[i - 1], "") for i in pos], positions=pos, widths=0.15,
        showfliers=False, showmeans=show_mean, patch_artist=True,
        boxprops={"facecolor": _SURFACE, "edgecolor": _INK, "linewidth": 1.2},
        medianprops={"color": _INK, "linewidth": 2},
        whiskerprops={"color": _INK, "linewidth": 1.2},
        capprops={"color": _INK, "linewidth": 1.2},
        meanprops={"marker": "o", "markersize": 8, "markerfacecolor": _FILL,
                   "markeredgecolor": _SURFACE, "markeredgewidth": 2},
    )


def plot_distribution(groups: dict[str, tuple[np.ndarray, dict]], path: Path,
                      n_sample: int, seed: int) -> None:
    """Linear and log10 panels, one box + violin per scene."""
    rng = np.random.default_rng(seed)
    labels = list(groups)
    lin = [v for v, _ in groups.values()]
    log = [np.log10(v[v > 0]) for v in lin]
    lin_s = [subsample(v, n_sample, rng) for v in lin]
    log_s = [subsample(v, n_sample, rng) for v in log]

    fig, (ax_lin, ax_log) = plt.subplots(
        1, 2, figsize=(max(8, 2.2 * len(labels) + 4), 5.5), facecolor=_SURFACE
    )
    _draw_panel(ax_lin, lin, lin_s, show_mean=True)
    _draw_panel(ax_log, log, log_s, show_mean=False)

    ticks = [
        f"{lbl}\nn = {s['n_px']:,}\nSE = 0: {100 * s['frac_zero']:.1f} %"
        if s["n_px"] else f"{lbl}\nn = 0"
        for lbl, (_, s) in groups.items()
    ]
    ax_lin.set_ylabel("squared error (p − y)²", color=_INK_2)
    ax_lin.set_title("linear  (● = mean = MSE)", color=_INK, loc="left")
    ax_log.set_ylabel("squared error (p − y)², SE > 0", color=_INK_2)
    ax_log.set_title("log10, pixels with SE > 0", color=_INK, loc="left")
    ax_log.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_log.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"$10^{{{y:g}}}$"))
    for ax in (ax_lin, ax_log):
        ax.set_facecolor(_SURFACE)
        ax.set_xticks(range(1, len(labels) + 1), ticks, color=_INK)
        ax.set_xlim(0.4, len(labels) + 0.6)
        ax.tick_params(colors=_INK_2, labelsize=9)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)

    fig.suptitle("Squared-error distribution per scene", color=_INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=_SURFACE)
    plt.close(fig)


def _print_table(stats: dict[str, dict]) -> None:
    keys = ("mean", "median", "p90", "p99", "max")
    header = f"{'scene':>10}  {'n_px':>12}  {'zero %':>7}  " + "  ".join(f"{k:>9}" for k in keys)
    print(header)
    print("-" * len(header))
    for scene, s in stats.items():
        if not s["n_px"]:
            print(f"{scene:>10}  {0:>12,}")
            continue
        print(f"{scene:>10}  {s['n_px']:>12,}  {100 * s['frac_zero']:>7.2f}  "
              + "  ".join(f"{s[k]:>9.4f}" for k in keys))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path,
                        help="error_<scene>.tif files or folders containing them")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG (default: <folder of first TIF>/error_distribution.png)")
    parser.add_argument("--sample", type=int, default=200_000,
                        help="pixels per scene for the violin KDE (stats use all pixels)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    tifs = find_tifs(args.paths)
    out = args.out or tifs[0].parent / "error_distribution.png"
    out_json = out.with_suffix(".json")
    if not args.overwrite:
        existing = [p for p in (out, out_json) if p.exists()]
        if existing:
            raise FileExistsError(f"{existing[0]} exists; pass --overwrite to replace it")

    groups: dict[str, tuple[np.ndarray, dict]] = {}
    for tif in tifs:
        values = read_valid(tif)
        groups[scene_label(tif)] = (values, se_stats(values))
        print(f"read {tif}: {values.size:,} valid px")

    stats = {scene: s for scene, (_, s) in groups.items()}
    _print_table(stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_distribution(groups, out, args.sample, args.seed)
    out_json.write_text(json.dumps(
        {"sources": [str(t) for t in tifs], "sample": args.sample, "seed": args.seed,
         "scenes": stats},
        indent=2,
    ))
    print(f"-> {out}\n-> {out_json}")


if __name__ == "__main__":
    main()
