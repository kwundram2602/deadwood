"""Every date's raw DSM on one grid: where do the surveys stop agreeing in height?

Nothing is fitted or converted here — no geoid, no co-registration — because a
survey offset or a height-system switch between campaigns is exactly what has
to show. Two views of the same stack:

- the height distribution per date (min, percentiles, max). Easy to read, but
  it carries the whole relief of the scene, so an offset smaller than the
  terrain range drowns in it, and a season with less canopy moves its upper
  percentiles without any offset at all.
- the pixel-wise difference to the stack: every pixel minus its own median over
  all dates. The terrain cancels, because each pixel is compared only with
  itself, and the median of that difference per date is the date's offset
  against the majority of the dates. A small IQR around it means the whole
  surface moved (offset); a large one means only parts did (canopy, tilt).

The reference is the majority, not the truth: when campaigns split into two
height systems of similar size, the offsets fall into two groups ~N apart and
zero means "like most dates". Which group is right is `vertical_datum`'s call.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the HPC
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.warp import reproject  # noqa: E402

from deadwood_spectral.grid import ReferenceGrid  # noqa: E402

logger = logging.getLogger(__name__)

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
LEVELS = ["min_m", *[f"p{q}_m" for q in PERCENTILES], "max_m"]
COLUMNS = [
    "date",
    "n_valid_px",
    *LEVELS,
    # This date's median height minus the median of every date's median: the
    # distribution view of the offset, relief and canopy share included.
    "p50_vs_median_m",
    # Median and IQR of (DSM - per-pixel median over all dates): the offset with
    # the terrain cancelled out, and how uniform it is.
    "offset_vs_stack_m",
    "offset_iqr_m",
]

# Offsets beyond this are labelled with their date in the figure — the dates to
# look up first. Well above what co-registered DSMs of one campaign scatter by.
FLAG_M = 1.0


def coarse_grid(grid: ReferenceGrid, resolution_m: float) -> ReferenceGrid:
    """The reference grid's extent at a coarser pixel size.

    A metre-scale offset does not need the 3.7 cm the DSMs come in, and dozens
    of full-resolution scenes do not fit in memory side by side; at 0.5 m one
    date is ~0.5 million pixels.
    """
    factor = resolution_m / abs(grid.transform.a)
    if factor < 1.0 - 1e-6:
        raise ValueError(
            f"resolution_m={resolution_m:g} is finer than the reference grid "
            f"({abs(grid.transform.a):g} m) — nothing to gain from upsampling"
        )
    # The tolerance keeps an exact multiple (12 m / 0.5 m) from rounding up to
    # one extra row of pure padding.
    height = int(np.ceil(grid.height / factor - 1e-6))
    width = int(np.ceil(grid.width / factor - 1e-6))
    return ReferenceGrid(height, width, grid.transform * Affine.scale(factor), grid.crs)


def load_stack(paths: dict[str, str | Path], grid: ReferenceGrid) -> np.ndarray:
    """(dates, height, width) heights on `grid`, NaN outside each DSM's footprint.

    Averaged rather than bilinear: going from centimetres to half a metre,
    bilinear samples a handful of source pixels per cell and keeps their noise,
    while the average uses all of them.
    """
    stack = np.full((len(paths), grid.height, grid.width), np.nan, dtype=np.float32)
    for index, (date, path) in enumerate(paths.items()):
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=stack[index],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=Resampling.average,
                dst_nodata=np.nan,
            )
        logger.info("%s: %.1f%% of the grid valid", date, 100.0 * np.isfinite(stack[index]).mean())
    return stack


def height_table(dates: list[str], stack: np.ndarray) -> pd.DataFrame:
    """One row per date, over the pixels valid on *every* date.

    The common footprint keeps the dates comparable: a date that covers one
    more strip of slope would shift its percentiles without any offset.
    """
    valid = np.isfinite(stack).all(axis=0)
    if not valid.any():
        raise ValueError("no pixel is valid on every date — the DSM footprints do not overlap")
    logger.info("%d px valid on all %d dates", int(valid.sum()), len(dates))

    values = stack[:, valid].astype(np.float64)
    levels = np.percentile(values, [0, *PERCENTILES, 100], axis=1)
    diff = values - np.median(values, axis=0)
    q25, offset, q75 = np.percentile(diff, [25, 50, 75], axis=1)

    table = pd.DataFrame({"date": dates, "n_valid_px": int(valid.sum())})
    for name, level in zip(LEVELS, levels):
        table[name] = level
    table["p50_vs_median_m"] = table["p50_m"] - table["p50_m"].median()
    table["offset_vs_stack_m"] = offset
    table["offset_iqr_m"] = q75 - q25
    return table[COLUMNS]


def timeseries_figure(table: pd.DataFrame):
    """Heights per date on top, the offset against the stack below, same dates."""
    when = mdates.date2num(pd.to_datetime(table["date"], format="%Y%m%d"))
    # Boxes a few days wide: the dates are irregular, and placing them at their
    # real time keeps a gap in the series visible as a gap.
    width = max(2.0, 0.6 * float(np.min(np.diff(when)))) if len(when) > 1 else 5.0

    fig, (heights, offsets) = plt.subplots(
        2, 1, figsize=(max(10.0, 0.35 * len(table)), 8), sharex=True, height_ratios=(3, 2)
    )
    stats = [
        {
            "med": row.p50_m,
            "q1": row.p25_m,
            "q3": row.p75_m,
            "whislo": row.p1_m,
            "whishi": row.p99_m,
            "fliers": [row.min_m, row.max_m],
        }
        for row in table.itertuples()
    ]
    heights.bxp(stats, positions=when, widths=width, manage_ticks=False)
    heights.set_ylabel("DSM height [m]")
    heights.set_title(
        "height distribution per date — box p25-p75, whiskers p1-p99, dots min/max", fontsize=10
    )
    heights.grid(axis="y", alpha=0.3)

    offsets.axhline(0.0, color="#7f8c8d", linewidth=1.0)
    offsets.errorbar(
        when,
        table["offset_vs_stack_m"],
        yerr=table["offset_iqr_m"] / 2.0,
        fmt="o",
        color="#c0392b",
        ecolor="#2c3e50",
        capsize=3,
        label="median (DSM - per-pixel median of all dates), bar = IQR",
    )
    for x, y, date in zip(when, table["offset_vs_stack_m"], table["date"]):
        if abs(y) > FLAG_M:
            offsets.annotate(date, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    offsets.set_ylabel("offset vs. stack [m]")
    offsets.legend(fontsize=8, loc="best")
    offsets.grid(axis="y", alpha=0.3)

    offsets.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
