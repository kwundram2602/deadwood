import os
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.plots import (  # noqa: E402
    plot_signature,
    plot_timeseries,
    timeseries_figure,
)

DATES = ["20240116", "20240613"]


def _class_df():
    records = []
    for name in ("deadwood", "living", "background"):
        for measure in ("ndvi", "ndre"):
            for i, date in enumerate(DATES):
                records.append(
                    {
                        "class": name,
                        "date": date,
                        "measure": measure,
                        "n_valid_px": 100,
                        "median": 0.3 + 0.1 * i,
                        "q25": 0.2,
                        "q75": 0.5,
                    }
                )
    return pd.DataFrame.from_records(records)


def _signature_df():
    records = []
    for name in ("deadwood", "living", "background"):
        for season in ("dry", "wet"):
            for band in ("Green", "Red", "RedEdge", "NIR"):
                records.append(
                    {
                        "class": name,
                        "season": season,
                        "band": band,
                        "n_valid_px": 100,
                        "mean": 0.3,
                        "std": 0.05,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_plot_timeseries_writes_a_file(tmp_path):
    path = plot_timeseries(_class_df(), "ndvi", tmp_path / "sub" / "ts_ndvi.png")
    assert path.exists() and path.stat().st_size > 0


def test_the_figure_draws_one_line_and_one_band_per_class():
    """Median plus IQR, nothing else — no per-tree curves behind them."""
    fig = timeseries_figure(_class_df(), "ndvi")
    ax = fig.axes[0]
    assert len(ax.lines) == 3
    assert len(ax.collections) == 3


def test_plot_timeseries_rejects_a_measure_that_was_not_computed(tmp_path):
    with pytest.raises(ValueError, match="evi"):
        plot_timeseries(_class_df(), "evi", tmp_path / "ts.png")


def test_plot_signature_writes_a_file(tmp_path):
    path = plot_signature(_signature_df(), tmp_path / "signature.png")
    assert path.exists() and path.stat().st_size > 0
