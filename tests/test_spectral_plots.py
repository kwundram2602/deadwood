import os
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.plots import plot_signature, plot_timeseries  # noqa: E402

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


def _tree_df():
    records = []
    for tree_id in ("4157", "4170"):
        for measure in ("ndvi", "ndre"):
            for i, date in enumerate(DATES):
                records.append(
                    {
                        "tree_id": tree_id,
                        "date": date,
                        "measure": measure,
                        "n_valid_px": 20,
                        "median": 0.25 + 0.05 * i,
                        "q25": 0.2,
                        "q75": 0.3,
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
    path = plot_timeseries(_class_df(), _tree_df(), "ndvi", tmp_path / "sub" / "ts_ndvi.png")
    assert path.exists() and path.stat().st_size > 0


def test_plot_timeseries_works_without_any_tree_curves(tmp_path):
    empty = _tree_df().iloc[:0]
    path = plot_timeseries(_class_df(), empty, "ndvi", tmp_path / "ts.png")
    assert path.exists()


def test_plot_timeseries_rejects_a_measure_that_was_not_computed(tmp_path):
    with pytest.raises(ValueError, match="evi"):
        plot_timeseries(_class_df(), _tree_df(), "evi", tmp_path / "ts.png")


def test_plot_signature_writes_a_file(tmp_path):
    path = plot_signature(_signature_df(), tmp_path / "signature.png")
    assert path.exists() and path.stat().st_size > 0
