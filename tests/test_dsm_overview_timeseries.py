"""Raw DSMs of every date side by side: where do their heights stop agreeing?

The synthetic scene is a sloping ground with a few trees, flown on several
dates. One date is lifted as a whole — the survey offset the mode exists to
find — and one has lost its trees, which must *not* read as an offset.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import load_reference_grid  # noqa: E402
from dsm_overview.overview import run_timeseries_overview  # noqa: E402
from dsm_overview.timeseries import (  # noqa: E402
    COLUMNS,
    coarse_grid,
    height_table,
    load_stack,
    timeseries_figure,
)

CRS = "EPSG:32736"
SIZE = 120
FINE = 0.1  # the DSMs and the reference grid
OFFSET = 5.6  # the lifted date, like 20260313
OFFSET_DATE = "20240310"
BARE_DATE = "20240320"
DATES = ("20240301", "20240305", OFFSET_DATE, "20240315", BARE_DATE)


def _raster(path, data, res=FINE):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=from_origin(1000.0, 2000.0, res, res),
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return str(path)


def _surface(trees: bool = True):
    yy = np.mgrid[0:SIZE, 0:SIZE][0].astype(np.float32)
    # 6 m of slope across the scene: wider than the offset, so a percentile of
    # the heights alone cannot pin the offset down.
    dsm = 500.0 + 0.05 * yy
    if trees:
        for r, c in ((10, 10), (70, 60)):
            dsm[r : r + 20, c : c + 20] += 12.0
    return dsm


@pytest.fixture
def scene(tmp_path):
    dsm_dir = tmp_path / "dsm"
    dsm_dir.mkdir()
    for date in DATES:
        dsm = _surface(trees=date != BARE_DATE)
        if date == OFFSET_DATE:
            dsm = dsm + OFFSET
        _raster(dsm_dir / f"{date}_Airport_DSM_coreg.tif", dsm)
    # Something that is not a DSM of the series and must not be picked up.
    _raster(dsm_dir / "20240301_Airport_DSM_merged.tif", _surface() + 100.0)
    return {
        "reference": _raster(tmp_path / "ref.tif", np.zeros((SIZE, SIZE))),
        "dsm_dir": str(dsm_dir),
        "out_dir": str(tmp_path / "out"),
    }


def _paths(scene, dates=DATES):
    return {d: f"{scene['dsm_dir']}/{d}_Airport_DSM_coreg.tif" for d in dates}


# --- the common grid ----------------------------------------------------------


def test_the_coarse_grid_keeps_the_extent_at_the_new_resolution(scene):
    grid = coarse_grid(load_reference_grid(scene["reference"]), 0.5)

    assert grid.shape == (SIZE * FINE / 0.5, SIZE * FINE / 0.5)
    assert grid.transform.a == pytest.approx(0.5)
    assert grid.transform.e == pytest.approx(-0.5)
    assert (grid.transform.c, grid.transform.f) == (1000.0, 2000.0)


def test_a_resolution_finer_than_the_reference_is_rejected(scene):
    with pytest.raises(ValueError, match="finer"):
        coarse_grid(load_reference_grid(scene["reference"]), 0.05)


def test_the_stack_holds_one_layer_per_date_on_the_grid(scene):
    grid = coarse_grid(load_reference_grid(scene["reference"]), 0.5)
    stack = load_stack(_paths(scene), grid)

    assert stack.shape == (len(DATES), *grid.shape)
    assert np.nanmedian(stack[DATES.index(OFFSET_DATE)] - stack[0]) == pytest.approx(
        OFFSET, abs=0.01
    )


# --- the table ----------------------------------------------------------------


def _table(scene, dates=DATES):
    grid = coarse_grid(load_reference_grid(scene["reference"]), 0.5)
    return height_table(list(dates), load_stack(_paths(scene, dates), grid)).set_index("date")


def test_the_table_has_one_row_per_date_in_a_fixed_column_order(scene):
    table = _table(scene).reset_index()

    assert list(table.columns) == COLUMNS
    assert list(table["date"]) == list(DATES)


def test_the_percentiles_are_ordered_from_min_to_max(scene):
    table = _table(scene)
    levels = ["min_m", "p1_m", "p5_m", "p25_m", "p50_m", "p75_m", "p95_m", "p99_m", "max_m"]

    assert (table[levels].diff(axis=1).iloc[:, 1:] >= 0).all().all()


def test_the_lifted_date_stands_out_by_the_offset(scene):
    table = _table(scene)

    assert table.loc[OFFSET_DATE, "offset_vs_stack_m"] == pytest.approx(OFFSET, abs=0.05)
    assert table.loc[OFFSET_DATE, "p50_vs_median_m"] == pytest.approx(OFFSET, abs=0.3)
    # Uniform: the whole surface moved, not a part of it.
    assert table.loc[OFFSET_DATE, "offset_iqr_m"] < 0.05


def test_the_other_dates_sit_at_zero(scene):
    table = _table(scene)
    others = table.drop(index=[OFFSET_DATE])

    assert np.abs(others["offset_vs_stack_m"]).max() < 0.05


def test_lost_canopy_does_not_read_as_an_offset(scene):
    # The bare date lacks the trees on ~6% of the scene: its upper percentiles
    # drop, but the pixel-wise median difference stays at zero.
    table = _table(scene)

    wooded = table.drop(index=[OFFSET_DATE, BARE_DATE])
    assert table.loc[BARE_DATE, "p99_m"] < wooded["p99_m"].min() - 5.0
    assert abs(table.loc[BARE_DATE, "offset_vs_stack_m"]) < 0.05


def test_only_pixels_valid_on_every_date_count(scene):
    grid = coarse_grid(load_reference_grid(scene["reference"]), 0.5)
    stack = load_stack(_paths(scene), grid)
    stack[1, :10, :] = np.nan  # one date misses a strip

    table = height_table(list(DATES), stack)

    assert table["n_valid_px"].nunique() == 1
    assert table["n_valid_px"].iloc[0] == (grid.height - 10) * grid.width


def test_no_common_pixel_raises(scene):
    grid = coarse_grid(load_reference_grid(scene["reference"]), 0.5)
    stack = load_stack(_paths(scene), grid)
    stack[0, : grid.height // 2] = np.nan
    stack[1, grid.height // 2 :] = np.nan

    with pytest.raises(ValueError, match="no pixel"):
        height_table(list(DATES), stack)


# --- the figure and the run ---------------------------------------------------


def test_the_figure_has_a_box_per_date_and_an_offset_panel(scene):
    fig = timeseries_figure(_table(scene).reset_index())
    try:
        heights, offsets = fig.axes
        assert len(heights.patches) + len(heights.lines) > 0
        assert len(offsets.lines) > 0
    finally:
        plt.close(fig)


def test_the_run_writes_csv_and_figure(scene):
    outputs = run_timeseries_overview(**scene, pattern="*_DSM_coreg.tif", resolution_m=0.5)

    assert set(outputs) == {"stats_csv", "plot"}
    assert all(path.exists() for path in outputs.values())

    table = pd.read_csv(outputs["stats_csv"], dtype={"date": str})
    assert list(table["date"]) == list(DATES)  # the *_merged file was not picked up


def test_the_run_can_be_restricted_to_some_dates(scene):
    outputs = run_timeseries_overview(
        **scene, pattern="*_DSM_coreg.tif", resolution_m=0.5, dates=[DATES[0], 20240305]
    )

    table = pd.read_csv(outputs["stats_csv"], dtype={"date": str})
    assert list(table["date"]) == [DATES[0], "20240305"]


def test_a_requested_date_without_a_dsm_raises(scene):
    with pytest.raises(ValueError, match="20991231"):
        run_timeseries_overview(
            **scene, pattern="*_DSM_coreg.tif", resolution_m=0.5, dates=["20991231"]
        )


def test_without_a_reference_the_first_dsm_sets_the_grid(scene):
    scene = {**scene, "reference": None}
    outputs = run_timeseries_overview(**scene, pattern="*_DSM_coreg.tif", resolution_m=0.5)

    table = pd.read_csv(outputs["stats_csv"], dtype={"date": str})
    assert (table["n_valid_px"] == (SIZE * FINE / 0.5) ** 2).all()
