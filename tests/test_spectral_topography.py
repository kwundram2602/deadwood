"""Per-tree nDSM check: does the photogrammetry reconstruct a soff crown at all?

The height numbers are secondary here. The first question a dead, leafless
crown raises is whether SfM produced a surface over it, and that shows up as
the share of nDSM pixels that are not NaN.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402
from deadwood_spectral.topography import (  # noqa: E402
    ALL_TREES,
    read_single_band,
    topography_table,
)

CRS = "EPSG:32736"
TRANSFORM = from_origin(1000.0, 2000.0, 1.0, 1.0)
SHAPE = (20, 20)


@pytest.fixture
def grid():
    return ReferenceGrid(SHAPE[0], SHAPE[1], TRANSFORM, rasterio.crs.CRS.from_string(CRS))


def _raster(path, data, transform=TRANSFORM, crs=CRS):
    data = np.asarray(data, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return path


def _pixels(rows, cols, tree_ids):
    return pd.DataFrame(
        {
            "row": np.asarray(rows, dtype=np.int64),
            "col": np.asarray(cols, dtype=np.int64),
            "class": "deadwood",
            "tree_id": pd.Series(tree_ids, dtype="string"),
        }
    )


def test_read_returns_the_value_under_every_pixel(tmp_path, grid):
    data = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    path = _raster(tmp_path / "ndsm.tif", data)
    rows = np.array([0, 5, 19, 12])
    cols = np.array([3, 0, 19, 7])

    values = read_single_band(path, grid, rows, cols, chunk_rows=4)

    assert values.shape == rows.shape
    np.testing.assert_allclose(values, data[rows, cols])


def test_read_carries_the_nodata_holes_through_as_nan(tmp_path, grid):
    data = np.ones(SHAPE, dtype=np.float32)
    data[7, 7] = np.nan
    path = _raster(tmp_path / "ndsm.tif", data)

    values = read_single_band(path, grid, np.array([7, 8]), np.array([7, 8]))

    assert np.isnan(values[0])
    assert values[1] == 1.0


def test_read_rejects_a_raster_off_the_reference_grid(tmp_path, grid):
    path = _raster(
        tmp_path / "shifted.tif", np.ones(SHAPE), transform=from_origin(1500.0, 2000.0, 1.0, 1.0)
    )
    with pytest.raises(ValueError, match="transform"):
        read_single_band(path, grid, np.array([0]), np.array([0]))


def test_the_table_reports_the_quartiles_of_each_tree():
    pixels = _pixels(
        rows=[0, 1, 2, 3, 4, 5, 6, 7],
        cols=[0, 1, 2, 3, 4, 5, 6, 7],
        tree_ids=["4157"] * 5 + ["4170"] * 3,
    )
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 10.0, 10.0])

    table = topography_table(values, pixels).set_index("tree_id")

    assert table.loc["4157", "median_m"] == 3.0
    assert table.loc["4157", "q25_m"] == 2.0
    assert table.loc["4157", "q75_m"] == 4.0
    assert table.loc["4157", "iqr_m"] == 2.0
    assert table.loc["4170", "median_m"] == 10.0
    assert table.loc["4170", "iqr_m"] == 0.0


def test_the_table_counts_the_photogrammetry_holes_per_tree():
    pixels = _pixels(rows=[0, 1, 2, 3], cols=[0, 1, 2, 3], tree_ids=["4157"] * 2 + ["4170"] * 2)
    values = np.array([np.nan, 2.0, 3.0, 4.0])

    table = topography_table(values, pixels).set_index("tree_id")

    assert table.loc["4157", "n_px"] == 2
    assert table.loc["4157", "n_valid_px"] == 1
    assert table.loc["4157", "valid_frac"] == 0.5
    assert table.loc["4170", "valid_frac"] == 1.0


def test_a_tree_without_a_single_reconstructed_pixel_is_kept_as_a_row():
    """The whole point of the check: a missing tree must be visible, not absent."""
    pixels = _pixels(rows=[0, 1], cols=[0, 1], tree_ids=["4157", "4157"])
    values = np.array([np.nan, np.nan])

    table = topography_table(values, pixels).set_index("tree_id")

    assert table.loc["4157", "n_valid_px"] == 0
    assert table.loc["4157", "valid_frac"] == 0.0
    assert np.isnan(table.loc["4157", "median_m"])


def test_the_summary_row_pools_every_soff_pixel():
    pixels = _pixels(rows=[0, 1, 2, 3], cols=[0, 1, 2, 3], tree_ids=["4157"] * 2 + ["4170"] * 2)
    values = np.array([1.0, 2.0, 3.0, 4.0])

    table = topography_table(values, pixels)

    assert table["tree_id"].iloc[-1] == ALL_TREES
    summary = table.set_index("tree_id").loc[ALL_TREES]
    assert summary["n_px"] == 4
    assert summary["median_m"] == 2.5


def test_the_table_ignores_everything_that_is_not_a_soff_tree():
    pixels = _pixels(rows=[0, 1, 2], cols=[0, 1, 2], tree_ids=["4157", pd.NA, pd.NA])
    pixels.loc[1:, "class"] = "living"
    values = np.array([1.0, 99.0, 99.0])

    table = topography_table(values, pixels).set_index("tree_id")

    assert set(table.index) == {"4157", ALL_TREES}
    assert table.loc[ALL_TREES, "n_px"] == 1


def test_the_trees_come_out_in_a_stable_order():
    pixels = _pixels(rows=[0, 1, 2], cols=[0, 1, 2], tree_ids=["4170", "4157", "4345"])
    table = topography_table(np.array([1.0, 2.0, 3.0]), pixels)
    assert list(table["tree_id"]) == ["4157", "4170", "4345", ALL_TREES]
