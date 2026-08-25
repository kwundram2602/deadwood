import os
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402
from deadwood_spectral.indices import BAND_NAMES  # noqa: E402
from deadwood_spectral.masks import ClassMasks  # noqa: E402
from deadwood_spectral.overview import (  # noqa: E402
    MEASURES,
    class_table,
    read_values,
    season_of,
    select_pixels,
    signature_table,
    stack_dates,
    stack_paths,
    tree_table,
)

GRID = ReferenceGrid(
    10, 10, from_origin(1000.0, 2000.0, 1.0, 1.0), rasterio.crs.CRS.from_epsg(32736)
)


def _write_stack(path, nir=0.5, red=0.1, nan_pixel=None):
    """A 7-band scene on GRID, constant per band, optionally one NaN pixel."""
    values = {"R": 0.1, "G": 0.2, "B": 0.15, "Green": 0.2, "Red": red, "RedEdge": 0.3, "NIR": nir}
    data = np.stack([np.full(GRID.shape, values[b], dtype=np.float32) for b in BAND_NAMES])
    if nan_pixel is not None:
        data[:, nan_pixel[0], nan_pixel[1]] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=GRID.height,
        width=GRID.width,
        count=len(BAND_NAMES),
        dtype="float32",
        crs=GRID.crs,
        transform=GRID.transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data)
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(i, name)
    return path


def _stack_dir(tmp_path, dates, nir_by_date=None, nan_pixel=None):
    directory = tmp_path / "timeseries"
    directory.mkdir(exist_ok=True)
    for date in dates:
        nir = 0.5 if nir_by_date is None else nir_by_date[date]
        _write_stack(directory / f"{date}_stack.tif", nir=nir, nan_pixel=nan_pixel)
    return directory


def _masks(deadwood_rows=(0,), living_rows=(3, 4), background_rows=(7, 8, 9)):
    """Row-band masks: whole rows belong to one class, so counts are obvious."""
    empty = lambda: np.zeros(GRID.shape, dtype=bool)  # noqa: E731
    deadwood, living, background = empty(), empty(), empty()
    for r in deadwood_rows:
        deadwood[r] = True
    for r in living_rows:
        living[r] = True
    for r in background_rows:
        background[r] = True
    tree_idx = np.zeros(GRID.shape, dtype=np.int32)
    for i, r in enumerate(deadwood_rows, start=1):
        tree_idx[r] = i
    return ClassMasks(
        deadwood=deadwood,
        living=living,
        background=background,
        tree_idx=tree_idx,
        tree_ids={i: f"tree{i}" for i in range(1, len(deadwood_rows) + 1)},
    )


# ── date discovery ─────────────────────────────────────────────────────────


def test_stack_dates_returns_every_scene_sorted(tmp_path):
    directory = _stack_dir(tmp_path, ["20240312", "20230824", "20240116"])
    assert stack_dates(directory) == ["20230824", "20240116", "20240312"]


def test_stack_dates_raises_on_an_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no aligned stack"):
        stack_dates(tmp_path / "empty")


def test_stack_paths_fails_before_reading_when_a_scene_is_missing(tmp_path):
    directory = _stack_dir(tmp_path, ["20240116"])
    with pytest.raises(FileNotFoundError, match="20240312"):
        stack_paths(directory, ["20240116", "20240312"])


# ── pixel selection ────────────────────────────────────────────────────────


def test_select_pixels_takes_every_deadwood_pixel():
    pixels = select_pixels(_masks(), max_pixels_per_class=2)
    assert (pixels["class"] == "deadwood").sum() == 10


def test_select_pixels_caps_the_reference_classes():
    pixels = select_pixels(_masks(), max_pixels_per_class=5)
    assert (pixels["class"] == "living").sum() == 5
    assert (pixels["class"] == "background").sum() == 5


def test_select_pixels_keeps_all_reference_pixels_below_the_cap():
    pixels = select_pixels(_masks(), max_pixels_per_class=1000)
    assert (pixels["class"] == "living").sum() == 20


def test_select_pixels_is_deterministic_for_a_seed():
    first = select_pixels(_masks(), max_pixels_per_class=5, seed=7)
    second = select_pixels(_masks(), max_pixels_per_class=5, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_select_pixels_draws_differently_for_another_seed():
    first = select_pixels(_masks(), max_pixels_per_class=5, seed=0)
    second = select_pixels(_masks(), max_pixels_per_class=5, seed=1)
    assert not first[["row", "col"]].equals(second[["row", "col"]])


def test_every_selected_pixel_lies_inside_its_own_mask():
    masks = _masks()
    pixels = select_pixels(masks, max_pixels_per_class=5)
    for name, group in pixels.groupby("class", observed=True):
        assert getattr(masks, str(name))[group["row"], group["col"]].all()


def test_only_deadwood_pixels_carry_a_tree_id():
    pixels = select_pixels(_masks(), max_pixels_per_class=5)
    assert pixels.loc[pixels["class"] == "deadwood", "tree_id"].notna().all()
    assert pixels.loc[pixels["class"] != "deadwood", "tree_id"].isna().all()


def test_tree_ids_come_from_the_field_attribute_not_the_burned_index():
    pixels = select_pixels(_masks(deadwood_rows=(0, 1)), max_pixels_per_class=5)
    assert set(pixels["tree_id"].dropna()) == {"tree1", "tree2"}


# ── reading the time series ────────────────────────────────────────────────


def test_read_values_returns_one_column_per_date(tmp_path):
    paths = stack_paths(_stack_dir(tmp_path, ["20240116", "20240312"]), ["20240116", "20240312"])
    values = read_values(paths, GRID, np.array([0, 5]), np.array([0, 5]), ["ndvi"])
    assert values["ndvi"].shape == (2, 2)


def test_read_values_computes_indices_from_the_scene(tmp_path):
    directory = _stack_dir(tmp_path, ["20240116"], nir_by_date={"20240116": 0.5})
    values = read_values(
        stack_paths(directory, ["20240116"]), GRID, np.array([0]), np.array([0]), ["ndvi"]
    )
    assert values["ndvi"][0, 0] == pytest.approx(0.4 / 0.6, abs=1e-6)


def test_read_values_also_returns_raw_bands(tmp_path):
    directory = _stack_dir(tmp_path, ["20240116"])
    values = read_values(
        stack_paths(directory, ["20240116"]), GRID, np.array([0]), np.array([0]), ["NIR"]
    )
    assert values["NIR"][0, 0] == pytest.approx(0.5, abs=1e-6)


def test_read_values_preserves_input_order_across_chunks(tmp_path):
    """Rows are read in chunks; the output must stay in the caller's order."""
    directory = _stack_dir(tmp_path, ["20240116"])
    rows = np.array([9, 0, 5])
    cols = np.array([1, 2, 3])
    values = read_values(
        stack_paths(directory, ["20240116"]), GRID, rows, cols, ["NIR"], chunk_rows=2
    )
    assert values["NIR"].shape == (3, 1)
    assert np.isfinite(values["NIR"]).all()


def test_read_values_yields_nan_where_the_scene_has_nodata(tmp_path):
    directory = _stack_dir(tmp_path, ["20240116"], nan_pixel=(0, 0))
    values = read_values(
        stack_paths(directory, ["20240116"]), GRID, np.array([0, 1]), np.array([0, 0]), ["ndvi"]
    )
    assert np.isnan(values["ndvi"][0, 0])
    assert np.isfinite(values["ndvi"][1, 0])


def test_read_values_rejects_a_scene_off_the_reference_grid(tmp_path):
    directory = _stack_dir(tmp_path, ["20240116"])
    other = ReferenceGrid(20, 20, GRID.transform, GRID.crs)
    with pytest.raises(ValueError, match="shape"):
        read_values(
            stack_paths(directory, ["20240116"]), other, np.array([0]), np.array([0]), ["ndvi"]
        )


# ── aggregation ────────────────────────────────────────────────────────────


def _values(n_px, n_dates, value=1.0):
    return {m: np.full((n_px, n_dates), value, dtype=np.float32) for m in MEASURES}


def test_class_table_has_a_row_per_class_date_and_measure():
    pixels = select_pixels(_masks(), max_pixels_per_class=5)
    values = _values(len(pixels), 2)
    table = class_table(values, pixels, ["20240116", "20240312"])
    assert len(table) == 3 * 2 * len(MEASURES)


def test_class_table_median_matches_a_hand_computed_value():
    pixels = select_pixels(_masks(deadwood_rows=(0,), living_rows=(), background_rows=()), 5)
    values = _values(len(pixels), 1)
    values["ndvi"][:, 0] = np.arange(len(pixels), dtype=np.float32)
    table = class_table(values, pixels, ["20240116"])
    row = table[(table["class"] == "deadwood") & (table["measure"] == "ndvi")].iloc[0]
    assert row["median"] == pytest.approx(np.median(np.arange(len(pixels))))
    assert row["q25"] == pytest.approx(np.quantile(np.arange(len(pixels)), 0.25))


def test_class_table_counts_only_finite_pixels():
    pixels = select_pixels(_masks(deadwood_rows=(0,), living_rows=(), background_rows=()), 5)
    values = _values(len(pixels), 1)
    values["ndvi"][:3, 0] = np.nan
    table = class_table(values, pixels, ["20240116"])
    row = table[table["measure"] == "ndvi"].iloc[0]
    assert row["n_valid_px"] == len(pixels) - 3


def test_class_table_median_ignores_nan_rather_than_propagating():
    pixels = select_pixels(_masks(deadwood_rows=(0,), living_rows=(), background_rows=()), 5)
    values = _values(len(pixels), 1, value=0.4)
    values["ndvi"][0, 0] = np.nan
    table = class_table(values, pixels, ["20240116"])
    assert table[table["measure"] == "ndvi"].iloc[0]["median"] == pytest.approx(0.4)


def test_tree_table_reports_each_tree_separately():
    pixels = select_pixels(_masks(deadwood_rows=(0, 1)), max_pixels_per_class=5)
    values = _values(len(pixels), 2)
    table = tree_table(values, pixels, ["20240116", "20240312"])
    assert set(table["tree_id"]) == {"tree1", "tree2"}
    assert len(table) == 2 * 2 * len(MEASURES)


def test_tree_table_covers_only_the_deadwood_pixels():
    pixels = select_pixels(_masks(), max_pixels_per_class=5)
    values = _values(len(pixels), 1)
    values["ndvi"][pixels["class"].to_numpy() != "deadwood", 0] = 99.0
    table = tree_table(values, pixels, ["20240116"])
    assert (table[table["measure"] == "ndvi"]["median"] != 99.0).all()


# ── seasons and band signature ─────────────────────────────────────────────


def test_season_of_splits_dry_and_wet():
    assert season_of("20240613", [5, 6, 7, 8, 9], [11, 12, 1, 2, 3]) == "dry"
    assert season_of("20240116", [5, 6, 7, 8, 9], [11, 12, 1, 2, 3]) == "wet"


def test_season_of_calls_april_and_october_transition():
    assert season_of("20240408", [5, 6, 7, 8, 9], [11, 12, 1, 2, 3]) == "transition"
    assert season_of("20241015", [5, 6, 7, 8, 9], [11, 12, 1, 2, 3]) == "transition"


def test_season_of_rejects_overlapping_month_lists():
    with pytest.raises(ValueError, match="overlap"):
        season_of("20240116", [1, 2], [2, 3])


def test_signature_table_averages_each_band_per_class_and_season():
    pixels = select_pixels(_masks(), max_pixels_per_class=5)
    dates = ["20240116", "20240613"]
    values = {b: np.full((len(pixels), 2), 0.3, dtype=np.float32) for b in BAND_NAMES}
    values["NIR"][:, 0] = 0.7
    table = signature_table(values, pixels, dates, [5, 6, 7, 8, 9], [11, 12, 1, 2, 3])
    wet_nir = table[
        (table["season"] == "wet") & (table["band"] == "NIR") & (table["class"] == "deadwood")
    ]
    assert wet_nir["mean"].iloc[0] == pytest.approx(0.7, abs=1e-6)


def test_signature_table_excludes_transition_dates():
    pixels = select_pixels(_masks(), max_pixels_per_class=5)
    dates = ["20240116", "20240408"]
    values = {b: np.full((len(pixels), 2), 0.3, dtype=np.float32) for b in BAND_NAMES}
    values["NIR"][:, 1] = 99.0
    table = signature_table(values, pixels, dates, [5, 6, 7, 8, 9], [11, 12, 1, 2, 3])
    assert set(table["season"]) == {"wet"}
    assert (table["mean"] != 99.0).all()


def test_signature_table_covers_every_band_and_class():
    pixels = select_pixels(_masks(), max_pixels_per_class=5)
    dates = ["20240116", "20240613"]
    values = {b: np.full((len(pixels), 2), 0.3, dtype=np.float32) for b in BAND_NAMES}
    table = signature_table(values, pixels, dates, [5, 6, 7, 8, 9], [11, 12, 1, 2, 3])
    assert len(table) == 3 * 2 * len(BAND_NAMES)
