import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.field_eval import (
    evaluate_polygons,
    plot_detection,
    plot_threshold_comparison,
    summarize,
    threshold_table,
)

# 1 m pixels, origin top-left at (0, 10): x -> col, (10 - y) -> row.
TRANSFORM = from_origin(0, 10, 1, 1)
SHAPE = (10, 10)
CRS = "EPSG:32736"


def _write_mask(path, mask):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SHAPE[1],
        height=SHAPE[0],
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=TRANSFORM,
        nodata=255,
    ) as dst:
        dst.write(mask, 1)


def _write_polygons(path, crs=CRS):
    gdf = gpd.GeoDataFrame(
        {"tree_id": ["a", "b", "c"]},
        geometry=[box(0, 8, 2, 10), box(4, 4, 6, 6), box(8, 0, 9, 1)],
        crs=CRS,
    )
    gdf.to_crs(crs).to_file(path, layer="polys", driver="GPKG")


def _mask():
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[0:2, 0:2] = 1  # polygon a fully covered
    mask[4, 4] = 1  # polygon b: 1 of 4
    mask[9, 8] = 255  # polygon c: nodata only
    return mask


def test_evaluate_polygons_scores_each_polygon(tmp_path):
    _write_mask(tmp_path / "m.tif", _mask())
    _write_polygons(tmp_path / "p.gpkg")
    df = evaluate_polygons(tmp_path / "m.tif", tmp_path / "p.gpkg")
    assert list(df["tree_id"]) == ["a", "b", "c"]
    assert np.allclose(df["area_m2"], [4.0, 4.0, 1.0])
    assert np.isclose(df["covered_fraction"].iloc[0], 1.0)
    assert np.isclose(df["covered_fraction"].iloc[1], 0.25)
    assert np.isnan(df["covered_fraction"].iloc[2])


def test_evaluate_polygons_reprojects_to_the_mask_crs(tmp_path):
    _write_mask(tmp_path / "m.tif", _mask())
    _write_polygons(tmp_path / "p.gpkg", crs="EPSG:4326")
    df = evaluate_polygons(tmp_path / "m.tif", tmp_path / "p.gpkg")
    assert np.isclose(df["covered_fraction"].iloc[0], 1.0)


def test_summarize_skips_unscored_polygons():
    per_polygon = pd.DataFrame(
        {"covered_fraction": [1.0, 0.25, np.nan], "area_m2": [4.0, 4.0, 1.0]}
    )
    s = summarize(per_polygon, [0.5, 0.2])
    assert s["n_polygons"] == 3
    assert s["n_evaluated"] == 2
    assert np.isclose(s["mean_covered_fraction"], 0.625)
    assert np.isclose(s["median_covered_fraction"], 0.625)
    assert np.isclose(s["area_weighted_covered_fraction"], 0.625)
    assert s["detected_ge_0.5"] == 1
    assert np.isclose(s["detected_rate_ge_0.5"], 0.5)
    assert s["detected_ge_0.2"] == 2


def test_area_weighting_favours_large_polygons():
    per_polygon = pd.DataFrame({"covered_fraction": [1.0, 0.0], "area_m2": [3.0, 1.0]})
    assert np.isclose(summarize(per_polygon, [0.5])["area_weighted_covered_fraction"], 0.75)


def test_plot_detection_writes_png(tmp_path):
    results = {
        "crown": pd.DataFrame({"covered_fraction": [0.9, 0.4, 0.7]}),
        "deadwood": pd.DataFrame({"covered_fraction": [0.1, np.nan, 0.6]}),
    }
    out = plot_detection(results, [0.5], tmp_path / "det.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_detection_refuses_more_series_than_colors(tmp_path):
    results = {f"s{i}": pd.DataFrame({"covered_fraction": [0.5]}) for i in range(5)}
    with pytest.raises(ValueError, match="at most 4"):
        plot_detection(results, [0.5], tmp_path / "det.png")


def _summary(rates):
    return pd.DataFrame(
        {
            "detected_rate_ge_0.5": rates + [0.9],
            "mean_covered_fraction": rates + [0.8],
        },
        index=pd.Index(["crown_t0.9", "crown_t0.6", "crown_t0.8", "deadwood"], name="target"),
    )


def test_threshold_table_keeps_thresholded_targets_sorted():
    table = threshold_table(_summary([0.4, 0.7, 0.6]))
    assert list(table.index) == [0.6, 0.8, 0.9]
    assert list(table["detected_rate_ge_0.5"]) == [0.7, 0.6, 0.4]


def test_plot_threshold_comparison_writes_png(tmp_path):
    summaries = {"a": _summary([0.4, 0.7, 0.6]), "b": _summary([0.3, 0.5, 0.4])}
    out = plot_threshold_comparison(summaries, 0.5, tmp_path / "cmp.png")
    assert out.exists() and out.stat().st_size > 0
