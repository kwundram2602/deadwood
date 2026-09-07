import os
import sys

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.crown_recall import crown_coverage, crown_pixel_index, threshold_sweep

# 1 m pixels, origin top-left at (0, 10): x -> col, (10 - y) -> row.
TRANSFORM = from_origin(0, 10, 1, 1)
SHAPE = (10, 10)


def _crowns():
    return gpd.GeoDataFrame(
        geometry=[box(2, 6, 4, 8), box(6, 2, 7, 3)],
        index=[21, 58],
        crs="EPSG:32736",
    )


def test_pixel_index_maps_geometry_to_grid():
    index = crown_pixel_index(_crowns(), TRANSFORM, SHAPE)
    assert set(index) == {21, 58}
    rows, cols = index[21]
    assert len(rows) == 4  # 2x2 m polygon on a 1 m grid
    assert sorted(set(rows.tolist())) == [2, 3]
    assert sorted(set(cols.tolist())) == [2, 3]
    rows58, cols58 = index[58]
    assert len(rows58) == 1
    assert (rows58[0], cols58[0]) == (7, 6)


def test_coverage_counts_hit_pixels_per_crown():
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[2, 2] = 1
    mask[2, 3] = 1  # 2 of crown 21's 4 pixels
    df = crown_coverage(mask, crown_pixel_index(_crowns(), TRANSFORM, SHAPE))
    assert df.loc[21, "n_px"] == 4
    assert df.loc[21, "covered_px"] == 2
    assert np.isclose(df.loc[21, "covered_fraction"], 0.5)
    assert bool(df.loc[21, "hit"])
    assert df.loc[58, "covered_px"] == 0
    assert not bool(df.loc[58, "hit"])


def test_hit_uses_the_min_overlap_threshold():
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[2, 2] = 1  # 1 of 4 = 0.25
    index = crown_pixel_index(_crowns(), TRANSFORM, SHAPE)
    assert bool(crown_coverage(mask, index, min_overlap=0.1).loc[21, "hit"])
    assert not bool(crown_coverage(mask, index, min_overlap=0.5).loc[21, "hit"])


def test_sweep_recall_falls_as_the_threshold_rises():
    probs = np.zeros(SHAPE, dtype=np.float32)
    probs[2, 2:4] = 0.8
    probs[7, 6] = 0.3
    df = threshold_sweep(probs, crown_pixel_index(_crowns(), TRANSFORM, SHAPE), [0.2, 0.5, 0.9])
    assert list(df["threshold"]) == [0.2, 0.5, 0.9]
    assert list(df["n_hit"]) == [2, 1, 0]
    assert (df["n_crowns"] == 2).all()
