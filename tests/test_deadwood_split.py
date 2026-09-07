import os
import sys

import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.deadwood_patches import split_crowns


def _line_of_crowns(n=9):
    # Strung out along +x so the principal axis is unambiguous.
    return gpd.GeoDataFrame(
        geometry=[Point(float(i), 0.0).buffer(0.2) for i in range(n)],
        index=list(range(100, 100 + n)),
        crs="EPSG:32736",
    )


def test_spatial_split_is_contiguous_along_the_principal_axis():
    splits = split_crowns(_line_of_crowns(), 5, 2, 2, mode="spatial")
    assert splits["train"] == [100, 101, 102, 103, 104]
    assert splits["val"] == [105, 106]
    assert splits["test"] == [107, 108]


def test_spatial_split_is_deterministic_regardless_of_row_order():
    crowns = _line_of_crowns()
    shuffled = crowns.iloc[[4, 0, 8, 2, 6, 1, 7, 3, 5]]
    assert split_crowns(crowns, 5, 2, 2) == split_crowns(shuffled, 5, 2, 2)


def test_counts_must_match_the_crown_total():
    import pytest

    with pytest.raises(ValueError, match="19"):
        split_crowns(_line_of_crowns(), 19, 6, 6, mode="spatial")


def test_random_split_is_seed_stable_and_covers_every_crown():
    a = split_crowns(_line_of_crowns(), 5, 2, 2, mode="random", seed=7)
    b = split_crowns(_line_of_crowns(), 5, 2, 2, mode="random", seed=7)
    assert a == b
    assert sorted(a["train"] + a["val"] + a["test"]) == list(range(100, 109))
