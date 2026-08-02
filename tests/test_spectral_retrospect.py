import os
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.retrospect import first_dead_cycle  # noqa: E402

CRS = rasterio.crs.CRS.from_epsg(32736)


def _objects():
    return gpd.GeoDataFrame(
        {"object_id": [1, 2], "geometry": [box(0, 0, 1, 1), box(2, 2, 3, 3)]},
        crs=CRS,
    )


def _labels():
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = 1
    labels[2, 2] = 2
    return labels


def _cycle(object_1_dead, object_2_dead):
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = object_1_dead
    mask[2, 2] = object_2_dead
    return mask


def test_first_dead_cycle_picks_the_earliest_cycle():
    masks = {
        "2023_24": _cycle(False, True),
        "2024_25": _cycle(True, True),
        "2025_26": _cycle(True, True),
    }
    result = first_dead_cycle(masks, _objects(), _labels()).set_index("object_id")
    assert result.loc[1, "first_dead_cycle"] == "2024_25"
    assert result.loc[2, "first_dead_cycle"] == "2023_24"


def test_cycles_dead_counts_all_positive_cycles():
    masks = {"a": _cycle(True, False), "b": _cycle(False, False), "c": _cycle(True, True)}
    result = first_dead_cycle(masks, _objects(), _labels()).set_index("object_id")
    assert result.loc[1, "cycles_dead"] == 2
    assert result.loc[2, "cycles_dead"] == 1


def test_object_never_dead_gets_null_first_cycle():
    masks = {"a": _cycle(False, False), "b": _cycle(False, True)}
    result = first_dead_cycle(masks, _objects(), _labels()).set_index("object_id")
    assert result.loc[1, "first_dead_cycle"] is None
    assert result.loc[1, "cycles_dead"] == 0


def test_cycles_are_evaluated_in_sorted_key_order():
    masks = {"2025_26": _cycle(True, True), "2023_24": _cycle(True, True)}
    result = first_dead_cycle(masks, _objects(), _labels()).set_index("object_id")
    assert result.loc[1, "first_dead_cycle"] == "2023_24"


def _validity(object_1_valid, object_2_valid):
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = object_1_valid
    valid[2, 2] = object_2_valid
    return valid


def test_low_coverage_in_first_dead_cycle_is_flagged():
    # Object 1 is reported dead in "a" (its footprint pixel is True there),
    # but that pixel was actually nodata in "a" (validity False), so the
    # "dead" verdict rests on no real observation and must be flagged.
    masks = {"a": _cycle(True, True), "b": _cycle(True, True)}
    validity_masks = {"a": _validity(False, True), "b": _validity(True, True)}
    result = first_dead_cycle(
        masks, _objects(), _labels(), validity_masks=validity_masks
    ).set_index("object_id")

    assert result.loc[1, "first_dead_cycle"] == "a"
    assert result.loc[1, "first_dead_cycle_coverage"] == pytest.approx(0.0)
    assert bool(result.loc[1, "low_confidence"]) is True

    # Object 2's footprint was fully observed in "a" — not flagged.
    assert result.loc[2, "first_dead_cycle_coverage"] == pytest.approx(1.0)
    assert bool(result.loc[2, "low_confidence"]) is False


def test_no_validity_masks_leaves_coverage_unknown_not_flagged():
    masks = {"a": _cycle(True, True)}
    result = first_dead_cycle(masks, _objects(), _labels()).set_index("object_id")
    assert np.isnan(result.loc[1, "first_dead_cycle_coverage"])
    assert bool(result.loc[1, "low_confidence"]) is False
