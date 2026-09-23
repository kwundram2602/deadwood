import os
import sys

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from explore_and_process.rasterize_crowns import (
    rasterize_binary,
    rescale_soft_floor,
    reshape_polygons,
    soft_mask_from_geoms,
)
from utils.nodata import MASK_OUTSIDE, MASK_UNLABELLED

TRANSFORM = from_origin(0, 20, 1, 1)


def test_crown_interior_is_positive():
    mask = soft_mask_from_geoms(
        [box(5, 5, 15, 15)], 20, 20, TRANSFORM, sigma=1.0, nodata_threshold=0.05
    )
    assert mask.dtype == np.float32
    assert mask[10, 10] == 1.0


def test_pixels_beyond_the_bleed_are_unlabelled_not_background():
    # The whole point of the sentinel: with sparse labels, "no polygon here"
    # must not be taught as background.
    mask = soft_mask_from_geoms(
        [box(9, 9, 11, 11)], 20, 20, TRANSFORM, sigma=1.0, nodata_threshold=0.05
    )
    assert mask[0, 0] == MASK_UNLABELLED


def test_the_falloff_produces_no_hard_zero():
    # Beyond 4*sigma the Gaussian is exactly 0, which falls under
    # nodata_threshold and becomes MASK_UNLABELLED — so this function emits no
    # pixel labelled 0.0 at all. Hard negatives come from the digitised
    # background layer via deadwood_scene_mask instead; asserting it here keeps
    # that from being rediscovered the hard way.
    mask = soft_mask_from_geoms(
        [box(9, 9, 11, 11)], 20, 20, TRANSFORM, sigma=1.0, nodata_threshold=0.05
    )
    assert not (mask == 0.0).any()
    falloff = mask[(mask > 0.0) & (mask < 1.0)]
    assert falloff.size > 0
    assert falloff.min() >= 0.05


def test_footprint_marks_outside_pixels():
    footprint = np.ones((20, 20), dtype=bool)
    footprint[:5, :] = False
    mask = soft_mask_from_geoms(
        [box(9, 9, 11, 11)],
        20,
        20,
        TRANSFORM,
        sigma=1.0,
        nodata_threshold=0.05,
        footprint=footprint,
    )
    assert (mask[:5, :] == MASK_OUTSIDE).all()
    assert (mask[5:, :] != MASK_OUTSIDE).all()


def test_lower_nodata_threshold_labels_more_pixels():
    # nodata_threshold decides how much of the Gaussian falloff counts as label
    # rather than "no statement": every pixel under it becomes MASK_UNLABELLED,
    # so lowering it can only grow the labelled region.
    #
    # sigma does NOT have this property — the burned mass is fixed, so a wider
    # blur spreads it thinner and can push the falloff back below the threshold.
    def n_labelled(nodata_threshold):
        mask = soft_mask_from_geoms(
            [box(9, 9, 11, 11)],
            20,
            20,
            TRANSFORM,
            sigma=2.0,
            nodata_threshold=nodata_threshold,
        )
        return int(((mask >= 0.0) & (mask <= 1.0)).sum())

    assert n_labelled(0.01) > n_labelled(0.05) > n_labelled(0.5)


def test_invalid_geometry_is_skipped_with_a_warning(capsys):
    # Confirmed: shapely does call this bowtie invalid.
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    valid = box(5, 5, 15, 15)

    binary = rasterize_binary([bowtie, valid], 20, 20, TRANSFORM)

    assert binary[10, 10] == 1.0
    captured = capsys.readouterr()
    assert "skipped 1 null/invalid geometry" in captured.out


def test_soft_floor_lifts_the_falloff_minimum():
    plain = soft_mask_from_geoms(
        [box(5, 5, 15, 15)], 20, 20, TRANSFORM, sigma=2.0, nodata_threshold=0.05
    )
    floored = soft_mask_from_geoms(
        [box(5, 5, 15, 15)], 20, 20, TRANSFORM, sigma=2.0, nodata_threshold=0.05,
        soft_floor=0.4,
    )
    labelled = (floored >= 0.0) & (floored <= 1.0)
    assert floored[labelled].min() >= 0.4
    assert plain[labelled].min() < 0.4


def test_soft_floor_keeps_the_labelled_region_identical():
    # Only the values move; which pixels carry a label is still decided on the
    # raw blur against nodata_threshold.
    kwargs = dict(sigma=2.0, nodata_threshold=0.05)
    plain = soft_mask_from_geoms([box(9, 9, 11, 11)], 20, 20, TRANSFORM, **kwargs)
    floored = soft_mask_from_geoms(
        [box(9, 9, 11, 11)], 20, 20, TRANSFORM, soft_floor=0.6, **kwargs
    )
    np.testing.assert_array_equal(
        plain == MASK_UNLABELLED, floored == MASK_UNLABELLED
    )


def test_soft_floor_keeps_the_interior_at_one():
    mask = soft_mask_from_geoms(
        [box(5, 5, 15, 15)], 20, 20, TRANSFORM, sigma=1.0, nodata_threshold=0.05,
        soft_floor=0.6,
    )
    assert mask[10, 10] == 1.0


def test_soft_floor_preserves_the_gradient_order():
    mask = soft_mask_from_geoms(
        [box(5, 5, 15, 15)], 20, 20, TRANSFORM, sigma=2.0, nodata_threshold=0.05,
        soft_floor=0.4,
    )
    # Walking outward from the centre along a row, the label may never rise
    row = mask[10, 10:]
    row = row[(row >= 0.0) & (row <= 1.0)]
    assert np.all(np.diff(row) <= 1e-6)


def test_soft_floor_none_is_identity():
    soft = np.array([0.05, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(rescale_soft_floor(soft, 0.05, None), soft)


def test_rescale_maps_endpoints_exactly():
    soft = np.array([0.05, 1.0], dtype=np.float32)
    out = rescale_soft_floor(soft, 0.05, 0.4)
    np.testing.assert_allclose(out, [0.4, 1.0], atol=1e-6)


def test_rescale_is_linear_in_between():
    # midpoint of [nodata_threshold, 1] lands on midpoint of [floor, 1]
    soft = np.array([0.525], dtype=np.float32)
    out = rescale_soft_floor(soft, 0.05, 0.4)
    np.testing.assert_allclose(out, [0.7], atol=1e-6)


def test_rescale_clips_values_below_the_threshold_to_the_floor():
    # a polygon narrower than its sigma peaks below nodata_threshold
    soft = np.array([0.0, 0.01], dtype=np.float32)
    out = rescale_soft_floor(soft, 0.05, 0.4)
    np.testing.assert_allclose(out, [0.4, 0.4], atol=1e-6)


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_rescale_rejects_out_of_range_floor(bad):
    with pytest.raises(ValueError, match="soft_floor"):
        rescale_soft_floor(np.array([0.5], dtype=np.float32), 0.05, bad)


# ---------------------------------------------------------------------------
# reshape_polygons: erode + rounding applied before the polygons are burned
# ---------------------------------------------------------------------------

SQUARE = box(0, 0, 10, 10)
# A 0.2 m wide spike sticking 4 m out of the square's right edge — the kind of
# artefact a stray vertex leaves in a hand-digitised crown.
SPIKED = SQUARE.union(Polygon([(10, 4.9), (14, 5.0), (10, 5.1)]))
# Two blobs joined by a 0.2 m wide bridge: rounding pinches it in two.
DUMBBELL = box(0, 0, 4, 4).union(box(6, 0, 10, 4)).union(box(4, 1.9, 6, 2.1))


def _series(*geoms):
    return gpd.GeoSeries(list(geoms))


def test_erode_shrinks_the_area_by_the_given_distance():
    out = reshape_polygons(_series(SQUARE), erode=1.0)
    # inward offset of a square is a square: 10x10 -> 8x8
    assert out.iloc[0].area == pytest.approx(64.0, abs=1e-6)


def test_zero_parameters_leave_the_geometry_untouched():
    out = reshape_polygons(_series(SQUARE), erode=0.0, round_radius=0.0)
    assert out.iloc[0].equals(SQUARE)


def test_round_radius_cuts_a_thin_spike():
    spike_tip = Point(13.5, 5.0)
    assert SPIKED.covers(spike_tip)
    out = reshape_polygons(_series(SPIKED), round_radius=0.5)
    assert not out.iloc[0].covers(spike_tip)


def test_round_radius_replaces_the_sharp_corner_with_an_arc():
    out = reshape_polygons(_series(SQUARE), round_radius=1.0)
    rounded = out.iloc[0]
    assert not rounded.covers(Point(0, 0))  # corner cut away
    # a square has 5 coords (closing point); an arc needs many more
    assert len(rounded.exterior.coords) > 5


def test_round_radius_roughly_preserves_the_area():
    # opening + closing trades the four corners for quarter circles:
    # 100 - (4 - pi) * r^2 for r = 1
    out = reshape_polygons(_series(SQUARE), round_radius=1.0)
    assert out.iloc[0].area == pytest.approx(100.0 - (4 - np.pi), abs=0.05)


def test_erode_and_round_compose():
    eroded_only = reshape_polygons(_series(SQUARE), erode=1.0).iloc[0]
    both = reshape_polygons(_series(SQUARE), erode=1.0, round_radius=0.5).iloc[0]
    assert both.area < eroded_only.area
    assert not both.covers(Point(1, 1))  # the eroded square's own corner


def test_vanished_polygon_is_dropped_with_a_warning(capsys):
    tiny = box(0, 0, 0.5, 0.5)
    out = reshape_polygons(_series(SQUARE, tiny), erode=1.0)
    assert len(out) == 1
    assert out.index.tolist() == [0]
    captured = capsys.readouterr()
    assert "1 of 2" in captured.out
    assert "vanished" in captured.out


def test_rounding_may_split_a_polygon_and_keeps_both_parts():
    out = reshape_polygons(_series(DUMBBELL), round_radius=0.5)
    assert len(out) == 1
    parts = out.iloc[0].geoms
    assert len(list(parts)) == 2


def test_invalid_geometry_passes_through_untouched():
    # buffering a bowtie yields garbage; rasterize_binary reports and skips it
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    out = reshape_polygons(_series(bowtie), erode=0.1, round_radius=0.1)
    assert out.iloc[0].equals_exact(bowtie, tolerance=0.0)


def test_index_travels_with_the_geometries():
    series = gpd.GeoSeries([SQUARE, box(20, 20, 30, 30)], index=[7, 9])
    out = reshape_polygons(series, erode=1.0)
    assert out.index.tolist() == [7, 9]


@pytest.mark.parametrize("kwargs", [{"erode": -0.1}, {"round_radius": -0.1}])
def test_negative_distance_raises(kwargs):
    with pytest.raises(ValueError, match="must be >= 0"):
        reshape_polygons(_series(SQUARE), **kwargs)
