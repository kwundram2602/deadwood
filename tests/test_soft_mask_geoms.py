import os
import sys

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.rasterize_crowns import soft_mask_from_geoms
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
