import os
import sys

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.deadwood_patches import deadwood_scene_mask
from utils.nodata import MASK_OUTSIDE, MASK_UNLABELLED, valid_target

# 1 m pixels, origin at the top-left corner (0, 100): row = 100 - y, col = x.
TRANSFORM = from_origin(0, 100, 1, 1)
KW = dict(sigma_pos=1.0, sigma_neg=4.0, pos_threshold=0.05, neg_threshold=0.9)


def test_positives_win_where_the_classes_overlap():
    # Same polygon in both layers. Application order is the contract: negatives
    # are laid down first and positives overwrite them, never the reverse.
    crown = box(10, 80, 20, 90)
    mask = deadwood_scene_mask([crown], [crown], 100, 100, TRANSFORM, **KW)
    assert mask[15, 15] == 1.0


def test_background_core_is_zero_but_its_border_is_unlabelled():
    # neg_threshold erodes the background inward: only the confident core counts
    # as 0.0, so a pixel one step inside the drawn edge says nothing at all.
    mask = deadwood_scene_mask([], [box(10, 60, 30, 80)], 100, 100, TRANSFORM,
                               sigma_pos=1.0, sigma_neg=2.0,
                               pos_threshold=0.05, neg_threshold=0.9)
    assert mask[30, 20] == 0.0
    assert mask[21, 20] == MASK_UNLABELLED


def test_touching_opposite_polygons_still_leave_an_unlabelled_band():
    # The guarantee the whole asymmetric-sigma decision exists for: even drawn
    # edge-to-edge, a hard 1.0 never lands next to a hard 0.0.
    mask = deadwood_scene_mask([box(0, 50, 20, 70)], [box(20, 50, 60, 70)],
                               100, 100, TRANSFORM, **KW)
    assert (mask[40, 22:25] == MASK_UNLABELLED).all()


def test_a_shared_sigma_would_let_the_classes_collide():
    # Why sigma_neg must exceed sigma_pos. Counted strictly between the two
    # polygons; beyond them everything is unlabelled for the ordinary reason.
    def band_px(sigma_neg):
        mask = deadwood_scene_mask([box(0, 50, 20, 70)], [box(20, 50, 60, 70)],
                                   100, 100, TRANSFORM, sigma_pos=1.0, sigma_neg=sigma_neg,
                                   pos_threshold=0.05, neg_threshold=0.9)
        return int((mask[40, 20:31] == MASK_UNLABELLED).sum())

    assert band_px(1.0) == 0
    assert band_px(2.0) == 1
    assert band_px(4.0) == 3


def test_a_crown_smaller_than_its_blur_is_still_labelled():
    # A 1x1 m crown under sigma=5 blurs to a peak of ~0.006, far below
    # pos_threshold. The binary raster is what keeps it a label.
    mask = deadwood_scene_mask([box(50, 50, 51, 51)], [], 100, 100, TRANSFORM,
                               sigma_pos=5.0, sigma_neg=4.0,
                               pos_threshold=0.05, neg_threshold=0.9)
    assert valid_target(mask[49, 50])
    assert 0.0 < mask[49, 50] < 0.05


def test_outside_the_footprint_overwrites_every_label():
    # MASK_OUTSIDE is applied last: the drone never saw this ground, so no
    # label of either class may survive there.
    footprint = np.ones((100, 100), dtype=bool)
    footprint[:20, :] = False
    mask = deadwood_scene_mask([box(0, 85, 100, 95)], [], 100, 100, TRANSFORM,
                               footprint=footprint, **KW)
    assert (mask[:20, :] == MASK_OUTSIDE).all()
    assert (mask[20:, :] != MASK_OUTSIDE).all()


def test_either_class_may_be_absent():
    mask = deadwood_scene_mask([box(10, 80, 20, 90)], [], 100, 100, TRANSFORM, **KW)
    assert mask[15, 15] == 1.0
    assert not (mask == 0.0).any()
