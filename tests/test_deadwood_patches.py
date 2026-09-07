import os
import sys

import numpy as np
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.deadwood_patches import (
    array_window,
    crown_crop_window,
    ring_negatives,
)
from utils.nodata import MASK_OUTSIDE, MASK_UNLABELLED

TRANSFORM = from_origin(0, 100, 1, 1)


def test_crop_window_is_centred_on_the_crown():
    win = crown_crop_window(box(9, 89, 11, 91), TRANSFORM, crop_size=4)
    assert (win.width, win.height) == (4, 4)
    assert (win.col_off, win.row_off) == (8, 8)


def test_jitter_offsets_the_window():
    win = crown_crop_window(box(9, 89, 11, 91), TRANSFORM, crop_size=4, jitter=(3, -2))
    assert (win.row_off, win.col_off) == (11, 6)


def test_array_window_pads_beyond_the_edge_with_fill():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = array_window(arr, Window(col_off=-1, row_off=-1, width=3, height=3), fill=-1.0)
    assert out.shape == (3, 3)
    assert (out[0, :] == -1.0).all()
    assert (out[:, 0] == -1.0).all()
    assert out[1, 1] == 0.0 and out[2, 2] == 5.0


def test_array_window_inside_the_array_is_a_plain_slice():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = array_window(arr, Window(col_off=1, row_off=1, width=2, height=2), fill=-1.0)
    assert out.tolist() == [[5.0, 6.0], [9.0, 10.0]]


def test_ring_negatives_labels_near_pixels_and_leaves_far_ones_unlabelled():
    mask = np.full((100, 100), MASK_UNLABELLED, dtype=np.float32)
    mask[49:51, 49:51] = 1.0  # the crown itself, already labelled
    out = ring_negatives(mask, [box(49, 49, 51, 51)], TRANSFORM, buffer_m=3.0)
    assert out[50, 52] == 0.0  # inside the 3 m buffer
    assert out[50, 60] == MASK_UNLABELLED  # beyond it — sparse labels stay sparse
    assert out[50, 50] == 1.0  # the crown is not overwritten


def test_ring_negatives_never_overwrites_an_existing_label():
    mask = np.full((100, 100), MASK_UNLABELLED, dtype=np.float32)
    mask[50, 50] = 0.4  # soft falloff value
    mask[50, 51] = MASK_OUTSIDE  # off-footprint
    out = ring_negatives(mask, [box(49, 49, 51, 51)], TRANSFORM, buffer_m=3.0)
    assert out[50, 50] == 0.4
    assert out[50, 51] == MASK_OUTSIDE
