import os
import sys

import numpy as np
from rasterio.windows import Window

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.deadwood_patches import array_window


def test_array_window_pads_beyond_the_edge_with_fill():
    # The scene mask lives in RAM, so grid tiles that overhang the array need
    # the boundless read rasterio gives datasets for free.
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


def test_array_window_entirely_outside_the_array_is_all_fill():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = array_window(arr, Window(col_off=99, row_off=99, width=2, height=2), fill=-1.0)
    assert (out == -1.0).all()
