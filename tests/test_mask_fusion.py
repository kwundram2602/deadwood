import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mask_fusion.codes import (  # noqa: E402
    COMBO_COLORS,
    COMBO_NAMES,
    N_CODES,
    NODATA,
    combo_codes,
    height_class,
)

H1, H2 = 1.0, 3.0


def test_combo_names_and_colors_cover_every_code():
    assert len(COMBO_NAMES) == N_CODES == 12
    assert len(set(COMBO_NAMES)) == N_CODES
    assert len(COMBO_COLORS) == N_CODES
    assert COMBO_NAMES[3] == "dead_crown_low"
    assert COMBO_NAMES[10] == "dead_tall"
    assert COMBO_NAMES[11] == "dead_crown_tall"


def test_height_class_boundaries():
    # Negative heights are ground noise, not an error: they are low.
    h = np.array([-0.3, 0.0, 0.999, 1.0, 2.999, 3.0, 15.1], dtype=np.float32)
    np.testing.assert_array_equal(height_class(h, H1, H2), [0, 0, 0, 1, 1, 2, 2])
    assert height_class(h, H1, H2).dtype == np.uint8


@pytest.mark.parametrize("h1,h2", [(0.0, 3.0), (-1.0, 2.0), (3.0, 3.0), (3.0, 1.0)])
def test_height_class_rejects_bad_thresholds(h1, h2):
    with pytest.raises(ValueError, match="0 < h1 < h2"):
        height_class(np.zeros(1, np.float32), h1, h2)


def test_combo_codes_truth_table():
    crown = np.array([0, 1, 0, 1] * 3, np.uint8)
    dead = np.array([0, 0, 1, 1] * 3, np.uint8)
    ndsm = np.repeat(np.array([0.5, 2.0, 5.0], np.float32), 4)
    out = combo_codes(crown, dead, ndsm, H1, H2)
    np.testing.assert_array_equal(out, np.arange(12))
    assert out.dtype == np.uint8


def test_combo_codes_keeps_2d_shape():
    crown = np.ones((2, 3), np.uint8)
    dead = np.zeros((2, 3), np.uint8)
    ndsm = np.full((2, 3), 5.0, np.float32)
    out = combo_codes(crown, dead, ndsm, H1, H2)
    assert out.shape == (2, 3)
    assert (out == 9).all()


@pytest.mark.parametrize("which", ["crown", "deadwood", "ndsm"])
def test_nodata_from_each_input(which):
    arrays = {
        "crown": np.array([1, 1], np.uint8),
        "deadwood": np.array([1, 1], np.uint8),
        "ndsm": np.array([5.0, 5.0], np.float32),
    }
    arrays[which][0] = np.nan if which == "ndsm" else NODATA
    out = combo_codes(arrays["crown"], arrays["deadwood"], arrays["ndsm"], H1, H2)
    np.testing.assert_array_equal(out, [NODATA, 11])


def test_numeric_ndsm_nodata_is_nodata_not_low():
    zeros = np.zeros(2, np.uint8)
    ndsm = np.array([-9999.0, 0.5], np.float32)
    out = combo_codes(zeros, zeros, ndsm, H1, H2, ndsm_nodata=-9999.0)
    np.testing.assert_array_equal(out, [NODATA, 0])


def test_nan_ndsm_nodata_value_is_harmless():
    zeros = np.zeros(2, np.uint8)
    ndsm = np.array([np.nan, 0.5], np.float32)
    out = combo_codes(zeros, zeros, ndsm, H1, H2, ndsm_nodata=float("nan"))
    np.testing.assert_array_equal(out, [NODATA, 0])


@pytest.mark.parametrize("which", ["crown", "deadwood"])
def test_mask_with_probabilities_raises(which):
    arrays = {"crown": np.zeros(2, np.uint8), "deadwood": np.zeros(2, np.uint8)}
    arrays[which] = np.array([0, 128], np.uint8)
    with pytest.raises(ValueError, match="probability raster"):
        combo_codes(arrays["crown"], arrays["deadwood"], np.zeros(2, np.float32), H1, H2)
