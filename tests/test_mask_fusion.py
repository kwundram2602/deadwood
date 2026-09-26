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
    EcoClass,
    build_lookup,
    combo_codes,
    height_class,
    parse_classes,
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


def _classes():
    return {
        1: {"name": "living", "color": "#00ff00", "codes": [1, 5, 9]},
        2: {"name": "dead", "color": "#ff0000", "codes": [2, 3, 6, 7, 10, 11]},
        3: {"name": "rest", "color": "#808080", "codes": [0, 4, 8]},
    }


def test_parse_classes_sorted_with_rgb():
    eco = parse_classes(dict(reversed(list(_classes().items()))))
    assert [e.value for e in eco] == [1, 2, 3]
    assert eco[1] == EcoClass(2, "dead", (255, 0, 0), (2, 3, 6, 7, 10, 11))


def test_parse_classes_accepts_string_keys():
    eco = parse_classes({str(k): v for k, v in _classes().items()})
    assert [e.value for e in eco] == [1, 2, 3]


def test_missing_code_raises():
    classes = _classes()
    classes[3]["codes"] = [0, 4]
    with pytest.raises(ValueError, match="missing"):
        parse_classes(classes)


def test_duplicate_code_raises():
    classes = _classes()
    classes[1]["codes"] = [0, 1, 5, 9]
    with pytest.raises(ValueError, match="more than once"):
        parse_classes(classes)


def test_unknown_code_raises():
    classes = _classes()
    classes[3]["codes"] = [0, 4, 8, 12]
    with pytest.raises(ValueError, match="unknown"):
        parse_classes(classes)


@pytest.mark.parametrize("value", [0, 255])
def test_reserved_eco_value_raises(value):
    classes = _classes()
    classes[value] = classes.pop(3)
    with pytest.raises(ValueError, match=r"1\.\.254"):
        parse_classes(classes)


def test_duplicate_name_raises():
    classes = _classes()
    classes[3]["name"] = "living"
    with pytest.raises(ValueError, match="duplicate class name"):
        parse_classes(classes)


def test_bad_color_raises():
    classes = _classes()
    classes[1]["color"] = "green"
    with pytest.raises(ValueError, match="#rrggbb"):
        parse_classes(classes)


def test_build_lookup_maps_every_code_and_keeps_nodata():
    lut = build_lookup(parse_classes(_classes()))
    assert lut.dtype == np.uint8
    assert lut.shape == (256,)
    np.testing.assert_array_equal(lut[:N_CODES], [3, 1, 2, 2, 3, 1, 2, 2, 3, 1, 2, 2])
    assert lut[NODATA] == NODATA
    assert (lut[N_CODES:] == NODATA).all()
