# tests/test_apply_dsm_mask.py
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from explore_and_process.apply_dsm_mask import apply_soft_blend


def _arr(*values):
    return np.array(values, dtype=np.float32)


def test_crown_pixel_multiplied_by_one_minus_conf():
    mask = _arr(0.8)
    conf = _arr(0.9)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.8 * 0.1, abs=1e-6)


def test_crown_pixel_zero_conf_unchanged():
    mask = _arr(0.8)
    conf = _arr(0.0)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.8, abs=1e-6)


def test_crown_pixel_full_conf_becomes_zero():
    mask = _arr(0.6)
    conf = _arr(1.0)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_nodata_resolved_when_conf_above_threshold():
    mask = _arr(255.0)
    conf = _arr(0.8)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(1.0 - 0.8, abs=1e-6)


def test_nodata_stays_255_when_conf_below_threshold():
    mask = _arr(255.0)
    conf = _arr(0.5)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(255.0, abs=1e-6)


def test_nodata_stays_255_when_conf_exactly_at_threshold():
    # threshold is strict: must be *greater than*, not equal
    mask = _arr(255.0)
    conf = _arr(0.7)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(255.0, abs=1e-6)


def test_existing_zero_crown_pixel_stays_zero():
    mask = _arr(0.0)
    conf = _arr(0.9)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_mixed_array():
    mask = _arr(0.8, 255.0, 255.0, 0.0)
    conf = _arr(0.5, 0.9,   0.4,   0.8)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.8 * 0.5, abs=1e-6)   # crown dampened
    assert result[1] == pytest.approx(1.0 - 0.9, abs=1e-6)   # noData resolved
    assert result[2] == pytest.approx(255.0, abs=1e-6)        # noData stays
    assert result[3] == pytest.approx(0.0, abs=1e-6)          # zero stays zero
