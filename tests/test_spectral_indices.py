import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.indices import (  # noqa: E402
    BAND_NAMES,
    INDEX_NAMES,
    compute_indices,
    normalized_difference,
)


def _stack(r=0.1, g=0.2, b=0.15, green=0.2, red=0.1, rededge=0.3, nir=0.5):
    values = [r, g, b, green, red, rededge, nir]
    return np.array(values, dtype=np.float32).reshape(7, 1, 1)


def test_normalized_difference_known_value():
    a = np.array([0.5], dtype=np.float32)
    b = np.array([0.1], dtype=np.float32)
    assert normalized_difference(a, b)[0] == pytest.approx(0.4 / 0.6, abs=1e-6)


def test_normalized_difference_zero_sum_is_nan_not_inf():
    a = np.array([0.0], dtype=np.float32)
    b = np.array([0.0], dtype=np.float32)
    out = normalized_difference(a, b)
    assert np.isnan(out[0])
    assert not np.isinf(out[0])


def test_ndvi_uses_nir_and_red():
    out = compute_indices(_stack(nir=0.5, red=0.1), BAND_NAMES)
    assert out["ndvi"][0, 0] == pytest.approx(0.4 / 0.6, abs=1e-6)


def test_ndre_uses_nir_and_rededge():
    out = compute_indices(_stack(nir=0.5, rededge=0.3), BAND_NAMES)
    assert out["ndre"][0, 0] == pytest.approx(0.2 / 0.8, abs=1e-6)


def test_gndvi_uses_nir_and_ms_green():
    out = compute_indices(_stack(nir=0.5, green=0.2), BAND_NAMES)
    assert out["gndvi"][0, 0] == pytest.approx(0.3 / 0.7, abs=1e-6)


def test_brightness_averages_rgb_composite_only():
    out = compute_indices(_stack(r=0.1, g=0.2, b=0.3, green=0.9), BAND_NAMES)
    assert out["brightness"][0, 0] == pytest.approx(0.2, abs=1e-6)


def test_green_red_ratio_uses_rgb_composite():
    out = compute_indices(_stack(r=0.1, g=0.2), BAND_NAMES)
    assert out["green_red"][0, 0] == pytest.approx(0.1 / 0.3, abs=1e-6)


def test_nir_red_ratio():
    out = compute_indices(_stack(nir=0.5, red=0.1), BAND_NAMES)
    assert out["nir_red_ratio"][0, 0] == pytest.approx(5.0, abs=1e-5)


def test_nir_red_ratio_zero_denominator_is_nan():
    out = compute_indices(_stack(nir=0.5, red=0.0), BAND_NAMES)
    assert np.isnan(out["nir_red_ratio"][0, 0])


def test_all_declared_indices_are_produced():
    out = compute_indices(_stack(), BAND_NAMES)
    assert set(out) == set(INDEX_NAMES)


def test_nan_input_propagates():
    stack = _stack()
    stack[6, 0, 0] = np.nan
    out = compute_indices(stack, BAND_NAMES)
    assert np.isnan(out["ndvi"][0, 0])


def test_missing_band_raises():
    with pytest.raises(ValueError, match="NIR"):
        compute_indices(_stack()[:6], BAND_NAMES[:6])
