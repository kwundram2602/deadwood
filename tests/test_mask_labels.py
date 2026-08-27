import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.apply_dsm_mask import (
    _smoothstep_confidence,
    apply_soft_blend,
    height_data_valid,
)

NODATA = 255.0


# ── _smoothstep_confidence ───────────────────────────────────────────────────


def test_smoothstep_default_ramp_falls_to_zero_at_double_threshold():
    ndsm = np.array([0.0, 1.2, 1.8, 2.4, 5.0], dtype=np.float32)
    conf = _smoothstep_confidence(ndsm, threshold=1.2)
    assert conf[0] == 1.0
    assert conf[1] == 1.0
    assert 0.0 < conf[2] < 1.0
    assert conf[3] == 0.0
    assert conf[4] == 0.0


def test_smoothstep_custom_ramp_narrows_transition():
    ndsm = np.array([1.2, 1.4, 1.6, 2.0], dtype=np.float32)
    conf = _smoothstep_confidence(ndsm, threshold=1.2, ramp=0.4)
    assert conf[0] == 1.0
    assert np.isclose(conf[1], 0.5, atol=1e-5)  # smoothstep midpoint
    assert np.isclose(conf[2], 0.0, atol=1e-5)
    assert conf[3] == 0.0


# ── apply_soft_blend: crown resolution ───────────────────────────────────────


def test_nodata_with_low_ground_conf_resolves_to_crown():
    mask = np.full((2, 2), NODATA, dtype=np.float32)
    conf = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    out = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.4, crown_resolve_threshold=0.2)
    assert out[0, 0] == 1.0
    assert np.isclose(out[0, 1], 0.9)
    assert np.isclose(out[1, 0], 0.8)
    assert out[1, 1] == NODATA  # gray zone stays excluded


def test_crown_resolution_skips_invalid_dsm_pixels():
    mask = np.full((1, 2), NODATA, dtype=np.float32)
    conf = np.zeros((1, 2), dtype=np.float32)  # NaN-DSM pixels get conf 0.0
    height_valid = np.array([[True, False]])
    out = apply_soft_blend(
        mask,
        conf,
        nodata_resolve_threshold=0.4,
        crown_resolve_threshold=0.2,
        height_valid=height_valid,
    )
    assert out[0, 0] == 1.0
    assert out[0, 1] == NODATA


def test_height_valid_excludes_dtm_gaps():
    # DSM present but the external DTM has a hole -> nDSM is NaN there, so the
    # pixel carries no height information and must not be resolved to crown.
    dsm = np.array([[10.0, 10.0, np.nan]], dtype=np.float32)
    ndsm = np.array([[2.0, np.nan, np.nan]], dtype=np.float32)
    assert height_data_valid(ndsm, dsm).tolist() == [[True, False, False]]


def test_height_valid_without_ndsm_falls_back_to_dsm():
    dsm = np.array([[10.0, np.nan]], dtype=np.float32)
    assert height_data_valid(None, dsm).tolist() == [[True, False]]


def test_crown_resolution_disabled_by_default():
    mask = np.full((1, 1), NODATA, dtype=np.float32)
    conf = np.zeros((1, 1), dtype=np.float32)
    out = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.4)
    assert out[0, 0] == NODATA


# ── apply_soft_blend: existing behaviour stays intact ────────────────────────


def test_nodata_with_high_ground_conf_still_resolves_to_ground():
    mask = np.full((1, 2), NODATA, dtype=np.float32)
    conf = np.array([[0.9, 0.5]], dtype=np.float32)
    out = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.4, crown_resolve_threshold=0.2)
    assert np.isclose(out[0, 0], 0.1)
    assert np.isclose(out[0, 1], 0.5)


def test_crown_pixels_are_dampened_by_ground_conf():
    mask = np.array([[1.0, 0.6]], dtype=np.float32)
    conf = np.array([[0.5, 0.0]], dtype=np.float32)
    out = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.4, crown_resolve_threshold=0.2)
    assert np.isclose(out[0, 0], 0.5)
    assert np.isclose(out[0, 1], 0.6)
