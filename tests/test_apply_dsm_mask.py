# tests/test_apply_dsm_mask.py
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from explore_and_process.apply_dsm_mask import (  # noqa: E402
    _smoothstep_confidence,
    apply_soft_blend,
)


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


def test_nodata_resolves_when_conf_exactly_at_threshold():
    # threshold is inclusive: ground_conf >= nodata_resolve_threshold resolves
    mask = _arr(255.0)
    conf = _arr(0.7)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.3, abs=1e-6)


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


def test_smoothstep_below_threshold_is_one():
    ndsm = _arr(0.0, 1.0, 2.0)
    result = _smoothstep_confidence(ndsm, threshold=2.0)
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0], atol=1e-6)


def test_smoothstep_at_double_threshold_is_zero():
    ndsm = _arr(4.0)
    result = _smoothstep_confidence(ndsm, threshold=2.0)
    np.testing.assert_allclose(result, [0.0], atol=1e-6)


def test_smoothstep_above_double_threshold_is_zero():
    ndsm = _arr(10.0)
    result = _smoothstep_confidence(ndsm, threshold=2.0)
    np.testing.assert_allclose(result, [0.0], atol=1e-6)


def test_smoothstep_midpoint_is_half():
    # t=0.5 → smoothstep = 3(0.25) - 2(0.125) = 0.5 → confidence = 0.5
    ndsm = _arr(3.0)  # = 1.5 × threshold
    result = _smoothstep_confidence(ndsm, threshold=2.0)
    np.testing.assert_allclose(result, [0.5], atol=1e-6)


def test_smoothstep_finite_for_finite_inputs():
    ndsm = _arr(0.0, 4.0)
    result = _smoothstep_confidence(ndsm, threshold=2.0)
    assert np.isfinite(result[0])
    assert np.isfinite(result[1])


# --- align_dtm_to_dsm: vertical co-registration of an external DTM ----------


def _synthetic_scene(offset=0.0, tilt_y=0.0, tilt_x=0.0, size=256, seed=0):
    """Flat ground at z=0 with a few tall blobs, plus a DTM that is mis-levelled
    by `offset` metres and tilted by `tilt_*` metres across the full extent."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    ground = np.zeros((size, size), dtype=np.float32)
    dsm = ground + rng.normal(0.0, 0.02, (size, size)).astype(np.float32)
    # canopy: three 40x40 blocks of 10 m vegetation (~18% of the scene)
    for r, c in ((10, 10), (100, 150), (200, 60)):
        dsm[r : r + 40, c : c + 40] += 10.0
    surface = (
        offset
        + tilt_y * (yy / (size - 1))
        + tilt_x * (xx / (size - 1))
    ).astype(np.float32)
    dtm = (ground - surface).astype(np.float32)
    return dsm, dtm


def test_constant_offset_is_removed_so_ground_sits_at_zero():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _synthetic_scene(offset=6.5)
    aligned, info = align_dtm_to_dsm(dsm, dtm)

    ndsm = dsm - aligned
    ground = ndsm[120:160, 0:40]  # a patch with no canopy
    assert np.abs(np.median(ground)) < 0.1
    assert info["mean_shift"] == pytest.approx(6.5, abs=0.1)


def test_tilted_offset_is_removed_across_the_whole_scene():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _synthetic_scene(offset=6.0, tilt_y=1.8)
    aligned, _ = align_dtm_to_dsm(dsm, dtm)

    ndsm = dsm - aligned
    # ground-only columns at the top and the bottom of the scene must both
    # land at zero - a constant shift would leave ~0.9 m of residual tilt
    top = ndsm[0:40, 100:140]
    bottom = ndsm[216:256, 100:140]
    assert np.abs(np.median(top)) < 0.15
    assert np.abs(np.median(bottom)) < 0.15


def test_canopy_height_is_preserved_after_alignment():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _synthetic_scene(offset=6.5)
    aligned, _ = align_dtm_to_dsm(dsm, dtm)

    ndsm = dsm - aligned
    assert np.median(ndsm[10:50, 10:50]) == pytest.approx(10.0, abs=0.15)


def test_already_aligned_dtm_is_left_alone():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _synthetic_scene(offset=0.0)
    aligned, info = align_dtm_to_dsm(dsm, dtm)

    assert np.abs(info["mean_shift"]) < 0.1
    assert np.nanmax(np.abs(aligned - dtm)) < 0.15


def test_nan_pixels_do_not_break_the_fit():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _synthetic_scene(offset=6.5)
    dsm[0:60, :] = np.nan
    dtm[:, 0:60] = np.nan
    aligned, info = align_dtm_to_dsm(dsm, dtm)

    assert info["mean_shift"] == pytest.approx(6.5, abs=0.15)
    assert np.isfinite(aligned[128, 128])


def test_implausible_shift_raises():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _synthetic_scene(offset=45.0)  # geoid-sized blunder
    with pytest.raises(ValueError, match=r"vertical mismatch of 4[45]\.\d+ m"):
        align_dtm_to_dsm(dsm, dtm, max_shift=20.0)


def test_falls_back_to_constant_shift_when_candidates_are_too_few():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _synthetic_scene(offset=6.5, tilt_y=1.8, size=64)
    dsm[8:, :] = np.nan  # only a thin strip of valid data survives
    aligned, info = align_dtm_to_dsm(dsm, dtm)

    assert info["mode"] == "constant"
    assert np.isfinite(aligned).any()


# --- local refinement of the residual warp ----------------------------------


def _warped_scene(amplitude=0.4, size=512, seed=1):
    """Like _synthetic_scene, but the DTM is additionally warped by a smooth
    bump no plane can absorb - the real-world case of two surveys flown on
    different dates."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dsm = rng.normal(0.0, 0.02, (size, size)).astype(np.float32)
    for r, c in ((20, 20), (200, 300), (400, 120)):
        dsm[r : r + 80, c : c + 80] += 10.0
    warp = amplitude * np.sin(2 * np.pi * yy / size) * np.cos(2 * np.pi * xx / size)
    dtm = (-(6.0 + warp)).astype(np.float32)
    return dsm, dtm.astype(np.float32)


def _ground_levels(ndsm, size=512, block=64):
    """Median nDSM per block, canopy blocks excluded, as a ground-level map."""
    levels = []
    for i in range(0, size, block):
        for j in range(0, size, block):
            med = np.median(ndsm[i : i + block, j : j + block])
            if med < 1.0:  # skip canopy blocks
                levels.append(med)
    return np.array(levels)


def test_local_refinement_flattens_a_warp_a_plane_cannot_absorb():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _warped_scene(amplitude=0.4)

    plane_only, _ = align_dtm_to_dsm(dsm, dtm, local_refine=False)
    refined, info = align_dtm_to_dsm(dsm, dtm, local_refine=True)

    spread_plane = np.abs(_ground_levels(dsm - plane_only)).max()
    spread_refined = np.abs(_ground_levels(dsm - refined)).max()

    assert spread_plane > 0.25  # the warp survives a plane fit
    assert spread_refined < 0.10  # and is removed by the local refinement
    assert info["local_rms"] > 0.0
    assert info["local_blocks"] > 0


def test_local_refinement_preserves_canopy_height():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _warped_scene(amplitude=0.4)
    refined, _ = align_dtm_to_dsm(dsm, dtm, local_refine=True)

    ndsm = dsm - refined
    # a whole closed-canopy block must not be pulled down towards zero
    assert np.median(ndsm[200:280, 300:380]) == pytest.approx(10.0, abs=0.25)


def test_local_refinement_correction_is_bounded():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _warped_scene(amplitude=0.4)
    plane_only, _ = align_dtm_to_dsm(dsm, dtm, local_refine=False)
    refined, _ = align_dtm_to_dsm(
        dsm, dtm, local_refine=True, local_max_correction=1.0
    )

    assert np.nanmax(np.abs(refined - plane_only)) <= 1.0 + 1e-5


def test_local_refinement_is_a_no_op_on_an_unwarped_dtm():
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _warped_scene(amplitude=0.0)
    plane_only, _ = align_dtm_to_dsm(dsm, dtm, local_refine=False)
    refined, info = align_dtm_to_dsm(dsm, dtm, local_refine=True)

    assert info["local_rms"] < 0.05
    assert np.nanmax(np.abs(refined - plane_only)) < 0.1


def test_detect_ground_dtm_aligns_before_differencing():
    from explore_and_process.apply_dsm_mask import detect_ground_dtm

    dsm, dtm = _synthetic_scene(offset=6.5)
    _, _, ndsm = detect_ground_dtm(dsm, dtm, height_threshold=1.0)

    assert np.abs(np.median(ndsm[120:160, 0:40])) < 0.1
    assert np.median(ndsm[10:50, 10:50]) == pytest.approx(10.0, abs=0.15)
