# tests/test_apply_dsm_mask.py
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from explore_and_process.apply_dsm_mask import (  # noqa: E402
    _smoothstep_confidence,
    apply_soft_blend,
    preserved_crown,
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
    conf = _arr(0.5, 0.9, 0.4, 0.8)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.8 * 0.5, abs=1e-6)  # crown dampened
    assert result[1] == pytest.approx(1.0 - 0.9, abs=1e-6)  # noData resolved
    assert result[2] == pytest.approx(255.0, abs=1e-6)  # noData stays
    assert result[3] == pytest.approx(0.0, abs=1e-6)  # zero stays zero


def test_preserved_crown_keeps_full_label_despite_ground_conf():
    mask = _arr(1.0)
    conf = _arr(0.9)
    result = apply_soft_blend(
        mask, conf, nodata_resolve_threshold=0.7, preserve_crown_threshold=1.0
    )
    assert result[0] == pytest.approx(1.0, abs=1e-6)


def test_preserve_disabled_by_default_dampens_full_label():
    mask = _arr(1.0)
    conf = _arr(0.9)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.1, abs=1e-6)


def test_preserve_leaves_soft_crown_below_threshold_dampened():
    mask = _arr(0.6)
    conf = _arr(0.5)
    result = apply_soft_blend(
        mask, conf, nodata_resolve_threshold=0.7, preserve_crown_threshold=1.0
    )
    assert result[0] == pytest.approx(0.3, abs=1e-6)


def test_preserve_threshold_below_one_protects_soft_edge():
    mask = _arr(0.8, 0.4)
    conf = _arr(0.9, 0.9)
    result = apply_soft_blend(
        mask, conf, nodata_resolve_threshold=0.7, preserve_crown_threshold=0.5
    )
    assert result[0] == pytest.approx(0.8, abs=1e-6)  # protected
    assert result[1] == pytest.approx(0.04, abs=1e-6)  # still dampened


def test_preserve_does_not_touch_sentinels():
    # 255 (unlabelled) and -1 (outside footprint) must never count as crown
    mask = _arr(255.0, -1.0)
    conf = _arr(0.2, 0.2)
    result = apply_soft_blend(
        mask,
        conf,
        nodata_resolve_threshold=0.7,
        crown_resolve_threshold=0.5,
        preserve_crown_threshold=1.0,
    )
    assert result[0] == pytest.approx(0.8, abs=1e-6)  # resolved to crown
    assert result[1] == pytest.approx(-1.0, abs=1e-6)  # untouched


def test_preserved_crown_excludes_unlabelled_sentinel():
    mask = _arr(1.0, 0.99, 255.0, -1.0)
    np.testing.assert_array_equal(
        preserved_crown(mask, 1.0), [True, False, False, False]
    )


def test_preserved_crown_none_selects_nothing():
    mask = _arr(1.0, 0.5)
    assert not preserved_crown(mask, None).any()


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_preserve_threshold_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="preserve_crown_threshold"):
        apply_soft_blend(
            _arr(1.0), _arr(0.5), nodata_resolve_threshold=0.7,
            preserve_crown_threshold=bad,
        )


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
    surface = (offset + tilt_y * (yy / (size - 1)) + tilt_x * (xx / (size - 1))).astype(np.float32)
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

    # 24 blocks, not the production default of 12: _warped_scene's warp is a
    # full sine period across the scene, far higher-frequency than the residual
    # warp between two real surveys. This test is about the mechanism, so it
    # uses a block grid that resolves its own test signal; the trade-off the
    # default makes is covered by test_coarser_blocks_track_a_warp_less_closely.
    plane_only, _ = align_dtm_to_dsm(dsm, dtm, local_refine=False)
    refined, info = align_dtm_to_dsm(dsm, dtm, local_refine=True, local_blocks=24)

    spread_plane = np.abs(_ground_levels(dsm - plane_only)).max()
    spread_refined = np.abs(_ground_levels(dsm - refined)).max()

    assert spread_plane > 0.25  # the warp survives a plane fit
    assert spread_refined < 0.10  # and is removed by the local refinement
    assert info["local_rms"] > 0.0
    assert info["local_blocks"] > 0


def test_coarser_blocks_track_a_warp_less_closely():
    """The price of the coarser default: less warp-tracking, in exchange for a
    block big enough to find bare ground in half-vegetated terrain."""
    from explore_and_process.apply_dsm_mask import align_dtm_to_dsm

    dsm, dtm = _warped_scene(amplitude=0.4)

    fine, _ = align_dtm_to_dsm(dsm, dtm, local_blocks=24)
    coarse, _ = align_dtm_to_dsm(dsm, dtm, local_blocks=12)

    spread_fine = np.abs(_ground_levels(dsm - fine)).max()
    spread_coarse = np.abs(_ground_levels(dsm - coarse)).max()

    assert spread_fine < spread_coarse
    # still a clear improvement over the plane alone (> 0.25 m above)
    assert spread_coarse < 0.20


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
    refined, _ = align_dtm_to_dsm(dsm, dtm, local_refine=True, local_max_correction=1.0)

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
    _, _, ndsm, _ = detect_ground_dtm(dsm, dtm, height_threshold=1.0)

    assert np.abs(np.median(ndsm[120:160, 0:40])) < 0.1
    assert np.median(ndsm[10:50, 10:50]) == pytest.approx(10.0, abs=0.15)


# --- both stages from one fit, and the DTM-above-DSM guard ------------------


def test_both_stages_come_back_from_one_call():
    from explore_and_process.apply_dsm_mask import align_dtm_stages, align_dtm_to_dsm

    dsm, dtm = _warped_scene(amplitude=0.4)
    stages, infos = align_dtm_stages(dsm, dtm, clamp_to_dsm=False)

    assert set(stages) == {"plane", "aligned"}
    # each stage matches the single-stage entry point it corresponds to
    plane_only, _ = align_dtm_to_dsm(dsm, dtm, local_refine=False, clamp_to_dsm=False)
    refined, _ = align_dtm_to_dsm(dsm, dtm, local_refine=True, clamp_to_dsm=False)
    assert np.allclose(stages["plane"], plane_only, equal_nan=True)
    assert np.allclose(stages["aligned"], refined, equal_nan=True)
    # the plane stage carries no local correction, by construction
    assert infos["plane"]["local_rms"] == 0.0
    assert infos["aligned"]["local_rms"] > 0.0


def _over_lifting_scene(size=512, seed=3):
    """A scene the refinement over-corrects on: mostly low vegetation, so the
    per-block ground anchor lands above the real ground."""
    rng = np.random.default_rng(seed)
    dsm = rng.normal(0.0, 0.02, (size, size)).astype(np.float32)
    # 70% of the scene covered by 0.6 m scrub — high enough to drag the anchor
    # up, low enough to stay inside local_max_correction
    scrub = rng.random((size, size)) < 0.7
    dsm[scrub] += 0.6
    dtm = np.full((size, size), -6.0, dtype=np.float32)
    return dsm, dtm


def test_the_clamp_keeps_the_dtm_at_or_below_the_dsm():
    from explore_and_process.apply_dsm_mask import align_dtm_stages

    dsm, dtm = _over_lifting_scene()

    loose, loose_info = align_dtm_stages(dsm, dtm, clamp_to_dsm=False)
    clamped, clamped_info = align_dtm_stages(dsm, dtm, clamp_to_dsm=True)

    # without the guard the refinement lifts the DTM above the DSM ...
    assert loose_info["aligned"]["n_above_dsm"] > 0
    assert (loose["aligned"] > dsm).any()
    # ... with it, nowhere, and the count is still reported (measured before
    # the clamp), so the bias stays visible instead of being papered over
    assert not (clamped["aligned"] > dsm).any()
    assert clamped_info["aligned"]["n_above_dsm"] == loose_info["aligned"]["n_above_dsm"]
    assert clamped_info["aligned"]["max_above_dsm"] > 0.0


def test_the_clamp_leaves_dsm_gaps_alone():
    from explore_and_process.apply_dsm_mask import align_dtm_stages

    dsm, dtm = _over_lifting_scene()
    dsm[100:140, 100:140] = np.nan

    stages, _ = align_dtm_stages(dsm, dtm, clamp_to_dsm=True)

    # a hole in the DSM must not punch a hole in the DTM
    assert np.isfinite(stages["aligned"][100:140, 100:140]).all()


def test_the_clamp_does_not_flatten_canopy():
    from explore_and_process.apply_dsm_mask import align_dtm_stages

    dsm, dtm = _warped_scene(amplitude=0.4)
    stages, _ = align_dtm_stages(dsm, dtm, clamp_to_dsm=True)

    ndsm = dsm - stages["aligned"]
    assert np.median(ndsm[200:280, 300:380]) == pytest.approx(10.0, abs=0.25)


def test_the_stage_decides_which_ndsm_detect_ground_dtm_builds():
    from explore_and_process.apply_dsm_mask import detect_ground_dtm

    dsm, dtm = _warped_scene(amplitude=0.4)
    _, _, ndsm_plane, stages = detect_ground_dtm(
        dsm, dtm, height_threshold=1.0, stage="plane", clamp_to_dsm=False
    )
    _, _, ndsm_aligned, _ = detect_ground_dtm(
        dsm, dtm, height_threshold=1.0, stage="aligned", clamp_to_dsm=False
    )

    assert np.allclose(ndsm_plane, dsm - stages["plane"], equal_nan=True)
    assert not np.allclose(ndsm_plane, ndsm_aligned, equal_nan=True)


def test_an_unknown_stage_is_rejected():
    from explore_and_process.apply_dsm_mask import detect_ground_dtm

    dsm, dtm = _synthetic_scene(offset=6.5)
    with pytest.raises(ValueError, match="unknown dtm stage"):
        detect_ground_dtm(dsm, dtm, height_threshold=1.0, stage="raw")


# ── validate_method ──────────────────────────────────────────────────────────


def _method_cfg(**kw):
    from omegaconf import OmegaConf

    base = {"method": "dtm_approx", "dtm_approx": {"cell": 0.5}}
    return OmegaConf.create({**base, **kw})


def test_method_dtm_uses_external_dtm():
    from explore_and_process.apply_dsm_mask import validate_method

    assert validate_method(_method_cfg(method="dtm", dtm_real={"path": "dtm.tif"})) == ("dtm", "dtm")


def test_dtm_approx_ignores_a_set_dtm_path():
    from explore_and_process.apply_dsm_mask import validate_method

    assert validate_method(_method_cfg(dtm_real={"path": "dtm.tif"})) == ("dtm_approx", "dtm_approx")


@pytest.mark.parametrize("method", ["gradient", "both"])
def test_removed_methods_raise(method):
    from explore_and_process.apply_dsm_mask import validate_method

    with pytest.raises(ValueError, match="expected one of"):
        validate_method(_method_cfg(method=method))


@pytest.mark.parametrize(
    "key",
    [
        "combine",
        "gradient_sigma",
        "gradient_threshold",
        "windows",
        "dtm",
        "dtm_stage",
        "dtm_local_blocks",
        "dtm_clamp_to_dsm",
        "dtm_local_refine",
    ],
)
def test_removed_keys_raise(key):
    from explore_and_process.apply_dsm_mask import validate_method

    with pytest.raises(ValueError, match="removed|replaced|moved"):
        validate_method(_method_cfg(**{key: None}))


def test_local_min_points_to_its_new_name():
    from explore_and_process.apply_dsm_mask import validate_method

    with pytest.raises(ValueError, match="renamed to dtm_approx"):
        validate_method(_method_cfg(method="local_min"))


def test_dtm_approx_without_block_raises():
    from explore_and_process.apply_dsm_mask import validate_method

    with pytest.raises(ValueError, match="needs a dsm_mask.dtm_approx"):
        validate_method(_method_cfg(dtm_approx=None))


def test_dtm_approx_unknown_key_raises():
    from explore_and_process.apply_dsm_mask import validate_method

    with pytest.raises(ValueError, match="unknown keys"):
        validate_method(_method_cfg(dtm_approx={"cell": 0.5, "window": 3}))


def test_method_dtm_without_path_raises():
    from explore_and_process.apply_dsm_mask import validate_method

    with pytest.raises(ValueError, match="needs a dtm_real block"):
        validate_method(_method_cfg(method="dtm"))
    with pytest.raises(ValueError, match="dtm_real.path is not set"):
        validate_method(_method_cfg(method="dtm", dtm_real={"stage": "plane"}))


def test_mask_from_dtm_channel_from_dtm_approx():
    from explore_and_process.apply_dsm_mask import validate_method

    cfg = _method_cfg(method="dtm", dtm_real={"path": "dtm.tif"}, ndsm_channel="dtm_approx")
    assert validate_method(cfg) == ("dtm", "dtm_approx")


def test_channel_dtm_needs_a_dtm_path():
    from explore_and_process.apply_dsm_mask import validate_method

    with pytest.raises(ValueError, match="needs a dtm_real block"):
        validate_method(_method_cfg(ndsm_channel="dtm"))


def test_channel_dtm_approx_needs_its_block():
    from explore_and_process.apply_dsm_mask import validate_method

    cfg = _method_cfg(
        method="dtm", dtm_real={"path": "dtm.tif"}, ndsm_channel="dtm_approx", dtm_approx=None
    )
    with pytest.raises(ValueError, match="needs a dsm_mask.dtm_approx"):
        validate_method(cfg)


def test_dtm_real_bad_stage_raises():
    from explore_and_process.apply_dsm_mask import validate_method

    cfg = _method_cfg(method="dtm", dtm_real={"path": "dtm.tif", "stage": "warped"})
    with pytest.raises(ValueError, match="dtm_real.stage"):
        validate_method(cfg)


def test_dtm_real_is_not_checked_for_pure_dtm_approx():
    """A broken dtm_real block must not stop a run that never uses it."""
    from explore_and_process.apply_dsm_mask import validate_method

    cfg = _method_cfg(dtm_real={"stage": "warped"})
    assert validate_method(cfg) == ("dtm_approx", "dtm_approx")


def test_unknown_channel_raises():
    from explore_and_process.apply_dsm_mask import validate_method

    with pytest.raises(ValueError, match="ndsm_channel"):
        validate_method(_method_cfg(ndsm_channel="lidar"))


# ── approximate_dtm (dtm_approx) ─────────────────────────────────────────────
#
# Synthetic 100 m × 100 m scenes on a 0.1 m grid (1000 × 1000 px) with the
# config defaults: 0.5 m cells, windows up to 30 m.

APPROX_GSD = 0.1
APPROX_N = 1000


def _xy():
    ax = np.arange(APPROX_N, dtype=np.float32) * APPROX_GSD
    return np.meshgrid(ax, ax, indexing="ij")


def _plane(slope=0.10):
    rows, cols = _xy()
    return (100.0 + slope * cols + 0.03 * rows).astype(np.float32)


def _approx(dsm, **kw):
    from explore_and_process.apply_dsm_mask import DtmApproxParams, approximate_dtm

    return approximate_dtm(dsm, APPROX_GSD, DtmApproxParams(**kw))


def test_approx_dtm_follows_a_tilted_plane():
    """The opening reproduces a plane exactly — the old minimum filter sat
    slope × window / 2 below it (1.25 m at 10 % and 25 m)."""
    plane = _plane(0.10)
    dtm = _approx(plane)
    inner = (slice(50, -50), slice(50, -50))
    assert np.max(np.abs(dtm[inner] - plane[inner])) < 0.1


def test_approx_dtm_keeps_a_gentle_embankment_as_terrain():
    """3 m high, 30° flanks, 6 m crest, 80 m long: terrain, not an object."""
    plane = _plane(0.05)
    rows, cols = _xy()
    half_crest, height, tan_flank = 3.0, 3.0, np.tan(np.radians(30.0))
    dist = np.abs(cols - 50.0)
    bank = np.clip(height - (dist - half_crest) * tan_flank, 0.0, height)
    bank[(rows < 10.0) | (rows > 90.0)] = 0.0
    dsm = plane + bank.astype(np.float32)

    ndsm = dsm - _approx(dsm)
    on_bank = (bank > 0.5) & (rows > 20.0) & (rows < 80.0)
    assert np.percentile(ndsm[on_bank], 95) < 0.7  # height_threshold


def test_approx_dtm_removes_a_tree():
    """8 m wide, 12 m tall cylinder: full height in the nDSM, ground around it."""
    plane = _plane(0.05)
    rows, cols = _xy()
    tree = (rows - 50.0) ** 2 + (cols - 50.0) ** 2 <= 4.0**2
    dsm = plane + np.where(tree, 12.0, 0.0).astype(np.float32)

    ndsm = dsm - _approx(dsm)
    assert np.median(ndsm[tree]) > 11.0
    ring = ((rows - 50.0) ** 2 + (cols - 50.0) ** 2 > 6.0**2) & (
        (rows - 50.0) ** 2 + (cols - 50.0) ** 2 < 10.0**2
    )
    assert np.percentile(np.abs(ndsm[ring]), 95) < 0.3


def test_approx_dtm_ignores_a_single_pit():
    plane = _plane(0.05)
    dsm = plane.copy()
    dsm[500, 500] -= 5.0
    dtm = _approx(dsm)
    err = np.abs(dtm - plane)[480:521, 480:521]
    err[20, 20] = 0.0  # the pit pixel itself: terrain is capped at the DSM there
    assert np.max(err) < 0.1


def test_approx_dtm_keeps_nan_and_stays_below_dsm():
    plane = _plane(0.05)
    rows, cols = _xy()
    dsm = plane + np.where((rows - 50.0) ** 2 + (cols - 50.0) ** 2 <= 9.0, 8.0, 0.0)
    dsm = dsm.astype(np.float32)
    dsm[:, :100] = np.nan  # outside the footprint
    dtm = _approx(dsm)
    assert np.isnan(dtm[:, :100]).all()
    inside = np.isfinite(dsm)
    assert np.isfinite(dtm[inside]).all()
    assert (dtm[inside] <= dsm[inside] + 1e-5).all()


def test_tree_at_footprint_edge_is_still_removed():
    """No-data must not act as ground or as canopy: a tree touching the
    footprint edge is opened away by the ground on its inner side."""
    plane = _plane(0.05)
    rows, cols = _xy()
    tree = (rows - 50.0) ** 2 + (cols - 12.0) ** 2 <= 4.0**2
    dsm = plane + np.where(tree, 12.0, 0.0).astype(np.float32)
    dsm[:, :100] = np.nan
    ndsm = dsm - _approx(dsm)
    assert np.nanmedian(ndsm[tree & np.isfinite(dsm)]) > 11.0


# ── main: mask from the DTM, input channel from dtm_approx ───────────────────


def test_main_builds_mask_from_dtm_and_channel_from_dtm_approx(tmp_path):
    """The mask follows the external DTM, the written nDSM channel follows
    dtm_approx, and a raw metre nDSM is written for each of them."""
    import rasterio
    from omegaconf import OmegaConf
    from rasterio.transform import from_origin

    from explore_and_process.apply_dsm_mask import (
        DtmApproxParams,
        approximate_dtm,
        main,
        normalize_ndsm,
    )

    n, gsd = 400, 0.1
    transform = from_origin(500000, 5400000, gsd, gsd)
    rows, cols = np.meshgrid(np.arange(n) * gsd, np.arange(n) * gsd, indexing="ij")
    ground = (100.0 + 0.05 * cols).astype(np.float32)
    tree = (rows - 20.0) ** 2 + (cols - 20.0) ** 2 <= 3.0**2
    dsm = ground + np.where(tree, 10.0, 0.0).astype(np.float32)
    # The DTM sits 2 m lower everywhere: co-registration must absorb it
    dtm = ground - 2.0

    def write(path, arr, nodata=None):
        with rasterio.open(
            path, "w", driver="GTiff", dtype="float32", width=n, height=n, count=1,
            crs="EPSG:32636", transform=transform, nodata=nodata,
        ) as dst:
            dst.write(arr[np.newaxis].astype(np.float32))
        return str(path)

    (tmp_path / "masks").mkdir()
    approx_block = {"cell": 0.5, "max_window": 12.0}
    cfg = OmegaConf.create({
        "mask": write(tmp_path / "masks" / "m.tif", np.full((n, n), 255.0), nodata=-1.0),
        "dsm": write(tmp_path / "dsm.tif", dsm),
        "dtm_real": {
            "path": write(tmp_path / "dtm.tif", dtm),
            "stage": "plane",
            "local_blocks": 4,
        },
        "out": None,
        "out_dsm": None,
        "method": "dtm",
        "ndsm_channel": "dtm_approx",
        "dtm_approx": approx_block,
        "height_threshold": 0.7,
        "height_ramp": 1.0,
        "max_ndsm_height": 16.0,
        "nodata_resolve_threshold": 0.5,
        "crown_resolve_threshold": 0.3,
        "preserve_crown_threshold": None,
    })
    compare = main(cfg)

    assert "_dtm_ht0.7" in Path(cfg.out).name  # mask labelled by the DTM
    assert Path(cfg.out_dsm).stem.endswith("_da")  # channel from dtm_approx
    with rasterio.open(cfg.out_dsm) as src:
        channel = src.read(1)
    expected = normalize_ndsm(
        dsm - approximate_dtm(dsm, gsd, DtmApproxParams(**approx_block)),
        dsm,
        16.0,
    )
    np.testing.assert_allclose(channel, expected, atol=1e-5)

    with rasterio.open(cfg.out) as src:
        mask = src.read(1)
    assert mask[tree].mean() > 0.9  # crown from the DTM nDSM
    assert mask[~tree & ((rows - 20.0) ** 2 + (cols - 20.0) ** 2 > 6.0**2)].max() < 0.1

    raw = sorted(p.name for p in (tmp_path / "ndsm_in_m").rglob("*.tif"))
    assert raw == ["dsm_ndsm_da_raw_m.tif", "dsm_ndsm_dtm_raw_m.tif"]

    # Both nDSMs exist, so they are compared in [0,1]
    assert compare is not None and compare["n_valid"] == n * n
    written = sorted(p.name for p in Path(compare["dir"]).iterdir())
    assert written == [
        "dsm_ndsm_err_da_minus_dtm.png",
        "dsm_ndsm_err_da_minus_dtm.tif",
        "dsm_ndsm_err_da_minus_dtm_m.tif",
        "ndsm_compare.json",
    ]
    with rasterio.open(Path(compare["dir"]) / "dsm_ndsm_err_da_minus_dtm.tif") as src:
        error = src.read(1)
    assert np.mean(error.astype(np.float64) ** 2) == pytest.approx(compare["mse"], rel=1e-5)
    with rasterio.open(Path(compare["dir"]) / "dsm_ndsm_err_da_minus_dtm_m.tif") as src:
        error_m = src.read(1)
    assert np.mean(error_m.astype(np.float64) ** 2) == pytest.approx(compare["mse_m"], rel=1e-5)

    # Rebuilt from the saved raw nDSMs alone, the comparison is the same
    from explore_and_process.apply_dsm_mask import recompare_ndsm

    again = recompare_ndsm(str(tmp_path), 16.0)
    assert again["dir"] == compare["dir"]
    for key in ("mse", "bias", "mse_m", "p95_abs_m", "n_valid", "ceiling_da"):
        assert again[key] == pytest.approx(compare[key], rel=1e-6)


def _flat_scene(n=50):
    dsm = np.full((n, n), 100.0, dtype=np.float32)
    ndsm = np.zeros((n, n), dtype=np.float32)
    ndsm[:, n // 2 :] = 10.0  # half the scene is 10 m canopy, so p95 = 10 m
    return dsm, ndsm


def test_compare_ndsm_identical_ndsms_have_zero_error():
    from explore_and_process.apply_dsm_mask import compare_ndsm

    dsm, ndsm = _flat_scene()
    cmp = compare_ndsm(ndsm, ndsm.copy(), dsm, 16.0)
    m = cmp.metrics
    assert m["mse"] == 0.0 and m["bias"] == 0.0 and m["mse_m"] == 0.0 and m["p95_abs_m"] == 0.0
    assert np.all(cmp.error == 0.0) and np.all(cmp.error_m == 0.0)


def test_compare_ndsm_scales_each_ndsm_with_its_own_ceiling():
    from explore_and_process.apply_dsm_mask import compare_ndsm

    dsm, ndsm_dtm = _flat_scene()
    # dtm_approx sees the canopy at 5 m: scaled by its own ceiling it is
    # identical in [0,1], the metre MSE is not
    m = compare_ndsm(ndsm_dtm, ndsm_dtm / 2.0, dsm, 16.0).metrics
    assert m["ceiling_dtm"] == pytest.approx(10.0)
    assert m["ceiling_da"] == pytest.approx(5.0)
    assert m["mse"] == pytest.approx(0.0)
    # in metres the halved canopy is visible: half the pixels 5 m too low
    assert m["mse_m"] == pytest.approx(0.5 * 5.0**2)
    assert m["rmse_m"] == pytest.approx(np.sqrt(0.5 * 5.0**2))
    assert m["mae_m"] == pytest.approx(2.5)
    assert m["bias_m"] == pytest.approx(-2.5)
    assert m["p95_abs_m"] == pytest.approx(5.0)


def test_compare_ndsm_known_offset():
    from explore_and_process.apply_dsm_mask import compare_ndsm

    dsm, ndsm_dtm = _flat_scene()
    ndsm_da = ndsm_dtm.copy()
    ndsm_da[:, : ndsm_da.shape[1] // 2] = 2.0  # ground lifted by 0.2 of the 10 m ceiling
    cmp = compare_ndsm(ndsm_dtm, ndsm_da, dsm, 16.0)
    m, error = cmp.metrics, cmp.error
    assert m["bias"] == pytest.approx(0.5 * 0.2)
    assert m["bias_m"] == pytest.approx(0.5 * 2.0)
    assert m["mse"] == pytest.approx(0.5 * 0.2**2)
    assert m["mae"] == pytest.approx(0.5 * 0.2)
    assert error.max() == pytest.approx(0.2)


def test_compare_ndsm_excludes_pixels_without_height_data():
    from explore_and_process.apply_dsm_mask import compare_ndsm

    dsm, ndsm_dtm = _flat_scene(10)
    ndsm_da = ndsm_dtm.copy()
    dsm[0, 0] = np.nan
    ndsm_dtm[1, 1] = np.nan  # external DTM gap
    ndsm_da[2, 2] = np.nan
    ndsm_da[3, 3] = 7.0  # the only real difference
    cmp = compare_ndsm(ndsm_dtm, ndsm_da, dsm, 16.0)
    m, error = cmp.metrics, cmp.error
    assert m["n_valid"] == 100 - 3
    for arr in (error, cmp.error_m):
        assert np.isnan(arr[0, 0]) and np.isnan(arr[1, 1]) and np.isnan(arr[2, 2])
        assert np.count_nonzero(arr[np.isfinite(arr)]) == 1
    assert cmp.error_m[3, 3] == pytest.approx(7.0)


def test_recompare_ndsm_needs_both_raw_ndsms(tmp_path):
    from explore_and_process.apply_dsm_mask import recompare_ndsm

    (tmp_path / "ndsm_in_m" / "20260101_000000").mkdir(parents=True)
    (tmp_path / "ndsm_in_m" / "20260101_000000" / "dsm_ndsm_da_raw_m.tif").touch()
    with pytest.raises(FileNotFoundError, match="both raw nDSMs"):
        recompare_ndsm(str(tmp_path), 16.0)
