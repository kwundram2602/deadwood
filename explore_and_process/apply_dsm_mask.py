"""
apply_dsm_mask.py

Refine the soft crown mask by labelling confirmed ground pixels using the DSM.

Ground detection supports three methods:

  local_min  — DTM approximation via multi-scale minimum filter:
                 nDSM_approx = DSM - minimum_filter(DSM, window)
                 pixels where nDSM_approx < height_threshold  =>  ground = 0.0

  dtm        — External DTM raster (resampled to DSM grid, then vertically
               co-registered onto the DSM in two stages — see
               align_dtm_stages; `dtm_stage` picks the one used, both are
               written to process_out/dtm_coreg/<run_id>/):
                 nDSM = DSM - DTM
                 pixels where nDSM < height_threshold  =>  ground = 0.0

  gradient   — Slope/edge filter: flat DSM regions are classified as ground.

  both       — Combines local_min (or dtm when --dtm is supplied) with gradient
               via OR or AND logic (see --combine).

Ground pixels soft-blend into the crown mask: crown pixels (0–1) are
multiplied by (1 − ground_conf); noData pixels are resolved to ground
only when ground_conf exceeds --nodata_resolve_threshold.

Remaining noData pixels (255) — no crown polygon, not confirmed ground —
are excluded from the loss during training.

Usage:
  python explore_and_process/apply_dsm_mask.py \\
      --mask  explore_and_process/out/crown_mask.tif \\
      --dsm   data/raster/DSM.tif \\
      --out   explore_and_process/out/crown_mask_final.tif \\
      [--method dtm --dtm data/raster/DTM.tif] \\
      [--window 200] [--height_threshold 2.0]
"""

import argparse
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from omegaconf import OmegaConf
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter, minimum_filter, sobel, uniform_filter1d

from utils.nodata import MASK_OUTSIDE, MASK_RASTER_NODATA, MASK_UNLABELLED

logger = logging.getLogger(__name__)

# The two stages of the DSM/DTM co-registration, in the order they are built:
# "plane" removes the global offset and tilt, "aligned" adds the blockwise
# residual warp on top. Both are written to disk; `dtm_stage` picks the one the
# nDSM and the mask are built on.
DTM_STAGES: tuple[str, ...] = ("plane", "aligned")


def resample_raster(path, h, w, transform, crs):
    """Reproject a single-band raster to exactly match the target grid."""
    out = np.full((h, w), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
    return out


def _embed_params(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def _find_valley_threshold(ndsm: np.ndarray, lo: float = 0.3, hi: float = 8.0, bins: int = 300) -> float:
    """Return the histogram valley between the ground and vegetation peaks."""
    valid = ndsm[(ndsm >= lo) & (ndsm <= hi) & ~np.isnan(ndsm)]
    if valid.size == 0:
        return lo
    counts, edges = np.histogram(valid, bins=bins, range=(lo, hi))
    smoothed = uniform_filter1d(counts.astype(float), size=10)
    valley_idx = int(np.argmin(smoothed))
    return float(edges[valley_idx] + (edges[1] - edges[0]) / 2)


def _otsu_threshold(arr: np.ndarray, bins: int = 256) -> float:
    """Return Otsu's threshold for a 1-D float array."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("_otsu_threshold: no finite values in input array")
    counts, edges = np.histogram(valid, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    total = counts.sum()
    w0 = np.cumsum(counts) / total
    w1 = 1.0 - w0
    mu0 = np.cumsum(counts * centers) / np.maximum(np.cumsum(counts), 1)
    mu_total = float((counts * centers).sum() / total)
    mu1 = (mu_total - w0 * mu0) / np.maximum(w1, 1e-8)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(sigma_b))])


def _smoothstep_confidence(
    ndsm: np.ndarray, threshold: float, ramp: float | None = None
) -> np.ndarray:
    """Ground confidence based on height threshold.

    Returns 1.0 for nDSM <= threshold, smoothly falls to 0.0 at
    threshold + ramp (default ramp = threshold, i.e. zero at 2*threshold).
    Uses the smoothstep curve (3t^2 - 2t^3) for a C1-continuous transition.
    NaN handling: caller is responsible for zeroing NaN pixels after this call.
    """
    ramp = threshold if ramp is None else ramp
    t = np.clip((ndsm - threshold) / ramp, 0.0, 1.0)
    return (1.0 - (3 * t**2 - 2 * t**3)).astype(np.float32)


def normalize_ndsm(
    ndsm: np.ndarray, dsm: np.ndarray, max_ndsm_height: float
) -> np.ndarray:
    """Normalise raw nDSM (m) to [0,1] with ceiling = min(p95, max_ndsm_height).

    Same procedure at train and predict time: pixels above the ceiling
    saturate to 1, DSM noData pixels become 0.
    """
    valid_pos = ndsm[~np.isnan(ndsm) & (ndsm > 0)]
    p95 = float(np.percentile(valid_pos, 95)) if valid_pos.size > 0 else max_ndsm_height
    ceiling = min(p95, max_ndsm_height)
    print(f"nDSM ceiling: {ceiling:.2f} m  (p95={p95:.2f} m, cap={max_ndsm_height} m)")
    ndsm_norm = np.clip(ndsm, 0.0, ceiling) / ceiling
    # nDSM is NaN wherever the DSM *or* the external DTM has a gap — both mean
    # "no height information", and neither may leak a NaN into the channel.
    ndsm_norm[np.isnan(ndsm) | np.isnan(dsm)] = 0.0
    return ndsm_norm.astype(np.float32)


def height_data_valid(ndsm: np.ndarray | None, dsm: np.ndarray) -> np.ndarray:
    """Pixels where the height detector actually had data.

    A DTM gap leaves nDSM NaN while the DSM is perfectly valid; the resulting
    ground_conf of 0.0 is then indistinguishable from "confidently canopy", so
    such pixels must stay noData instead of being resolved to crown.
    """
    valid = ~np.isnan(dsm)
    if ndsm is not None:
        valid &= ~np.isnan(ndsm)
    return valid


def detect_ground_local_min(
    dsm: np.ndarray,
    windows: list[int],
    height_threshold: float,
    height_ramp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DTM approximation via multi-scale minimum filter.

    Returns:
        binary:     bool array, True where pixel is ground
        confidence: float32 [0,1], higher = more likely ground
        ndsm:       raw nDSM (m) for diagnostics and nDSM output
    """
    dsm_filled = np.where(np.isnan(dsm), np.nanmax(dsm), dsm)
    local_mins = [minimum_filter(dsm_filled, size=w) for w in windows]
    local_min = np.mean(local_mins, axis=0) if len(local_mins) > 1 else local_mins[0]
    ndsm = dsm - local_min

    confidence = _smoothstep_confidence(ndsm, height_threshold, height_ramp)
    confidence[np.isnan(dsm)] = 0.0

    binary = (ndsm < height_threshold) & ~np.isnan(dsm)
    return binary, confidence, ndsm


def _block_ground_candidates(
    diff: np.ndarray, blocks: int, quantile: float
) -> np.ndarray:
    """Boolean mask of likely-ground pixels, picked per block rather than
    globally so the candidates are spread over the whole scene: a global
    quantile lands entirely in the lowest-lying corner and leaves the rest of
    the fit unconstrained."""
    h, w = diff.shape
    picked = np.zeros((h, w), dtype=bool)
    for rows in np.array_split(np.arange(h), min(blocks, h)):
        for cols in np.array_split(np.arange(w), min(blocks, w)):
            block = diff[np.ix_(rows, cols)]
            valid = block[np.isfinite(block)]
            if valid.size == 0:
                continue
            cut = np.percentile(valid, quantile)
            picked[np.ix_(rows, cols)] = np.isfinite(block) & (block <= cut)
    return picked


def _local_ground_residual(
    resid: np.ndarray,
    blocks: int,
    quantile: float,
    min_valid_frac: float,
    max_correction: float,
    smooth_sigma: float,
    ground_band: float,
) -> tuple[np.ndarray, int]:
    """Low-frequency map of how far bare ground still sits off zero.

    A single plane only removes the first-order mis-levelling; a DTM from a
    different survey (different date, coarser GSD) stays locally warped against
    the DSM by a few decimetres. This estimates that warp on a coarse block
    grid and returns it at full resolution.

    Blocks whose low quantile lands above ``max_correction`` saw no bare ground
    (closed canopy, buildings) and are dropped, then filled from their
    neighbours by the NaN-aware Gaussian below — so canopy is never flattened.

    Returns:
        correction: same shape as resid, clipped to +-max_correction
        n_blocks:   how many blocks contributed a direct estimate
    """
    h, w = resid.shape
    row_edges = np.array_split(np.arange(h), min(blocks, h))
    col_edges = np.array_split(np.arange(w), min(blocks, w))
    coarse = np.full((len(row_edges), len(col_edges)), np.nan, dtype=np.float32)

    for i, rows in enumerate(row_edges):
        for j, cols in enumerate(col_edges):
            block = resid[np.ix_(rows, cols)]
            valid = block[np.isfinite(block)]
            if valid.size < min_valid_frac * block.size:
                continue
            # The low quantile finds the ground population even in a block that
            # is mostly canopy, but sits ~1.6 sigma below its centre. Taking the
            # median of everything within +-ground_band of it recovers the true
            # ground level without dragging canopy in.
            anchor = float(np.percentile(valid, quantile))
            near = valid[np.abs(valid - anchor) <= ground_band]
            level = float(np.median(near)) if near.size else anchor
            if abs(level) <= max_correction:
                coarse[i, j] = level

    n_blocks = int(np.isfinite(coarse).sum())
    if n_blocks == 0:
        return np.zeros_like(resid), 0

    # NaN-aware Gaussian: smooth values and weights separately, then divide.
    # This both denoises the estimates and extrapolates into the dropped blocks.
    filled = np.where(np.isfinite(coarse), coarse, 0.0).astype(np.float32)
    weight = np.isfinite(coarse).astype(np.float32)
    num = gaussian_filter(filled, sigma=smooth_sigma, mode="nearest")
    den = gaussian_filter(weight, sigma=smooth_sigma, mode="nearest")
    smooth = num / np.maximum(den, 1e-6)
    # Blocks too far from any estimate fall back to the scene median
    smooth[den < 1e-6] = float(np.nanmedian(coarse))
    smooth = np.clip(smooth, -max_correction, max_correction)

    correction = np.empty((h, w), dtype=np.float32)
    for i, rows in enumerate(row_edges):
        for j, cols in enumerate(col_edges):
            correction[np.ix_(rows, cols)] = smooth[i, j]
    # Block edges would show up as steps in the nDSM without this
    px_sigma = smooth_sigma * max(h / len(row_edges), w / len(col_edges)) / 2.0
    correction = gaussian_filter(correction, sigma=px_sigma, mode="nearest")
    return correction, n_blocks


def _clamp_to_dsm(
    stage: np.ndarray, dsm: np.ndarray, clamp: bool
) -> tuple[np.ndarray, int, float]:
    """Keep a co-registered DTM from sitting above the DSM.

    A DTM above the DSM means negative nDSM over bare ground, which is
    physically impossible: nothing is below the surface. It happens when the
    local refinement over-lifts, so the count is worth reporting even when the
    clamp is off — it is the number that says how biased the block estimate
    still is.

    Deliberately not ``np.minimum``: that propagates a NaN from the DSM into a
    perfectly valid DTM pixel, turning a DSM gap into a DTM gap.
    """
    above = np.isfinite(dsm) & np.isfinite(stage) & (stage > dsm)
    n_above = int(above.sum())
    max_above = float((stage - dsm)[above].max()) if n_above else 0.0
    if clamp and n_above:
        stage = np.where(above, dsm, stage)
    return stage, n_above, max_above


def align_dtm_stages(
    dsm: np.ndarray,
    dtm: np.ndarray,
    max_shift: float = 20.0,
    blocks: int = 32,
    ground_quantile: float = 2.0,
    min_candidates: int = 50,
    min_extent: float = 0.25,
    clamp_to_dsm: bool = True,
    local_blocks: int = 12,
    local_quantile: float = 5.0,
    local_min_valid_frac: float = 0.2,
    local_max_correction: float = 1.0,
    local_smooth_sigma: float = 1.0,
    local_ground_band: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Vertically co-register an external DTM to the DSM, both stages at once.

    A DTM flown on a different date, or referenced to a different vertical
    datum, sits at a constant offset (and often a slight tilt) from the DSM.
    Differencing them directly then puts bare ground at several metres, which
    the [0,1] nDSM normalisation turns into "everything is canopy".

    Estimates that mis-levelling as a plane through the scene's ground pixels
    and returns the DTM lifted onto the DSM's ground level. Both inputs must
    already be on the same grid. The DTM on disk is never touched.

    A plane alone still leaves bare ground a few decimetres off zero, varying
    across the scene, because the two surveys are warped against each other.
    That residual warp is estimated on a coarse block grid and removed on top
    of the plane (see _local_ground_residual).

    The plane is fitted once and both stages are returned from it, so `plane`
    and `aligned` differ by exactly the local correction and nothing else.

    `local_blocks` defaults to 12 rather than a finer grid on purpose: on the
    5 cm reference grid that is a ~29 m block, and in half-vegetated terrain a
    block needs that much area before its low quantile lands on real bare
    ground instead of on low vegetation. Finer blocks measurably over-lift the
    DTM (the ring offsets in out/dsm_overview/dem_offsets.csv went from +0.17 m
    after the plane to -0.12 m after a 24-block refinement).

    Returns:
        stages: {"plane": DTM + plane, "aligned": DTM + plane + local warp},
                each the same shape/dtype as the input DTM
        infos:  the same keys, each {"mode", "mean_shift", "tilt",
                "n_candidates", "local_rms", "local_blocks", "n_above_dsm",
                "max_above_dsm"} for logging. The two `*_above_dsm` entries are
                measured *before* the clamp — clamping first would hide exactly
                the bias they exist to expose.
    """
    diff = (dsm - dtm).astype(np.float32)
    diff[np.isnan(dsm) | np.isnan(dtm)] = np.nan
    if not np.isfinite(diff).any():
        raise ValueError("DSM and DTM do not overlap — no valid pixels to align on")

    h, w = diff.shape
    # normalised pixel coordinates in [-1, 1] keep the least-squares well conditioned
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy = yy * (2.0 / max(h - 1, 1)) - 1.0
    xx = xx * (2.0 / max(w - 1, 1)) - 1.0

    cand = _block_ground_candidates(diff, blocks, ground_quantile)
    cy, cx, cd = yy[cand], xx[cand], diff[cand]

    # A plane needs candidates spread over both axes. When the valid data is a
    # thin strip the tilt terms are unconstrained and would extrapolate wildly,
    # so fall back to a constant shift.
    spread_ok = cd.size >= min_candidates and (
        cy.size > 0
        and (cy.max() - cy.min()) >= 2.0 * min_extent
        and (cx.max() - cx.min()) >= 2.0 * min_extent
    )

    if spread_ok:
        mode = "plane"
        keep = np.ones(cd.shape, dtype=bool)
        coef = None
        # Blocks lying entirely inside a canopy patch contribute candidates that
        # are metres too high. Two robust re-fits drop them.
        for _ in range(3):
            design = np.column_stack(
                [cx[keep], cy[keep], np.ones(int(keep.sum()), dtype=np.float32)]
            )
            coef, *_ = np.linalg.lstsq(design, cd[keep], rcond=None)
            resid = cd - (coef[0] * cx + coef[1] * cy + coef[2])
            centre = np.median(resid)
            mad = 1.4826 * np.median(np.abs(resid - centre))
            tol = max(2.5 * mad, 0.25)  # 0.25 m floor: real ground is rough
            new_keep = np.abs(resid - centre) <= tol
            if new_keep.sum() < min_candidates or np.array_equal(new_keep, keep):
                break
            keep = new_keep
        surface = coef[0] * xx + coef[1] * yy + coef[2]
        tilt = (abs(coef[0]) + abs(coef[1])) * 2.0  # metres across the full extent
        n_used = int(keep.sum())
    else:
        mode = "constant"
        pool = cd if cd.size else diff[np.isfinite(diff)]
        surface = np.full((h, w), float(np.median(pool)), dtype=np.float32)
        tilt = 0.0
        n_used = int(pool.size)

    # Plausibility check on the plane alone: a datum blunder shows up here, and
    # the local refinement is bounded to +-local_max_correction anyway.
    peak = float(np.nanmax(np.abs(surface)))
    if peak > max_shift:
        raise ValueError(
            f"DSM/DTM vertical mismatch of {peak:.2f} m exceeds max_shift="
            f"{max_shift:g} m — the two rasters are probably on different "
            "vertical datums or the DTM does not belong to this scene"
        )

    local, n_local_blocks = _local_ground_residual(
        diff - surface,
        blocks=local_blocks,
        quantile=local_quantile,
        min_valid_frac=local_min_valid_frac,
        max_correction=local_max_correction,
        smooth_sigma=local_smooth_sigma,
        ground_band=local_ground_band,
    )

    stages: dict[str, np.ndarray] = {}
    infos: dict[str, dict] = {}
    for name, correction, local_rms, n_blocks in (
        ("plane", surface, 0.0, 0),
        ("aligned", surface + local, float(np.sqrt(np.mean(local**2))), n_local_blocks),
    ):
        stage = (dtm + correction).astype(dtm.dtype)
        stage, n_above, max_above = _clamp_to_dsm(stage, dsm, clamp_to_dsm)
        stages[name] = stage.astype(dtm.dtype)
        infos[name] = {
            "mode": mode,
            "mean_shift": float(
                np.nanmean(np.where(np.isfinite(diff), correction, np.nan))
            ),
            "tilt": float(tilt),
            "n_candidates": n_used,
            "local_rms": local_rms,
            "local_blocks": n_blocks,
            "n_above_dsm": n_above,
            "max_above_dsm": max_above,
        }
    return stages, infos


def align_dtm_to_dsm(
    dsm: np.ndarray,
    dtm: np.ndarray,
    local_refine: bool = True,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """One stage of `align_dtm_stages`: the plane, or the plane plus the warp.

    Kept as the single-stage entry point for callers that only want the surface
    they are going to subtract. `local_refine=False` selects "plane".
    """
    stages, infos = align_dtm_stages(dsm, dtm, **kwargs)
    stage = "aligned" if local_refine else "plane"
    return stages[stage], infos[stage]


def detect_ground_dtm(
    dsm: np.ndarray,
    dtm: np.ndarray,
    height_threshold: float,
    height_ramp: float | None = None,
    max_shift: float = 20.0,
    stage: str = "aligned",
    clamp_to_dsm: bool = True,
    local_blocks: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Ground detection using an external DTM: nDSM = DSM - DTM.

    The DTM is vertically co-registered to the DSM first (see
    align_dtm_stages) — without it any datum offset between the two surveys
    propagates straight into the nDSM. `stage` picks which co-registration
    stage the nDSM is built on; both are returned either way so the caller can
    write them out and compare.

    Prediction time must pass the same `stage`, `clamp_to_dsm` and
    `local_blocks` as training did, or the two nDSMs disagree.

    Returns:
        binary:     bool array, True where pixel is ground
        confidence: float32 [0,1], higher = more likely ground
        ndsm:       raw nDSM (m) for diagnostics and nDSM output
        stages:     {"plane": ..., "aligned": ...}, the co-registered DTMs
    """
    if stage not in DTM_STAGES:
        raise ValueError(f"unknown dtm stage {stage!r}; expected one of {DTM_STAGES}")
    stages, infos = align_dtm_stages(
        dsm,
        dtm,
        max_shift=max_shift,
        clamp_to_dsm=clamp_to_dsm,
        local_blocks=local_blocks,
    )
    info = infos[stage]
    print(
        f"  DTM alignment ({info['mode']}, stage={stage}): mean shift "
        f"{info['mean_shift']:+.2f} m, tilt {info['tilt']:.2f} m "
        f"across scene, {info['n_candidates']:,} ground candidates"
    )
    if stage == "aligned":
        print(
            f"  Local refinement: RMS {info['local_rms']:.2f} m over "
            f"{info['local_blocks']:,} blocks with visible ground "
            f"({local_blocks} blocks per axis)"
        )
    for name in DTM_STAGES:
        print(
            f"  DTM above DSM ({name}): {infos[name]['n_above_dsm']:,} px, "
            f"worst {infos[name]['max_above_dsm']:.2f} m"
            f"{' — clamped' if clamp_to_dsm else ' — not clamped'}"
        )

    dtm = stages[stage]
    ndsm = (dsm - dtm).astype(np.float32)
    # NaN where either input is NaN
    ndsm[np.isnan(dsm) | np.isnan(dtm)] = np.nan

    confidence = _smoothstep_confidence(ndsm, height_threshold, height_ramp)
    confidence[np.isnan(ndsm)] = 0.0

    binary = (ndsm < height_threshold) & ~np.isnan(ndsm)
    return binary, confidence, ndsm, stages


def detect_ground_gradient(
    dsm: np.ndarray, gradient_sigma: float, gradient_threshold: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slope/edge filter: flat DSM regions are classified as ground.

    Computes Sobel gradient magnitude, smooths with Gaussian, then classifies
    flat pixels (low gradient) as ground.

    Returns:
        binary:      bool array, True where pixel is ground
        confidence:  float32 [0,1], higher = more likely ground
        grad_smooth: smoothed gradient magnitude (for diagnostics)
    """
    dsm_filled = np.where(np.isnan(dsm), 0.0, dsm)
    gx = sobel(dsm_filled, axis=1)
    gy = sobel(dsm_filled, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    grad_smooth = gaussian_filter(grad_mag, sigma=gradient_sigma)

    threshold = gradient_threshold if gradient_threshold is not None else _otsu_threshold(grad_smooth)
    print(f"  Gradient threshold: {threshold:.4f}  ({'manual' if gradient_threshold is not None else 'Otsu'})")

    valid_pos = grad_smooth[grad_smooth > 0]
    p95 = float(np.percentile(valid_pos, 95)) if valid_pos.size > 0 else float(grad_smooth.max())
    # normiert auf [0, p95]; 1e-8 verhindert Division durch 0; invertiert: niedriger Gradient → hohe Boden-Confidence
    confidence = (1.0 - np.clip(grad_smooth / max(p95, 1e-8), 0.0, 1.0)).astype(np.float32)
    confidence[np.isnan(dsm)] = 0.0

    binary = (grad_smooth < threshold) & ~np.isnan(dsm)
    return binary, confidence, grad_smooth


def combine(
    a_bin: np.ndarray, a_conf: np.ndarray,
    b_bin: np.ndarray, b_conf: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two (binary, confidence) ground-detection results.

    mode='or':  ground if EITHER method agrees; confidence = max
    mode='and': ground if BOTH methods agree;   confidence = min
    """
    if mode == "or":
        return a_bin | b_bin, np.maximum(a_conf, b_conf)
    return a_bin & b_bin, np.minimum(a_conf, b_conf)


def apply_soft_blend(
    mask: np.ndarray,
    ground_conf: np.ndarray,
    nodata_resolve_threshold: float,
    crown_resolve_threshold: float | None = None,
    height_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Soft-blend ground confidence into the crown mask.

    Out-of-footprint pixels (MASK_OUTSIDE = -1) are never touched: no amount of
    DSM confidence can invent imagery the drone did not record. Both guards
    below (`mask >= 0.0` and `mask == MASK_UNLABELLED`) exclude them already.

    Crown pixels (0–1): multiplied by (1 - ground_conf).
    Unlabelled pixels (255): resolved to (1 - ground_conf) when
      ground_conf >= nodata_resolve_threshold (confident ground), or when
      ground_conf <= crown_resolve_threshold (confident crown, requires
      height_valid there — pixels without DSM or DTM data also carry conf 0.0
      and must stay 255).
      The gray zone in between is kept at 255.
    """
    if crown_resolve_threshold is not None and not (
        crown_resolve_threshold < nodata_resolve_threshold
    ):
        raise ValueError(
            "crown_resolve_threshold must be < nodata_resolve_threshold"
        )
    result = mask.copy()

    # Alle gültigen Kronenpixel (Konfidenz 0–1, kein noData-Sentinel)
    crown = (mask >= 0.0) & (mask < MASK_UNLABELLED)
    # Krone × (1 – Bodenwahrscheinlichkeit): hohe Bodenkonf. → Kronenwert sinkt gegen 0
    result[crown] = mask[crown] * (1.0 - ground_conf[crown])

    # noData-Pixel (Sentinel 255): außerhalb des Bildbereichs oder nicht klassifiziert
    nodata = mask == MASK_UNLABELLED
    # Wenn der DSM-Detektor trotzdem sicher Boden erkennt, Pixel auflösen statt 255 zu behalten
    resolve = nodata & (ground_conf >= nodata_resolve_threshold)
    # Aufgelöste noData-Pixel bekommen Bodenwahrscheinlichkeit als invertierte Kronenkonfidenz
    result[resolve] = 1.0 - ground_conf[resolve]

    # Symmetrische Auflösung Richtung Krone: sicher NICHT Boden (hohe Vegetation
    # ohne Polygon) wird als Krone gelabelt statt vom Loss ausgeschlossen
    if crown_resolve_threshold is not None:
        resolve_crown = nodata & (ground_conf <= crown_resolve_threshold)
        if height_valid is not None:
            resolve_crown &= height_valid
        result[resolve_crown] = 1.0 - ground_conf[resolve_crown]
    return result


def _save_diagnostic(
    values: np.ndarray,
    process_out_dir: str,
    run_id: str,
    mask_stem: str,
    label: str,
    used_threshold: float,
    suggested_threshold: float | None = None,
    xlabel: str = "value",
    title: str = "distribution",
) -> None:
    """Save a histogram diagnostic PNG to diag_graphs/<run_id>/ under process_out_dir."""
    diag_dir = os.path.join(process_out_dir, "diag_graphs", run_id)
    os.makedirs(diag_dir, exist_ok=True)

    diag_path = os.path.join(diag_dir, f"{mask_stem}_{label}_diag.png")

    valid = values[np.isfinite(values) & (values >= 0)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid.ravel(), bins=300, color="steelblue", alpha=0.7, log=True)
    ax.axvline(used_threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"used  {used_threshold:.3f}")
    if suggested_threshold is not None:
        ax.axvline(suggested_threshold, color="orange", linestyle=":", linewidth=1.5,
                   label=f"suggested  {suggested_threshold:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pixel count (log)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(diag_path, dpi=120)
    plt.close(fig)
    print(f"  Diagnostic plot: {diag_path}")


def _save_float_tif(
    arr: np.ndarray, path: str, profile: dict, label: str, nodata=None
) -> None:
    """Write a single-band float32 raster on the mask grid to disk."""
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=nodata)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr[np.newaxis].astype(np.float32))
    print(f"  Saved {label}: {path}")


def _save_conf_tif(arr: np.ndarray, path: str, profile: dict) -> None:
    """Write a float32 confidence raster [0,1] to disk."""
    _save_float_tif(arr, path, profile, label="confidence")


def main(args):
    logger.info("Config:\n%s", OmegaConf.to_yaml(args))

    # --- Run ID (timestamp) groups all outputs of this run -------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Auto-derive output paths from mask path if not configured -----------
    _mask_abs = os.path.abspath(args.mask)
    _mask_dir = os.path.dirname(_mask_abs)
    _mask_stem = os.path.splitext(os.path.basename(_mask_abs))[0]
    process_out_dir = os.path.dirname(_mask_dir)

    if not getattr(args, "out", None):
        args.out = os.path.join(_mask_dir, f"{_mask_stem}_final.tif")
    if not getattr(args, "out_dsm", None):
        args.out_dsm = os.path.join(process_out_dir, "ndsm", "dsm_ndsm.tif")

    # --- Build output filename suffix ----------------------------------------
    w_tag = "-".join(str(w) for w in args.windows)
    lm_tag = f"_lm_w{w_tag}_ht{args.height_threshold}"
    dt_tag = f"_dtm_ht{args.height_threshold}"
    gr_tag = f"_gr_s{args.gradient_sigma}"

    # Determine which height-based tag to use (local_min or external dtm)
    use_external_dtm = bool(getattr(args, "dtm", None))
    ht_tag = dt_tag if use_external_dtm else lm_tag

    if args.method == "local_min":
        mask_suffix = lm_tag
    elif args.method == "dtm":
        mask_suffix = dt_tag
    elif args.method == "gradient":
        mask_suffix = gr_tag
    else:  # both
        mask_suffix = f"{ht_tag}{gr_tag}_{args.combine}"

    height_ramp = args.get("height_ramp", None)
    if args.get("dtm_local_refine", None) is not None:
        raise ValueError(
            "dsm_mask.dtm_local_refine was replaced by dtm_stage: use "
            "'dtm_stage: aligned' for the old true, 'dtm_stage: plane' for false"
        )
    dtm_stage = str(args.get("dtm_stage", "aligned"))
    if dtm_stage not in DTM_STAGES:
        raise ValueError(
            f"dsm_mask.dtm_stage is {dtm_stage!r}; expected one of {DTM_STAGES}"
        )
    dtm_local_blocks = int(args.get("dtm_local_blocks", 12))
    dtm_clamp_to_dsm = bool(args.get("dtm_clamp_to_dsm", True))
    crown_resolve_threshold = args.get("crown_resolve_threshold", None)
    if use_external_dtm and args.method in ("dtm", "both"):
        # Two runs that differ only in the stage must not overwrite each other
        mask_suffix += f"_{dtm_stage}_b{dtm_local_blocks}"
    if height_ramp is not None:
        mask_suffix += f"_hr{height_ramp}"
    if crown_resolve_threshold is not None:
        mask_suffix += f"_cr{crown_resolve_threshold}"

    args.out = _embed_params(args.out, mask_suffix)
    mask_dir, mask_file = os.path.dirname(args.out), os.path.basename(args.out)
    args.out = os.path.join(mask_dir, run_id, mask_file)

    if args.out_dsm:
        ndsm_suffix = "_dtm" if use_external_dtm else f"_w{w_tag}"
        args.out_dsm = _embed_params(args.out_dsm, ndsm_suffix)
        dsm_dir, dsm_file = os.path.dirname(args.out_dsm), os.path.basename(args.out_dsm)
        args.out_dsm = os.path.join(dsm_dir, run_id, dsm_file)

    # Derive confidence output path
    conf_dir = os.path.join(process_out_dir, "ground_confidence", run_id)
    conf_stem = os.path.splitext(os.path.basename(args.out))[0] + "_conf.tif"
    conf_path = os.path.join(conf_dir, conf_stem)

    # --- Load mask + DSM -----------------------------------------------------
    with rasterio.open(args.mask) as src:
        mask = src.read(1).astype(np.float32)
        h, w = src.height, src.width
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    print(f"Mask grid: {h} x {w}")
    print("Resampling DSM to mask grid...")
    dsm = resample_raster(args.dsm, h, w, transform, crs)

    dtm = None
    if use_external_dtm:
        print("Resampling external DTM to mask grid...")
        dtm = resample_raster(args.dtm, h, w, transform, crs)

    # --- Run detection method(s) ---------------------------------------------
    lm_bin = lm_conf = ndsm = None
    gr_bin = gr_conf = grad_smooth = None
    dtm_stages: dict[str, np.ndarray] = {}

    mask_stem = os.path.splitext(os.path.basename(args.out))[0]

    if args.method in ("dtm", "both") and use_external_dtm and dtm is not None:
        print(f"\n[dtm] external DTM  height_threshold={args.height_threshold} m  ramp={height_ramp}")
        lm_bin, lm_conf, ndsm, dtm_stages = detect_ground_dtm(
            dsm,
            dtm,
            args.height_threshold,
            height_ramp,
            stage=dtm_stage,
            clamp_to_dsm=dtm_clamp_to_dsm,
            local_blocks=dtm_local_blocks,
        )

    if args.method in ("local_min", "both") and not use_external_dtm:
        print(f"\n[local_min] windows={args.windows} px  height_threshold={args.height_threshold} m  ramp={height_ramp}")
        lm_bin, lm_conf, ndsm = detect_ground_local_min(dsm, args.windows, args.height_threshold, height_ramp)

    if args.method in ("local_min", "dtm", "both") and ndsm is not None:
        valid_ndsm = ndsm[~np.isnan(ndsm)]
        suggested_ht = _find_valley_threshold(ndsm)
        print(f"  nDSM  min={np.min(valid_ndsm):.2f} m  "
              f"p5={np.percentile(valid_ndsm, 5):.2f} m  "
              f"p25={np.percentile(valid_ndsm, 25):.2f} m  "
              f"median={np.median(valid_ndsm):.2f} m  "
              f"p75={np.percentile(valid_ndsm, 75):.2f} m  "
              f"p95={np.percentile(valid_ndsm, 95):.2f} m  "
              f"max={np.max(valid_ndsm):.2f} m")
        print(f"  Suggested threshold (valley): {suggested_ht:.2f} m  (used: {args.height_threshold} m)")
        ndsm_diag_label = "dtm" if use_external_dtm else "lm"
        ndsm_diag_title = ("dtm — nDSM distribution" if use_external_dtm
                           else "local_min — nDSM distribution")
        _save_diagnostic(
            ndsm, process_out_dir, run_id, mask_stem, label=ndsm_diag_label,
            used_threshold=args.height_threshold,
            suggested_threshold=suggested_ht,
            xlabel="nDSM [m]",
            title=ndsm_diag_title,
        )

    if args.method in ("gradient", "both"):
        print(f"\n[gradient] sigma={args.gradient_sigma} px")
        gr_bin, gr_conf, grad_smooth = detect_ground_gradient(
            dsm, args.gradient_sigma, args.gradient_threshold
        )
        # NOTE: _otsu_threshold is also called inside detect_ground_gradient when gradient_threshold is None
        used_gr_threshold = (args.gradient_threshold if args.gradient_threshold is not None
                             else _otsu_threshold(grad_smooth))
        _save_diagnostic(
            grad_smooth, process_out_dir, run_id, mask_stem, label="gr",
            used_threshold=used_gr_threshold,
            xlabel="gradient magnitude",
            title="gradient — slope distribution",
        )

    # --- Save both co-registered DTM stages ----------------------------------
    # Both are written whatever `dtm_stage` selects: comparing them is how the
    # co-registration is judged (see dsm_overview), and the one not used here
    # is exactly the comparison partner.
    if dtm_stages:
        coreg_dir = os.path.join(process_out_dir, "dtm_coreg", run_id)
        for name, stage_dtm in dtm_stages.items():
            _save_float_tif(
                stage_dtm,
                os.path.join(coreg_dir, f"{mask_stem}_dtm_{name}.tif"),
                profile,
                label=f"co-registered DTM ({name}{', used' if name == dtm_stage else ''})",
                nodata=float("nan"),
            )

    # --- Save individual confidences -----------------------------------------
    if lm_conf is not None:
        lm_conf_label = "dtm" if use_external_dtm else "lm"
        _save_conf_tif(lm_conf, os.path.join(conf_dir, f"{mask_stem}_{lm_conf_label}_conf.tif"), profile)
    if gr_conf is not None:
        _save_conf_tif(gr_conf, os.path.join(conf_dir, f"{mask_stem}_gr_conf.tif"), profile)

    # --- Combine -------------------------------------------------------------
    if args.method in ("local_min", "dtm"):
        ground_bin, ground_conf = lm_bin, lm_conf
    elif args.method == "gradient":
        ground_bin, ground_conf = gr_bin, gr_conf
    else:
        print(f"\n[combine] mode={args.combine}")
        ground_bin, ground_conf = combine(lm_bin, lm_conf, gr_bin, gr_conf, mode=args.combine)

    # --- Apply soft ground blend to crown mask --------------------------------
    n_crown_dampened = int(np.sum((mask >= 0.0) & (mask < MASK_UNLABELLED) & (ground_conf > 0.0)))
    nodata_before = mask == MASK_UNLABELLED
    height_valid = height_data_valid(ndsm, dsm)
    n_no_height = int(np.sum(~height_valid & ~np.isnan(dsm)))
    if n_no_height:
        print(f"Pixels with DSM but no DTM data (kept as noData): {n_no_height:,}")
    mask = apply_soft_blend(
        mask,
        ground_conf,
        args.nodata_resolve_threshold,
        crown_resolve_threshold=crown_resolve_threshold,
        height_valid=height_valid,
    )
    n_ground_resolved = int(np.sum(nodata_before & (ground_conf >= args.nodata_resolve_threshold)))
    n_crown_resolved = 0
    if crown_resolve_threshold is not None:
        n_crown_resolved = int(np.sum(nodata_before & height_valid & (ground_conf <= crown_resolve_threshold)))
    n_crown  = int(np.sum((mask > 0) & (mask < MASK_UNLABELLED)))
    n_ground = int(np.sum(mask == 0.0))
    n_nodata = int(np.sum(mask == MASK_UNLABELLED))
    n_outside = int(np.sum(mask == MASK_OUTSIDE))
    print(f"\nCrown pixels dampened by DSM:     {n_crown_dampened:,}  (multiplicative blend)")
    print(f"noData pixels resolved to ground: {n_ground_resolved:,}  (ground_conf >= {args.nodata_resolve_threshold:.2f})")
    if crown_resolve_threshold is not None:
        print(f"noData pixels resolved to crown:  {n_crown_resolved:,}  (ground_conf <= {crown_resolve_threshold:.2f})")
    print(f"Final  =>  Crown: {n_crown:,}  Ground: {n_ground:,}  "
          f"unlabelled: {n_nodata:,}  outside footprint: {n_outside:,}")

    # --- Write binary mask ---------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # The out-of-footprint sentinel is what a GIS should render transparent;
    # unlabelled pixels are inside the scene and stay visible as 255.
    profile.update(dtype="float32", count=1, nodata=MASK_RASTER_NODATA)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(mask[np.newaxis])
    print(f"Saved mask: {args.out}")

    # --- Write combined confidence raster ------------------------------------
    _save_conf_tif(ground_conf, conf_path, profile)

    # --- Write normalised nDSM (local_min method only) -----------------------
    if args.out_dsm and ndsm is not None:
        ndsm_norm = normalize_ndsm(ndsm, dsm, args.max_ndsm_height)
        dsm_profile = profile.copy()
        dsm_profile.update(dtype="float32", count=1, nodata=None)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_dsm)), exist_ok=True)
        with rasterio.open(args.out_dsm, "w", **dsm_profile) as dst:
            dst.write(ndsm_norm[np.newaxis].astype(np.float32))
        print(f"Saved nDSM: {args.out_dsm}  (range [0,1])")

    # --- Write raw nDSM in metres ---------------------------------------------
    if ndsm is not None:
        ndsm_m_dir = os.path.join(process_out_dir, "ndsm_in_m", run_id)
        os.makedirs(ndsm_m_dir, exist_ok=True)
        ndsm_stem = os.path.splitext(os.path.basename(args.out_dsm))[0] if args.out_dsm else os.path.splitext(os.path.basename(args.out))[0]
        ndsm_m_path = os.path.join(ndsm_m_dir, f"{ndsm_stem}_raw_m.tif")
        dsm_profile_m = profile.copy()
        dsm_profile_m.update(dtype="float32", count=1, nodata=float("nan"))
        ndsm_out = ndsm.astype(np.float32)
        ndsm_out[np.isnan(dsm)] = float("nan")
        with rasterio.open(ndsm_m_path, "w", **dsm_profile_m) as dst:
            dst.write(ndsm_out[np.newaxis])
        print(f"Saved raw nDSM (m): {ndsm_m_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Stage 1b: apply DSM ground mask.")
    p.add_argument("--config", required=True, help="Path to preprocess.yaml")
    cfg = OmegaConf.load(p.parse_args().config)
    main(cfg.dsm_mask)
