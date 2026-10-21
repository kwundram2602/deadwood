"""
apply_dsm_mask.py

Refine the soft crown mask by labelling confirmed ground pixels using the DSM.

Ground detection supports two methods (dsm_mask.method):

  dtm_approx — terrain approximated from the DSM alone by a progressive
               morphological filter (see approximate_dtm; written to
               process_out/dtm_approx/<run_id>/):
                 nDSM = DSM - approximate_dtm(DSM)
                 pixels where nDSM < height_threshold  =>  ground = 0.0

  dtm        — External DTM raster (resampled to DSM grid, then vertically
               co-registered onto the DSM in two stages — see
               align_dtm_stages; `dtm_real.stage` picks the one used, both are
               written to process_out/dtm_coreg/<run_id>/):
                 nDSM = DSM - DTM
                 pixels where nDSM < height_threshold  =>  ground = 0.0

`method` only decides the ground truth (the mask). The nDSM written as the
model's input channel comes from `ndsm_channel` (dtm_approx | dtm, default:
same as `method`). Masks from the external DTM with an input channel from
dtm_approx is the setup for scenes where prediction has no DTM: the labels
use the best terrain available, the channel uses what inference will have.

Ground pixels soft-blend into the crown mask: crown pixels (0–1) are
multiplied by (1 − ground_conf); noData pixels are resolved to ground
only when ground_conf exceeds --nodata_resolve_threshold.

preserve_crown_threshold exempts the confident stage-1a crown pixels from that
blend: with 1.0, everything a crown polygon covers outright keeps its label
whatever the height model says.

Remaining noData pixels (255) — no crown polygon, not confirmed ground —
are excluded from the loss during training.

Usage:
  python explore_and_process/apply_dsm_mask.py \\
      --mask  explore_and_process/out/crown_mask.tif \\
      --dsm   data/raster/DSM.tif \\
      --out   explore_and_process/out/crown_mask_final.tif \\
      [--method dtm --dtm data/raster/DTM.tif] \\
      [--height_threshold 2.0]
"""

import argparse
import hashlib
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import rasterio
from omegaconf import OmegaConf
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.interpolate import griddata
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
    maximum_filter,
    minimum_filter,
)

from utils.nodata import MASK_OUTSIDE, MASK_RASTER_NODATA, MASK_UNLABELLED

logger = logging.getLogger(__name__)

# The two stages of the DSM/DTM co-registration, in the order they are built:
# "plane" removes the global offset and tilt, "aligned" adds the blockwise
# residual warp on top. Both are written to disk; `dtm_real.stage` picks the one the
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


@dataclass(frozen=True)
class DtmApproxParams:
    """Parameters of the DSM-only terrain approximation (dsm_mask.dtm_approx).

    cell:         coarse-grid cell size (m) the morphology runs on
    percentile:   per-cell percentile of the DSM — a robust minimum
    max_window:   widest object (m) the opening removes
    window_step:  growth of the opening window per step (m)
    max_slope:    tan of the steepest flank still counted as terrain
    dh0:          height tolerance per step (m) for DSM noise
    smooth_sigma: Gaussian sigma (m) of the final terrain surface
    """

    cell: float = 0.5
    percentile: float = 5.0
    max_window: float = 30.0
    window_step: float = 1.0
    max_slope: float = 0.5
    dh0: float = 0.2
    smooth_sigma: float = 1.0

    @classmethod
    def from_cfg(cls, cfg) -> "DtmApproxParams":
        if cfg is None:
            raise ValueError("method: dtm_approx needs a dsm_mask.dtm_approx block")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"dtm_approx: unknown keys {unknown}; expected {sorted(known)}")
        params = cls(**{k: float(v) for k, v in dict(cfg).items()})
        if not 0.0 <= params.percentile <= 100.0:
            raise ValueError(f"dtm_approx.percentile must be in [0, 100], got {params.percentile}")
        for name in ("cell", "max_window", "window_step", "max_slope"):
            if getattr(params, name) <= 0.0:
                raise ValueError(f"dtm_approx.{name} must be > 0")
        if params.dh0 < 0.0 or params.smooth_sigma < 0.0:
            raise ValueError("dtm_approx.dh0 and smooth_sigma must be >= 0")
        return params

    def tag(self) -> str:
        """Short filename tag; the full set goes into the cache hash."""
        return f"_da_w{self.max_window:g}_s{self.max_slope:g}"

    def digest(self) -> str:
        return hashlib.md5(repr(self).encode()).hexdigest()[:8]


@dataclass(frozen=True)
class DtmRealParams:
    """The external DTM and its co-registration onto the DSM (dsm_mask.dtm_real).

    path:         the external DTM raster (not needed by predict, whose
                  scene inputs carry it)
    stage:        co-registration stage the nDSM is built on (DTM_STAGES):
                  plane = global offset + tilt, aligned = plus blockwise warp
    local_blocks: blocks per axis of the residual-warp correction
    clamp_to_dsm: keep the co-registered DTM from rising above the DSM
    """

    path: str | None = None
    stage: str = "aligned"
    local_blocks: int = 12
    clamp_to_dsm: bool = True

    @classmethod
    def from_cfg(cls, cfg, need_path: bool = True) -> "DtmRealParams":
        if cfg is None:
            raise ValueError("the external DTM needs a dtm_real block")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"dtm_real: unknown keys {unknown}; expected {sorted(known)}")
        raw = dict(cfg)
        params = cls(
            path=str(raw["path"]) if raw.get("path") else None,
            stage=str(raw.get("stage", cls.stage)),
            local_blocks=int(raw.get("local_blocks", cls.local_blocks)),
            clamp_to_dsm=bool(raw.get("clamp_to_dsm", cls.clamp_to_dsm)),
        )
        if need_path and not params.path:
            raise ValueError("dtm_real.path is not set")
        if params.stage not in DTM_STAGES:
            raise ValueError(f"dtm_real.stage is {params.stage!r}; expected one of {DTM_STAGES}")
        if params.local_blocks < 1:
            raise ValueError("dtm_real.local_blocks must be >= 1")
        return params

    def tag(self) -> str:
        """Filename tag: runs that differ in co-registration must not share names."""
        return f"_{self.stage}_b{self.local_blocks}{'' if self.clamp_to_dsm else '_noclamp'}"


def _block_percentile(dsm: np.ndarray, f: int, q: float) -> np.ndarray:
    """q-th percentile of the finite pixels in every f×f block; NaN for empty blocks."""
    h, w = dsm.shape
    hc, wc = -(-h // f), -(-w // f)
    padded = np.full((hc * f, wc * f), np.nan, dtype=np.float32)
    padded[:h, :w] = dsm
    blocks = padded.reshape(hc, f, wc, f).transpose(0, 2, 1, 3).reshape(hc, wc, f * f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
        return np.nanpercentile(blocks, q, axis=2).astype(np.float32)


def _masked_opening(surface: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    """Grey opening with a size×size square that ignores cells outside *valid*.

    Outside cells count as +inf for the erosion and −inf for the dilation, so
    "no data" never acts as ground: a tree at the footprint edge is still
    opened away by the ground on its inner side, and the edge is not pulled
    down by a fill value.
    """
    eroded = minimum_filter(np.where(valid, surface, np.inf), size=size, mode="nearest")
    eroded[~valid] = -np.inf
    opened = maximum_filter(eroded, size=size, mode="nearest")
    opened[~valid] = np.nan
    return opened


def _fill_nearest(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with the value of the nearest finite cell."""
    missing = ~np.isfinite(arr)
    if not missing.any() or missing.all():
        return arr
    idx = distance_transform_edt(missing, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def _nan_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN cells and keeps them NaN."""
    if sigma <= 0.0:
        return arr
    valid = np.isfinite(arr)
    num = gaussian_filter(np.where(valid, arr, 0.0), sigma=sigma, mode="nearest")
    den = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    out = num / np.maximum(den, 1e-6)
    out[~valid] = np.nan
    return out.astype(np.float32)


def approximate_dtm(dsm: np.ndarray, gsd: float, params: DtmApproxParams) -> np.ndarray:
    """Terrain surface from the DSM alone: a progressive morphological filter.

    After Zhang et al. (2003, PMF) and Pingel et al. (2013, SMRF), on a coarse
    grid of `cell` m:

    1. z_0 = per-cell `percentile` of the DSM (robust minimum: single pits and
       holes do not pull the cell down).
    2. For k = 1..K, window w_k = 2·k·r + 1 cells (r = window_step / 2 / cell,
       up to max_window):
           z_k = opening(z_{k-1}, w_k)
           object  if  z_{k-1} − z_k > dh,   dh = dh0 + max_slope · window_step / 2
       The opening keeps everything wider than w_k and lowers a narrower
       feature to a plateau whose edge moves by window_step/2 per step. Cells
       on that plateau therefore drop by (flank slope) · window_step/2 per step:
       an embankment with gentle flanks never exceeds dh and stays terrain,
       however narrow; a crown or hedge with a steep flank drops by more than
       dh once the window outgrows it, and every cell above that flank is
       marked. Marks are permanent.
    3. Marked cells are re-interpolated (linear, nearest outside the hull)
       from the unmarked cells, which keep their z_0 value.
    4. Gaussian smoothing (smooth_sigma), bilinear upsampling to the DSM grid,
       terrain ≤ DSM, NaN where the DSM has no data.

    Limits: under closed canopy wider than max_window the DSM shows no
    ground, and the result there is the canopy. A feature counts as terrain
    when its flanks are no steeper than about
    max_slope + 2·dh0 / window_step — not by its width.
    """
    f = max(1, int(round(params.cell / gsd)))
    cell = f * gsd
    r_step = max(1, int(round(params.window_step / 2.0 / cell)))
    n_steps = max(1, int(round(params.max_window / (2.0 * r_step * cell))))
    dh = params.dh0 + params.max_slope * r_step * cell

    z0 = _block_percentile(dsm, f, params.percentile)
    valid = np.isfinite(z0)
    if not valid.any():
        return np.full(dsm.shape, np.nan, dtype=np.float32)

    surface = z0.copy()
    is_object = np.zeros(z0.shape, dtype=bool)
    for k in range(1, n_steps + 1):
        opened = _masked_opening(surface, valid, 2 * k * r_step + 1)
        is_object |= valid & (surface - opened > dh)
        surface = opened

    ground = valid & ~is_object
    print(
        f"  dtm_approx: {cell:g} m cells, {n_steps} openings up to "
        f"{(2 * n_steps * r_step + 1) * cell:g} m, dh={dh:.2f} m — "
        f"{int(is_object.sum()):,} / {int(valid.sum()):,} cells marked as object"
    )

    terrain = np.where(ground, z0, np.nan).astype(np.float32)
    fill = valid & ~ground
    if fill.any() and ground.sum() >= 3:
        pts = np.argwhere(ground)
        vals = z0[ground]
        targets = np.argwhere(fill)
        lin = griddata(pts, vals, targets, method="linear")
        hull_miss = ~np.isfinite(lin)
        if hull_miss.any():
            lin[hull_miss] = griddata(pts, vals, targets[hull_miss], method="nearest")
        terrain[fill] = lin
    elif fill.any():
        terrain[fill] = z0[fill]

    terrain = _nan_gaussian(terrain, params.smooth_sigma / cell)

    # Back to the DSM grid: coarse cell i covers fine rows i·f .. i·f+f−1,
    # its centre sits at fine coordinate i·f + (f−1)/2.
    h, w = dsm.shape
    rows = (np.arange(h, dtype=np.float32) - (f - 1) / 2.0) / f
    cols = (np.arange(w, dtype=np.float32) - (f - 1) / 2.0) / f
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    dtm = map_coordinates(_fill_nearest(terrain), [rr, cc], order=1, mode="nearest")
    dtm = dtm.astype(np.float32)
    finite = np.isfinite(dsm)
    dtm[finite] = np.minimum(dtm[finite], dsm[finite])
    dtm[~finite] = np.nan
    return dtm


def detect_ground_dtm_approx(
    dsm: np.ndarray,
    gsd: float,
    params: DtmApproxParams,
    height_threshold: float,
    height_ramp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ground detection against the DSM-only terrain approximation.

    nDSM = DSM − approximate_dtm(DSM); confidence as for method 'dtm'.

    Returns:
        binary:     bool array, True where pixel is ground
        confidence: float32 [0,1], higher = more likely ground
        ndsm:       raw nDSM (m) for diagnostics and nDSM output
        dtm:        the approximated terrain surface (m)
    """
    dtm = approximate_dtm(dsm, gsd, params)
    ndsm = dsm - dtm

    confidence = _smoothstep_confidence(ndsm, height_threshold, height_ramp)
    confidence[np.isnan(ndsm)] = 0.0

    binary = (ndsm < height_threshold) & ~np.isnan(ndsm)
    return binary, confidence, ndsm, dtm


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


def preserved_crown(
    mask: np.ndarray, preserve_crown_threshold: float | None
) -> np.ndarray:
    """Stage-1a crown pixels the DSM is not allowed to overwrite.

    Everything the polygons labelled at or above `preserve_crown_threshold`
    (1.0 = only the pixels a polygon covers outright) keeps its stage-1a value:
    the digitised crown outranks the height model there. None disables it.
    """
    if preserve_crown_threshold is None:
        return np.zeros_like(mask, dtype=bool)
    return (mask >= preserve_crown_threshold) & (mask < MASK_UNLABELLED)


def apply_soft_blend(
    mask: np.ndarray,
    ground_conf: np.ndarray,
    nodata_resolve_threshold: float,
    crown_resolve_threshold: float | None = None,
    height_valid: np.ndarray | None = None,
    preserve_crown_threshold: float | None = None,
) -> np.ndarray:
    """Soft-blend ground confidence into the crown mask.

    Out-of-footprint pixels (MASK_OUTSIDE = -1) are never touched: no amount of
    DSM confidence can invent imagery the drone did not record. Both guards
    below (`mask >= 0.0` and `mask == MASK_UNLABELLED`) exclude them already.

    Crown pixels (0–1): multiplied by (1 - ground_conf), except those at or
      above `preserve_crown_threshold`, which keep their stage-1a value.
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
    if preserve_crown_threshold is not None and not (
        0.0 < preserve_crown_threshold <= 1.0
    ):
        raise ValueError(
            "preserve_crown_threshold must be in (0, 1] or null"
        )
    result = mask.copy()

    # labelled crown pixels (0–1), minus preserved stage-1a crowns
    crown = (mask >= 0.0) & (mask < MASK_UNLABELLED)
    crown &= ~preserved_crown(mask, preserve_crown_threshold)
    result[crown] = mask[crown] * (1.0 - ground_conf[crown])

    # unlabelled pixels: confident ground → label 1 - ground_conf
    nodata = mask == MASK_UNLABELLED
    resolve = nodata & (ground_conf >= nodata_resolve_threshold)
    result[resolve] = 1.0 - ground_conf[resolve]

    # confident non-ground (tall vegetation without polygon) → crown label
    if crown_resolve_threshold is not None:
        resolve_crown = nodata & (ground_conf <= crown_resolve_threshold)
        if height_valid is not None:
            resolve_crown &= height_valid
        result[resolve_crown] = 1.0 - ground_conf[resolve_crown]
    return result


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


# Keys of removed methods, with the hint shown when a config still has them
REMOVED_KEYS = {
    "combine": "removed together with the gradient method — delete it",
    "gradient_sigma": "removed together with the gradient method — delete it",
    "gradient_threshold": "removed together with the gradient method — delete it",
    "windows": "local_min was replaced by dtm_approx — use the dtm_approx block",
    "dtm": "moved to dtm_real.path",
    "dtm_stage": "moved to dtm_real.stage",
    "dtm_local_blocks": "moved to dtm_real.local_blocks",
    "dtm_clamp_to_dsm": "moved to dtm_real.clamp_to_dsm",
    "dtm_local_refine": "was replaced by dtm_real.stage (aligned = old true, plane = old false)",
}
METHODS = ("dtm_approx", "dtm")


def validate_method(args) -> tuple[str, str]:
    """Check dsm_mask.method / ndsm_channel and their keys.

    Returns (method, ndsm_channel): `method` builds the mask (ground truth),
    `ndsm_channel` the nDSM written as model input. The external DTM and its
    co-registration settings (dtm_real) are read only where one of them is
    'dtm', never because a path happens to be set.
    """
    stale = [k for k in REMOVED_KEYS if k in args]
    if stale:
        raise ValueError(
            "dsm_mask: " + "; ".join(f"'{k}' {REMOVED_KEYS[k]}" for k in stale)
        )
    method = str(args.method)
    if method == "local_min":
        raise ValueError("dsm_mask.method: local_min was renamed to dtm_approx")
    if method not in METHODS:
        raise ValueError(f"dsm_mask.method is {method!r}; expected one of {METHODS}")
    channel = str(args.get("ndsm_channel", None) or method)
    if channel not in METHODS:
        raise ValueError(
            f"dsm_mask.ndsm_channel is {channel!r}; expected one of {METHODS} or null"
        )
    used = {method, channel}
    if "dtm" in used:
        try:
            DtmRealParams.from_cfg(args.get("dtm_real", None))
        except ValueError as e:
            raise ValueError(f"dsm_mask.method / ndsm_channel 'dtm': {e}") from e
    elif args.get("dtm_real", None) is not None:
        print("[dtm_approx] dsm_mask.dtm_real is ignored (no 'dtm' in method / ndsm_channel)")
    if "dtm_approx" in used:
        DtmApproxParams.from_cfg(args.get("dtm_approx", None))
    return method, channel


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

    # --- Method (mask) and nDSM channel --------------------------------------
    method, channel = validate_method(args)
    used = {method, channel}
    real = DtmRealParams.from_cfg(args.dtm_real) if "dtm" in used else None
    approx = DtmApproxParams.from_cfg(args.dtm_approx) if "dtm_approx" in used else None
    if channel != method:
        print(f"Mask from '{method}', nDSM input channel from '{channel}'")

    # --- Build output filename suffix ----------------------------------------
    if method == "dtm":
        mask_suffix = f"_dtm_ht{args.height_threshold}"
    else:
        assert approx is not None
        mask_suffix = f"{approx.tag()}_ht{args.height_threshold}"

    height_ramp = args.get("height_ramp", None)
    crown_resolve_threshold = args.get("crown_resolve_threshold", None)
    preserve_crown_threshold = args.get("preserve_crown_threshold", None)
    if preserve_crown_threshold is not None:
        preserve_crown_threshold = float(preserve_crown_threshold)
        if not 0.0 < preserve_crown_threshold <= 1.0:
            raise ValueError(
                "dsm_mask.preserve_crown_threshold must be in (0, 1] or null; "
                f"got {preserve_crown_threshold}"
            )
    if method == "dtm":
        # Two runs that differ only in the co-registration must not share names
        assert real is not None
        mask_suffix += real.tag()
    if height_ramp is not None:
        mask_suffix += f"_hr{height_ramp}"
    if crown_resolve_threshold is not None:
        mask_suffix += f"_cr{crown_resolve_threshold}"
    if preserve_crown_threshold is not None:
        mask_suffix += f"_pc{preserve_crown_threshold}"

    args.out = _embed_params(args.out, mask_suffix)
    mask_dir, mask_file = os.path.dirname(args.out), os.path.basename(args.out)
    args.out = os.path.join(mask_dir, run_id, mask_file)

    ndsm_tags = {"dtm": "_dtm", "dtm_approx": "_da"}
    ndsm_base = os.path.splitext(os.path.basename(args.out_dsm or "dsm_ndsm.tif"))[0]
    if args.out_dsm:
        channel_tag = ndsm_tags[channel] + (real.tag() if channel == "dtm" and real else "")
        args.out_dsm = _embed_params(args.out_dsm, channel_tag)
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

    # --- Ground detection ------------------------------------------------------
    dtm_stages: dict[str, np.ndarray] = {}

    mask_stem = os.path.splitext(os.path.basename(args.out))[0]

    # Every nDSM this run needs: the mask's, and the channel's if it differs.
    # name -> (ground_conf, ndsm)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if real is not None:
        print("Resampling external DTM to mask grid...")
        dtm = resample_raster(real.path, h, w, transform, crs)
        print(f"\n[dtm] external DTM  height_threshold={args.height_threshold} m  ramp={height_ramp}")
        _, conf, nd, dtm_stages = detect_ground_dtm(
            dsm,
            dtm,
            args.height_threshold,
            height_ramp,
            stage=real.stage,
            clamp_to_dsm=real.clamp_to_dsm,
            local_blocks=real.local_blocks,
        )
        results["dtm"] = (conf, nd)
    if "dtm_approx" in used:
        assert approx is not None
        print(f"\n[dtm_approx] {approx}  height_threshold={args.height_threshold} m  ramp={height_ramp}")
        _, conf, nd, approx_dtm = detect_ground_dtm_approx(
            dsm, abs(transform.a), approx, args.height_threshold, height_ramp
        )
        results["dtm_approx"] = (conf, nd)
        _save_float_tif(
            approx_dtm,
            os.path.join(process_out_dir, "dtm_approx", run_id, f"{mask_stem}_dtm_approx.tif"),
            profile,
            label="approximated DTM",
            nodata=float("nan"),
        )

    ground_conf, ndsm = results[method]
    channel_ndsm = results[channel][1]

    for name, (_, nd) in results.items():
        valid_ndsm = nd[~np.isnan(nd)]
        if not valid_ndsm.size:
            continue
        print(f"  nDSM [{name}]  min={np.min(valid_ndsm):.2f} m  "
              f"p5={np.percentile(valid_ndsm, 5):.2f} m  "
              f"p25={np.percentile(valid_ndsm, 25):.2f} m  "
              f"median={np.median(valid_ndsm):.2f} m  "
              f"p75={np.percentile(valid_ndsm, 75):.2f} m  "
              f"p95={np.percentile(valid_ndsm, 95):.2f} m  "
              f"max={np.max(valid_ndsm):.2f} m")

    # --- Save both co-registered DTM stages ----------------------------------
    # Both are written whatever `dtm_real.stage` selects: comparing them is how the
    # co-registration is judged (see dsm_overview), and the one not used here
    # is exactly the comparison partner.
    if dtm_stages:
        coreg_dir = os.path.join(process_out_dir, "dtm_coreg", run_id)
        for name, stage_dtm in dtm_stages.items():
            _save_float_tif(
                stage_dtm,
                os.path.join(coreg_dir, f"{mask_stem}_dtm_{name}.tif"),
                profile,
                label=f"co-registered DTM ({name}{', used' if real and name == real.stage else ''})",
                nodata=float("nan"),
            )

    # --- Apply soft ground blend to crown mask --------------------------------
    preserved = preserved_crown(mask, preserve_crown_threshold)
    n_crown_dampened = int(np.sum(
        (mask >= 0.0) & (mask < MASK_UNLABELLED) & (ground_conf > 0.0) & ~preserved
    ))
    n_crown_preserved = int(np.sum(preserved & (ground_conf > 0.0)))
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
        preserve_crown_threshold=preserve_crown_threshold,
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
    if preserve_crown_threshold is not None:
        print(f"Crown pixels kept from stage 1a:  {n_crown_preserved:,}  "
              f"(mask >= {preserve_crown_threshold:.2f}, DSM ignored)")
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

    # --- Write ground confidence raster --------------------------------------
    _save_conf_tif(ground_conf, conf_path, profile)

    # --- Write normalised nDSM (the model's input channel) --------------------
    if args.out_dsm:
        ndsm_norm = normalize_ndsm(channel_ndsm, dsm, args.max_ndsm_height)
        dsm_profile = profile.copy()
        dsm_profile.update(dtype="float32", count=1, nodata=None)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_dsm)), exist_ok=True)
        with rasterio.open(args.out_dsm, "w", **dsm_profile) as dst:
            dst.write(ndsm_norm[np.newaxis].astype(np.float32))
        print(f"Saved nDSM ({channel}): {args.out_dsm}  (range [0,1])")

    # --- Write raw nDSM in metres, one per computed nDSM ----------------------
    ndsm_m_dir = os.path.join(process_out_dir, "ndsm_in_m", run_id)
    os.makedirs(ndsm_m_dir, exist_ok=True)
    dsm_profile_m = profile.copy()
    dsm_profile_m.update(dtype="float32", count=1, nodata=float("nan"))
    for name, (_, nd) in results.items():
        ndsm_m_path = os.path.join(ndsm_m_dir, f"{ndsm_base}{ndsm_tags[name]}_raw_m.tif")
        ndsm_out = nd.astype(np.float32)
        ndsm_out[np.isnan(dsm)] = float("nan")
        with rasterio.open(ndsm_m_path, "w", **dsm_profile_m) as dst:
            dst.write(ndsm_out[np.newaxis])
        print(f"Saved raw nDSM {name} (m): {ndsm_m_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Stage 1b: apply DSM ground mask.")
    p.add_argument("--config", required=True, help="Path to preprocess.yaml")
    cfg = OmegaConf.load(p.parse_args().config)
    main(cfg.dsm_mask)
