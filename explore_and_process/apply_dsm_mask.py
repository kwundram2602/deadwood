"""
apply_dsm_mask.py

Refine the soft crown mask by labelling confirmed ground pixels using the DSM.

Ground detection uses a local-minimum approximation of the terrain surface:
  nDSM_approx = DSM - minimum_filter(DSM, window)
  pixels where nDSM_approx < height_threshold  =>  ground = 0.0

Ground pixels soft-blend into the crown mask: crown pixels (0–1) are
multiplied by (1 − ground_conf); noData pixels are resolved to ground
only when ground_conf exceeds --nodata_resolve_threshold.

Remaining noData pixels (255) — no crown polygon, not confirmed ground —
are excluded from the loss during training.

Usage:
  python explore_and_process/apply_dsm_mask.py \\
      --mask  explore_and_process/out/crown_mask.tif \\
      --dsm   data/raster/20230824_Airport_Main_MAVICM3MFIXEDM3M_DSM_coregReference.tif \\
      --out   explore_and_process/out/crown_mask_final.tif \\
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

logger = logging.getLogger(__name__)


def resample_dsm(dsm_path, h, w, transform, crs):
    """Reproject DSM to exactly match the mask grid."""
    out = np.full((h, w), np.nan, dtype=np.float32)
    with rasterio.open(dsm_path) as src:
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


def _smoothstep_confidence(ndsm: np.ndarray, threshold: float) -> np.ndarray:
    """Ground confidence based on height threshold.

    Returns 1.0 for nDSM <= threshold, smoothly falls to 0.0 at 2*threshold.
    Uses the smoothstep curve (3t^2 - 2t^3) for a C1-continuous transition.
    NaN handling: caller is responsible for zeroing NaN pixels after this call.
    """
    t = np.clip((ndsm - threshold) / threshold, 0.0, 1.0)
    return (1.0 - (3 * t**2 - 2 * t**3)).astype(np.float32)


def detect_ground_local_min(
    dsm: np.ndarray, windows: list[int], height_threshold: float
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

    confidence = _smoothstep_confidence(ndsm, height_threshold)
    confidence[np.isnan(dsm)] = 0.0

    binary = (ndsm < height_threshold) & ~np.isnan(dsm)
    return binary, confidence, ndsm


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
) -> np.ndarray:
    """Soft-blend ground confidence into the crown mask.

    Crown pixels (0–1): multiplied by (1 - ground_conf).
    noData pixels (255): resolved to (1 - ground_conf) only when
      ground_conf > nodata_resolve_threshold, otherwise kept at 255.
    """
    result = mask.copy()

    # Alle gültigen Kronenpixel (Konfidenz 0–1, kein noData-Sentinel)
    crown = (mask >= 0.0) & (mask < 255.0)
    # Krone × (1 – Bodenwahrscheinlichkeit): hohe Bodenkonf. → Kronenwert sinkt gegen 0
    result[crown] = mask[crown] * (1.0 - ground_conf[crown])

    # noData-Pixel (Sentinel 255): außerhalb des Bildbereichs oder nicht klassifiziert
    nodata = mask == 255.0
    # Wenn der DSM-Detektor trotzdem sicher Boden erkennt, Pixel auflösen statt 255 zu behalten
    resolve = nodata & (ground_conf > nodata_resolve_threshold)
    # Aufgelöste noData-Pixel bekommen Bodenwahrscheinlichkeit als invertierte Kronenkonfidenz
    result[resolve] = 1.0 - ground_conf[resolve]
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


def _save_conf_tif(arr: np.ndarray, path: str, profile: dict) -> None:
    """Write a float32 confidence raster [0,1] to disk."""
    conf_profile = profile.copy()
    conf_profile.update(dtype="float32", count=1, nodata=None)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **conf_profile) as dst:
        dst.write(arr[np.newaxis])
    print(f"  Saved confidence: {path}")


def main(args):
    logger.info("Config:\n%s", OmegaConf.to_yaml(args))

    # --- Run ID (timestamp) groups all outputs of this run -------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_out_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.out)))

    # --- Build output filename suffix ----------------------------------------
    w_tag = "-".join(str(w) for w in args.windows)
    lm_tag = f"_lm_w{w_tag}_ht{args.height_threshold}"
    gr_tag = f"_gr_s{args.gradient_sigma}"

    if args.method == "local_min":
        mask_suffix = lm_tag
    elif args.method == "gradient":
        mask_suffix = gr_tag
    else:  # both
        mask_suffix = f"{lm_tag}{gr_tag}_{args.combine}"

    args.out = _embed_params(args.out, mask_suffix)
    mask_dir, mask_file = os.path.dirname(args.out), os.path.basename(args.out)
    args.out = os.path.join(mask_dir, run_id, mask_file)

    if args.out_dsm:
        args.out_dsm = _embed_params(args.out_dsm, f"_w{w_tag}")
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
    dsm = resample_dsm(args.dsm, h, w, transform, crs)

    # --- Run detection method(s) ---------------------------------------------
    lm_bin = lm_conf = ndsm = None
    gr_bin = gr_conf = grad_smooth = None

    mask_stem = os.path.splitext(os.path.basename(args.out))[0]

    if args.method in ("local_min", "both"):
        print(f"\n[local_min] windows={args.windows} px  height_threshold={args.height_threshold} m")
        lm_bin, lm_conf, ndsm = detect_ground_local_min(dsm, args.windows, args.height_threshold)
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
        _save_diagnostic(
            ndsm, process_out_dir, run_id, mask_stem, label="lm",
            used_threshold=args.height_threshold,
            suggested_threshold=suggested_ht,
            xlabel="nDSM [m]",
            title="local_min — nDSM distribution",
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

    # --- Save individual confidences -----------------------------------------
    if lm_conf is not None:
        _save_conf_tif(lm_conf, os.path.join(conf_dir, f"{mask_stem}_lm_conf.tif"), profile)
    if gr_conf is not None:
        _save_conf_tif(gr_conf, os.path.join(conf_dir, f"{mask_stem}_gr_conf.tif"), profile)

    # --- Combine -------------------------------------------------------------
    if args.method == "local_min":
        ground_bin, ground_conf = lm_bin, lm_conf
    elif args.method == "gradient":
        ground_bin, ground_conf = gr_bin, gr_conf
    else:
        print(f"\n[combine] mode={args.combine}")
        ground_bin, ground_conf = combine(lm_bin, lm_conf, gr_bin, gr_conf, mode=args.combine)

    # --- Apply soft ground blend to crown mask --------------------------------
    n_crown_dampened = int(np.sum((mask >= 0.0) & (mask < 255.0) & (ground_conf > 0.0)))
    n_nodata_before = int(np.sum(mask == 255.0))
    mask = apply_soft_blend(mask, ground_conf, args.nodata_resolve_threshold)
    n_nodata_resolved = n_nodata_before - int(np.sum(mask == 255.0))
    n_crown  = int(np.sum((mask > 0) & (mask < 255)))
    n_ground = int(np.sum(mask == 0.0))
    n_nodata = int(np.sum(mask == 255.0))
    print(f"\nCrown pixels dampened by DSM:     {n_crown_dampened:,}  (multiplicative blend)")
    print(f"noData pixels resolved to ground: {n_nodata_resolved:,}  (ground_conf > {args.nodata_resolve_threshold:.2f})")
    print(f"Final  =>  Crown: {n_crown:,}  Ground: {n_ground:,}  noData: {n_nodata:,}")

    # --- Write binary mask ---------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=255.0)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(mask[np.newaxis])
    print(f"Saved mask: {args.out}")

    # --- Write combined confidence raster ------------------------------------
    _save_conf_tif(ground_conf, conf_path, profile)

    # --- Write normalised nDSM (local_min method only) -----------------------
    if args.out_dsm and ndsm is not None:
        valid_pos = ndsm[~np.isnan(ndsm) & (ndsm > 0)]
        p95 = float(np.percentile(valid_pos, 95)) if valid_pos.size > 0 else args.max_ndsm_height
        ceiling = min(p95, args.max_ndsm_height)
        print(f"nDSM ceiling: {ceiling:.2f} m  (p95={p95:.2f} m, cap={args.max_ndsm_height} m)")
        ndsm_norm = np.clip(ndsm, 0.0, ceiling) / ceiling
        ndsm_norm[np.isnan(dsm)] = 0.0
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
