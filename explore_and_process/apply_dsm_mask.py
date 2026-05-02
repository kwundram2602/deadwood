"""
apply_dsm_mask.py

Refine the soft crown mask by labelling confirmed ground pixels using the DSM.

Ground detection uses a local-minimum approximation of the terrain surface:
  nDSM_approx = DSM - minimum_filter(DSM, window)
  pixels where nDSM_approx < height_threshold  →  ground = 0.0

Ground pixels overwrite both noData (255) and any existing crown label,
since a pixel at ground height cannot be a tree crown.

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
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter, minimum_filter, sobel, uniform_filter1d


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
    mu1 = np.where(w1 > 1e-8, (mu_total - w0 * mu0) / w1, 0.0)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(sigma_b))])


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
    local_min = np.minimum.reduce(local_mins) if len(local_mins) > 1 else local_mins[0]
    ndsm = dsm - local_min

    valid_pos = ndsm[~np.isnan(ndsm) & (ndsm > 0)]
    p95 = float(np.percentile(valid_pos, 95)) if valid_pos.size > 0 else height_threshold * 5
    confidence = (1.0 - np.clip(ndsm / p95, 0.0, 1.0)).astype(np.float32)
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


def _save_diagnostic(
    values: np.ndarray,
    out_mask_path: str,
    label: str,
    used_threshold: float,
    suggested_threshold: float | None = None,
    xlabel: str = "value",
    title: str = "distribution",
) -> None:
    """Save a histogram diagnostic PNG to diag_graphs/ sibling of the masks/ dir."""
    process_out_dir = os.path.dirname(os.path.dirname(os.path.abspath(out_mask_path)))
    diag_dir = os.path.join(process_out_dir, "diag_graphs")
    os.makedirs(diag_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(out_mask_path))[0]
    diag_path = os.path.join(diag_dir, f"{stem}_{label}_diag.png")

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


def main(args):
    w_tag = "-".join(str(w) for w in args.windows)
    args.out = _embed_params(args.out, f"_w{w_tag}_ht{args.height_threshold}")
    if args.out_dsm:
        args.out_dsm = _embed_params(args.out_dsm, f"_w{w_tag}")

    with rasterio.open(args.mask) as src:
        mask = src.read(1).astype(np.float32)
        h, w = src.height, src.width
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    print(f"Mask grid: {h} x {w}")

    print("Resampling DSM to mask grid...")
    dsm = resample_dsm(args.dsm, h, w, transform, crs)

    # Replace NaN with local max so the filter doesn't suppress nearby valid values
    dsm_filled = np.where(np.isnan(dsm), np.nanmax(dsm), dsm)

    print(f"Computing local minimum (windows={args.windows} px)...")
    local_mins = [minimum_filter(dsm_filled, size=w) for w in args.windows]
    local_min = np.minimum.reduce(local_mins) if len(local_mins) > 1 else local_mins[0]
    ndsm = dsm - local_min  # approximate height above local terrain
    # if the threshold is higher,
    # more pixels are labelled ground, which may include low vegetation or crown edges;
    suggested_ht = _find_valley_threshold(ndsm)
    print(f"\nnDSM stats (valid pixels):")
    valid_ndsm = ndsm[~np.isnan(ndsm)]
    print(f"  Median: {np.median(valid_ndsm):.2f} m  |  5th pct: {np.percentile(valid_ndsm, 5):.2f} m  |  95th pct: {np.percentile(valid_ndsm, 95):.2f} m")
    print(f"  Suggested threshold (histogram valley): {suggested_ht:.2f} m  (used: {args.height_threshold} m)")
    _save_diagnostic(ndsm, args.out, args.height_threshold, suggested_ht)

    ground = (ndsm < args.height_threshold) & ~np.isnan(dsm)
    mask[ground] = 0.0

    n_crown  = int(np.sum((mask > 0) & (mask < 255)))
    n_ground = int(np.sum(mask == 0.0))
    n_nodata = int(np.sum(mask == 255.0))
    print(f"Ground pixels forced to 0: {int(ground.sum()):,}")
    print(f"Final  →  Crown: {n_crown:,}  Ground: {n_ground:,}  noData: {n_nodata:,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=255.0)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(mask[np.newaxis])
    print(f"Saved: {args.out}")

    if args.out_dsm:
        valid_pos = ndsm[~np.isnan(ndsm) & (ndsm > 0)]
        p95 = float(np.percentile(valid_pos, 95)) if valid_pos.size > 0 else args.max_ndsm_height
        ceiling = min(p95, args.max_ndsm_height)
        print(f"nDSM normalisation ceiling: {ceiling:.2f} m  (95th pct={p95:.2f} m, cap={args.max_ndsm_height} m)")
        ndsm_norm = np.clip(ndsm, 0.0, ceiling) / ceiling
        ndsm_norm[np.isnan(dsm)] = 0.0  # fill DSM voids with 0
        dsm_profile = profile.copy()
        dsm_profile.update(dtype="float32", count=1, nodata=None)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_dsm)), exist_ok=True)
        with rasterio.open(args.out_dsm, "w", **dsm_profile) as dst:
            dst.write(ndsm_norm[np.newaxis].astype(np.float32))
        print(f"nDSM saved: {args.out_dsm}  (range [0,1])")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", required=True,
                   help="Soft crown mask from rasterize_crowns.py")
    p.add_argument("--dsm", required=True,
                   help="DSM raster (.tif)")
    p.add_argument("--out", required=True,
                   help="Base output path for binary mask (.tif) — params embedded automatically")
    p.add_argument("--method", default="local_min", choices=["local_min", "gradient", "both"],
                   help="Ground detection method (default: local_min)")
    p.add_argument("--combine", default="or", choices=["or", "and"],
                   help="How to merge methods when --method both (default: or)")
    p.add_argument("--windows", type=int, nargs="+", default=[700],
                   help="Window size(s) in px for local-min filter — element-wise min used "
                        "when multiple given (default: [700] → 35 m at 5 cm GSD)")
    p.add_argument("--height_threshold", type=float, default=2.0,
                   help="nDSM threshold (m) below which pixel is ground (default: 2.0)")
    p.add_argument("--gradient_sigma", type=float, default=3.0,
                   help="Gaussian smoothing sigma (px) before gradient computation (default: 3)")
    p.add_argument("--gradient_threshold", type=float, default=None,
                   help="Gradient magnitude threshold for ground; auto-Otsu if omitted")
    p.add_argument("--out_dsm", default=None,
                   help="Save normalised nDSM [0,1] here (local_min method only)")
    p.add_argument("--max_ndsm_height", type=float, default=50.0,
                   help="nDSM cap (m) for normalisation — overrides p95 ceiling if lower (default: 50)")
    main(p.parse_args())
