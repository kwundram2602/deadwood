"""Score a deadwood checkpoint against the crown polygons.

Runs the same measurement for the raw published model and for a fine-tuned
checkpoint, so "beats the baseline" is one script's output rather than two
incomparable ones. Recall only — see training/crown_recall for why.

Usage:
    uv run python scripts/evaluate_deadwood.py \\
        --config configs/predict/raw_deadwood.yaml --working_dir .

    # score a fine-tuned checkpoint, reusing a cached probability raster
    uv run python scripts/evaluate_deadwood.py \\
        --config configs/predict/raw_deadwood.yaml --working_dir . \\
        --weights experiments/deadwood_ft/tl_best.pt --probs out/eval/probs.tif
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.raw_predict_deadwood import load_model, predict_probs, prepare_rgb8  # noqa: E402
from training.crown_recall import crown_coverage, crown_pixel_index, threshold_sweep  # noqa: E402
from utils.device import get_device  # noqa: E402

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _probability_raster(cfg, out_dir: Path, weights: Path, cache: Path | None):
    """Return (probs, transform, crs).

    Reuses `cache` when it exists — inference is the expensive part of this
    script and a threshold sweep does not need it repeated.
    """
    if cache is not None and cache.exists():
        print(f"Reusing cached probabilities: {cache}")
        with rasterio.open(cache) as src:
            return src.read(1), src.transform, src.crs

    stem = Path(str(cfg.source.path)).stem
    rgb8_path = prepare_rgb8(cfg, out_dir / f"{stem}_rgb8.tif")
    device = get_device()
    model = load_model(weights, device)
    with rasterio.open(rgb8_path) as src:
        probs, valid = predict_probs(src, model, device, cfg)
        transform, crs, profile = src.transform, src.crs, src.profile
    probs[~valid] = 0.0

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=1, dtype="float32", nodata=None)
        with rasterio.open(cache, "w", **profile) as dst:
            dst.write(probs, 1)
        print(f"  wrote {cache}")
    return probs, transform, crs


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-crown recall for a deadwood checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--working_dir", default=".")
    parser.add_argument("--weights", help="override the checkpoint (.safetensors or .pt)")
    parser.add_argument("--crowns", help="override the label layer")
    parser.add_argument("--exclude_fids", default="21", help="comma-separated fids to skip")
    parser.add_argument("--probs", help="cache path for the probability raster")
    parser.add_argument("--out_dir", default="out/eval")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{args.config} must hold a mapping, not a list")
    cfg = cfg.get("raw_deadwood", cfg)

    root = Path(args.working_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = (root / (args.weights or str(cfg.weights))).resolve()
    # Its own key, not the clip extent it used to borrow: an extent to predict
    # over and a label set to score against are unrelated things that merely
    # happened to be the same file.
    crowns_path = root / (args.crowns or str(cfg.labels.deadwood_path))
    cache = (root / args.probs).resolve() if args.probs else None

    probs, transform, crs = _probability_raster(cfg, out_dir, weights, cache)

    crowns = gpd.read_file(crowns_path, fid_as_index=True)
    excluded = [int(f) for f in args.exclude_fids.split(",") if f.strip()]
    crowns = crowns.drop(index=[f for f in excluded if f in crowns.index])
    # A CRS mismatch here would place every crown somewhere else on the grid and
    # score as a silent total miss, so reproject rather than assume.
    crowns = crowns.to_crs(crs)
    print(f"{len(crowns)} crowns scored (excluded {excluded})")

    index = crown_pixel_index(crowns, transform, probs.shape)
    threshold = float(cfg.get("threshold", 0.5))
    coverage = crown_coverage(probs > threshold, index)
    sweep = threshold_sweep(probs, index, DEFAULT_THRESHOLDS)

    print(f"\nPer-crown coverage at threshold {threshold}")
    print(coverage.to_string())
    print(
        f"\n  hit >10%: {int(coverage['hit'].sum())}/{len(coverage)}   "
        f"mean covered fraction: {coverage['covered_fraction'].mean():.2f}"
    )
    print("\nThreshold sweep")
    print(sweep.to_string(index=False))

    coverage.to_csv(out_dir / f"{weights.stem}_crown_coverage.csv")
    sweep.to_csv(out_dir / f"{weights.stem}_threshold_sweep.csv", index=False)
    print(f"\nWrote {out_dir}/{weights.stem}_*.csv")


if __name__ == "__main__":
    main()
