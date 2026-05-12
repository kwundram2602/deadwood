"""Standalone evaluation script.

Usage:
    uv run python scripts/evaluate.py \\
        --config configs/train_config/crown_ms.yaml \\
        --working_dir .

CLI flags override config values when provided:
    --weights    path to .pt checkpoint  (falls back to cfg.evaluate.weights)
    --threshold  float                   (falls back to cfg.metrics.threshold)
    --n_samples  int                     (falls back to cfg.evaluate.n_samples)
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import make_loaders
from models.model import build_model
from scripts.train import _make_experiment_id
from training.losses import CombinedLoss
from training.metrics import MetricAccumulator
from utils.device import get_device
from utils.viz import plot_final_bars_multi, plot_samples

_NODATA: int = 255


def _collect_split(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> MetricAccumulator:
    """Run one full pass through loader and return a filled accumulator (no thresholds applied yet)."""
    accumulator = MetricAccumulator()
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            n_valid = int((masks != _NODATA).sum().item())
            accumulator.update(logits.detach(), masks, loss.item(), n_valid)
    return accumulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained crown segmentation model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--working_dir", default=".")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(args.working_dir).resolve()
    data_root = root / cfg.dataset.path

    metrics_cfg = cfg.get("metrics", OmegaConf.create({}))
    # Support both list form (thresholds) and legacy scalar form (threshold)
    if "thresholds" in metrics_cfg:
        thresholds = list(float(v) for v in metrics_cfg.thresholds)
    else:
        thresholds = [args.threshold if args.threshold is not None else float(metrics_cfg.get("threshold", 0.5))]
    if "target_thresholds" in metrics_cfg:
        target_thresholds = list(float(v) for v in metrics_cfg.target_thresholds)
    else:
        target_thresholds = [float(metrics_cfg.get("target_threshold", 0.5))]

    eval_cfg = cfg.get("evaluate", OmegaConf.create({}))
    n_samples = args.n_samples if args.n_samples is not None else int(eval_cfg.get("n_samples", 6))

    weights_str = args.weights or eval_cfg.get("weights", None)
    if not weights_str:
        experiment_id = _make_experiment_id(cfg)
        ckpt_name = "ft_best.pt" if cfg.fine_tune.get("enabled", True) else "tl_best.pt"
        weights_str = str(Path(cfg.output_dir) / experiment_id / ckpt_name)
        print(f"Weights inferred: {weights_str}")

    device = get_device()
    train_loader, val_loader, test_loader = make_loaders(cfg, data_root)

    model = build_model(cfg, device)
    weights_path = Path(weights_str)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad = False

    criterion = CombinedLoss(cfg.loss)
    out_dir = weights_path.parent / "eval"
    out_dir.mkdir(exist_ok=True)

    # Run each split once, then compute metrics for every (tt, t) combo
    print("Collecting train split...")
    train_acc = _collect_split(model, train_loader, device, criterion)
    print("Collecting val split...")
    val_acc = _collect_split(model, val_loader, device, criterion)
    print("Collecting test split...")
    test_acc = _collect_split(model, test_loader, device, criterion)

    results: dict[tuple[float, float], dict[str, dict[str, float]]] = {}
    for tt in target_thresholds:
        for t in thresholds:
            results[(tt, t)] = {
                "train": train_acc.compute(threshold=t, target_threshold=tt),
                "val": val_acc.compute(threshold=t, target_threshold=tt),
                "test": test_acc.compute(threshold=t, target_threshold=tt),
            }

    # Print table for the first combination
    tt0, t0 = target_thresholds[0], thresholds[0]
    first = results[(tt0, t0)]
    print(f"\n=== Evaluation Results (tgt≥{tt0}, pred≥{t0}) ===")
    header = f"{'metric':>10}  {'train':>8}  {'val':>8}  {'test':>8}"
    print(header)
    print("-" * len(header))
    for k in ("loss", "auc_pr", "f1", "iou", "prec", "rec", "acc"):
        print(
            f"{k:>10}  {first['train'].get(k, 0):>8.4f}  "
            f"{first['val'].get(k, 0):>8.4f}  {first['test'].get(k, 0):>8.4f}"
        )

    plot_final_bars_multi(
        results, thresholds, target_thresholds, out_dir, stem=weights_path.stem
    )

    # Sample visualisations use the first threshold
    val_samples_path = out_dir / f"{weights_path.stem}_val_samples.png"
    plot_samples(
        model, val_loader, device, n=n_samples, threshold=t0, save_path=val_samples_path
    )

    test_samples_path = out_dir / f"{weights_path.stem}_test_samples.png"
    plot_samples(
        model,
        test_loader,
        device,
        n=n_samples,
        threshold=t0,
        save_path=test_samples_path,
    )

    txt_path = out_dir / f"{weights_path.stem}_eval.txt"
    with open(txt_path, "w") as f:
        for (tt, t), splits in results.items():
            f.write(f"[tgt>={tt}  pred>={t}]\n")
            for split, metrics in splits.items():
                f.write(f"  {split}:\n")
                for k, v in metrics.items():
                    f.write(f"    {k}: {v:.4f}\n")
            f.write("\n")
    print(f"\nSaved results to {txt_path}")


if __name__ == "__main__":
    main()
