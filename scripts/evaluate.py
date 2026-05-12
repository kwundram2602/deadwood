"""Standalone evaluation script.

Usage:
    uv run --active python deadwood/scripts/evaluate.py \\
        --config deadwood/configs/train_config/crown_ms.yaml \\
        --weights deadwood/out/crown_ms/ft_best.pt \\
        --working_dir D:/EAGLE/InnoLab_DL

Optional flags:
    --threshold  float  (default from cfg.metrics.threshold)
    --n_samples  int    (default 6)
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import make_loaders
from models.model import build_model
from training.losses import CombinedLoss
from training.metrics import MetricAccumulator
from utils.device import get_device
from utils.viz import plot_final_bars, plot_samples


def _evaluate_split(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    criterion: torch.nn.Module,
    threshold: float,
) -> dict[str, float]:
    accumulator = MetricAccumulator()
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            accumulator.update(logits.detach(), masks, loss.item(), images.size(0))
    return accumulator.compute(threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained crown segmentation model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--working_dir", default=".")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n_samples", type=int, default=6)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(args.working_dir).resolve()
    data_root = root / cfg.dataset.path
    threshold = args.threshold if args.threshold is not None else float(cfg.metrics.threshold)

    device = get_device()
    train_loader, val_loader, test_loader = make_loaders(cfg, data_root)

    model = build_model(cfg, device)
    weights_path = Path(args.weights)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad = False

    criterion = CombinedLoss(cfg.loss)
    out_dir = weights_path.parent

    train_m = _evaluate_split(model, train_loader, device, criterion, threshold)
    val_m = _evaluate_split(model, val_loader, device, criterion, threshold)
    test_m = _evaluate_split(model, test_loader, device, criterion, threshold)

    print("\n=== Evaluation Results ===")
    header = f"{'metric':>10}  {'train':>8}  {'val':>8}  {'test':>8}"
    print(header)
    print("-" * len(header))
    for k in ("loss", "auc_pr", "f1", "iou", "prec", "rec", "acc"):
        print(
            f"{k:>10}  {train_m.get(k, 0):>8.4f}  {val_m.get(k, 0):>8.4f}  {test_m.get(k, 0):>8.4f}"
        )

    bars_path = out_dir / f"{weights_path.stem}_eval_bars.png"
    plot_final_bars(train_m, val_m, test_m, bars_path)

    val_samples_path = out_dir / f"{weights_path.stem}_val_samples.png"
    plot_samples(
        model, val_loader, device, n=args.n_samples, threshold=threshold, save_path=val_samples_path
    )

    test_samples_path = out_dir / f"{weights_path.stem}_test_samples.png"
    plot_samples(
        model,
        test_loader,
        device,
        n=args.n_samples,
        threshold=threshold,
        save_path=test_samples_path,
    )

    txt_path = out_dir / f"{weights_path.stem}_eval.txt"
    with open(txt_path, "w") as f:
        f.write(f"threshold: {threshold}\n\n")
        for split, m in (("train", train_m), ("val", val_m), ("test", test_m)):
            f.write(f"[{split}]\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")
    print(f"\nSaved results to {txt_path}")


if __name__ == "__main__":
    main()
