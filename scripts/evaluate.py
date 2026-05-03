"""Standalone evaluation script.

Usage:
    uv run --active python deadwood/scripts/evaluate.py \
        --config deadwood/configs/crown_ms.yaml \
        --weights deadwood/out/crown_ms/ft_best.pt \
        --working_dir D:/EAGLE/InnoLab_DL
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import make_loaders
from models.model import build_model
from training.losses import MaskedBCELoss
from training.metrics import pixel_metrics
from utils.device import get_device
from utils.viz import plot_training_curves


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    criterion = MaskedBCELoss()
    model.eval()

    total_loss = 0.0
    agg = {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
    n = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            m = pixel_metrics(logits, masks)

            bs = images.size(0)
            total_loss += loss.item() * bs
            for k in agg:
                agg[k] += m[k] * bs
            n += bs

    results = {k: v / n for k, v in agg.items()}
    results["loss"] = total_loss / n
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained crown segmentation model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--weights", required=True, help="Path to .pt state dict")
    parser.add_argument("--working_dir", default=".", help="Root for relative paths")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(args.working_dir).resolve()
    data_root = root / cfg.dataset.path

    device = get_device()
    _, _, test_loader = make_loaders(cfg, data_root)

    model = build_model(cfg, device)
    weights_path = Path(args.weights)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    # Unfreeze all params for evaluation
    for p in model.parameters():
        p.requires_grad = False

    results = evaluate(model, test_loader, device)

    print("\n=== Test Results ===")
    for k, v in results.items():
        print(f"  {k:>8}: {v:.4f}")

    # Save results next to the weights file
    out_path = weights_path.parent / f"{weights_path.stem}_eval.txt"
    with open(out_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
