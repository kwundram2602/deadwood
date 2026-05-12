"""Crown segmentation training script.

Usage (local):
    uv run --active python deadwood/scripts/train.py \\
        --config deadwood/configs/train_config/crown_ms.yaml \\
        --working_dir D:/EAGLE/InnoLab_DL

Usage (HPC via sbatch):
    see deadwood/hpc/train_torch.sh
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import make_loaders
from models.model import build_model
from training.learning_configurator import LearningConfigurator
from training.losses import CombinedLoss
from training.trainer import train
from utils.device import get_device
from utils.logging import init_wandb


def _reload_best(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Load best-checkpoint weights back into model in-place (strips DataParallel prefix)."""
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(state, strict=True)


def _make_experiment_id(cfg) -> str:
    parts = [cfg.model_name]

    weights = cfg.model.get("weights_name") or cfg.model.get("weights_path")
    if weights:
        w = str(weights)
        parts.append(Path(w).stem if ("/" in w or "\\" in w) else w)

    loss = cfg.loss
    loss_parts = [
        f"{term}{float(getattr(loss, term, 0.0)):g}"
        for term in ("bce", "dice", "iou", "mae")
        if float(getattr(loss, term, 0.0)) > 0
    ]
    if loss_parts:
        parts.append("_".join(loss_parts))

    tag = cfg.get("run_tag")
    if tag:
        parts.append(str(tag))

    return "__".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crown segmentation training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--working_dir", default=".", help="Root directory for dataset.path in config"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(args.working_dir).resolve()

    data_root = root / cfg.dataset.path

    experiment_id = _make_experiment_id(cfg)
    out_dir = Path(cfg.output_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {experiment_id}")
    print(f"Output dir: {out_dir}")

    if cfg.model.weights_path is not None:
        cfg.model.weights_path = str((root / cfg.model.weights_path).resolve())

    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    train_loader, val_loader, _ = make_loaders(cfg, data_root)

    model = build_model(cfg, device)
    lc = LearningConfigurator()
    criterion = CombinedLoss(cfg.loss)
    threshold = float(cfg.metrics.threshold)
    target_threshold = float(cfg.metrics.get("target_threshold", 0.5))

    if cfg.logging.use_wandb:
        init_wandb(cfg, model)

    # ------------------------------------------------------------------
    # Phase 1: Transfer Learning — encoder frozen, decoder + head only
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1: Transfer Learning")
    print("=" * 60)
    lc.prepare_model_for_transfer_learning(model)
    tl_result = train(
        model,
        train_loader,
        val_loader,
        cfg.transfer,
        out_dir,
        "tl",
        device,
        criterion=criterion,
        threshold=threshold,
        target_threshold=target_threshold,
    )
    _reload_best(model, tl_result["ckpt_path"], device)

    # ------------------------------------------------------------------
    # Phase 2: Fine-tuning — partial encoder unfreeze
    # ------------------------------------------------------------------
    if cfg.fine_tune.enabled:
        print("\n" + "=" * 60)
        print("Phase 2: Fine-tuning")
        print("=" * 60)
        lc.prepare_model_for_fine_tuning(model, cfg.fine_tune.unfreeze_blocks)
        ft_result = train(
            model,
            train_loader,
            val_loader,
            cfg.fine_tune,
            out_dir,
            "ft",
            device,
            criterion=criterion,
            threshold=threshold,
            target_threshold=target_threshold,
        )
        _reload_best(model, ft_result["ckpt_path"], device)

    print(f"\nTraining complete.")
    print(f"Experiment : {experiment_id}")
    print(f"Output dir : {out_dir}")


if __name__ == "__main__":
    main()
