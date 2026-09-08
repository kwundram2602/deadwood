"""Fine-tune the pretrained deadtrees deadwood model on our crowns.

Mirrors scripts/train.py but for the deadtrees architecture and its uint8/
ImageNet input contract. Everything downstream of the model and the loaders —
freezing, loss, training loop, metrics, checkpointing — is the existing stack,
unchanged.

Usage:
    uv run python scripts/finetune_deadwood.py \\
        --config configs/finetune/deadwood.yaml --working_dir .
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# isort: split  — must be set before torch loads libc10_cuda
import argparse
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.deadwood_dataset import make_deadwood_loaders  # noqa: E402
from scripts.raw_predict_deadwood import load_model  # noqa: E402
from training.learning_configurator import LearningConfigurator  # noqa: E402
from training.losses import CombinedLoss  # noqa: E402
from training.trainer import train  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.logger import init_wandb  # noqa: E402


def _reload_best(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(state, strict=True)


def _experiment_id(cfg) -> str:
    parts = [str(cfg.model_name), str(cfg.stage)]
    loss_parts = [
        f"{term}{float(cfg.loss.get(term, 0.0)):g}"
        for term in ("bce", "dice", "iou", "mae")
        if float(cfg.loss.get(term, 0.0)) > 0
    ]
    if loss_parts:
        parts.append("_".join(loss_parts))
    if cfg.get("run_tag"):
        parts.append(str(cfg.run_tag))
    return "__".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deadwood model fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--working_dir", default=".")
    parser.add_argument("--stage", choices=["head", "decoder"], help="override stage")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{args.config} must hold a mapping, not a list")
    if args.stage:
        cfg.stage = args.stage

    root = Path(args.working_dir).resolve()
    data_root = root / str(cfg.dataset.path)
    if not (data_root / "meta.json").exists():
        raise FileNotFoundError(
            f"{data_root} has no meta.json — run scripts/preprocess_deadwood.py first"
        )

    out_dir = root / str(cfg.output_dir) / _experiment_id(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(f"Experiment: {out_dir.name}\nOutput dir: {out_dir}")

    device = get_device()
    train_loader, val_loader, _ = make_deadwood_loaders(cfg, data_root)

    print(f"Loading weights: {cfg.weights}")
    model = load_model(root / str(cfg.weights), device)
    model.train()

    lc = LearningConfigurator()
    criterion = CombinedLoss(cfg.loss)
    threshold = float(cfg.metrics.thresholds[0])
    target_threshold = float(cfg.metrics.target_thresholds[0])
    amp = bool(OmegaConf.select(cfg, "train.amp", default=True))

    if cfg.logging.use_wandb:
        init_wandb(cfg, model)

    print("\n" + "=" * 60)
    print(f"Phase 1: {cfg.stage}")
    print("=" * 60)
    if str(cfg.stage) == "head":
        lc.prepare_model_for_head_only(model)
    else:
        lc.prepare_model_for_transfer_learning(model)

    tl = train(
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
        amp=amp,
    )
    _reload_best(model, tl["ckpt_path"], device)

    if cfg.fine_tune.enabled:
        print("\n" + "=" * 60)
        print("Phase 2: encoder unfreeze")
        print("=" * 60)
        lc.prepare_model_for_fine_tuning(model, list(cfg.fine_tune.unfreeze_keys))
        ft = train(
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
            amp=amp,
        )
        _reload_best(model, ft["ckpt_path"], device)

    print(f"\nTraining complete.\nOutput dir : {out_dir}")


if __name__ == "__main__":
    main()
