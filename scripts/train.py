"""Crown segmentation training script.

Usage (local):
    uv run --active python deadwood/scripts/train.py \\
        --config deadwood/configs/train_config/crown_ms.yaml \\
        --working_dir D:/EAGLE/InnoLab_DL

Usage (HPC via sbatch):
    see deadwood/hpc/train_torch.sh
"""

import argparse
import copy
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import make_loaders
from models.model import build_model
from training.learning_configurator import LearningConfigurator
from training.losses import CombinedLoss
from training.trainer import train
from utils.device import get_device
from utils.logging import init_wandb


def main() -> None:
    parser = argparse.ArgumentParser(description="Crown segmentation training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--working_dir", default=".", help="Root directory for relative paths in config"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(args.working_dir).resolve()

    data_root = root / cfg.dataset.path
    out_dir = root / cfg.output_dir / cfg.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model.weights_path is not None:
        cfg.model.weights_path = str((root / cfg.model.weights_path).resolve())

    OmegaConf.save(cfg, out_dir / f"{cfg.model_name}.yaml")
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    train_loader, val_loader, _ = make_loaders(cfg, data_root)

    model = build_model(cfg, device)
    lc = LearningConfigurator()
    criterion = CombinedLoss(cfg.loss)
    threshold = float(cfg.metrics.threshold)

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
    )
    model = copy.deepcopy(tl_result["best_model"])

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
        )
        model = copy.deepcopy(ft_result["best_model"])

    print(f"\nTraining complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
