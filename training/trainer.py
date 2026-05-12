import copy
from pathlib import Path

import torch
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

from training.losses import MaskedBCELoss
from training.metrics import MetricAccumulator


def train(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    phase_cfg: DictConfig,
    out_dir: Path,
    prefix: str,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
    threshold: float = 0.5,
) -> dict:
    """Run one training phase (transfer learning or fine-tuning).

    Args:
        model:        model with frozen/unfrozen params already configured
        train_loader / val_loader: DataLoaders
        phase_cfg:    OmegaConf node with epochs, lr, weight_decay, optimizer,
                      scheduler, patience
        out_dir:      directory to write checkpoints and plots
        prefix:       filename prefix for saved files (e.g. "tl", "ft")
        device:       training device
        criterion:    loss function; defaults to MaskedBCELoss if None
        threshold:    sigmoid threshold for binary metrics

    Returns:
        dict with keys "history" (metric lists) and "best_model" (deepcopy)
    """
    if criterion is None:
        criterion = MaskedBCELoss()

    opt, sched = _build_optimizer(phase_cfg, model)

    history: dict[str, list] = {
        "loss": [],
        "val_loss": [],
        "auc_pr": [],
        "val_auc_pr": [],
        "f1": [],
        "val_f1": [],
        "iou": [],
        "val_iou": [],
        "prec": [],
        "val_prec": [],
        "rec": [],
        "val_rec": [],
    }
    best_val_loss = float("inf")
    best_model: torch.nn.Module | None = None
    patience_counter = 0

    for epoch in range(phase_cfg.epochs):
        t_m = _run_epoch(
            model,
            train_loader,
            criterion,
            opt,
            device,
            train=True,
            threshold=threshold,
        )
        sched.step()
        v_m = _run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            train=False,
            threshold=threshold,
        )

        for key in ("loss", "auc_pr", "f1", "iou", "prec", "rec"):
            history[key].append(t_m[key])
            history[f"val_{key}"].append(v_m[key])

        print(
            f"[{prefix}] {epoch + 1}/{phase_cfg.epochs}  "
            f"loss={t_m['loss']:.4f} auc_pr={t_m['auc_pr']:.3f} "
            f"f1={t_m['f1']:.3f} iou={t_m['iou']:.3f}  "
            f"val_loss={v_m['loss']:.4f} val_auc_pr={v_m['auc_pr']:.3f} "
            f"val_f1={v_m['f1']:.3f} val_iou={v_m['iou']:.3f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

        if v_m["loss"] < best_val_loss:
            best_val_loss = v_m["loss"]
            patience_counter = 0
            best_model = copy.deepcopy(model)
            ckpt_path = out_dir / f"{prefix}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved {ckpt_path.name} (epoch {epoch + 1})")
        else:
            patience_counter += 1
            if patience_counter >= phase_cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    _save_dashboard(history, out_dir / f"{prefix}_dashboard.png")
    return {"history": history, "best_model": best_model}


def _save_dashboard(history: dict, save_path: Path) -> None:
    try:
        from utils.viz import plot_dashboard

        plot_dashboard(history, save_path)
    except Exception as e:
        print(f"Warning: could not save dashboard: {e}")


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    *,
    train: bool,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.train(train)
    accumulator = MetricAccumulator()

    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        desc = "train" if train else "val"
        for images, masks in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

            accumulator.update(logits.detach(), masks, loss.item(), images.size(0))

    return accumulator.compute(threshold=threshold)


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(
            "No trainable parameters — check LearningConfigurator freeze/unfreeze."
        )

    lr, wd = cfg.lr, cfg.weight_decay

    if cfg.optimizer == "adam":
        opt = optim.Adam(trainable, lr=lr, weight_decay=wd)
    elif cfg.optimizer == "adamw":
        opt = optim.AdamW(trainable, lr=lr, weight_decay=wd)
    elif cfg.optimizer == "sgd":
        opt = optim.SGD(trainable, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    if cfg.scheduler == "cos":
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=lr * 0.01
        )
    elif cfg.scheduler == "step":
        sched = optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)
    elif cfg.scheduler == "multistep":
        sched = optim.lr_scheduler.MultiStepLR(opt, list(range(5, 26)), gamma=0.85)
    elif cfg.scheduler == "anneal":
        sched = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    return opt, sched
