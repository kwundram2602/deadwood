import copy
from pathlib import Path

import torch
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

from training.losses import MaskedBCELoss
from training.metrics import pixel_metrics


def train(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    phase_cfg: DictConfig,
    out_dir: Path,
    prefix: str,
    device: torch.device,
) -> dict:
    """Run one training phase (transfer learning or fine-tuning).

    Args:
        model:       model with frozen/unfrozen params already configured
        train_loader / val_loader: DataLoaders
        phase_cfg:   OmegaConf node with epochs, lr, weight_decay, optimizer,
                     scheduler, patience
        out_dir:     directory to write checkpoints
        prefix:      filename prefix for saved checkpoints (e.g. "tl", "ft")
        device:      training device

    Returns:
        dict with keys "history" (loss/acc lists) and "best_model" (deepcopy)
    """
    criterion = MaskedBCELoss()
    opt, sched = _build_optimizer(phase_cfg, model)

    history: dict[str, list] = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model: torch.nn.Module | None = None
    patience_counter = 0

    for epoch in range(phase_cfg.epochs):
        t_loss, t_acc = _run_epoch(
            model, train_loader, criterion, opt, device, train=True
        )
        sched.step()
        v_loss, v_acc = _run_epoch(
            model, val_loader, criterion, None, device, train=False
        )

        history["loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(
            f"[{prefix}] {epoch + 1}/{phase_cfg.epochs}  "
            f"loss={t_loss:.4f} acc={t_acc:.3f}  "
            f"val_loss={v_loss:.4f} val_acc={v_acc:.3f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
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

    return {"history": history, "best_model": best_model}


def _run_epoch(model, loader, criterion, optimizer, device, *, train: bool):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0

    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for images, masks in tqdm(loader, desc="train" if train else "val", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

            m = pixel_metrics(logits.detach(), masks)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_acc += m["acc"] * bs
            n += bs

    return total_loss / n, total_acc / n


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
