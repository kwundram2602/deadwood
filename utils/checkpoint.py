from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    path: str | Path,
    is_best: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metric": metric,
    }
    torch.save(state, path)
    if is_best:
        best = path.parent / "best.pt"
        torch.save(state, best)
        print(f"New best checkpoint -> {best}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded checkpoint: epoch {checkpoint.get('epoch', '?')}, metric {checkpoint.get('metric', '?'):.4f}")
    return checkpoint
