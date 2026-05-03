from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str | Path = "training_curves.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves -> {save_path}")


def show_batch(
    images: torch.Tensor,
    labels: torch.Tensor | None = None,
    n: int = 8,
    title: str = "",
) -> None:
    """Preview a batch. images: (B, C, H, W) float tensor."""
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    for i, ax in enumerate(axes if n > 1 else [axes]):
        img = images[i].detach().cpu()
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img.numpy().squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        ax.axis("off")
        if labels is not None:
            ax.set_title(str(labels[i].item()), fontsize=8)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
