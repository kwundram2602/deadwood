from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_NODATA = 255


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


def plot_dashboard(
    history: dict[str, list],
    save_path: str | Path = "dashboard.png",
    threshold: float = 0.5,
    target_threshold: float = 0.5,
) -> None:
    """2x3 dashboard: Loss, AUC-PR, F1, Precision, Recall, IoU (train + val per panel)."""
    panels = [
        ("loss", "Loss"),
        ("auc_pr", "AUC-PR"),
        ("f1", "F1"),
        ("prec", "Precision"),
        ("rec", "Recall"),
        ("iou", "IoU"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, label) in zip(axes.flat, panels):
        if key in history:
            ax.plot(history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="val", linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"pred_thresh={threshold}  target_thresh={target_threshold}", fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved dashboard -> {save_path}")


def plot_loss_parts(
    history: dict[str, list],
    save_path: str | Path = "loss_parts.png",
) -> None:
    """One panel per active loss term, showing weighted train + val curves."""
    part_keys = [k for k in history if k.startswith("loss_") and k != "loss"]
    if not part_keys:
        return
    n = len(part_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, key in zip(axes[0], part_keys):
        name = key[len("loss_"):]
        ax.plot(history[key], label="train")
        val_key = f"val_{key}"
        if val_key in history:
            ax.plot(history[val_key], label="val", linestyle="--")
        ax.set_title(f"{name} (weighted)")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss parts -> {save_path}")


def plot_final_bars(
    train_m: dict[str, float],
    val_m: dict[str, float],
    test_m: dict[str, float],
    save_path: str | Path = "eval_bars.png",
) -> None:
    """Grouped bar chart: one group per metric, three bars (train/val/test)."""
    keys = ["auc_pr", "f1", "iou", "prec", "rec"]
    labels = ["AUC-PR", "F1", "IoU", "Precision", "Recall"]
    x = np.arange(len(keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_groups = [
        (x - width, train_m, "train"),
        (x, val_m, "val"),
        (x + width, test_m, "test"),
    ]
    for positions, metrics, split_label in bar_groups:
        bars = ax.bar(
            positions,
            [metrics.get(k, 0.0) for k in keys],
            width,
            label=split_label,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved final bars -> {save_path}")


def plot_samples(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    n: int = 6,
    threshold: float = 0.5,
    save_path: str | Path = "samples.png",
) -> None:
    """5-panel per-sample visualization: Pseudo-RGB | CIR | nDSM | GT Mask | Model sigma.

    Band layout of the 5-channel input tensor:
        0=R, 1=G, 2=RedEdge, 3=NIR, 4=nDSM
    Pseudo-RGB: bands [0,1,2] -> R,G,B (RedEdge fills the missing blue channel)
    CIR:        bands [3,0,1] -> R,G,B (NIR,R,G -- standard vegetation false-colour)
    """
    model.eval()
    images_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []
    preds_list: list[torch.Tensor] = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            for i in range(images.size(0)):
                images_list.append(images[i].cpu())
                masks_list.append(masks[i].cpu())
                preds_list.append(probs[i].cpu())
                if len(images_list) >= n:
                    break
            if len(images_list) >= n:
                break

    n_actual = len(images_list)
    fig, axes = plt.subplots(n_actual, 5, figsize=(15, 3 * n_actual))
    if n_actual == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Pseudo-RGB", "CIR", "nDSM", "GT Mask", "Model sigma"]
    im_pred = None
    for row, (img, mask, pred) in enumerate(zip(images_list, masks_list, preds_list)):
        rgb = img[[0, 1, 2]].permute(1, 2, 0).numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        cir = img[[3, 0, 1]].permute(1, 2, 0).numpy()
        cir = (cir - cir.min()) / (cir.max() - cir.min() + 1e-8)

        ndsm = img[4].numpy()

        gt = mask.squeeze(0).numpy()
        nodata_mask = gt == _NODATA
        gt_display = np.where(nodata_mask, np.nan, gt)

        pred_np = pred.squeeze(0).numpy()
        pred_display = np.where(nodata_mask, np.nan, pred_np)

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(cir)
        axes[row, 2].imshow(ndsm, cmap="gray")

        axes[row, 3].imshow(gt_display, cmap="viridis", vmin=0, vmax=1)
        axes[row, 3].imshow(
            np.where(nodata_mask, 1.0, np.nan),
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.4,
        )

        im_pred = axes[row, 4].imshow(pred_display, cmap="viridis", vmin=0, vmax=1)
        axes[row, 4].imshow(
            np.where(nodata_mask, 1.0, np.nan),
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.4,
        )

        for col in range(5):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10)

    if im_pred is not None:
        fig.colorbar(im_pred, ax=axes[:, 3:], shrink=0.6, label="probability")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved samples -> {save_path}")
