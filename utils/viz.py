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
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    lines, labels = [], []
    for ax, (key, label) in zip(axes.flat, panels):
        if key in history:
            (l,) = ax.plot(history[key], label="train")
            if not labels:
                lines.append(l); labels.append("train")
        if f"val_{key}" in history:
            (l,) = ax.plot(history[f"val_{key}"], label="val", linestyle="--")
            if len(labels) < 2:
                lines.append(l); labels.append("val")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.legend(lines, labels, loc="lower center", ncol=2, fontsize=9, framealpha=0.9)
    fig.suptitle(f"pred_thresh={threshold}  target_thresh={target_threshold}", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
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


def plot_final_bars_multi(
    results: dict[tuple[float, float], dict[str, dict[str, float]]],
    thresholds: list[float],
    target_thresholds: list[float],
    out_dir: Path,
    stem: str = "eval",
) -> None:
    """One figure per metric; subplots arranged as rows=target_thresholds × cols=thresholds.

    Args:
        results: keyed by (target_threshold, threshold) → {"train": metrics, "val": metrics, "test": metrics}
        thresholds: ordered list of prediction thresholds (columns)
        target_thresholds: ordered list of GT binarisation thresholds (rows)
        out_dir: directory to write <stem>_<metric>.png files
        stem: filename prefix
    """
    metrics_cfg = [
        ("auc_pr", "AUC-PR"),
        ("f1", "F1"),
        ("iou", "IoU"),
        ("prec", "Precision"),
        ("rec", "Recall"),
        ("acc", "Accuracy"),
    ]
    n_rows = len(target_thresholds)
    n_cols = len(thresholds)
    width = 0.25
    x = np.arange(3)  # train / val / test

    for metric_key, metric_label in metrics_cfg:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(metric_label, fontsize=13, y=1.01)

        for r, tt in enumerate(target_thresholds):
            for c, t in enumerate(thresholds):
                ax = axes[r, c]
                split_data = results.get((tt, t), {})
                values = [
                    split_data.get("train", {}).get(metric_key, 0.0),
                    split_data.get("val", {}).get(metric_key, 0.0),
                    split_data.get("test", {}).get(metric_key, 0.0),
                ]
                colors = ["#4c72b0", "#dd8452", "#55a868"]
                for i, (v, color) in enumerate(zip(values, colors)):
                    bar = ax.bar(x[i], v, width * 2.5, color=color, label=["train", "val", "test"][i])
                    ax.annotate(
                        f"{v:.3f}",
                        xy=(x[i], v),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
                ax.set_title(f"tgt≥{tt}  pred≥{t}", fontsize=9)
                ax.set_xticks(x)
                ax.set_xticklabels(["train", "val", "test"], fontsize=8)
                ax.set_ylim(0, 1.15)
                ax.grid(alpha=0.3, axis="y")
                if r == 0 and c == n_cols - 1:
                    ax.legend(
                        fontsize=7,
                        loc="upper left",
                        bbox_to_anchor=(1.04, 1),
                        borderaxespad=0,
                    )

        fig.tight_layout(rect=[0, 0, 0.85, 1])
        save_path = Path(out_dir) / f"{stem}_{metric_key}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {metric_label} plot -> {save_path}")


def plot_samples(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    n: int = 6,
    threshold: float = 0.5,
    save_path: str | Path = "samples.png",
) -> None:
    """4-panel per-sample visualization: Pseudo-RGB | nDSM | GT Mask | Model sigma.

    Band layout of the 5-channel input tensor:
        0=R, 1=G, 2=RedEdge, 3=NIR, 4=nDSM
    Pseudo-RGB: bands [0,1,2] -> R,G,B (RedEdge fills the missing blue channel)
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

    col_titles = ["Pseudo-RGB", "nDSM", "GT Mask", "Model sigma (masked)", "Model sigma (full)"]
    im_pred = None
    for row, (img, mask, pred) in enumerate(zip(images_list, masks_list, preds_list)):
        rgb = img[[0, 1, 2]].permute(1, 2, 0).numpy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        ndsm = img[4].numpy()

        gt = mask.squeeze(0).numpy()
        nodata_mask = gt == _NODATA
        gt_display = np.where(nodata_mask, np.nan, gt)

        pred_np = pred.squeeze(0).numpy()
        pred_display = np.where(nodata_mask, np.nan, pred_np)

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(ndsm, cmap="gray")

        axes[row, 2].imshow(gt_display, cmap="viridis", vmin=0, vmax=1)
        axes[row, 2].imshow(
            np.where(nodata_mask, 1.0, np.nan),
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.4,
        )

        im_pred = axes[row, 3].imshow(pred_display, cmap="viridis", vmin=0, vmax=1)
        axes[row, 3].imshow(
            np.where(nodata_mask, 1.0, np.nan),
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.4,
        )

        axes[row, 4].imshow(pred_np, cmap="viridis", vmin=0, vmax=1)

        for col in range(5):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10)

    if im_pred is not None:
        fig.colorbar(im_pred, ax=axes[:, 2:4], shrink=0.6, label="probability")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved samples -> {save_path}")
