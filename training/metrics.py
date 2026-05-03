import torch

NODATA: int = 255


def pixel_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 over valid (non-noData) pixels.

    Args:
        logits: raw model output (N, 1, H, W)
        target: soft mask (N, 1, H, W), 255 = noData
        threshold: sigmoid threshold for positive class
    """
    valid = (target != NODATA).squeeze(1)  # (N, H, W) bool
    preds = (torch.sigmoid(logits.squeeze(1)) >= threshold)
    tgt = (target.squeeze(1) > 0.5)

    p = preds[valid]
    t = tgt[valid]

    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    tn = (~p & ~t).sum().item()
    total = tp + fp + fn + tn

    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
