import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve

NODATA: int = 255


def pixel_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, IoU over valid (non-noData) pixels.

    Args:
        logits: raw model output (N, 1, H, W)
        target: soft mask (N, 1, H, W), 255 = noData
        threshold: sigmoid threshold for positive class
    """
    valid = (target != NODATA).squeeze(1)
    preds = torch.sigmoid(logits.squeeze(1)) >= threshold
    tgt = target.squeeze(1) > 0.5

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
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "iou": iou}


class MetricAccumulator:
    """Collects per-batch logits/targets across an epoch, computes all metrics at end.

    Usage:
        acc = MetricAccumulator()
        for images, masks in loader:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward(); optimizer.step()
            acc.update(logits.detach(), masks, loss.item(), images.size(0))
        scores = acc.compute(threshold=0.5)
        acc.reset()
    """

    def __init__(self) -> None:
        self._probs: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._loss_sum: float = 0.0
        self._n: int = 0

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_val: float,
        batch_size: int,
    ) -> None:
        self._probs.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        self._targets.append(target.detach().cpu().numpy().ravel())
        self._loss_sum += loss_val * batch_size
        self._n += batch_size

    def compute(self, threshold: float = 0.5) -> dict[str, float]:
        probs = np.concatenate(self._probs)
        tgts = np.concatenate(self._targets)
        valid = tgts != NODATA

        if not valid.any():
            return {
                "loss": self._loss_sum / self._n if self._n > 0 else 0.0,
                "acc": 0.0, "prec": 0.0, "rec": 0.0,
                "f1": 0.0, "iou": 0.0, "auc_pr": 0.0,
            }

        probs_v = probs[valid]
        tgts_v = tgts[valid]
        t_bin = tgts_v > 0.5
        preds = probs_v >= threshold

        tp = int((preds & t_bin).sum())
        fp = int((preds & ~t_bin).sum())
        fn = int((~preds & t_bin).sum())
        tn = int((~preds & ~t_bin).sum())
        total = tp + fp + fn + tn

        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        has_pos = t_bin.sum() > 0
        has_neg = (~t_bin).sum() > 0
        if has_pos and has_neg:
            prec_curve, rec_curve, _ = precision_recall_curve(
                t_bin.astype(int), probs_v
            )
            auc_pr = float(auc(rec_curve, prec_curve))
        else:
            auc_pr = 0.0

        return {
            "loss": self._loss_sum / self._n if self._n > 0 else 0.0,
            "acc": acc, "prec": prec, "rec": rec,
            "f1": f1, "iou": iou, "auc_pr": auc_pr,
        }

    def reset(self) -> None:
        self._probs.clear()
        self._targets.clear()
        self._loss_sum = 0.0
        self._n = 0
