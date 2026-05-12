import numpy as np
import torch
from sklearn.metrics import average_precision_score

NODATA: int = 255
_EPS: float = 1e-6


class MetricAccumulator:
    """Collects per-batch logits/targets across an epoch, computes all metrics at end.

    Usage:
        acc = MetricAccumulator()
        for images, masks in loader:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward(); optimizer.step()
            n_valid = (masks != 255).sum().item()
            acc.update(logits.detach(), masks, loss.item(), n_valid)
        scores = acc.compute(threshold=0.5)
        acc.reset()
    """

    def __init__(self) -> None:
        self._probs: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._loss_sum: float = 0.0
        self._n_valid: int = 0

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_val: float,
        n_valid_pixels: int,
    ) -> None:
        if n_valid_pixels <= 0:
            return
        self._probs.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        self._targets.append(target.detach().cpu().numpy().ravel())
        self._loss_sum += loss_val * n_valid_pixels
        self._n_valid += n_valid_pixels

    def compute(self, threshold: float = 0.5, target_threshold: float = 0.5) -> dict[str, float]:
        _zero = {
            "loss": self._loss_sum / self._n_valid if self._n_valid > 0 else 0.0,
            "acc": 0.0,
            "prec": 0.0,
            "rec": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "soft_iou": 0.0,
            "auc_pr": 0.0,
        }

        if not self._probs:
            return _zero

        probs = np.concatenate(self._probs)
        tgts = np.concatenate(self._targets)
        valid = tgts != NODATA

        if not valid.any():
            return _zero

        probs_v = probs[valid]
        tgts_v = tgts[valid]
        t_bin = tgts_v > target_threshold
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

        inter = float((probs_v * tgts_v).sum())
        union = float((probs_v + tgts_v - probs_v * tgts_v).sum())
        soft_iou = (inter + _EPS) / (union + _EPS)

        has_pos = t_bin.sum() > 0
        has_neg = (~t_bin).sum() > 0
        auc_pr = (
            float(average_precision_score(t_bin.astype(int), probs_v))
            if has_pos and has_neg
            else 0.0
        )

        return {
            "loss": self._loss_sum / self._n_valid if self._n_valid > 0 else 0.0,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "iou": iou,
            "soft_iou": soft_iou,
            "auc_pr": auc_pr,
        }

    def reset(self) -> None:
        self._probs.clear()
        self._targets.clear()
        self._loss_sum = 0.0
        self._n_valid = 0
