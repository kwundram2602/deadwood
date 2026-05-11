import torch
import torch.nn as nn

NODATA: int = 255


class MaskedBCELoss(nn.Module):
    """BCEWithLogitsLoss that ignores pixels where target == 255 (noData sentinel)."""

    def __init__(self):
        super().__init__()
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != NODATA
        if not valid.any():
            return logits.sum() * 0.0
        return self._bce(logits, target.float())[valid].mean()


class SoftDiceLoss(nn.Module):
    """Sørensen-Dice loss on continuous sigmoid outputs vs soft targets.

    Args:
        squared: if True uses squared denominator (p²+t²); default False uses
                 linear denominator (p+t) — more honest for fractional coverage.
    """

    def __init__(self, squared: bool = False, eps: float = 1e-6):
        super().__init__()
        self.squared = squared
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != NODATA
        if not valid.any():
            return logits.sum() * 0.0
        p = torch.sigmoid(logits)[valid]
        t = target[valid]
        inter = (p * t).sum()
        if self.squared:
            denom = (p**2).sum() + (t**2).sum()
        else:
            denom = p.sum() + t.sum()
        return 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)


class SoftIoULoss(nn.Module):
    """Soft Jaccard loss on continuous sigmoid outputs vs soft targets."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != NODATA
        if not valid.any():
            return logits.sum() * 0.0
        p = torch.sigmoid(logits)[valid]
        t = target[valid]
        inter = (p * t).sum()
        union = (p + t - p * t).sum()
        return 1.0 - (inter + self.eps) / (union + self.eps)


class MaskedMAELoss(nn.Module):
    """Mean absolute error between sigmoid(logits) and soft target, noData masked."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != NODATA
        if not valid.any():
            return logits.sum() * 0.0
        p = torch.sigmoid(logits)[valid]
        t = target[valid]
        return (p - t).abs().mean()


class CombinedLoss(nn.Module):
    """Weighted sum of active loss terms, configured from a DictConfig.

    Expected config keys: bce, dice, dice_squared, iou, mae.
    Terms with weight 0.0 are skipped. Raises if all weights are 0.
    """

    def __init__(self, cfg_loss) -> None:
        super().__init__()
        terms: list[tuple[float, nn.Module]] = []
        if float(cfg_loss.get("bce", 0.0)) > 0:
            terms.append((float(cfg_loss.bce), MaskedBCELoss()))
        if float(cfg_loss.get("dice", 0.0)) > 0:
            squared = bool(cfg_loss.get("dice_squared", False))
            terms.append((float(cfg_loss.dice), SoftDiceLoss(squared=squared)))
        if float(cfg_loss.get("iou", 0.0)) > 0:
            terms.append((float(cfg_loss.iou), SoftIoULoss()))
        if float(cfg_loss.get("mae", 0.0)) > 0:
            terms.append((float(cfg_loss.mae), MaskedMAELoss()))
        if not terms:
            raise ValueError("CombinedLoss: all loss weights are 0")
        self._terms = terms

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sum(w * loss_fn(logits, target) for w, loss_fn in self._terms)
