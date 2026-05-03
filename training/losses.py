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
            return logits.sum() * 0.0  # differentiable zero
        return self._bce(logits, target.float())[valid].mean()
