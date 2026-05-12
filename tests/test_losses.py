import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.losses import (
    CombinedLoss,
    MaskedMAELoss,
    SoftDiceLoss,
    SoftIoULoss,
)


def test_soft_dice_perfect_prediction():
    logits = torch.full((2, 1, 4, 4), 5.0)
    target = torch.ones(2, 1, 4, 4)
    assert SoftDiceLoss()(logits, target).item() < 0.05


def test_soft_dice_zero_prediction():
    logits = torch.full((2, 1, 4, 4), -5.0)
    target = torch.ones(2, 1, 4, 4)
    assert SoftDiceLoss()(logits, target).item() > 0.9


def test_soft_dice_nodata_returns_zero():
    logits = torch.full((2, 1, 4, 4), -5.0)
    target = torch.full((2, 1, 4, 4), 255.0)
    assert SoftDiceLoss()(logits, target).item() == pytest.approx(0.0)


def test_soft_dice_squared_differs_from_linear():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    linear = SoftDiceLoss(squared=False)(logits, target).item()
    squared = SoftDiceLoss(squared=True)(logits, target).item()
    assert abs(linear - squared) > 1e-4


def test_soft_iou_perfect_prediction():
    logits = torch.full((2, 1, 4, 4), 5.0)
    target = torch.ones(2, 1, 4, 4)
    assert SoftIoULoss()(logits, target).item() < 0.05


def test_soft_iou_nodata_returns_zero():
    logits = torch.full((2, 1, 4, 4), 5.0)
    target = torch.full((2, 1, 4, 4), 255.0)
    assert SoftIoULoss()(logits, target).item() == pytest.approx(0.0)


def test_masked_mae_perfect_prediction():
    logits = torch.full((2, 1, 4, 4), 5.0)
    target = torch.ones(2, 1, 4, 4)
    assert MaskedMAELoss()(logits, target).item() < 0.05


def test_masked_mae_nodata_returns_zero():
    logits = torch.randn(2, 1, 4, 4)
    target = torch.full((2, 1, 4, 4), 255.0)
    assert MaskedMAELoss()(logits, target).item() == pytest.approx(0.0)


def test_combined_loss_single_dice_matches_dice_alone():
    cfg = OmegaConf.create(
        {"bce": 0.0, "dice": 1.0, "dice_squared": False, "iou": 0.0, "mae": 0.0}
    )
    logits = torch.randn(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)
    combined = CombinedLoss(cfg)(logits, target).item()
    expected = SoftDiceLoss()(logits, target).item()
    assert combined == pytest.approx(expected, abs=1e-5)


def test_combined_loss_all_zero_weights_raises():
    cfg = OmegaConf.create(
        {"bce": 0.0, "dice": 0.0, "dice_squared": False, "iou": 0.0, "mae": 0.0}
    )
    with pytest.raises(ValueError, match="all loss weights are 0"):
        CombinedLoss(cfg)


def test_combined_loss_weighted_sum():
    cfg = OmegaConf.create(
        {"bce": 0.0, "dice": 0.5, "dice_squared": False, "iou": 0.5, "mae": 0.0}
    )
    logits = torch.randn(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)
    combined = CombinedLoss(cfg)(logits, target).item()
    expected = (
        0.5 * SoftDiceLoss()(logits, target).item()
        + 0.5 * SoftIoULoss()(logits, target).item()
    )
    assert combined == pytest.approx(expected, abs=1e-5)
