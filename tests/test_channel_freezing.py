import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.model import build_model
from training.learning_configurator import LearningConfigurator
from training.trainer import _build_optimizer


def _make_cfg(frozen_channels):
    return OmegaConf.create(
        {
            "model": {
                "type": "unet",
                "in_channels": 5,
                "num_classes": 1,
                "weights_name": None,
                "weights_path": None,
                "frozen_pretrained_channels": frozen_channels,
            }
        }
    )


def _first_conv(model):
    m = model.module if hasattr(model, "module") else model
    return m.encoder.conv1


def test_frozen_channels_survive_training_step():
    device = torch.device("cpu")
    model = build_model(_make_cfg([0, 1]), device)
    model = LearningConfigurator().prepare_model_for_transfer_learning(model)

    phase_cfg = OmegaConf.create(
        {
            "lr": 0.01,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "scheduler": "cos",
            "warmup_epochs": 0,
        }
    )
    opt, _ = _build_optimizer(phase_cfg, model)

    conv1 = _first_conv(model)
    before = conv1.weight.detach().clone()

    model.train()
    x = torch.randn(2, 5, 64, 64)
    opt.zero_grad()
    loss = model(x).mean()
    loss.backward()

    grad = conv1.weight.grad
    assert grad is not None, "conv1 must receive gradients"
    assert torch.all(grad[:, :2] == 0), "frozen channel grads must be exactly zero"
    assert torch.any(grad[:, 2:] != 0), "trainable channel grads must be nonzero"

    opt.step()

    assert torch.equal(conv1.weight[:, :2], before[:, :2]), (
        "frozen channel weights must be bit-identical after optimizer step"
    )
    assert not torch.equal(conv1.weight[:, 2:], before[:, 2:]), (
        "trainable channel weights must change after optimizer step"
    )


def test_bn1_trainable_alongside_conv1():
    device = torch.device("cpu")
    model = build_model(_make_cfg([0, 1]), device)
    model = LearningConfigurator().prepare_model_for_transfer_learning(model)

    m = model.module if hasattr(model, "module") else model
    assert all(p.requires_grad for p in m.encoder.bn1.parameters())


def test_invalid_frozen_channel_index_raises():
    with pytest.raises(ValueError, match="frozen_pretrained_channels"):
        build_model(_make_cfg([0, 5]), torch.device("cpu"))


def test_empty_frozen_list_trains_all_channels():
    device = torch.device("cpu")
    model = build_model(_make_cfg([]), device)
    model = LearningConfigurator().prepare_model_for_transfer_learning(model)

    conv1 = _first_conv(model)
    model.train()
    x = torch.randn(2, 5, 64, 64)
    model(x).mean().backward()

    assert conv1.weight.grad is not None
    assert torch.any(conv1.weight.grad[:, :2] != 0), (
        "with empty frozen list all channel slices must receive gradients"
    )
