import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.model import build_model
from training.learning_configurator import LearningConfigurator


def _make_model():
    cfg = OmegaConf.create(
        {
            "model": {
                "type": "unet",
                "in_channels": 5,
                "num_classes": 1,
                "weights_name": None,
                "weights_path": None,
                "frozen_pretrained_channels": [],
            }
        }
    )
    return build_model(cfg, torch.device("cpu"))


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _all_trainable(module):
    return all(p.requires_grad for p in module.parameters())


def _all_frozen(module):
    return all(not p.requires_grad for p in module.parameters())


def test_subblock_key_unfreezes_only_that_subblock():
    model = _make_model()
    m = _unwrap(model)
    last = str(len(m.encoder.layer4) - 1)

    LearningConfigurator().prepare_model_for_fine_tuning(model, [f"layer4.{last}"])

    assert _all_trainable(m.encoder.layer4[int(last)])
    for idx in range(len(m.encoder.layer4) - 1):
        assert _all_frozen(m.encoder.layer4[idx]), f"layer4.{idx} must stay frozen"
    assert _all_frozen(m.encoder.layer3)
    assert _all_trainable(m.decoder)
    assert _all_trainable(m.segmentation_head)
    assert _all_trainable(m.encoder.conv1)
    assert _all_trainable(m.encoder.bn1)


def test_full_block_key_unfreezes_whole_block():
    model = _make_model()
    m = _unwrap(model)

    LearningConfigurator().prepare_model_for_fine_tuning(model, ["layer4"])

    assert _all_trainable(m.encoder.layer4)
    assert _all_frozen(m.encoder.layer3)
    assert _all_frozen(m.encoder.layer2)
    assert _all_frozen(m.encoder.layer1)


def test_unknown_key_raises_with_valid_keys_listed():
    model = _make_model()
    with pytest.raises(ValueError, match="layer9"):
        LearningConfigurator().prepare_model_for_fine_tuning(model, ["layer9"])
    # error message doubles as key discovery: it must list real keys
    with pytest.raises(ValueError, match=r"layer4\.0"):
        LearningConfigurator().prepare_model_for_fine_tuning(model, ["layer9"])


def test_empty_keys_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="unfreeze_keys"):
        LearningConfigurator().prepare_model_for_fine_tuning(model, [])


def test_table_shows_subblock_rows_and_partial_status(capsys):
    model = _make_model()
    m = _unwrap(model)
    last = str(len(m.encoder.layer4) - 1)

    LearningConfigurator().prepare_model_for_fine_tuning(model, [f"layer4.{last}"])

    out = capsys.readouterr().out
    assert "PARTIAL" in out, "mixed block must show PARTIAL status"
    assert f"layer4.{last}" in out, "targeted sub-block must have its own row"
    assert "layer4.0" in out, "frozen sibling sub-blocks must have rows too"
    assert "layer1.0" in out, "sub-block rows must appear for all encoder layers"
