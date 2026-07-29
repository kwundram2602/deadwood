# tests/test_adapt_first_conv.py
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.channels import ChannelSpec  # noqa: E402
from models.model import _adapt_first_conv, _find_first_conv, build_model  # noqa: E402


def _tiny_encoder():
    enc = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
    with torch.no_grad():
        # slot-identifiable pretrained weights: column c filled with c+1
        for c in range(3):
            enc[0].weight[:, c] = float(c + 1)
    return enc


def test_pretrained_slots_land_at_assigned_positions():
    enc = _tiny_encoder()
    # 5 inputs; position 1 ← slot R(0), position 0 ← slot G(1); rest random
    _adapt_first_conv(enc, to_ch=5, assignment={1: 0, 0: 1})
    _, conv = _find_first_conv(enc)
    assert conv.in_channels == 5
    assert torch.all(conv.weight[:, 1] == 1.0)   # slot R copied to pos 1
    assert torch.all(conv.weight[:, 0] == 2.0)   # slot G copied to pos 0
    for pos in (2, 3, 4):                        # unassigned: kaiming, not constant
        col = conv.weight[:, pos]
        assert col.std() > 0


def test_identity_three_channel_assignment():
    enc = _tiny_encoder()
    _adapt_first_conv(enc, to_ch=3, assignment={0: 0, 1: 1, 2: 2})
    _, conv = _find_first_conv(enc)
    for c in range(3):
        assert torch.all(conv.weight[:, c] == float(c + 1))


def _model_cfg():
    return OmegaConf.create(
        {
            "model": {
                "type": "unet",
                "num_classes": 1,
                "weights_name": None,
                "weights_path": None,
                "frozen_pretrained_channels": None,
            }
        }
    )


def test_build_model_lands_pretrained_columns_at_swapped_positions_not_positionally():
    """End-to-end regression test through the public build_model() entry point.

    The stack is in native Mavic M3M order (green, red, rededge, nir), and the
    input spec deliberately does NOT list channels in R,G,B order. Under the
    naming convention (data/channels.py), 'red_ms' resolves to pretrained slot
    0 (red) but sits at input position 1, and 'green_ms' resolves to slot 1
    (green) but sits at input position 0 -- a genuine position/slot swap.

    Since weights_name=None means the encoder's "pretrained" first conv is
    just randomly initialised, we can't identify columns by value the way the
    unit tests above do with a constant-filled fixture. Instead we build an
    identity-spec (3-channel, no adaptation) reference model with the same
    torch seed to capture what the base encoder's first-conv columns are
    *before* any adaptation, then build the real (swapped) model with the
    same seed and check that column 0 (red) of the reference landed at
    position 1, and column 1 (green) landed at position 0 -- not at their
    own positional index. If the copy in _adapt_first_conv were positional
    (assignment ignored, or transposed) rather than slot-based, this would
    fail.
    """
    stack_names = ["green_ms", "red_ms", "rededge", "nir"]
    input_channels = ["green_ms", "red_ms", "rededge", "nir", "ndsm"]
    spec = ChannelSpec(stack_names, input_channels)
    # Sanity: this really is a swap, not a coincidental identity mapping.
    assert spec.pretrained_assignment == {1: 0, 0: 1}

    identity_spec = ChannelSpec(["red", "green", "blue"], ["red", "green", "blue"])
    cfg = _model_cfg()

    torch.manual_seed(1234)
    ref_model = build_model(cfg, torch.device("cpu"), identity_spec)
    _, ref_conv = _find_first_conv(ref_model.encoder)
    ref_weight = ref_conv.weight.detach().clone()

    torch.manual_seed(1234)
    model = build_model(cfg, torch.device("cpu"), spec)
    _, conv = _find_first_conv(model.encoder)

    assert conv.in_channels == 5
    # red_ms (pretrained slot 0 / red) must land at input position 1.
    assert torch.allclose(conv.weight[:, 1], ref_weight[:, 0])
    # green_ms (pretrained slot 1 / green) must land at input position 0.
    assert torch.allclose(conv.weight[:, 0], ref_weight[:, 1])
    # And NOT at their own positional index -- this is the regression check:
    # a positional (rather than name/slot-based) copy would put slot 0 at
    # position 0 and slot 1 at position 1.
    assert not torch.allclose(conv.weight[:, 0], ref_weight[:, 0])
    assert not torch.allclose(conv.weight[:, 1], ref_weight[:, 1])
