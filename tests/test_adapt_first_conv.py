# tests/test_adapt_first_conv.py
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model import _adapt_first_conv, _find_first_conv  # noqa: E402


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
