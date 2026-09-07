import os
import sys

import safetensors.torch
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.raw_predict_deadwood import _load_state_dict


def _tiny_state():
    return {"encoder.weight": torch.ones(2, 2), "head.bias": torch.zeros(3)}


def test_loads_safetensors(tmp_path):
    path = tmp_path / "w.safetensors"
    safetensors.torch.save_file(_tiny_state(), str(path))
    state = _load_state_dict(path)
    assert set(state) == {"encoder.weight", "head.bias"}
    assert torch.equal(state["encoder.weight"], torch.ones(2, 2))


def test_loads_torch_pt(tmp_path):
    path = tmp_path / "ft_best.pt"
    torch.save(_tiny_state(), path)
    state = _load_state_dict(path)
    assert set(state) == {"encoder.weight", "head.bias"}
    assert torch.equal(state["head.bias"], torch.zeros(3))


def test_strips_torch_compile_prefix(tmp_path):
    # The published checkpoint was saved from a torch.compile wrapper, so every
    # key carries _orig_mod.; a plain module cannot load it strictly otherwise.
    path = tmp_path / "w.safetensors"
    safetensors.torch.save_file({"_orig_mod.encoder.weight": torch.ones(2, 2)}, str(path))
    assert set(_load_state_dict(path)) == {"encoder.weight"}
