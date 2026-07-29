# tests/test_channels.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.channels import NDSM, ChannelSpec, load_manifest  # noqa: E402

STACK7 = ["red", "green", "blue", "green_ms", "red_ms", "rededge", "nir"]
STACK_MS = ["green_ms", "red_ms", "rededge", "nir"]


# ---------------------------------------------------------------- manifest

def test_load_manifest_roundtrip(tmp_path):
    p = tmp_path / "channels.json"
    p.write_text(json.dumps({"names": STACK_MS}))
    assert load_manifest(p) == STACK_MS


def test_load_manifest_rejects_duplicates(tmp_path):
    p = tmp_path / "channels.json"
    p.write_text(json.dumps({"names": ["red", "red"]}))
    with pytest.raises(ValueError, match="unique"):
        load_manifest(p)


# ---------------------------------------------------------------- selection

def test_basic_selection_and_indexes():
    spec = ChannelSpec(STACK7, ["red", "green", "blue", "rededge", "nir", NDSM])
    assert spec.in_channels == 6
    assert spec.use_ndsm is True
    # 1-based indexes into STACK7, in input order, ndsm excluded
    assert spec.stack_indexes == [1, 2, 3, 6, 7]


def test_selection_without_ndsm():
    spec = ChannelSpec(STACK7, ["red", "green", "blue"])
    assert spec.use_ndsm is False
    assert spec.stack_indexes == [1, 2, 3]


def test_unknown_channel_rejected():
    with pytest.raises(ValueError, match="Unknown input_channels.*'nir_typo'"):
        ChannelSpec(STACK7, ["red", "nir_typo"])


def test_duplicate_input_channels_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        ChannelSpec(STACK7, ["red", "red"])


def test_ndsm_reserved_in_stack():
    with pytest.raises(ValueError, match="reserved"):
        ChannelSpec(["red", NDSM], ["red"])


def test_string_input_channels_rejected():
    """A YAML list missing its brackets must not be split into characters."""
    with pytest.raises(ValueError, match="list of channel names"):
        ChannelSpec(STACK7, "red, green, blue")


def test_string_stack_names_rejected():
    with pytest.raises(ValueError, match="list of channel names"):
        ChannelSpec("red, green, blue", ["red"])


def test_duplicate_stack_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        ChannelSpec(["red", "green", "red"], ["red", "green"])


# ---------------------------------------------------------------- pretrained

def test_exact_names_win_slots():
    spec = ChannelSpec(STACK7, ["red", "green", "blue", "green_ms", "red_ms", "rededge", "nir"])
    # position→slot: red@0→R(0), green@1→G(1), blue@2→B(2); *_ms unassigned
    assert spec.pretrained_assignment == {0: 0, 1: 1, 2: 2}


def test_prefix_match_when_no_exact():
    spec = ChannelSpec(STACK_MS, ["green_ms", "red_ms", "rededge", "nir", NDSM])
    # red_ms@1→R(0), green_ms@0→G(1); no blue candidate
    assert spec.pretrained_assignment == {1: 0, 0: 1}


def test_unassigned_slots_are_fine():
    spec = ChannelSpec(STACK7, ["rededge", "nir"])
    assert spec.pretrained_assignment == {}


def test_ambiguous_prefix_raises():
    stack = ["red_ms", "red_dji", "nir"]
    with pytest.raises(ValueError, match="pretrained_channel_map"):
        ChannelSpec(stack, ["red_ms", "red_dji", "nir"])


def test_explicit_map_overrides_convention():
    spec = ChannelSpec(
        STACK7,
        ["red", "green", "green_ms", "red_ms"],
        pretrained_channel_map={"red": "red_ms", "green": "green_ms"},
    )
    # map wins for red/green; blue falls back to convention (no candidate)
    assert spec.pretrained_assignment == {3: 0, 2: 1}


def test_explicit_map_none_suppresses_slot():
    spec = ChannelSpec(STACK7, ["red", "green", "blue"], pretrained_channel_map={"blue": None})
    assert spec.pretrained_assignment == {0: 0, 1: 1}


def test_map_rejects_bad_slot_and_bad_name():
    with pytest.raises(ValueError, match="keys"):
        ChannelSpec(STACK7, ["red"], pretrained_channel_map={"nir": "red"})
    with pytest.raises(ValueError, match="not in input_channels"):
        ChannelSpec(STACK7, ["red"], pretrained_channel_map={"green": "green_ms"})


# ---------------------------------------------------------------- assemble

def test_assemble_inserts_ndsm_at_position():
    spec = ChannelSpec(STACK_MS, ["green_ms", NDSM, "nir"])
    stack = np.stack([np.full((2, 2), 0.1), np.full((2, 2), 0.3)])  # green_ms, nir
    ndsm = np.full((1, 2, 2), 0.9)
    out = spec.assemble(stack.astype(np.float32), ndsm.astype(np.float32))
    assert out.shape == (3, 2, 2)
    assert out[0, 0, 0] == np.float32(0.1)
    assert out[1, 0, 0] == np.float32(0.9)
    assert out[2, 0, 0] == np.float32(0.3)


def test_assemble_without_ndsm_passthrough():
    spec = ChannelSpec(STACK_MS, ["rededge", "nir"])
    stack = np.zeros((2, 4, 4), dtype=np.float32)
    assert spec.assemble(stack, None).shape == (2, 4, 4)


def test_assemble_requires_ndsm_when_selected():
    spec = ChannelSpec(STACK_MS, ["nir", NDSM])
    with pytest.raises(ValueError, match="ndsm"):
        spec.assemble(np.zeros((1, 2, 2), dtype=np.float32), None)


# ---------------------------------------------------------------- stats

def test_norm_stats_subset_by_name_in_input_order():
    spec = ChannelSpec(STACK_MS, ["nir", "green_ms", NDSM])
    stats = {
        "names": STACK_MS + [NDSM],
        "mean": [0.1, 0.2, 0.3, 0.4, 0.5],
        "std": [1.1, 1.2, 1.3, 1.4, 1.5],
    }
    out = spec.norm_stats(stats)
    assert out["mean"] == [0.4, 0.1, 0.5]
    assert out["std"] == [1.4, 1.1, 1.5]


def test_norm_stats_missing_names_key_raises():
    spec = ChannelSpec(STACK_MS, ["nir"])
    with pytest.raises(ValueError, match="names"):
        spec.norm_stats({"mean": [0.0] * 5, "std": [1.0] * 5})


def test_norm_stats_missing_channel_raises():
    spec = ChannelSpec(STACK_MS, ["nir", NDSM])
    stats = {"names": ["nir"], "mean": [0.1], "std": [1.0]}
    with pytest.raises(ValueError, match="ndsm"):
        spec.norm_stats(stats)


# ---------------------------------------------------------------- freezing

def test_frozen_indices_resolved():
    spec = ChannelSpec(STACK_MS, ["green_ms", "red_ms", "rededge", "nir", NDSM])
    assert spec.frozen_indices(["red_ms", "green_ms"]) == [1, 0]


def test_frozen_unselected_channel_raises():
    spec = ChannelSpec(STACK_MS, ["rededge", "nir"])
    with pytest.raises(ValueError, match="not in input_channels"):
        spec.frozen_indices(["red_ms"])


def test_frozen_unpretrained_channel_raises():
    spec = ChannelSpec(STACK_MS, ["green_ms", "red_ms", "rededge", "nir"])
    with pytest.raises(ValueError, match="did not receive pretrained weights"):
        spec.frozen_indices(["nir"])


def test_frozen_empty_list_ok():
    spec = ChannelSpec(STACK_MS, ["nir"])
    assert spec.frozen_indices([]) == []
    assert spec.frozen_indices(None) == []


# ------------------------------------------------------------------ display

def test_ndsm_position_none_when_unused():
    assert ChannelSpec(STACK7, ["red", "green", "blue"]).ndsm_position is None


def test_ndsm_position_tracks_input_order():
    assert ChannelSpec(STACK7, ["red", NDSM, "green"]).ndsm_position == 1


def test_display_rgb_positions_true_colour():
    spec = ChannelSpec(STACK7, ["red", "green", "blue", NDSM])
    assert spec.display_rgb_positions == [0, 1, 2]


def test_display_rgb_positions_reorders_to_rgb():
    """Display order is R,G,B regardless of input order."""
    spec = ChannelSpec(STACK7, ["blue", "green", "red"])
    assert spec.display_rgb_positions == [2, 1, 0]


def test_display_rgb_positions_uses_ms_slots():
    """green_ms/red_ms win their slots; blue is absent, so it falls back."""
    spec = ChannelSpec(STACK_MS, ["green_ms", "red_ms", "rededge", "nir", NDSM])
    assert spec.display_rgb_positions == [0, 1, 2]


def test_display_rgb_positions_never_selects_ndsm():
    spec = ChannelSpec(STACK_MS, ["nir", NDSM])
    assert NDSM not in [spec.input_channels[p] for p in spec.display_rgb_positions]


def test_display_rgb_positions_pads_single_channel():
    spec = ChannelSpec(STACK_MS, ["nir"])
    assert spec.display_rgb_positions == [0, 0, 0]
