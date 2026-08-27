# tests/test_plot_samples.py
"""plot_samples must adapt to any input_channels selection, not a fixed 5-band layout."""

import sys
from pathlib import Path

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.channels import NDSM, ChannelSpec  # noqa: E402
from utils.viz import plot_samples  # noqa: E402

STACK = ["red", "green", "blue", "rededge", "nir"]


class _OneChannelOut(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, 1, 1)

    def forward(self, x):
        return self.conv(x)


def _loader(in_ch, n=2, size=8):
    images = torch.rand(n, in_ch, size, size)
    masks = torch.randint(0, 2, (n, 1, size, size)).float()
    return [(images, masks)]


@pytest.mark.parametrize(
    "input_channels",
    [
        ["red", "green", "blue", NDSM],  # nDSM panel present
        ["red", "green", "blue"],  # no nDSM channel -> panel dropped
        ["nir", NDSM],  # single spectral band, padded pseudo-RGB
        ["red", "green", "blue", "rededge", "nir", NDSM],  # 6 channels
    ],
)
def test_plot_samples_matches_channel_selection(tmp_path, input_channels):
    spec = ChannelSpec(STACK, input_channels)
    model = _OneChannelOut(spec.in_channels)
    out = tmp_path / "samples.png"

    plot_samples(model, _loader(spec.in_channels), torch.device("cpu"), spec, n=2, save_path=out)

    assert out.exists() and out.stat().st_size > 0
    assert len(matplotlib.pyplot.get_fignums()) == 0, "figure was not closed"


def test_plot_samples_column_titles_follow_ndsm(tmp_path, monkeypatch):
    """The nDSM column exists only when ndsm is a selected input channel."""
    seen: list[str] = []
    orig = matplotlib.axes.Axes.set_title
    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "set_title",
        lambda self, label, *a, **k: (seen.append(label), orig(self, label, *a, **k))[1],
    )

    spec = ChannelSpec(STACK, ["red", "green", "blue", NDSM])
    plot_samples(
        _OneChannelOut(4),
        _loader(4),
        torch.device("cpu"),
        spec,
        n=1,
        save_path=tmp_path / "a.png",
    )
    assert "nDSM" in seen

    seen.clear()
    spec = ChannelSpec(STACK, ["red", "green", "blue"])
    plot_samples(
        _OneChannelOut(3),
        _loader(3),
        torch.device("cpu"),
        spec,
        n=1,
        save_path=tmp_path / "b.png",
    )
    assert "nDSM" not in seen
    assert seen == ["Pseudo-RGB", "GT Mask", "Model sigma (masked)", "Model sigma (full)"]
