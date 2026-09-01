# tests/test_plot_samples.py
"""plot_samples must adapt to any input_channels selection, not a fixed 5-band layout."""

import sys
from pathlib import Path

import matplotlib
import numpy
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
    assert seen == ["Pseudo-RGB", "GT Mask", "Model sigma"]


def test_plot_samples_blanks_prediction_only_outside_footprint(tmp_path, monkeypatch):
    """Out-of-footprint pixels are dropped from the sigma panel; unlabelled are not.

    Predicting over unlabelled ground is the intended behaviour — the imagery is
    real, only the label is missing — so that panel must keep its values.
    """
    from utils.nodata import MASK_OUTSIDE, MASK_UNLABELLED

    spec = ChannelSpec(STACK, ["red", "green", "blue"])
    images = torch.rand(1, 3, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, 0] = MASK_OUTSIDE
    mask[0, 0, 1] = MASK_UNLABELLED

    drawn: list = []
    orig = matplotlib.axes.Axes.imshow
    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "imshow",
        lambda self, X, *a, **k: (drawn.append(X), orig(self, X, *a, **k))[1],
    )

    plot_samples(
        _OneChannelOut(3), [(images, mask)], torch.device("cpu"), spec,
        n=1, save_path=tmp_path / "s.png",
    )

    # panels in order: pseudo-RGB, GT, GT overlay, sigma
    sigma = drawn[-1]
    assert numpy.isnan(sigma[0]).all(), "out-of-footprint row still drawn"
    assert not numpy.isnan(sigma[1]).any(), "unlabelled row was wrongly blanked"
    assert not numpy.isnan(sigma[2:]).any()
