# tests/test_nodata_footprint.py
"""The pipeline must tell "outside the recorded scene" apart from
"inside the scene but unlabelled" — see utils/nodata.py."""
import os
import sys

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from explore_and_process.rasterize_crowns import (  # noqa: E402
    read_source_footprint,
    scene_footprint,
    stack_sources,
)
from explore_and_process.tile_patches import tile_raster  # noqa: E402
from training.losses import MaskedBCELoss  # noqa: E402
from utils.data import compute_channel_stats  # noqa: E402
from utils.nodata import (  # noqa: E402
    MASK_OUTSIDE,
    MASK_UNLABELLED,
    footprint_from_stack,
    valid_target,
)

CRS = "EPSG:25832"
TRANSFORM = from_bounds(0, 0, 8, 8, 8, 8)


def _write(path, data, nodata=np.nan):
    if data.ndim == 2:
        data = data[np.newaxis]
    profile = dict(
        driver="GTiff", dtype="float32", width=data.shape[2], height=data.shape[1],
        count=data.shape[0], crs=CRS, transform=TRANSFORM, nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
    return str(path)


# ------------------------------------------------------------ valid_target
def test_valid_target_excludes_both_sentinels():
    mask = np.array([0.0, 0.5, 1.0, MASK_UNLABELLED, MASK_OUTSIDE], dtype=np.float32)
    assert valid_target(mask).tolist() == [True, True, True, False, False]


def test_valid_target_works_on_torch_tensors():
    mask = torch.tensor([0.0, 1.0, MASK_UNLABELLED, MASK_OUTSIDE])
    assert valid_target(mask).tolist() == [True, True, False, False]


def test_loss_ignores_outside_footprint_pixels():
    """A patch differing only beyond the footprint must give the same loss."""
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 4, 4)
    base = torch.full((1, 1, 4, 4), 0.5)
    base[0, 0, :2] = MASK_OUTSIDE
    other = base.clone()
    other[0, 0, :2] = MASK_UNLABELLED  # same pixels, other sentinel

    loss = MaskedBCELoss()
    assert torch.allclose(loss(logits, base), loss(logits, other))


# --------------------------------------------------------- footprint_from_stack
def test_footprint_from_stack_uses_nan_when_present():
    bands = np.zeros((3, 4, 4), dtype=np.float32)
    bands[:, 0, 0] = np.nan
    # a genuine all-zero pixel inside the footprint must stay valid
    fp = footprint_from_stack(bands)
    assert not fp[0, 0]
    assert fp[1, 1]


def test_footprint_from_stack_falls_back_for_legacy_stacks():
    """Patches written before the footprint was tracked carry no NaN."""
    bands = np.ones((3, 4, 4), dtype=np.float32)
    bands[:, 0, 0] = 0.0
    fp = footprint_from_stack(bands)
    assert not fp[0, 0]
    assert fp[1, 1]


def test_footprint_requires_all_bands(tmp_path):
    """A pixel valid in one band but not another is outside the footprint."""
    bands = np.ones((2, 4, 4), dtype=np.float32)
    bands[0, 2, 2] = np.nan  # only the first band has a gap
    assert not footprint_from_stack(bands)[2, 2]


# ------------------------------------------------------------ scene_footprint
def test_scene_footprint_intersects_sources(tmp_path):
    a = np.ones((8, 8), dtype=np.float32)
    a[:3] = np.nan
    b = np.ones((8, 8), dtype=np.float32)
    b[:, :2] = np.nan
    pa = _write(tmp_path / "a.tif", a)
    pb = _write(tmp_path / "b.tif", b)

    fp = scene_footprint([(pa, [1], ["a"]), (pb, [1], ["b"])], 8, 8)
    assert not fp[0, 5]  # missing in a
    assert not fp[5, 0]  # missing in b
    assert fp[5, 5]  # present in both
    assert fp.sum() == 5 * 6


def test_read_source_footprint_is_crisp(tmp_path):
    """Nearest resampling: no half-valid fringe when the grid is upsampled."""
    a = np.ones((8, 8), dtype=np.float32)
    a[:4] = np.nan
    pa = _write(tmp_path / "a.tif", a)
    fp = read_source_footprint(pa, [1], 16, 16)
    assert set(np.unique(fp).tolist()) == {False, True}
    assert fp[:8].sum() == 0
    assert fp[8:].all()


# -------------------------------------------------------------- stack_sources
def test_stack_sources_writes_footprint_as_nan(tmp_path):
    a = np.ones((8, 8), dtype=np.float32)
    a[:2] = np.nan
    pa = _write(tmp_path / "a.tif", a)
    out = tmp_path / "stack.tif"
    stack_sources([(pa, [1], ["a"])], 8, 8, TRANSFORM, CRS, str(out))

    with rasterio.open(out) as src:
        assert np.isnan(src.nodata)
        data = src.read()
    assert np.isnan(data[:, :2]).all()
    assert not np.isnan(data[:, 2:]).any()


# ----------------------------------------------------------------- tile_raster
def test_tile_raster_pads_with_requested_fill(tmp_path):
    p = _write(tmp_path / "small.tif", np.ones((8, 8), dtype=np.float32))
    with rasterio.open(p) as src:
        padded = tile_raster(src, 0, 0, 16, fill=np.nan)
    assert np.isnan(padded[:, 8:]).all()
    assert not np.isnan(padded[:, :8, :8]).any()


def test_tile_raster_default_fill_is_zero(tmp_path):
    p = _write(tmp_path / "small.tif", np.ones((8, 8), dtype=np.float32))
    with rasterio.open(p) as src:
        padded = tile_raster(src, 0, 0, 16)
    assert (padded[:, 8:] == 0).all()


# ------------------------------------------------------- compute_channel_stats
def _patch_dirs(tmp_path, img, dsm):
    (tmp_path / "images").mkdir()
    (tmp_path / "dsm").mkdir()
    _write(tmp_path / "images" / "0_0.tif", img)
    _write(tmp_path / "dsm" / "0_0_dsm.tif", dsm)
    return tmp_path


def test_channel_stats_ignore_out_of_footprint_pixels(tmp_path):
    """Half the patch is outside the footprint; the mean must not be halved."""
    img = np.full((2, 8, 8), 0.6, dtype=np.float32)
    img[:, :4] = np.nan
    dsm = np.full((1, 8, 8), 0.2, dtype=np.float32)
    stats = compute_channel_stats(_patch_dirs(tmp_path, img, dsm), ["a", "b"])

    assert stats["mean"][0] == pytest.approx(0.6)
    assert stats["mean"][1] == pytest.approx(0.6)
    assert stats["std"][0] == pytest.approx(0.0, abs=1e-5)


def test_channel_stats_unchanged_for_legacy_patches(tmp_path):
    """Patches with no NaN keep the previous all-zero-heuristic behaviour."""
    img = np.full((2, 8, 8), 0.6, dtype=np.float32)
    img[:, :4] = 0.0
    dsm = np.full((1, 8, 8), 0.2, dtype=np.float32)
    stats = compute_channel_stats(_patch_dirs(tmp_path, img, dsm), ["a", "b"])
    assert stats["mean"][0] == pytest.approx(0.6)


# ---------------------------------------------------------------- predict_scene
class _AlwaysCrown(torch.nn.Module):
    """Predicts crown everywhere — isolates the footprint from model behaviour."""

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), 10.0)


def test_predict_scene_excludes_out_of_footprint_pixels(tmp_path):
    from data.channels import ChannelSpec
    from scripts.predict import binarize, predict_scene

    scene = np.ones((2, 8, 8), dtype=np.float32)
    scene[:, :3] = np.nan  # top three rows were never recorded
    path = tmp_path / "scene.tif"
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", width=8, height=8, count=2,
        crs=CRS, transform=TRANSFORM, nodata=np.nan,
    ) as dst:
        dst.write(scene)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "green")

    spec = ChannelSpec(["red", "green"], ["red", "green"])
    prob, valid_mask, _ = predict_scene(
        _AlwaysCrown(), path, None, torch.device("cpu"), None, spec,
        tile_size=4, stride=4, batch_size=2,
    )

    assert not valid_mask[:3].any(), "predicted over ground outside the footprint"
    assert valid_mask[3:].all()

    binary = binarize(prob, 0.5, valid_mask)
    assert (binary[:3] == 255).all()
    assert (binary[3:] == 1).all()


def test_predict_scene_requires_data_in_every_band(tmp_path):
    """Regression: the old `not all bands zero` test was a union over bands.

    With two mosaics whose extents differ, a pixel recorded by only one of them
    passed as valid and the model was fed zero-filled bands for the other.
    """
    from data.channels import ChannelSpec
    from scripts.predict import predict_scene

    scene = np.ones((2, 8, 8), dtype=np.float32)
    scene[0, :3] = np.nan  # only the first source is missing up there
    path = tmp_path / "scene.tif"
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", width=8, height=8, count=2,
        crs=CRS, transform=TRANSFORM, nodata=np.nan,
    ) as dst:
        dst.write(scene)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "green")

    spec = ChannelSpec(["red", "green"], ["red", "green"])
    _, valid_mask, _ = predict_scene(
        _AlwaysCrown(), path, None, torch.device("cpu"), None, spec,
        tile_size=4, stride=4, batch_size=2,
    )
    assert not valid_mask[:3].any()
    assert valid_mask[3:].all()
