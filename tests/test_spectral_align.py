import json
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.align import (  # noqa: E402
    align_all,
    align_scene,
    is_aligned,
    parse_date,
)
from deadwood_spectral.grid import load_reference_grid  # noqa: E402

BANDS = ("R", "G", "B", "Green", "Red", "RedEdge", "NIR")


def _write_reference(path, left=1000.0, top=2000.0, res=1.0, size=4):
    profile = dict(
        driver="GTiff", dtype="float32", width=size, height=size, count=1,
        crs="EPSG:32736", transform=from_origin(left, top, res, res),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((1, size, size), dtype="float32"))
    return path


def _write_source(path, data_uint16, left=1000.0, top=2000.0, res=1.0):
    count, h, w = data_uint16.shape
    profile = dict(
        driver="GTiff", dtype="uint16", width=w, height=h, count=count,
        crs="EPSG:32736", transform=from_origin(left, top, res, res),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data_uint16)
        for i, name in enumerate(BANDS[:count], start=1):
            dst.set_band_description(i, name)
    return path


def _marker_source(size=4):
    """All zeros except a single bright pixel at row 0, col 0 of every band."""
    data = np.zeros((7, size, size), dtype="uint16")
    data[:, 0, 0] = 65535
    return data


def test_parse_date_from_filename():
    assert parse_date("20230824_Airport_Main_OM_coreg.tif") == "20230824"


def test_parse_date_without_leading_date_raises():
    with pytest.raises(ValueError, match="date"):
        parse_date("Airport_20230824_OM.tif")


def test_aligned_output_sits_on_reference_grid(tmp_path):
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif"))
    src = _write_source(tmp_path / "20230824_x.tif", _marker_source())
    out = align_scene(src, grid, tmp_path / "20230824_stack.tif")
    with rasterio.open(out) as dst:
        assert (dst.height, dst.width) == grid.shape
        assert dst.transform.almost_equals(grid.transform)
        assert dst.crs == grid.crs
        assert dst.count == 7
        assert list(dst.descriptions) == list(BANDS)


def test_offset_source_lands_in_the_correct_pixel(tmp_path):
    """The core regression: a source shifted by exactly one pixel of ground
    distance must land one pixel over, not be rescaled onto the same corner."""
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif", size=4))
    # Source origin is one 1 m pixel right and one down from the reference.
    src = _write_source(
        tmp_path / "20230824_x.tif", _marker_source(), left=1001.0, top=1999.0
    )
    out = align_scene(src, grid, tmp_path / "20230824_stack.tif")
    with rasterio.open(out) as dst:
        band = dst.read(1)
    assert band[1, 1] == pytest.approx(1.0, abs=1e-3)
    assert band[0, 0] == pytest.approx(0.0, abs=1e-3) or np.isnan(band[0, 0])


def test_values_are_scaled_to_unit_range(tmp_path):
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif"))
    data = np.full((7, 4, 4), 32768, dtype="uint16")
    src = _write_source(tmp_path / "20230824_x.tif", data)
    out = align_scene(src, grid, tmp_path / "20230824_stack.tif")
    with rasterio.open(out) as dst:
        assert dst.read(1).max() == pytest.approx(0.5, abs=1e-3)


def test_area_outside_the_source_becomes_nan(tmp_path):
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif", size=8))
    src = _write_source(tmp_path / "20230824_x.tif", _marker_source(size=2))
    out = align_scene(src, grid, tmp_path / "20230824_stack.tif")
    with rasterio.open(out) as dst:
        assert np.isnan(dst.read(1)[7, 7])


def test_is_aligned_true_for_matching_output(tmp_path):
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif"))
    src = _write_source(tmp_path / "20230824_x.tif", _marker_source())
    out = align_scene(src, grid, tmp_path / "20230824_stack.tif")
    assert is_aligned(out, grid) is True


def test_is_aligned_false_for_missing_output(tmp_path):
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif"))
    assert is_aligned(tmp_path / "nope.tif", grid) is False


def test_align_all_writes_manifest_and_skips_existing(tmp_path):
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif"))
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_source(src_dir / "20230824_a_OM_coreg.tif", _marker_source())
    _write_source(src_dir / "20230906_b_OM_coreg.tif", _marker_source())

    first = align_all(src_dir, grid, out_dir)
    assert len(first) == 2
    manifest = json.loads((out_dir / "channels.json").read_text())
    assert manifest["names"] == list(BANDS)

    mtimes = {p: p.stat().st_mtime_ns for p in first}
    second = align_all(src_dir, grid, out_dir)
    assert second == []  # nothing re-written
    assert all(p.stat().st_mtime_ns == mtimes[p] for p in first)


def test_align_all_rejects_wrong_crs(tmp_path):
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif"))
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    path = src_dir / "20230824_a_OM_coreg.tif"
    profile = dict(
        driver="GTiff", dtype="uint16", width=4, height=4, count=7,
        crs="EPSG:4326", transform=from_origin(30.0, -24.0, 0.001, 0.001),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(_marker_source())
    with pytest.raises(ValueError, match="CRS"):
        align_all(src_dir, grid, tmp_path / "out")


def test_aligned_output_is_band_interleaved(tmp_path):
    """Band interleaving is a correctness requirement, not a preference.

    align_scene writes one band at a time. In a pixel-interleaved tiled file a
    single tile holds all seven bands, so each band write forces a read-modify-
    recompress-append cycle over every tile. Once the scene outgrows GDAL's
    block cache the superseded bytes are never reclaimed: the real 6962 x 6459
    output measured 3.35 GB against 0.78 GB of bit-identical data.

    Asserted on the profile rather than on file size, because the bloat only
    appears above the cache threshold and a size-based test would pass on a
    small fixture regardless.
    """
    grid = load_reference_grid(_write_reference(tmp_path / "ref.tif"))
    src = _write_source(tmp_path / "20230824_x.tif", _marker_source())
    out = align_scene(src, grid, tmp_path / "20230824_stack.tif")
    with rasterio.open(out) as dst:
        assert dst.profile["interleave"] == "band"
