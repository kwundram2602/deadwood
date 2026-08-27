# tests/test_resample_image.py
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_and_process.rasterize_crowns import (  # noqa: E402
    read_scaled_bands,
    stack_sources,
    validate_sources,
)

TRANSFORM = from_origin(500000, 5400000, 0.05, 0.05)
CRS = "EPSG:32636"


def _write_raster(path, n_bands, h=8, w=8, fill=1000.0):
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=w,
        height=h,
        count=n_bands,
        crs=CRS,
        transform=TRANSFORM,
    )
    data = np.stack([np.full((h, w), fill * (i + 1), np.float32) for i in range(n_bands)])
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_read_scaled_bands_selects_and_scales(tmp_path):
    src = _write_raster(tmp_path / "ms.tif", 4)
    out = read_scaled_bands(str(src), [2, 4], 8, 8)
    assert out.shape == (2, 8, 8)
    assert out.dtype == np.float32
    assert np.allclose(out[0], 2000.0 / 65535.0)
    assert np.allclose(out[1], 4000.0 / 65535.0)


def test_read_scaled_bands_clips_artifacts(tmp_path):
    src = _write_raster(tmp_path / "hot.tif", 1, fill=1e23)
    out = read_scaled_bands(str(src), [1], 8, 8)
    assert out.max() <= 1.0


def test_stack_sources_order_descriptions_manifestable(tmp_path):
    rgb = _write_raster(tmp_path / "rgb.tif", 3, fill=100.0)
    ms = _write_raster(tmp_path / "ms.tif", 4, fill=1000.0)
    out_path = tmp_path / "out_stack.tif"
    specs = [
        (str(rgb), [1, 2, 3], ["red", "green", "blue"]),
        (str(ms), [3, 4], ["rededge", "nir"]),
    ]
    names = stack_sources(specs, 8, 8, TRANSFORM, CRS, str(out_path))
    assert names == ["red", "green", "blue", "rededge", "nir"]
    with rasterio.open(out_path) as src:
        assert src.count == 5
        assert list(src.descriptions) == names
        data = src.read()
    # source order preserved: band 4 = ms band 3 (fill 3000), band 5 = ms band 4
    assert np.allclose(data[3], 3000.0 / 65535.0)
    assert np.allclose(data[4], 4000.0 / 65535.0)


def _src_entry(path, bands, names):
    from omegaconf import OmegaConf

    return OmegaConf.create({"path": str(path), "bands": bands, "names": names})


def test_validate_sources_ok():
    sources = [
        _src_entry("a.tif", [1, 2, 3], ["red", "green", "blue"]),
        _src_entry("b.tif", [1, 2], ["green_ms", "red_ms"]),
    ]
    assert validate_sources(sources) == ["red", "green", "blue", "green_ms", "red_ms"]


def test_validate_sources_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        validate_sources([_src_entry("a.tif", [1, 2], ["red"])])


def test_validate_sources_duplicate_names():
    sources = [
        _src_entry("a.tif", [1], ["red"]),
        _src_entry("b.tif", [2], ["red"]),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_sources(sources)


def test_validate_sources_ndsm_reserved():
    with pytest.raises(ValueError, match="reserved"):
        validate_sources([_src_entry("a.tif", [1], ["ndsm"])])


def test_validate_sources_raster_dir_single_source_only():
    sources = [
        _src_entry("a.tif", [1], ["red"]),
        _src_entry("b.tif", [1], ["nir"]),
    ]
    with pytest.raises(ValueError, match="raster_dir"):
        validate_sources(sources, raster_dir="some/dir")


def test_validate_sources_empty():
    with pytest.raises(ValueError, match="at least one"):
        validate_sources([])
