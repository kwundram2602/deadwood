# tests/test_tile_naming.py
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_and_process.tile_patches import main as tile_main  # noqa: E402
from utils.data import split_patches  # noqa: E402

CRS = "EPSG:32636"
TRANSFORM = from_origin(500000, 5400000, 0.05, 0.05)


def _write(path, data, nodata=None):
    profile = dict(
        driver="GTiff", dtype="float32", width=data.shape[2],
        height=data.shape[1], count=data.shape[0], crs=CRS,
        transform=TRANSFORM, nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def _make_inputs(root, n_bands=5, size=8):
    img = np.random.rand(n_bands, size, size).astype(np.float32) * 0.5 + 0.25
    mask = np.zeros((1, size, size), np.float32)
    dsm = np.random.rand(1, size, size).astype(np.float32)
    _write(root / "scene_stack.tif", img)
    _write(root / "mask.tif", mask, nodata=255.0)
    _write(root / "dsm.tif", dsm)
    (root / "channels.json").write_text(
        json.dumps({"names": ["red", "green", "blue", "rededge", "nir"]})
    )


def test_tile_naming_and_manifest_copy(tmp_path):
    _make_inputs(tmp_path)
    out = tmp_path / "patches"
    cfg = OmegaConf.create({
        "image": str(tmp_path / "scene_stack.tif"),
        "mask": str(tmp_path / "mask.tif"),
        "dsm": str(tmp_path / "dsm.tif"),
        "out": str(out),
        "size": 4,
        "nodata_thresh": 0.9,
        "img_nodata_thresh": 0.9,
    })
    tile_main(cfg)

    tifs = sorted(p.name for p in out.glob("*.tif"))
    # 2x2 tiles of an 8x8 scene → 4 triples, no _ms4 anywhere
    assert "0_0.tif" in tifs
    assert "0_0_mask.tif" in tifs
    assert "0_0_dsm.tif" in tifs
    assert not [n for n in tifs if "_ms4" in n]
    assert json.loads((out / "channels.json").read_text())["names"][0] == "red"
    with rasterio.open(out / "0_0.tif") as src:
        assert src.count == 5


def test_split_copies_manifest(tmp_path):
    _make_inputs(tmp_path)
    out = tmp_path / "patches"
    cfg = OmegaConf.create({
        "image": str(tmp_path / "scene_stack.tif"),
        "mask": str(tmp_path / "mask.tif"),
        "dsm": str(tmp_path / "dsm.tif"),
        "out": str(out),
        "size": 4,
        "nodata_thresh": 0.9,
        "img_nodata_thresh": 0.9,
    })
    tile_main(cfg)
    split_root = tmp_path / "split"
    split_patches(out, split_root, train_ratio=0.5, val_ratio=0.25, seed=1, mode="copy")
    assert (split_root / "channels.json").exists()
