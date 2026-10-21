# tests/test_rasterize_unlabelled.py
import sys
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_and_process.rasterize_crowns import main as rasterize_main  # noqa: E402
from explore_and_process.rasterize_crowns import unlabelled_mask  # noqa: E402
from utils.nodata import MASK_OUTSIDE, MASK_UNLABELLED  # noqa: E402


def test_unlabelled_mask_values():
    footprint = np.zeros((4, 4), bool)
    footprint[:2] = True
    mask = unlabelled_mask(footprint)
    assert mask.dtype == np.float32
    assert (mask[:2] == MASK_UNLABELLED).all()
    assert (mask[2:] == MASK_OUTSIDE).all()


def test_main_without_polygons_writes_unlabelled_mask(tmp_path):
    om = tmp_path / "20260226_OM_coreg.tif"
    data = np.full((3, 8, 8), 30000, np.uint16)
    data[:, :, :2] = 0  # left two columns: outside the footprint
    profile = dict(
        driver="GTiff", dtype="uint16", width=8, height=8, count=3,
        crs="EPSG:32636", transform=from_origin(500000, 5400000, 0.05, 0.05), nodata=0,
    )
    with rasterio.open(om, "w", **profile) as dst:
        dst.write(data)

    cfg = OmegaConf.create(
        {
            "use_polygons": False,
            # never opened: a missing file proves the polygons are skipped
            "crowns": [str(tmp_path / "missing.gpkg")],
            "reference": str(om),
            "sources": [{"path": str(om), "bands": [1, 2, 3], "names": ["r", "g", "b"]}],
            "out_mask": str(tmp_path / "mask.tif"),
            "out_image_dir": str(tmp_path / "images"),
            "raster_dir": None,
            "target_gsd": 0.05,
            "sigma": 3.0,
            "nodata_threshold": 0.05,
        }
    )
    rasterize_main(cfg)

    with rasterio.open(tmp_path / "mask.tif") as src:
        mask = src.read(1)
    assert set(np.unique(mask)) == {MASK_OUTSIDE, MASK_UNLABELLED}
    assert (mask[:, :2] == MASK_OUTSIDE).all()
    assert (mask[:, 2:] == MASK_UNLABELLED).all()
    assert (tmp_path / "images" / "20260226_OM_coreg_stack.tif").exists()
