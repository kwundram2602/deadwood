# tests/test_predict_dtm.py
"""prepare_inputs must build the nDSM from an external DTM when one is given,
mirroring apply_dsm_mask.py's `method: dtm`, and fall back to the multi-scale
minimum filter otherwise.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from omegaconf import OmegaConf
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.predict import prepare_inputs  # noqa: E402

CRS = "EPSG:32736"
GSD = 0.05
H = W = 64


def _write(path, arr, *, descriptions=None, gsd=GSD):
    arr = np.atleast_3d(arr.T).T if arr.ndim == 2 else arr
    count, h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=count,
        dtype="float32", crs=CRS, transform=from_origin(0, 0, gsd, gsd),
    ) as dst:
        dst.write(arr.astype(np.float32))
        if descriptions:
            for i, d in enumerate(descriptions, start=1):
                dst.set_band_description(i, d)
    return Path(path)


@pytest.fixture
def scene(tmp_path):
    """RGB source, a DSM with a 10 m block on sloped ground, and the true DTM."""
    rgb = _write(tmp_path / "rgb.tif", np.full((3, H, W), 0.5, np.float32),
                 descriptions=["red", "green", "blue"])

    ground = np.tile(np.linspace(100.0, 140.0, W, dtype=np.float32), (H, 1))
    dsm_arr = ground.copy()
    dsm_arr[20:40, 20:40] += 10.0  # a 10 m tall object
    dsm = _write(tmp_path / "dsm.tif", dsm_arr)
    dtm = _write(tmp_path / "dtm.tif", ground)

    sources = [(str(rgb), [1, 2, 3], ["red", "green", "blue"])]
    cfg = OmegaConf.create({"target_gsd": GSD, "windows": [15], "max_ndsm_height": 16.0})
    return sources, dsm, dtm, cfg, tmp_path / "prep"


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1)


def test_dtm_ndsm_recovers_true_object_height(scene):
    """DSM - DTM on sloped ground gives a flat 0 background and the true height."""
    sources, dsm, dtm, cfg, prep = scene
    _, ndsm_path = prepare_inputs(sources, dsm, dtm, cfg, prep, need_ndsm=True)
    ndsm = _read(ndsm_path)

    # ceiling = min(p95, 16.0); background is exactly ground level -> 0
    assert ndsm[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert ndsm[5, 60] == pytest.approx(0.0, abs=1e-5)
    # the object saturates the [0,1] range relative to the 10 m ceiling
    assert ndsm[30, 30] == pytest.approx(1.0, abs=1e-5)


def test_minimum_filter_leaves_slope_residual(scene):
    """Contrast: without a DTM the sloped terrain bleeds into the nDSM."""
    sources, dsm, _dtm, cfg, prep = scene
    _, ndsm_path = prepare_inputs(sources, dsm, None, cfg, prep, need_ndsm=True)
    ndsm = _read(ndsm_path)

    # the minimum filter cannot remove the slope: flat ground is not 0
    assert ndsm[5, 60] > 0.01


def test_dtm_and_minimum_filter_use_distinct_caches(scene):
    """A DTM run must not be served from a minimum-filter cache entry."""
    sources, dsm, dtm, cfg, prep = scene
    _, lm_path = prepare_inputs(sources, dsm, None, cfg, prep, need_ndsm=True)
    _, dtm_path = prepare_inputs(sources, dsm, dtm, cfg, prep, need_ndsm=True)

    assert lm_path != dtm_path
    assert "_dtm" in dtm_path.name
    assert not np.allclose(_read(lm_path), _read(dtm_path))


def test_dtm_ignored_when_model_has_no_ndsm(scene):
    sources, dsm, dtm, cfg, prep = scene
    _, ndsm_path = prepare_inputs(sources, dsm, dtm, cfg, prep, need_ndsm=False)
    assert ndsm_path is None


def test_dtm_without_dsm_is_rejected(scene):
    """A DTM alone cannot produce an nDSM — the DSM is the minuend."""
    sources, _dsm, dtm, cfg, prep = scene
    with pytest.raises(ValueError, match="dsm"):
        prepare_inputs(sources, None, dtm, cfg, prep, need_ndsm=True)


def test_dtm_grid_mismatch_is_resampled_not_rejected(scene, tmp_path):
    """An external DTM on a coarser grid is reprojected onto the target grid."""
    sources, dsm, _dtm, cfg, prep = scene
    coarse = np.tile(np.linspace(100.0, 140.0, W // 4, dtype=np.float32), (H // 4, 1))
    coarse_dtm = _write(tmp_path / "dtm_coarse.tif", coarse, gsd=GSD * 4)

    _, ndsm_path = prepare_inputs(sources, dsm, coarse_dtm, cfg, prep, need_ndsm=True)
    with rasterio.open(ndsm_path) as src:
        assert (src.height, src.width) == (H, W)
