"""The three DTM stages the pipeline actually walks through.

`raw` is the DTM as it comes off disk, `plane` after the global levelling,
`aligned` after the local block refinement — the one apply_dsm_mask subtracts.
Seeing them next to each other is the only way to tell which step swallowed a
crown.
"""

import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402
from dsm_overview.surfaces import STAGES, build_surfaces, load_surfaces  # noqa: E402

CRS = "EPSG:32736"
TRANSFORM = from_origin(1000.0, 2000.0, 0.5, 0.5)
SIZE = 256


@pytest.fixture
def grid():
    return ReferenceGrid(SIZE, SIZE, TRANSFORM, rasterio.crs.CRS.from_string(CRS))


def _scene(offset=6.5):
    """Sloping bare ground, three canopy blocks, DTM sitting `offset` m low."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    ground = 100.0 + 0.01 * yy + 0.005 * xx
    dsm = ground + rng.normal(0.0, 0.02, (SIZE, SIZE)).astype(np.float32)
    for r, c in ((10, 10), (100, 150), (200, 60)):
        dsm[r : r + 40, c : c + 40] += 10.0
    dtm = (ground - offset).astype(np.float32)
    return dsm.astype(np.float32), dtm


def _raster(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def test_every_stage_is_present(grid):
    dsm, dtm = _scene()
    surfaces = build_surfaces(dsm, dtm, grid)
    assert tuple(surfaces.dtm) == STAGES
    for stage in STAGES:
        assert surfaces.dtm[stage].shape == (SIZE, SIZE)


def test_the_raw_stage_still_carries_the_full_offset(grid):
    dsm, dtm = _scene(offset=6.5)
    surfaces = build_surfaces(dsm, dtm, grid)
    ground = surfaces.ndsm("raw")[120:160, 0:40]
    assert np.median(ground) == pytest.approx(6.5, abs=0.1)


def test_the_aligned_stage_puts_bare_ground_at_zero(grid):
    dsm, dtm = _scene(offset=6.5)
    surfaces = build_surfaces(dsm, dtm, grid)
    ground = surfaces.ndsm("aligned")[120:160, 0:40]
    assert abs(np.median(ground)) < 0.1


def test_canopy_height_survives_every_stage(grid):
    dsm, dtm = _scene(offset=6.5)
    surfaces = build_surfaces(dsm, dtm, grid)
    assert np.median(surfaces.ndsm("aligned")[10:50, 10:50]) == pytest.approx(10.0, abs=0.2)


def test_the_alignment_info_is_kept_per_stage(grid):
    dsm, dtm = _scene(offset=6.5)
    surfaces = build_surfaces(dsm, dtm, grid)
    assert surfaces.info["plane"]["mean_shift"] == pytest.approx(6.5, abs=0.1)
    assert surfaces.info["plane"]["local_blocks"] == 0
    assert surfaces.info["aligned"]["local_blocks"] > 0


def test_an_unknown_stage_is_rejected(grid):
    dsm, dtm = _scene()
    with pytest.raises(KeyError, match="nonsense"):
        build_surfaces(dsm, dtm, grid).ndsm("nonsense")


def test_loading_resamples_both_rasters_onto_the_reference_grid(tmp_path, grid):
    """The DTM comes in at a coarser GSD, like the real one at 0.5 m."""
    dsm, dtm = _scene(offset=6.5)
    _raster(tmp_path / "dsm.tif", dsm)
    coarse = dtm[::2, ::2]
    with rasterio.open(
        tmp_path / "dtm.tif",
        "w",
        driver="GTiff",
        height=coarse.shape[0],
        width=coarse.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=from_origin(1000.0, 2000.0, 1.0, 1.0),
        nodata=np.nan,
    ) as dst:
        dst.write(coarse, 1)

    surfaces = load_surfaces(
        _raster(tmp_path / "ref.tif", dsm), tmp_path / "dsm.tif", tmp_path / "dtm.tif"
    )

    assert surfaces.dsm.shape == (SIZE, SIZE)
    assert surfaces.dtm["raw"].shape == (SIZE, SIZE)
    assert abs(np.median(surfaces.ndsm("aligned")[120:160, 0:40])) < 0.15
