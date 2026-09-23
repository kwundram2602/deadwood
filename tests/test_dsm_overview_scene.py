"""The whole scene instead of one crown, from DTM stages already on disk.

`apply_dsm_mask` writes both co-registered DTM stages to
`process_out/dtm_coreg/<run_id>/`. Re-fitting them here would answer a
different question than the one production actually asked, so the scene run
reads them verbatim and only resamples the two rasters that are not on the
reference grid: the DSM and the raw DTM.
"""

import os
import sys

import matplotlib
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsm_overview.overview import run_scene_overview  # noqa: E402
from dsm_overview.plot3d import dem_figure  # noqa: E402
from dsm_overview.surfaces import load_surfaces_from_stages  # noqa: E402
from dsm_overview.window import scene_aoi  # noqa: E402

CRS = "EPSG:32736"
TRANSFORM = from_origin(1000.0, 2000.0, 0.5, 0.5)
SIZE = 128

# The raw DTM sits this far below the ground the DSM sees; the stages on disk
# close that gap only partly, so a re-fit would be plain to see.
RAW_OFFSET = 6.5
PLANE_OFFSET = 3.0


def _raster(path, data, transform=TRANSFORM):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return str(path)


def _ground():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return 100.0 + 0.01 * yy + 0.005 * xx


@pytest.fixture
def scene(tmp_path):
    """Paths for one scene: reference, DSM, raw DTM, and both stages."""
    ground = _ground()
    dsm = ground.copy()
    for r, c in ((10, 10), (60, 80)):
        dsm[r : r + 30, c : c + 30] += 10.0

    # The raw DTM comes in coarse, like the real 0.5 m one against a 0.05 m grid.
    coarse = (ground - RAW_OFFSET)[::2, ::2]
    return {
        "reference": _raster(tmp_path / "ref.tif", ground),
        "dsm": _raster(tmp_path / "dsm.tif", dsm),
        "dtm": _raster(tmp_path / "dtm.tif", coarse, from_origin(1000.0, 2000.0, 1.0, 1.0)),
        "dtm_plane": _raster(tmp_path / "plane.tif", ground - PLANE_OFFSET),
        "dtm_aligned": _raster(tmp_path / "aligned.tif", ground),
        "dsm_array": dsm,
    }


def test_the_scene_aoi_covers_the_whole_grid():
    from deadwood_spectral.grid import ReferenceGrid

    grid = ReferenceGrid(SIZE, SIZE, TRANSFORM, rasterio.crs.CRS.from_string(CRS))

    aoi = scene_aoi(grid)

    assert (aoi.window.col_off, aoi.window.row_off) == (0, 0)
    assert (aoi.window.width, aoi.window.height) == (SIZE, SIZE)


def test_the_stages_are_read_verbatim_rather_than_re_fitted(scene):
    """A re-fit would lift the plane stage onto the DSM and erase the 3 m gap."""
    surfaces = load_surfaces_from_stages(
        scene["reference"],
        scene["dsm"],
        scene["dtm"],
        scene["dtm_plane"],
        scene["dtm_aligned"],
    )

    assert np.nanmedian(surfaces.ndsm("plane")) == pytest.approx(PLANE_OFFSET, abs=0.1)
    assert abs(np.nanmedian(surfaces.ndsm("aligned"))) < 0.1


def test_no_alignment_info_is_invented_for_stages_read_from_disk(scene):
    surfaces = load_surfaces_from_stages(
        scene["reference"],
        scene["dsm"],
        scene["dtm"],
        scene["dtm_plane"],
        scene["dtm_aligned"],
    )

    assert surfaces.info["plane"] == {}
    assert surfaces.info["aligned"] == {}


def test_the_raw_stage_is_resampled_onto_the_reference_grid(scene):
    surfaces = load_surfaces_from_stages(
        scene["reference"],
        scene["dsm"],
        scene["dtm"],
        scene["dtm_plane"],
        scene["dtm_aligned"],
    )

    assert surfaces.dsm.shape == (SIZE, SIZE)
    assert surfaces.dtm["raw"].shape == (SIZE, SIZE)
    assert np.nanmedian(surfaces.ndsm("raw")) == pytest.approx(RAW_OFFSET, abs=0.2)


def test_a_stage_raster_off_the_reference_grid_is_rejected(scene, tmp_path):
    """A stale dtm_coreg run dir must fail loudly, not be silently resampled."""
    off_grid = _raster(tmp_path / "off.tif", _ground()[: SIZE // 2, : SIZE // 2])

    with pytest.raises(ValueError, match="aligned"):
        load_surfaces_from_stages(
            scene["reference"], scene["dsm"], scene["dtm"], scene["dtm_plane"], off_grid
        )


def test_the_figure_takes_an_explicit_label(scene):
    from deadwood_spectral.grid import load_reference_grid

    surfaces = load_surfaces_from_stages(
        scene["reference"],
        scene["dsm"],
        scene["dtm"],
        scene["dtm_plane"],
        scene["dtm_aligned"],
    )
    aoi = scene_aoi(load_reference_grid(scene["reference"]))

    fig = dem_figure(surfaces, aoi, max_side=32, label="whole scene")

    assert fig._suptitle.get_text().startswith("whole scene")
    plt.close(fig)


def test_the_scene_run_writes_one_figure_and_no_csv(scene, tmp_path):
    out_dir = tmp_path / "out"

    outputs = run_scene_overview(
        reference=scene["reference"],
        dsm=scene["dsm"],
        dtm=scene["dtm"],
        dtm_plane=scene["dtm_plane"],
        dtm_aligned=scene["dtm_aligned"],
        out_dir=out_dir,
        max_side=32,
    )

    assert list(outputs) == ["plot_scene"]
    assert outputs["plot_scene"].exists()
    assert sorted(p.name for p in out_dir.iterdir()) == ["dem_scene.png"]
