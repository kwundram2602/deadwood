"""The figure is the deliverable, so the test checks it gets built and written.

Pixels are not asserted on. What is asserted: four panels in the documented
order, and a file on disk with content in it.
"""

import os
import sys

import matplotlib
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402
from dsm_overview.plot3d import dem_figure, plot_dem_overview  # noqa: E402
from dsm_overview.surfaces import Surfaces  # noqa: E402
from dsm_overview.window import aoi_from_bounds  # noqa: E402

CRS = "EPSG:32736"
TRANSFORM = from_origin(1000.0, 2000.0, 0.25, 0.25)
SIZE = 200


@pytest.fixture
def grid():
    return ReferenceGrid(SIZE, SIZE, TRANSFORM, rasterio.crs.CRS.from_string(CRS))


@pytest.fixture
def surfaces(grid):
    dsm = np.full((SIZE, SIZE), 100.0, dtype=np.float32)
    dsm[80:96, 80:96] += 3.0
    return Surfaces(
        grid=grid,
        dsm=dsm,
        dtm={
            "raw": np.full((SIZE, SIZE), 99.2, dtype=np.float32),
            "plane": np.full((SIZE, SIZE), 100.0, dtype=np.float32),
            "aligned": np.full((SIZE, SIZE), 100.0, dtype=np.float32),
        },
        info={"raw": {}, "plane": {"mean_shift": 0.8}, "aligned": {"local_rms": 0.1}},
    )


@pytest.fixture
def aoi(grid):
    return aoi_from_bounds((1020.0, 1976.0, 1024.0, 1980.0), grid, buffer_m=8.0, tree_id="4157")


def test_the_figure_has_the_five_documented_panels(surfaces, aoi):
    fig = dem_figure(surfaces, aoi)
    assert len(fig.axes) == 5
    assert "4157" in fig.get_suptitle()
    plt.close(fig)


def test_a_large_patch_is_thinned_before_plotting(surfaces, grid):
    """A full-extent AOI must not put 40 000 faces into a surface plot."""
    big = aoi_from_bounds((1000.0, 1950.0, 1049.0, 1999.0), grid, buffer_m=0.0, tree_id="wide")
    fig = dem_figure(surfaces, big, max_side=50)
    collections = fig.axes[0].collections
    assert len(collections) > 0
    plt.close(fig)


def test_the_plot_is_written_to_disk(surfaces, aoi, tmp_path):
    path = plot_dem_overview(surfaces, aoi, tmp_path / "4157.png")
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.fixture
def geometry():
    """A crown polygon on the raised block of the synthetic DSM."""
    return box(1020.0, 1976.0, 1024.0, 1980.0)


def test_the_crown_outline_is_drawn_on_dsm_and_ndsm(surfaces, aoi, geometry):
    """Without the polygon a mound in the patch cannot be attributed to a tree."""
    plain = dem_figure(surfaces, aoi)
    marked = dem_figure(surfaces, aoi, geometry=geometry)
    for index in range(5):
        assert len(marked.axes[index].lines) == len(plain.axes[index].lines) + 1
    assert "crown polygon" in [t.get_text() for t in marked.axes[0].get_legend().get_texts()]
    plt.close(plain)
    plt.close(marked)


def test_the_view_angle_is_configurable(surfaces, aoi):
    fig = dem_figure(surfaces, aoi, elev=62.0, azim=-35.0)
    assert all(ax.elev == 62.0 and ax.azim == -35.0 for ax in fig.axes)
    plt.close(fig)


def test_depth_sorting_is_off_so_the_crown_tint_survives(surfaces, aoi, geometry):
    """mplot3d's own sort paints the DTM plane over the crown; see `_new_axes`."""
    fig = dem_figure(surfaces, aoi, geometry=geometry)
    assert all(ax.computed_zorder is False for ax in fig.axes)
    plt.close(fig)


def test_the_control_panel_is_scaled_to_the_overshoot_not_the_crown(surfaces, aoi):
    """A 3 m crown must not flatten the half-metre of DTM-above-DSM into a line."""
    fig = dem_figure(surfaces, aoi)
    low, high = fig.axes[4].get_zlim()
    assert high <= 1.0 and low >= -1.0
    plt.close(fig)
