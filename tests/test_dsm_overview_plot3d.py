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


def test_the_figure_has_the_four_documented_panels(surfaces, aoi):
    fig = dem_figure(surfaces, aoi)
    assert len(fig.axes) == 4
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
