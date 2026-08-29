"""Numbers that separate the two candidate explanations.

If bare ground next to a crown sits at zero after the alignment but the crown
still reads flat, the co-registration did its job and the DSM never resolved
the tree. If the ground itself is off by half a metre, the alignment is the
problem. `crown_above_ring_m` is the DTM-free control: how far the crown stands
above its own surroundings in the DSM alone.
"""

import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402
from dsm_overview.stats import aoi_stats, crown_ring_masks, stats_table  # noqa: E402
from dsm_overview.surfaces import Surfaces  # noqa: E402
from dsm_overview.window import aoi_from_bounds  # noqa: E402

CRS = "EPSG:32736"
# 200 x 200 px at 0.25 m: 50 m across, enough for a crown plus a ring.
TRANSFORM = from_origin(1000.0, 2000.0, 0.25, 0.25)
SIZE = 200


@pytest.fixture
def grid():
    return ReferenceGrid(SIZE, SIZE, TRANSFORM, rasterio.crs.CRS.from_string(CRS))


def _surfaces(grid, crown_height=3.0, ground_offset=0.0):
    """Flat ground at 100 m with one 4 m x 4 m crown; DTM off by `ground_offset`."""
    dsm = np.full((SIZE, SIZE), 100.0, dtype=np.float32)
    dsm[80:96, 80:96] += crown_height  # x 1020..1024, y 1980..1976
    dtm = np.full((SIZE, SIZE), 100.0 - ground_offset, dtype=np.float32)
    return Surfaces(
        grid=grid,
        dsm=dsm,
        dtm={"raw": dtm, "plane": dtm, "aligned": dtm},
        info={"raw": {}, "plane": {}, "aligned": {}},
    )


CROWN = box(1020.0, 1976.0, 1024.0, 1980.0)


def test_the_masks_are_disjoint_and_leave_the_gap_empty(grid):
    aoi = aoi_from_bounds(CROWN.bounds, grid, buffer_m=8.0, tree_id="4157")
    crown, ring = crown_ring_masks(CROWN, aoi, grid, ring_gap_m=2.0, ring_width_m=5.0)

    assert crown.shape == ring.shape
    assert not (crown & ring).any()
    assert crown.sum() > 0
    assert ring.sum() > crown.sum()


def test_a_standing_crown_is_reported_at_its_height(grid):
    aoi = aoi_from_bounds(CROWN.bounds, grid, buffer_m=8.0, tree_id="4157")
    stats = aoi_stats(_surfaces(grid, crown_height=3.0), aoi, CROWN)

    assert stats["tree_id"] == "4157"
    assert stats["crown_ndsm_m"] == pytest.approx(3.0, abs=0.05)
    assert stats["crown_above_ring_m"] == pytest.approx(3.0, abs=0.05)
    assert stats["offset_aligned_m"] == pytest.approx(0.0, abs=0.05)


def test_a_dtm_sitting_low_shows_up_as_a_ground_offset(grid):
    aoi = aoi_from_bounds(CROWN.bounds, grid, buffer_m=8.0, tree_id="4157")
    stats = aoi_stats(_surfaces(grid, crown_height=3.0, ground_offset=0.8), aoi, CROWN)

    assert stats["offset_raw_m"] == pytest.approx(0.8, abs=0.05)
    # The nDSM inherits the offset one-to-one — this is the failure mode the
    # whole check exists for.
    assert stats["crown_ndsm_m"] == pytest.approx(3.8, abs=0.05)
    # ...but the DTM-free control is untouched by it.
    assert stats["crown_above_ring_m"] == pytest.approx(3.0, abs=0.05)


def test_a_crown_the_dsm_never_resolved_reads_flat_in_both(grid):
    aoi = aoi_from_bounds(CROWN.bounds, grid, buffer_m=8.0, tree_id="4157")
    stats = aoi_stats(_surfaces(grid, crown_height=0.0), aoi, CROWN)

    assert stats["crown_ndsm_m"] == pytest.approx(0.0, abs=0.05)
    assert stats["crown_above_ring_m"] == pytest.approx(0.0, abs=0.05)


def _sloped_surfaces(grid, crown_height=3.0, slope=0.08, res=0.25):
    """Crown on an ~8% slope, matching the real survey's 6-9% terrain.

    A flat-ground fixture cannot catch a bias that only shows up as relief
    across the AOI, which is exactly what happened here: every fixture in
    this file used to be flat, so `crown_above_ring_m`'s bias against sloping
    terrain shipped unnoticed.
    """
    rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    ground = 100.0 + slope * rows * res
    dsm = ground.copy()
    dsm[80:96, 80:96] += crown_height
    return Surfaces(
        grid=grid,
        dsm=dsm.astype(np.float32),
        dtm={
            "raw": ground.astype(np.float32),
            "plane": ground.astype(np.float32),
            "aligned": ground.astype(np.float32),
        },
        info={"raw": {}, "plane": {}, "aligned": {}},
    )


def test_a_standing_crown_on_sloping_ground_is_still_reported_at_its_height(grid):
    # buffer_m=15.0 matches the production config default — the AOI size the
    # bias was actually measured at.
    aoi = aoi_from_bounds(CROWN.bounds, grid, buffer_m=15.0, tree_id="4157")
    stats = aoi_stats(_sloped_surfaces(grid, crown_height=3.0), aoi, CROWN)

    assert stats["crown_above_ring_m"] == pytest.approx(3.0, abs=0.1)


def test_the_pixel_counts_are_reported(grid):
    aoi = aoi_from_bounds(CROWN.bounds, grid, buffer_m=8.0, tree_id="4157")
    stats = aoi_stats(_surfaces(grid), aoi, CROWN)
    assert stats["n_crown_px"] > 0
    assert stats["n_ring_px"] > 0


def test_the_table_keeps_the_column_order(grid):
    aoi = aoi_from_bounds(CROWN.bounds, grid, buffer_m=8.0, tree_id="4157")
    table = stats_table([aoi_stats(_surfaces(grid), aoi, CROWN)])
    assert list(table.columns)[:4] == [
        "tree_id",
        "n_crown_px",
        "n_ring_px",
        "crown_ndsm_m",
    ]
