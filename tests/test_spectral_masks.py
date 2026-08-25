import os
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.grid import ReferenceGrid  # noqa: E402
from deadwood_spectral.masks import (  # noqa: E402
    CLASS_NAMES,
    binarize_crown_mask,
    build_masks,
    erode_by_area,
    load_crowns,
    rasterize_polygons,
)

# 100 x 100 px at 1 m, origin (1000, 2000) — 1 px == 1 m keeps the arithmetic
# in the tests readable.
GRID = ReferenceGrid(
    100, 100, from_origin(1000.0, 2000.0, 1.0, 1.0), rasterio.crs.CRS.from_epsg(32736)
)


def _gdf(tmp_path, rows):
    """rows: list of (tree_id, category, coverage, x, y, size)."""
    records = [
        {
            "tree_id": tree_id,
            "crown_category": cat,
            "coverage": cov,
            "geometry": box(x, y, x + size, y + size),
        }
        for tree_id, cat, cov, x, y, size in rows
    ]
    path = tmp_path / "crowns.gpkg"
    gpd.GeoDataFrame(records, crs="EPSG:32736").to_file(path, driver="GPKG")
    return path


def _default_rows():
    return [
        ("4170", "soff", "nc", 1010.0, 1910.0, 10.0),
        ("4178", "soff", "pc ", 1030.0, 1910.0, 10.0),
        ("4345", "son", "nc", 1060.0, 1910.0, 10.0),
        ("4999", "dead_fallen", "nc", 1080.0, 1910.0, 10.0),
    ]


def _crown_prediction(tmp_path, value=1.0, nodata_corner=False):
    """A raster on GRID standing in for the binarized torch output.

    Crown in the top 90 rows only — the bottom strip has to stay non-crown, or
    there is no background left for the third class.
    """
    path = tmp_path / "crown_pred.tif"
    data = np.zeros(GRID.shape, dtype=np.float32)
    data[:90] = value
    if nodata_corner:
        data[:5, :5] = 255.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=GRID.height,
        width=GRID.width,
        count=1,
        dtype="float32",
        crs=GRID.crs,
        transform=GRID.transform,
        nodata=255.0,
    ) as dst:
        dst.write(data, 1)
    return path


def _built(tmp_path, rows=None, **kwargs):
    gdf = load_crowns([_gdf(tmp_path, rows or _default_rows())], GRID)
    crown, valid = binarize_crown_mask(_crown_prediction(tmp_path), GRID)
    return build_masks(crown, valid, gdf, GRID, **kwargs)


# ── load_crowns ────────────────────────────────────────────────────────────


def test_load_crowns_keeps_only_son_and_soff(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    assert set(gdf["crown_category"]) == {"son", "soff"}


def test_load_crowns_normalises_coverage_whitespace_and_case(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    assert set(gdf["coverage"]) == {"nc", "pc"}


def test_load_crowns_assigns_unique_one_based_poly_idx(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    assert sorted(gdf["poly_idx"]) == [1, 2, 3]


# ── rasterize_polygons ─────────────────────────────────────────────────────


def test_rasterize_polygons_burns_poly_idx(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    burned = rasterize_polygons(gdf, GRID, values=gdf["poly_idx"])
    assert set(np.unique(burned)) == {0, 1, 2, 3}


def test_negative_buffer_shrinks_the_burned_area(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    full = rasterize_polygons(gdf, GRID).sum()
    shrunk = rasterize_polygons(gdf, GRID, buffer_m=-2.0).sum()
    assert 0 < shrunk < full


# ── erode_by_area ──────────────────────────────────────────────────────────


def test_erode_by_area_erodes_polygons_above_the_threshold(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, [("a", "soff", "nc", 1010.0, 1910.0, 10.0)])], GRID)
    eroded = erode_by_area(gdf, erode_m=1.0, min_area_m2=1.0)
    assert eroded.area.iloc[0] == pytest.approx(64.0, abs=1e-6)


def test_erode_by_area_leaves_a_polygon_under_the_threshold_untouched(tmp_path):
    """Eroding 10 cm off a 0.02 m2 crown deletes the tree — tree 4157's bug."""
    rows = [("4157", "soff", "nc", 1010.0, 1910.0, 0.1)]
    gdf = load_crowns([_gdf(tmp_path, rows)], GRID)
    eroded = erode_by_area(gdf, erode_m=0.1, min_area_m2=1.0)
    assert not eroded.iloc[0].is_empty
    assert eroded.area.iloc[0] == pytest.approx(0.01, abs=1e-9)


def test_erode_by_area_on_an_empty_frame(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    eroded = erode_by_area(gdf.iloc[:0], erode_m=1.0, min_area_m2=1.0)
    assert len(eroded) == 0


# ── binarize_crown_mask ────────────────────────────────────────────────────


def test_binarize_crown_mask_splits_crown_and_validity(tmp_path):
    crown, valid = binarize_crown_mask(_crown_prediction(tmp_path, nodata_corner=True), GRID)
    assert not valid[:5, :5].any()
    assert not crown[:5, :5].any()
    assert crown[50, 50]


def test_binarize_crown_mask_rejects_a_foreign_grid(tmp_path):
    other = ReferenceGrid(50, 50, GRID.transform, GRID.crs)
    with pytest.raises(ValueError, match="shape"):
        binarize_crown_mask(_crown_prediction(tmp_path), other)


# ── build_masks ────────────────────────────────────────────────────────────


def test_masks_are_disjoint_and_nonempty(tmp_path):
    masks = _built(tmp_path)
    arrays = [getattr(masks, name) for name in CLASS_NAMES]
    for array in arrays:
        assert array.any()
    assert (arrays[0].astype(int) + arrays[1].astype(int) + arrays[2].astype(int)).max() == 1


def test_living_excludes_the_buffered_soff_polygons(tmp_path):
    masks = _built(tmp_path, exclude_buffer_m=2.0)
    # 1 m outside the 4170 crown (which spans x 1010..1020, y 1910..1920).
    row, col = GRID.transform.__invert__() * (1021.0, 1919.0)
    assert not masks.living[int(col), int(row)]


def test_background_excludes_every_crown_polygon(tmp_path):
    masks = _built(tmp_path, edge_buffer_m=0.0)
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    inside_any = rasterize_polygons(gdf, GRID).astype(bool)
    assert not (masks.background & inside_any).any()


def test_masks_never_include_invalid_pixels(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    crown, valid = binarize_crown_mask(_crown_prediction(tmp_path, nodata_corner=True), GRID)
    masks = build_masks(crown, valid, gdf, GRID)
    for name in CLASS_NAMES:
        assert not (getattr(masks, name) & ~valid).any()


def test_tree_idx_labels_deadwood_pixels_with_their_polygon(tmp_path):
    masks = _built(tmp_path)
    labels = set(np.unique(masks.tree_idx[masks.deadwood]).tolist())
    assert labels == {1, 2}
    assert (masks.tree_idx[~masks.deadwood] == 0).all()


def test_tree_ids_map_the_burned_index_back_to_the_field_id(tmp_path):
    masks = _built(tmp_path)
    assert set(masks.tree_ids.values()) == {"4170", "4178"}


def test_an_empty_deadwood_mask_raises(tmp_path):
    rows = [("4345", "son", "nc", 1060.0, 1910.0, 10.0)]
    with pytest.raises(ValueError, match="no deadwood"):
        _built(tmp_path, rows=rows)


def test_erosion_that_would_empty_a_crown_keeps_it_in_the_deadwood_mask(tmp_path):
    """The whole point of erode_min_area_m2: no soff tree may silently vanish."""
    rows = [
        ("4157", "soff", "nc", 1010.5, 1910.5, 2.0),
        ("4170", "soff", "nc", 1030.0, 1910.0, 10.0),
    ]
    masks = _built(tmp_path, rows=rows, erode_m=3.0, erode_min_area_m2=100.0)
    assert set(np.unique(masks.tree_idx[masks.deadwood]).tolist()) == {1, 2}
