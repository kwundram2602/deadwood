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
from deadwood_spectral.sampling import (  # noqa: E402
    CLASS_CODES,
    apply_quality_filter,
    binarize_crown_mask,
    block_ids,
    build_pools,
    draw_samples,
    load_crowns,
    rasterize_polygons,
)

# 100 x 100 px at 1 m, origin (1000, 2000) — 1 px == 1 m keeps the arithmetic
# in the tests readable.
GRID = ReferenceGrid(100, 100, from_origin(1000.0, 2000.0, 1.0, 1.0), rasterio.crs.CRS.from_epsg(32736))


def _gdf(tmp_path, rows):
    """rows: list of (tree_id, category, certainty, coverage, species, x, y, size)."""
    records = []
    for tree_id, cat, cert, cov, species, x, y, size in rows:
        records.append(
            {
                "tree_id": tree_id,
                "crown_category": cat,
                "certaintyLP": cert,
                "coverage": cov,
                "species": species,
                "geometry": box(x, y, x + size, y + size),
            }
        )
    path = tmp_path / "crowns.gpkg"
    gpd.GeoDataFrame(records, crs="EPSG:32736").to_file(path, driver="GPKG")
    return path


def _default_rows():
    return [
        ("4170", "soff", 100, "nc", "acanig", 1010.0, 1910.0, 10.0),
        ("4178", "soff", 0, "pc ", "sclbir", 1030.0, 1910.0, 10.0),
        ("4345", "son", 100, "nc", "diccin", 1060.0, 1910.0, 10.0),
        ("4999", "dead_fallen", 100, "nc", "terser", 1080.0, 1910.0, 10.0),
    ]


def _crown_mask(tmp_path, value=1.0, nodata_corner=False):
    path = tmp_path / "crown_pred.tif"
    data = np.zeros((1, 100, 100), dtype="float32")
    data[0, 80:100, 0:50] = value          # a crown blob, rows 80-99, cols 0-49
    data[0, 0:20, 60:100] = value          # a second blob
    if nodata_corner:
        data[0, 0:5, 0:5] = 255.0
    profile = dict(
        driver="GTiff", dtype="float32", width=100, height=100, count=1,
        crs="EPSG:32736", transform=GRID.transform, nodata=255.0,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_load_crowns_keeps_only_son_and_soff(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    assert set(gdf["crown_category"]) == {"son", "soff"}
    assert len(gdf) == 3


def test_load_crowns_normalises_coverage_whitespace_and_case(tmp_path):
    rows = _default_rows()
    rows[1] = ("4178", "soff", 0, "NC ", "sclbir", 1030.0, 1910.0, 10.0)
    gdf = load_crowns([_gdf(tmp_path, rows)], GRID)
    assert gdf.loc[gdf["tree_id"] == "4178", "coverage"].item() == "nc"


def test_load_crowns_assigns_unique_poly_idx(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    assert sorted(gdf["poly_idx"]) == [1, 2, 3]


def test_quality_filter_flags_but_does_not_drop(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    assert len(gdf) == 3
    assert gdf.loc[gdf["tree_id"] == "4170", "quality_ok"].item()
    assert not gdf.loc[gdf["tree_id"] == "4178", "quality_ok"].item()


def test_rasterize_polygons_burns_poly_idx(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    soff = gdf[gdf["crown_category"] == "soff"]
    raster = rasterize_polygons(soff, GRID, values=soff["poly_idx"])
    assert raster.shape == GRID.shape
    assert set(np.unique(raster)) == {0, 1, 2}


def test_negative_buffer_shrinks_the_burned_area(tmp_path):
    gdf = load_crowns([_gdf(tmp_path, _default_rows())], GRID)
    soff = gdf[gdf["crown_category"] == "soff"]
    full = rasterize_polygons(soff, GRID).sum()
    eroded = rasterize_polygons(soff, GRID, buffer_m=-2.0).sum()
    assert 0 < eroded < full


def test_binarize_crown_mask_splits_crown_and_validity(tmp_path):
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path, nodata_corner=True), GRID)
    assert crown.dtype == bool and valid.dtype == bool
    assert crown[90, 10]
    assert not crown[50, 50]
    assert not valid[2, 2]      # the 255 corner
    assert valid[50, 50]


def test_pools_are_disjoint_and_nonempty(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    pools = build_pools(crown, valid, gdf, GRID)
    assert set(pools) == set(CLASS_CODES)
    for name, pool in pools.items():
        assert pool.any(), f"{name} pool is empty"
    assert not (pools["deadwood"] & pools["living"]).any()
    assert not (pools["deadwood"] & pools["background"]).any()
    assert not (pools["living"] & pools["background"]).any()


def test_living_pool_excludes_buffered_soff(tmp_path):
    # A soff polygon placed inside the crown blob must be removed from `living`
    # together with its exclusion buffer.
    rows = _default_rows() + [("5000", "soff", 100, "nc", "acanig", 1010.0, 1905.0, 6.0)]
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, rows)], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    pools = build_pools(crown, valid, gdf, GRID, exclude_buffer_m=3.0)
    soff_area = rasterize_polygons(gdf[gdf["crown_category"] == "soff"], GRID).astype(bool)
    assert not (pools["living"] & soff_area).any()


def test_background_excludes_all_crown_polygons(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    pools = build_pools(crown, valid, gdf, GRID)
    all_polys = rasterize_polygons(gdf, GRID).astype(bool)
    assert not (pools["background"] & all_polys).any()


def test_pools_never_include_invalid_pixels(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path, nodata_corner=True), GRID)
    pools = build_pools(crown, valid, gdf, GRID)
    for pool in pools.values():
        assert not pool[~valid].any()


def test_block_ids_partition_the_grid():
    ids = block_ids(GRID, block_m=20.0)
    assert ids.shape == GRID.shape
    assert ids[0, 0] != ids[0, 99]
    assert ids[0, 0] == ids[5, 5]
    assert ids.min() == 0


def test_draw_samples_columns_and_class_codes(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    df = draw_samples(build_pools(crown, valid, gdf, GRID), gdf, GRID)
    expected = {
        "row", "col", "class_name", "class_code", "tree_id",
        "group_id", "species", "certaintyLP", "coverage", "quality_ok",
    }
    assert expected <= set(df.columns)
    assert set(df["class_name"]) == set(CLASS_CODES)
    assert df["class_code"].tolist() == [CLASS_CODES[n] for n in df["class_name"]]


def test_draw_samples_respects_negative_ratio(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    df = draw_samples(build_pools(crown, valid, gdf, GRID), gdf, GRID, negative_ratio=2.0)
    counts = df["class_name"].value_counts()
    assert counts["living"] <= 2 * counts["deadwood"]
    assert counts["background"] <= 2 * counts["deadwood"]


def test_draw_samples_assigns_tree_id_groups_to_deadwood(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    df = draw_samples(build_pools(crown, valid, gdf, GRID), gdf, GRID)
    dead = df[df["class_name"] == "deadwood"]
    assert set(dead["group_id"]) == {"tree:4170", "tree:4178"}
    assert dead["tree_id"].notna().all()


def test_draw_samples_assigns_block_groups_to_negatives(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    df = draw_samples(build_pools(crown, valid, gdf, GRID), gdf, GRID)
    neg = df[df["class_name"] != "deadwood"]
    assert neg["group_id"].str.startswith("block:").all()
    assert neg["tree_id"].isna().all()


def test_draw_samples_is_deterministic_for_a_seed(tmp_path):
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, _default_rows())], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    pools = build_pools(crown, valid, gdf, GRID)
    a = draw_samples(pools, gdf, GRID, seed=7)
    b = draw_samples(pools, gdf, GRID, seed=7)
    assert a.equals(b)


def test_empty_deadwood_pool_raises(tmp_path):
    rows = [("4345", "son", 100, "nc", "diccin", 1060.0, 1910.0, 10.0)]
    gdf = apply_quality_filter(load_crowns([_gdf(tmp_path, rows)], GRID))
    crown, valid = binarize_crown_mask(_crown_mask(tmp_path), GRID)
    with pytest.raises(ValueError, match="deadwood"):
        draw_samples(build_pools(crown, valid, gdf, GRID), gdf, GRID)
