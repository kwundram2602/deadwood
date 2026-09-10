import os
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.deadwood_patches import (
    keep_tile,
    scan_tiles,
    tile_footprints,
    tile_kind,
    tile_stats,
    write_tiles,
)
from utils.nodata import MASK_OUTSIDE, MASK_RASTER_NODATA, MASK_UNLABELLED


def _crop(labelled=0, negatives=0, outside=0, size=10):
    crop = np.full((size, size), MASK_UNLABELLED, dtype=np.float32)
    flat = crop.reshape(-1)
    flat[:labelled] = 1.0
    flat[labelled : labelled + negatives] = 0.0
    flat[size * size - outside :] = MASK_OUTSIDE
    return crop


def test_stats_count_each_sentinel_separately():
    stats = tile_stats(_crop(labelled=4, negatives=6, outside=10))
    assert stats["pos_px"] == 4
    assert stats["neg_px"] == 6
    assert stats["labelled_px"] == 10
    assert stats["outside_frac"] == 0.1


def test_unlabelled_pixels_are_not_counted_as_outside():
    # The two sentinels mean opposite things: MASK_OUTSIDE says the imagery is
    # absent, MASK_UNLABELLED says only the label is. Conflating them would
    # drop every tile, since 97-99% of even the best tile is unlabelled.
    stats = tile_stats(_crop(labelled=1))
    assert stats["outside_frac"] == 0.0
    assert stats["labelled_px"] == 1


def test_tile_kind_distinguishes_the_three_useful_cases():
    assert tile_kind(tile_stats(_crop(labelled=4, negatives=6))) == "both"
    assert tile_kind(tile_stats(_crop(labelled=4))) == "pos"
    assert tile_kind(tile_stats(_crop(negatives=6))) == "neg"
    assert tile_kind(tile_stats(_crop())) == "empty"


def test_background_only_tiles_survive_the_filter():
    # The regression guard for the bug this rework exists to fix. A tile holding
    # only negatives is the entire false-positive-suppression signal; a filter
    # that required positives would silently discard it.
    stats = tile_stats(_crop(negatives=60))
    assert keep_tile(stats, min_labelled_px=50, max_outside_frac=0.5)


def test_filter_drops_thin_labels_and_mostly_absent_imagery():
    assert not keep_tile(tile_stats(_crop(labelled=4)), 50, 0.5)
    assert not keep_tile(tile_stats(_crop(labelled=60, outside=60)), 50, 0.5)


def test_scan_covers_the_grid_with_padded_edge_tiles():
    # 25x25 at size 10 is a ragged 3x3 grid: the last row and column run past
    # the array and must be padded with MASK_OUTSIDE, not silently shrunk —
    # the model requires every patch to be exactly size x size.
    mask = np.full((25, 25), MASK_UNLABELLED, dtype=np.float32)
    mask[0, 0] = 1.0
    scan = scan_tiles(mask, size=10)
    assert len(scan) == 9
    assert sorted(scan)[0] == "0_0"
    assert scan["0_0"]["pos_px"] == 1
    assert scan["2_2"]["outside_frac"] > 0.0
    assert scan["0_0"]["row"] == 0 and scan["2_2"]["col"] == 2


def test_scan_ids_are_zero_padded_so_they_sort_in_grid_order():
    mask = np.full((110, 110), MASK_UNLABELLED, dtype=np.float32)
    scan = scan_tiles(mask, size=10)
    assert sorted(scan)[:2] == ["00_00", "00_01"]


def _scene(tmp_path, size=20):
    """A tiny uint8 RGB scene on disk plus a matching mask in memory."""
    transform = from_origin(0, size, 1, 1)
    image = np.full((3, size, size), 7, dtype=np.uint8)
    path = tmp_path / "scene.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype="uint8", crs="EPSG:32736", transform=transform, nodata=0,
    ) as dst:
        dst.write(image)
    mask = np.full((size, size), MASK_UNLABELLED, dtype=np.float32)
    mask[:5, :5] = 1.0
    mask[15:, 15:] = 0.0
    return path, mask, transform


def test_write_tiles_writes_a_patch_pair_per_kept_tile(tmp_path):
    path, mask, transform = _scene(tmp_path)
    scan = scan_tiles(mask, size=10)
    splits = {"0_0": "train", "1_1": "val"}
    counts = write_tiles(path, mask, transform, "EPSG:32736", scan, splits, tmp_path / "out", 10)

    assert counts == {"train": 1, "val": 1, "test": 0}
    assert (tmp_path / "out" / "train" / "images" / "0_0.tif").exists()
    assert (tmp_path / "out" / "train" / "masks" / "0_0_mask.tif").exists()
    # Tiles not in `splits` were dropped by the filters and must not be written.
    assert not (tmp_path / "out" / "train" / "images" / "0_1.tif").exists()


def test_written_mask_keeps_its_sentinels_and_georeference(tmp_path):
    path, mask, transform = _scene(tmp_path)
    scan = scan_tiles(mask, size=10)
    write_tiles(path, mask, transform, "EPSG:32736", scan, {"0_0": "train"}, tmp_path / "out", 10)

    with rasterio.open(tmp_path / "out" / "train" / "masks" / "0_0_mask.tif") as src:
        written = src.read(1)
        assert src.nodata == MASK_RASTER_NODATA
        assert src.transform == transform
    assert (written[:5, :5] == 1.0).all()
    assert (written[5:, 5:] == MASK_UNLABELLED).all()


def test_edge_tiles_are_padded_to_full_size(tmp_path):
    # A 25 px scene at size 10 leaves a 5 px overhang. The patch must still be
    # 10x10, padded with MASK_OUTSIDE on the mask side and 0 on the image side.
    path, mask, transform = _scene(tmp_path, size=25)
    scan = scan_tiles(mask, size=10)
    write_tiles(path, mask, transform, "EPSG:32736", scan, {"2_2": "test"}, tmp_path / "out", 10)

    with rasterio.open(tmp_path / "out" / "test" / "masks" / "2_2_mask.tif") as src:
        written = src.read(1)
    with rasterio.open(tmp_path / "out" / "test" / "images" / "2_2.tif") as src:
        image = src.read()
    assert written.shape == (10, 10)
    assert image.shape == (3, 10, 10)
    assert (written[5:, :] == MASK_OUTSIDE).all()
    assert (image[:, 5:, :] == 0).all()


def test_tile_footprints_labels_every_tile_including_dropped_ones(tmp_path):
    # The point of the layer is answering "why is there no patch here?" in QGIS,
    # so dropped tiles must appear too, with the numbers that dropped them.
    _, mask, transform = _scene(tmp_path)
    scan = scan_tiles(mask, size=10)
    gdf = tile_footprints(scan, {"0_0": "train"}, transform, "EPSG:32736", 10)

    assert len(gdf) == len(scan)
    assert set(gdf.columns) == {
        "tile_id", "split", "kind", "labelled_px", "outside_frac", "geometry"
    }
    row = gdf.set_index("tile_id").loc["0_0"]
    assert row["split"] == "train"
    assert row["kind"] == "pos"
    assert row["labelled_px"] == 25
    assert gdf.set_index("tile_id").loc["0_1"]["split"] == "dropped"


def test_tile_footprint_geometry_matches_the_tile_window(tmp_path):
    _, mask, transform = _scene(tmp_path)
    scan = scan_tiles(mask, size=10)
    gdf = tile_footprints(scan, {}, transform, "EPSG:32736", 10)

    assert gdf.crs.to_string() == "EPSG:32736"
    bounds = gdf.set_index("tile_id").geometry.bounds
    # Scene origin is (0, 20) with 1 m pixels, so tile 0_0 spans x 0..10, y 10..20.
    assert tuple(bounds.loc["0_0"]) == (0.0, 10.0, 10.0, 20.0)
    # 0_0 alone cannot catch a row/col transpose — its row and col are both 0, so
    # swapped offsets land on the same square. These two are mirror images, so a
    # transpose swaps them and both assertions fail.
    assert tuple(bounds.loc["0_1"]) == (10.0, 10.0, 20.0, 20.0)
    assert tuple(bounds.loc["1_0"]) == (0.0, 0.0, 10.0, 10.0)
