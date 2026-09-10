import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.deadwood_patches import (
    keep_tile,
    scan_tiles,
    tile_kind,
    tile_stats,
)
from utils.nodata import MASK_OUTSIDE, MASK_UNLABELLED


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
