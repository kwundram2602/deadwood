import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_and_process.deadwood_patches import SPLIT_NAMES, assign_splits

FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def _kinds(n_pos, n_neg, n_both=0):
    kinds = {f"p{i}": "pos" for i in range(n_pos)}
    kinds.update({f"n{i}": "neg" for i in range(n_neg)})
    kinds.update({f"b{i}": "both" for i in range(n_both)})
    return kinds


def test_fractions_must_sum_to_one():
    # The old config took explicit crown counts and raised on a mismatch. The
    # unit changed to tiles, so the guard moved to the fractions — but it stays
    # loud rather than silently renormalising a typo.
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_splits(_kinds(6, 6), {"train": 0.6, "val": 0.2, "test": 0.3})


def test_every_tile_lands_in_exactly_one_split():
    kinds = _kinds(10, 10, 5)
    splits = assign_splits(kinds, FRACTIONS, seed=0)
    assert set(splits) == set(kinds)
    assert set(splits.values()) <= set(SPLIT_NAMES)


def test_stratified_split_gives_each_split_its_share_of_each_kind():
    # 20 of each kind at 60/20/20 -> 12/4/4 within every kind, not by luck.
    splits = assign_splits(_kinds(20, 20, 20), FRACTIONS, seed=0, stratify=True)
    for kind in ("p", "n", "b"):
        counts = Counter(
            split for tile, split in splits.items() if tile.startswith(kind)
        )
        assert counts["train"] == 12
        assert counts["val"] == 4
        assert counts["test"] == 4


def test_unstratified_split_ignores_kind():
    # Kept as an option so the cost of stratifying can be measured, but it is
    # what lets one split come up short of a kind, which is why it is not the
    # default.
    splits = assign_splits(_kinds(20, 20), FRACTIONS, seed=0, stratify=False)
    assert Counter(splits.values())["train"] == 24


def test_same_seed_gives_the_same_assignment():
    kinds = _kinds(15, 15, 6)
    assert assign_splits(kinds, FRACTIONS, seed=7) == assign_splits(kinds, FRACTIONS, seed=7)


def test_different_seeds_generally_differ():
    kinds = _kinds(15, 15, 6)
    assert assign_splits(kinds, FRACTIONS, seed=0) != assign_splits(kinds, FRACTIONS, seed=1)


def test_assignment_does_not_depend_on_input_ordering():
    # dict iteration order must not leak into the split, or re-running after
    # adding one polygon would reshuffle unrelated tiles.
    kinds = _kinds(9, 9)
    reversed_kinds = dict(reversed(list(kinds.items())))
    assert assign_splits(kinds, FRACTIONS, seed=3) == assign_splits(
        reversed_kinds, FRACTIONS, seed=3
    )


def test_a_kind_too_small_to_divide_still_places_every_tile():
    splits = assign_splits(_kinds(1, 2), FRACTIONS, seed=0)
    assert len(splits) == 3
