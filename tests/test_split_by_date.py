# tests/test_split_by_date.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import assign_dates, split_patches  # noqa: E402

CRS = "EPSG:32636"
TRANSFORM = from_origin(500000, 5400000, 0.05, 0.05)


def _write(path, data):
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        crs=CRS,
        transform=TRANSFORM,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))


def _tiles(root, dates, n_per_date=3):
    """<root>/<date>/<date>_0_<i>{,_mask,_dsm}.tif, like the time-series tiling."""
    for d in dates:
        folder = root / d
        folder.mkdir(parents=True)
        for i in range(n_per_date):
            stem = f"{d}_0_{i}"
            _write(folder / f"{stem}.tif", np.full((2, 4, 4), 0.5))
            _write(folder / f"{stem}_mask.tif", np.zeros((1, 4, 4)))
            _write(folder / f"{stem}_dsm.tif", np.zeros((1, 4, 4)))
    (root / "channels.json").write_text(json.dumps({"names": ["a", "b"]}))


# --- assign_dates ----------------------------------------------------------


def test_explicit_mapping_is_used_as_given():
    got = assign_dates(
        ["20260313", "20260226"],
        explicit={"train": [20260226], "val": ["20260313"], "test": []},
    )
    assert got == {"20260226": "train", "20260313": "val"}


def test_explicit_mapping_rejects_double_assignment():
    with pytest.raises(ValueError, match="both"):
        assign_dates(["1", "2"], explicit={"train": ["1"], "val": ["1", "2"]})


def test_explicit_mapping_rejects_unknown_split():
    with pytest.raises(ValueError, match="unknown"):
        assign_dates(["1", "2"], explicit={"train": ["1"], "holdout": ["2"]})


def test_explicit_mapping_must_cover_selected_dates():
    with pytest.raises(ValueError, match="missing"):
        assign_dates(["1", "2", "3"], explicit={"train": ["1"], "val": ["2"]})


def test_explicit_mapping_needs_val():
    with pytest.raises(ValueError, match="'val'"):
        assign_dates(["1", "2"], explicit={"train": ["1", "2"], "val": []})


def test_shuffle_two_dates_gives_train_and_val():
    got = assign_dates(["20260226", "20260313"], seed=0)
    assert sorted(got.values()) == ["train", "val"]


def test_shuffle_counts_for_ten_dates():
    dates = [f"202601{i:02d}" for i in range(1, 11)]
    got = assign_dates(dates, train_ratio=0.7, val_ratio=0.2, seed=1)
    counts = {s: list(got.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 7, "val": 2, "test": 1}


def test_shuffle_is_deterministic_per_seed_and_order_free():
    dates = [f"202601{i:02d}" for i in range(1, 9)]
    a = assign_dates(dates, seed=3)
    b = assign_dates(list(reversed(dates)), seed=3)
    assert a == b
    assert any(assign_dates(dates, seed=s) != a for s in range(4, 20))


def test_shuffle_needs_two_dates():
    with pytest.raises(ValueError, match=">= 2"):
        assign_dates(["20260313"])


# --- split_patches ---------------------------------------------------------


def test_by_date_keeps_each_date_in_one_split(tmp_path):
    tiles = tmp_path / "tiles"
    _tiles(tiles, ["20260226", "20260313", "20260126"])
    date_splits = {"20260226": "train", "20260313": "val", "20260126": "test"}
    out = tmp_path / "split"
    written = split_patches(tiles, out, strategy="by_date", date_splits=date_splits)

    for split in ("train", "val", "test"):
        dates = {p.name[:8] for p in (out / split / "images").glob("*.tif")}
        assert len(dates) == 1
        assert date_splits[dates.pop()] == split
        assert len(written[split]) == 3
    assert (out / "channels.json").exists()

    record = json.loads((out / "split.json").read_text())
    assert record["strategy"] == "by_date"
    assert record["date_splits"] == date_splits
    assert record["tiles"]["val"] == ["20260313_0_0", "20260313_0_1", "20260313_0_2"]


def test_by_date_unassigned_date_raises(tmp_path):
    tiles = tmp_path / "tiles"
    _tiles(tiles, ["20260226", "20260313"])
    with pytest.raises(ValueError, match="20260313"):
        split_patches(
            tiles, tmp_path / "split", strategy="by_date", date_splits={"20260226": "train"}
        )


def test_random_strategy_also_writes_split_file(tmp_path):
    tiles = tmp_path / "tiles"
    _tiles(tiles, ["20260226"], n_per_date=10)
    out = tmp_path / "split"
    split_patches(tiles, out, train_ratio=0.6, val_ratio=0.2, seed=7)
    record = json.loads((out / "split.json").read_text())
    assert record["strategy"] == "random"
    assert record["date_splits"] is None
    assert sum(len(v) for v in record["tiles"].values()) == 10


def test_unknown_strategy_raises(tmp_path):
    with pytest.raises(ValueError, match="strategy"):
        split_patches(tmp_path, tmp_path / "split", strategy="spatial")
