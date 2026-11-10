# tests/test_plot_error_distribution.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_error_distribution import (  # noqa: E402
    box_stats,
    find_tifs,
    main,
    read_valid,
    scene_label,
    se_stats,
)


def _write(path, data):
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", width=data.shape[1], height=data.shape[0],
        count=1, crs="EPSG:32636", nodata=float("nan"),
        transform=from_origin(500000, 5400000, 0.05, 0.05),
    ) as dst:
        dst.write(data.astype(np.float32)[np.newaxis])


@pytest.fixture
def scene_dir(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.random((40, 30)).astype(np.float32) ** 4
    a[:5] = 0.0
    a[-3:] = np.nan
    _write(tmp_path / "error_20260226.tif", a)
    _write(tmp_path / "error_20260313.tif", np.zeros((10, 10), np.float32))
    (tmp_path / "other.tif").write_bytes(b"")
    return tmp_path, a


def test_read_valid_drops_nan_and_stats_are_exact(scene_dir):
    tmp, a = scene_dir
    values = read_valid(tmp / "error_20260226.tif")
    finite = a[np.isfinite(a)]
    assert values.size == finite.size
    s = se_stats(values)
    assert s["n_px"] == finite.size
    assert s["mean"] == pytest.approx(float(finite.mean()), rel=1e-6)
    assert s["frac_zero"] == pytest.approx(float((finite == 0).mean()))
    assert s["max"] == pytest.approx(float(finite.max()))


def test_box_whiskers_stay_inside_data():
    v = np.array([0.0, 1, 2, 3, 4, 100])
    b = box_stats(v, "x")
    assert b["whislo"] == 0.0
    assert b["whishi"] == 4.0  # 100 lies beyond q3 + 1.5 IQR


def test_find_tifs_expands_folder(scene_dir):
    tmp, _ = scene_dir
    tifs = find_tifs([tmp])
    assert [scene_label(t) for t in tifs] == ["20260226", "20260313"]


def test_main_writes_outputs_and_refuses_overwrite(scene_dir):
    tmp, _ = scene_dir
    main([str(tmp), "--sample", "100"])  # error_20260313 is all zero: empty log panel
    out = tmp / "error_distribution.png"
    stats = json.loads(out.with_suffix(".json").read_text())["scenes"]
    assert out.stat().st_size > 0
    assert stats["20260313"]["frac_zero"] == 1.0
    with pytest.raises(FileExistsError):
        main([str(tmp)])
    main([str(tmp), "--overwrite"])
