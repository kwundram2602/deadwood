import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.predict_deadwood_timeseries import find_scenes, run_timeseries, scene_config


class ConstantLogit(torch.nn.Module):
    """Stand-in for the real model: emits one fixed logit everywhere."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.logit)


def _write_scene(path: Path, size=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((3, size, size), 30000.0, dtype="float32")
    data[:, :10, :] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=3,
        dtype="float32",
        crs="EPSG:32736",
        transform=from_origin(0, size * 0.05, 0.05, 0.05),
        nodata=np.nan,
    ) as dst:
        dst.write(data)


def _cfg(out_dir: str):
    return OmegaConf.create(
        {
            "source": {"path": None, "bands": [1, 2, 3]},
            "target_gsd": 0.05,
            "scale": {"mode": "linear", "source_max": 65535},
            "tile_size": 64,
            "padding": 16,
            "batch_size": 2,
            "threshold": 0.5,
            "min_polygon_area": 0.0,
            "out_dir": out_dir,
            "save_input": False,
            "save_mask": True,
        }
    )


def _series(tmp_path: Path) -> Path:
    series = tmp_path / "series"
    for date in ("20260601", "20260313"):
        _write_scene(series / date / f"{date}_Site_OM_RGB_merged.tif")
    return series


def test_find_scenes_is_recursive_sorted_and_ignores_own_outputs(tmp_path):
    series = _series(tmp_path)
    (series / "20260313" / "20260313_Site_OM_MS_merged.tif").touch()
    (series / "20260313" / "x_OM_RGB_merged_rgb8.tif").touch()

    scenes = find_scenes(series)

    assert [p.parent.name for p in scenes] == ["20260313", "20260601"]
    assert find_scenes(series, "**/*_rgb8.tif") == []


def test_find_scenes_max_items_keeps_the_most_recent(tmp_path):
    for date in ("20240101", "20230101", "20250101"):
        (tmp_path / f"{date}_OM_coreg.tif").touch()

    scenes = find_scenes(tmp_path, "*_OM_coreg.tif", max_items=2)

    assert [p.name[:8] for p in scenes] == ["20240101", "20250101"]
    assert len(find_scenes(tmp_path, "*_OM_coreg.tif", max_items=10)) == 3


def test_scene_config_writes_into_a_per_date_folder(tmp_path):
    scene = tmp_path / "20260313" / "a.tif"
    cfg = scene_config(_cfg("out/ts"), scene)

    assert cfg.source.path == str(scene)
    assert cfg.out_dir == "out/ts/20260313"


def _masks(tmp_path: Path, out: str) -> dict[str, np.ndarray]:
    masks = {}
    for path in sorted((tmp_path / out).glob("*/*_deadwood_mask.tif")):
        with rasterio.open(path) as src:
            masks[path.parent.name] = src.read(1)
    return masks


def test_run_writes_every_date_and_prefetch_matches_sequential(tmp_path):
    scenes = find_scenes(_series(tmp_path))
    model, device = ConstantLogit(5.0), torch.device("cpu")

    rows_seq = run_timeseries(_cfg("seq"), tmp_path, scenes, model, device, prefetch=False)
    rows_pre = run_timeseries(_cfg("pre"), tmp_path, scenes, model, device, prefetch=True)

    assert [r["status"] for r in rows_seq + rows_pre] == ["ok"] * 4
    assert all(r["polygons"] == 1 for r in rows_seq + rows_pre)
    seq, pre = _masks(tmp_path, "seq"), _masks(tmp_path, "pre")
    assert list(seq) == ["20260313", "20260601"]
    for date in seq:
        np.testing.assert_array_equal(seq[date], pre[date])
    # save_input is off, so the intermediate uint8 inputs are gone.
    assert not list((tmp_path / "pre").glob("*/*_rgb8.tif"))


def test_rerun_skips_done_scenes_unless_overwrite(tmp_path):
    scenes = find_scenes(_series(tmp_path))
    model, device = ConstantLogit(5.0), torch.device("cpu")
    run_timeseries(_cfg("out"), tmp_path, scenes, model, device)

    rows = run_timeseries(_cfg("out"), tmp_path, scenes, model, device)
    assert [r["status"] for r in rows] == ["skipped", "skipped"]

    rows = run_timeseries(_cfg("out"), tmp_path, scenes, model, device, overwrite=True)
    assert [r["status"] for r in rows] == ["ok", "ok"]


def test_a_failing_scene_does_not_stop_the_series(tmp_path):
    series = _series(tmp_path)
    broken = series / "20260101" / "20260101_Site_OM_RGB_merged.tif"
    broken.parent.mkdir()
    broken.write_bytes(b"not a tiff")
    scenes = find_scenes(series)

    rows = run_timeseries(_cfg("out"), tmp_path, scenes, ConstantLogit(5.0), torch.device("cpu"))

    assert [r["status"] for r in rows] == ["failed", "ok", "ok"]
