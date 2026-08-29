"""End-to-end: paths in, one PNG per crown and one CSV out."""

import os
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsm_overview.overview import run_dsm_overview  # noqa: E402

CRS = "EPSG:32736"
TRANSFORM = from_origin(1000.0, 2000.0, 0.25, 0.25)
SIZE = 256


def _raster(path, data, transform=TRANSFORM):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


@pytest.fixture
def project(tmp_path):
    """Flat ground at 100 m, a 3 m crown on 4157, nothing on 4170, DTM 0.8 m low."""
    rng = np.random.default_rng(0)
    dsm = 100.0 + rng.normal(0.0, 0.02, (SIZE, SIZE)).astype(np.float32)
    dsm[80:96, 80:96] += 3.0
    dtm = np.full((SIZE, SIZE), 99.2, dtype=np.float32)

    crowns = gpd.GeoDataFrame(
        [
            {
                "tree_id": "4157",
                "crown_category": "soff",
                "coverage": "nc",
                "geometry": box(1020.0, 1976.0, 1024.0, 1980.0),
            },
            {
                "tree_id": "4170",
                "crown_category": "soff",
                "coverage": "nc",
                "geometry": box(1035.0, 1976.0, 1039.0, 1980.0),
            },
            {
                "tree_id": "4345",
                "crown_category": "son",
                "coverage": "nc",
                "geometry": box(1005.0, 1976.0, 1009.0, 1980.0),
            },
        ],
        crs=CRS,
    )
    crowns.to_file(tmp_path / "crowns.gpkg", driver="GPKG")

    return {
        "reference": _raster(tmp_path / "ref.tif", np.zeros((SIZE, SIZE), dtype=np.float32)),
        "dsm": _raster(tmp_path / "dsm.tif", dsm),
        "dtm": _raster(tmp_path / "dtm.tif", dtm),
        "crowns": [tmp_path / "crowns.gpkg"],
        "out_dir": tmp_path / "out",
        "buffer_m": 8.0,
        "max_side": 60,
    }


def test_the_run_writes_every_declared_output(project):
    outputs = run_dsm_overview(**project)
    for key, path in outputs.items():
        assert path.exists(), f"{key} missing"
        assert path.stat().st_size > 0, f"{key} empty"


def test_one_plot_per_soff_crown_and_none_for_the_son(project):
    outputs = run_dsm_overview(**project)
    assert set(k for k in outputs if k.startswith("plot_")) == {"plot_4157", "plot_4170"}


def test_the_stats_csv_has_one_row_per_crown(project):
    outputs = run_dsm_overview(**project)
    table = pd.read_csv(outputs["stats_csv"], dtype={"tree_id": str})
    assert list(table["tree_id"]) == ["4157", "4170"]


def test_the_alignment_removes_the_offset_the_raw_dtm_carries(project):
    outputs = run_dsm_overview(**project)
    table = pd.read_csv(outputs["stats_csv"], dtype={"tree_id": str}).set_index("tree_id")
    assert table.loc["4157", "offset_raw_m"] == pytest.approx(0.8, abs=0.1)
    assert abs(table.loc["4157", "offset_aligned_m"]) < 0.15


def test_the_dtm_free_control_separates_the_two_crowns(project):
    outputs = run_dsm_overview(**project)
    table = pd.read_csv(outputs["stats_csv"], dtype={"tree_id": str}).set_index("tree_id")
    assert table.loc["4157", "crown_above_ring_m"] == pytest.approx(3.0, abs=0.15)
    assert abs(table.loc["4170", "crown_above_ring_m"]) < 0.15


def test_a_tree_id_selection_restricts_the_run(project):
    outputs = run_dsm_overview(**project, tree_ids=["4170"])
    assert set(k for k in outputs if k.startswith("plot_")) == {"plot_4170"}


def test_an_unknown_tree_id_is_rejected(project):
    with pytest.raises(ValueError, match="9999"):
        run_dsm_overview(**project, tree_ids=["9999"])


def test_an_int_tree_id_from_unquoted_yaml_still_matches(project):
    """`tree_ids: [4170]` in YAML parses as int, but the crown table's
    tree_id column is str — the config must not have to be hand-quoted."""
    outputs = run_dsm_overview(**project, tree_ids=[4170])
    assert set(k for k in outputs if k.startswith("plot_")) == {"plot_4170"}
