"""End-to-end: config-shaped arguments in, CSVs and PNGs out.

The unit tests cover each stage; this one covers the wiring between them, which
is where a rebuilt pipeline actually breaks.
"""

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
from deadwood_spectral.indices import BAND_NAMES  # noqa: E402
from deadwood_spectral.overview import MEASURES, run_overview  # noqa: E402

CRS = "EPSG:32736"
TRANSFORM = from_origin(1000.0, 2000.0, 1.0, 1.0)
SHAPE = (40, 40)
# Wet, dry, wet — enough for both seasons to appear in the signature.
DATES = ["20240116", "20240613", "20241203"]


def _reference(path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SHAPE[0],
        width=SHAPE[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
    ) as dst:
        dst.write(np.zeros(SHAPE, dtype=np.float32), 1)
    return path


def _crown_prediction(path):
    data = np.zeros(SHAPE, dtype=np.float32)
    data[:25] = 1.0  # crown in the top rows, bare ground below
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SHAPE[0],
        width=SHAPE[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
        nodata=255.0,
    ) as dst:
        dst.write(data, 1)
    return path


def _crowns(path):
    rows = [
        # y 1995..1990 is rows 5..10 — inside the crown region.
        ("4157", "soff", "nc", box(1005.0, 1990.0, 1013.0, 1995.0)),
        ("4170", "soff", "nc", box(1020.0, 1990.0, 1028.0, 1995.0)),
        ("4345", "son", "nc", box(1005.0, 1980.0, 1013.0, 1985.0)),
    ]
    gpd.GeoDataFrame(
        [
            {"tree_id": t, "crown_category": c, "coverage": cov, "geometry": g}
            for t, c, cov, g in rows
        ],
        crs=CRS,
    ).to_file(path, driver="GPKG")
    return path


def _stack(path, nir):
    values = {"R": 0.1, "G": 0.2, "B": 0.15, "Green": 0.2, "Red": 0.1, "RedEdge": 0.3, "NIR": nir}
    data = np.stack([np.full(SHAPE, values[b], dtype=np.float32) for b in BAND_NAMES])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SHAPE[0],
        width=SHAPE[1],
        count=len(BAND_NAMES),
        dtype="float32",
        crs=CRS,
        transform=TRANSFORM,
        nodata=np.nan,
    ) as dst:
        dst.write(data)
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(i, name)
    return path


@pytest.fixture
def project(tmp_path):
    stack_dir = tmp_path / "timeseries"
    stack_dir.mkdir()
    for nir, date in zip([0.5, 0.2, 0.6], DATES):
        _stack(stack_dir / f"{date}_stack.tif", nir)
    return {
        "reference": _reference(tmp_path / "crown_mask.tif"),
        "stack_dir": stack_dir,
        "crowns": [_crowns(tmp_path / "crowns.gpkg")],
        "crown_prediction": _crown_prediction(tmp_path / "crown_pred.tif"),
        "out_dir": tmp_path / "out",
        "erode_m": 0.5,
        "erode_min_area_m2": 1.0,
        "max_pixels_per_class": 50,
    }


def test_run_writes_every_declared_output(project):
    outputs = run_overview(**project)
    for key, path in outputs.items():
        assert path.exists(), f"{key} missing"
        assert path.stat().st_size > 0, f"{key} empty"


def test_run_writes_one_timeseries_plot_per_measure(project):
    outputs = run_overview(**project)
    assert {k for k in outputs if k.startswith("plot_ts_")} == {f"plot_ts_{m}" for m in MEASURES}


def test_class_csv_covers_all_three_classes_at_every_date(project):
    outputs = run_overview(**project)
    table = pd.read_csv(outputs["class_csv"], dtype={"date": str})
    assert set(table["class"]) == {"deadwood", "living", "background"}
    assert set(table["date"]) == set(DATES)


def test_tree_csv_carries_one_curve_per_soff_tree(project):
    outputs = run_overview(**project)
    table = pd.read_csv(outputs["tree_csv"], dtype={"tree_id": str})
    assert set(table["tree_id"]) == {"4157", "4170"}


def test_tree_csv_never_contains_the_son_polygon(project):
    outputs = run_overview(**project)
    table = pd.read_csv(outputs["tree_csv"], dtype={"tree_id": str})
    assert "4345" not in set(table["tree_id"])


def test_signature_csv_splits_the_two_seasons(project):
    outputs = run_overview(**project)
    table = pd.read_csv(outputs["signature_csv"])
    assert set(table["season"]) == {"dry", "wet"}
    assert set(table["band"]) == set(BAND_NAMES)


def test_ndvi_tracks_the_nir_the_scenes_were_written_with(project):
    """The dry date was given the lowest NIR, so its NDVI must be the lowest."""
    outputs = run_overview(**project)
    table = pd.read_csv(outputs["class_csv"], dtype={"date": str})
    ndvi = table[(table["measure"] == "ndvi") & (table["class"] == "deadwood")]
    assert ndvi.set_index("date").loc["20240613", "median"] == ndvi["median"].min()


def test_run_is_reproducible_for_a_seed(project, tmp_path):
    first = pd.read_csv(run_overview(**project)["class_csv"])
    second = pd.read_csv(run_overview(**{**project, "out_dir": tmp_path / "out2"})["class_csv"])
    pd.testing.assert_frame_equal(first, second)
