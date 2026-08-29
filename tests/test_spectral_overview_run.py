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


def test_without_an_anchor_the_run_covers_the_whole_time_series(project):
    outputs = run_overview(**project)
    table = pd.read_csv(outputs["class_csv"], dtype={"date": str})
    assert set(table["date"]) == set(DATES)


def test_an_anchor_and_window_restrict_the_run(project):
    outputs = run_overview(**project, label_date="20241203", window_months=6)
    table = pd.read_csv(outputs["class_csv"], dtype={"date": str})
    assert set(table["date"]) == {"20240613", "20241203"}


def test_the_signature_follows_the_window_too(project):
    """A window with no dry-season acquisition must not report a dry signature."""
    outputs = run_overview(**project, label_date="20241203", window_months=3)
    table = pd.read_csv(outputs["signature_csv"])
    assert set(table["season"]) == {"wet"}


def test_an_anchor_without_an_acquisition_is_rejected(project):
    with pytest.raises(FileNotFoundError, match="20241204"):
        run_overview(**project, label_date="20241204", window_months=6)


def test_the_run_writes_the_sample_geometry(project):
    outputs = run_overview(**project)
    gdf = gpd.read_file(outputs["sample_gpkg"])
    assert set(gdf["class"]) == {"deadwood", "living", "background"}
    assert gdf.crs.to_epsg() == 32736


def test_the_sample_geometry_has_one_point_per_sampled_pixel(project):
    outputs = run_overview(**project)
    gdf = gpd.read_file(outputs["sample_gpkg"])
    # 50 living + 50 background by the cap, plus every deadwood pixel.
    assert (gdf["class"] == "living").sum() == 50
    assert (gdf["class"] == "background").sum() == 50
    assert (gdf.geometry.geom_type == "Point").all()


def _ndsm(path, holes=()):
    """A height surface over the crown region, with optional photogrammetry holes."""
    data = np.zeros(SHAPE, dtype=np.float32)
    data[:25] = 8.0
    for row, col in holes:
        data[row, col] = np.nan
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
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return path


def test_without_an_ndsm_the_run_skips_the_topography_table(project):
    outputs = run_overview(**project)
    assert "topography_csv" not in outputs
    assert "plot_topography" not in outputs


def test_the_ndsm_yields_one_row_per_soff_tree_plus_the_pooled_row(project, tmp_path):
    outputs = run_overview(**project, ndsm=_ndsm(tmp_path / "ndsm.tif"))
    table = pd.read_csv(outputs["topography_csv"], dtype={"tree_id": str})
    assert list(table["tree_id"]) == ["4157", "4170", "all_soff"]
    assert (table["median_m"] == 8.0).all()
    assert (table["valid_frac"] == 1.0).all()
    assert outputs["plot_topography"].exists()


def test_a_tree_the_photogrammetry_missed_is_reported_not_dropped(project, tmp_path):
    """4157 sits on rows 5..10, cols 5..13; blanking the whole band empties it."""
    holes = [(row, col) for row in range(5, 11) for col in range(5, 14)]
    outputs = run_overview(**project, ndsm=_ndsm(tmp_path / "holes.tif", holes=holes))
    table = pd.read_csv(outputs["topography_csv"], dtype={"tree_id": str}).set_index("tree_id")
    assert table.loc["4157", "n_valid_px"] == 0
    assert table.loc["4157", "valid_frac"] == 0.0
    assert np.isnan(table.loc["4157", "median_m"])
    assert table.loc["4170", "valid_frac"] == 1.0
