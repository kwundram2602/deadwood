import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.coreg import estimate_shift, flagged_dates  # noqa: E402


def _textured(size=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((size, size)).astype(np.float32)


def test_zero_shift_recovered():
    ref = _textured()
    drow, dcol = estimate_shift(ref, ref.copy())
    assert drow == pytest.approx(0.0, abs=0.05)
    assert dcol == pytest.approx(0.0, abs=0.05)


def test_known_integer_shift_recovered():
    ref = _textured()
    moving = np.roll(np.roll(ref, 3, axis=0), -2, axis=1)
    drow, dcol = estimate_shift(ref, moving)
    assert drow == pytest.approx(3.0, abs=0.2)
    assert dcol == pytest.approx(-2.0, abs=0.2)


def test_nan_tiles_do_not_crash():
    ref = _textured()
    moving = ref.copy()
    moving[:5, :5] = np.nan
    drow, dcol = estimate_shift(ref, moving)
    assert np.isfinite(drow) and np.isfinite(dcol)


def test_flagged_dates_filters_on_the_flag_column():
    import pandas as pd

    report = pd.DataFrame(
        {
            "date": ["20230824", "20230906", "20230918"],
            "flagged": [False, True, True],
        }
    )
    assert flagged_dates(report) == ["20230906", "20230918"]


def _write_stack(path, data):
    import rasterio
    from rasterio.transform import from_origin

    profile = dict(
        driver="GTiff", dtype="float32", width=data.shape[2], height=data.shape[1],
        count=data.shape[0], crs="EPSG:32736",
        transform=from_origin(1000.0, 2000.0, 1.0, 1.0), nodata=np.nan,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"))
    return path


def _coreg_fixture(tmp_path, nan_dates=(), nan_block=48):
    """Two dates on a 64x64 grid, 7 bands, optional NaN block on some dates."""
    import rasterio
    from rasterio.transform import from_origin

    from deadwood_spectral.grid import ReferenceGrid

    grid = ReferenceGrid(64, 64, from_origin(1000.0, 2000.0, 1.0, 1.0),
                         rasterio.crs.CRS.from_epsg(32736))
    d = tmp_path / "ts"
    d.mkdir(exist_ok=True)
    for date in ("20230824", "20230906"):
        data = np.repeat(_textured(64)[None], 7, axis=0)
        if date in nan_dates:
            data[:, :nan_block, :] = np.nan
        _write_stack(d / f"{date}_stack.tif", data)
    return d, grid


def test_nan_heavy_tile_is_rejected_and_reported(tmp_path):
    """estimate_shift fills NaN with the tile mean; a large filled block
    abutting real data is a step edge that can bias the peak. Such a tile must
    never reach estimate_shift, and the report must say it was dropped."""
    from deadwood_spectral.coreg import coreg_report, nan_fraction

    # NaN block covers rows 0-31, i.e. y from 2000 down to 1968. Two tiles sit
    # inside it (y=1990) and two sit fully in clean data (y=1960).
    d, grid = _coreg_fixture(tmp_path, nan_dates=("20230906",), nan_block=32)
    tiles = [(1016.0, 1990.0), (1048.0, 1990.0), (1016.0, 1960.0), (1048.0, 1960.0)]
    report = coreg_report(
        d, grid, tiles, tile_size_px=16, reference_date="20230824",
        max_tile_nan_frac=0.02, min_tiles=1,
    ).set_index("date")

    assert nan_fraction(np.array([np.nan, 1.0, 2.0, 3.0])) == pytest.approx(0.25)
    # The reference date keeps every tile; the NaN-blocked date loses the
    # tiles that sit inside the blocked rows.
    assert report.loc["20230824", "n_tiles_rejected_nan"] == 0
    assert report.loc["20230906", "n_tiles_rejected_nan"] > 0
    assert report.loc["20230906", "n_tiles_rejected_nan"] == 2
    assert report.loc["20230906", "n_tiles"] == 2
    assert report.loc["20230906", "n_tiles_total"] == 4
    assert "rejected" in report.loc["20230906", "status"]
    # The surviving estimate is the clean-tile one: no step-edge bias.
    assert report.loc["20230906", "dx_m"] == pytest.approx(0.0, abs=0.2)


def test_date_with_too_few_usable_tiles_is_skipped_and_flagged(tmp_path):
    from deadwood_spectral.coreg import coreg_report, flagged_dates

    d, grid = _coreg_fixture(tmp_path, nan_dates=("20230906",), nan_block=64)
    tiles = [(1016.0, 1960.0), (1048.0, 1960.0), (1016.0, 1990.0)]
    report = coreg_report(
        d, grid, tiles, tile_size_px=16, reference_date="20230824", min_tiles=3,
    )
    row = report.set_index("date").loc["20230906"]

    assert row["n_tiles"] == 0
    assert np.isnan(row["dx_m"]) and np.isnan(row["dy_m"])
    assert bool(row["flagged"]) is True
    assert "min_tiles" in row["status"]
    # Flagged means excluded downstream, which is the point of skipping.
    assert "20230906" in flagged_dates(report)


def test_single_tile_spread_is_nan_not_zero(tmp_path):
    """spread_m over one tile is identically 0 — false confidence."""
    from deadwood_spectral.coreg import coreg_report

    d, grid = _coreg_fixture(tmp_path)
    report = coreg_report(
        d, grid, [(1016.0, 1960.0)], tile_size_px=16,
        reference_date="20230824", min_tiles=1,
    )
    assert report["n_tiles"].eq(1).all()
    assert report["spread_m"].isna().all()
