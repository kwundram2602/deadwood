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
