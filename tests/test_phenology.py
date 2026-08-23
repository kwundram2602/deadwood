import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.phenology import (  # noqa: E402
    FEATURE_NAMES,
    stack_paths,
    window_dates,
)

ALL_DATES = (
    "20231114",
    "20240220",  # zwei Zyklen früher
    "20250417",
    "20250907",
    "20251121",
    "20260226",
    "20260313",
    "20260401",  # nach dem Ankerdatum
)


def _stack_dir(tmp_path, dates=ALL_DATES):
    d = tmp_path / "ts"
    d.mkdir()
    for date in dates:
        (d / f"{date}_stack.tif").write_bytes(b"")
    (d / "channels.json").write_text("{}")
    return d


def test_window_dates_keeps_only_the_window(tmp_path):
    dates = window_dates(_stack_dir(tmp_path), "20260313", 12)
    assert dates == ["20250417", "20250907", "20251121", "20260226", "20260313"]


def test_window_dates_includes_the_label_date_and_excludes_the_window_start(tmp_path):
    d = _stack_dir(tmp_path, ("20250313", "20250314", "20260313"))
    assert window_dates(d, "20260313", 12) == ["20250314", "20260313"]


def test_window_dates_raises_when_the_window_is_empty(tmp_path):
    d = _stack_dir(tmp_path, ("20231114",))
    with pytest.raises(ValueError, match="no aligned stack"):
        window_dates(d, "20260313", 12)


def test_window_dates_handles_a_month_end_anchor(tmp_path):
    # 20260331 minus 1 Monat ist der 28.02.2026 — kein 31. Februar.
    d = _stack_dir(tmp_path, ("20260227", "20260301", "20260331"))
    assert window_dates(d, "20260331", 1) == ["20260301", "20260331"]


def test_feature_names_are_fixed_and_carry_no_date():
    assert len(FEATURE_NAMES) == 31
    assert FEATURE_NAMES[-1] == "ndsm"
    assert not any(c.isdigit() for name in FEATURE_NAMES for c in name)


def test_stack_paths_fails_loudly_on_a_missing_stack(tmp_path):
    d = _stack_dir(tmp_path, ("20260313",))
    with pytest.raises(FileNotFoundError, match="20250417"):
        stack_paths(d, ["20250417", "20260313"])
