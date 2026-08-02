"""Table -> fixed-length feature matrix for one seasonal cycle.

Column order is deterministic and is persisted next to the trained model. A
RandomForest silently accepts a reordered matrix and produces garbage, so
inference asserts the order rather than trusting it.

nDSM carries more weight than its single column suggests: without height, a
dead trunk and a dry grass patch look much alike.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from deadwood_spectral.extract import feature_column

logger = logging.getLogger(__name__)

PER_DATE_MEASURES: tuple[str, ...] = (
    "ndvi", "ndre", "NIR", "Red", "brightness", "green_red",
)
TEMPORAL_MEASURES: tuple[str, ...] = ("ndvi", "ndre")
TEMPORAL_STATS: tuple[str, ...] = (
    "max", "min", "amplitude", "mean", "std", "greenup_slope",
)
STATIC_FEATURES: tuple[str, ...] = ("ndsm",)


def feature_names(
    dates: Sequence[str],
    per_date: bool = True,
    temporal: bool = True,
    static: bool = True,
) -> list[str]:
    """The exact column order build_features produces."""
    names: list[str] = []
    if per_date:
        names += [feature_column(m, d) for m in PER_DATE_MEASURES for d in dates]
    if temporal:
        names += [f"{m}_{s}" for m in TEMPORAL_MEASURES for s in TEMPORAL_STATS]
    if static:
        names += list(STATIC_FEATURES)
    return names


def _measure_matrix(df: pd.DataFrame, measure: str, dates: Sequence[str]) -> np.ndarray:
    columns = []
    for date in dates:
        column = feature_column(measure, date)
        if column not in df.columns:
            raise ValueError(f"missing column {column!r} for date {date}")
        columns.append(column)
    return df[columns].to_numpy(dtype=np.float64)


def _greenup_slope(values: np.ndarray) -> np.ndarray:
    """Least-squares slope over evenly spaced dates, per row.

    Dates are treated as equally spaced steps rather than calendar days: the
    flights are roughly biweekly, and a day-scaled slope would be dominated by
    the occasional long gap rather than by phenology.
    """
    n = values.shape[1]
    x = np.arange(n, dtype=np.float64)
    x_centred = x - x.mean()
    denominator = float((x_centred**2).sum())
    if denominator == 0:
        return np.zeros(values.shape[0])
    y_centred = values - values.mean(axis=1, keepdims=True)
    return (y_centred * x_centred).sum(axis=1) / denominator


def build_features(
    df: pd.DataFrame,
    dates: Sequence[str],
    per_date: bool = True,
    temporal: bool = True,
    static: bool = True,
) -> pd.DataFrame:
    """Build the feature matrix in the exact order of feature_names()."""
    dates = list(dates)
    out = pd.DataFrame(index=df.index)

    if per_date:
        for measure in PER_DATE_MEASURES:
            values = _measure_matrix(df, measure, dates)
            for i, date in enumerate(dates):
                out[feature_column(measure, date)] = values[:, i]

    if temporal:
        for measure in TEMPORAL_MEASURES:
            values = _measure_matrix(df, measure, dates)
            out[f"{measure}_max"] = values.max(axis=1)
            out[f"{measure}_min"] = values.min(axis=1)
            out[f"{measure}_amplitude"] = values.max(axis=1) - values.min(axis=1)
            out[f"{measure}_mean"] = values.mean(axis=1)
            out[f"{measure}_std"] = values.std(axis=1, ddof=0)
            out[f"{measure}_greenup_slope"] = _greenup_slope(values)

    if static:
        for name in STATIC_FEATURES:
            if name not in df.columns:
                raise ValueError(f"missing static feature column {name!r}")
            out[name] = df[name].to_numpy(dtype=np.float64)

    expected = feature_names(dates, per_date, temporal, static)
    return out[expected]


def save_feature_names(names: Sequence[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"names": list(names)}, indent=2))


def load_feature_names(path: str | Path) -> list[str]:
    return list(json.loads(Path(path).read_text())["names"])


def assert_feature_names(actual: Sequence[str], expected: Sequence[str]) -> None:
    """Fail unless the feature columns match exactly, in order."""
    actual, expected = list(actual), list(expected)
    if actual == expected:
        return
    if set(actual) == set(expected):
        raise ValueError("feature columns match but their order differs from the trained model")
    missing = [n for n in expected if n not in actual]
    extra = [n for n in actual if n not in expected]
    raise ValueError(f"feature mismatch — missing {missing[:5]}, unexpected {extra[:5]}")
