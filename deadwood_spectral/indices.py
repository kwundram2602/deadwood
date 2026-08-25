"""Spectral indices over a named band stack.

Pure functions: no I/O, no config. `green_red` and `brightness` come from the
RGB composite (bands R/G/B), the vegetation indices from the multispectral
bands (Green/Red/RedEdge/NIR) — the two groups are separate sensors in the
same file and must not be mixed.

Deadwood signature, for orientation: high visible reflectance, collapsed NIR,
and — the point of the whole project — no seasonal swing in ndvi/ndre.
"""

from collections.abc import Sequence

import numpy as np

# Band order of the time-series orthomosaics, from their GeoTIFF descriptions.
BAND_NAMES: tuple[str, ...] = ("R", "G", "B", "Green", "Red", "RedEdge", "NIR")

INDEX_NAMES: tuple[str, ...] = (
    "ndvi",
    "ndre",
    "gndvi",
    "nir_red_ratio",
    "brightness",
    "green_red",
)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num/den with a zero denominator yielding NaN rather than inf.

    inf would survive into the feature table and silently blow up the
    RandomForest; NaN is handled explicitly downstream.
    """
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=(den != 0))
    return out


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where the sum is zero."""
    return _safe_divide(a - b, a + b)


def _select(stack: np.ndarray, band_names: Sequence[str], name: str) -> np.ndarray:
    try:
        return stack[list(band_names).index(name)]
    except ValueError:
        raise ValueError(
            f"band {name!r} missing from stack; available: {list(band_names)}"
        ) from None


def compute_indices(stack: np.ndarray, band_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Compute every index in INDEX_NAMES from a (C, H, W) float32 stack."""
    r = _select(stack, band_names, "R")
    g = _select(stack, band_names, "G")
    b = _select(stack, band_names, "B")
    green = _select(stack, band_names, "Green")
    red = _select(stack, band_names, "Red")
    rededge = _select(stack, band_names, "RedEdge")
    nir = _select(stack, band_names, "NIR")

    return {
        "ndvi": normalized_difference(nir, red),
        "ndre": normalized_difference(nir, rededge),
        "gndvi": normalized_difference(nir, green),
        "nir_red_ratio": _safe_divide(nir, red),
        "brightness": ((r + g + b) / 3.0).astype(np.float32),
        "green_red": normalized_difference(g, r),
    }
