"""Pixel rules: crown x deadwood x height class -> combo code -> eco class.

Pure numpy, no I/O, so every rule is testable on a handful of pixels.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

NODATA = 255
N_CODES = 12
MASK_VALUES = (0, 1, NODATA)

# Index = combo code = C + 2*D + 4*Hc.
COMBO_NAMES: tuple[str, ...] = (
    "ground",
    "crown_low",
    "dead_low",
    "dead_crown_low",
    "unlabelled_mid",
    "crown_mid",
    "dead_mid",
    "dead_crown_mid",
    "unlabelled_tall",
    "crown_tall",
    "dead_tall",
    "dead_crown_tall",
)
COMBO_COLORS: tuple[str, ...] = (
    "#f0f0f0",
    "#d9f0a3",
    "#fdae61",
    "#d6604d",
    "#998ec3",
    "#a6d96a",
    "#f46d43",
    "#b2182b",
    "#542788",
    "#1a9641",
    "#d7191c",
    "#67001f",
)


def check_thresholds(h1: float, h2: float) -> None:
    if not 0 < h1 < h2:
        raise ValueError(f"height thresholds must satisfy 0 < h1 < h2, got h1={h1}, h2={h2}")


def height_class(ndsm: np.ndarray, h1: float, h2: float) -> np.ndarray:
    """0 below h1, 1 from h1, 2 from h2. NaN compares False and lands in 0;
    combo_codes masks it out as nodata."""
    check_thresholds(h1, h2)
    return (ndsm >= h1).astype(np.uint8) + (ndsm >= h2).astype(np.uint8)


def _check_mask(values: np.ndarray, name: str) -> None:
    bad = ~np.isin(values, MASK_VALUES)
    if bad.any():
        raise ValueError(
            f"{name}: values {np.unique(values[bad])[:5].tolist()} outside {{0, 1, 255}} "
            "- a probability raster instead of a binary mask?"
        )


def combo_codes(
    crown: np.ndarray,
    deadwood: np.ndarray,
    ndsm: np.ndarray,
    h1: float,
    h2: float,
    ndsm_nodata: float | None = None,
) -> np.ndarray:
    """C + 2*D + 4*Hc as uint8, NODATA wherever any input is nodata."""
    _check_mask(crown, "crown")
    _check_mask(deadwood, "deadwood")
    valid = (crown != NODATA) & (deadwood != NODATA) & np.isfinite(ndsm)
    if ndsm_nodata is not None and np.isfinite(ndsm_nodata):
        valid &= ndsm != ndsm_nodata
    code = (
        (crown == 1).astype(np.uint8)
        + 2 * (deadwood == 1).astype(np.uint8)
        + 4 * height_class(ndsm, h1, h2)
    )
    return np.where(valid, code, NODATA).astype(np.uint8)


_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class EcoClass:
    value: int
    name: str
    color: tuple[int, int, int]
    codes: tuple[int, ...]


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))


def parse_classes(classes: Mapping) -> list[EcoClass]:
    """Validate the YAML mapping: every combo code exactly once, values 1..254."""
    eco = []
    for key, spec in classes.items():
        value = int(key)
        if not 1 <= value <= 254:
            raise ValueError(f"eco value {value} must be in 1..254 (255 is nodata)")
        color = str(spec["color"])
        if not _HEX.match(color):
            raise ValueError(f"eco class {value}: color {color!r} is not #rrggbb")
        codes = tuple(int(c) for c in spec["codes"])
        eco.append(EcoClass(value, str(spec["name"]), hex_to_rgb(color), codes))

    names = [e.name for e in eco]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    if dup_names:
        raise ValueError(f"duplicate class name(s): {dup_names}")

    seen = [c for e in eco for c in e.codes]
    unknown = sorted({c for c in seen if not 0 <= c < N_CODES})
    if unknown:
        raise ValueError(f"unknown combo code(s) {unknown}, valid codes are 0..{N_CODES - 1}")
    dup = sorted({c for c in seen if seen.count(c) > 1})
    if dup:
        raise ValueError(f"combo code(s) {dup} mapped more than once")
    missing = sorted(set(range(N_CODES)) - set(seen))
    if missing:
        raise ValueError(
            f"combo code(s) {missing} missing from the mapping: {[COMBO_NAMES[c] for c in missing]}"
        )
    return sorted(eco, key=lambda e: e.value)


def build_lookup(eco: Sequence[EcoClass]) -> np.ndarray:
    """256-entry table: lut[combo] -> eco value; NODATA stays NODATA."""
    lut = np.full(256, NODATA, dtype=np.uint8)
    for e in eco:
        lut[list(e.codes)] = e.value
    return lut
