"""Pixel rules: crown x deadwood x height class -> combo code -> eco class.

Pure numpy, no I/O, so every rule is testable on a handful of pixels.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

NODATA = 255
MASK_VALUES = (0, 1, NODATA)

# Height levels per number of nDSM thresholds.
HEIGHT_LEVELS: dict[int, tuple[str, ...]] = {1: ("low", "tall"), 2: ("low", "mid", "tall")}
# Per level: (neither, crown, dead, dead+crown); the unlabelled low pixel is "ground".
_LEVEL_COLORS: dict[str, tuple[str, str, str, str]] = {
    "low": ("#f0f0f0", "#d9f0a3", "#fdae61", "#d6604d"),
    "mid": ("#998ec3", "#a6d96a", "#f46d43", "#b2182b"),
    "tall": ("#542788", "#1a9641", "#d7191c", "#67001f"),
}


def _levels(n_thresholds: int) -> tuple[str, ...]:
    if n_thresholds not in HEIGHT_LEVELS:
        raise ValueError(f"need 1 or 2 height thresholds, got {n_thresholds}")
    return HEIGHT_LEVELS[n_thresholds]


def combo_names(n_thresholds: int) -> tuple[str, ...]:
    """Index = combo code = C + 2*D + 4*Hc."""
    names: list[str] = []
    for level in _levels(n_thresholds):
        unlabelled = "ground" if level == "low" else f"unlabelled_{level}"
        names += [unlabelled, f"crown_{level}", f"dead_{level}", f"dead_crown_{level}"]
    return tuple(names)


def combo_colors(n_thresholds: int) -> tuple[str, ...]:
    return tuple(c for level in _levels(n_thresholds) for c in _LEVEL_COLORS[level])


# The default two-threshold scheme.
COMBO_NAMES = combo_names(2)
COMBO_COLORS = combo_colors(2)
N_CODES = len(COMBO_NAMES)


def check_thresholds(thresholds: Sequence[float]) -> None:
    _levels(len(thresholds))
    if not (0 < thresholds[0] and all(a < b for a, b in zip(thresholds, thresholds[1:]))):
        raise ValueError(
            f"height thresholds must be positive and increasing, got {list(thresholds)}"
        )


def height_class(ndsm: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Number of thresholds reached: 0 below the first, len(thresholds) at the last.
    NaN compares False and lands in 0; combo_codes masks it out as nodata."""
    check_thresholds(thresholds)
    hc = np.zeros(ndsm.shape, dtype=np.uint8)
    for t in thresholds:
        hc += (ndsm >= t).astype(np.uint8)
    return hc


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
    thresholds: Sequence[float],
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
        + 4 * height_class(ndsm, thresholds)
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


def parse_classes(classes: Mapping, n_thresholds: int = 2) -> list[EcoClass]:
    """Validate the YAML mapping: every combo code exactly once, values 1..254."""
    names_by_code = combo_names(n_thresholds)
    n_codes = len(names_by_code)
    eco = []
    for key, spec in classes.items():
        value = int(key)
        if not 1 <= value <= 254:
            raise ValueError(f"eco value {value} must be in 1..254 (255 is nodata)")
        missing_keys = [k for k in ("name", "color", "codes") if k not in spec]
        if missing_keys:
            raise ValueError(f"eco class {value}: missing key(s) {missing_keys}")
        color = str(spec["color"])
        if not _HEX.match(color):
            raise ValueError(f"eco class {value}: color {color!r} is not #rrggbb")
        if not isinstance(spec["codes"], list | tuple):
            raise ValueError(f"eco class {value}: codes must be a list")
        codes = tuple(int(c) for c in spec["codes"])
        eco.append(EcoClass(value, str(spec["name"]), hex_to_rgb(color), codes))

    names = [e.name for e in eco]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    if dup_names:
        raise ValueError(f"duplicate class name(s): {dup_names}")

    seen = [c for e in eco for c in e.codes]
    unknown = sorted({c for c in seen if not 0 <= c < n_codes})
    if unknown:
        raise ValueError(f"unknown combo code(s) {unknown}, valid codes are 0..{n_codes - 1}")
    dup = sorted({c for c in seen if seen.count(c) > 1})
    if dup:
        raise ValueError(f"combo code(s) {dup} mapped more than once")
    missing = sorted(set(range(n_codes)) - set(seen))
    if missing:
        raise ValueError(
            f"combo code(s) {missing} missing from the mapping: "
            f"{[names_by_code[c] for c in missing]}"
        )
    return sorted(eco, key=lambda e: e.value)


def build_lookup(eco: Sequence[EcoClass]) -> np.ndarray:
    """256-entry table: lut[combo] -> eco value; NODATA stays NODATA."""
    lut = np.full(256, NODATA, dtype=np.uint8)
    for e in eco:
        lut[list(e.codes)] = e.value
    return lut
