"""Which height system a DSM or DTM is in, and the geoid conversion between them.

Nothing is fitted here. Each raster's coarse ground surface is compared with
Copernicus GLO-30 (orthometric, EGM2008) under both hypotheses, "these heights
are orthometric" and "these heights are ellipsoidal" (= minus the geoid
undulation N). The nearer hypothesis wins when it is nearer than N/2 — the two
lie N apart, so that is the point where the call stops being decisive. A
winning residual above SURVEY_OFFSET_WARN_M is a survey offset of that
campaign (GNSS base, no ground control), reported but not a reason to doubt
the height system. `harmonise` converts a terrain raster into a DSM's height
system by the geoid alone; any remaining offset is the co-registration's job.
The building blocks are those of scripts/check_vertical_datum.py; this module
turns them into a result the pipeline can record and the sweep can use.

Results are cached per file (path, size, mtime) in a JSON file: the check
downloads geoid grids (cdn.proj.org) and reference tiles (AWS), and the
rasters do not change between runs.

    h_ellipsoidal = H_orthometric + N
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyproj.network
import rasterio
import rasterio.transform
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from scripts.check_vertical_datum import geoid_undulation, verify_absolute

logger = logging.getLogger(__name__)

MODEL = "EGM2008"  # the geoid of the Copernicus reference
SURVEY_OFFSET_WARN_M = 5.0  # winning residual above this: flagged as a survey offset
CACHE_SCHEMA = "v2"  # bump when the classification rule changes; older cache entries are recomputed
REFERENCE_RESOLUTION_M = 0.5
REFERENCE_BLOCK_M = 30.0
SYSTEMS = ("orthometric", "ellipsoidal", "unknown")


@dataclass(frozen=True)
class HeightSystem:
    """Result for one raster."""

    system: str  # orthometric | ellipsoidal | unknown (neither hypothesis nearer than N/2)
    residual_m: float  # ground minus reference under the winning hypothesis (else the smaller one)
    # = the campaign's survey offset against GLO-30 when the system is known
    as_orthometric_m: float
    as_ellipsoidal_m: float
    tilt_m: float  # against the reference over the raster's extent
    scatter_m: float


@dataclass(frozen=True)
class DatumCheck:
    """DSM and DTM of one scene: their height systems and whether they differ."""

    dsm: HeightSystem
    dtm: HeightSystem
    undulation_m: float  # N at the scene centre
    model: str

    @property
    def mixed(self) -> bool:
        return "unknown" not in (self.dsm.system, self.dtm.system) and self.dsm.system != self.dtm.system

    def summary(self) -> str:
        def one(h: HeightSystem) -> str:
            flag = ", SURVEY OFFSET" if h.system != "unknown" and abs(h.residual_m) > SURVEY_OFFSET_WARN_M else ""
            return f"{h.system} ({h.residual_m:+.2f} m vs. GLO-30{flag}, tilt {h.tilt_m:.2f} m)"

        verdict = (
            f"DATUM MIX (N = {self.undulation_m:+.2f} m, {self.model})" if self.mixed
            else "height system UNKNOWN for at least one raster" if "unknown" in (self.dsm.system, self.dtm.system)
            else "same height system"
        )  # fmt: skip
        return f"DSM {one(self.dsm)} | DTM {one(self.dtm)} -> {verdict}"

    def to_dict(self) -> dict:
        return {**asdict(self), "mixed": self.mixed}


def _centre_lonlat(path: str | Path) -> tuple[float, float]:
    with rasterio.open(path) as src:
        b = src.bounds
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        return to_wgs84.transform((b.left + b.right) / 2, (b.bottom + b.top) / 2)


def height_system(path: str | Path, undulation: float) -> HeightSystem:
    """Classify one raster against Copernicus GLO-30 (needs network)."""
    result = verify_absolute(Path(path), REFERENCE_RESOLUTION_M, REFERENCE_BLOCK_M, undulation)
    if result is None:
        nan = float("nan")
        return HeightSystem("unknown", nan, nan, nan, nan, nan)
    ortho, ellip = result["as_orthometric"], result["as_ellipsoidal"]
    best = "orthometric" if abs(ortho) <= abs(ellip) else "ellipsoidal"
    residual = ortho if best == "orthometric" else ellip
    system = best if abs(residual) < abs(undulation) / 2.0 else "unknown"
    return HeightSystem(system, residual, ortho, ellip, result["tilt"], result["scatter"])


def _cache_key(path: str | Path) -> str:
    st = os.stat(path)
    return f"{CACHE_SCHEMA}|{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


def _numbers(entry: dict) -> list[float]:
    return [float(v) for v in entry.values() if isinstance(v, (int, float))]


def check_pair(dsm: str | Path, dtm: str | Path, cache: str | Path | None = None) -> DatumCheck:
    """Height systems of a DSM and a DTM; per-raster results come from `cache` when present."""
    pyproj.network.set_network_enabled(True)
    store: dict = {}
    if cache is not None and Path(cache).is_file():
        store = json.loads(Path(cache).read_text())
        # Entries of an older rule, or written with an infinite undulation (PROJ without its
        # geoid grid), are void
        store = {
            k: v for k, v in store.items()
            if k.startswith(f"{CACHE_SCHEMA}|") and all(np.isfinite(x) or np.isnan(x) for x in _numbers(v))
        }  # fmt: skip
    undulation = geoid_undulation(*_centre_lonlat(dsm), MODEL)
    if not np.isfinite(undulation):
        raise RuntimeError(f"{MODEL} undulation is {undulation}: PROJ could not use its geoid grid")

    def lookup(path) -> HeightSystem:
        key = _cache_key(path)
        if key not in store:
            store[key] = asdict(height_system(path, undulation))
        return HeightSystem(**store[key])

    check = DatumCheck(lookup(dsm), lookup(dtm), undulation, MODEL)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(store, indent=2))
    return check


def undulation_grid(shape: tuple[int, int], transform, crs, model: str = MODEL, samples: int = 5) -> np.ndarray:
    """Geoid undulation N (m) on a raster grid.

    Evaluated on a samples x samples lattice over the extent and interpolated
    bilinearly — N changes by centimetres per kilometre, so this is exact to
    well below a millimetre at scene scale.
    """
    pyproj.network.set_network_enabled(True)
    h, w = shape
    rows = np.linspace(0, h - 1, samples)
    cols = np.linspace(0, w - 1, samples)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lattice = np.empty((samples, samples))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            x, y = rasterio.transform.xy(transform, r, c)
            lattice[i, j] = geoid_undulation(*to_wgs84.transform(x, y), model)
    interp = RegularGridInterpolator((rows, cols), lattice)
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return interp(np.stack([rr.ravel(), cc.ravel()], axis=1)).reshape(h, w).astype(np.float32)


def to_dsm_system(dtm: np.ndarray, check: DatumCheck, undulation: np.ndarray) -> np.ndarray:
    """The DTM converted into the DSM's height system by the geoid alone (no fit).

    Raises if either height system is unknown: then no conversion is defined.
    """
    if not check.mixed:
        if "unknown" in (check.dsm.system, check.dtm.system):
            raise ValueError(f"cannot convert heights: {check.summary()}")
        return dtm
    if check.dtm.system == "ellipsoidal":  # DSM orthometric: H = h - N
        return (dtm - undulation).astype(np.float32)
    return (dtm + undulation).astype(np.float32)  # DTM orthometric, DSM ellipsoidal: h = H + N


def harmonise(
    terrain: np.ndarray,
    dsm_path: str | Path,
    terrain_path: str | Path,
    shape: tuple[int, int],
    transform,
    crs,
    cache: str | Path | None = None,
) -> tuple[np.ndarray, dict | None]:
    """A terrain raster (on the DSM's grid) converted into the DSM's height system by the geoid.

    Nothing is fitted: offset, tilt or doming beyond the datum stay for the
    co-registration. When the check fails or a height system is unknown, the
    terrain is returned unchanged with a warning. Returns the terrain and the
    check as a dict (None if it failed), with "converted" set.
    """
    try:
        check = check_pair(dsm_path, terrain_path, cache)
    except Exception as exc:  # check only: no conversion without it
        print(f"  [WARN] vertical datum check failed, no geoid conversion: {type(exc).__name__}: {exc}")
        return terrain, None
    print(f"  vertical datum: {check.summary()}")
    if not check.mixed:
        if "unknown" in (check.dsm.system, check.dtm.system):
            print("  [WARN] height system unknown: no geoid conversion, the co-registration absorbs the offset")
        return terrain, {**check.to_dict(), "converted": False}
    converted = to_dsm_system(terrain, check, undulation_grid(shape, transform, crs))
    print(f"  terrain converted {check.dtm.system} -> {check.dsm.system} by the geoid "
          f"(N = {check.undulation_m:+.2f} m, {check.model})")
    return converted, {**check.to_dict(), "converted": True}
