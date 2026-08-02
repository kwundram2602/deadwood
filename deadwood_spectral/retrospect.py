"""Apply an existing model to earlier seasonal cycles.

Same feature definition, different dates. For each deadwood object detected in
the current cycle, this reports the earliest cycle in which it already looked
dead — a rough date of mortality, at the resolution of a season.

Strictly a bonus product: the model was trained on labels valid near the field
survey, and applying it backwards assumes the spectral signature of deadwood
did not drift. Treat the dates as indicative.
"""

import logging
import re
from collections.abc import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cycle keys are sorted lexicographically (see `first_dead_cycle`) and that
# order is trusted to be chronological. This shape — a zero-padded four-digit
# start year, an underscore, a zero-padded two-digit end year — is the only
# one for which lexicographic order is guaranteed to match chronological
# order (e.g. "2023_24" < "2024_25"). Keys outside this shape (e.g. "2023_9")
# can sort wrong with no error, so they only get a warning, not a hard
# failure — see `first_dead_cycle`'s docstring for the full rationale.
_CYCLE_KEY_SHAPE = re.compile(r"^\d{4}_\d{2}$")


def first_dead_cycle(
    object_masks: Mapping[str, np.ndarray],
    objects: gpd.GeoDataFrame,
    labels: np.ndarray,
    validity_masks: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Earliest cycle in which each object was classified deadwood.

    object_masks: cycle name -> boolean deadwood mask on the reference grid.
    labels: connected-component label raster whose ids match objects.object_id.
    An object counts as dead in a cycle when the majority of its pixels are.
    validity_masks: optional cycle name -> boolean valid-data mask (True where
    the cycle's classification was not nodata). When given, the returned frame
    also reports the fraction of each object's footprint that was actually
    observed (not nodata) in the cycle it was first reported dead, and flags
    rows where that fraction is low. Nodata pixels read as "not dead" in the
    majority vote above, so a low-coverage cycle can silently look "alive"
    when the honest answer is "unknown" — this is how that uncertainty is
    surfaced in the output rather than only in a log line.

    Cycle-key ordering: cycles are ranked by `sorted(object_masks)`, i.e.
    plain lexicographic order on the key strings. This only matches
    chronological order when every key is a zero-padded "YYYY_YY" season
    label (e.g. "2023_24" < "2024_25"); an unpadded key such as "2023_9"
    would sort after "2023_10" and silently produce a wrong "first dead"
    date. Keys that don't match that shape trigger a `logger.warning` (not
    an exception, since this module also accepts arbitrary keys for testing
    and non-seasonal callers) so the risk isn't silent.
    """
    bad_keys = [k for k in object_masks if not _CYCLE_KEY_SHAPE.match(k)]
    if bad_keys:
        logger.warning(
            "retrospect cycle key(s) %s do not match the zero-padded 'YYYY_YY' "
            "shape; first_dead_cycle sorts keys lexicographically and trusts "
            "that to be chronological order, so results may be wrong if these "
            "keys aren't already in chronological order as plain strings",
            bad_keys,
        )

    cycles = sorted(object_masks)
    records = []
    for object_id in objects["object_id"].astype(int):
        footprint = labels == object_id
        n_pixels = int(footprint.sum())
        dead_in = [
            cycle
            for cycle in cycles
            if n_pixels and object_masks[cycle][footprint].mean() > 0.5
        ]
        first_cycle = dead_in[0] if dead_in else None

        coverage = float("nan")
        if validity_masks is not None and first_cycle is not None and n_pixels:
            coverage = float(validity_masks[first_cycle][footprint].mean())

        records.append(
            {
                "object_id": int(object_id),
                "first_dead_cycle": first_cycle,
                "cycles_dead": len(dead_in),
                "n_cycles": len(cycles),
                "first_dead_cycle_coverage": coverage,
                "low_confidence": bool(np.isfinite(coverage) and coverage < 0.5),
            }
        )
    # Built column-wise (rather than pd.DataFrame(records)) so that
    # "first_dead_cycle" stays a plain object column: pandas' string-dtype
    # inference (default since pandas 3.0) would otherwise silently turn a
    # missing entry's `None` into a float `NaN`, breaking `is None` checks
    # for objects that were never observed dead.
    return pd.DataFrame(
        {
            "object_id": [r["object_id"] for r in records],
            "first_dead_cycle": pd.Series(
                [r["first_dead_cycle"] for r in records], dtype=object
            ),
            "cycles_dead": [r["cycles_dead"] for r in records],
            "n_cycles": [r["n_cycles"] for r in records],
            "first_dead_cycle_coverage": [r["first_dead_cycle_coverage"] for r in records],
            "low_confidence": [r["low_confidence"] for r in records],
        }
    )
