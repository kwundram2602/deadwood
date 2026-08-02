"""Apply an existing model to earlier seasonal cycles.

Same feature definition, different dates. For each deadwood object detected in
the current cycle, this reports the earliest cycle in which it already looked
dead — a rough date of mortality, at the resolution of a season.

Strictly a bonus product: the model was trained on labels valid near the field
survey, and applying it backwards assumes the spectral signature of deadwood
did not drift. Treat the dates as indicative.
"""

import logging
from collections.abc import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def first_dead_cycle(
    object_masks: Mapping[str, np.ndarray],
    objects: gpd.GeoDataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Earliest cycle in which each object was classified deadwood.

    object_masks: cycle name -> boolean deadwood mask on the reference grid.
    labels: connected-component label raster whose ids match objects.object_id.
    An object counts as dead in a cycle when the majority of its pixels are.
    """
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
        records.append(
            {
                "object_id": int(object_id),
                "first_dead_cycle": dead_in[0] if dead_in else None,
                "cycles_dead": len(dead_in),
                "n_cycles": len(cycles),
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
        }
    )
