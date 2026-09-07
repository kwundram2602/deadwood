"""Per-crown scoring for sparse deadwood labels.

The labels are sparse-sampled, so predictions outside the crown polygons are
not false positives: a large share is likely real, unsurveyed deadwood. Every
number here is therefore recall-side only — coverage of known crowns — and no
precision figure is derivable from it.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features


def crown_pixel_index(
    crowns: gpd.GeoDataFrame, transform, shape: tuple[int, int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Row/col indices of each crown's pixels on the given grid, keyed by fid.

    Computed once and reused across thresholds — rasterising 31 crowns per
    sweep step would dominate the runtime of an otherwise trivial loop.
    """
    index: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fid, geom in zip(crowns.index, crowns.geometry):
        if geom is None or geom.is_empty:
            index[int(fid)] = (np.array([], dtype=int), np.array([], dtype=int))
            continue
        burn = features.rasterize(
            [(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8"
        )
        index[int(fid)] = np.nonzero(burn)
    return index


def crown_coverage(
    mask: np.ndarray, index: dict[int, tuple[np.ndarray, np.ndarray]], min_overlap: float = 0.1
) -> pd.DataFrame:
    """Fraction of each crown covered by the binary mask, and whether it counts as hit."""
    rows = []
    for fid, (r, c) in index.items():
        n_px = int(r.size)
        covered = int(mask[r, c].astype(bool).sum()) if n_px else 0
        fraction = covered / n_px if n_px else 0.0
        rows.append(
            {
                "fid": fid,
                "n_px": n_px,
                "covered_px": covered,
                "covered_fraction": fraction,
                "hit": fraction > min_overlap,
            }
        )
    return pd.DataFrame(rows).set_index("fid")


def threshold_sweep(
    probs: np.ndarray,
    index: dict[int, tuple[np.ndarray, np.ndarray]],
    thresholds,
    min_overlap: float = 0.1,
) -> pd.DataFrame:
    """Recall against decision threshold, from one probability raster."""
    rows = []
    for t in thresholds:
        cov = crown_coverage(probs > float(t), index, min_overlap=min_overlap)
        rows.append(
            {
                "threshold": float(t),
                "n_hit": int(cov["hit"].sum()),
                "n_crowns": int(len(cov)),
                "mean_covered_fraction": float(cov["covered_fraction"].mean()),
            }
        )
    return pd.DataFrame(rows)
