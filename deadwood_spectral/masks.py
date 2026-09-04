"""The three class masks the overview compares, built once on the reference grid.

None of them depends on a date: the field polygons and the crown prediction are
static layers, so they are built before the time loop and reused at every
acquisition. Rebuilding them per date would also make the pixel sample drift,
and a drifting sample turns every step in a curve into a question.

`deadwood` is ground truth (the soff field polygons). `living` is the *model's*
crown prediction, not the son polygons: that is the surface a classifier meets
at inference time. `background` is the reference floor, and it is not optional —
a leafless dead crown collapses spectrally toward bare ground, so without the
ground curve the deadwood curve has nothing to be distinguished from.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from scipy.ndimage import binary_dilation

from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid

logger = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, ...] = ("deadwood", "living", "background")
INCLUDE_CATEGORIES = ("son", "soff")
# Coverage classes that cannot be measured: `fc` is fully covered, the crown
# sits under a closed canopy. What DSM and sensor see there is the tree above
# it, not the dead one — so those crowns are dropped from the measured class.
# Their footprints are still cut out of the reference classes, see build_masks.
EXCLUDE_COVERAGE: tuple[str, ...] = ("fc",)
# Both the soft [0,1] label mask and predict.py's uint8 {0,1} output use 255.
MASK_NODATA = 255.0


@dataclass(frozen=True)
class ClassMasks:
    """Three disjoint boolean masks plus the per-tree attribution of `deadwood`."""

    deadwood: np.ndarray
    living: np.ndarray
    background: np.ndarray
    # 0 means "no soff polygon here"; every other value indexes `tree_ids`.
    tree_idx: np.ndarray
    tree_ids: dict[int, str]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in CLASS_NAMES}


def load_crowns(paths: Sequence[str | Path], grid: ReferenceGrid) -> gpd.GeoDataFrame:
    """Load crown polygons, keep son/soff, normalise the messy attribute columns."""
    frames = [gpd.read_file(p) for p in paths]
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=frames[0].crs).to_crs(grid.crs)

    gdf["crown_category"] = gdf["crown_category"].astype("string").str.strip().str.lower()
    gdf = gdf[gdf["crown_category"].isin(INCLUDE_CATEGORIES)].copy()
    # The field table contains 'nc ' with a trailing space and mixed case.
    gdf["coverage"] = gdf["coverage"].astype("string").str.strip().str.lower()
    gdf["tree_id"] = gdf["tree_id"].astype("string")
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].reset_index(drop=True)
    # 1-based: 0 is the rasterize fill value meaning "no polygon".
    gdf["poly_idx"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    logger.info("%d crown polygons (%s)", len(gdf), dict(gdf["crown_category"].value_counts()))
    return gdf


def rasterize_polygons(
    gdf: gpd.GeoDataFrame,
    grid: ReferenceGrid,
    buffer_m: float = 0.0,
    values=None,
    fill: int = 0,
    geometry=None,
) -> np.ndarray:
    """Burn polygons onto the reference grid. Negative buffer_m erodes."""
    if len(gdf) == 0:
        return np.full(grid.shape, fill, dtype=np.int32)
    geoms = gdf.geometry if geometry is None else geometry
    if buffer_m:
        geoms = geoms.buffer(buffer_m)
    burn = (
        np.ones(len(gdf), dtype=np.int32) if values is None else np.asarray(values, dtype=np.int32)
    )
    shapes = [(g, int(v)) for g, v in zip(geoms, burn) if g is not None and not g.is_empty]
    if not shapes:
        return np.full(grid.shape, fill, dtype=np.int32)
    return rio_rasterize(
        shapes, out_shape=grid.shape, transform=grid.transform, fill=fill, dtype="int32"
    )


def erode_by_area(gdf: gpd.GeoDataFrame, erode_m: float, min_area_m2: float) -> gpd.GeoSeries:
    """Erode only the crowns that survive it; leave the small ones raw.

    The soff polygons span 0.02 to 13.9 m2. Taking 10 cm off 0.02 m2 deletes the
    tree outright, and that is exactly what happened to 4157 — the model reported
    17 deadwood groups instead of 18. Below the threshold the raw geometry stays,
    edge mixed pixels and all: a noisier tree beats a missing one.
    """
    if len(gdf) == 0:
        return gdf.geometry
    eroded = gdf.geometry.buffer(-abs(erode_m))
    keep_raw = (gdf.geometry.area < abs(min_area_m2)) | eroded.isna() | eroded.is_empty
    if keep_raw.any():
        logger.info(
            "erode_m=%.3f skipped for %d/%d crown(s) under %.2f m2: %s",
            erode_m,
            int(keep_raw.sum()),
            len(gdf),
            min_area_m2,
            list(gdf.loc[keep_raw, "tree_id"]),
        )
    return eroded.where(~keep_raw, gdf.geometry)


def binarize_crown_mask(
    path: str | Path, grid: ReferenceGrid, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """(crown, valid) booleans from the binarized crown prediction.

    Feed this the thresholded `*_pred_t*.tif`, not the `*_prob.tif` beside it:
    the probability raster carries nodata = -1.0, which this function knows
    nothing about, so a -1 pixel would read as "valid, but below threshold".
    """
    with rasterio.open(path) as src:
        assert_matches_grid(src, grid, str(path))
        data = src.read(1).astype(np.float32)
    valid = np.isfinite(data) & (data != MASK_NODATA)
    crown = valid & (data >= threshold)
    logger.info(
        "crown prediction: %d crown px, %d valid px of %d",
        int(crown.sum()),
        int(valid.sum()),
        data.size,
    )
    return crown, valid


def build_masks(
    crown: np.ndarray,
    valid: np.ndarray,
    gdf: gpd.GeoDataFrame,
    grid: ReferenceGrid,
    erode_m: float = 0.10,
    erode_min_area_m2: float = 1.0,
    exclude_buffer_m: float = 1.0,
    edge_buffer_m: float = 0.25,
    exclude_coverage: Sequence[str] = EXCLUDE_COVERAGE,
) -> ClassMasks:
    """Three disjoint masks on the reference grid, plus per-tree attribution.

    `exclude_coverage` drops crowns from the *measured* class only. It must not
    be applied by removing them from `gdf`: their pixels would then fall into
    `background` or `living`, and a dead crown under a closed canopy would
    become reference ground — exactly the contamination `exclude_buffer_m`
    exists to prevent. So `measured` feeds the deadwood mask while `soff` and
    `gdf` keep every polygon for the cut-outs.
    """
    soff = gdf[gdf["crown_category"] == "soff"]
    # NA coverage stays in: unknown is not the same as fully covered, and
    # `.isin()` returns False for NA, which is the behaviour wanted here — it
    # reads like an accident otherwise.
    measured = soff[~soff["coverage"].isin(list(exclude_coverage))]
    dropped = sorted(set(soff["tree_id"]) - set(measured["tree_id"]))
    if dropped:
        logger.info(
            "coverage %s excludes %d of %d soff crown(s) from the deadwood class: %s",
            "/".join(exclude_coverage),
            len(dropped),
            len(soff),
            ", ".join(dropped),
        )
    # Only when coverage is what emptied it. No soff polygon at all is a
    # different fault and already has its own message further down.
    if not soff.empty and measured.empty:
        raise ValueError(f"every soff crown is excluded by coverage {tuple(exclude_coverage)}")

    tree_idx = rasterize_polygons(
        measured,
        grid,
        values=measured["poly_idx"],
        geometry=erode_by_area(measured, erode_m, erode_min_area_m2),
    )
    deadwood = (tree_idx > 0) & valid
    tree_idx = np.where(deadwood, tree_idx, 0).astype(np.int32)
    if not deadwood.any():
        raise ValueError("no deadwood pixels: no soff polygon survives on the valid grid")

    soff_excluded = rasterize_polygons(soff, grid, buffer_m=abs(exclude_buffer_m)).astype(bool)
    all_polys = rasterize_polygons(gdf, grid).astype(bool)

    if edge_buffer_m == 0:
        crown_dilated = crown
    else:
        px = max(1, int(round(abs(edge_buffer_m) / abs(grid.transform.a))))
        crown_dilated = binary_dilation(crown, iterations=px)

    living = crown & ~soff_excluded & valid
    background = ~crown_dilated & ~all_polys & ~soff_excluded & valid

    masks = ClassMasks(
        deadwood=deadwood,
        living=living,
        background=background,
        tree_idx=tree_idx,
        tree_ids={int(i): str(t) for i, t in zip(measured["poly_idx"], measured["tree_id"])},
    )
    for name, array in masks.as_dict().items():
        logger.info("mask %-10s %d px", name, int(array.sum()))
    return masks
