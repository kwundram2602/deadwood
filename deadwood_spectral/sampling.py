"""Build the three label pools and draw a balanced, grouped sample.

`deadwood` comes from the soff field polygons — the only real ground truth,
18 trees. `living` comes from the *model's* crown prediction, not from the son
polygons: that is the surface the classifier meets at inference time. It is
knowingly noisy, since the prediction can contain undetected deadwood; the
exclusion buffer removes the deadwood we know about, the rest is residual risk.

Every sample carries a group_id so no tree's pixels can straddle a CV split.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from scipy.ndimage import binary_dilation

from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid

logger = logging.getLogger(__name__)

CLASS_CODES: dict[str, int] = {"background": 0, "living": 1, "deadwood": 2}
INCLUDE_CATEGORIES = ("son", "soff")
MASK_NODATA = 255.0


def load_crowns(paths: Sequence[str | Path], grid: ReferenceGrid) -> gpd.GeoDataFrame:
    """Load crown polygons, keep son/soff, normalise the messy attribute columns."""
    frames = [gpd.read_file(p) for p in paths]
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=frames[0].crs).to_crs(grid.crs)

    gdf["crown_category"] = gdf["crown_category"].astype("string").str.strip().str.lower()
    gdf = gdf[gdf["crown_category"].isin(INCLUDE_CATEGORIES)].copy()
    # The field table contains 'nc ' with a trailing space and mixed case.
    gdf["coverage"] = gdf["coverage"].astype("string").str.strip().str.lower()
    gdf["certaintyLP"] = pd.to_numeric(gdf["certaintyLP"], errors="coerce").fillna(0).astype(int)
    gdf["tree_id"] = gdf["tree_id"].astype("string")
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].reset_index(drop=True)
    # 1-based: 0 is the rasterize fill value meaning "no polygon".
    gdf["poly_idx"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    logger.info(
        "%d crown polygons (%s)",
        len(gdf),
        dict(gdf["crown_category"].value_counts()),
    )
    return gdf


def apply_quality_filter(
    gdf: gpd.GeoDataFrame,
    min_certainty: int = 50,
    coverages: Sequence[str] = ("nc",),
) -> gpd.GeoDataFrame:
    """Flag polygons that pass the label-quality bar. Drops nothing.

    With 18 positives it must stay visible what the filter costs, so the
    excluded trees are carried through and reported separately.
    """
    gdf = gdf.copy()
    gdf["quality_ok"] = (gdf["certaintyLP"] >= min_certainty) & gdf["coverage"].isin(
        list(coverages)
    )
    logger.info(
        "quality filter: %d/%d polygons pass (certaintyLP >= %d, coverage in %s)",
        int(gdf["quality_ok"].sum()), len(gdf), min_certainty, list(coverages),
    )
    return gdf


def rasterize_polygons(
    gdf: gpd.GeoDataFrame,
    grid: ReferenceGrid,
    buffer_m: float = 0.0,
    values=None,
    fill: int = 0,
) -> np.ndarray:
    """Burn polygons onto the reference grid. Negative buffer_m erodes."""
    if len(gdf) == 0:
        return np.full(grid.shape, fill, dtype=np.int32)
    geoms = gdf.geometry.buffer(buffer_m) if buffer_m else gdf.geometry
    burn = np.ones(len(gdf), dtype=np.int32) if values is None else np.asarray(values, dtype=np.int32)
    shapes = [(g, int(v)) for g, v in zip(geoms, burn) if g is not None and not g.is_empty]
    if not shapes:
        return np.full(grid.shape, fill, dtype=np.int32)
    return rio_rasterize(
        shapes, out_shape=grid.shape, transform=grid.transform, fill=fill, dtype="int32"
    )


def binarize_crown_mask(
    path: str | Path, grid: ReferenceGrid, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """(crown, valid) booleans from a crown raster.

    Handles both the soft [0,1] label mask and predict.py's uint8 {0,1} output;
    both use 255 as the nodata sentinel.
    """
    with rasterio.open(path) as src:
        assert_matches_grid(src, grid, str(path))
        data = src.read(1).astype(np.float32)
    valid = np.isfinite(data) & (data != MASK_NODATA)
    crown = valid & (data >= threshold)
    logger.info(
        "crown mask: %d crown px, %d valid px of %d",
        int(crown.sum()), int(valid.sum()), data.size,
    )
    return crown, valid


def build_pools(
    crown: np.ndarray,
    valid: np.ndarray,
    gdf: gpd.GeoDataFrame,
    grid: ReferenceGrid,
    erode_m: float = 0.10,
    exclude_buffer_m: float = 1.0,
    edge_buffer_m: float = 0.25,
) -> dict[str, np.ndarray]:
    """Three disjoint boolean pools on the reference grid."""
    soff = gdf[gdf["crown_category"] == "soff"]

    deadwood = rasterize_polygons(soff, grid, buffer_m=-abs(erode_m)).astype(bool) & valid
    soff_excluded = rasterize_polygons(soff, grid, buffer_m=abs(exclude_buffer_m)).astype(bool)
    all_polys = rasterize_polygons(gdf, grid).astype(bool)

    px = max(1, int(round(abs(edge_buffer_m) / abs(grid.transform.a))))
    crown_dilated = binary_dilation(crown, iterations=px)

    living = crown & ~soff_excluded & valid
    background = ~crown_dilated & ~all_polys & ~soff_excluded & valid

    pools = {"background": background, "living": living, "deadwood": deadwood}
    for name, pool in pools.items():
        logger.info("pool %-10s %d px", name, int(pool.sum()))
    return pools


def block_ids(grid: ReferenceGrid, block_m: float = 20.0) -> np.ndarray:
    """Integer block index per pixel, for spatially stratified subsampling."""
    px = max(1, int(round(block_m / abs(grid.transform.a))))
    rows = np.arange(grid.height)[:, None] // px
    cols = np.arange(grid.width)[None, :] // px
    n_col_blocks = int(np.ceil(grid.width / px))
    return (rows * n_col_blocks + cols).astype(np.int32)


def _stratified_choice(
    flat_idx: np.ndarray, block_flat: np.ndarray, target: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw `target` indices spread as evenly as possible over the blocks."""
    if len(flat_idx) <= target:
        return flat_idx
    order = np.argsort(block_flat[flat_idx], kind="stable")
    by_block: dict[int, list[int]] = {}
    for i in flat_idx[order]:
        by_block.setdefault(int(block_flat[i]), []).append(int(i))
    for values in by_block.values():
        rng.shuffle(values)

    picked: list[int] = []
    keys = sorted(by_block)
    while len(picked) < target:
        drained = True
        for key in keys:
            bucket = by_block[key]
            if bucket:
                picked.append(bucket.pop())
                drained = False
                if len(picked) == target:
                    break
        if drained:
            break
    return np.array(sorted(picked), dtype=np.int64)


def draw_samples(
    pools: dict[str, np.ndarray],
    gdf: gpd.GeoDataFrame,
    grid: ReferenceGrid,
    negative_ratio: float = 5.0,
    max_pixels_per_class: int = 200_000,
    block_m: float = 20.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample pixels from the pools into a tidy table with group IDs.

    The deadwood pool is taken in full (a few thousand pixels at most); the two
    negative classes are cut to negative_ratio x that. Without this the imbalance
    is roughly 1:1000 and the forest collapses onto the majority classes.
    """
    rng = np.random.default_rng(seed)
    blocks = block_ids(grid, block_m).ravel()
    poly_raster = rasterize_polygons(gdf, grid, values=gdf["poly_idx"]).ravel()
    attrs = gdf.set_index("poly_idx")

    dead_idx = np.flatnonzero(pools["deadwood"].ravel())
    if dead_idx.size == 0:
        raise ValueError(
            "deadwood pool is empty — check erode_m, the quality filter and the "
            "polygon/reference CRS match"
        )
    dead_idx = _stratified_choice(dead_idx, blocks, max_pixels_per_class, rng)
    target_negative = min(max_pixels_per_class, int(negative_ratio * dead_idx.size))

    selected = {"deadwood": dead_idx}
    for name in ("living", "background"):
        idx = np.flatnonzero(pools[name].ravel())
        if idx.size == 0:
            raise ValueError(f"{name} pool is empty")
        selected[name] = _stratified_choice(idx, blocks, target_negative, rng)

    frames = []
    for name, idx in selected.items():
        rows, cols = np.unravel_index(idx, grid.shape)
        frame = pd.DataFrame(
            {
                "row": rows.astype(np.int32),
                "col": cols.astype(np.int32),
                "class_name": name,
                "class_code": CLASS_CODES[name],
            }
        )
        if name == "deadwood":
            poly = poly_raster[idx]
            frame["tree_id"] = attrs.loc[poly, "tree_id"].to_numpy()
            frame["species"] = attrs.loc[poly, "species"].to_numpy()
            frame["certaintyLP"] = attrs.loc[poly, "certaintyLP"].to_numpy()
            frame["coverage"] = attrs.loc[poly, "coverage"].to_numpy()
            frame["quality_ok"] = attrs.loc[poly, "quality_ok"].to_numpy()
            frame["group_id"] = "tree:" + frame["tree_id"].astype(str)
        else:
            frame["tree_id"] = pd.NA
            frame["species"] = pd.NA
            frame["certaintyLP"] = pd.NA
            frame["coverage"] = pd.NA
            frame["quality_ok"] = True
            frame["group_id"] = "block:" + pd.Series(blocks[idx]).astype(str)
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True).sort_values(["class_code", "row", "col"])
    out = out.reset_index(drop=True)
    logger.info("samples drawn: %s", dict(out["class_name"].value_counts()))
    return out
