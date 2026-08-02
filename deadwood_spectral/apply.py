"""Apply the classifier to the whole scene and aggregate deadwood objects.

The classifier deliberately runs over the entire AOI rather than inside the
crown mask: the mask can contain deadwood, and deadwood can stand outside it.

Tiling here is a memory device, not a smoothing device — unlike predict.py's
Hann-blended CNN tiles, per-pixel RandomForest predictions do not depend on
their neighbours, so tiles are written straight into the output and the result
is identical to a whole-array run.
"""

import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.windows import Window
from shapely.geometry import shape as shapely_shape
from skimage.measure import label as cc_label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral.extract import feature_column
from deadwood_spectral.features import assert_feature_names, build_features, feature_names
from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid
from deadwood_spectral.indices import BAND_NAMES, compute_indices
from deadwood_spectral.sampling import CLASS_CODES
from scripts.predict import make_windows  # noqa: E402

logger = logging.getLogger(__name__)

DEADWOOD_CODE = CLASS_CODES["deadwood"]
N_CLASSES = len(CLASS_CODES)


def window_table(
    stack_dir: str | Path,
    dates: Sequence[str],
    grid: ReferenceGrid,
    row_off: int,
    col_off: int,
    height: int,
    width: int,
    ndsm_path: str | Path | None = None,
) -> pd.DataFrame:
    """One row per pixel of the window, columns as build_features expects."""
    stack_dir = Path(stack_dir)
    window = Window(col_off, row_off, width, height)
    out = pd.DataFrame(index=pd.RangeIndex(height * width))

    for date in dates:
        path = stack_dir / f"{date}_stack.tif"
        with rasterio.open(path) as src:
            assert_matches_grid(src, grid, str(path))
            block = src.read(window=window).astype(np.float32)
            names = [d or n for d, n in zip(src.descriptions, BAND_NAMES)]
        for name, band in zip(names, block):
            out[feature_column(name, date)] = band.reshape(-1)
        for name, arr in compute_indices(block, names).items():
            out[feature_column(name, date)] = arr.reshape(-1)

    if ndsm_path is not None:
        with rasterio.open(ndsm_path) as src:
            assert_matches_grid(src, grid, str(ndsm_path))
            out["ndsm"] = src.read(1, window=window).astype(np.float32).reshape(-1)
    return out


def predict_scene(
    stack_dir: str | Path,
    dates: Sequence[str],
    grid: ReferenceGrid,
    model,
    features: Sequence[str],
    ndsm_path: str | Path | None,
    switches: dict,
    tile_size: int = 1024,
    stride: int = 896,
) -> np.ndarray:
    """(N_CLASSES, H, W) class probabilities over the whole grid."""
    dates = list(dates)
    assert_feature_names(feature_names(dates, **switches), list(features))

    proba = np.zeros((N_CLASSES, grid.height, grid.width), dtype=np.float32)
    offsets = make_windows(grid.height, grid.width, tile_size, stride)
    for n, (row_off, col_off) in enumerate(offsets, start=1):
        height = min(tile_size, grid.height - row_off)
        width = min(tile_size, grid.width - col_off)
        table = window_table(stack_dir, dates, grid, row_off, col_off, height, width, ndsm_path)
        matrix = build_features(table, dates, **switches)
        assert_feature_names(list(matrix.columns), list(features))

        values = matrix.to_numpy(dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        # Non-finite pixels (outside the source footprint) must stay NaN, not
        # fall back to a zero vector: argmax(zeros) picks class 0 (background),
        # which would silently turn "unknown" into a confident prediction.
        tile = np.full((height * width, N_CLASSES), np.nan, dtype=np.float32)
        if finite.any():
            tile[finite] = 0.0
            predicted = model.predict_proba(values[finite])
            for column, class_label in enumerate(np.asarray(model.classes_, dtype=int)):
                tile[finite, class_label] = predicted[:, column]
        # Overlapping tiles overwrite with an identical value: per-pixel
        # predictions carry no neighbourhood context, so no blending is needed.
        proba[:, row_off : row_off + height, col_off : col_off + width] = (
            tile.reshape(height, width, N_CLASSES).transpose(2, 0, 1)
        )
        if n % 10 == 0 or n == len(offsets):
            logger.info("predicted tile %d/%d", n, len(offsets))
    return proba


def aggregate_objects(
    class_raster: np.ndarray,
    prob_raster: np.ndarray,
    grid: ReferenceGrid,
    ndsm: np.ndarray | None = None,
    min_object_m2: float = 0.5,
) -> gpd.GeoDataFrame:
    """Connected deadwood pixels -> countable tree objects.

    Connected components rather than watershed: a deliberate simplification.
    Touching deadwood crowns merge into one object, which is honest about what
    the data supports at 18 training trees.
    """
    pixel_area = abs(grid.transform.a) * abs(grid.transform.e)
    deadwood = class_raster == DEADWOOD_CODE
    labels = cc_label(deadwood, connectivity=2)

    records = []
    for label_id in range(1, int(labels.max()) + 1):
        mask = labels == label_id
        area = float(mask.sum()) * pixel_area
        if area < min_object_m2:
            continue
        polygons = [
            shapely_shape(geom)
            for geom, _ in rio_shapes(
                mask.astype(np.uint8), mask=mask, transform=grid.transform, connectivity=8
            )
        ]
        geometry = polygons[0] if len(polygons) == 1 else max(polygons, key=lambda p: p.area)
        record = {
            "object_id": label_id,
            "area_m2": area,
            "n_pixels": int(mask.sum()),
            "mean_prob": float(prob_raster[DEADWOOD_CODE][mask].mean()),
            "geometry": geometry,
        }
        if ndsm is not None:
            values = ndsm[mask]
            values = values[np.isfinite(values)]
            record["mean_height_m"] = float(values.mean()) if values.size else float("nan")
        records.append(record)

    columns = ["object_id", "area_m2", "n_pixels", "mean_prob", "geometry"]
    if ndsm is not None:
        columns.insert(-1, "mean_height_m")
    if not records:
        return gpd.GeoDataFrame(
            {c: pd.Series(dtype="float64") for c in columns if c != "geometry"},
            geometry=gpd.GeoSeries([], crs=grid.crs),
            crs=grid.crs,
        )
    logger.info("aggregated %d deadwood object(s)", len(records))
    return gpd.GeoDataFrame(records, geometry="geometry", crs=grid.crs)[columns]
