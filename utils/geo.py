from pathlib import Path

import numpy as np
import rasterio
import torch


def read_raster(path: str | Path, bands: list[int] | None = None) -> np.ndarray:
    """Read a raster file into a (C, H, W) float32 array.

    bands: 1-indexed list of bands to read. None = all bands.
    """
    with rasterio.open(path) as src:
        data = src.read(bands) if bands else src.read()
    return data.astype(np.float32)


def raster_to_tensor(path: str | Path, bands: list[int] | None = None) -> torch.Tensor:
    """Read raster and return a float32 tensor normalised to [0, 1]."""
    arr = read_raster(path, bands)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return torch.from_numpy(arr)


def check_crs(path: str | Path, expected_epsg: int = 4326) -> bool:
    """Return True if raster CRS matches expected_epsg, warn otherwise."""
    with rasterio.open(path) as src:
        actual = src.crs.to_epsg() if src.crs else None
    if actual != expected_epsg:
        print(f"Warning: {path} — CRS is EPSG:{actual}, expected EPSG:{expected_epsg}")
        return False
    return True


def get_raster_info(path: str | Path) -> dict:
    """Return a summary dict: crs, shape, dtype, bounds, nodata."""
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "shape": (src.count, src.height, src.width),
            "dtype": src.dtypes[0],
            "bounds": src.bounds,
            "nodata": src.nodata,
        }
