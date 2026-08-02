"""The reference-grid contract.

Every raster in the spectral pipeline lives on exactly one grid: the crown
mask's. Alignment, extraction and inference all assert against it, so the
comparison lives here rather than being re-implemented per module.
"""

from dataclasses import dataclass
from pathlib import Path

import rasterio
from affine import Affine
from rasterio.crs import CRS

# Sub-millimetre: tighter than any real georeference error, loose enough for
# float round-trips through the GeoTIFF header.
TRANSFORM_TOL = 1e-6


@dataclass(frozen=True)
class ReferenceGrid:
    height: int
    width: int
    transform: Affine
    crs: CRS

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def load_reference_grid(path: str | Path) -> ReferenceGrid:
    """Read the grid definition from the reference raster (the crown mask)."""
    with rasterio.open(path) as src:
        return ReferenceGrid(src.height, src.width, src.transform, src.crs)


def assert_matches_grid(src, grid: ReferenceGrid, context: str) -> None:
    """Fail unless an open raster sits on exactly the reference grid."""
    if (src.height, src.width) != grid.shape:
        raise ValueError(
            f"{context}: shape {(src.height, src.width)} != reference {grid.shape}"
        )
    if src.crs != grid.crs:
        raise ValueError(f"{context}: CRS {src.crs} != reference {grid.crs}")
    deltas = [abs(a - b) for a, b in zip(tuple(src.transform), tuple(grid.transform))]
    if max(deltas) > TRANSFORM_TOL:
        raise ValueError(
            f"{context}: transform {tuple(src.transform)} != reference "
            f"{tuple(grid.transform)} (max delta {max(deltas):.3g})"
        )
