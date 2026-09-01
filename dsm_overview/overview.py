"""The whole check in one call: load once, then one figure and one row per crown.

The expensive part — resampling two full-survey rasters and fitting the
co-registration twice — happens once for the scene. Everything after that is a
cut-out, so asking for all eighteen crowns costs barely more than asking for
one.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from deadwood_spectral.grid import load_reference_grid
from deadwood_spectral.masks import load_crowns
from dsm_overview.plot3d import DEFAULT_AZIM, DEFAULT_ELEV, plot_dem_overview
from dsm_overview.stats import aoi_stats, stats_table
from dsm_overview.surfaces import load_surfaces
from dsm_overview.window import aoi_from_bounds

logger = logging.getLogger(__name__)


def run_dsm_overview(
    reference: str | Path,
    dsm: str | Path,
    dtm: str | Path,
    crowns: Sequence[str | Path],
    out_dir: str | Path,
    tree_ids: Sequence[str] | None = None,
    categories: Sequence[str] = ("soff",),
    buffer_m: float = 15.0,
    ring_gap_m: float = 2.0,
    ring_width_m: float = 8.0,
    height_threshold: float = 0.5,
    max_side: int = 200,
    local_blocks: int = 12,
    clamp_to_dsm: bool = False,
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
) -> dict[str, Path]:
    """Every artefact the check writes, keyed by name."""
    # The grid is read here for the crown reprojection and again inside
    # load_surfaces — two header reads, not two raster reads.
    gdf = load_crowns(crowns, load_reference_grid(reference))
    gdf = gdf[gdf["crown_category"].isin(list(categories))]
    if tree_ids is not None:
        # Coerced to str: gdf["tree_id"] is a pandas string column, but an
        # unquoted YAML list like `tree_ids: [4144]` parses as int. Comparing
        # int against str always misses, so every configured id would read as
        # missing even for a crown that plainly exists.
        tree_ids = [str(t) for t in tree_ids]
        missing = sorted(set(tree_ids) - set(gdf["tree_id"]))
        if missing:
            raise ValueError(f"no crown polygon for tree_id(s): {', '.join(missing)}")
        gdf = gdf[gdf["tree_id"].isin(list(tree_ids))]
    if gdf.empty:
        raise ValueError("no crown selected — check categories/tree_ids")
    gdf = gdf.sort_values("tree_id")

    surfaces = load_surfaces(reference, dsm, dtm, local_blocks, clamp_to_dsm)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    rows = []
    for tree_id, geometry in zip(gdf["tree_id"], gdf.geometry):
        aoi = aoi_from_bounds(geometry.bounds, surfaces.grid, buffer_m, str(tree_id))
        rows.append(aoi_stats(surfaces, aoi, geometry, ring_gap_m, ring_width_m))
        outputs[f"plot_{tree_id}"] = plot_dem_overview(
            surfaces,
            aoi,
            out_dir / f"dem_{tree_id}.png",
            height_threshold,
            max_side,
            geometry=geometry,
            elev=elev,
            azim=azim,
        )

    table = stats_table(rows)
    outputs["stats_csv"] = out_dir / "dem_offsets.csv"
    table.to_csv(outputs["stats_csv"], index=False)
    logger.info("wrote %s (%d rows)", outputs["stats_csv"], len(table))
    return outputs
