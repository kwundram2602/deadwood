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
from deadwood_spectral.masks import EXCLUDE_COVERAGE, load_crowns
from deadwood_spectral.plots import _save
from dsm_overview.plot3d import DEFAULT_AZIM, DEFAULT_ELEV, plot_dem_overview
from dsm_overview.stats import aoi_stats, stats_table
from dsm_overview.surfaces import load_surfaces, load_surfaces_from_stages
from dsm_overview.timeseries import coarse_grid, height_table, load_stack, timeseries_figure
from dsm_overview.window import aoi_from_bounds, scene_aoi
from explore_and_process.timeseries import _by_date

logger = logging.getLogger(__name__)


def run_dsm_overview(
    reference: str | Path,
    dsm: str | Path,
    dtm: str | Path,
    crowns: Sequence[str | Path],
    out_dir: str | Path,
    tree_ids: Sequence[str] | None = None,
    categories: Sequence[str] = ("soff",),
    exclude_coverage: Sequence[str] = EXCLUDE_COVERAGE,
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
    # Before the tree_ids filter, so an explicitly requested crown that is
    # excluded here fails the `missing` check below instead of vanishing.
    excluded = gdf[gdf["coverage"].isin(list(exclude_coverage))]
    if not excluded.empty:
        logger.info(
            "coverage %s excludes %d crown(s): %s",
            "/".join(exclude_coverage),
            len(excluded),
            ", ".join(sorted(excluded["tree_id"])),
        )
    gdf = gdf.drop(index=excluded.index)
    if tree_ids is not None:
        # Coerced to str: gdf["tree_id"] is a pandas string column, but an
        # unquoted YAML list like `tree_ids: [4144]` parses as int. Comparing
        # int against str always misses, so every configured id would read as
        # missing even for a crown that plainly exists.
        tree_ids = [str(t) for t in tree_ids]
        missing = sorted(set(tree_ids) - set(gdf["tree_id"]))
        if missing:
            raise ValueError(
                "no crown polygon for tree_id(s), or excluded by category/coverage: "
                + ", ".join(missing)
            )
        gdf = gdf[gdf["tree_id"].isin(list(tree_ids))]
    if gdf.empty:
        raise ValueError("no crown selected — check categories/tree_ids/exclude_coverage")
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


def run_scene_overview(
    reference: str | Path,
    dsm: str | Path,
    dtm: str | Path,
    dtm_plane: str | Path,
    dtm_aligned: str | Path,
    out_dir: str | Path,
    height_threshold: float = 0.5,
    max_side: int = 400,
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
    label: str = "whole scene",
) -> dict[str, Path]:
    """The same five panels, once, over the entire reference grid.

    The per-crown run asks whether the ground under *one* tree ends up at zero;
    this one asks what the co-registration did to the survey as a whole. There
    is no crown and therefore no ring, so there is nothing to tabulate either —
    the figure is the whole output.

    The DTM stages come from `apply_dsm_mask`'s `dtm_coreg` run directory
    rather than from a fresh fit; see `load_surfaces_from_stages`.
    """
    surfaces = load_surfaces_from_stages(reference, dsm, dtm, dtm_plane, dtm_aligned)
    aoi = scene_aoi(surfaces.grid)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "plot_scene": plot_dem_overview(
            surfaces,
            aoi,
            out_dir / "dem_scene.png",
            height_threshold,
            max_side,
            geometry=None,
            elev=elev,
            azim=azim,
            label=label,
        )
    }


def run_timeseries_overview(
    reference: str | Path | None,
    dsm_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "*_DSM_coreg.tif",
    dates: Sequence[str] | None = None,
    resolution_m: float = 0.5,
) -> dict[str, Path]:
    """Every date's raw DSM side by side: one CSV row per date, one figure.

    No DTM, no crown, no fit — the raw DSMs against each other, so survey
    offsets and height-system switches between campaigns stay visible. See
    `dsm_overview.timeseries` for what the two views measure.

    `reference` only sets extent and CRS of the common grid. None takes them
    from the first date's DSM: the footprints roughly agree, and the pixels
    valid on every date are all that is counted anyway.
    """
    found = _by_date(dsm_dir, pattern)
    if not found:
        raise ValueError(f"no DSM matching {pattern!r} in {dsm_dir}")
    if dates is not None:
        # str for the same reason as tree_ids above: an unquoted YAML date list
        # parses as int.
        dates = sorted(str(d) for d in dates)
        missing = [d for d in dates if d not in found]
        if missing:
            raise ValueError(
                f"no DSM matching {pattern!r} in {dsm_dir} for date(s): {', '.join(missing)}"
            )
        found = {d: found[d] for d in dates}

    reference = next(iter(found.values())) if reference is None else reference
    grid = coarse_grid(load_reference_grid(reference), resolution_m)
    logger.info("grid from %s", reference)
    logger.info("%d date(s) on a %s grid at %g m", len(found), grid.shape, resolution_m)
    table = height_table(list(found), load_stack(found, grid))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"stats_csv": out_dir / "dsm_heights.csv"}
    table.to_csv(outputs["stats_csv"], index=False, float_format="%.3f")
    logger.info("wrote %s (%d rows)", outputs["stats_csv"], len(table))
    outputs["plot"] = _save(timeseries_figure(table), out_dir / "dsm_heights.png")
    return outputs
