"""Reduce the aligned time series to comparable spectral curves per class.

Reduce-on-read: the pixel set is fixed once, then every scene is streamed in
row chunks and collapsed to numbers immediately. Nothing intermediate is
written, so memory stays flat regardless of how many acquisitions arrive.

Object-wise detail exists only for the soff trees, which are the sure ground
truth and few enough to plot individually. `living` and `background` are
summarised as a class curve with an interquartile band — their per-object
spread would be the spread of the crown segmenter, not of the phenology.
"""

import calendar
import datetime as dt
import logging
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point

from deadwood_spectral.align import parse_date
from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid
from deadwood_spectral.indices import BAND_NAMES, INDEX_NAMES, compute_indices
from deadwood_spectral.masks import CLASS_NAMES, ClassMasks

logger = logging.getLogger(__name__)

# The index set plus raw NIR: a collapsed NIR is the most direct deadwood
# signature there is, and it is lost inside every normalised difference.
MEASURES: tuple[str, ...] = INDEX_NAMES + ("NIR",)
SEASONS: tuple[str, ...] = ("dry", "wet")


def stack_dates(stack_dir: str | Path) -> list[str]:
    """Every aligned acquisition in the directory, chronologically."""
    dates = sorted(parse_date(p) for p in Path(stack_dir).glob("*_stack.tif"))
    if not dates:
        raise FileNotFoundError(f"no aligned stack in {stack_dir}")
    logger.info("%d acquisition(s), %s..%s", len(dates), dates[0], dates[-1])
    return dates


def _shift_months(date: dt.date, months: int) -> dt.date:
    """`date` moved back by `months`, without inventing a 31st of February."""
    index = date.year * 12 + (date.month - 1) - months
    year, month0 = divmod(index, 12)
    day = min(date.day, calendar.monthrange(year, month0 + 1)[1])
    return dt.date(year, month0 + 1, day)


def window_dates(stack_dir: str | Path, label_date: str, window_months: int) -> list[str]:
    """Acquisitions in the half-open window (label_date - window_months, label_date].

    Half-open because the anchor is where the labels come from: the acquisition
    on the anchor date belongs in, the one exactly a window back belongs to the
    previous cycle. The anchor itself must have a scene — a label_date without
    an acquisition is a typo far more often than it is an intention, and a
    silent shift to the nearest neighbour would hide it.
    """
    anchor = dt.datetime.strptime(label_date, "%Y%m%d").date()
    start = _shift_months(anchor, int(window_months))
    available = stack_dates(stack_dir)
    if label_date not in available:
        raise FileNotFoundError(f"no aligned stack for the anchor date {label_date}")
    dates = [d for d in available if start < dt.datetime.strptime(d, "%Y%m%d").date() <= anchor]
    if not dates:
        raise FileNotFoundError(
            f"no aligned stack in ({start:%Y%m%d}, {label_date}] under {stack_dir}"
        )
    logger.info("window (%s, %s]: %d acquisition(s)", f"{start:%Y%m%d}", label_date, len(dates))
    return dates


def stack_paths(stack_dir: str | Path, dates: Sequence[str]) -> list[Path]:
    """Stack paths for the dates, checked for existence *before* the first read.

    A real run streams sixty ~800 MB scenes; a missing path noticed only
    afterwards throws that work away.
    """
    paths = [Path(stack_dir) / f"{date}_stack.tif" for date in dates]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing aligned stack(s): " + ", ".join(missing))
    return paths


def select_pixels(
    masks: ClassMasks,
    max_pixels_per_class: int = 50_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Fix the pixel set once: row, col, class, tree_id.

    The deadwood mask is taken whole — a few thousand pixels across eighteen
    trees. The two reference classes run to millions and are cut to a seeded
    random draw. Drawing once rather than per date is what makes the curves
    comparable across time: otherwise a step in a curve could equally well be a
    change in which pixels were asked.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for name in CLASS_NAMES:
        rows, cols = np.nonzero(getattr(masks, name))
        if rows.size == 0:
            logger.warning("class %s has no pixels", name)
            continue
        if name != "deadwood" and rows.size > max_pixels_per_class:
            keep = np.sort(rng.choice(rows.size, size=max_pixels_per_class, replace=False))
            rows, cols = rows[keep], cols[keep]
        frame = pd.DataFrame({"row": rows, "col": cols})
        frame["class"] = name
        frame["tree_id"] = (
            pd.Series(masks.tree_idx[rows, cols]).map(masks.tree_ids).astype("string")
            if name == "deadwood"
            else pd.Series([pd.NA] * len(frame), dtype="string")
        )
        frames.append(frame)

    pixels = pd.concat(frames, ignore_index=True)
    logger.info("pixel sample: %s", dict(pixels["class"].value_counts()))
    return pixels


def sample_points(pixels: pd.DataFrame, grid: ReferenceGrid) -> gpd.GeoDataFrame:
    """The drawn pixels as points on the reference grid, for inspection in QGIS.

    One point per sampled pixel at its centre, not a polygonised mask: the mask
    says where a class *could* have been drawn, this says where it *was*. For a
    seeded draw that difference is the whole question — a mask covering the
    whole survey tells you nothing about whether the 50 000 pixels landed evenly
    or piled into one corner.
    """
    xs, ys = grid.transform * (pixels["col"].to_numpy() + 0.5, pixels["row"].to_numpy() + 0.5)
    return gpd.GeoDataFrame(
        pixels.copy(),
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=grid.crs,
    )


def read_values(
    paths: Sequence[Path],
    grid: ReferenceGrid,
    rows: np.ndarray,
    cols: np.ndarray,
    names: Sequence[str],
    chunk_rows: int = 512,
) -> dict[str, np.ndarray]:
    """Gather the requested bands/indices for a pixel set -> {name: (n_px, n_dates)}.

    Reads in row chunks across all scenes at once, so a pixel's whole time
    series is assembled without ever holding a full scene in memory. `names`
    may mix raw band names with index names; both come out of the same
    per-chunk dictionary.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.shape != cols.shape:
        raise ValueError(f"rows/cols length mismatch: {rows.shape} vs {cols.shape}")
    if len(paths) == 0:
        raise ValueError("no stacks given")

    out = {name: np.full((rows.size, len(paths)), np.nan, dtype=np.float32) for name in names}
    order = np.argsort(rows, kind="stable")

    with ExitStack() as stack:
        sources = []
        for path in paths:
            src = stack.enter_context(rasterio.open(path))
            assert_matches_grid(src, grid, str(path))
            sources.append(src)

        for start in range(0, grid.height, chunk_rows):
            stop = min(start + chunk_rows, grid.height)
            sel = order[(rows[order] >= start) & (rows[order] < stop)]
            if sel.size == 0:
                continue
            local_rows, local_cols = rows[sel] - start, cols[sel]
            window = Window.from_slices((start, stop), (0, grid.width))
            for date_idx, src in enumerate(sources):
                block = src.read(window=window).astype(np.float32)
                band_names = [d or n for d, n in zip(src.descriptions, BAND_NAMES)]
                values = dict(zip(band_names, block))
                values.update(compute_indices(block, band_names))
                for name in names:
                    out[name][sel, date_idx] = values[name][local_rows, local_cols]
    return out


def _reduce(values: dict[str, np.ndarray], member: np.ndarray, dates: Sequence[str], measures):
    """Per-date median/quartiles/count over a subset of rows, in long form."""
    records = []
    for measure in measures:
        block = np.asarray(values[measure], dtype=np.float64)[member]
        finite = np.isfinite(block)
        counts = finite.sum(axis=0)
        # An all-NaN column is expected at a date with a data hole, not
        # exceptional; nanquantile would only warn about it.
        safe = np.where(finite, block, np.nan)
        quantiles = np.full((3, len(dates)), np.nan)
        has_data = counts > 0
        if has_data.any():
            quantiles[:, has_data] = np.nanquantile(safe[:, has_data], [0.25, 0.5, 0.75], axis=0)
        for date_idx, date in enumerate(dates):
            records.append(
                {
                    "date": date,
                    "measure": measure,
                    "n_valid_px": int(counts[date_idx]),
                    "median": quantiles[1, date_idx],
                    "q25": quantiles[0, date_idx],
                    "q75": quantiles[2, date_idx],
                }
            )
    return records


def class_table(
    values: dict[str, np.ndarray],
    pixels: pd.DataFrame,
    dates: Sequence[str],
    measures: Sequence[str] = MEASURES,
) -> pd.DataFrame:
    """One row per class x date x measure: median with its interquartile band."""
    records = []
    classes = pixels["class"].to_numpy()
    for name in CLASS_NAMES:
        member = classes == name
        if not member.any():
            continue
        for record in _reduce(values, member, dates, measures):
            records.append({"class": name, **record})
    return pd.DataFrame.from_records(records)


def tree_table(
    values: dict[str, np.ndarray],
    pixels: pd.DataFrame,
    dates: Sequence[str],
    measures: Sequence[str] = MEASURES,
) -> pd.DataFrame:
    """One row per soff tree x date x measure. Deadwood only, by design."""
    records = []
    # Kept as a pandas Series: the column is nullable, and comparing an object
    # array holding pd.NA raises rather than yielding False.
    tree_ids = pixels["tree_id"]
    for tree_id in sorted(tree_ids.dropna().unique()):
        member = (tree_ids == tree_id).fillna(False).to_numpy()
        for record in _reduce(values, member, dates, measures):
            records.append({"tree_id": tree_id, **record})
    return pd.DataFrame.from_records(records)


def season_of(date: str, dry_months: Sequence[int], wet_months: Sequence[int]) -> str:
    """'dry', 'wet' or 'transition' for an acquisition date.

    April and October belong to neither season: labelling a transition month as
    one or the other would blur the very contrast the signature is meant to show.
    """
    overlap = set(dry_months) & set(wet_months)
    if overlap:
        raise ValueError(f"dry and wet months overlap: {sorted(overlap)}")
    month = dt.datetime.strptime(date, "%Y%m%d").month
    if month in set(dry_months):
        return "dry"
    if month in set(wet_months):
        return "wet"
    return "transition"


def signature_table(
    values: dict[str, np.ndarray],
    pixels: pd.DataFrame,
    dates: Sequence[str],
    dry_months: Sequence[int],
    wet_months: Sequence[int],
    bands: Sequence[str] = BAND_NAMES,
) -> pd.DataFrame:
    """Mean reflectance per band, class and season — the spectrum proper.

    The time-series tables answer "when does it swing"; this one answers "what
    does it look like", by pooling every acquisition of a season into one curve
    across the seven bands.
    """
    seasons = np.array([season_of(date, dry_months, wet_months) for date in dates])
    classes = pixels["class"].to_numpy()
    records = []
    for name in CLASS_NAMES:
        member = classes == name
        if not member.any():
            continue
        for season in SEASONS:
            columns = seasons == season
            if not columns.any():
                continue
            for band in bands:
                block = np.asarray(values[band], dtype=np.float64)[np.ix_(member, columns)]
                finite = np.isfinite(block)
                records.append(
                    {
                        "class": name,
                        "season": season,
                        "band": band,
                        "n_valid_px": int(finite.sum()),
                        "mean": float(np.nanmean(block)) if finite.any() else np.nan,
                        "std": float(np.nanstd(block)) if finite.any() else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def run_overview(
    reference: str | Path,
    stack_dir: str | Path,
    crowns: Sequence[str | Path],
    crown_prediction: str | Path,
    out_dir: str | Path,
    crown_threshold: float = 0.5,
    erode_m: float = 0.10,
    erode_min_area_m2: float = 1.0,
    exclude_buffer_m: float = 1.0,
    edge_buffer_m: float = 0.25,
    max_pixels_per_class: int = 50_000,
    seed: int = 0,
    chunk_rows: int = 512,
    label_date: str | None = None,
    window_months: int = 12,
    dry_months: Sequence[int] = (5, 6, 7, 8, 9),
    wet_months: Sequence[int] = (11, 12, 1, 2, 3),
) -> dict[str, Path]:
    """The whole overview stage: masks, sample, reduce, write.

    Returns every artefact it wrote, keyed by name, so a caller can report the
    paths without reconstructing the naming scheme.
    """
    # Imported here rather than at module scope so the reduce can run on a node
    # without a matplotlib install getting in the way of the numbers.
    from deadwood_spectral.grid import load_reference_grid
    from deadwood_spectral.masks import binarize_crown_mask, build_masks, load_crowns
    from deadwood_spectral.plots import plot_signature, plot_timeseries

    grid = load_reference_grid(reference)
    logger.info("reference grid %s from %s", grid.shape, reference)

    gdf = load_crowns(crowns, grid)
    crown, valid = binarize_crown_mask(crown_prediction, grid, threshold=crown_threshold)
    masks = build_masks(
        crown,
        valid,
        gdf,
        grid,
        erode_m=erode_m,
        erode_min_area_m2=erode_min_area_m2,
        exclude_buffer_m=exclude_buffer_m,
        edge_buffer_m=edge_buffer_m,
    )

    pixels = select_pixels(masks, max_pixels_per_class=max_pixels_per_class, seed=seed)
    # No anchor means the whole time series. An anchor restricts the run to the
    # half-open window ending on it, which is how a stage that trains on labels
    # from one date has to see the data; a descriptive first look usually does
    # not want that restriction, so it is off by default.
    dates = (
        stack_dates(stack_dir)
        if label_date is None
        else window_dates(stack_dir, label_date, window_months)
    )
    paths = stack_paths(stack_dir, dates)
    # Bands and indices in one pass: the signature needs the raw reflectances,
    # the curves need the indices, and reading the scenes twice for that would
    # double the only expensive part of the stage.
    names = list(dict.fromkeys((*MEASURES, *BAND_NAMES)))
    values = read_values(
        paths, grid, pixels["row"].to_numpy(), pixels["col"].to_numpy(), names, chunk_rows
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = class_table(values, pixels, dates)
    trees = tree_table(values, pixels, dates)
    signature = signature_table(values, pixels, dates, dry_months, wet_months)

    sample_gpkg = out_dir / "sample_pixels.gpkg"
    sample_points(pixels, grid).to_file(sample_gpkg, driver="GPKG", layer="sample_pixels")
    logger.info("wrote %s (%d point(s))", sample_gpkg, len(pixels))

    outputs = {
        "sample_gpkg": sample_gpkg,
        "class_csv": out_dir / "overview_class.csv",
        "tree_csv": out_dir / "overview_tree.csv",
        "signature_csv": out_dir / "signature_class.csv",
    }
    csv_keys = ("class_csv", "tree_csv", "signature_csv")
    for table, key in zip((classes, trees, signature), csv_keys):
        path = outputs[key]
        table.to_csv(path, index=False)
        logger.info("wrote %s (%d rows)", path, len(table))

    for measure in MEASURES:
        outputs[f"plot_ts_{measure}"] = plot_timeseries(
            classes, measure, out_dir / f"ts_{measure}.png"
        )
    outputs["plot_signature"] = plot_signature(signature, out_dir / "signature.png")
    return outputs
