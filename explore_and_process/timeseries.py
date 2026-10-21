"""Select the dates of a time-series crown run and derive per-date stage configs.

Each date pairs one orthomosaic (OM) with the DSM of the same acquisition, by
the leading YYYYMMDD of their filenames. The DTM is static and shared by all
dates. Every date runs stages 1a, 1b and 2 of the crown pipeline on its own
grid; its tiles carry the date as stem prefix so the split can group by it.
"""

import copy
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from deadwood_spectral.align import parse_date
from utils.data import SPLIT_STRATEGIES, assign_dates


@dataclass(frozen=True)
class DateScene:
    date: str
    om: Path
    dsm: Path


def _by_date(directory: str | Path, pattern: str) -> dict[str, Path]:
    """Map YYYYMMDD -> file for every match of *pattern* in *directory*."""
    found: dict[str, Path] = {}
    for path in sorted(Path(directory).glob(pattern)):
        date = parse_date(path)
        if date in found:
            raise ValueError(f"two files for date {date}: {found[date].name}, {path.name}")
        found[date] = path
    return found


def select_dates(ts: DictConfig) -> list[DateScene]:
    """Pair OM and DSM by date and pick the dates of this run.

    Candidates are ``ts.dates`` if given (every listed date must have both an
    OM and a DSM), otherwise every date that has both, sorted. ``ts.max_dates``
    then keeps the first N candidates.
    """
    oms = _by_date(ts.om_dir, ts.om_pattern)
    dsms = _by_date(ts.dsm_dir, ts.dsm_pattern)

    explicit = ts.get("dates", None)
    if explicit is not None:
        dates = [str(d) for d in explicit]
        if len(set(dates)) != len(dates):
            raise ValueError(f"timeseries.dates contains duplicates: {dates}")
        missing = [d for d in dates if d not in oms or d not in dsms]
        if missing:
            raise ValueError(
                f"timeseries.dates without OM in {ts.om_dir} or DSM in {ts.dsm_dir}: {missing}"
            )
    else:
        dates = sorted(set(oms) & set(dsms))
        if not dates:
            raise ValueError(f"no date has both an OM in {ts.om_dir} and a DSM in {ts.dsm_dir}")

    max_dates = ts.get("max_dates", None)
    if max_dates is not None:
        if int(max_dates) < 1:
            raise ValueError(f"timeseries.max_dates must be >= 1 or null, got {max_dates}")
        dates = dates[: int(max_dates)]
    return [DateScene(d, oms[d], dsms[d]) for d in dates]


def date_config(cfg: DictConfig, scene: DateScene) -> DictConfig:
    """Copy of the pipeline config with all per-date input and output paths set.

    Intermediate rasters go to ``<timeseries.work_dir>/<date>/``, tiles to
    ``<tiling.out>/<date>/`` with the date as stem prefix.
    """
    dcfg = copy.deepcopy(cfg)
    work = os.path.join(str(cfg.timeseries.work_dir), scene.date)
    om = str(scene.om)

    r = dcfg.rasterize
    r.reference = om
    r.sources[0].path = om
    r.raster_dir = None
    r.out_mask = os.path.join(work, "masks", "crown_mask_intermediate.tif")
    r.out_image_dir = os.path.join(work, "images")

    d = dcfg.dsm_mask
    d.mask = r.out_mask
    d.dsm = str(scene.dsm)
    # apply_dsm_mask derives both from the mask path when unset
    d.out = None
    d.out_dsm = os.path.join(work, "ndsm", "dsm_ndsm.tif")

    t = dcfg.tiling
    t.image = os.path.join(r.out_image_dir, f"{scene.om.stem}_stack.tif")
    t.out = os.path.join(str(cfg.tiling.out), scene.date)
    t.prefix = scene.date
    return dcfg


def validate_timeseries_cfg(cfg: DictConfig, dates: list[str]) -> None:
    """Fail fast on a time-series config before any raster is read."""
    if len(cfg.rasterize.sources) != 1:
        raise ValueError("time-series mode needs exactly one rasterize.sources entry (the OM)")
    validate_label_cfg(cfg)

    sp = cfg.split
    if os.path.normpath(str(sp.input)) != os.path.normpath(str(cfg.tiling.out)):
        raise ValueError(
            f"split.input ({sp.input}) must be tiling.out ({cfg.tiling.out}): "
            "the split collects the per-date tile folders below it"
        )
    strategy = str(sp.get("strategy", "random"))
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split.strategy is {strategy!r}; expected one of {SPLIT_STRATEGIES}")
    explicit = sp.get("dates", None)
    if strategy == "by_date":
        # The real assignment runs after tiling; this only surfaces config errors early
        assign_dates(
            dates,
            explicit=explicit,
            train_ratio=float(sp.train_ratio),
            val_ratio=float(sp.val_ratio),
            seed=int(sp.seed),
        )


def validate_label_cfg(cfg: DictConfig) -> None:
    """Without polygons, crown labels exist only through crown_resolve_threshold."""
    if not cfg.rasterize.get("use_polygons", True) and (
        cfg.dsm_mask.get("crown_resolve_threshold", None) is None
    ):
        raise ValueError(
            "rasterize.use_polygons: false needs dsm_mask.crown_resolve_threshold — "
            "otherwise the mask contains no crown labels at all"
        )
