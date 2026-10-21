"""Parameter sweep for the DSM-only terrain approximation (dtm_approx) on one scene.

dtm_approx runs on its own here: the scene's DSM goes in, `approximate_dtm`
builds the terrain, and nDSM = DSM - terrain is compared against a fixed
reference, the external-DTM nDSM that stage 1b (apply_dsm_mask) already wrote
to <process_out>/ndsm_in_m/<run>/*_dtm_raw_m.tif. Its grid is the scene grid;
the DSM is resampled onto it exactly as stage 1b does. No mask, tiling or
co-registration is recomputed.

Search: one parameter at a time around the baseline (the preprocess config's
dsm_mask.dtm_approx block). Every run is scored with `compare_ndsm`; the
target metric (default rmse_m) ranks them. Output, one folder per sweep:

  <out_dir>/<date>/<YYYYMMDD-HHMMSS>/
    config.yaml         resolved sweep config (+ the baseline it started from)
    meta.json           git commit/dirty, host, versions, times, inputs
    results_runs.csv    one row per run in run order, appended as runs finish
    results.csv         the same rows sorted by the metric, with delta to baseline
    metrics.json        baseline and best run
    (zones)             with sweep.zones, every row also has rmse_m / mae_m / bias_m /
                        n_valid per zone (suffix _<zone>) and outside all zones (_rest),
                        so a gain in one area can be weighed against a loss elsewhere
    best_params.yaml    dtm_approx block of the best run, ready for the preprocess config
    sensitivity.png     metric over each parameter's values, baseline marked
    plots/<run>.png     two-row comparison plot ([0,1] and m) per run
    maps/best_*.tif     error map (m) and terrain of the best run
"""

import csv
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from explore_and_process.apply_dsm_mask import (
    DtmApproxParams,
    _plot_ndsm_compare,
    _save_float_tif,
    approximate_dtm,
    compare_ndsm,
    resample_raster,
)
from explore_and_process.timeseries import select_dates

PARAM_NAMES: tuple[str, ...] = tuple(f.name for f in fields(DtmApproxParams))
METRIC_KEYS = (
    "rmse_m", "mae_m", "bias_m", "p95_abs_m", "mse_m",
    "rmse", "mae", "bias", "mse", "n_valid", "ceiling_dtm", "ceiling_da",
)  # fmt: skip
REFERENCE_SUFFIX = "_dtm_raw_m.tif"
ZONE_METRICS = ("rmse_m", "mae_m", "bias_m", "n_valid")


@dataclass(frozen=True)
class SweepRun:
    name: str  # "baseline" or "<param>=<value>"
    varied: str | None  # the parameter changed against the baseline
    params: DtmApproxParams


@dataclass
class Scene:
    """One scene on the reference grid: DSM, reference nDSM (m) and grid metadata."""

    date: str
    dsm: np.ndarray
    reference: np.ndarray  # external-DTM nDSM (m), NaN where DSM or DTM has no data
    gsd: float
    profile: dict
    dsm_path: str
    reference_path: str
    zones: dict[str, np.ndarray] = field(default_factory=dict)  # name -> pixel mask


def zone_metrics(error_m: np.ndarray, zones: dict[str, np.ndarray]) -> dict:
    """rmse_m, mae_m, bias_m, n_valid of the metre error inside every zone and outside all (rest)."""
    if not zones:
        return {}
    outside = ~np.logical_or.reduce(list(zones.values()))
    out = {}
    for name, mask in {**zones, "rest": outside}.items():
        e = error_m[mask & np.isfinite(error_m)].astype(np.float64)
        nan = float("nan")
        out |= {
            f"rmse_m_{name}": float(np.sqrt(np.mean(e**2))) if e.size else nan,
            f"mae_m_{name}": float(np.mean(np.abs(e))) if e.size else nan,
            f"bias_m_{name}": float(np.mean(e)) if e.size else nan,
            f"n_valid_{name}": int(e.size),
        }
    return out


def _rasterize_zones(paths: dict, shape: tuple[int, int], transform, crs) -> dict[str, np.ndarray]:
    """Every polygon file (name -> path) burnt onto the scene grid as a boolean mask."""
    zones = {}
    for name, path in paths.items():
        if name == "rest":
            raise ValueError("sweep.zones: 'rest' is reserved for the area outside all zones")
        gdf = gpd.read_file(path).to_crs(crs)
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty],
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
        ).astype(bool)
        if not mask.any():
            raise ValueError(f"sweep.zones.{name}: {path} does not overlap the scene")
        zones[str(name)] = mask
        print(f"  zone {name}: {int(mask.sum()):,} px from {path}")
    return zones


def one_at_a_time(baseline: DtmApproxParams, values: dict) -> list[SweepRun]:
    """The baseline, then one run per alternative value of each parameter.

    Every run differs from the baseline in exactly one parameter; values equal
    to the baseline and repeated values are dropped. Each run is validated by
    DtmApproxParams.from_cfg (ranges, types).
    """
    unknown = sorted(set(values) - set(PARAM_NAMES))
    if unknown:
        raise ValueError(f"sweep.values: unknown dtm_approx parameters {unknown}; expected {list(PARAM_NAMES)}")
    runs = [SweepRun("baseline", None, baseline)]
    seen = {baseline}
    for name, options in values.items():
        for value in options:
            params = DtmApproxParams.from_cfg({**asdict(baseline), name: value})
            if params in seen:
                continue
            seen.add(params)
            runs.append(SweepRun(f"{name}={getattr(params, name):g}", name, params))
    return runs


def _latest_reference(process_out: Path, run_id: str | None) -> Path:
    """The external-DTM nDSM (m) of stage-1b run `run_id`, default the latest run that has one."""
    runs_dir = process_out / "ndsm_in_m"
    runs = [runs_dir / run_id] if run_id else sorted(runs_dir.glob("*"), reverse=True)
    for run in runs:
        found = sorted(run.glob(f"*{REFERENCE_SUFFIX}"))
        if found:
            return found[0]
    raise FileNotFoundError(
        f"no *{REFERENCE_SUFFIX} in {runs_dir}{'/' + run_id if run_id else ''}: run stage 1b with "
        "dsm_mask.method or ndsm_channel 'dtm' for this scene first"
    )


def scene_inputs(prep_cfg: DictConfig, date: str | None) -> tuple[str, Path]:
    """(DSM path, stage-1b process_out folder) of one scene of a preprocess config.

    Time-series configs pick the date's DSM from timeseries.dsm_dir; a
    single-scene config uses dsm_mask.dsm and the folder above the mask.
    """
    if prep_cfg.get("timeseries", None) is not None:
        if date is None:
            raise ValueError("sweep.date is required for a time-series preprocess config")
        ts = OmegaConf.merge(prep_cfg.timeseries, {"dates": [str(date)], "max_dates": None})
        assert isinstance(ts, DictConfig)
        (scene,) = select_dates(ts)
        return str(scene.dsm), Path(str(prep_cfg.timeseries.work_dir)) / scene.date
    mask = os.path.abspath(str(prep_cfg.dsm_mask.mask))
    return str(prep_cfg.dsm_mask.dsm), Path(os.path.dirname(os.path.dirname(mask)))


def load_scene(
    dsm_path: str,
    process_out: Path,
    date: str,
    reference_run: str | None = None,
    window_px: int | None = None,
    zones: dict | None = None,
) -> Scene:
    """Reference nDSM and the DSM resampled onto its grid.

    `window_px` crops a centred square of that size (smoke tests); the DSM is
    then resampled onto the cropped grid directly. `zones` (name -> polygon
    file) are rasterized onto the same grid.
    """
    ref_path = _latest_reference(process_out, reference_run)
    with rasterio.open(ref_path) as src:
        window = Window(0, 0, src.width, src.height)
        if window_px is not None:
            size_r, size_c = min(window_px, src.height), min(window_px, src.width)
            window = Window((src.width - size_c) // 2, (src.height - size_r) // 2, size_c, size_r)
        reference = src.read(1, window=window).astype(np.float32)
        transform = window_transform(window, src.transform)
        profile = src.profile.copy()
        crs = src.crs
    h, w = reference.shape
    profile.update(height=h, width=w, transform=transform)
    print(f"Scene {date}: grid {h} x {w} at {abs(transform.a):.3f} m, reference {ref_path}")
    dsm = resample_raster(dsm_path, h, w, transform, crs)
    masks = _rasterize_zones(zones or {}, (h, w), transform, crs)
    return Scene(date, dsm, reference, abs(transform.a), profile, dsm_path, str(ref_path), masks)


def run_one(scene: Scene, params: DtmApproxParams, max_ndsm_height: float):
    """dtm_approx with `params` on the scene; returns (terrain, nDSM, NdsmComparison)."""
    terrain = approximate_dtm(scene.dsm, scene.gsd, params)
    ndsm = (scene.dsm - terrain).astype(np.float32)
    return terrain, ndsm, compare_ndsm(scene.reference, ndsm, scene.dsm, max_ndsm_height)


def _row(run: SweepRun, metrics: dict, zones: dict, seconds: float) -> dict:
    value = getattr(run.params, run.varied) if run.varied else None
    return {
        "name": run.name,
        "varied": run.varied or "",
        "value": "" if value is None else value,
        **asdict(run.params),
        **{k: metrics[k] for k in METRIC_KEYS},
        **zones,
        "seconds": round(seconds, 1),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_sensitivity(rows: list[dict], metric: str, path: Path) -> None:
    """One panel per varied parameter: metric and bias_m over its values, baseline starred."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    base = next(r for r in rows if r["name"] == "baseline")
    varied = list(dict.fromkeys(r["varied"] for r in rows if r["varied"]))
    if not varied:
        return
    cols = min(4, len(varied))
    nrows = -(-len(varied) // cols)
    fig = Figure(figsize=(4.2 * cols, 3.4 * nrows), layout="constrained")
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(nrows, cols, squeeze=False)).ravel()
    for ax, name in zip(axes, varied):
        pts = sorted([(r[name], r[metric], r["bias_m"]) for r in rows if r["varied"] == name] + [
            (base[name], base[metric], base["bias_m"])
        ])
        x, y, bias = zip(*pts)
        ax.plot(x, y, "o-", label=metric)
        ax.plot(x, bias, "s--", color="tab:orange", label="bias_m")
        ax.plot([base[name]], [base[metric]], "*", color="black", markersize=14, label="baseline")
        ax.axhline(0.0, color="grey", linewidth=0.6)
        ax.set_xlabel(name)
        ax.set_ylabel("m")
        ax.grid(alpha=0.3)
    for ax in axes[len(varied):]:
        ax.set_axis_off()
    axes[0].legend(fontsize=8)
    fig.suptitle(f"dtm_approx one-at-a-time sensitivity — {metric} (lower is better)")
    fig.savefig(path, dpi=110)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _meta(scene: Scene, n_runs: int, started: datetime) -> dict:
    import scipy

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": socket.gethostname(),
        "started": started.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "rasterio": rasterio.__version__,
        "date": scene.date,
        "dsm": scene.dsm_path,
        "reference": scene.reference_path,
        "grid": list(scene.dsm.shape),
        "gsd_m": scene.gsd,
        "n_runs": n_runs,
    }


def run_sweep(
    scene: Scene,
    runs: list[SweepRun],
    max_ndsm_height: float,
    out_dir: Path,
    metric: str = "rmse_m",
    config: dict | None = None,
) -> list[dict]:
    """Run every SweepRun on the scene and write the run folder (see module docstring).

    Returns the result rows sorted by `metric` (ascending).
    """
    allowed = list(METRIC_KEYS) + list(zone_metrics(np.zeros(1), {z: np.ones(1, bool) for z in scene.zones}))
    if metric not in allowed:
        raise ValueError(f"sweep.metric {metric!r}; expected one of {allowed}")
    if not runs or runs[0].name != "baseline":
        raise ValueError("the first run must be the baseline")
    started = datetime.now()
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "plots").mkdir()
    OmegaConf.save(OmegaConf.create(config or {}), out_dir / "config.yaml")
    meta = _meta(scene, len(runs), started)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    rows: list[dict] = []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {run.name}")
        t0 = time.perf_counter()
        _, ndsm, cmp = run_one(scene, run.params, max_ndsm_height)
        seconds = time.perf_counter() - t0
        rows.append(_row(run, cmp.metrics, zone_metrics(cmp.error_m, scene.zones), seconds))
        _write_csv(rows, out_dir / "results_runs.csv")
        _plot_ndsm_compare(scene.reference, ndsm, cmp, str(out_dir / "plots" / f"{i:02d}_{run.name}.png"))
        m = cmp.metrics
        zone_text = "".join(
            f"  rmse_m_{z}={rows[-1][f'rmse_m_{z}']:.3f}" for z in [*scene.zones, "rest"] if scene.zones
        )
        print(f"  rmse_m={m['rmse_m']:.3f}  mae_m={m['mae_m']:.3f}  bias_m={m['bias_m']:+.3f}"
              f"{zone_text}  ({seconds:.0f} s)")
        if i == 1 and len(runs) > 1:
            print(f"  -> about {seconds * (len(runs) - 1) / 60:.0f} min for the remaining {len(runs) - 1} run(s)")
        del ndsm, cmp

    base = rows[0]
    ranked = sorted(rows, key=lambda r: r[metric])
    for r in ranked:
        r[f"delta_{metric}"] = r[metric] - base[metric]
    _write_csv(ranked, out_dir / "results.csv")
    _plot_sensitivity(rows, metric, out_dir / "sensitivity.png")

    best = ranked[0]
    best_run = next(r for r in runs if r.name == best["name"])
    (out_dir / "best_params.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create({"dtm_approx": asdict(best_run.params)}))
    )
    (out_dir / "metrics.json").write_text(
        json.dumps({"metric": metric, "baseline": base, "best": best}, indent=2)
    )
    if best_run.name != "baseline":
        terrain, _, cmp = run_one(scene, best_run.params, max_ndsm_height)
        nan = float("nan")
        _save_float_tif(cmp.error_m, str(out_dir / "maps" / "best_err_da_minus_dtm_m.tif"),
                        scene.profile, label="best error map (m)", nodata=nan)
        _save_float_tif(terrain, str(out_dir / "maps" / "best_dtm_approx.tif"),
                        scene.profile, label="best dtm_approx", nodata=nan)

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nBest by {metric}: {best['name']}  {metric}={best[metric]:.3f} "
          f"(baseline {base[metric]:.3f})  -> {out_dir}")
    return ranked
