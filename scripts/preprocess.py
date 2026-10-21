"""Run the full preprocessing pipeline from a single config file.

Usage:
    uv run python scripts/preprocess.py --config configs/preprocess.yaml --working_dir .
    # only rebuild the nDSM comparison (dtm_approx vs. dtm) from the last run's raw nDSMs:
    uv run python scripts/preprocess.py --config configs/preprocess.yaml --ndsm_compare_only

With a `timeseries` section (configs/preprocess/timeseries.yaml), stages 1a–2
run once per selected date and the split can group tiles by date.
"""
import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf

from data.channels import load_manifest
from explore_and_process.apply_dsm_mask import main as dsm_main
from explore_and_process.apply_dsm_mask import recompare_ndsm
from explore_and_process.rasterize_crowns import main as rasterize_main
from explore_and_process.tile_patches import main as tile_main
from explore_and_process.timeseries import (
    date_config,
    select_dates,
    validate_label_cfg,
    validate_timeseries_cfg,
)
from utils.data import assign_dates, compute_channel_stats, split_patches

logger = logging.getLogger(__name__)


def run_scene(cfg) -> tuple[int, dict | None]:
    """Stages 1a, 1b and 2 for one scene.

    Returns the number of kept tiles and the nDSM comparison metrics of stage
    1b (dtm_approx vs. dtm in [0,1]; None unless both nDSMs were computed).
    """
    print("=== Stage 1a: rasterize_crowns ===")
    rasterize_main(cfg.rasterize)

    print("\n=== Stage 1b: apply_dsm_mask ===")
    compare = dsm_main(cfg.dsm_mask)
    # _embed_params inside apply_dsm_mask.main() mutated cfg.dsm_mask.out / out_dsm in-place.
    # Propagate the actual (param-embedded) paths to the tiling stage.
    cfg.tiling.mask = cfg.dsm_mask.out
    if cfg.dsm_mask.out_dsm is not None:
        cfg.tiling.dsm = cfg.dsm_mask.out_dsm

    print("\n=== Stage 2: tile_patches ===")
    return tile_main(cfg.tiling), compare


def write_ndsm_compare_csv(rows: list[dict], path: Path) -> None:
    """One row per date: nDSM comparison metrics (dtm_approx vs. dtm, [0,1])."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] nDSM comparison per date (dtm_approx vs. dtm) -> {path}")
    for r in rows:
        print(f"  {r['date']}  [0,1] RMSE={r['rmse']:.4f} bias={r['bias']:+.4f}  |  "
              f"m RMSE={r['rmse_m']:.3f} MAE={r['mae_m']:.3f} bias={r['bias_m']:+.3f} "
              f"p95|err|={r['p95_abs_m']:.2f}")


def run_timeseries(cfg) -> dict[str, str]:
    """Stages 1a–2 once per selected date; returns date -> split for by_date splits.

    Tiles land in <tiling.out>/<date>/ with the date as stem prefix; the
    shared channels.json and dates.json (date, OM, DSM, kept tiles, nDSM
    comparison) go to <tiling.out>/, plus ndsm_compare.csv with one row of
    nDSM comparison metrics per date when stage 1b computed both nDSMs.
    """
    scenes = select_dates(cfg.timeseries)
    dates = [s.date for s in scenes]
    validate_timeseries_cfg(cfg, dates)
    print(f"[INFO] Time series: {len(scenes)} date(s): {', '.join(dates)}")

    tiles_root = Path(cfg.tiling.out)
    if tiles_root.exists():
        shutil.rmtree(tiles_root)
    tiles_root.mkdir(parents=True)

    records = []
    compare_rows = []
    for scene in scenes:
        print(f"\n##### Date {scene.date} #####")
        dcfg = date_config(cfg, scene)
        kept, compare = run_scene(dcfg)
        records.append({
            "date": scene.date,
            "om": str(scene.om),
            "dsm": str(scene.dsm),
            "mask": str(dcfg.dsm_mask.out),
            "ndsm": str(dcfg.dsm_mask.out_dsm),
            "tiles": kept,
            "ndsm_compare": compare,
        })
        if compare is not None:
            compare_rows.append({"date": scene.date, **compare})

    manifests = {s.date: tiles_root / s.date / "channels.json" for s in scenes}
    names = {d: load_manifest(m) for d, m in manifests.items()}
    if len({tuple(n) for n in names.values()}) != 1:
        raise RuntimeError(f"channel manifests differ between dates: {names}")
    shutil.copy2(manifests[dates[0]], tiles_root / "channels.json")
    (tiles_root / "dates.json").write_text(json.dumps(records, indent=2))
    if compare_rows:
        write_ndsm_compare_csv(compare_rows, tiles_root / "ndsm_compare.csv")

    sp = cfg.split
    if str(sp.get("strategy", "random")) != "by_date":
        return {}
    return assign_dates(
        dates,
        explicit=sp.get("dates", None),
        train_ratio=float(sp.train_ratio),
        val_ratio=float(sp.val_ratio),
        seed=int(sp.seed),
    )


def rerun_ndsm_compare(cfg) -> None:
    """Rebuild only the nDSM comparison (metrics, error maps, plots) of the last run.

    Reads the raw nDSMs stage 1b already wrote, per date in time-series mode;
    nothing else is recomputed. Updates ndsm_compare.csv and the ndsm_compare
    entries of dates.json in <tiling.out>/ as well.
    """
    max_h = float(cfg.dsm_mask.max_ndsm_height)
    if cfg.get("timeseries", None) is None:
        process_out = os.path.dirname(os.path.dirname(os.path.abspath(cfg.dsm_mask.mask)))
        recompare_ndsm(process_out, max_h)
        return

    rows = []
    for scene in select_dates(cfg.timeseries):
        print(f"\n##### Date {scene.date} #####")
        metrics = recompare_ndsm(os.path.join(str(cfg.timeseries.work_dir), scene.date), max_h)
        rows.append({"date": scene.date, **metrics})

    tiles_root = Path(cfg.tiling.out)
    tiles_root.mkdir(parents=True, exist_ok=True)
    write_ndsm_compare_csv(rows, tiles_root / "ndsm_compare.csv")
    dates_json = tiles_root / "dates.json"
    if dates_json.is_file():
        by_date = {r["date"]: {k: v for k, v in r.items() if k != "date"} for r in rows}
        records = json.loads(dates_json.read_text())
        for record in records:
            if record["date"] in by_date:
                record["ndsm_compare"] = by_date[record["date"]]
        dates_json.write_text(json.dumps(records, indent=2))


def run(config_path: str, ndsm_compare_only: bool = False) -> None:
    cfg = OmegaConf.load(config_path)
    if ndsm_compare_only:
        rerun_ndsm_compare(cfg)
        return
    logger.info("Full config:\n%s", OmegaConf.to_yaml(cfg))
    validate_label_cfg(cfg)

    if cfg.get("timeseries", None) is not None:
        date_splits = run_timeseries(cfg)
    else:
        run_scene(cfg)
        date_splits = None

    print("\n=== Stage 3: split patches ===")
    sp = cfg.split
    split_patches(
        Path(sp.input),
        Path(sp.output),
        train_ratio=float(sp.train_ratio),
        val_ratio=float(sp.val_ratio),
        seed=int(sp.seed),
        mode=str(sp.mode),
        strategy=str(sp.get("strategy", "random")),
        date_splits=date_splits,
    )

    print("[INFO] Computing per-channel normalisation stats from training split …")
    names = load_manifest(Path(sp.output) / "channels.json")
    stats = compute_channel_stats(Path(sp.output) / "train", names)
    stats_path = Path(sp.output) / "train_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    for name, m, s in zip(stats["names"], stats["mean"], stats["std"]):
        print(f"  {name:10s}  mean={m:.4f}  std={s:.4f}")
    print(f"[DONE] Stats saved to {stats_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to preprocess.yaml")
    p.add_argument("--working_dir", default=".",
                   help="Working directory — all config paths are relative to this (default: .)")
    p.add_argument("--ndsm_compare_only", action="store_true",
                   help="only rebuild the nDSM comparison (dtm_approx vs. dtm) from the raw "
                        "nDSMs of the last stage-1b run; nothing else is recomputed")
    args = p.parse_args()
    os.chdir(args.working_dir)
    run(args.config, ndsm_compare_only=args.ndsm_compare_only)
