"""Deadwood prediction over every scene of a time-series folder.

Wraps scripts/raw_predict_deadwood.py: same model settings, same outputs, but
its own config — source names a folder (source.dir) instead of one scene. Every
raster under source.dir matching source.pattern is predicted in turn, and the
model is loaded once for all of them.

Scenes run one after another on the GPU: the 4 GB card holds a single
inference pass at batch_size 2, so running scenes side by side does not fit.
What does overlap is the CPU-bound input preparation (resampling + uint8
scaling): while scene n is on the GPU, a background thread prepares scene n+1.
At most two scenes are held in RAM at once. --no_prefetch turns this off.

Outputs land in out_dir/<scene parent folder>/, so the dates of a series stay
apart. A scene whose polygons or mask already exist is skipped unless
--overwrite is given, which makes an interrupted run resumable.

Usage:
    uv run python scripts/predict_deadwood_timeseries.py \\
        --config configs/predict/deadwood/raw_deadwood_timeseries.yaml \\
        --working_dir .

Scenes are ordered by sorted path, so date-prefixed names come out
chronologically; source.max_items N keeps only the N most recent of them.

CLI flags override config values when provided:
    --input_dir --pattern --max_items --weights --threshold --batch_size --out_dir
"""

import argparse
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pyogrio
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.raw_predict_deadwood import (  # noqa: E402
    load_model,
    output_paths,
    predict_scene,
    prepare_input,
)
from utils.device import get_device  # noqa: E402

DEFAULT_PATTERN = "**/*_OM_RGB_merged.tif"

# Files this pipeline writes itself; never treat them as input scenes.
_OUTPUT_SUFFIXES = ("_rgb8.tif", "_deadwood_mask.tif")


def find_scenes(
    input_dir: Path, pattern: str = DEFAULT_PATTERN, max_items: int | None = None
) -> list[Path]:
    """Matching rasters under input_dir, sorted by path (date prefixes -> chronological).

    max_items keeps only the last N, i.e. the most recent scenes of the series.
    """
    scenes = sorted(
        path
        for path in Path(input_dir).glob(pattern)
        if path.is_file() and not path.name.endswith(_OUTPUT_SUFFIXES)
    )
    if max_items is not None:
        if max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        scenes = scenes[-max_items:]
    return scenes


def scene_config(base_cfg, scene: Path) -> DictConfig:
    """A copy of base_cfg pointed at one scene, writing to out_dir/<scene parent>.

    raw_predict_deadwood reads the scene from source.path; the series-level keys
    (dir, pattern, max_items) stay in source but are not used past this point.
    """
    cfg = base_cfg.copy()
    cfg.source.path = str(scene)
    cfg.out_dir = str(Path(str(base_cfg.get("out_dir", "out/raw_deadwood"))) / scene.parent.name)
    return cfg


def is_done(cfg, root: Path) -> bool:
    """A scene counts as done once its polygons or its mask were written."""
    paths = output_paths(cfg, root)
    return paths["polygons"].exists() or paths["mask"].exists()


def run_timeseries(
    base_cfg,
    root: Path,
    scenes: list[Path],
    model,
    device,
    prefetch: bool = True,
    overwrite: bool = False,
) -> list[dict]:
    """Predict every scene in order and return one summary row per scene."""
    configs = [scene_config(base_cfg, scene) for scene in scenes]
    todo = [i for i, cfg in enumerate(configs) if overwrite or not is_done(cfg, root)]

    rows = [
        {"scene": scene.name, "status": "skipped", "polygons": None, "seconds": 0.0}
        for scene in scenes
    ]

    executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
    pending: Future | None = None
    try:
        for k, i in enumerate(todo):
            cfg = configs[i]
            print(f"\n[{k + 1}/{len(todo)}] {scenes[i]}")
            start = time.perf_counter()
            try:
                if pending is not None:
                    future, pending = pending, None
                    rgb8_path = future.result()
                else:
                    rgb8_path = prepare_input(cfg, root)

                # Queue the next scene's preparation before the GPU pass starts.
                if executor is not None and k + 1 < len(todo):
                    pending = executor.submit(prepare_input, configs[todo[k + 1]], root)

                poly_path = predict_scene(cfg, root, rgb8_path, model, device)
                n_polygons = pyogrio.read_info(poly_path)["features"] if poly_path.exists() else 0
                rows[i].update(status="ok", polygons=n_polygons)
            except Exception:
                traceback.print_exc()
                rows[i]["status"] = "failed"
            finally:
                rows[i]["seconds"] = time.perf_counter() - start
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    return rows


def print_summary(rows: list[dict]) -> None:
    width = max([len(r["scene"]) for r in rows] + [5])
    print(f"\n{'scene':<{width}}  {'status':<8}  {'polygons':>8}  {'time':>8}")
    for r in rows:
        polygons = "" if r["polygons"] is None else str(r["polygons"])
        print(f"{r['scene']:<{width}}  {r['status']:<8}  {polygons:>8}  {r['seconds']:>7.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict deadwood for every scene of a folder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", help="override source.dir")
    parser.add_argument("--pattern", help="override source.pattern")
    parser.add_argument("--max_items", type=int, help="override source.max_items")
    parser.add_argument("--working_dir", default=".")
    parser.add_argument("--weights", help="override weights")
    parser.add_argument("--threshold", type=float, help="override threshold")
    parser.add_argument("--batch_size", type=int, help="override batch_size")
    parser.add_argument("--out_dir", help="override out_dir")
    parser.add_argument("--overwrite", action="store_true", help="redo scenes already predicted")
    parser.add_argument(
        "--no_prefetch", action="store_true", help="prepare inputs strictly one at a time"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{args.config} must hold a mapping, not a list")
    if "deadwood_timeseries" not in cfg:
        raise ValueError(
            f"{args.config} has no deadwood_timeseries block — single-scene configs"
            " (raw_deadwood) belong to raw_predict_deadwood.py; see"
            " configs/predict/deadwood/raw_deadwood_timeseries.yaml"
        )
    cfg = cfg.deadwood_timeseries

    if args.input_dir:
        cfg.source.dir = args.input_dir
    if args.pattern:
        cfg.source.pattern = args.pattern
    if args.max_items is not None:
        cfg.source.max_items = args.max_items
    if args.weights:
        cfg.weights = args.weights
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.out_dir:
        cfg.out_dir = args.out_dir

    root = Path(args.working_dir).resolve()
    if cfg.source.get("dir") is None:
        raise ValueError("source.dir is not set (config or --input_dir)")
    input_dir = (root / str(cfg.source.dir)).resolve()
    pattern = str(cfg.source.get("pattern", DEFAULT_PATTERN))
    scenes = find_scenes(input_dir, pattern, cfg.source.get("max_items"))
    if not scenes:
        raise SystemExit(f"no scenes matching {pattern!r} under {input_dir}")
    print(f"{len(scenes)} scene(s) under {input_dir}")

    device = get_device()
    print(f"Loading weights: {cfg.weights}")
    model = load_model((root / str(cfg.weights)).resolve(), device)

    rows = run_timeseries(
        cfg, root, scenes, model, device, prefetch=not args.no_prefetch, overwrite=args.overwrite
    )
    print_summary(rows)
    if any(r["status"] == "failed" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
