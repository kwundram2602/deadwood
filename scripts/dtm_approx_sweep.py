"""dtm_approx parameter sweep on one scene (one parameter at a time).

uv run python scripts/dtm_approx_sweep.py --config configs/dtm_approx_sweep/sweep.yaml [key=value ...]
uv run python scripts/dtm_approx_sweep.py --config configs/dtm_approx_sweep/sweep.yaml --smoke

--smoke: a centred 1000 x 1000 px crop and the first 3 runs, into <out_dir>/smoke/.
"""

import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explore_and_process.apply_dsm_mask import DtmApproxParams  # noqa: E402
from explore_and_process.dtm_approx_sweep import (  # noqa: E402
    load_scene,
    one_at_a_time,
    run_sweep,
    scene_inputs,
)

logger = logging.getLogger(__name__)

SMOKE_WINDOW_PX = 1000
SMOKE_RUNS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="dtm_approx parameter sweep on one scene.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--working_dir", default=".", help="config paths are relative to this (default: .)")
    parser.add_argument("--smoke", action="store_true", help="tiny crop and 3 runs, end to end")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides, e.g. sweep.date=20260210")
    args = parser.parse_args()
    os.chdir(args.working_dir)

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    sw = cfg.sweep
    prep = cast(DictConfig, OmegaConf.load(str(sw.preprocess_config)))
    # sweep.baseline overrides single values of the preprocess config's dtm_approx block
    override = OmegaConf.to_container(sw.baseline) if sw.get("baseline") else {}
    baseline = DtmApproxParams.from_cfg({**OmegaConf.to_container(prep.dsm_mask.dtm_approx), **override})
    runs = one_at_a_time(baseline, dict(sw["values"]))  # validates before any raster is read
    date = None if sw.get("date") is None else str(sw.date)

    dsm_path, process_out = scene_inputs(prep, date)
    out_root = Path(str(sw.out_dir)) / ("smoke" if args.smoke else "")
    if args.smoke:
        runs = runs[:SMOKE_RUNS]
    print(f"{len(runs)} run(s): {', '.join(r.name for r in runs)}")

    scene = load_scene(
        dsm_path,
        process_out,
        date or "scene",
        reference_run=sw.get("reference_run"),
        window_px=SMOKE_WINDOW_PX if args.smoke else None,
        zones=dict(sw.zones) if sw.get("zones") else None,
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    resolved = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    resolved |= {"baseline_dtm_approx": asdict(baseline), "smoke": args.smoke}
    run_sweep(
        scene,
        runs,
        float(prep.dsm_mask.max_ndsm_height),
        out_root / scene.date / stamp,
        metric=str(sw.get("metric", "rmse_m")),
        config=resolved,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
