"""Raw DSMs of every date side by side: height distribution and offset per date.

Finds survey offsets and height-system switches between campaigns. The per-crown
and whole-scene counterparts are scripts/dsm_overview.py and
scripts/dsm_overview_scene.py.

uv run python scripts/dsm_overview_timeseries.py --config configs/dsm_overview/dsm_overview_timeseries.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsm_overview.overview import run_timeseries_overview  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw DSM heights per date.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    dates = cfg.timeseries.get("dates")
    outputs = run_timeseries_overview(
        reference=cfg.timeseries.get("reference"),
        dsm_dir=cfg.timeseries.dsm_dir,
        out_dir=cfg.timeseries.out_dir,
        pattern=cfg.timeseries.get("pattern", "*_DSM_coreg.tif"),
        dates=None if dates is None else list(dates),
        resolution_m=cfg.timeseries.get("resolution_m", 0.5),
    )
    for name, path in outputs.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
