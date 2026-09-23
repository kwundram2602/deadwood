"""DSM/DTM co-registration check for the whole scene: five 3D panels, no CSV.

The per-crown counterpart is scripts/dsm_overview.py.

uv run python scripts/dsm_overview_scene.py --config configs/dsm_overview/dsm_overview_scene.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsm_overview.overview import run_scene_overview  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene-wide DSM/DTM co-registration check.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    outputs = run_scene_overview(
        reference=cfg.scene.reference,
        dsm=cfg.scene.dsm,
        dtm=cfg.scene.dtm,
        dtm_plane=cfg.scene.dtm_plane,
        dtm_aligned=cfg.scene.dtm_aligned,
        out_dir=cfg.scene.out_dir,
        height_threshold=cfg.plot.height_threshold,
        max_side=cfg.plot.max_side,
        elev=cfg.plot.get("elev", 45.0),
        azim=cfg.plot.get("azim", -60.0),
        label=cfg.scene.get("label", "whole scene"),
    )
    logger.info("wrote %s", outputs["plot_scene"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
