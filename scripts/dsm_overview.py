"""DSM/DTM co-registration check: 3D panels and ground offsets per soff crown.

uv run python scripts/dsm_overview.py --config configs/dsm_overview/dsm_overview.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsm_overview.overview import run_dsm_overview  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSM/DTM co-registration check.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    tree_ids = cfg.overview.get("tree_ids")
    outputs = run_dsm_overview(
        reference=cfg.overview.reference,
        dsm=cfg.overview.dsm,
        dtm=cfg.overview.dtm,
        crowns=list(cfg.overview.crowns),
        out_dir=cfg.overview.out_dir,
        tree_ids=None if tree_ids is None else list(tree_ids),
        categories=list(cfg.overview.categories),
        buffer_m=cfg.aoi.buffer_m,
        ring_gap_m=cfg.aoi.ring_gap_m,
        ring_width_m=cfg.aoi.ring_width_m,
        height_threshold=cfg.aoi.height_threshold,
        max_side=cfg.plot.max_side,
    )
    logger.info("wrote %d artefact(s) to %s", len(outputs), cfg.overview.out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
