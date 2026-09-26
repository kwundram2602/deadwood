"""Fuse crown mask, deadwood mask and nDSM into combo codes and an ecological class map.

uv run python scripts/fuse_masks.py --config configs/fusion/fuse.yaml --working_dir .
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mask_fusion.run import run_fusion  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse crown, deadwood and nDSM into eco classes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--working_dir", default=".")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config).fuse
    wd = Path(args.working_dir)

    outputs = run_fusion(
        crown=wd / cfg.crown,
        deadwood=wd / cfg.deadwood,
        ndsm=wd / cfg.ndsm,
        height_m=list(cfg.height_m),
        classes=OmegaConf.to_container(cfg.classes),
        out_dir=wd / cfg.out_dir,
        out_stem=cfg.get("out_stem"),
        chunk_rows=cfg.get("chunk_rows", 512),
    )
    for kind, path in outputs.items():
        logger.info("%s -> %s", kind, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
