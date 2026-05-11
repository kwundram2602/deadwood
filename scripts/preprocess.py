"""Run the full preprocessing pipeline from a single config file.

Usage:
    uv run python scripts/preprocess.py --config configs/preprocess.yaml --working_dir .
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf

from explore_and_process.apply_dsm_mask import main as dsm_main
from explore_and_process.rasterize_crowns import main as rasterize_main
from explore_and_process.tile_patches import main as tile_main

logger = logging.getLogger(__name__)


def run(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    logger.info("Full config:\n%s", OmegaConf.to_yaml(cfg))

    print("=== Stage 1a: rasterize_crowns ===")
    rasterize_main(cfg.rasterize)

    print("\n=== Stage 1b: apply_dsm_mask ===")
    dsm_main(cfg.dsm_mask)
    # _embed_params inside apply_dsm_mask.main() mutated cfg.dsm_mask.out / out_dsm in-place.
    # Propagate the actual (param-embedded) paths to the tiling stage.
    cfg.tiling.mask = cfg.dsm_mask.out
    if cfg.dsm_mask.out_dsm is not None:
        cfg.tiling.dsm = cfg.dsm_mask.out_dsm
    # else: cfg.tiling.dsm keeps the value set directly in [tiling] config

    print("\n=== Stage 2: tile_patches ===")
    tile_main(cfg.tiling)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to preprocess.yaml")
    p.add_argument("--working_dir", default=".",
                   help="Working directory — all config paths are relative to this (default: .)")
    args = p.parse_args()
    os.chdir(args.working_dir)
    run(args.config)
