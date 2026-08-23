"""Stage A: bring every time-series scene onto the crown-mask grid.
  Although the the orginal files are DTM aligned, the need to be realigned since the crown
  tif is on 5cm spatial resolution
uv run python scripts/spectral_align.py --config configs/spectral/align.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral.align import align_all  # noqa: E402
from deadwood_spectral.grid import load_reference_grid  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align the time series to the reference grid.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    grid = load_reference_grid(cfg.align.reference)
    logger.info("Reference grid: %s at %s", grid.shape, cfg.align.reference)

    written = align_all(cfg.align.src_dir, grid, cfg.align.out_dir, force=bool(cfg.align.force))
    logger.info("Aligned %d scene(s) into %s", len(written), cfg.align.out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
