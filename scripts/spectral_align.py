"""Stage A: bring every time-series scene onto the crown-mask grid.

  uv run python scripts/spectral_align.py --config configs/spectral/align.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral.align import align_all  # noqa: E402
from deadwood_spectral.coreg import (  # noqa: E402
    DEFAULT_MAX_TILE_NAN_FRAC,
    DEFAULT_MIN_TILES,
    coreg_report,
)
from deadwood_spectral.grid import load_reference_grid  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align the time series to the reference grid.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    grid = load_reference_grid(cfg.align.reference)
    logger.info("Reference grid: %s at %s", grid.shape, cfg.align.reference)

    written = align_all(
        cfg.align.src_dir, grid, cfg.align.out_dir, force=bool(cfg.align.force)
    )
    logger.info("Aligned %d scene(s) into %s", len(written), cfg.align.out_dir)

    tiles = [tuple(t) for t in (cfg.coreg.tiles or [])]
    if not tiles:
        logger.warning(
            "coreg.tiles is empty — skipping the co-registration report. "
            "Pick stable, non-vegetated tiles in QGIS and list them as [x, y] pairs."
        )
        return
    report = coreg_report(
        cfg.align.out_dir,
        grid,
        tiles,
        tile_size_px=int(cfg.coreg.tile_size_px),
        reference_date=cfg.coreg.reference_date,
        max_shift_m=float(cfg.coreg.max_shift_m),
        max_tile_nan_frac=float(
            cfg.coreg.get("max_tile_nan_frac", DEFAULT_MAX_TILE_NAN_FRAC)
        ),
        min_tiles=int(cfg.coreg.get("min_tiles", DEFAULT_MIN_TILES)),
    )
    Path(cfg.coreg.report).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(cfg.coreg.report, index=False)
    logger.info("Co-registration report -> %s", cfg.coreg.report)
    logger.info("\n%s", report.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
