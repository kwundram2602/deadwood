"""Stage B1: spectral overview of the aligned time series.

Compares the seasonal course of the soff deadwood crowns against the crown
model's living canopy and against bare ground, per index and per band.

uv run python scripts/spectral_overview.py --config configs/spectral/overview.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral.overview import run_overview  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral overview over the aligned time series.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    outputs = run_overview(
        reference=cfg.overview.reference,
        stack_dir=cfg.overview.stack_dir,
        crowns=list(cfg.overview.crowns),
        crown_prediction=cfg.overview.crown_prediction,
        ndsm=cfg.overview.get("ndsm"),
        out_dir=cfg.overview.out_dir,
        crown_threshold=cfg.sampling.crown_threshold,
        erode_m=cfg.sampling.erode_m,
        erode_min_area_m2=cfg.sampling.erode_min_area_m2,
        exclude_buffer_m=cfg.sampling.exclude_buffer_m,
        edge_buffer_m=cfg.sampling.edge_buffer_m,
        exclude_coverage=list(cfg.sampling.get("exclude_coverage", ["fc"])),
        max_pixels_per_class=cfg.sampling.max_pixels_per_class,
        seed=cfg.sampling.seed,
        chunk_rows=cfg.read.chunk_rows,
        label_date=cfg.window.label_date,
        window_months=cfg.window.window_months,
        dry_months=list(cfg.season.dry_months),
        wet_months=list(cfg.season.wet_months),
    )
    logger.info("wrote %d artefact(s) to %s", len(outputs), cfg.overview.out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
