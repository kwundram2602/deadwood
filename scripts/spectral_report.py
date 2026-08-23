"""Stage B: sample, extract and describe.

  uv run python scripts/spectral_report.py --config configs/spectral/analysis.yaml
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral.coreg import flagged_dates  # noqa: E402
from deadwood_spectral.extract import (  # noqa: E402
    available_dates,
    extract_samples,
    ndsm_signature,
    samples_ndsm_reference_path,
    save_ndsm_reference,
)
from deadwood_spectral.grid import load_reference_grid  # noqa: E402
from deadwood_spectral.report import run_report  # noqa: E402
from deadwood_spectral.sampling import (  # noqa: E402
    binarize_crown_mask,
    build_pools,
    draw_samples,
    load_crowns,
)

logger = logging.getLogger(__name__)


def _excluded_dates(cfg) -> list[str]:
    report_path = Path(cfg.paths.coreg_report)
    if not cfg.extract.exclude_flagged_dates or not report_path.exists():
        return []
    excluded = flagged_dates(pd.read_csv(report_path, dtype={"date": str}))
    if excluded:
        logger.warning("excluding %d date(s) flagged by co-registration: %s",
                       len(excluded), excluded)
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample, extract and describe.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    grid = load_reference_grid(cfg.paths.reference)
    gdf = load_crowns(list(cfg.paths.crowns), grid)

    crown, valid = binarize_crown_mask(
        cfg.paths.crown_prediction, grid, threshold=float(cfg.sampling.crown_threshold)
    )
    pools = build_pools(
        crown, valid, gdf, grid,
        erode_m=float(cfg.sampling.erode_m),
        exclude_buffer_m=float(cfg.sampling.exclude_buffer_m),
        edge_buffer_m=float(cfg.sampling.edge_buffer_m),
    )
    samples = draw_samples(
        pools, gdf, grid,
        negative_ratio=float(cfg.sampling.negative_ratio),
        max_pixels_per_class=int(cfg.sampling.max_pixels_per_class),
        block_m=float(cfg.sampling.block_m),
        seed=int(cfg.sampling.seed),
    )

    excluded = _excluded_dates(cfg)
    dates = available_dates(cfg.paths.stack_dir, exclude=excluded)
    table = extract_samples(
        samples, cfg.paths.stack_dir, grid,
        dates=dates,
        ndsm_path=cfg.paths.ndsm,
        chunk_rows=int(cfg.extract.chunk_rows),
    )
    out_path = Path(cfg.paths.samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    logger.info("samples -> %s (%d rows, %d columns)", out_path, len(table), table.shape[1])

    # Which nDSM these samples carry. paths.ndsm is declared independently in
    # analysis.yaml and classify.yaml and two variants (metres, normalized)
    # sit on the same grid, so this sidecar is the only thing that can tell
    # apply.py it was handed the wrong one.
    reference_path = samples_ndsm_reference_path(out_path)
    save_ndsm_reference(ndsm_signature(cfg.paths.ndsm), reference_path)
    logger.info("nDSM identity -> %s", reference_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_report(table, dates, Path(cfg.paths.out_dir) / stamp)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
