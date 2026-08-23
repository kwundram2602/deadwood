"""Stage B: Totholz aus der Kronen-Prediction und der Zeitreihe.

uv run python scripts/spectral_deadwood.py train   --config configs/spectral/deadwood.yaml
uv run python scripts/spectral_deadwood.py predict --config configs/spectral/deadwood.yaml
uv run python scripts/spectral_deadwood.py all     --config configs/spectral/deadwood.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import rasterio
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral import plots  # noqa: E402
from deadwood_spectral.grid import load_reference_grid  # noqa: E402
from deadwood_spectral.model import (  # noqa: E402
    aggregate_crowns,
    load_model,
    predict_crown_pixels,
    save_model,
    train,
    write_probability_raster,
)
from deadwood_spectral.phenology import (  # noqa: E402
    FEATURE_NAMES,
    measure_series,
    pixel_features,
    stack_paths,
    window_dates,
)
from deadwood_spectral.samples import (  # noqa: E402
    binarize_crown_mask,
    build_pools,
    draw_samples,
    load_crowns,
    sampling_fingerprint,
)

logger = logging.getLogger(__name__)


def _window(cfg):
    dates = window_dates(
        cfg.paths.stack_dir, str(cfg.window.label_date), int(cfg.window.window_months)
    )
    return dates, stack_paths(cfg.paths.stack_dir, dates)


def _sample_table(cfg, grid, dates) -> pd.DataFrame:
    """Stichprobe plus Features — aus dem Cache, wenn er zum Fenster passt.

    Der Cache ist eine Datei neben dem Modell, keine eigene Pipeline-Stufe.
    """
    cache = Path(cfg.paths.model_dir) / "samples.parquet"
    sampling_container = OmegaConf.to_container(cfg.sampling, resolve=True)
    assert isinstance(sampling_container, dict)
    params = dict(sampling_container)
    params["min_valid_dates"] = int(cfg.window.min_valid_dates)
    params["ndsm"] = str(cfg.paths.ndsm)
    params["crown_prediction"] = str(cfg.paths.crown_prediction)
    fingerprint = sampling_fingerprint(dates, params)

    if cache.exists():
        cached = pd.read_parquet(cache)
        if cached.attrs.get("fingerprint") == fingerprint:
            logger.info("reusing %s (%d rows)", cache, len(cached))
            return cached
        logger.info("%s was drawn with different parameters — redrawing", cache)

    crown, valid = binarize_crown_mask(
        cfg.paths.crown_prediction, grid, float(cfg.sampling.crown_threshold)
    )
    gdf = load_crowns(list(cfg.paths.crowns), grid)
    pools = build_pools(
        crown,
        valid,
        gdf,
        grid,
        erode_m=float(cfg.sampling.erode_m),
        exclude_buffer_m=float(cfg.sampling.exclude_buffer_m),
        edge_buffer_m=float(cfg.sampling.edge_buffer_m),
    )
    drawn = draw_samples(
        pools,
        gdf,
        grid,
        negative_ratio=float(cfg.sampling.negative_ratio),
        max_pixels_per_class=int(cfg.sampling.max_pixels_per_class),
        block_m=float(cfg.sampling.block_m),
        seed=int(cfg.sampling.seed),
    )
    features = pixel_features(
        stack_paths(cfg.paths.stack_dir, dates),
        grid,
        drawn["row"].to_numpy(),
        drawn["col"].to_numpy(),
        ndsm_path=cfg.paths.ndsm,
        chunk_rows=int(cfg.predict.chunk_rows),
        min_valid_dates=int(cfg.window.min_valid_dates),
    )
    table = pd.concat([drawn.reset_index(drop=True), features], axis=1)
    table.attrs["fingerprint"] = fingerprint
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache, index=False)
    logger.info("wrote %s (%d rows)", cache, len(table))
    return table


def run_train(cfg) -> None:
    grid = load_reference_grid(cfg.paths.reference)
    dates, _ = _window(cfg)
    table = _sample_table(cfg, grid, dates)

    result = train(
        table[list(FEATURE_NAMES)],
        table,
        seed=int(cfg.model.seed),
        n_splits=int(cfg.model.n_splits),
        n_estimators=int(cfg.model.n_estimators),
        permutation_repeats=int(cfg.model.permutation_repeats),
    )
    save_model(result, cfg.paths.model_dir)
    logger.info("\n%s", result["metrics"].to_string(index=False))

    out_dir = Path(cfg.paths.out_dir)
    plots.plot_importances(result["importances"], out_dir / "importances.png")
    plots.plot_precision_recall(result["y"], result["oof_proba"], out_dir / "precision_recall.png")

    # Der Phaenologie-Plot braucht den rohen Verlauf, nicht die Aggregate — auf
    # einer Teilstichprobe, weil die Kurve das Bild ist und nicht die Datenmenge.
    subset = table.groupby("class_name", group_keys=False).head(2000)
    series = measure_series(
        stack_paths(cfg.paths.stack_dir, dates),
        grid,
        subset["row"].to_numpy(),
        subset["col"].to_numpy(),
        chunk_rows=int(cfg.predict.chunk_rows),
    )
    plots.plot_phenology(series, subset["class_code"].to_numpy(), dates, out_dir / "phenology.png")


def run_predict(cfg) -> None:
    grid = load_reference_grid(cfg.paths.reference)
    dates, paths = _window(cfg)
    fitted = load_model(cfg.paths.model_dir)

    crown, _ = binarize_crown_mask(
        cfg.paths.crown_prediction, grid, float(cfg.sampling.crown_threshold)
    )
    rows, cols, proba = predict_crown_pixels(
        fitted,
        paths,
        grid,
        crown,
        ndsm_path=cfg.paths.ndsm,
        chunk_rows=int(cfg.predict.chunk_rows),
        min_valid_dates=int(cfg.window.min_valid_dates),
        pixel_batch=int(cfg.predict.pixel_batch),
    )

    out_dir = Path(cfg.paths.out_dir)
    write_probability_raster(rows, cols, proba, grid, out_dir / "p_deadwood.tif")

    with rasterio.open(cfg.paths.ndsm) as src:
        ndsm = src.read(1)
    crowns = aggregate_crowns(
        crown,
        rows,
        cols,
        proba,
        grid,
        ndsm=ndsm,
        dead_frac_threshold=float(cfg.predict.dead_frac_threshold),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    crowns.to_file(out_dir / "crowns.gpkg", driver="GPKG")
    logger.info("wrote %s — %s", out_dir / "crowns.gpkg", dict(crowns["label"].value_counts()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deadwood from the spectral time series.")
    parser.add_argument("stage", choices=["train", "predict", "all"])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--refresh-samples",
        action="store_true",
        help="Stichprobe neu ziehen, auch wenn der Cache passt.",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.refresh_samples:
        cache = Path(cfg.paths.model_dir) / "samples.parquet"
        if cache.exists():
            cache.unlink()
            logger.info("removed %s", cache)

    if args.stage in ("train", "all"):
        run_train(cfg)
    if args.stage in ("predict", "all"):
        run_predict(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
