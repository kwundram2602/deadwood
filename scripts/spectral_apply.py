"""Stage C, part 2: apply the classifier to the whole AOI.

  uv run python scripts/spectral_apply.py --config configs/spectral/classify.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf
from skimage.measure import label as cc_label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral.apply import DEADWOOD_CODE, aggregate_objects, predict_scene  # noqa: E402
from deadwood_spectral.classify import load_model, variant_spec  # noqa: E402
from deadwood_spectral.grid import load_reference_grid  # noqa: E402
from deadwood_spectral.retrospect import first_dead_cycle  # noqa: E402

logger = logging.getLogger(__name__)

_NODATA_CLASS = 255


def class_raster_from_proba(proba: np.ndarray) -> np.ndarray:
    """argmax over classes; non-finite (off-footprint) pixels stay unknown.

    Precedent: scripts/predict.py's `binarize` sets invalid pixels to a 255
    sentinel via an explicit valid mask rather than letting them fall out of
    an argmax. `proba` already carries NaN at pixels predict_scene could not
    classify, so validity is read straight off it rather than reconstructed
    from a separate source.
    """
    valid = np.isfinite(proba).all(axis=0)
    out = np.full(proba.shape[1:], _NODATA_CLASS, dtype=np.uint8)
    out[valid] = np.argmax(proba[:, valid], axis=0).astype(np.uint8)
    return out


def _write(path, data, grid, dtype, nodata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis]
    profile = dict(
        driver="GTiff", dtype=dtype, width=grid.width, height=grid.height,
        count=data.shape[0], crs=grid.crs, transform=grid.transform, nodata=nodata,
        compress="lzw", tiled=True, blockxsize=512, blockysize=512,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the deadwood classifier to the scene.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    grid = load_reference_grid(cfg.paths.reference)
    model, features = load_model(cfg.paths.model_dir)
    dates = [str(d) for d in cfg.classify.cycle.dates]
    baseline = str(cfg.classify.baseline_date) if cfg.classify.baseline_date else dates[-1]
    variant_dates, switches = variant_spec(str(cfg.classify.primary_variant), dates, baseline)

    proba = predict_scene(
        cfg.paths.stack_dir, variant_dates, grid, model, features, cfg.paths.ndsm, switches,
        tile_size=int(cfg.apply.tile_size), stride=int(cfg.apply.stride),
    )
    class_raster = class_raster_from_proba(proba)

    _write(cfg.apply.prob_raster, proba, grid, "float32", np.nan)
    _write(cfg.apply.class_raster, class_raster, grid, "uint8", _NODATA_CLASS)

    with rasterio.open(cfg.paths.ndsm) as src:
        ndsm = src.read(1).astype(np.float32)

    objects = aggregate_objects(
        class_raster, proba, grid, ndsm=ndsm, min_object_m2=float(cfg.apply.min_object_m2)
    )
    Path(cfg.apply.objects).parent.mkdir(parents=True, exist_ok=True)
    objects.to_file(cfg.apply.objects, driver="GPKG")
    logger.info("%d deadwood object(s) -> %s", len(objects), cfg.apply.objects)

    if not cfg.retrospect.enabled:
        return

    labels = cc_label(class_raster == DEADWOOD_CODE, connectivity=2)
    out_dir = Path(cfg.retrospect.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = {}
    for cycle_name, cycle_dates in cfg.retrospect.cycles.items():
        cycle_dates = [str(d) for d in cycle_dates]
        if len(cycle_dates) != len(variant_dates):
            raise ValueError(
                f"retrospect cycle {cycle_name!r} has {len(cycle_dates)} dates; "
                f"the model needs exactly {len(variant_dates)} dates (the same "
                "count as the training cycle) because the feature vector has a "
                "fixed length"
            )
        cycle_proba = predict_scene(
            cfg.paths.stack_dir, cycle_dates, grid, model, features, cfg.paths.ndsm, switches,
            tile_size=int(cfg.apply.tile_size), stride=int(cfg.apply.stride),
        )
        cycle_class = class_raster_from_proba(cycle_proba)
        _write(out_dir / f"class_{cycle_name}.tif", cycle_class, grid, "uint8", _NODATA_CLASS)
        masks[str(cycle_name)] = cycle_class == DEADWOOD_CODE

        # class_raster_from_proba leaves non-finite pixels at the 255 nodata
        # sentinel, which reads as "not dead" in first_dead_cycle's majority
        # vote. For an object whose footprint is mostly nodata in this cycle
        # (e.g. a cloud/shadow gap in the older imagery), that silently biases
        # the object toward "alive" instead of "unknown". This is a bonus,
        # indicative-only product, so we don't try to represent "unknown" in
        # the output — but we do surface it in the log rather than hide it.
        valid = cycle_class != _NODATA_CLASS
        for object_id in np.unique(labels[labels > 0]):
            footprint = labels == object_id
            n_pixels = int(footprint.sum())
            if n_pixels and valid[footprint].mean() < 0.5:
                logger.warning(
                    "retrospect cycle %s: object %d has <50%% valid coverage "
                    "(nodata-heavy); its 'alive' verdict for this cycle is unreliable",
                    cycle_name, int(object_id),
                )
        logger.info("retrospect cycle %s done", cycle_name)

    timing = first_dead_cycle(masks, objects, labels)
    timing.to_csv(out_dir / "mortality_timing.csv", index=False)
    logger.info("mortality timing -> %s", out_dir / "mortality_timing.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
