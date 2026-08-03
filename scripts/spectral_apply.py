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
from deadwood_spectral.apply import (  # noqa: E402
    DEADWOOD_CODE,
    aggregate_objects,
    assert_labels_match_objects,
    predict_scene,
)
from deadwood_spectral.classify import load_model, variant_spec  # noqa: E402
from deadwood_spectral.extract import (  # noqa: E402
    NDSM_REFERENCE_FILE,
    assert_same_ndsm,
    load_ndsm_reference,
)
from deadwood_spectral.grid import load_reference_grid  # noqa: E402
from deadwood_spectral.labels import label_box, label_boxes, label_count  # noqa: E402
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

    # paths.ndsm is set independently in analysis.yaml (training) and
    # classify.yaml (inference), and the two nDSM variants on disk — metres
    # and normalized — share the reference grid, so a mix-up produces
    # silently wrong scene-wide predictions with no error. Refuse it here.
    ndsm_reference = load_ndsm_reference(Path(cfg.paths.model_dir) / NDSM_REFERENCE_FILE)
    if ndsm_reference is None:
        logger.warning(
            "%s carries no %s — cannot verify that paths.ndsm is the nDSM this "
            "model was trained on. Re-run scripts/spectral_report.py and "
            "scripts/spectral_classify.py to record it.",
            cfg.paths.model_dir, NDSM_REFERENCE_FILE,
        )
    else:
        assert_same_ndsm(ndsm_reference, cfg.paths.ndsm)
        logger.info("nDSM matches the one used for training")

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
    # aggregate_objects computed its own connected-component labels internally
    # and does not expose them; this recomputation is only guaranteed to line
    # up with objects.object_id because both calls are deterministic over the
    # same class_raster. Pin that coupling explicitly rather than trust it.
    assert_labels_match_objects(labels, objects)
    out_dir = Path(cfg.retrospect.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bounding boxes once, for the kept objects only. The previous loop rebuilt
    # a whole-scene `labels == object_id` mask for EVERY connected component
    # (including the sub-threshold ones aggregate_objects discarded), once per
    # cycle — ~59 ms per mask on the real grid, tens of thousands of components.
    counts, boxes = label_boxes(labels)
    kept_ids = [int(o) for o in objects["object_id"]]

    masks = {}
    validity_masks = {}
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
        # indicative-only product, so we don't invent an "unknown" verdict —
        # but the coverage is both logged here and threaded into
        # mortality_timing.csv below (first_dead_cycle_coverage/low_confidence)
        # so a reader of the CSV sees the uncertainty without the log.
        valid = cycle_class != _NODATA_CLASS
        validity_masks[str(cycle_name)] = valid
        for object_id in kept_ids:
            box = label_box(boxes, object_id)
            if box is None:
                continue
            footprint = labels[box] == object_id
            n_pixels = label_count(counts, object_id)
            if n_pixels and valid[box][footprint].mean() < 0.5:
                logger.warning(
                    "retrospect cycle %s: object %d has <50%% valid coverage "
                    "(nodata-heavy); its 'alive' verdict for this cycle is unreliable",
                    cycle_name, object_id,
                )
        logger.info("retrospect cycle %s done", cycle_name)

    timing = first_dead_cycle(masks, objects, labels, validity_masks=validity_masks)
    timing.to_csv(out_dir / "mortality_timing.csv", index=False)
    logger.info("mortality timing -> %s", out_dir / "mortality_timing.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
