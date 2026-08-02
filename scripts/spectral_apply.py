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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral.apply import aggregate_objects, predict_scene  # noqa: E402
from deadwood_spectral.classify import load_model, variant_spec  # noqa: E402
from deadwood_spectral.grid import load_reference_grid  # noqa: E402

logger = logging.getLogger(__name__)


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
    class_raster = proba.argmax(axis=0).astype(np.uint8)

    _write(cfg.apply.prob_raster, proba, grid, "float32", np.nan)
    _write(cfg.apply.class_raster, class_raster, grid, "uint8", 255)

    with rasterio.open(cfg.paths.ndsm) as src:
        ndsm = src.read(1).astype(np.float32)

    objects = aggregate_objects(
        class_raster, proba, grid, ndsm=ndsm, min_object_m2=float(cfg.apply.min_object_m2)
    )
    Path(cfg.apply.objects).parent.mkdir(parents=True, exist_ok=True)
    objects.to_file(cfg.apply.objects, driver="GPKG")
    logger.info("%d deadwood object(s) -> %s", len(objects), cfg.apply.objects)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
