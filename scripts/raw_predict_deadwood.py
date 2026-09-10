"""Deadwood prediction with the pretrained deadtrees.earth model, no fine-tuning.

Runs the published SegFormer-B5/UNet deadwood model
(https://data2.deadtrees.earth/assets/v1/segformer_b5_full_epoch_100.safetensors)
over one of our orthomosaics and writes the predicted deadwood polygons. This
is the "raw" baseline: the model is used exactly as trained on the deadtrees
corpus, so its output is the reference point any fine-tuning has to beat.

The model takes a single input: a 3-band uint8 RGB image. That is the whole
interface — no DSM, no multispectral bands, no crown polygons. Our orthos are
float32 in a ~16-bit value range with NaN noData, so the scaling to uint8 is
part of this script rather than an afterthought: feeding the float scene
straight into the model's uint8 pipeline saturates every pixel to 255.

Scaling modes (scale.mode):
    linear      value / (source_max / 255), clipped to [1, 255]. The default.
    percentile  contrast stretch between shared percentiles. The percentiles
                are computed over all three bands jointly, not per band —
                per-band percentiles differ enough on our scenes (blue sits
                lower than red) that they shift the colour balance visibly.
0 is reserved for noData in both modes.

Outputs (out_dir):
    <scene>_rgb8.tif           the uint8 model input          (save_input)
    <scene>_deadwood_mask.tif  uint8 0/1, noData = 255        (save_mask)
    <scene>_deadwood.gpkg      the predicted polygons

Usage:
    uv run python scripts/raw_predict_deadwood.py \\
        --config configs/predict/raw_deadwood.yaml \\
        --working_dir .

CLI flags override config values when provided:
    --source --weights --threshold --batch_size --out_dir
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import safetensors.torch
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig, OmegaConf
from rasterio import features, windows
from rasterio.enums import ColorInterp, Resampling
from shapely.geometry import shape
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.deadwood_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from utils.device import get_device  # noqa: E402

_NODATA_MASK = 255


# ── input preparation ───────────────────────────────────────────────────────


def _scale_to_uint8(data: np.ndarray, valid: np.ndarray, cfg) -> np.ndarray:
    """Map the source values onto [1, 255]; 0 stays reserved for noData."""
    filled = np.where(np.isfinite(data), data, 0.0)
    mode = cfg.get("mode", "linear")

    if mode == "linear":
        source_max = cfg.get("source_max", None)
        if source_max is None:
            source_max = float(np.nanmax(data[:, valid])) if valid.any() else 255.0
        scaled = filled / (float(source_max) / 255.0)
    elif mode == "percentile":
        lo_p, hi_p = cfg.get("percentiles", [2, 98])
        # Shared across bands on purpose — see the module docstring.
        lo, hi = np.percentile(data[:, valid], [lo_p, hi_p])
        print(f"  percentile stretch (shared): {lo:.0f} .. {hi:.0f}")
        scaled = (filled - lo) / max(hi - lo, 1e-9) * 254.0 + 1.0
    else:
        raise ValueError(f"scale.mode must be 'linear' or 'percentile', got {mode!r}")

    out = np.clip(scaled, 1, 255).astype(np.uint8)
    out[:, ~valid] = 0
    return out


def prepare_rgb8(cfg, out_path: Path) -> Path:
    """Resample the source ortho to target_gsd and write it as 3-band uint8 RGB."""
    src_cfg = cfg.source
    bands = list(src_cfg.get("bands", [1, 2, 3]))
    if len(bands) != 3:
        raise ValueError(f"source.bands must name exactly 3 RGB bands, got {bands}")

    with rasterio.open(src_cfg.path) as src:
        window = windows.Window(col_off=0, row_off=0, width=src.width, height=src.height)

        src_gsd = abs(src.transform.a)
        gsd = float(cfg.get("target_gsd", src_gsd))
        out_h = max(1, round(window.height * src_gsd / gsd))
        out_w = max(1, round(window.width * abs(src.transform.e) / gsd))
        print(
            f"  {int(window.width)}x{int(window.height)} px @ {src_gsd:.4f} m"
            f" -> {out_w}x{out_h} @ {gsd:.4f} m"
        )

        data = src.read(
            indexes=bands,
            window=window,
            out_shape=(3, out_h, out_w),
            resampling=Resampling.average,
            boundless=True,
            fill_value=np.nan,
            masked=False,
        ).astype(np.float32)
        transform = src.window_transform(window) * rasterio.Affine.scale(
            window.width / out_w, window.height / out_h
        )
        crs = src.crs
        src_nodata = src.nodata

    valid = np.isfinite(data).all(axis=0)
    if src_nodata is not None and np.isfinite(src_nodata):
        valid &= (data != src_nodata).all(axis=0)
    print(f"  valid pixels: {valid.mean():.1%}")
    if not valid.any():
        raise ValueError("no valid pixels in the requested extent")

    rgb8 = _scale_to_uint8(data, valid, cfg.get("scale", OmegaConf.create({})))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=out_h,
        width=out_w,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(rgb8)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
    print(f"  wrote {out_path}")
    return out_path


# ── tiled inference ─────────────────────────────────────────────────────────


class RGBTileDataset(Dataset):
    """Overlapping tiles of a uint8 RGB scene, ImageNet-normalised.

    Each item covers `tile_size - 2*padding` output pixels; the padding is
    context the model sees but whose predictions are discarded, so tile seams
    do not land in the middle of a prediction.
    """

    def __init__(self, src, tile_size: int, padding: int):
        if tile_size <= 2 * padding:
            raise ValueError(f"tile_size {tile_size} must exceed 2*padding {2 * padding}")
        self.src = src
        self.tile_size = tile_size
        self.padding = padding
        self.height = src.height
        self.width = src.width

        step = tile_size - 2 * padding
        self.offsets = [
            (row, col) for row in range(0, self.height, step) for col in range(0, self.width, step)
        ]

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int):
        row, col = self.offsets[index]
        window = windows.Window(
            col_off=col - self.padding,
            row_off=row - self.padding,
            width=self.tile_size,
            height=self.tile_size,
        )
        image = self.src.read((1, 2, 3), window=window, boundless=True, fill_value=0)
        valid = self.src.dataset_mask(window=window, boundless=True) == 255

        tensor = image.astype(np.float32).transpose(1, 2, 0) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).contiguous()
        return tensor, torch.from_numpy(valid), row, col


def _load_state_dict(weights: str | Path) -> dict[str, torch.Tensor]:
    """Read a checkpoint and strip the torch.compile wrapper prefix.

    Two formats reach this: the published .safetensors checkpoint, and the .pt
    state dicts training/trainer.py writes. Both carry plain module keys once
    the `_orig_mod.` prefix a compiled model adds is removed.
    """
    path = Path(weights)
    if path.suffix == ".safetensors":
        state = safetensors.torch.load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    return {k.removeprefix("_orig_mod."): v for k, v in state.items()}


def load_model(weights: str | Path, device: torch.device) -> torch.nn.Module:
    """Build the deadtrees architecture and load a checkpoint into it.

    Loading is strict: a silent partial load here would look like a badly
    performing model rather than a bug.
    """
    model = smp.Unet(encoder_name="mit_b5", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(_load_state_dict(weights), strict=True)
    return model.to(device).eval()


def predict_probs(src, model, device, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Slide the model over the scene and return (probabilities, valid).

    Kept separate from thresholding so a decision-threshold sweep can reuse one
    expensive pass instead of re-running inference per threshold.
    """
    tile_size = int(cfg.get("tile_size", 1024))
    padding = int(cfg.get("padding", 256))

    dataset = RGBTileDataset(src, tile_size, padding)
    # num_workers stays 0: forked workers share the parent's GDAL handle and
    # the concurrent reads corrupt each other ("ZIPDecode: Decoding error").
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 2)),
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    probs_out = np.zeros((dataset.height, dataset.width), dtype=np.float32)
    valid_out = np.zeros((dataset.height, dataset.width), dtype=bool)
    step = tile_size - 2 * padding

    for images, valid, rows, cols in tqdm(loader, desc="inference"):
        images = images.to(device=device, memory_format=torch.channels_last)
        with torch.inference_mode():
            probs = torch.sigmoid(model(images)).cpu().numpy()[:, 0]

        for i in range(probs.shape[0]):
            row, col = int(rows[i]), int(cols[i])
            core = probs[i, padding : padding + step, padding : padding + step]
            core_valid = valid[i].numpy()[padding : padding + step, padding : padding + step]

            h = min(step, dataset.height - row)
            w = min(step, dataset.width - col)
            probs_out[row : row + h, col : col + w] = core[:h, :w]
            valid_out[row : row + h, col : col + w] = core_valid[:h, :w]

    return probs_out, valid_out


def predict_mask(src, model, device, cfg) -> np.ndarray:
    """Slide the model over the scene and return the thresholded deadwood mask."""
    probs, valid = predict_probs(src, model, device, cfg)
    mask = (probs > float(cfg.get("threshold", 0.5))).astype(np.uint8)
    mask[~valid] = 0
    return mask


# ── vectorisation ───────────────────────────────────────────────────────────


def mask_to_polygons(mask: np.ndarray, transform, crs, min_area: float) -> gpd.GeoDataFrame:
    """Vectorise the 0/1 mask and drop polygons below min_area (CRS units squared)."""
    geoms = [
        shape(geom)
        for geom, value in features.shapes(mask, mask=mask.astype(bool), transform=transform)
        if value == 1
    ]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if gdf.empty:
        return gdf

    before = len(gdf)
    gdf = gdf[gdf.area >= min_area].reset_index(drop=True)
    print(f"  {before} polygons, {before - len(gdf)} dropped below {min_area} m2")
    gdf["area_m2"] = gdf.area.round(3)
    return gdf


# ── entry point ─────────────────────────────────────────────────────────────


def run(cfg, root: Path) -> Path:
    def resolve(value):
        return (root / str(value)).resolve()

    out_dir = resolve(cfg.get("out_dir", "out/raw_deadwood"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(str(cfg.source.path)).stem

    print("Preparing model input")
    rgb8_path = prepare_rgb8(cfg, out_dir / f"{stem}_rgb8.tif")

    device = get_device()
    print(f"Loading weights: {cfg.weights}")
    model = load_model(resolve(cfg.weights), device)

    with rasterio.open(rgb8_path) as src:
        mask = predict_mask(src, model, device, cfg)
        transform, crs = src.transform, src.crs
        profile = src.profile

    print("Vectorising")
    gdf = mask_to_polygons(mask, transform, crs, float(cfg.get("min_polygon_area", 0.1)))

    if cfg.get("save_mask", True):
        mask_path = out_dir / f"{stem}_deadwood_mask.tif"
        profile.update(count=1, dtype="uint8", nodata=_NODATA_MASK)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask, 1)
        print(f"  wrote {mask_path}")

    if not cfg.get("save_input", True):
        rgb8_path.unlink()

    poly_path = out_dir / f"{stem}_deadwood.gpkg"
    if gdf.empty:
        print("No deadwood polygons found — nothing written")
        return poly_path

    gdf.to_file(poly_path, driver="GPKG")
    print(f"  {len(gdf)} polygons, {gdf.area.sum():.0f} m2 total -> {poly_path}")
    return poly_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict deadwood with the pretrained deadtrees.earth model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--working_dir", default=".")
    parser.add_argument("--source", help="override source.path")
    parser.add_argument("--weights", help="override weights")
    parser.add_argument("--threshold", type=float, help="override threshold")
    parser.add_argument("--batch_size", type=int, help="override batch_size")
    parser.add_argument("--out_dir", help="override out_dir")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{args.config} must hold a mapping, not a list")
    cfg = cfg.get("raw_deadwood", cfg)

    if args.source:
        cfg.source.path = args.source
    if args.weights:
        cfg.weights = args.weights
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.out_dir:
        cfg.out_dir = args.out_dir

    run(cfg, Path(args.working_dir).resolve())


if __name__ == "__main__":
    main()
