"""Full-scene inference: tile, predict, merge, binarize.

Takes the full-res named-channel scene stack (+ an aligned nDSM on the same
grid, only when the model's channels include 'ndsm' — both [0,1], the
rasterize/dsm_mask stage outputs), slides an overlapping tile window over the
scene, runs the trained UNet on each tile, and blends the sigmoid outputs back
into one georeferenced probability map. The channels the model was trained on
are read from the experiment's channels.json manifest and selected out of the
stack by name. The probability map is binarized with the configured threshold
into a 0/1 crown mask (255 = noData).

Outputs (written to out:, default <weights_dir>/predict/):
    <scene>_prob.tif        float32 [0,1] crown probability, noData = -1
    <scene>_pred_t<T>.tif   uint8 0/1 crown mask,             noData = 255
    <scene>_quicklook.png   Pseudo-RGB | probability | binary preview

Usage:
    uv run python scripts/predict.py \\
        --config configs/predict/predict.yaml \\
        --working_dir .

CLI flags override config values when provided:
    --image --dsm --dtm --weights --threshold --out
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.channels import NDSM, ChannelSpec, load_manifest
from explore_and_process.tile_patches import tile_raster

_NODATA_BIN = 255
_NODATA_PROB = -1.0


def make_windows(
    height: int, width: int, tile_size: int, stride: int
) -> list[tuple[int, int]]:
    """(row_off, col_off) offsets covering the scene with the given stride.

    The last offset per axis is clamped so the window ends exactly at the
    scene edge; scenes smaller than tile_size get a single (0, 0) window
    (tile_raster pads the read).
    """

    def axis_offsets(dim: int) -> list[int]:
        if dim <= tile_size:
            return [0]
        offs = list(range(0, dim - tile_size, stride))
        offs.append(dim - tile_size)
        return sorted(set(offs))

    return [(r, c) for r in axis_offsets(height) for c in axis_offsets(width)]


def blend_weights(tile_size: int, eps: float = 1e-3) -> np.ndarray:
    """2D Hann window + eps: center-weighted blending of overlapping tiles.

    The eps keeps border weights non-zero so pixels covered by a single tile
    (scene borders) normalise exactly to that tile's prediction.
    """
    hann = np.hanning(tile_size).astype(np.float32)
    return np.outer(hann, hann) + eps


class TileMerger:
    """Accumulate weighted tile predictions into one full-scene map."""

    def __init__(self, height: int, width: int, tile_size: int):
        self.height = height
        self.width = width
        self.weights = blend_weights(tile_size)
        self._prob_sum = np.zeros((height, width), dtype=np.float32)
        self._weight_sum = np.zeros((height, width), dtype=np.float32)

    def add(self, prob_tile: np.ndarray, row_off: int, col_off: int) -> None:
        h = min(prob_tile.shape[0], self.height - row_off)
        w = min(prob_tile.shape[1], self.width - col_off)
        window = self.weights[:h, :w]
        self._prob_sum[row_off : row_off + h, col_off : col_off + w] += (
            prob_tile[:h, :w] * window
        )
        self._weight_sum[row_off : row_off + h, col_off : col_off + w] += window

    def merge(self) -> np.ndarray:
        return self._prob_sum / np.maximum(self._weight_sum, 1e-8)


def binarize(
    prob: np.ndarray, threshold: float, valid_mask: np.ndarray
) -> np.ndarray:
    """[0,1] probability → uint8 {0, 1}; invalid pixels → 255."""
    out = (prob >= threshold).astype(np.uint8)
    out[~valid_mask] = _NODATA_BIN
    return out


def validate_grid(img_src, dsm_src) -> None:
    """Image and nDSM must share shape, CRS, and geotransform (sub-pixel tol)."""
    if (img_src.height, img_src.width) != (dsm_src.height, dsm_src.width):
        raise ValueError(
            f"Grid mismatch: image {img_src.height}x{img_src.width} vs "
            f"DSM {dsm_src.height}x{dsm_src.width}"
        )
    if img_src.crs != dsm_src.crs:
        raise ValueError(f"CRS mismatch: image {img_src.crs} vs DSM {dsm_src.crs}")
    a, b = img_src.transform, dsm_src.transform
    tol = abs(a.a) * 0.5  # half a pixel
    if any(abs(x - y) > tol for x, y in zip(a[:6], b[:6])):
        raise ValueError(
            f"Geotransform mismatch beyond half a pixel:\n  image: {a}\n  dsm:   {b}\n"
            "Set preprocess.enabled: true to resample both onto one grid."
        )


def prepare_inputs(
    sources: list[tuple[str, list[int], list[str]]],
    raw_dsm: Path | None,
    raw_dtm: Path | None,
    prep_cfg,
    prep_dir: Path,
    need_ndsm: bool,
) -> tuple[Path, Path | None]:
    """Replicate the training preprocessing on raw merged rasters.

    Sources: (path, bands, names) tuples — resampled to target_gsd, scaled to
    [0,1], stacked with band descriptions (rasterize stage). DSM (only when
    need_ndsm): reproject to the same grid, then nDSM = DSM - DTM when an
    external DTM is given (apply_dsm_mask's `method: dtm`) or DSM - multi-scale
    minimum filter otherwise (`method: local_min`), normalised to [0,1] the
    same way in both cases (dsm_mask stage). Prepared rasters are cached in
    prep_dir with param-tagged names and reused on the next run.
    """
    from explore_and_process.apply_dsm_mask import (
        detect_ground_dtm,
        detect_ground_local_min,
        normalize_ndsm,
        resample_raster,
    )
    from explore_and_process.rasterize_crowns import stack_sources, target_grid

    gsd = float(prep_cfg.target_gsd)
    gsd_tag = f"{gsd * 100:g}cm"
    src_tag = hashlib.md5(repr(sources).encode()).hexdigest()[:8]
    img_out = prep_dir / f"{Path(sources[0][0]).stem}_stack_{src_tag}_{gsd_tag}.tif"
    prep_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(sources[0][0]) as ref:
        h, w, transform = target_grid(ref, gsd)
        crs = ref.crs
        print(
            f"Prep grid: {h} x {w} at {gsd * 100:g} cm GSD "
            f"(native {ref.res[0] * 100:.2f} cm)"
        )

    if img_out.exists():
        print(f"Prep cache hit: {img_out.name}")
    else:
        print(f"Stacking {len(sources)} source(s) to {gsd * 100:g} cm...")
        stack_sources(sources, h, w, transform, crs, str(img_out))

    if not need_ndsm:
        return img_out, None

    if raw_dsm is None:
        raise ValueError("model uses 'ndsm' but no dsm is set (config dsm: or --dsm)")
    windows = [int(x) for x in prep_cfg.windows]
    max_h = float(prep_cfg.max_ndsm_height)
    w_tag = "-".join(str(x) for x in windows)
    # Tag mirrors apply_dsm_mask.py so a DTM run never reuses a local_min cache
    method_tag = "dtm" if raw_dtm is not None else f"w{w_tag}"
    dsm_out = prep_dir / f"{raw_dsm.stem}_ndsm_{gsd_tag}_{method_tag}_max{max_h:g}.tif"
    if dsm_out.exists():
        print(f"Prep cache hit: {dsm_out.name}")
    else:
        dsm = resample_raster(str(raw_dsm), h, w, transform, crs)
        # height_threshold only shapes the (discarded) ground confidence;
        # the nDSM itself is just dsm - ground
        if raw_dtm is not None:
            print(f"Computing nDSM (external DTM {raw_dtm.name})...")
            dtm = resample_raster(str(raw_dtm), h, w, transform, crs)
            _, _, ndsm = detect_ground_dtm(dsm, dtm, height_threshold=1.0)
        else:
            print(f"Computing nDSM (minimum filter windows {windows} px)...")
            _, _, ndsm = detect_ground_local_min(dsm, windows, height_threshold=1.0)
        ndsm_norm = normalize_ndsm(ndsm, dsm, max_h)
        profile = dict(
            driver="GTiff", dtype="float32", width=w, height=h, count=1,
            crs=crs, transform=transform, nodata=None, compress="lzw",
        )
        with rasterio.open(dsm_out, "w", **profile) as dst:
            dst.write(ndsm_norm[np.newaxis])
        print(f"Saved nDSM: {dsm_out}")
    return img_out, dsm_out


def infer_architecture(state: dict) -> tuple[str, int, int]:
    """Read (encoder_name, in_channels, num_classes) from a UNet state dict.

    The architecture is not stored in a .pt checkpoint, but the tensor shapes
    determine it: first-conv input dim → in_channels, segmentation-head output
    dim → num_classes, presence of bottleneck conv3 → resnet50 vs resnet34.
    """
    try:
        in_channels = state["encoder.conv1.weight"].shape[1]
        num_classes = state["segmentation_head.0.weight"].shape[0]
    except KeyError as e:
        raise ValueError(
            f"Checkpoint is missing expected UNet key {e} — not a model trained "
            "by this repo (scripts/train.py)?"
        ) from e
    if "encoder.layer1.0.conv3.weight" in state:
        encoder_name = "resnet50"
    elif "encoder.layer3.2.conv1.weight" in state:
        encoder_name = "resnet34"
    else:
        raise ValueError(
            "Cannot infer encoder backbone from checkpoint (expected resnet50 "
            "or resnet34 key layout)"
        )
    return encoder_name, int(in_channels), int(num_classes)


def build_model_from_checkpoint(
    weights_path: Path, device: torch.device
) -> torch.nn.Module:
    """Rebuild the trained UNet purely from a .pt checkpoint and load it."""
    import segmentation_models_pytorch as smp

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    encoder_name, in_channels, num_classes = infer_architecture(state)
    print(
        f"Checkpoint architecture: unet/{encoder_name}, "
        f"in_channels={in_channels}, num_classes={num_classes}"
    )
    # torchgeo's unet() is a thin smp.Unet wrapper → identical state dict keys
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded checkpoint: {weights_path}")
    return model.to(device)


def predict_scene(
    model: torch.nn.Module,
    image_path: Path,
    dsm_path: Path | None,
    device: torch.device,
    norm_stats: dict | None,
    spec: ChannelSpec,
    tile_size: int,
    stride: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sliding-window inference. Returns (prob, valid_mask, raster profile)."""
    import contextlib

    if norm_stats is not None:
        mean = np.asarray(norm_stats["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(norm_stats["std"], dtype=np.float32)[:, None, None]
    else:
        mean = std = None

    idx0 = np.asarray(spec.stack_indexes) - 1  # 0-based positions in the stack tile

    model.eval()
    with contextlib.ExitStack() as ctx:
        img_src = ctx.enter_context(rasterio.open(image_path))
        dsm_src = ctx.enter_context(rasterio.open(dsm_path)) if dsm_path else None
        if dsm_src is not None:
            validate_grid(img_src, dsm_src)
        h, w = img_src.height, img_src.width
        profile = dict(crs=img_src.crs, transform=img_src.transform)

        offsets = make_windows(h, w, tile_size, stride)
        print(f"Scene {h}x{w} → {len(offsets)} tiles ({tile_size}px, stride {stride})")

        merger = TileMerger(h, w, tile_size)
        valid_mask = np.zeros((h, w), dtype=bool)

        with torch.no_grad():
            for start in range(0, len(offsets), batch_size):
                batch_offs = offsets[start : start + batch_size]
                tiles = []
                for r, c in batch_offs:
                    img = tile_raster(img_src, r, c, tile_size).astype(np.float32)[idx0]
                    np.nan_to_num(img, copy=False, nan=0.0)
                    ndsm = None
                    if dsm_src is not None:
                        ndsm = tile_raster(dsm_src, r, c, tile_size).astype(np.float32)
                        np.nan_to_num(ndsm, copy=False, nan=0.0)

                    th = min(tile_size, h - r)
                    tw = min(tile_size, w - c)
                    valid_mask[r : r + th, c : c + tw] |= ~np.all(
                        img[:, :th, :tw] == 0, axis=0
                    )

                    tile = spec.assemble(img, ndsm)  # (in_channels, H, W)
                    if mean is not None:
                        tile = (tile - mean) / std
                    tiles.append(tile)

                batch = torch.from_numpy(np.stack(tiles)).to(device)
                probs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()
                for (r, c), prob_tile in zip(batch_offs, probs):
                    merger.add(prob_tile, r, c)

                done = min(start + batch_size, len(offsets))
                print(f"\r  {done}/{len(offsets)} tiles", end="", flush=True)
        print()

    return merger.merge(), valid_mask, profile


def write_geotiff(path: Path, data: np.ndarray, profile: dict, nodata) -> None:
    full = dict(
        driver="GTiff",
        dtype=str(data.dtype),
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        nodata=nodata,
        compress="lzw",
        **profile,
    )
    with rasterio.open(path, "w", **full) as dst:
        dst.write(data, 1)
    print(f"Saved {path}")


def quicklook_band_indexes(stack_names: list[str]) -> list[int]:
    """1-based display bands: true red/green/blue if all present, else the
    first three stack bands (last one repeated if fewer exist)."""
    if all(n in stack_names for n in ("red", "green", "blue")):
        return [stack_names.index(n) + 1 for n in ("red", "green", "blue")]
    idx = list(range(1, min(3, len(stack_names)) + 1))
    while len(idx) < 3:
        idx.append(idx[-1])
    return idx


def save_quicklook(
    image_path: Path,
    prob: np.ndarray,
    binary: np.ndarray,
    save_path: Path,
    stack_names: list[str],
    max_dim: int = 2000,
) -> None:
    """Pseudo-RGB | probability | binary side-by-side preview."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    ds = max(1, max(prob.shape) // max_dim)
    with rasterio.open(image_path) as src:
        rgb = src.read(quicklook_band_indexes(stack_names)).astype(np.float32)
    rgb = rgb[:, ::ds, ::ds].transpose(1, 2, 0)
    rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)

    prob_ds = prob[::ds, ::ds]
    bin_ds = binary[::ds, ::ds].astype(np.float32)
    bin_ds[bin_ds == _NODATA_BIN] = np.nan

    # 0 = background (white), 1 = crown (green), NaN/noData = light grey
    bin_cmap = ListedColormap(["#ffffff", "#2e8b57"])
    bin_cmap.set_bad("#c8c8c8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Pseudo-RGB")
    im = axes[1].imshow(prob_ds, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Crown probability")
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    axes[2].imshow(bin_ds, cmap=bin_cmap, vmin=0, vmax=1)
    axes[2].set_title("Binary mask")
    axes[2].legend(
        handles=[
            Patch(facecolor="#2e8b57", label="crown"),
            Patch(facecolor="#ffffff", edgecolor="#999999", label="background"),
            Patch(facecolor="#c8c8c8", label="noData"),
        ],
        loc="lower right",
        fontsize=9,
    )
    for ax in axes:
        ax.axis("off")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict crown mask for a full scene")
    parser.add_argument("--config", required=True, help="Path to configs/predict/*.yaml")
    parser.add_argument("--working_dir", default=".")
    parser.add_argument(
        "--image",
        default=None,
        help="Override: pre-stacked scene GeoTIFF (preprocess.enabled: false)",
    )
    parser.add_argument("--dsm", default=None, help="Override: aligned nDSM GeoTIFF")
    parser.add_argument(
        "--dtm",
        default=None,
        help="Override: external DTM GeoTIFF; nDSM becomes DSM-DTM instead of "
        "DSM minus a minimum filter (requires preprocess.enabled)",
    )
    parser.add_argument("--weights", default=None, help="Override: .pt checkpoint")
    parser.add_argument("--threshold", type=float, default=None, help="Override: binarization threshold")
    parser.add_argument("--out", default=None, help="Override: output dir")
    args = parser.parse_args()

    from utils.device import get_device

    pcfg = OmegaConf.load(args.config)
    root = Path(args.working_dir).resolve()

    threshold = args.threshold if args.threshold is not None else float(pcfg.get("threshold", 0.5))

    weights_str = args.weights or pcfg.get("weights", None)
    if not weights_str:
        raise ValueError("weights must be set (predict config or --weights)")
    weights_path = Path(weights_str)
    if not weights_path.is_absolute():
        weights_path = root / weights_path

    tile_size = int(pcfg.get("tile_size", 512))
    overlap = float(pcfg.get("overlap", 0.5))
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    stride = max(1, int(tile_size * (1 - overlap)))
    batch_size = int(pcfg.get("batch_size", 8))

    # --- resolve channels the model was trained with -----------------------
    channels_str = pcfg.get("channels", None)
    channels_path = (
        root / channels_str if channels_str else weights_path.parent / "channels.json"
    )
    if not channels_path.exists():
        raise ValueError(
            f"Channel manifest not found: {channels_path}\n"
            "scripts/train.py writes channels.json next to the checkpoint; "
            "for other checkpoints set 'channels:' explicitly."
        )
    input_channels = load_manifest(channels_path)
    print(f"Model input channels: {input_channels}")
    need_ndsm = NDSM in input_channels

    dsm_str = args.dsm or pcfg.get("dsm", None)
    dsm_path = root / dsm_str if dsm_str else None
    dtm_str = args.dtm or pcfg.get("dtm", None)
    dtm_path = root / dtm_str if dtm_str else None

    # --- build / load the scene stack --------------------------------------
    prep_cfg = pcfg.get("preprocess", None)
    if prep_cfg is not None and prep_cfg.get("enabled", False):
        sources = [
            (str(root / s.path), [int(b) for b in s.bands], [str(n) for n in s.names])
            for s in prep_cfg.sources
        ]
        stack_names = [n for _, _, names in sources for n in names]
        prep_dir_str = prep_cfg.get("dir", None)
        prep_dir = (
            root / prep_dir_str
            if prep_dir_str
            else root / "datafiles/process_out/predict_prep"
        )
        image_path, dsm_path = prepare_inputs(
            sources, dsm_path, dtm_path, prep_cfg, prep_dir, need_ndsm
        )
    else:
        image_str = args.image or pcfg.get("image", None)
        if not image_str:
            raise ValueError("preprocess.enabled is false — set image: (pre-stacked scene)")
        image_path = root / image_str
        with rasterio.open(image_path) as src:
            stack_names = list(src.descriptions)
        if not all(stack_names):
            raise ValueError(
                f"{image_path} has no band descriptions — produce it with "
                "rasterize_crowns.py or set preprocess.enabled: true"
            )
        if need_ndsm and dsm_path is None:
            raise ValueError("model uses 'ndsm' but no dsm is set (config dsm: or --dsm)")
        if dtm_path is not None:
            raise ValueError(
                "dtm: is only used when preprocess.enabled is true — with a "
                "pre-stacked scene the nDSM must already be built (use "
                "apply_dsm_mask.py --method dtm)"
            )

    spec = ChannelSpec(stack_names, input_channels)
    if not spec.use_ndsm:
        dsm_path = None

    # --- stats --------------------------------------------------------------
    stats_str = pcfg.get("stats", None)
    if stats_str:
        stats_path = root / stats_str
        norm_stats = spec.norm_stats(json.loads(stats_path.read_text()))
        print(f"Normalisation stats: {stats_path}")
    else:
        norm_stats = None
        print("[WARN] stats not set — predicting without normalisation. "
              "If the model was trained with train_stats.json this will degrade results.")

    device = get_device()
    model = build_model_from_checkpoint(weights_path, device)
    ckpt_in = model.encoder.conv1.in_channels
    if ckpt_in != spec.in_channels:
        raise ValueError(
            f"Checkpoint expects {ckpt_in} input channels but channels.json "
            f"lists {spec.in_channels}: {spec.input_channels}"
        )

    prob, valid_mask, profile = predict_scene(
        model, image_path, dsm_path, device, norm_stats, spec,
        tile_size=tile_size, stride=stride, batch_size=batch_size,
    )
    binary = binarize(prob, threshold, valid_mask)
    prob_out = np.where(valid_mask, prob, _NODATA_PROB).astype(np.float32)

    out_str = args.out or pcfg.get("out", None)
    out_dir = root / out_str if out_str else weights_path.parent / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    write_geotiff(out_dir / f"{stem}_prob.tif", prob_out, profile, nodata=_NODATA_PROB)
    write_geotiff(
        out_dir / f"{stem}_pred_t{threshold:g}.tif", binary, profile, nodata=_NODATA_BIN
    )

    crown_frac = float((binary == 1).sum()) / max(int(valid_mask.sum()), 1)
    print(f"Threshold {threshold:g}: {crown_frac:.1%} of valid pixels classified as crown")

    if pcfg.get("quicklook", True):
        save_quicklook(
            image_path, prob, binary, out_dir / f"{stem}_quicklook.png", stack_names
        )


if __name__ == "__main__":
    main()
