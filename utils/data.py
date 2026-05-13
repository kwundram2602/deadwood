import os
import random
import shutil
from pathlib import Path

import numpy as np
import rasterio

_IMAGE_SUFFIX = ".tif"
_MASK_ID = "_mask"
_DSM_ID = "_dsm"


def find_triples(input_root: Path) -> list[tuple[Path, Path, Path]]:
    """Return (image, mask, dsm) triples found under *input_root*.

    Images are identified as .tif files whose stems do NOT contain the
    mask/dsm identifiers.  Triples are returned in deterministic sorted order
    so that a fixed random seed produces a reproducible shuffle.
    """
    triples: list[tuple[Path, Path, Path]] = []
    dirs = [input_root] + sorted(d for d in input_root.rglob("*") if d.is_dir())

    for folder in dirs:
        for tif in sorted(folder.glob(f"*{_IMAGE_SUFFIX}")):
            if _MASK_ID in tif.stem or _DSM_ID in tif.stem:
                continue
            mask_path = tif.with_name(tif.stem + _MASK_ID + tif.suffix)
            dsm_path = tif.with_name(tif.stem + _DSM_ID + tif.suffix)
            if not mask_path.exists():
                print(f"[WARN] No mask for: {tif}")
                continue
            if not dsm_path.exists():
                print(f"[WARN] No DSM for: {tif}")
                continue
            triples.append((tif, mask_path, dsm_path))

    return triples


def _ensure_split_dirs(base: Path) -> None:
    for split in ("train", "val", "test"):
        for sub in ("images", "masks", "dsm"):
            (base / split / sub).mkdir(parents=True, exist_ok=True)


def _transfer(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.unlink(missing_ok=True)
        os.symlink(src, dst)
    else:
        raise ValueError(f"mode must be 'copy' or 'symlink', got {mode!r}")


def split_patches(
    input_root: Path,
    output_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    mode: str = "copy",
) -> dict[str, list[tuple[Path, Path, Path]]]:
    """Split image/mask/dsm patches into train/val/test directories.

    Returns a dict mapping split name to the list of (image, mask, dsm) triples
    written for that split — useful for downstream stats computation.
    """
    triples = find_triples(input_root)
    if not triples:
        raise RuntimeError(
            f"No image/mask/dsm triples found under {input_root}. "
            "Run tile_patches.py first."
        )

    random.seed(seed)
    random.shuffle(triples)

    n = len(triples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": triples[:n_train],
        "val": triples[n_train : n_train + n_val],
        "test": triples[n_train + n_val :],
    }

    print(
        f"[INFO] Found {n} triples → "
        f"train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}"
    )

    _ensure_split_dirs(output_root)

    written: dict[str, list[tuple[Path, Path, Path]]] = {}
    for split_name, split_triples in splits.items():
        out_imgs: list[tuple[Path, Path, Path]] = []
        for img, mask, dsm in split_triples:
            dst_img = output_root / split_name / "images" / img.name
            dst_mask = output_root / split_name / "masks" / mask.name
            dst_dsm = output_root / split_name / "dsm" / dsm.name
            _transfer(img, dst_img, mode)
            _transfer(mask, dst_mask, mode)
            _transfer(dsm, dst_dsm, mode)
            out_imgs.append((dst_img, dst_mask, dst_dsm))
        written[split_name] = out_imgs

    print(f"[DONE] Split written to: {output_root}")
    return written


def compute_channel_stats(split_dir: Path) -> dict[str, list[float]]:
    """Compute per-channel mean and std from all patches in *split_dir*.

    Reads both the 4-band MS image and the 1-band DSM for each patch so that
    the returned stats cover all 5 input channels (R, G, RE, NIR, nDSM).
    Uses the sum-of-squares identity (E[X²] − E[X]²) to avoid holding all
    data in memory.
    """
    image_dir = split_dir / "images"
    dsm_dir = split_dir / "dsm"

    stems = sorted(f.stem for f in image_dir.iterdir() if f.suffix == ".tif")
    if not stems:
        raise RuntimeError(f"No images found in {image_dir}")

    n_ch = 5
    ch_sum = np.zeros(n_ch, dtype=np.float64)
    ch_sum_sq = np.zeros(n_ch, dtype=np.float64)
    n_pixels = 0

    for stem in stems:
        img_path = image_dir / f"{stem}.tif"
        dsm_path = dsm_dir / f"{stem}_dsm.tif"

        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float64)   # (4, H, W)
        with rasterio.open(dsm_path) as src:
            dsm = src.read().astype(np.float64)   # (1, H, W)

        combined = np.concatenate([img, dsm], axis=0)  # (5, H, W)
        np.nan_to_num(combined, copy=False)

        hw = combined.shape[1] * combined.shape[2]
        for c in range(n_ch):
            ch_sum[c] += combined[c].sum()
            ch_sum_sq[c] += (combined[c] ** 2).sum()
        n_pixels += hw

    mean = ch_sum / n_pixels
    variance = ch_sum_sq / n_pixels - mean**2
    std = np.sqrt(np.maximum(variance, 1e-12))

    return {"mean": mean.tolist(), "std": std.tolist()}


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Split crown-MS patches into train/val/test.")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=["copy", "symlink"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f).get("split", {})

    def _get(key: str, fallback):
        return getattr(args, key) if getattr(args, key) is not None else cfg.get(key, fallback)

    split_patches(
        input_root=Path(_get("input", "out/crown_ms_patches")),
        output_root=Path(_get("output", "out/crown_ms")),
        train_ratio=_get("train_ratio", 0.7),
        val_ratio=_get("val_ratio", 0.2),
        seed=_get("seed", 42),
        mode=_get("mode", "copy"),
    )
