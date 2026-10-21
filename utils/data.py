import json
import os
import random
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import rasterio

from deadwood_spectral.align import parse_date
from utils.nodata import footprint_from_stack

_IMAGE_SUFFIX = ".tif"
_MASK_ID = "_mask"
_DSM_ID = "_dsm"

SPLIT_NAMES = ("train", "val", "test")
SPLIT_STRATEGIES = ("random", "by_date")


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


def _explicit_date_splits(mapping: Mapping, dates: list[str]) -> dict[str, str]:
    """Validate an explicit {split: [dates]} mapping; return date -> split."""
    as_dict = {str(k): list(v or []) for k, v in dict(mapping).items()}
    unknown = set(as_dict) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"split.dates has unknown splits {sorted(unknown)}")
    assigned: dict[str, str] = {}
    for split in SPLIT_NAMES:
        for d in as_dict.get(split, []):
            d = str(d)
            if d in assigned:
                raise ValueError(f"date {d} is assigned to both {assigned[d]} and {split}")
            assigned[d] = split
    if set(assigned) != set(dates):
        raise ValueError(
            "split.dates must assign exactly the selected dates: "
            f"missing {sorted(set(dates) - set(assigned))}, "
            f"not selected {sorted(set(assigned) - set(dates))}"
        )
    return assigned


def assign_dates(
    dates: list[str],
    explicit: Mapping | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole acquisition dates to splits; return date -> split.

    explicit: {train: [...], val: [...], test: [...]} used as given; every
      date must appear exactly once.
    otherwise: seeded shuffle of the sorted dates, with
      n_test = min(round(n * (1 - train - val)), n - 2),
      n_val  = max(1, min(round(n * val), n - 1 - n_test)),
      n_train = the rest — so train and val always get at least one date.

    Only test may stay empty: val drives model selection.

    All tiles of one date land in one split, but the same trees appear on
    every date. This split measures generalisation to unseen dates, not to
    unseen places.
    """
    dates = sorted(str(d) for d in dates)
    if explicit is not None:
        assigned = _explicit_date_splits(explicit, dates)
    else:
        n = len(dates)
        if n < 2:
            raise ValueError(f"split.strategy by_date needs >= 2 dates, got {dates}")
        n_test = min(round(n * (1.0 - train_ratio - val_ratio)), n - 2)
        n_test = max(n_test, 0)
        n_val = max(1, min(round(n * val_ratio), n - 1 - n_test))
        n_train = n - n_val - n_test
        order = list(np.random.default_rng(seed).permutation(dates))
        assigned = {}
        for d in order[:n_train]:
            assigned[str(d)] = "train"
        for d in order[n_train : n_train + n_val]:
            assigned[str(d)] = "val"
        for d in order[n_train + n_val :]:
            assigned[str(d)] = "test"
    for split in ("train", "val"):
        if split not in assigned.values():
            raise ValueError(f"no date assigned to '{split}'; only test may be empty")
    return assigned


def split_patches(
    input_root: Path,
    output_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    mode: str = "copy",
    strategy: str = "random",
    date_splits: Mapping[str, str] | None = None,
) -> dict[str, list[tuple[Path, Path, Path]]]:
    """Split image/mask/dsm patches into train/val/test directories.

    strategy:
      random  — seeded shuffle over all tiles, sliced by the ratios.
      by_date — every tile goes to date_splits[<YYYYMMDD stem prefix>], so
                tiles of one date never end up in different splits.

    Writes split.json (strategy, seed, date -> split, tile stems per split)
    next to the split directories. Returns a dict mapping split name to the
    list of (image, mask, dsm) triples written for that split — useful for
    downstream stats computation.
    """
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"strategy must be one of {SPLIT_STRATEGIES}, got {strategy!r}")
    triples = find_triples(input_root)
    if not triples:
        raise RuntimeError(
            f"No image/mask/dsm triples found under {input_root}. "
            "Run tile_patches.py first."
        )

    if strategy == "random":
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
    else:
        if date_splits is None:
            raise ValueError("strategy 'by_date' needs date_splits (date -> split)")
        splits = {name: [] for name in SPLIT_NAMES}
        for triple in triples:
            date = parse_date(triple[0])
            if date not in date_splits:
                raise ValueError(f"{triple[0].name}: date {date} has no split assigned")
            splits[date_splits[date]].append(triple)

    print(
        f"[INFO] Found {len(triples)} triples ({strategy}) → "
        f"train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}"
    )

    if output_root.exists():
        shutil.rmtree(output_root)
        print(f"[INFO] Cleared existing split directory: {output_root}")
    _ensure_split_dirs(output_root)

    manifest = input_root / "channels.json"
    if manifest.exists():
        shutil.copy2(manifest, output_root / "channels.json")

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

    split_record = {
        "strategy": strategy,
        "seed": seed,
        "input": str(input_root),
        "date_splits": dict(sorted(date_splits.items())) if date_splits else None,
        "tiles": {name: sorted(t[0].stem for t in ts) for name, ts in splits.items()},
    }
    if strategy == "random":
        split_record.update(train_ratio=train_ratio, val_ratio=val_ratio)
    (output_root / "split.json").write_text(json.dumps(split_record, indent=2))

    print(f"[DONE] Split written to: {output_root}")
    return written


def compute_channel_stats(split_dir: Path, names: list[str]) -> dict:
    """Per-channel mean/std over all patches: stack channels (*names*) + ndsm.

    Uses the sum-of-squares identity (E[X²] − E[X]²) to avoid holding all
    data in memory. Returns {"names": names + ["ndsm"], "mean", "std"}.

    Only pixels inside the scene footprint contribute. Out-of-footprint pixels
    carry no measurement; averaging their filler values in drags every mean
    toward that filler and inflates the variance, which then bakes a bias into
    the normalisation the model is trained with. Patches written before the
    footprint was tracked have no NaN and fall back to the all-bands-zero
    heuristic, so their stats are unchanged.
    """
    image_dir = split_dir / "images"
    dsm_dir = split_dir / "dsm"

    stems = sorted(f.stem for f in image_dir.iterdir() if f.suffix == ".tif")
    if not stems:
        raise RuntimeError(f"No images found in {image_dir}")

    all_names = list(names) + ["ndsm"]
    n_ch = len(all_names)
    ch_sum = np.zeros(n_ch, dtype=np.float64)
    ch_sum_sq = np.zeros(n_ch, dtype=np.float64)
    n_pixels = np.zeros(n_ch, dtype=np.int64)

    for stem in stems:
        with rasterio.open(image_dir / f"{stem}.tif") as src:
            img = src.read().astype(np.float64)
        if img.shape[0] != len(names):
            raise ValueError(
                f"{stem}.tif has {img.shape[0]} bands but manifest lists "
                f"{len(names)} names: {list(names)}"
            )
        with rasterio.open(dsm_dir / f"{stem}_dsm.tif") as src:
            dsm = src.read().astype(np.float64)

        # The footprint comes from the image bands; the nDSM is 0 outside the
        # DSM anyway and shares the scene extent, so it uses the same mask.
        inside = footprint_from_stack(img)
        combined = np.concatenate([img, dsm], axis=0)
        np.nan_to_num(combined, copy=False)

        n_inside = int(inside.sum())
        if n_inside == 0:
            continue
        for c in range(n_ch):
            vals = combined[c][inside]
            ch_sum[c] += vals.sum()
            ch_sum_sq[c] += (vals**2).sum()
        n_pixels += n_inside

    if not n_pixels.all():
        raise RuntimeError(
            f"No in-footprint pixels found in {image_dir} — every patch is "
            "entirely outside the scene footprint?"
        )
    mean = ch_sum / n_pixels
    variance = ch_sum_sq / n_pixels - mean**2
    std = np.sqrt(np.maximum(variance, 1e-12))

    return {"names": all_names, "mean": mean.tolist(), "std": std.tolist()}


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
