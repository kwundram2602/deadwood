"""Cut crown-centred training patches for deadwood fine-tuning.

Runs separately from the crown preprocessing pipeline: it reuses functions from
it but shares no config and no execution path.

Each split gets its own label mask, carrying only that split's crowns. A
validation crown falling inside a training crop is therefore MASK_UNLABELLED
there and contributes nothing to the loss. That is the leakage guarantee — with
crowns 7-12 m apart and a 51 m crop, spatially disjoint crops do not exist on
this site.

Usage:
    uv run python scripts/preprocess_deadwood.py \\
        --config configs/preprocess/deadwood.yaml --working_dir .
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_and_process.deadwood_patches import (  # noqa: E402
    SPLIT_NAMES,
    cut_split,
    ring_negatives,
    split_crowns,
)
from explore_and_process.rasterize_crowns import soft_mask_from_geoms  # noqa: E402
from scripts.raw_predict_deadwood import prepare_rgb8  # noqa: E402


def _git_rev(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run(cfg, root: Path) -> Path:
    out_dir = (root / str(cfg.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing model input")
    stem = Path(str(cfg.source.path)).stem
    # prepare_rgb8 opens these paths directly, so resolve them against
    # --working_dir rather than relying on the process cwd.
    cfg.source.path = str((root / str(cfg.source.path)).resolve())
    if cfg.get("clip", None) and cfg.clip.get("vector", None):
        cfg.clip.vector = str((root / str(cfg.clip.vector)).resolve())
    rgb8_path = prepare_rgb8(cfg, out_dir / f"{stem}_rgb8.tif")

    crowns = gpd.read_file(root / str(cfg.labels.path), fid_as_index=True)
    excluded = [int(f) for f in cfg.labels.get("exclude_fids", [])]
    crowns = crowns.drop(index=[f for f in excluded if f in crowns.index])
    with rasterio.open(rgb8_path) as src:
        crowns = crowns.to_crs(src.crs)
        h, w, transform, crs = src.height, src.width, src.transform, src.crs
    print(f"{len(crowns)} crowns (excluded {excluded})")

    sp = cfg.split
    splits = split_crowns(
        crowns,
        int(sp.train),
        int(sp.val),
        int(sp.test),
        mode=str(sp.mode),
        seed=int(sp.get("seed", 0)),
    )

    counts = {}
    for name in SPLIT_NAMES:
        fids = splits[name]
        subset = crowns.loc[fids]
        print(f"\n{name}: {len(fids)} crowns {fids}")
        # Only this split's crowns are burned, so crowns from the other splits
        # stay MASK_UNLABELLED even where a crop overlaps them.
        mask = soft_mask_from_geoms(
            subset.geometry,
            h,
            w,
            transform,
            float(cfg.mask.sigma),
            float(cfg.mask.nodata_threshold),
        )
        mask = ring_negatives(mask, subset.geometry, transform, float(cfg.mask.negative_buffer_m))
        n_pos = int(((mask > 0.0) & (mask <= 1.0)).sum())
        n_neg = int((mask == 0.0).sum())
        print(f"  positives {n_pos:,}  hard negatives {n_neg:,}")
        counts[name] = cut_split(
            rgb8_path,
            mask,
            transform,
            crs,
            subset,
            out_dir / name,
            crop_size=int(cfg.patches.crop_size),
            n_jitter=int(cfg.patches.get("n_jitter", 1)),
            jitter_px=int(cfg.patches.get("jitter_px", 0)),
            seed=int(sp.get("seed", 0)),
        )
        print(f"  wrote {counts[name]} patches -> {out_dir / name}")

    meta = {
        "scene": str(cfg.source.path),
        "target_gsd": float(cfg.target_gsd),
        "scale": OmegaConf.to_container(cfg.scale),
        "labels": str(cfg.labels.path),
        "exclude_fids": excluded,
        "split_mode": str(sp.mode),
        "splits": splits,
        "patch_counts": counts,
        "crop_size": int(cfg.patches.crop_size),
        "mask": {
            "sigma": float(cfg.mask.sigma),
            "nodata_threshold": float(cfg.mask.nodata_threshold),
            "negative_buffer_m": float(cfg.mask.negative_buffer_m),
        },
        "git_rev": _git_rev(root),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_dir / 'meta.json'}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut deadwood fine-tuning patches")
    parser.add_argument("--config", required=True)
    parser.add_argument("--working_dir", default=".")
    parser.add_argument("--out_dir", help="override out_dir")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{args.config} must hold a mapping, not a list")
    cfg = cfg.get("deadwood_preprocess", cfg)
    if args.out_dir:
        cfg.out_dir = args.out_dir

    run(cfg, Path(args.working_dir).resolve())


if __name__ == "__main__":
    main()
