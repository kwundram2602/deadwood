"""Cut training tiles for deadwood fine-tuning from a whole-scene two-class mask.

Runs separately from the crown preprocessing pipeline: it reuses functions from
it but shares no config and no execution path.

One mask covers the scene, carrying deadwood crowns as soft positives and the
digitised background layer as hard negatives, with an unlabelled band between
them. It is cut on a disjoint grid, so tiles share no ground and the split can
be assigned per tile — no per-split masks, and no leakage to guard against.

Usage:
    uv run python scripts/preprocess_deadwood.py \\
        --config configs/preprocess/deadwood.yaml --working_dir .
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_and_process.deadwood_patches import (  # noqa: E402
    SPLIT_NAMES,
    assign_splits,
    deadwood_scene_mask,
    keep_tile,
    scan_tiles,
    tile_footprints,
    tile_kind,
    write_tiles,
)
from scripts.raw_predict_deadwood import prepare_rgb8  # noqa: E402
from utils.nodata import MASK_RASTER_NODATA  # noqa: E402


def _git_rev(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_scene_mask(path: Path, mask, transform, crs) -> None:
    """The whole-scene mask as a GeoTIFF, for checking the labels in QGIS.

    MASK_OUTSIDE is declared as the GDAL noData value so a GIS renders the
    off-footprint region transparent rather than as a dark band.
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=MASK_RASTER_NODATA,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(mask[None, ...])
    print(f"  wrote {path}")


def run(cfg, root: Path) -> Path:
    out_dir = (root / str(cfg.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tile ids are grid positions and so are stable across runs. Without this,
    # a re-run with a different seed or filter threshold moves a tile to
    # another split and leaves the previous copy behind — the same ground in
    # two splits at once, which nothing downstream would notice.
    for name in SPLIT_NAMES:
        shutil.rmtree(out_dir / name, ignore_errors=True)

    print("Preparing model input")
    stem = Path(str(cfg.source.path)).stem
    scene_rel = str(cfg.source.path)
    # prepare_rgb8 opens this path directly, so resolve it against
    # --working_dir rather than relying on the process cwd.
    cfg.source.path = str((root / scene_rel).resolve())
    rgb8_path = prepare_rgb8(cfg, out_dir / f"{stem}_rgb8.tif")

    with rasterio.open(rgb8_path) as src:
        h, w, transform, crs = src.height, src.width, src.transform, src.crs
        # prepare_rgb8 maps valid data onto [1, 255] and reserves 0 for noData,
        # so an all-bands-positive test is the footprint, no heuristics needed.
        footprint = np.all(src.read((1, 2, 3)) > 0, axis=0)
    print(f"Scene {w}x{h} px, footprint {footprint.mean():.1%}")

    crowns = gpd.read_file(root / str(cfg.labels.deadwood_path), fid_as_index=True)
    excluded = [int(f) for f in cfg.labels.get("exclude_fids", [])]
    crowns = crowns.drop(index=[f for f in excluded if f in crowns.index]).to_crs(crs)
    background = gpd.read_file(root / str(cfg.labels.background_path)).to_crs(crs)
    print(f"{len(crowns)} crowns (excluded {excluded}), {len(background)} background polygons")

    mask = deadwood_scene_mask(
        crowns.geometry,
        background.geometry,
        h,
        w,
        transform,
        sigma_pos=float(cfg.mask.sigma_pos),
        sigma_neg=float(cfg.mask.sigma_neg),
        pos_threshold=float(cfg.mask.pos_threshold),
        neg_threshold=float(cfg.mask.neg_threshold),
        footprint=footprint,
    )
    n_pos = int(((mask > 0.0) & (mask <= 1.0)).sum())
    n_neg = int((mask == 0.0).sum())
    print(f"Mask: positives {n_pos:,}  negatives {n_neg:,}")
    _write_scene_mask(out_dir / "deadwood_mask_scene.tif", mask, transform, crs)

    size = int(cfg.tiling.size)
    scan = scan_tiles(mask, size)
    kinds = {tile_id: tile_kind(stats) for tile_id, stats in scan.items()}
    kept = {
        tile_id: kinds[tile_id]
        for tile_id, stats in scan.items()
        if kinds[tile_id] != "empty"
        and keep_tile(
            stats,
            int(cfg.tiling.min_labelled_px),
            float(cfg.tiling.max_outside_frac),
        )
    }
    dropped = Counter(kinds[tile_id] for tile_id in scan if tile_id not in kept)
    n_empty = dropped.pop("empty", 0)
    print(f"\nTiles: {len(scan)} in the grid, {len(kept)} kept")
    print(f"  kept by kind      : {dict(Counter(kept.values()))}")
    print(f"  dropped by filter : {dict(dropped)}")
    print(f"  no polygon at all : {n_empty}")

    sp = cfg.split
    splits = assign_splits(
        kept,
        {name: float(sp[name]) for name in SPLIT_NAMES},
        seed=int(sp.get("seed", 0)),
        stratify=bool(sp.get("stratify", True)),
    )
    counts = write_tiles(rgb8_path, mask, transform, crs, scan, splits, out_dir, size)
    for name in SPLIT_NAMES:
        by_kind = Counter(kept[t] for t, s in splits.items() if s == name)
        print(f"  {name:5s}: {counts[name]:3d} tiles  {dict(by_kind)}")

    tile_footprints(scan, splits, transform, crs, size).to_file(
        out_dir / "tiles.gpkg", driver="GPKG"
    )
    print(f"Wrote {out_dir / 'tiles.gpkg'}")

    meta = {
        "scene": scene_rel,
        "target_gsd": float(cfg.target_gsd),
        "scale": OmegaConf.to_container(cfg.scale),
        "labels": {
            "deadwood_path": str(cfg.labels.deadwood_path),
            "background_path": str(cfg.labels.background_path),
            "exclude_fids": excluded,
        },
        "mask": {
            "sigma_pos": float(cfg.mask.sigma_pos),
            "sigma_neg": float(cfg.mask.sigma_neg),
            "pos_threshold": float(cfg.mask.pos_threshold),
            "neg_threshold": float(cfg.mask.neg_threshold),
            "positive_px": n_pos,
            "negative_px": n_neg,
        },
        "tiling": {
            "size": size,
            "min_labelled_px": int(cfg.tiling.min_labelled_px),
            "max_outside_frac": float(cfg.tiling.max_outside_frac),
            "grid_tiles": len(scan),
            "kept_tiles": len(kept),
            "dropped_by_kind": dict(dropped),
        },
        "split": {
            "fractions": {name: float(sp[name]) for name in SPLIT_NAMES},
            "seed": int(sp.get("seed", 0)),
            "stratify": bool(sp.get("stratify", True)),
            "counts": counts,
            "by_kind": {
                name: dict(Counter(kept[t] for t, s in splits.items() if s == name))
                for name in SPLIT_NAMES
            },
            "tiles": {
                name: sorted(t for t, s in splits.items() if s == name) for name in SPLIT_NAMES
            },
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
