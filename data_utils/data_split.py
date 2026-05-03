import os
import random
import shutil
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
_DEADWOOD   = Path(__file__).resolve().parents[1]
INPUT_ROOT  = _DEADWOOD / "out" / "crown_ms_patches"
OUTPUT_ROOT = _DEADWOOD / "out" / "crown_ms"

IMAGE_SUFFIX    = ".tif"
MASK_IDENTIFIER = "_mask"   # mask files end with _mask.tif
DSM_IDENTIFIER  = "_dsm"    # DSM files end with _dsm.tif

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1

SEED = 42

MODE = "copy"   # "copy" or "symlink"

# ----------------------------
# UTILS
# ----------------------------
def find_triples(input_root: Path):
    """Find (image, mask, dsm) triples. Image stem ends with _ms4."""
    triples = []

    for folder in [input_root] + [f for f in input_root.rglob("*") if f.is_dir()]:
        tifs = list(folder.glob(f"*{IMAGE_SUFFIX}"))

        for tif in tifs:
            # skip mask and dsm files — only process the base image
            if MASK_IDENTIFIER in tif.stem or DSM_IDENTIFIER in tif.stem:
                continue

            mask_path = tif.with_name(tif.stem + MASK_IDENTIFIER + tif.suffix)
            dsm_path  = tif.with_name(tif.stem + DSM_IDENTIFIER  + tif.suffix)

            if not mask_path.exists():
                print(f"[WARN] No mask found for: {tif}")
                continue
            if not dsm_path.exists():
                print(f"[WARN] No DSM found for: {tif}")
                continue

            triples.append((tif, mask_path, dsm_path))

    return triples


def ensure_dirs(base: Path):
    for split in ["train", "val", "test"]:
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "masks").mkdir(parents=True, exist_ok=True)
        (base / split / "dsm").mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, mode="copy"):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    else:
        raise ValueError("mode must be 'copy' or 'symlink'")


# ----------------------------
# MAIN
# ----------------------------
def main():
    random.seed(SEED)

    ensure_dirs(OUTPUT_ROOT)

    triples = find_triples(INPUT_ROOT)

    if len(triples) == 0:
        raise RuntimeError(
            "No image/mask/dsm triples found. "
            "Check that tile_patches.py has been run and INPUT_ROOT is correct."
        )

    print(f"[INFO] Found triples: {len(triples)}")

    random.shuffle(triples)

    n_total = len(triples)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_triples = triples[:n_train]
    val_triples   = triples[n_train:n_train + n_val]
    test_triples  = triples[n_train + n_val:]

    print(f"[INFO] Train: {len(train_triples)} | Val: {len(val_triples)} | Test: {len(test_triples)}")

    def write_split(split_name, split_triples):
        for img_path, mask_path, dsm_path in split_triples:
            transfer_file(img_path,  OUTPUT_ROOT / split_name / "images" / img_path.name,  MODE)
            transfer_file(mask_path, OUTPUT_ROOT / split_name / "masks"  / mask_path.name, MODE)
            transfer_file(dsm_path,  OUTPUT_ROOT / split_name / "dsm"    / dsm_path.name,  MODE)

    write_split("train", train_triples)
    write_split("val",   val_triples)
    write_split("test",  test_triples)

    print(f"[DONE] Dataset split into: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
