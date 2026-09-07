"""Patch dataset for the pretrained deadtrees deadwood model.

Deliberately not CrownDataset: that one expects [0,1] float stacks with a
channels.json manifest and per-channel train_stats normalisation. This model
was trained on uint8 RGB with ImageNet statistics, and feeding it train_stats
input silently destroys the value of the pretrained weights.
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

# The checkpoint's normalisation. Not tunable — changing these invalidates the
# encoder's pretrained weights.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_deadwood_train_transform():
    """The D4 dihedral group only.

    Aerial crown maps are genuinely invariant under all 8 square symmetries, and
    these are the transforms that cannot introduce a value the mask sentinels do
    not already allow. Radiometric jitter is left out on purpose: it would move
    the input away from the ImageNet statistics the frozen encoder relies on.
    """
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ]
    )


class DeadwoodPatchDataset(Dataset):
    """uint8 RGB crops with soft crown masks, as written by preprocess_deadwood."""

    def __init__(self, split_dir: Path, transform=None):
        self.image_dir = Path(split_dir) / "images"
        self.mask_dir = Path(split_dir) / "masks"
        self.transform = transform
        self.stems = sorted(f.stem for f in self.image_dir.iterdir() if f.suffix == ".tif")
        if not self.stems:
            raise RuntimeError(f"no patches under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        with rasterio.open(self.image_dir / f"{stem}.tif") as src:
            image = src.read((1, 2, 3))
        with rasterio.open(self.mask_dir / f"{stem}_mask.tif") as src:
            mask = src.read(1).astype(np.float32)

        image = image.astype(np.float32).transpose(1, 2, 0) / 255.0
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0),
        )


def make_deadwood_loaders(cfg, data_root: Path):
    """Train/val/test loaders over out/deadwood_patches."""
    train_ds = DeadwoodPatchDataset(data_root / "train", transform=get_deadwood_train_transform())
    val_ds = DeadwoodPatchDataset(data_root / "val")
    test_ds = DeadwoodPatchDataset(data_root / "test")

    kw = dict(
        batch_size=int(cfg.dataset.batch_size),
        num_workers=int(cfg.dataset.num_workers),
        persistent_workers=int(cfg.dataset.num_workers) > 0,
    )
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
