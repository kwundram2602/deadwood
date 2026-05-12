import json
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class CrownDataset(Dataset):
    """5-band (MS + nDSM) segmentation dataset.

    Expects split_dir/{images,masks,dsm}/ layout produced by tile_patches.py
    and data_split via scripts/preprocess.py.  Masks contain soft crown
    probability [0,1] and the noData sentinel 255 for pixels to ignore in loss.

    If *norm_stats* is provided ({"mean": [...5], "std": [...5]}), each image
    channel is z-score normalised before being returned as a tensor.  Compute
    stats from the training split with utils.data.compute_channel_stats and
    store them in <data_root>/train_stats.json — preprocess.py does this
    automatically as part of Stage 3.
    """

    def __init__(
        self,
        split_dir: Path,
        transform=None,
        norm_stats: dict | None = None,
    ):
        self.image_dir = split_dir / "images"
        self.mask_dir = split_dir / "masks"
        self.dsm_dir = split_dir / "dsm"
        self.transform = transform
        self.stems = sorted(
            f.stem for f in self.image_dir.iterdir() if f.suffix == ".tif"
        )

        if norm_stats is not None:
            self._norm_mean = torch.tensor(norm_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
            self._norm_std = torch.tensor(norm_stats["std"], dtype=torch.float32).view(-1, 1, 1)
        else:
            self._norm_mean = None
            self._norm_std = None

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img_path = self.image_dir / f"{stem}.tif"
        mask_path = self.mask_dir / f"{stem}_mask.tif"
        dsm_path = self.dsm_dir / f"{stem}_dsm.tif"

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)  # (4, H, W), already [0,1]
        np.nan_to_num(image, copy=False, nan=0.0)

        with rasterio.open(dsm_path) as src:
            dsm = src.read().astype(np.float32)    # (1, H, W), already [0,1]
        np.nan_to_num(dsm, copy=False, nan=0.0)

        image = np.concatenate([image, dsm], axis=0)  # (5, H, W)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)  # (H, W), 255 = noData

        if self.transform is not None:
            aug = self.transform(image=image.transpose(1, 2, 0), mask=mask)
            image = aug["image"].transpose(2, 0, 1)
            mask = aug["mask"]

        img_tensor = torch.from_numpy(np.ascontiguousarray(image))  # (5, H, W)

        if self._norm_mean is not None:
            img_tensor = (img_tensor - self._norm_mean) / self._norm_std

        return (
            img_tensor,
            torch.from_numpy(mask).unsqueeze(0),  # (1, H, W)
        )


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(
    cfg: DictConfig, data_root: Path
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders.

    Looks for <data_root>/train_stats.json (written by Stage 3 of preprocess.py)
    and applies per-channel z-score normalisation when found.
    """
    from data.transforms import get_train_transform

    stats_path = data_root / "train_stats.json"
    norm_stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if norm_stats is None:
        print("[WARN] train_stats.json not found — running without per-channel normalisation")

    train_ds = CrownDataset(data_root / "train", transform=get_train_transform(), norm_stats=norm_stats)
    val_ds = CrownDataset(data_root / "val", norm_stats=norm_stats)
    test_ds = CrownDataset(data_root / "test", norm_stats=norm_stats)

    g = torch.Generator()
    g.manual_seed(0)
    kw = dict(
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        persistent_workers=cfg.dataset.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
