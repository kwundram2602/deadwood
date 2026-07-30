import json
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from data.channels import ChannelSpec


class CrownDataset(Dataset):
    """Named-channel segmentation dataset.

    Expects split_dir/{images,masks,dsm}/ produced by tile_patches.py and
    split via scripts/preprocess.py. The ChannelSpec decides which stack
    bands are read and whether the ndsm patch is appended; tensor channel
    order equals spec.input_channels.

    norm_stats, when given, is the already-subset {"mean", "std"} dict from
    spec.norm_stats(train_stats) — one value per input channel, in order.
    """

    def __init__(
        self,
        split_dir: Path,
        spec: ChannelSpec,
        transform=None,
        norm_stats: dict | None = None,
    ):
        self.image_dir = split_dir / "images"
        self.mask_dir = split_dir / "masks"
        self.dsm_dir = split_dir / "dsm"
        self.spec = spec
        self._stack_indexes = spec.stack_indexes
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

        with rasterio.open(self.image_dir / f"{stem}.tif") as src:
            image = src.read(indexes=self._stack_indexes).astype(np.float32)
        np.nan_to_num(image, copy=False, nan=0.0)

        ndsm = None
        if self.spec.use_ndsm:
            with rasterio.open(self.dsm_dir / f"{stem}_dsm.tif") as src:
                ndsm = src.read().astype(np.float32)
            np.nan_to_num(ndsm, copy=False, nan=0.0)

        image = self.spec.assemble(image, ndsm)  # (in_channels, H, W)

        with rasterio.open(self.mask_dir / f"{stem}_mask.tif") as src:
            mask = src.read(1).astype(np.float32)  # (H, W), 255 = noData

        if self.transform is not None:
            aug = self.transform(image=image.transpose(1, 2, 0), mask=mask)
            image = aug["image"].transpose(2, 0, 1)
            mask = aug["mask"]

        img_tensor = torch.from_numpy(np.ascontiguousarray(image))

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
    cfg: DictConfig, data_root: Path, spec: ChannelSpec
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for the given channel selection.

    Looks for <data_root>/train_stats.json (written by Stage 3 of preprocess.py)
    and applies per-channel z-score normalisation, subset to spec, when found.
    """
    from data.transforms import get_train_transform

    stats_path = data_root / "train_stats.json"
    if stats_path.exists():
        norm_stats = spec.norm_stats(json.loads(stats_path.read_text()))
    else:
        norm_stats = None
        print("[WARN] train_stats.json not found — running without per-channel normalisation")

    train_ds = CrownDataset(data_root / "train", spec, transform=get_train_transform(), norm_stats=norm_stats)
    val_ds = CrownDataset(data_root / "val", spec, norm_stats=norm_stats)
    test_ds = CrownDataset(data_root / "test", spec, norm_stats=norm_stats)

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
