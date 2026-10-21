"""Per-scene error evaluation: MSE and binary error rate per scene × split, error maps.

Every tile of the split folder is predicted once; the errors are reported per
scene (acquisition date for time-series tiles, ``"scene"`` otherwise) and per
split, and the squared error is mosaicked back onto each scene's grid.

Definitions, over the valid pixels i of a scene × split
(``valid_target(y)``: y in [0, 1]; the 255 / -1 noData sentinels are excluded):

    p_i = sigmoid(logit_i)            model probability (the network emits logits)
    y_i                               soft target from the crown mask

    mse          = (1/N) Σ_i (p_i − y_i)²                     Brier score
    mse_binary   = (1/N) Σ_i [ (p_i ≥ t) ≠ (y_i > t_y) ]      = error rate = 1 − acc
    mse_baseline        = (1/N) Σ_i (c − y_i)²,  c = mean of the valid train targets
    mse_binary_baseline = (1/N) Σ_i [ m ≠ (y_i > t_y) ],  m = majority class of train

t = metrics.thresholds[0] (prediction), t_y = metrics.target_thresholds[0]
(target binarisation). On 0/1 values the squared error is the 0/1 error, so
mse_binary is the binary error rate. The baselines are what a model that
learned nothing — the same prediction for every pixel — would score; a model
is only useful where it lies clearly below them.

Outputs in ``out_dir``:
    scene_errors.json    all numbers, per scene × split plus pixel-weighted overall
    error_<scene>.tif    (p − y)² on the scene grid, NaN where no valid label / no tile
    error_<scene>.png    heatmap of the same, tile outlines by split
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from rasterio.transform import from_origin
from torch.utils.data import DataLoader, Dataset

from data.channels import ChannelSpec
from data.dataset import CrownDataset
from deadwood_spectral.align import DATE_RE
from utils.nodata import valid_target

SPLITS = ("train", "val", "test")
SINGLE_SCENE = "scene"

# Sequential magnitude ramp: one hue (blue), light -> dark. The split outlines
# carry identity with categorical slots 2-4 (orange, aqua, yellow) because slot
# 1's blue is taken by the magnitude; the line style is a second encoding.
_ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
_SPLIT_STYLE = {
    "train": ("#eb6834", "-"),
    "val": ("#1baf7a", "--"),
    "test": ("#eda100", ":"),
}
_NAN_COLOR = "#f9f9f7"
_INK = "#0b0b0b"
_INK_2 = "#52514e"


def scene_of(stem: str) -> str:
    """Scene key of a tile stem: its YYYYMMDD prefix, or "scene" without one."""
    match = DATE_RE.match(stem)
    return match.group(1) if match else SINGLE_SCENE


@dataclass
class _Sums:
    """Running sums for one scene × split."""

    sq: float = 0.0
    bin_err: float = 0.0
    sq_base: float = 0.0
    bin_base: float = 0.0
    n: int = 0

    def add(self, other: "_Sums") -> None:
        self.sq += other.sq
        self.bin_err += other.bin_err
        self.sq_base += other.sq_base
        self.bin_base += other.bin_base
        self.n += other.n

    def as_dict(self) -> dict:
        if self.n == 0:
            return {"mse": None, "mse_baseline": None, "mse_binary": None,
                    "mse_binary_baseline": None, "n_px": 0}
        return {
            "mse": self.sq / self.n,
            "mse_baseline": self.sq_base / self.n,
            "mse_binary": self.bin_err / self.n,
            "mse_binary_baseline": self.bin_base / self.n,
            "n_px": self.n,
        }


@dataclass
class _Tile:
    split: str
    dataset: CrownDataset
    index: int
    stem: str
    row: int = 0
    col: int = 0
    height: int = 0
    width: int = 0


@dataclass
class _SceneGrid:
    tiles: list[_Tile] = field(default_factory=list)
    height: int = 0
    width: int = 0
    transform: object = None
    crs: object = None


class _TileList(Dataset):
    """Tiles of one scene across splits, in a fixed order; yields the list index."""

    def __init__(self, tiles: list[_Tile]):
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int):
        tile = self.tiles[index]
        image, mask = tile.dataset[tile.index]
        return image, mask, index


def train_baseline(data_root: Path, target_threshold: float) -> tuple[float, bool]:
    """(c, majority) from the valid train targets: mean value and majority class."""
    total = n = n_pos = 0
    for path in sorted((data_root / "train" / "masks").glob("*_mask.tif")):
        with rasterio.open(path) as src:
            y = src.read(1)
        y = y[valid_target(y)]
        total += float(y.sum())
        n += y.size
        n_pos += int((y > target_threshold).sum())
    if n == 0:
        raise RuntimeError(f"no valid train targets under {data_root / 'train' / 'masks'}")
    return total / n, n_pos > n / 2


def _scene_grids(datasets: dict[str, CrownDataset]) -> dict[str, _SceneGrid]:
    """Group the tiles of all splits by scene and place them on a common grid."""
    grids: dict[str, _SceneGrid] = defaultdict(_SceneGrid)
    for split, ds in datasets.items():
        for i, stem in enumerate(ds.stems):
            grids[scene_of(stem)].tiles.append(_Tile(split, ds, i, stem))

    for grid in grids.values():
        bounds = []
        for tile in grid.tiles:
            with rasterio.open(tile.dataset.mask_dir / f"{tile.stem}_mask.tif") as src:
                bounds.append((src.transform, src.height, src.width))
                grid.crs = src.crs
        res_x = bounds[0][0].a
        res_y = -bounds[0][0].e
        left = min(t.c for t, _, _ in bounds)
        top = max(t.f for t, _, _ in bounds)
        right = max(t.c + w * t.a for t, _, w in bounds)
        bottom = min(t.f + h * t.e for t, h, _ in bounds)
        grid.width = int(round((right - left) / res_x))
        grid.height = int(round((top - bottom) / res_y))
        grid.transform = from_origin(left, top, res_x, res_y)
        for tile, (t, h, w) in zip(grid.tiles, bounds):
            tile.row = int(round((top - t.f) / res_y))
            tile.col = int(round((t.c - left) / res_x))
            tile.height, tile.width = h, w
    return dict(sorted(grids.items()))


def _block_nanmean(arr: np.ndarray, max_side: int) -> np.ndarray:
    """Downsample by an integer factor with a NaN-aware block mean."""
    f = max(1, int(np.ceil(max(arr.shape) / max_side)))
    if f == 1:
        return arr
    h, w = arr.shape
    hc, wc = -(-h // f), -(-w // f)
    padded = np.full((hc * f, wc * f), np.nan, dtype=np.float32)
    padded[:h, :w] = arr
    blocks = padded.reshape(hc, f, wc, f)
    valid = np.isfinite(blocks)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan).astype(np.float32)


def _plot_scene(
    scene: str,
    mosaic: np.ndarray,
    grid: _SceneGrid,
    per_split: dict[str, dict],
    vmax: float,
    path: Path,
    max_side: int = 2000,
) -> None:
    small = _block_nanmean(mosaic, max_side)
    scale = small.shape[0] / mosaic.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10 * small.shape[0] / max(small.shape[1], 1) + 1.2))
    cmap = _ERROR_CMAP.with_extremes(bad=_NAN_COLOR, over=_ERROR_CMAP(1.0))
    im = ax.imshow(small, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")

    for tile in grid.tiles:
        color, style = _SPLIT_STYLE.get(tile.split, (_INK_2, "-"))
        ax.add_patch(plt.Rectangle(
            (tile.col * scale - 0.5, tile.row * scale - 0.5),
            tile.width * scale, tile.height * scale,
            fill=False, edgecolor=color, linestyle=style, linewidth=1.0,
        ))

    handles = []
    for split, stats in per_split.items():
        color, style = _SPLIT_STYLE.get(split, (_INK_2, "-"))
        mse = stats["mse"]
        label = f"{split}: MSE {mse:.4f}" if mse is not None else f"{split}: —"
        handles.append(Line2D([0], [0], color=color, linestyle=style, linewidth=2, label=label))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, labelcolor=_INK, title="tiles by split")
    cbar = fig.colorbar(im, ax=ax, extend="max", shrink=0.6, pad=0.02,
                        location="bottom")
    cbar.set_label(f"squared error (p − y)², scale fixed at 0–{vmax:g}", color=_INK_2)
    cbar.outline.set_visible(False)
    ax.set_title(f"Error map — {scene}", color=_INK, loc="left")
    ax.set_axis_off()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def evaluate_scene_errors(
    model: torch.nn.Module,
    data_root: Path,
    spec: ChannelSpec,
    norm_stats: dict | None,
    device: torch.device,
    out_dir: Path,
    splits: list[str] | tuple[str, ...] = SPLITS,
    threshold: float = 0.5,
    target_threshold: float = 0.5,
    batch_size: int = 4,
    num_workers: int = 0,
    amp: bool = True,
    vmax: float = 0.25,
    checkpoint: str | None = None,
) -> dict:
    """Predict every tile once; MSE / binary error per scene × split; error maps.

    See the module docstring for the definitions. Returns the dict that is also
    written to ``out_dir/scene_errors.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    c, majority = train_baseline(data_root, target_threshold)
    print(f"Baseline (train): constant c = {c:.4f}, majority class = {int(majority)}")

    datasets = {
        s: CrownDataset(data_root / s, spec, norm_stats=norm_stats)
        for s in splits
        if (data_root / s / "images").is_dir()
    }
    grids = _scene_grids(datasets)
    use_amp = amp and device.type == "cuda"
    model.eval()

    scenes: dict[str, dict[str, dict]] = {}
    overall: dict[str, _Sums] = {s: _Sums() for s in datasets}
    for scene, grid in grids.items():
        mosaic = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        sums: dict[str, _Sums] = {s: _Sums() for s in datasets}
        loader = DataLoader(
            _TileList(grid.tiles), batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        with torch.no_grad():
            for images, masks, idx in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(images.to(device, non_blocking=True))
                probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
                targets = masks.numpy()[:, 0]
                for p, y, i in zip(probs, targets, idx.tolist()):
                    tile = grid.tiles[i]
                    valid = valid_target(y)
                    sq = (p - y) ** 2
                    t_bin = y > target_threshold
                    s = sums[tile.split]
                    s.sq += float(sq[valid].sum())
                    s.bin_err += float(((p >= threshold) != t_bin)[valid].sum())
                    s.sq_base += float(((c - y) ** 2)[valid].sum())
                    s.bin_base += float((t_bin != majority)[valid].sum())
                    s.n += int(valid.sum())
                    h = min(tile.height, grid.height - tile.row)
                    w = min(tile.width, grid.width - tile.col)
                    mosaic[tile.row:tile.row + h, tile.col:tile.col + w] = np.where(
                        valid, sq, np.nan
                    )[:h, :w]

        scenes[scene] = {s: v.as_dict() for s, v in sums.items()}
        for s, v in sums.items():
            overall[s].add(v)

        profile = dict(
            driver="GTiff", dtype="float32", width=grid.width, height=grid.height, count=1,
            crs=grid.crs, transform=grid.transform, nodata=float("nan"), compress="lzw",
        )
        with rasterio.open(out_dir / f"error_{scene}.tif", "w", **profile) as dst:
            dst.write(mosaic[np.newaxis])
        _plot_scene(scene, mosaic, grid, scenes[scene], vmax, out_dir / f"error_{scene}.png")
        print(f"  {scene}: {len(grid.tiles)} tiles -> error_{scene}.tif / .png")

    result = {
        "checkpoint": checkpoint,
        "threshold": threshold,
        "target_threshold": target_threshold,
        "baseline": {"constant": c, "majority_class": int(majority)},
        "scenes": scenes,
        "overall": {s: v.as_dict() for s, v in overall.items()},
    }
    (out_dir / "scene_errors.json").write_text(json.dumps(result, indent=2))
    _print_table(result)
    return result


def _print_table(result: dict) -> None:
    def fmt(v):
        return f"{v:>10.4f}" if v is not None else f"{'—':>10}"

    print(f"\n=== Scene errors (pred ≥ {result['threshold']}, tgt > {result['target_threshold']}) ===")
    header = (f"{'scene':>10}  {'split':>5}  {'mse':>10}  {'baseline':>10}  "
              f"{'mse_bin':>10}  {'baseline':>10}  {'n_px':>12}")
    print(header)
    print("-" * len(header))
    rows = [(scene, s, v) for scene, splits in result["scenes"].items() for s, v in splits.items()]
    rows += [("overall", s, v) for s, v in result["overall"].items()]
    for scene, split, v in rows:
        print(f"{scene:>10}  {split:>5}  {fmt(v['mse'])}  {fmt(v['mse_baseline'])}  "
              f"{fmt(v['mse_binary'])}  {fmt(v['mse_binary_baseline'])}  {v['n_px']:>12,}")
