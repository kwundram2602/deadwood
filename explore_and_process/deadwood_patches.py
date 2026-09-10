"""Crown splitting and crop cutting for the deadwood fine-tuning path.

Separate from the crown pipeline's tile_patches on purpose: that one grid-tiles
a whole scene, which here would yield ~20 tiles of which most hold no crown at
all. Crops are centred on crowns instead, one per crown.
"""

import math
from collections import defaultdict

import numpy as np
import rasterio
from rasterio import features, windows
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_filter

from explore_and_process.rasterize_crowns import rasterize_binary
from utils.nodata import (
    MASK_OUTSIDE,
    MASK_RASTER_NODATA,
    MASK_UNLABELLED,
    valid_target,
)

SPLIT_NAMES = ("train", "val", "test")


def deadwood_scene_mask(
    pos_geoms,
    neg_geoms,
    h,
    w,
    transform,
    *,
    sigma_pos,
    sigma_neg,
    pos_threshold,
    neg_threshold,
    footprint=None,
):
    """One whole-scene mask carrying both label classes.

    The two classes are treated asymmetrically on purpose. Positives keep their
    blur as a soft target and reach past the drawn edge, so a crown boundary
    that is a little wrong is not a hard error. Negatives are thresholded high,
    which erodes them back to a confident core well inside their own edge. What
    is left between the two is MASK_UNLABELLED and draws no gradient.

    That band is never empty, even for polygons drawn edge to edge, which is the
    whole reason sigma_neg exceeds sigma_pos. A hard 1.0 adjacent to a hard 0.0
    is what the deleted ring negatives produced, and it taught the model to
    shrink predictions it had right.

    Application order is the contract: negatives, then positives over them, then
    MASK_OUTSIDE over everything.
    """
    pos_binary = rasterize_binary(pos_geoms, h, w, transform)
    neg_binary = rasterize_binary(neg_geoms, h, w, transform)
    pos = gaussian_filter(pos_binary, sigma=sigma_pos)
    neg = gaussian_filter(neg_binary, sigma=sigma_neg)

    mask = np.full((h, w), MASK_UNLABELLED, dtype=np.float32)
    mask[neg >= neg_threshold] = 0.0
    # `| pos_binary > 0` keeps a polygon narrower than its own sigma labelled:
    # its blurred peak can fall below pos_threshold, but it was still drawn.
    labelled = (pos >= pos_threshold) | (pos_binary > 0)
    mask[labelled] = pos[labelled]
    if footprint is not None:
        mask[~footprint] = MASK_OUTSIDE
    return mask


def tile_stats(mask_crop):
    """Per-tile pixel accounting, keeping the two sentinels apart.

    ``outside_frac`` is about the imagery being absent; ``labelled_px`` is about
    a label being present. They are separate numbers because they drive separate
    filters — see keep_tile.
    """
    valid = valid_target(mask_crop)
    return {
        "outside_frac": float(np.mean(mask_crop == MASK_OUTSIDE)),
        "labelled_px": int(valid.sum()),
        "pos_px": int((valid & (mask_crop > 0.0)).sum()),
        "neg_px": int((mask_crop == 0.0).sum()),
    }


def tile_kind(stats):
    """Which label classes a tile carries: both, pos, neg, or empty."""
    if stats["pos_px"] and stats["neg_px"]:
        return "both"
    if stats["pos_px"]:
        return "pos"
    if stats["neg_px"]:
        return "neg"
    return "empty"


def keep_tile(stats, min_labelled_px, max_outside_frac):
    """Two independent filters, never one combined "noData fraction".

    Deliberately does NOT require positives: a tile holding only background is
    the false-positive-suppression signal, and dropping those is exactly the
    failure this pipeline was rewritten to fix.
    """
    return stats["labelled_px"] >= min_labelled_px and stats["outside_frac"] <= max_outside_frac


def scan_tiles(mask, size):
    """tile id -> stats for every window of a disjoint grid over ``mask``.

    The last row and column overhang the array when the shape is not a multiple
    of ``size``; array_window pads them with MASK_OUTSIDE so every tile stays
    exactly size x size. Ids are zero-padded so lexical order is grid order.
    """
    h, w = mask.shape
    n_rows = math.ceil(h / size)
    n_cols = math.ceil(w / size)
    pad_r = len(str(n_rows - 1))
    pad_c = len(str(n_cols - 1))

    scan = {}
    for row in range(n_rows):
        for col in range(n_cols):
            window = windows.Window(
                col_off=col * size, row_off=row * size, width=size, height=size
            )
            stats = tile_stats(array_window(mask, window, fill=MASK_OUTSIDE))
            stats["row"] = row
            stats["col"] = col
            scan[f"{row:0{pad_r}d}_{col:0{pad_c}d}"] = stats
    return scan


def assign_splits(kinds, fractions, seed=0, stratify=True):
    """Deal tile ids into train/val/test.

    Random rather than spatial: on this site the crowns cluster along one end of
    the principal axis, so a spatial band cut hands one split almost every crown
    tile and another almost none. Grid tiles are disjoint on the ground, so the
    overlap that made a random split unsafe for crown-centred crops is gone.

    ``stratify`` deals each kind separately, so a split cannot come up short of
    one class by luck. Ids are sorted before shuffling so the result depends on
    the seed alone, never on dict ordering.
    """
    total = sum(float(fractions[name]) for name in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"split fractions must sum to 1.0, got {total} "
            f"({', '.join(f'{n}={float(fractions[n])}' for n in SPLIT_NAMES)})"
        )

    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for tile_id in sorted(kinds):
        groups[kinds[tile_id] if stratify else "all"].append(tile_id)

    splits = {}
    for _, tile_ids in sorted(groups.items()):
        rng.shuffle(tile_ids)
        n = len(tile_ids)
        n_train = min(int(round(n * float(fractions["train"]))), n)
        n_val = min(int(round(n * float(fractions["val"]))), n - n_train)
        for i, tile_id in enumerate(tile_ids):
            if i < n_train:
                splits[tile_id] = "train"
            elif i < n_train + n_val:
                splits[tile_id] = "val"
            else:
                splits[tile_id] = "test"
    return splits


def split_crowns(crowns, n_train, n_val, n_test, mode="spatial", seed=0):
    """Assign crown fids to train/val/test.

    ``spatial`` orders crowns along the principal axis of their centroids and
    cuts contiguous bands, so a held-out crown sits at one end of the site
    rather than interleaved with training crowns. ``random`` is a seeded
    shuffle, for checking how much the spatial arrangement is worth.
    """
    total = n_train + n_val + n_test
    if total != len(crowns):
        raise ValueError(
            f"split counts sum to {total} but got {len(crowns)} crowns "
            f"(train {n_train} + val {n_val} + test {n_test})"
        )

    fids = np.asarray([int(f) for f in crowns.index])
    if mode == "spatial":
        xy = np.c_[crowns.geometry.centroid.x, crowns.geometry.centroid.y]
        centred = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        axis = vt[0]
        # SVD leaves the component's sign arbitrary; pin it so the same crowns
        # land in the same split across numpy versions and row orderings.
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        order = np.argsort(centred @ axis, kind="stable")
    elif mode == "random":
        order = np.random.default_rng(seed).permutation(len(fids))
    else:
        raise ValueError(f"mode must be 'spatial' or 'random', got {mode!r}")

    ordered = fids[order]
    return {
        "train": ordered[:n_train].tolist(),
        "val": ordered[n_train : n_train + n_val].tolist(),
        "test": ordered[n_train + n_val :].tolist(),
    }


def crown_crop_window(geom, transform, crop_size, jitter=(0, 0)):
    """A crop_size square window centred on the crown's centroid pixel."""
    row, col = rowcol(transform, geom.centroid.x, geom.centroid.y)
    half = crop_size // 2
    return windows.Window(
        col_off=int(col) - half + int(jitter[1]),
        row_off=int(row) - half + int(jitter[0]),
        width=crop_size,
        height=crop_size,
    )


def array_window(arr, window, fill):
    """Boundless slice of an in-memory 2D array, padded with ``fill``.

    rasterio does this for datasets; the split masks live in RAM, so crops that
    run off the scene edge need the same treatment here.
    """
    out = np.full((int(window.height), int(window.width)), fill, dtype=arr.dtype)
    r0, c0 = int(window.row_off), int(window.col_off)
    src_r0, src_c0 = max(r0, 0), max(c0, 0)
    src_r1 = min(r0 + int(window.height), arr.shape[0])
    src_c1 = min(c0 + int(window.width), arr.shape[1])
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return out
    out[src_r0 - r0 : src_r1 - r0, src_c0 - c0 : src_c1 - c0] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


def ring_negatives(mask, geoms, transform, buffer_m):
    """Turn unlabelled pixels within buffer_m of a crown into hard 0.0.

    soft_mask_from_geoms emits no 0.0 at all: beyond 4*sigma the Gaussian is
    exactly zero, which falls under nodata_threshold and becomes
    MASK_UNLABELLED. Without this step the lowest target the model ever sees is
    nodata_threshold, so it is never told that anything is *not* deadwood.

    Only MASK_UNLABELLED pixels are touched. Crown pixels, the soft falloff, and
    off-footprint pixels keep what they have — and everything beyond the buffer
    stays unlabelled, which is the whole point of ring-only negatives under
    sparse labelling.
    """
    buffered = [g.buffer(buffer_m) for g in geoms if g is not None and not g.is_empty]
    if not buffered:
        return mask
    near = features.rasterize(
        [(g, 1) for g in buffered],
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    out = mask.copy()
    out[near & (mask == MASK_UNLABELLED)] = 0.0
    return out


def _write(path, data, transform, crs, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[-2],
        width=data.shape[-1],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(data)


def cut_split(
    rgb_path, mask, transform, crs, crowns, out_dir, crop_size, n_jitter=1, jitter_px=0, seed=0
):
    """Write one image/mask patch per crown (times n_jitter) under out_dir."""
    rng = np.random.default_rng(seed)
    written = 0
    with rasterio.open(rgb_path) as src:
        for fid, geom in zip(crowns.index, crowns.geometry):
            for rep in range(n_jitter):
                jitter = (0, 0)
                if rep > 0 and jitter_px > 0:
                    jitter = tuple(rng.integers(-jitter_px, jitter_px + 1, size=2).tolist())
                win = crown_crop_window(geom, transform, crop_size, jitter)
                image = src.read((1, 2, 3), window=win, boundless=True, fill_value=0)
                mask_crop = array_window(mask, win, fill=MASK_OUTSIDE)
                pt = windows.transform(win, transform)

                stem = f"{int(fid)}" if n_jitter == 1 else f"{int(fid)}_j{rep}"
                _write(out_dir / "images" / f"{stem}.tif", image, pt, crs, 0)
                _write(
                    out_dir / "masks" / f"{stem}_mask.tif",
                    mask_crop[None, ...],
                    pt,
                    crs,
                    MASK_RASTER_NODATA,
                )
                written += 1
    return written
