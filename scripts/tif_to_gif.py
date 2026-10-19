"""Render a time series of masks on their RGB scenes to PNGs, then to a GIF.

Every file in DIR whose name ends with --suffix is one frame: a single-band mask
(1 = foreground) drawn in red over the RGB scene <prefix><rgb_suffix>. The RGB
is always the background. With --suffix equal to --rgb_suffix, the plain RGB
series is rendered.

With --persistent, the pixels that are foreground in every scene are drawn in
purple with a red outline on top of each frame. They are also written as a GeoTIFF
(persistent_<tag>.tif: 1 = in all scenes, 0 = not, 255 = nodata in any scene)
and as a map on the latest RGB scene (persistent_<tag>_map.png). All masks must
share one grid.

Frames are sorted by name, so date-prefixed scenes come out chronologically, and
each frame is labelled with the part of the name before the suffix.

RGB rasters are read decimated to --max_size px on the long side, so the
full-res stacks never land in RAM. Masks are downsampled so that an output pixel
is foreground if any pixel it covers is, which keeps thin deadwood visible.

Usage:
    uv run python scripts/tif_to_gif.py out/raw_deadwood/ms_timeseries_5cm_aligned \\
        --suffix _deadwood_mask.tif --persistent
    uv run python scripts/tif_to_gif.py out/raw_deadwood/ms_timeseries_5cm_aligned \\
        --suffix _rgb8.tif
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rasterio.enums import Resampling

MASK_COLOR = np.array([230, 30, 30], dtype=np.float32)
PERSISTENT_COLOR = np.array([160, 32, 240], dtype=np.float32)
PERSISTENT_OUTLINE = np.array([255, 0, 0], dtype=np.uint8)


def render_rgb(path: Path, max_size: int) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"{path.name}: RGB background needs 3 bands, has {src.count}")
        scale = min(1.0, max_size / max(src.width, src.height))
        shape = (3, round(src.height * scale), round(src.width * scale))
        data = src.read([1, 2, 3], out_shape=shape, resampling=Resampling.average)
        nodata = src.nodata
    rgb = np.moveaxis(data, 0, -1)
    if rgb.dtype != np.uint8:
        lo, hi = np.percentile(rgb[rgb != nodata] if nodata is not None else rgb, [2, 98])
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    return rgb


def read_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Full-res (foreground, valid) of a mask; masks are small enough for that."""
    with rasterio.open(path) as src:
        band, nodata = src.read(1), src.nodata
    valid = band != nodata if nodata is not None else np.ones(band.shape, bool)
    return (band == 1) & valid, valid


def downsample(fg: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # A box filter > 0 marks every output pixel that covers any foreground pixel.
    small = Image.fromarray(fg.astype(np.uint8) * 255).resize(size, Image.Resampling.BOX)
    return np.asarray(small) > 0


def paint(
    img: np.ndarray,
    fg: np.ndarray,
    color: np.ndarray,
    alpha: float,
    outline: np.ndarray | None = None,
    outline_px: int = 2,
) -> np.ndarray:
    fg = downsample(fg, (img.shape[1], img.shape[0]))
    out = img.astype(np.float32)
    out[fg] = (1 - alpha) * out[fg] + alpha * color
    out = out.astype(np.uint8)
    if outline is not None:
        # Ring of outline_px pixels just outside each patch, drawn opaque.
        grown = Image.fromarray(fg.astype(np.uint8) * 255).filter(
            ImageFilter.MaxFilter(2 * outline_px + 1)
        )
        out[(np.asarray(grown) > 0) & ~fg] = outline
    return out


def persistent_mask(paths: list[Path], out: Path) -> np.ndarray:
    """AND of all masks, written as GeoTIFF; returns the full-res foreground."""
    with rasterio.open(paths[0]) as ref:
        profile, transform, shape = ref.profile, ref.transform, ref.shape
    fg_all = np.ones(shape, bool)
    valid_all = np.ones(shape, bool)
    for path in paths:
        with rasterio.open(path) as src:
            if src.shape != shape or src.transform != transform:
                raise ValueError(f"{path.name}: grid differs from {paths[0].name}")
        fg, valid = read_mask(path)
        fg_all &= fg
        valid_all &= valid
    data = np.where(valid_all, fg_all, 255).astype(np.uint8)
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
    return fg_all


def pretty_date(prefix: str) -> str:
    m = re.match(r"(\d{4})(\d{2})(\d{2})", prefix)
    return f"{m[1]}-{m[2]}-{m[3]}" if m else prefix


def plot_map(
    rgb_path: Path, fg: np.ndarray, prefixes: list[str], max_size: int, alpha: float, out: Path
) -> None:
    with rasterio.open(rgb_path) as src:
        b, crs, res = src.bounds, src.crs, src.res
    img = render_rgb(rgb_path, max_size)
    img = paint(img, fg, PERSISTENT_COLOR, alpha, PERSISTENT_OUTLINE)
    area = fg.sum() * abs(res[0] * res[1])

    fig, ax = plt.subplots(figsize=(9, 9 * img.shape[0] / img.shape[1] + 0.8), dpi=150)
    ax.imshow(img, extent=(b.left, b.right, b.bottom, b.top))
    ax.set_title(
        f"Deadwood detected in all {len(prefixes)} scenes\n"
        f"{pretty_date(prefixes[0])} – {pretty_date(prefixes[-1])}"
        f", background {pretty_date(prefixes[-1])}",
        fontsize=11,
    )
    unit = crs.linear_units if crs else "units"
    ax.set_xlabel(f"Easting [{unit}]  ({crs.to_string() if crs else 'no CRS'})", fontsize=8)
    ax.set_ylabel(f"Northing [{unit}]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(
        handles=[
            Patch(
                facecolor="#a020f0",  # PERSISTENT_COLOR
                edgecolor="#ff0000",  # PERSISTENT_OUTLINE
                linewidth=1.5,
                label=f"deadwood in every scene ({area:,.0f} m²)",
            )
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def label(img: np.ndarray, lines: list[tuple[str, tuple[int, ...]]]) -> Image.Image:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=max(16, pil.height // 30))
    y = 10
    for text, color in lines:
        x0, y0, x1, y1 = draw.textbbox((10, y), text, font=font)
        draw.rectangle((x0 - 6, y0 - 6, x1 + 6, y1 + 6), fill=(0, 0, 0))
        draw.text((10, y), text, fill=color, font=font)
        y = y1 + 12
    return pil


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dir", type=Path, help="folder holding the rasters")
    p.add_argument(
        "--suffix",
        required=True,
        help="file name ending of the frame masks, e.g. _deadwood_mask.tif",
    )
    p.add_argument(
        "--rgb_suffix", default="_rgb8.tif", help="file name ending of the RGB background"
    )
    p.add_argument(
        "--persistent",
        action="store_true",
        help="also overlay, save and map the pixels that are foreground in all scenes",
    )
    p.add_argument("--alpha", type=float, default=0.7, help="mask opacity")
    p.add_argument("--max_size", type=int, default=1200, help="long side of the frames in px")
    p.add_argument("--duration", type=int, default=800, help="ms per frame")
    p.add_argument("--out_dir", type=Path, help="default: DIR/gif_<suffix stem>")
    args = p.parse_args()

    frames = sorted(f for f in args.dir.iterdir() if f.name.endswith(args.suffix))
    if not frames:
        raise SystemExit(f"no file in {args.dir} ends with {args.suffix!r}")
    has_mask = args.suffix != args.rgb_suffix
    if args.persistent and not has_mask:
        raise SystemExit("--persistent needs a mask suffix, not the RGB one")

    tag = Path(args.suffix).stem.strip("_")
    out_dir = args.out_dir or args.dir / f"gif_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefixes = [f.name.removesuffix(args.suffix) for f in frames]

    persistent = None
    if args.persistent:
        tif = out_dir / f"persistent_{tag}.tif"
        persistent = persistent_mask(frames, tif)
        print(f"wrote {tif}")
        png = out_dir / f"persistent_{tag}_map.png"
        rgb_last = args.dir / f"{prefixes[-1]}{args.rgb_suffix}"
        plot_map(rgb_last, persistent, prefixes, 2 * args.max_size, args.alpha, png)
        print(f"wrote {png}")

    images = []
    for f, prefix in zip(frames, prefixes, strict=True):
        img = render_rgb(args.dir / f"{prefix}{args.rgb_suffix}", args.max_size)
        lines = [(prefix, (255, 255, 255))]
        if has_mask:
            img = paint(img, read_mask(f)[0], MASK_COLOR, args.alpha)
            lines.append(("deadwood this date", tuple(MASK_COLOR.astype(int))))
        if persistent is not None:
            img = paint(img, persistent, PERSISTENT_COLOR, args.alpha, PERSISTENT_OUTLINE)
            lines.append(("deadwood in all scenes", tuple(PERSISTENT_COLOR.astype(int))))
        frame = label(img, lines)
        png = out_dir / f"{prefix}.png"
        frame.save(png)
        images.append(frame)
        print(f"wrote {png}")

    gif = out_dir / f"{tag}.gif"
    images[0].save(
        gif, save_all=True, append_images=images[1:], duration=args.duration, loop=0, optimize=True
    )
    print(f"wrote {gif} ({len(images)} frames)")


if __name__ == "__main__":
    main()
