"""Read the three inputs block by block, write combo and eco rasters plus area stats.

The crown mask is the reference grid; the other two must match it exactly,
since resampling would hide the kind of misregistration this step depends on.
"""

import csv
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from deadwood_spectral.grid import assert_matches_grid, load_reference_grid
from mask_fusion.codes import (
    COMBO_COLORS,
    COMBO_NAMES,
    N_CODES,
    NODATA,
    EcoClass,
    build_lookup,
    check_thresholds,
    combo_codes,
    hex_to_rgb,
    parse_classes,
)
from mask_fusion.figures import compile_tex, decision_tree_tex, legend_png, pdf_to_png

logger = logging.getLogger(__name__)

MASK_SUFFIX = "_deadwood_mask"


def default_stem(deadwood: str | Path) -> str:
    return Path(deadwood).stem.removesuffix(MASK_SUFFIX)


def _colormap(colors: Mapping[int, tuple[int, int, int]]) -> dict[int, tuple[int, int, int, int]]:
    cmap = {value: (*rgb, 255) for value, rgb in colors.items()}
    cmap[NODATA] = (0, 0, 0, 0)
    return cmap


def _write_stats(
    path: Path,
    combo_counts: np.ndarray,
    eco_pixels: Sequence[tuple[EcoClass, int]],
    pixel_area: float,
) -> None:
    valid = int(combo_counts[:N_CODES].sum())
    rows = [("combo", code, COMBO_NAMES[code], int(combo_counts[code])) for code in range(N_CODES)]
    rows += [("eco", e.value, e.name, pixels) for e, pixels in eco_pixels]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "value", "name", "pixels", "area_m2", "share_pct"])
        for layer, value, name, pixels in rows:
            writer.writerow(
                [
                    layer,
                    value,
                    name,
                    pixels,
                    round(pixels * pixel_area, 4),
                    round(100 * pixels / valid, 4),
                ]
            )


def run_fusion(
    crown: str | Path,
    deadwood: str | Path,
    ndsm: str | Path,
    height_m: Sequence[float],
    classes: Mapping,
    out_dir: str | Path,
    out_stem: str | None = None,
    chunk_rows: int = 512,
) -> dict[str, Path]:
    """Write {stem}_combo.tif, {stem}_eco.tif, {stem}_stats.csv, {stem}_legend.png and
    {stem}_decision_tree.tex (+ .pdf/.png when pdflatex/pdftoppm exist) into out_dir."""
    if len(height_m) != 2:
        raise ValueError(f"height_m must be [h1, h2], got {list(height_m)}")
    h1, h2 = (float(h) for h in height_m)
    check_thresholds(h1, h2)
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    eco = parse_classes(classes)
    lut = build_lookup(eco)

    grid = load_reference_grid(crown)
    out_dir = Path(out_dir)
    stem = out_stem or default_stem(deadwood)
    paths = {
        "combo": out_dir / f"{stem}_combo.tif",
        "eco": out_dir / f"{stem}_eco.tif",
        "stats": out_dir / f"{stem}_stats.csv",
    }
    profile = dict(
        driver="GTiff",
        dtype="uint8",
        count=1,
        height=grid.height,
        width=grid.width,
        crs=grid.crs,
        transform=grid.transform,
        nodata=NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    combo_counts = np.zeros(256, dtype=np.int64)
    with (
        rasterio.open(crown) as c_src,
        rasterio.open(deadwood) as d_src,
        rasterio.open(ndsm) as n_src,
    ):
        assert_matches_grid(d_src, grid, "deadwood")
        assert_matches_grid(n_src, grid, "ndsm")
        for name, src in (("crown", c_src), ("deadwood", d_src)):
            if src.nodata is not None and src.nodata != NODATA:
                raise ValueError(f"{name}: declared nodata {src.nodata} != {NODATA}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with (
            rasterio.open(paths["combo"], "w", **profile) as combo_dst,
            rasterio.open(paths["eco"], "w", **profile) as eco_dst,
        ):
            for row in range(0, grid.height, chunk_rows):
                win = Window(
                    col_off=0,
                    row_off=row,
                    width=grid.width,
                    height=min(chunk_rows, grid.height - row),
                )
                combo = combo_codes(
                    c_src.read(1, window=win),
                    d_src.read(1, window=win),
                    n_src.read(1, window=win),
                    h1,
                    h2,
                    ndsm_nodata=n_src.nodata,
                )
                combo_dst.write(combo, 1, window=win)
                eco_dst.write(lut[combo], 1, window=win)
                combo_counts += np.bincount(combo.ravel(), minlength=256)
            combo_dst.write_colormap(
                1, _colormap({code: hex_to_rgb(col) for code, col in enumerate(COMBO_COLORS)})
            )
            eco_dst.write_colormap(1, _colormap({e.value: e.color for e in eco}))

    valid = int(combo_counts[:N_CODES].sum())
    if valid == 0:
        raise ValueError("no pixel is valid in all three inputs - disjoint footprints?")
    pixel_area = abs(grid.transform.a * grid.transform.e)
    eco_pixels = [(e, int(combo_counts[list(e.codes)].sum())) for e in eco]
    _write_stats(paths["stats"], combo_counts, eco_pixels, pixel_area)

    paths["legend"] = legend_png(
        [(e, px * pixel_area, 100 * px / valid) for e, px in eco_pixels],
        out_dir / f"{stem}_legend.png",
    )
    tex = decision_tree_tex(eco, h1, h2)
    paths["tree_tex"] = out_dir / f"{stem}_decision_tree.tex"
    paths["tree_tex"].write_text(tex, encoding="utf-8")
    pdf = compile_tex(tex, out_dir / f"{stem}_decision_tree.pdf")
    if pdf is not None:
        paths["tree_pdf"] = pdf
        png = pdf_to_png(pdf)
        if png is not None:
            paths["tree_png"] = png
    logger.info(
        "fused %d valid px (%.1f m2) into %d eco classes",
        valid,
        valid * pixel_area,
        len(eco),
    )
    return paths
