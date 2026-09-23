"""
rasterize_crowns.py

Rasterise crown polygons to a soft training mask and (optionally)
batch-export band-selected, normalised, resampled MS images.

Steps:
  1. Load crown polygons, keep only son/soff
  2. Reproject polygons to raster CRS
  3. Optionally erode and round the polygons, then rasterize to a binary
     mask at target GSD
  4. Gaussian blur for soft crown boundaries, optionally rescaled so the
     falloff bottoms out at `soft_floor` instead of 0
  5. Set noData=255 for pixels inside the footprint but far from any crown,
     and noData=-1 for pixels outside the recorded scene footprint
  6. Save mask
  7. (optional) Resample all OM tifs in --raster_dir to target GSD,
     select 4 MS bands, normalise to [0,1], save as float32 with the
     out-of-footprint pixels left as NaN (nodata=nan)

Usage (config-driven; sources replace the old numeric `bands:` list):
  python explore_and_process/rasterize_crowns.py --config configs/preprocess/preprocess.yaml

  # rasterize.sources (in configs/preprocess/preprocess.yaml):
  #   sources:
  #     - path: data/raster/20230824_..._OM_RGB.tif
  #       bands: [1, 2, 3]
  #       names: [red, green, blue]
  #     - path: data/raster/20230824_..._OM_MS.tif
  #       bands: [1, 2, 3, 4]
  #       names: [green_ms, red_ms, rededge, nir]
"""

# python explore_and_process/rasterize_crowns.py \\     --crowns  datafiles/crown_poly/2_crown_main_20260409_editLP.gpkg --reference datafiles/raster/20260313/20260313_Airport_Main_MAVICM3MFIXEDM3M_tile001_OM_shift.tif --out_mask  datafiles/process_out/crown_mask.tif --raster_dir    data/raster --out_image_dir explore_and_process/out/images --target_gsd 0.05
import argparse
import json
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from omegaconf import OmegaConf
from rasterio.enums import Resampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter

from utils.nodata import MASK_OUTSIDE, MASK_RASTER_NODATA, MASK_UNLABELLED

logger = logging.getLogger(__name__)

# Only these crown categories map to class=1; everything else is excluded
INCLUDE_CATEGORIES = {"son", "soff"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def target_grid(src, gsd):
    """Return (height, width, transform) for a resampled grid at gsd metres."""
    h = int(round(src.height * src.res[0] / gsd))
    w = int(round(src.width * src.res[1] / gsd))
    return h, w, from_bounds(*src.bounds, w, h)


def assert_same_bounds(src, reference_bounds, tol: float = 1e-6) -> None:
    """Fail if a source raster does not share the reference extent.

    read_scaled_bands resamples a source's own extent onto the target shape; it
    does not reproject onto the target transform. That is correct only when the
    source and the reference cover the same ground. A differently-bounded source
    would be silently shifted while keeping a correct-looking georeference.
    """
    deltas = [abs(a - b) for a, b in zip(src.bounds, reference_bounds)]
    if max(deltas) > tol:
        raise ValueError(
            f"{src.name}: bounds {tuple(src.bounds)} differ from the reference "
            f"{tuple(reference_bounds)} by up to {max(deltas):.3f} m. "
            f"read_scaled_bands cannot reproject; use deadwood_spectral.align instead."
        )


def write_tif(path, data, transform, crs, nodata=None, descriptions=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis]
    profile = dict(
        driver="GTiff",
        dtype="float32",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------


def reshape_polygons(geoms, erode: float = 0.0, round_radius: float = 0.0):
    """Shrink and round crown polygons before they are burned to raster.

    Hand-digitised crowns are angular: straight runs between clicked vertices,
    plus the occasional thin spike where one vertex landed on a branch. Both are
    artefacts of drawing rather than shape the model should learn.

    ``erode`` shrinks every polygon by that distance (CRS units — metres in a
    UTM scene). ``round_radius`` runs a morphological opening followed by a
    closing — buffer(-r), buffer(2r), buffer(-r), all with round joins — which
    rounds convex corners, cuts spikes thinner than 2r, and fills notches while
    leaving the area roughly intact (a square loses (4 - π)r² to its corners).

    Invalid geometries pass through untouched: buffering them yields garbage,
    and rasterize_binary already reports and skips them.

    Polygons that vanish are dropped with a warning — their pixels fall back to
    MASK_UNLABELLED rather than being taught as background, but a label is gone,
    so the count is the signal that the distance is too large. Rounding can also
    pinch a polygon into two parts; both are kept.
    """
    if erode < 0.0 or round_radius < 0.0:
        raise ValueError(
            f"erode and round_radius must be >= 0; got {erode}, {round_radius}"
        )
    if erode == 0.0 and round_radius == 0.0:
        return geoms

    kept, index, vanished = [], [], []
    for idx, geom in geoms.items():
        if geom is None or not geom.is_valid:
            kept.append(geom)
            index.append(idx)
            continue
        shaped = geom
        if erode > 0.0:
            shaped = shaped.buffer(-erode, join_style="round")
        if round_radius > 0.0:
            r = round_radius
            shaped = (
                shaped.buffer(-r, join_style="round")
                .buffer(2 * r, join_style="round")
                .buffer(-r, join_style="round")
            )
        if shaped.is_empty:
            vanished.append(idx)
            continue
        kept.append(shaped)
        index.append(idx)

    if vanished:
        shown = ", ".join(str(i) for i in vanished[:10])
        more = f", ... (+{len(vanished) - 10})" if len(vanished) > 10 else ""
        print(
            f"  [WARN] reshape_polygons: {len(vanished)} of {len(geoms)} polygon(s) "
            f"vanished (erode={erode} m, round_radius={round_radius} m); "
            f"index: {shown}{more}"
        )
    return gpd.GeoSeries(kept, index=index, crs=geoms.crs)


def rasterize_binary(geoms, h, w, transform):
    """Burn geometries into a binary raster: inside = 1.0, background = 0.0.

    Split out of soft_mask_from_geoms because the deadwood mask builds two of
    these — one per label class — and needs each binary raster as well as its
    blur: a polygon narrower than its own sigma blurs to a peak below any
    sensible threshold, and the binary is what keeps it labelled anyway.

    An empty geometry list yields all zeros rather than raising, so a scene with
    only one of the two classes digitised is a valid input.
    """
    geoms = list(geoms)
    shapes = [(geom, 1.0) for geom in geoms if geom is not None and geom.is_valid]
    n_skipped = len(geoms) - len(shapes)
    if n_skipped:
        print(f"  [WARN] rasterize_binary skipped {n_skipped} null/invalid geometry(ies)")
    if not shapes:
        return np.zeros((h, w), dtype=np.float32)
    return rio_rasterize(shapes, out_shape=(h, w), transform=transform, fill=0.0, dtype="float32")


def rescale_soft_floor(
    soft: np.ndarray, nodata_threshold: float, soft_floor: float | None
) -> np.ndarray:
    """Stretch the labelled blur range [nodata_threshold, 1] onto [soft_floor, 1].

    The Gaussian falloff reaches 0 at the outer edge of a crown, which labels
    the rim of every polygon as confident background — a statement the blur was
    never meant to make. Lifting the bottom of the range keeps the rim a weak
    *positive*: the gradient stays monotonic everywhere, only its minimum moves.

    Must run before the noData sentinels are written, because afterwards the
    values that decide what is unlabelled no longer exist. None disables it.
    """
    if soft_floor is None:
        return soft
    if not 0.0 <= soft_floor < 1.0:
        raise ValueError(
            f"soft_floor must be in [0, 1) or null; got {soft_floor}"
        )
    scaled = soft_floor + (1.0 - soft_floor) * (soft - nodata_threshold) / (
        1.0 - nodata_threshold
    )
    # Polygons narrower than their own sigma peak below nodata_threshold and
    # would map under the floor; the floor is a floor for them too.
    return np.clip(scaled, soft_floor, 1.0).astype(np.float32)


def soft_mask_from_geoms(
    geoms, h, w, transform, sigma, nodata_threshold, footprint=None, soft_floor=None
):
    """Rasterize geometries → Gaussian blur → noData sentinels.

    ``footprint`` is the boolean scene footprint from the image sources. Pixels
    outside it get MASK_OUTSIDE (the drone never saw that ground); pixels
    inside it that no polygon reaches get MASK_UNLABELLED. Keeping the two
    apart is what lets predict suppress output beyond the flight extent — with
    one sentinel, "outside the scene" and "unlabelled background" are
    indistinguishable and the model is never told the difference.

    ``soft_floor`` is the value the blur is allowed to sink to at most; see
    rescale_soft_floor.
    """
    binary = rasterize_binary(geoms, h, w, transform)

    soft = gaussian_filter(binary, sigma=sigma)
    # Pixels outside all polygons that received no Gaussian bleed-over are
    # unlabelled: valid imagery, no statement about membership. Decided on the
    # raw blur, before the floor lifts every labelled value above it.
    unlabelled = (binary == 0) & (soft < nodata_threshold)
    soft = rescale_soft_floor(soft, nodata_threshold, soft_floor)
    soft[unlabelled] = MASK_UNLABELLED
    if footprint is not None:
        soft[~footprint] = MASK_OUTSIDE

    n_crown = int(np.sum((soft > 0) & (soft < MASK_UNLABELLED)))
    soft_zero = int(np.sum(soft == 0.0))
    n_unlabelled = int(np.sum(soft == MASK_UNLABELLED))
    n_outside = int(np.sum(soft == MASK_OUTSIDE))
    print(
        f"  Crown: {n_crown:,}  Soft == 0.0: {soft_zero:,}  "
        f"unlabelled: {n_unlabelled:,}  outside footprint: {n_outside:,}"
    )
    return soft


def build_mask(
    crowns_paths, src, h, w, transform, sigma, nodata_threshold,
    footprint=None, soft_floor=None, erode=0.0, round_radius=0.0,
):
    """Load son/soff crown polygons and rasterize them to a soft training mask."""
    gdfs = [gpd.read_file(p) for p in crowns_paths]
    gdf = pd.concat(gdfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdfs[0].crs)
    gdf = gdf[gdf["crown_category"].isin(INCLUDE_CATEGORIES)].to_crs(src.crs)
    print(f"  {len(gdf)} crown polygons (son/soff) from {len(crowns_paths)} file(s)")
    if erode > 0.0 or round_radius > 0.0:
        print(f"  reshaping polygons: erode={erode} m, round_radius={round_radius} m")
    geoms = reshape_polygons(gdf.geometry, erode=erode, round_radius=round_radius)
    return soft_mask_from_geoms(
        geoms, h, w, transform, sigma, nodata_threshold,
        footprint=footprint, soft_floor=soft_floor,
    )


def read_scaled_bands(path, bands, h, w):
    """Read selected bands, resample to (h, w), scale uint16-range to [0,1]."""
    with rasterio.open(path) as src:
        data = src.read(
            indexes=bands, out_shape=(len(bands), h, w), resampling=Resampling.bilinear
        ).astype(np.float32)
    data /= 65535.0  # uint16-range → [0, 1]
    data = np.where(np.isnan(data), 0.0, data)
    # Sensor/calibration artifacts can produce physically impossible
    # reflectance (hot pixels up to ~1e23 in the raw mosaic); clip to [0,1]
    n_clipped = int(np.sum((data < 0.0) | (data > 1.0)))
    if n_clipped:
        print(f"  [WARN] clipped {n_clipped} out-of-range pixel value(s) to [0,1]")
    np.clip(data, 0.0, 1.0, out=data)
    return data


def read_source_footprint(path, bands, h, w):
    """Boolean (h, w) footprint of one source: True where every band has data.

    Read with nearest-neighbour resampling on purpose — the bilinear read used
    for the pixel values would smear the footprint edge into a soft fringe of
    half-valid pixels, and a footprint has to be a crisp yes/no.
    """
    with rasterio.open(path) as src:
        masks = src.read_masks(
            indexes=bands,
            out_shape=(len(bands), h, w),
            resampling=Resampling.nearest,
        )
    return np.all(masks > 0, axis=0)


def scene_footprint(specs, h, w):
    """Intersection of every source's footprint on the target grid.

    Intersection, not union: a pixel the model is asked to predict on must have
    real data in *all* its input bands. With RGB and MS mosaics whose extents
    differ slightly, a union would feed the model zero-filled bands at the
    fringe and call the result valid.
    """
    footprint = np.ones((h, w), dtype=bool)
    for path, bands, _ in specs:
        footprint &= read_source_footprint(path, bands, h, w)
    return footprint


def validate_sources(sources, raster_dir=None):
    """Check a rasterize.sources config list; return the combined channel names."""
    if not sources:
        raise ValueError("rasterize.sources must list at least one source")
    if raster_dir and len(sources) > 1:
        raise ValueError("raster_dir batch mode requires exactly one source entry")
    names = []
    for s in sources:
        s_names = [str(n) for n in s.names]
        if len(list(s.bands)) != len(s_names):
            raise ValueError(
                f"{s.path}: bands/names length mismatch ({list(s.bands)} vs {s_names})"
            )
        names.extend(s_names)
    if "ndsm" in names:
        raise ValueError("channel name 'ndsm' is reserved for the DSM channel")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate channel names across sources: {names}")
    return names


def stack_sources(specs, h, w, transform, crs, out_path, footprint=None):
    """Resample each (path, bands, names) source to the target grid and stack
    all bands into one float32 [0,1] GeoTIFF with named band descriptions.

    Out-of-footprint pixels are written as NaN (nodata=nan) so that "the drone
    never recorded here" survives into the stack instead of collapsing into an
    ordinary 0.0. Everything downstream — tiling, normalisation stats,
    prediction — then reads the footprint off the imagery rather than guessing
    it back from an all-bands-zero heuristic.

    ``footprint`` defaults to the intersection of the sources' own footprints.
    """
    arrays, names = [], []
    for path, bands, band_names in specs:
        arrays.append(read_scaled_bands(path, bands, h, w))
        names.extend(band_names)
    data = np.concatenate(arrays, axis=0)
    if footprint is None:
        footprint = scene_footprint(specs, h, w)
    data[:, ~footprint] = np.nan
    write_tif(out_path, data, transform, crs, nodata=np.nan, descriptions=names)
    inside = int(footprint.sum())
    print(
        f"  -> {os.path.basename(out_path)}  ({len(names)} ch: {', '.join(names)}) "
        f"— {inside:,} / {footprint.size:,} px inside footprint"
    )
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    logger.info("Config:\n%s", OmegaConf.to_yaml(args))
    names = validate_sources(args.sources, args.raster_dir) if args.out_image_dir else None

    specs = [
        (str(s.path), [int(b) for b in s.bands], [str(n) for n in s.names]) for s in args.sources
    ]

    with rasterio.open(args.reference) as ref:
        crs = ref.crs
        h, w, transform = target_grid(ref, args.target_gsd)
        print(
            f"Target grid: {h} x {w} at {args.target_gsd * 100:.1f} cm GSD "
            f"(native {ref.res[0] * 100:.2f} cm -> {args.target_gsd * 100:.1f} cm)"
        )

        # The mask needs the footprint before it can tell "outside the scene"
        # from "inside but unlabelled", so derive it from the sources first.
        print("\nDeriving scene footprint from sources...")
        footprint = scene_footprint(specs, h, w)
        print(f"  {int(footprint.sum()):,} / {footprint.size:,} px inside the footprint")

        print("\nBuilding crown mask...")
        mask = build_mask(
            args.crowns,
            ref,
            h,
            w,
            transform,
            args.sigma,
            args.nodata_threshold,
            footprint=footprint,
            soft_floor=args.get("soft_floor", None),
            erode=float(args.get("erode", 0.0) or 0.0),
            round_radius=float(args.get("round_radius", 0.0) or 0.0),
        )  # args.crowns is a list
        write_tif(args.out_mask, mask, transform, crs, nodata=MASK_RASTER_NODATA)
        print(f"Mask saved: {args.out_mask}")

    if args.out_image_dir:
        if args.raster_dir:
            om_files = sorted(
                os.path.join(args.raster_dir, f)
                for f in os.listdir(args.raster_dir)
                if "_OM_" in f and f.endswith(".tif")
            )
            with rasterio.open(args.reference) as _ref:
                _ref_bounds = _ref.bounds
            for _f in om_files:
                with rasterio.open(_f) as _src:
                    assert_same_bounds(_src, _ref_bounds)
            _, bands, band_names = specs[0]
            jobs = [([(f, bands, band_names)], f) for f in om_files]
        else:
            jobs = [(specs, str(args.reference))]

        print(f"\nStacking {len(jobs)} image(s) at {args.target_gsd * 100:.1f} cm...")
        for job_specs, stem_src in jobs:
            stem = os.path.splitext(os.path.basename(stem_src))[0]
            out_path = os.path.join(args.out_image_dir, f"{stem}_stack.tif")
            # In raster_dir batch mode each OM file brings its own footprint;
            # only the single-job case shares the one the mask was built with.
            job_footprint = footprint if job_specs is specs else None
            stack_sources(job_specs, h, w, transform, crs, out_path, footprint=job_footprint)

        os.makedirs(args.out_image_dir, exist_ok=True)
        manifest = os.path.join(args.out_image_dir, "channels.json")
        with open(manifest, "w") as f:
            json.dump({"names": names}, f, indent=2)
        print(f"\nDone. {len(jobs)} stack(s) + channels.json written to {args.out_image_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Stage 1a: rasterize crown polygons to soft mask.")
    p.add_argument("--config", required=True, help="Path to preprocess.yaml")
    cfg = OmegaConf.load(p.parse_args().config)
    main(cfg.rasterize)
