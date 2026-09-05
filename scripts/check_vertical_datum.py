"""Decide whether the DSM/DTM height offset is a vertical-datum mismatch.

The question this answers: is the ~6 m gap between the photogrammetric DSM and
the LiDAR DTM an ellipsoidal-vs-orthometric datum mix (fix it with a geoid
model, never with a fit), or is it a survey/reconstruction error (fix it by
co-registration)?

Three pieces of evidence, in order:

  1. What the files declare. A vertical datum lives in a CompoundCRS or in the
     GeoTIFF VerticalCSTypeGeoKey. If neither raster declares one, the metadata
     cannot answer the question and the remaining checks have to.

  2. What a datum mix would look like here. The geoid undulation N is computed
     from EGM96 and EGM2008 at the scene centre, and its gradient across the
     scene extent. Both numbers are predictions: a datum mix shows up as an
     offset of exactly +-N, and — because N varies by centimetres per kilometre
     — as essentially zero tilt. Over a 350 m scene the geoid is a plane to
     within a millimetre or two.

  3. What the rasters actually show. Both are put on a common grid and a plane
     is fitted robustly through the ground candidates, giving the observed
     offset and tilt.

The verdict compares 3 against 2. An observed tilt far above the geoid's own
tilt rules out the datum explanation on its own, whatever the offset does.

Usage:
  uv run python scripts/check_vertical_datum.py --dsm <DSM.tif> --dtm <DTM.tif>
"""

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup
from rasterio.enums import Resampling
from rasterio.warp import reproject

logger = logging.getLogger(__name__)

# Geoid models to test the offset against. EGM96 is what most consumer GNSS
# receivers and drone controllers report; EGM2008 is the modern default.
GEOID_MODELS: dict[str, str] = {"EGM96": "EPSG:4326+5773", "EGM2008": "EPSG:4326+3855"}

# A datum offset has to match +-N this closely to count as an explanation. Loose
# on purpose: it is meant to catch "13.8 vs 6.9", not to grade a fit.
OFFSET_TOL_M = 1.5

# How much larger than the geoid's own tilt the observed tilt has to be before
# the datum explanation is called impossible rather than merely unsupported.
TILT_FACTOR = 10.0


@dataclass(frozen=True)
class PlaneFit:
    """A plane through the ground candidates of DSM - DTM."""

    offset: float  # m, at the centre of the overlap
    dz_dx: float  # m/m, eastward
    dz_dy: float  # m/m, northward
    span_x: float  # m
    span_y: float  # m
    residual_std: float  # m, ground candidates after removing the plane
    n_ground: int
    n_valid: int
    diff_min: float
    diff_median: float

    @property
    def tilt(self) -> float:
        """Total gradient magnitude in m/m."""
        return float(np.hypot(self.dz_dx, self.dz_dy))

    @property
    def tilt_over_scene(self) -> float:
        """Height change the tilt produces across the diagonal of the overlap, in m."""
        return float(np.hypot(self.dz_dx * self.span_x, self.dz_dy * self.span_y))


def declared_vertical_crs(path: Path) -> dict:
    """Report the vertical component the file declares, if any.

    Checks both places it can hide: the CRS pyproj parses out of the file, and
    the raw VerticalCSTypeGeoKey, which GDAL drops from the CRS when it cannot
    resolve it.
    """
    with rasterio.open(path) as src:
        crs = CRS.from_wkt(src.crs.to_wkt())
    vertical = [sub for sub in (crs.sub_crs_list or []) if sub.is_vertical]
    geokeys = _geokeys(path)
    return {
        "name": crs.name,
        "is_compound": crs.is_compound,
        "vertical_crs": vertical[0].name if vertical else None,
        "vertical_geokey": geokeys.get("VerticalCSTypeGeoKey"),
        "citation": geokeys.get("GTCitationGeoKey"),
    }


def _geokeys(path: Path) -> dict[str, str]:
    """Parse the raw GeoTIFF keyed information out of `listgeo`.

    listgeo is used rather than rasterio because the point is to see keys that
    GDAL's CRS parsing may have discarded.
    """
    try:
        out = subprocess.run(
            ["listgeo", str(path)], capture_output=True, text=True, timeout=60, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        logger.warning("listgeo unavailable; raw GeoTIFF keys not inspected")
        return {}
    keys = {}
    for line in out.splitlines():
        if "GeoKey" in line and ":" in line:
            name, _, value = line.strip().partition(":")
            keys[name.split(" ")[0]] = value.strip().strip('"')
    return keys


def geoid_undulation(lon: float, lat: float, model: str) -> float:
    """Geoid undulation N in metres: h_ellipsoidal - H_orthometric.

    Raises when PROJ has no grid for the model. That matters: without
    allow_ballpark=False, PROJ silently returns the identity and N comes out as
    a very convincing 0.000 m.
    """
    group = TransformerGroup("EPSG:4979", GEOID_MODELS[model], always_xy=True)
    if not group.transformers:
        raise RuntimeError(
            f"{model}: no usable transformation "
            f"({len(group.unavailable_operations)} operation(s) need grids PROJ does not have). "
            "Enable the PROJ network (--network) or install proj-data."
        )
    transformer = Transformer.from_crs(
        "EPSG:4979", GEOID_MODELS[model], always_xy=True, allow_ballpark=False
    )
    return -float(transformer.transform(lon, lat, 0.0)[2])


def overlap_grid(dsm_path: Path, dtm_path: Path, resolution: float) -> tuple:
    """Common grid over the intersection of both rasters, at `resolution`.

    Comparing on a third, coarser grid rather than on either native grid keeps
    the 3.7 cm DSM from being carried around at full size and keeps the 0.5 m
    DTM from being interpolated tenfold — the interpolation would smooth exactly
    the ground detail the plane is fitted through.
    """
    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        if dsm.crs != dtm.crs:
            raise ValueError(f"horizontal CRS differ: {dsm.crs} vs {dtm.crs}")
        left = max(dsm.bounds.left, dtm.bounds.left)
        bottom = max(dsm.bounds.bottom, dtm.bounds.bottom)
        right = min(dsm.bounds.right, dtm.bounds.right)
        top = min(dsm.bounds.top, dtm.bounds.top)
        crs = dsm.crs
    if right <= left or top <= bottom:
        raise ValueError("DSM and DTM do not overlap")
    width = int((right - left) / resolution)
    height = int((top - bottom) / resolution)
    transform = rasterio.transform.from_origin(left, top, resolution, resolution)
    return transform, width, height, crs


def _to_grid(path: Path, transform, width: int, height: int, crs, resampling) -> np.ndarray:
    out = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling,
            dst_nodata=np.nan,
        )
    return out


def fit_ground_plane(diff: np.ndarray, transform, ground_quantile: float = 0.2) -> PlaneFit:
    """Fit a plane through the ground candidates of a DSM - DTM difference.

    Ground is not known in advance, so it is found by iteration: fit a plane to
    everything, keep the lowest `ground_quantile` of residuals, refit. Vegetation
    only ever pushes the difference up, so the low tail is ground, and after a
    few passes the plane sits on it. Taking a plain minimum instead would fit
    the single noisiest pixel of the scene.
    """
    valid = np.isfinite(diff)
    values = diff[valid].astype(np.float64)
    if values.size < 100:
        raise ValueError(f"only {values.size} overlapping valid pixels; nothing to fit")
    rows, cols = np.nonzero(valid)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.asarray(xs) - np.mean(xs)
    ys = np.asarray(ys) - np.mean(ys)
    design = np.column_stack([np.ones_like(values), xs, ys])

    keep = np.ones(values.size, dtype=bool)
    coef = np.zeros(3)
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        residual = values - design @ coef
        keep = residual <= np.percentile(residual, 100 * ground_quantile)

    residual = values[keep] - design[keep] @ coef
    return PlaneFit(
        offset=float(coef[0]),
        dz_dx=float(coef[1]),
        dz_dy=float(coef[2]),
        span_x=float(xs.max() - xs.min()),
        span_y=float(ys.max() - ys.min()),
        residual_std=float(residual.std()),
        n_ground=int(keep.sum()),
        n_valid=int(values.size),
        diff_min=float(values.min()),
        diff_median=float(np.median(values)),
    )


# Copernicus GLO-30, the external yardstick for check 4. Public, unauthenticated,
# and a COG, so only the few kilobytes covering the scene are ever read. Its
# heights are orthometric on EGM2008 -- that is what makes it able to tell an
# ellipsoidal raster from an orthometric one.
COP30_URL = (
    "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)

# Copernicus is itself a surface model: it sees canopy, and in savanna it sits a
# metre or two above true ground. Its absolute accuracy is a few metres. Both are
# far smaller than the 13.8 m that separates the two datum hypotheses, which is
# why a coarse reference can still settle a datum question it could never be used
# to co-register against.
REFERENCE_TOL_M = 5.0


def copernicus_tiles(bounds: tuple[float, float, float, float], crs) -> list[str]:
    """URLs of the GLO-30 tiles covering `bounds`, given in `crs`."""
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    left, bottom, right, top = bounds
    corners = [to_wgs84.transform(x, y) for x in (left, right) for y in (bottom, top)]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    urls = []
    for lat in range(int(np.floor(min(lats))), int(np.floor(max(lats))) + 1):
        for lon in range(int(np.floor(min(lons))), int(np.floor(max(lons))) + 1):
            urls.append(
                COP30_URL.format(
                    ns="S" if lat < 0 else "N",
                    lat=abs(lat),
                    ew="W" if lon < 0 else "E",
                    lon=abs(lon),
                )
            )
    return urls


def reference_on_grid(urls: list[str], transform, width: int, height: int, crs) -> np.ndarray:
    """Mosaic the reference tiles onto one grid, filling gaps tile by tile."""
    out = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for url in urls:
            tile = _to_grid(Path(url), transform, width, height, crs, Resampling.bilinear)
            out = np.where(np.isnan(out), tile, out)
    return out


def ground_surface(
    path: Path, resolution: float, block: float, quantile: float = 0.05
) -> tuple[np.ndarray, object, object]:
    """Coarse ground surface of one raster over its own full extent.

    A low per-block quantile rather than the block minimum: the minimum of a
    30 m cell is whichever pixel the reconstruction dropped furthest, and in a
    photogrammetric DSM that is a hole, not the ground. The 5th percentile keeps
    the same "lowest thing in the cell" idea without handing the answer to a
    single bad pixel.
    """
    with rasterio.open(path) as src:
        bounds, crs = src.bounds, src.crs
    width = int((bounds.right - bounds.left) / resolution)
    height = int((bounds.top - bounds.bottom) / resolution)
    transform = rasterio.transform.from_origin(bounds.left, bounds.top, resolution, resolution)
    fine = _to_grid(path, transform, width, height, crs, Resampling.average)

    factor = max(1, int(round(block / resolution)))
    rows, cols = fine.shape[0] // factor, fine.shape[1] // factor
    blocks = fine[: rows * factor, : cols * factor].reshape(rows, factor, cols, factor)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(rows, cols, factor * factor)
    with np.errstate(all="ignore"):
        coarse = np.nanquantile(blocks, quantile, axis=2).astype(np.float32)
    coarse_transform = rasterio.transform.from_origin(
        bounds.left, bounds.top, resolution * factor, resolution * factor
    )
    return coarse, coarse_transform, crs


def verify_absolute(path: Path, resolution: float, block: float, undulation: float) -> dict | None:
    """Test one raster's ground against the external reference under both hypotheses.

    Returns the median residual for "these heights are already orthometric" and
    for "these heights are ellipsoidal", plus the tilt against the reference.
    """
    ground, transform, crs = ground_surface(path, resolution, block)
    with rasterio.open(path) as src:
        urls = copernicus_tiles(tuple(src.bounds), crs)
    reference = reference_on_grid(urls, transform, ground.shape[1], ground.shape[0], crs)

    valid = np.isfinite(ground) & np.isfinite(reference)
    if valid.sum() < 20:
        logger.warning("%s: only %d cells overlap the reference", path.name, valid.sum())
        return None
    diff = (ground[valid] - reference[valid]).astype(np.float64)

    rows, cols = np.nonzero(valid)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.asarray(xs) - np.mean(xs)
    ys = np.asarray(ys) - np.mean(ys)
    design = np.column_stack([np.ones_like(diff), xs, ys])
    coef, *_ = np.linalg.lstsq(design, diff, rcond=None)
    return {
        "n": int(valid.sum()),
        "as_orthometric": float(np.median(diff)),
        "as_ellipsoidal": float(np.median(diff) - undulation),
        "tilt": float(np.hypot(coef[1] * (xs.max() - xs.min()), coef[2] * (ys.max() - ys.min()))),
        "scatter": float((diff - design @ coef).std()),
        "span": (float(xs.max() - xs.min()), float(ys.max() - ys.min())),
    }


def _print_attribution(absolute: dict[str, dict]) -> None:
    """Name the raster carrying the error, when check 4 has the evidence for it."""
    if len(absolute) < 2:
        return
    errors = {
        label: min(result["as_orthometric"], result["as_ellipsoidal"], key=abs)
        for label, result in absolute.items()
    }
    worst = max(errors, key=lambda label: abs(errors[label]))
    best = min(errors, key=lambda label: abs(errors[label]))
    print()
    print(
        f"  Check 4 puts the error on the {worst}: it sits {errors[worst]:+.2f} m off the external"
    )
    print(f"  reference, while the {best} agrees to {errors[best]:+.2f} m. Correct the {worst}.")


def run(
    dsm_path: Path,
    dtm_path: Path,
    resolution: float,
    use_network: bool,
    use_reference: bool,
    reference_block: float = 30.0,
) -> int:
    """Print the three checks and the verdict. Returns a shell exit code."""
    if use_network:
        pyproj.network.set_network_enabled(True)

    print("=" * 72)
    print("1. DECLARED VERTICAL DATUM")
    print("=" * 72)
    declared = {}
    for label, path in (("DSM", dsm_path), ("DTM", dtm_path)):
        info = declared_vertical_crs(path)
        declared[label] = info
        print(f"{label}  {path.name}")
        print(f"     CRS               : {info['name']}")
        print(f"     compound (3D)     : {info['is_compound']}")
        print(f"     vertical CRS      : {info['vertical_crs'] or '-- none --'}")
        print(f"     VerticalCSTypeGeoKey: {info['vertical_geokey'] or '-- absent --'}")
    both_undeclared = not any(d["vertical_crs"] or d["vertical_geokey"] for d in declared.values())
    if both_undeclared:
        print("\n  -> Neither file declares a vertical datum. The metadata cannot")
        print("     settle this; heights are whatever the survey delivered.")
    else:
        print("\n  -> A vertical datum is declared. Compare the two above before fitting.")

    transform, width, height, crs = overlap_grid(dsm_path, dtm_path, resolution)
    centre_x = transform.c + width * resolution / 2
    centre_y = transform.f - height * resolution / 2
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(centre_x, centre_y)

    print()
    print("=" * 72)
    print("2. WHAT A DATUM MISMATCH WOULD LOOK LIKE HERE")
    print("=" * 72)
    print(f"scene centre       : {lon:.5f} E, {lat:.5f} N")
    print(f"overlap            : {width * resolution:.0f} x {height * resolution:.0f} m")
    undulations: dict[str, float] = {}
    geoid_tilt = 0.0
    for model in GEOID_MODELS:
        try:
            n_centre = geoid_undulation(lon, lat, model)
        except (RuntimeError, pyproj.exceptions.ProjError) as exc:
            print(f"{model:<18} : UNAVAILABLE -- {exc}")
            continue
        undulations[model] = n_centre
        # Sample N at the overlap corners to get the geoid's own gradient.
        dx = width * resolution / 2
        dy = height * resolution / 2
        corners = [
            geoid_undulation(*to_wgs84.transform(centre_x + sx * dx, centre_y + sy * dy), model)
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1))
        ]
        spread = max(corners) - min(corners)
        geoid_tilt = max(geoid_tilt, spread)
        print(
            f"{model:<18} : N = {n_centre:+.3f} m at centre, varies {spread * 100:.2f} cm across the scene"
        )
    if not undulations:
        print("\n  -> No geoid grid available; check 2 is inconclusive.")
        print("     Re-run with --network, or install proj-data.")
    else:
        print("\n  -> A datum mix would show up as an offset of exactly +-N")
        print(
            f"     ({', '.join(f'{abs(v):.2f} m' for v in undulations.values())}) and, since N is"
        )
        print(f"     flat to {geoid_tilt * 100:.2f} cm over this extent, as no meaningful tilt.")

    print()
    print("=" * 72)
    print("3. WHAT THE RASTERS SHOW")
    print("=" * 72)
    print(f"comparison grid    : {width} x {height} px at {resolution} m")
    dsm = _to_grid(dsm_path, transform, width, height, crs, Resampling.average)
    dtm = _to_grid(dtm_path, transform, width, height, crs, Resampling.bilinear)
    fit = fit_ground_plane(dsm - dtm, transform)
    print(f"overlapping pixels : {fit.n_valid} valid, {fit.n_ground} kept as ground")
    print(f"DSM - DTM          : min {fit.diff_min:+.2f} m, median {fit.diff_median:+.2f} m")
    print(f"ground offset      : {fit.offset:+.3f} m at the scene centre")
    print(
        f"ground tilt        : {fit.dz_dx * 1000:+.2f} mm/m east, {fit.dz_dy * 1000:+.2f} mm/m north"
    )
    print(f"                     = {fit.tilt_over_scene:.2f} m across the scene")
    print(f"ground flatness    : {fit.residual_std * 100:.1f} cm std after removing the plane")

    absolute: dict[str, dict] = {}
    if use_reference and undulations:
        print()
        print("=" * 72)
        print("4. WHICH RASTER IS ACTUALLY WRONG (external reference)")
        print("=" * 72)
        print("reference          : Copernicus GLO-30, orthometric on EGM2008")
        n_ref = undulations.get("EGM2008", next(iter(undulations.values())))
        for label, path in (("DSM", dsm_path), ("DTM", dtm_path)):
            result = verify_absolute(path, resolution, reference_block, n_ref)
            if result is None:
                print(f"{label}: too little overlap with the reference")
                continue
            absolute[label] = result
            ortho, ellip = result["as_orthometric"], result["as_ellipsoidal"]
            verdict = (
                "ELLIPSOIDAL (WGS 84 h)" if abs(ellip) < abs(ortho) else "ORTHOMETRIC (EGM2008 H)"
            )
            print(
                f"{label}  ground vs reference over {result['span'][0]:.0f} x {result['span'][1]:.0f} m, {result['n']} cells"
            )
            print(f"     read as orthometric : {ortho:+7.2f} m residual")
            print(f"     read as ellipsoidal : {ellip:+7.2f} m residual")
            print(f"     -> heights are {verdict}, off by {min(abs(ortho), abs(ellip)):.2f} m")
            print(
                f"     tilt vs reference   : {result['tilt']:.2f} m over the extent (scatter {result['scatter']:.2f} m)"
            )
        print()
        print(f"  -> The two hypotheses are {abs(n_ref):.1f} m apart and the reference is good to")
        print(
            f"     a few metres, so the datum call is safe. A residual under ~{REFERENCE_TOL_M:.0f} m is"
        )
        print("     agreement -- the reference is a canopy surface, not bare ground.")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if not undulations:
        print("INCONCLUSIVE -- no geoid model available to compare against.")
        return 2

    matches = {m: n for m, n in undulations.items() if abs(abs(fit.offset) - abs(n)) < OFFSET_TOL_M}
    tilt_excess = fit.tilt_over_scene / max(geoid_tilt, 1e-6)

    if tilt_excess > TILT_FACTOR:
        print("NOT a vertical datum problem.")
        print(f"  The ground surfaces are tilted against each other by {fit.tilt_over_scene:.2f} m")
        print(f"  across the scene. The geoid changes by {geoid_tilt * 100:.2f} cm over the same")
        print(f"  extent -- {tilt_excess:.0f}x less. No geoid model can produce this tilt,")
        print("  so the mismatch is a survey/reconstruction error, not a datum.")
        if matches:
            print(f"  (The offset does happen to sit near N for {', '.join(matches)};")
            print("   the tilt still rules the datum explanation out.)")
        _print_attribution(absolute)
        print("\n  Look at: GNSS base coordinates and antenna height of each campaign,")
        print("  and whether the photogrammetric block was controlled by GCPs.")
        return 1

    if matches:
        model = next(iter(matches))
        print(f"LIKELY a vertical datum mismatch against {model}.")
        print(f"  Observed offset {fit.offset:+.3f} m matches N = {undulations[model]:+.3f} m,")
        print(f"  and the tilt ({fit.tilt_over_scene:.2f} m) is consistent with a flat geoid.")
        print("\n  Fix by converting the ellipsoidal raster with the geoid model --")
        print("  do not absorb this into a fitted plane.")
        return 1

    print("NOT explained by a vertical datum.")
    print(f"  Observed offset {fit.offset:+.3f} m matches neither model:")
    for model, n_centre in undulations.items():
        print(
            f"    {model}: N = {n_centre:+.3f} m  (off by {abs(abs(fit.offset) - abs(n_centre)):.2f} m)"
        )
    print("\n  A datum mix is all-or-nothing: the offset is either +-N or it is not")
    print("  a datum. Look for a survey error -- base station height, or a")
    print("  photogrammetric block that drifted without ground control.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dsm", type=Path, required=True, help="DSM raster")
    parser.add_argument("--dtm", type=Path, required=True, help="DTM raster")
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="comparison grid size in metres (default: 0.5, the DTM's native size)",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="let PROJ download geoid grids from cdn.proj.org if they are missing",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="also check each raster against Copernicus GLO-30 to find which one is wrong "
        "(downloads a few kB from a public AWS bucket)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    code = run(args.dsm, args.dtm, args.resolution, args.network, args.reference)
    return code


if __name__ == "__main__":
    sys.exit(main())
