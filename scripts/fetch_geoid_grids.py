"""Download the geoid grids PROJ needs for EGM96 and EGM2008 heights.

Without them PROJ has to fetch the grids from cdn.proj.org on every run; with
no network (sandbox, cluster node) the geoid undulation comes out as inf and
the vertical datum check fails. The files land in datafiles/proj, which
scripts/check_vertical_datum.py registers as a PROJ data directory.

Usage:
  uv run python scripts/fetch_geoid_grids.py [--model EGM2008] [--force]
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_vertical_datum import GEOID_GRIDS, GRID_DIR  # noqa: E402

CDN = "https://cdn.proj.org"


def fetch(name: str, target_dir: Path, force: bool = False) -> Path:
    """Download one grid file into target_dir; skips files already there."""
    target = target_dir / name
    if target.is_file() and not force:
        print(f"{name}: present ({target.stat().st_size / 1e6:.1f} MB)")
        return target
    target_dir.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    print(f"{name}: downloading from {CDN} ...")
    with urllib.request.urlopen(f"{CDN}/{name}") as resp, part.open("wb") as out:
        shutil.copyfileobj(resp, out)
    part.replace(target)  # only complete files get the final name
    print(f"{name}: {target.stat().st_size / 1e6:.1f} MB -> {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=sorted(GEOID_GRIDS),
        action="append",
        help="geoid model to fetch, repeatable (default: all)",
    )
    parser.add_argument("--out", type=Path, default=GRID_DIR, help=f"target directory (default: {GRID_DIR})")
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args(argv)

    for model in args.model or sorted(GEOID_GRIDS):
        fetch(GEOID_GRIDS[model], args.out, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
