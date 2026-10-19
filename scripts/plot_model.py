"""Render a paper-style model figure from a torchview model_graph.gv.

Usage:
    uv run python scripts/plot_model.py sample_exp/<run>/model_graph.gv [-o fig.pdf] [--keep-tex]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.model_figure import plot_model_from_gv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a model figure from model_graph.gv")
    parser.add_argument("gv", type=Path, help="model_graph.gv written by save_model_graph")
    parser.add_argument("-o", "--out", type=Path, default=None, help="output PDF path")
    parser.add_argument(
        "--keep-tex", action="store_true", help="keep the LaTeX source as <out>.tex"
    )
    args = parser.parse_args()
    pdf = plot_model_from_gv(args.gv, args.out, keep_tex=args.keep_tex)
    print(f"Saved model figure -> {pdf}")


if __name__ == "__main__":
    main()
