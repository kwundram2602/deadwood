"""Score the crown and deadwood masks against field polygons (covered area share).

uv run python scripts/evaluate_field.py --config configs/evaluate/field.yaml --working_dir .
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.field_eval import evaluate_polygons, plot_detection, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score masks against field polygons.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--working_dir", default=".")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config).field_eval
    wd = Path(args.working_dir)
    out_dir = wd / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(t) for t in cfg.thresholds]

    results: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.Series] = {}
    for name, target in cfg.targets.items():
        per_polygon = evaluate_polygons(
            wd / target.mask, wd / target.polygons, layer=target.get("layer")
        )
        per_polygon.to_csv(out_dir / f"{name}_per_polygon.csv")
        results[str(name)] = per_polygon
        summaries[str(name)] = summarize(per_polygon, thresholds)

    summary = pd.DataFrame(summaries).T.rename_axis("target")
    summary.to_csv(out_dir / "field_eval_summary.csv")
    plot = plot_detection(results, thresholds, out_dir / "field_eval_detection.png")

    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary.T.to_string())
    print(f"\nWrote {out_dir}/{{<target>_per_polygon,field_eval_summary}}.csv and {plot.name}")


if __name__ == "__main__":
    main()
