"""Stage C, part 1: build features, train, validate.

  uv run python scripts/spectral_classify.py --config configs/spectral/classify.yaml

Trains every (feature variant x label set) combination side by side. Feature
variants are `full`, `reduced`, `baseline` (Task 10's original scope). Label
sets are `filtered` (the default certaintyLP/coverage quality filter) and
`all` (every sampled row) — added because the quality filter turned out to
remove 11 of 18 deadwood ground-truth trees on the real data, and that cost
needs to stay visible rather than assumed.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deadwood_spectral.classify import (  # noqa: E402
    apply_label_set,
    save_model,
    train_variant,
    variant_spec,
)
from deadwood_spectral.extract import (  # noqa: E402
    load_ndsm_reference,
    samples_ndsm_reference_path,
)
from deadwood_spectral.features import build_features  # noqa: E402
from deadwood_spectral.report import best_date, separability_table  # noqa: E402

logger = logging.getLogger(__name__)


def _resolve_baseline(cfg, table, dates: list[str]) -> str:
    if cfg.classify.baseline_date:
        return str(cfg.classify.baseline_date)
    chosen = best_date(separability_table(table, dates, measures=["ndvi"]))
    logger.info("baseline_date not set — using highest-AUC date %s", chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the deadwood classifier.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    table = pd.read_parquet(cfg.paths.samples)
    dates = [str(d) for d in cfg.classify.cycle.dates]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.paths.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    label_sets = [str(s) for s in cfg.classify.label_sets]
    summaries = []
    results = {}
    baselines = {}
    for label_set in label_sets:
        label_table = apply_label_set(table, label_set)
        baseline = _resolve_baseline(cfg, label_table, dates)
        baselines[label_set] = baseline
        for variant in cfg.classify.variants:
            variant = str(variant)
            result = train_variant(
                label_table, dates, variant, baseline,
                seed=int(cfg.classify.seed), n_splits=int(cfg.classify.n_splits),
                n_estimators=int(cfg.classify.n_estimators),
            )
            key = (variant, label_set)
            results[key] = result
            result["metrics"].to_csv(out_dir / f"metrics_{variant}_{label_set}.csv", index=False)
            result["loto"].to_csv(out_dir / f"loto_{variant}_{label_set}.csv", index=False)
            deadwood = result["metrics"].set_index("class_name").loc["deadwood"]
            summaries.append(
                {
                    "variant": variant,
                    "label_set": label_set,
                    "n_features": result["n_features"],
                    # n_samples / n_deadwood_groups make the population each
                    # row was fitted on visible. They are NOT the same across
                    # variants: the NaN drop in train_variant removes any row
                    # missing on ANY of the variant's dates, so the 1-date
                    # `baseline` keeps rows the 12-date `full` drops, and can
                    # keep whole trees `full` loses. Compare rows with the
                    # same two numbers; where they differ, the score
                    # difference is partly a population difference.
                    "n_samples": result["n_samples"],
                    "n_deadwood_groups": result["n_deadwood_groups"],
                    "n_groups": result["n_groups"],
                    "deadwood_precision": deadwood["precision"],
                    "deadwood_recall": deadwood["recall"],
                    "deadwood_f1": deadwood["f1"],
                    "loto_recall_median": result["loto"]["recall"].median(),
                    "loto_recall_min": result["loto"]["recall"].min(),
                    "loto_recall_max": result["loto"]["recall"].max(),
                }
            )

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(out_dir / "variant_comparison.csv", index=False)
    logger.info("\n%s", comparison.to_string(index=False))

    # The comparison this branch exists to make — 12 dates vs. 1 — is only
    # honest if both models saw the same pixels. They do not have to: the
    # NaN drop is per variant. Say so explicitly rather than leaving it to
    # whoever reads the CSV.
    for label_set, block in comparison.groupby("label_set"):
        if block["n_samples"].nunique() > 1 or block["n_deadwood_groups"].nunique() > 1:
            logger.warning(
                "label_set=%s: variants were fitted on DIFFERENT populations "
                "(n_samples %s, n_deadwood_groups %s). A variant using fewer "
                "dates loses fewer rows to the NaN drop, so part of any score "
                "difference is a population difference, not a feature-set "
                "difference. See n_samples/n_deadwood_groups in "
                "variant_comparison.csv.",
                label_set,
                dict(zip(block["variant"], block["n_samples"])),
                dict(zip(block["variant"], block["n_deadwood_groups"])),
            )

    primary_key = (str(cfg.classify.primary_variant), str(cfg.classify.primary_label_set))
    primary = results[primary_key]
    # Carry the training nDSM's identity into the model directory so apply.py
    # can refuse a different nDSM at inference (see assert_same_ndsm).
    ndsm_reference = load_ndsm_reference(samples_ndsm_reference_path(cfg.paths.samples))
    if ndsm_reference is None:
        logger.warning(
            "no nDSM identity sidecar next to %s — apply.py will not be able to "
            "verify that inference uses the nDSM this model was trained on. "
            "Re-run scripts/spectral_report.py to write it.",
            cfg.paths.samples,
        )
    save_model(
        primary["model"], primary["features"], cfg.paths.model_dir,
        ndsm_reference=ndsm_reference,
    )
    logger.info(
        "model (variant=%s, label_set=%s) -> %s",
        cfg.classify.primary_variant, cfg.classify.primary_label_set, cfg.paths.model_dir,
    )

    primary_label_set = str(cfg.classify.primary_label_set)
    primary_table = apply_label_set(table, primary_label_set)
    _, primary_switches = variant_spec(primary["variant"], dates, baselines[primary_label_set])
    matrix = build_features(primary_table, primary["dates"], **primary_switches)
    finite = matrix.notna().all(axis=1)
    importance = permutation_importance(
        primary["model"],
        matrix[finite].to_numpy(dtype=float),
        primary_table.loc[finite, "class_code"].to_numpy(),
        n_repeats=int(cfg.classify.permutation_repeats),
        random_state=int(cfg.classify.seed),
        n_jobs=-1,
    )
    ranked = (
        pd.DataFrame({"feature": primary["features"], "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
    )
    ranked.to_csv(out_dir / "feature_importance.csv", index=False)

    top = ranked.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("permutation importance")
    ax.set_title(f"Top features — {cfg.classify.primary_variant}/{cfg.classify.primary_label_set}")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=140)
    plt.close(fig)

    logger.info("results -> %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
