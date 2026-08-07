"""Stage C, part 1: build features, train, validate.

  uv run python scripts/spectral_classify.py --config configs/spectral/classify.yaml

Trains one RandomForest on the full feature set (every configured cycle date,
per-date values plus temporal aggregates plus nDSM) over every sampled row.
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
from deadwood_spectral.classify import save_model, train_model  # noqa: E402
from deadwood_spectral.extract import (  # noqa: E402
    load_ndsm_reference,
    samples_ndsm_reference_path,
)
from deadwood_spectral.features import build_features  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the deadwood classifier.")
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)

    table = pd.read_parquet(cfg.paths.samples)
    dates = [str(d) for d in cfg.classify.cycle.dates]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.paths.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    result = train_model(
        table, dates,
        seed=int(cfg.classify.seed), n_splits=int(cfg.classify.n_splits),
        n_estimators=int(cfg.classify.n_estimators),
    )
    result["metrics"].to_csv(out_dir / "metrics.csv", index=False)
    result["loto"].to_csv(out_dir / "loto.csv", index=False)

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
        result["model"], result["features"], cfg.paths.model_dir,
        ndsm_reference=ndsm_reference,
    )
    logger.info("model -> %s", cfg.paths.model_dir)

    matrix = build_features(table, dates, per_date=True, temporal=True, static=True)
    finite = matrix.notna().all(axis=1)
    importance = permutation_importance(
        result["model"],
        matrix[finite].to_numpy(dtype=float),
        table.loc[finite, "class_code"].to_numpy(),
        n_repeats=int(cfg.classify.permutation_repeats),
        random_state=int(cfg.classify.seed),
        n_jobs=-1,
    )
    ranked = (
        pd.DataFrame({"feature": result["features"], "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
    )
    ranked.to_csv(out_dir / "feature_importance.csv", index=False)

    top = ranked.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("permutation importance")
    ax.set_title("Top features")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=140)
    plt.close(fig)

    logger.info("results -> %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    main()
