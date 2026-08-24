# Deadwood

## Installation 

1. Install uv 
*(https://docs.astral.sh/uv/getting-started/installation/)*
2. set up .venv
```
cd path/to/deadwood && uv sync
```
3. Get sample weights
```
git lfs install          
git lfs pull             
```
## Crown Segmentation

### Preprocessing


- 1a: Rasterizing Crown field data
- 1b: Creating a nDSM based crown ground truth mask together with results of 1a
- Stage 2 — Tile full-res outputs into 512×512 patches

*Run full pipeline (1 - 2 +  data split)*
```
uv run python scripts/preprocess.py --config ./configs/preprocess.yaml
```
### Training

#### Local training (from deadwood/)
The train config sets input channels (e.g rgb + multispectral), training parameters and also some evaluation parameters.

```
uv run python scripts/train.py --config configs/train_config/crown_rgb_ms.yaml --working_dir .
```

### Evaluation
Reuse your train config.
```
uv run python scripts/evaluate.py --config configs/train_config/user/crown_ms.yaml --working_dir .
```

### Prediction

Using a model trained on RGB+MS4:
(sample_exp/crown_rgb_ms__OAM_RGB_RESNET50_TCD__bce0.3_dice0.6/ft_best.pt)
If you trained your model on rgb and the multispectral scene, you also must provide both scenes, or a pre-stacked 7 band scene that you would like to predict on.

```
uv run python scripts/predict.py --config configs/predict/predict_sample_rgb_ms.yaml
```

## Spectral Analysis

Standing deadwood detection from the multi-year orthomosaic time series. The
crown segmentation above supplies the living-vegetation reference; the `soff`
field polygons supply the deadwood ground truth. Because a deciduous savanna
tree is leafless in the dry season too, a single date cannot separate dead from
dormant — a window of dates can.

The pipeline is two scripts, one config each.

### Stage A — Align the time series

Reprojects every scene in `datafiles/OM_domAligned/` onto the `crown_mask.tif`
grid (5 cm, EPSG:32736). Idempotent: re-run it as new dates arrive.

```
uv run python scripts/spectral_align.py --config configs/spectral/align.yaml
```

Stage A does only reprojection now — there is no co-registration check anymore.
Registration rests on the upstream alignment plus a manual visual check: overlay
one aligned date and `crown_mask.tif` in QGIS and confirm the crowns coincide.
The **residual registration error is unquantified** — nothing in the pipeline
measures it any more; the QGIS check is the only safeguard.

### Stage B — Sample, train, predict

```
uv run python scripts/spectral_deadwood.py train   --config configs/spectral/deadwood.yaml
uv run python scripts/spectral_deadwood.py predict --config configs/spectral/deadwood.yaml
uv run python scripts/spectral_deadwood.py all     --config configs/spectral/deadwood.yaml
```

Draws grouped, balanced samples from three pools (deadwood from the eroded
`soff` polygons, living from the crown prediction, background from the rest),
then computes, per sampled pixel, 31 date-invariant phenology statistics over
the half-open window `(label_date - window_months, label_date]` set in
`deadwood.yaml`'s `window` block — there is no fixed list of dates any more,
only the window's width and its end. Pixels with fewer than `min_valid_dates`
observations in the window get NaN features and are dropped from training /
marked `unevaluated` at prediction time. The sample table is cached next to the
model as `samples.parquet` and is redrawn automatically whenever the window or
sampling parameters change.

`train` fits one RandomForest, validated grouped by tree (`StratifiedGroupKFold`
plus leave-one-tree-out over the `soff` trees), and writes the model plus
`importances.png`, `precision_recall.png` and `phenology.png` (the raw,
un-aggregated seasonal course for a subsample — the aggregated features have
already thrown that shape away) under `out/spectral/`. `predict` applies the
saved model across every crown and writes:

- `out/spectral/p_deadwood.tif` — per-pixel deadwood probability, NaN outside
  the valid footprint.
- `out/spectral/crowns.gpkg` — one row per crown, with the aggregated dead
  fraction and a `label` column with four values, decided in this order:
  - `rejected` — the crown's `background_frac >= 0.5`, i.e. the spectral
    classifier read most of the crown as background. This almost always means
    the crown itself is a false positive from the upstream torch segmentation
    model, not that the tree is alive or dead — which is why the label set
    keeps three outcomes instead of a simple living/dead binary.
  - `deadwood` — checked only once a crown is not `rejected`: its
    `dead_frac >= dead_frac_threshold`.
  - `living` — neither of the above.
  - `unevaluated` — no pixel of the crown had enough valid dates to be
    classified at all (see `min_valid_dates` above).

*"When did a tree die?"* is no longer answered by dedicated code. Since the
features are date-invariant (bound only to a window's end, not to a fixed
calendar list), point `window.label_date` at successive dates and run `predict`
again for each — the same model, over several windows, traces when a crown's
predicted state changed.

### Caveats

The field data holds 18 `soff` (deadwood) crown polygons, one per tree, but
the one real training run so far reported 17 distinct deadwood tree groups —
one tree silently dropped out somewhere between the polygon file and the
sample table. Two mechanisms in this pipeline can do that: `erode_m` can erode
a small crown down to an empty geometry, and pixels with fewer than
`min_valid_dates` observations are dropped before labels are assigned. Which
of the two (or something else) accounts for the missing tree has not been
confirmed. Either way, every metric is validated grouped by tree
(`StratifiedGroupKFold` plus leave-one-tree-out over the `soff` trees) because
the deadwood group count is this small, so the per-tree spread matters more
than the mean. The living labels come from the model's crown prediction, not
from field polygons, and can contain undetected deadwood.
