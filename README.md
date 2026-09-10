# Deadwood
This project aims to detect deadwood based on sparse labeled field data. The code was tested on field data retrieved in the Kruger Nationalpark in South Afric.
Besides corwn field data
In a first step a TorchGeo UNet with pretrained weights () is used for a binary crown segmentation. 
The crown segmentation output mask and the crowns that were classified as deadwood in the field campaign then feed the spectral analysis step. The different amplitude in the spectral signal over time 
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

## Deadwood Fine-Tuning

Adapts the pretrained deadtrees.earth model (`smp.Unet(mit_b5)`, 84.7 M params) to our
site. Separate from the crown pipeline above — shares code, not configs or execution.

Input contract, fixed by the checkpoint: 3-band **uint8** RGB, `/255`, ImageNet
mean/std, 0.05 m GSD, 1024 px tiles with 256 px discarded padding. Our orthos are
float32 in a ~16-bit range, so `scale.mode: linear` + `source_max: 65535` is mandatory.

### Baseline prediction (no fine-tuning)
```
uv run python scripts/raw_predict_deadwood.py --config configs/predict/deadwood/raw_deadwood.yaml --working_dir .
```
Always the whole scene (~10 min). Writes `*_rgb8.tif`, `*_deadwood_mask.tif`,
`*_deadwood.gpkg`. Keep `num_workers` at 0 (forked workers corrupt the shared GDAL handle)
and `batch_size` at 2 on a 4 GB card.

### Preprocessing — whole-scene mask, grid tiles
```
uv run python scripts/preprocess_deadwood.py --config configs/preprocess/deadwood.yaml --working_dir .
```
One mask over the whole scene, cut into a disjoint 512 px grid into
`out/deadwood_patches/{train,val,test}/{images,masks}/`. Tiles share no ground, so the
split is assigned per tile (stratified random, by fraction) and there is no leakage to
guard against.

Two label layers: `labels.deadwood_path` (positives) and `labels.background_path`
(digitised confirmed non-deadwood). Mask values: `1.0` crown core · `0.05–1.0` Gaussian
falloff (`sigma_pos`) · `0.0` background core (`sigma_neg` + `neg_threshold`) · `255.0`
unlabelled · `-1.0` outside footprint. Only `[0,1]` enters the loss
(`utils.nodata.valid_target`).

The two sigmas differ on purpose. Blurring conserves burned area, so a wide sigma flattens
small polygons — at 50 cm a 0.32 m² crown peaks at 0.18 — which caps `sigma_pos` near 4 px
(20 cm). Background polygons are ~5× larger and absorb 10 px (50 cm), and that wider blur
is what guarantees an unlabelled band between the classes even where they are drawn edge to
edge. A hard `1.0` next to a hard `0.0` is what the old ring negatives produced.

Tiles are dropped by two independent filters: `min_labelled_px` (a label must be present —
it does **not** require positives, since background-only tiles are the false-positive
signal) and `max_outside_frac` (the imagery must be present).

Written alongside the patches for visual QA: `deadwood_mask_scene.tif` (the whole mask,
noData transparent in QGIS) and `tiles.gpkg` (every grid tile with its `split`, `kind`,
`labelled_px` and `outside_frac`, dropped tiles included).

Drop bad crowns with `labels.exclude_fids` — real GPKG fids, not row indices.

### Training
```
uv run python scripts/finetune_deadwood.py --config configs/finetune/deadwood.yaml --working_dir .
```
`stage: head|decoder` (145 vs 3.28 M trainable params). `fine_tune.enabled: true` adds an
encoder-unfreeze phase; valid `unfreeze_keys` are `patch_embed1..4`, `block1..4`,
`norm1..4`. Checkpoint → `experiments/<id>/tl_best.pt`. Patches are 512² since the mask
rework, so `batch_size: 8` costs about what 1024² at batch 2 did (~1.7 GiB) — worth raising,
because most tiles carry only one label class and small batches make the gradient very noisy.

### Prediction with a fine-tuned checkpoint
```
uv run python scripts/raw_predict_deadwood.py --config configs/predict/deadwood/finetuned_deadwood.yaml --working_dir .
```
Any checkpoint without editing a config (`--weights` takes `.pt` and `.safetensors`):
```
uv run python scripts/raw_predict_deadwood.py --config configs/predict/deadwood/raw_deadwood.yaml --working_dir . \
    --weights experiments/<run>/tl_best.pt --out_dir out/predict_<run> --threshold 0.3
```

### Evaluation — per-crown recall
```
uv run python scripts/evaluate_deadwood.py --config configs/predict/deadwood/finetuned_deadwood.yaml --working_dir . --probs out/eval/ft_probs.tif
```
`--probs` caches the probability raster and reuses it, so re-scoring at another threshold
skips inference. Writes `<ckpt>_crown_coverage.csv` and `<ckpt>_threshold_sweep.csv` to
`out/eval/`. **Recall only** — the labels are sparse-sampled, so predictions outside the
polygons are not false positives and no precision figure is derivable from them.

### Status (2026-09-07)

Raw model: **20/31** crowns hit at >10 % overlap. The threshold sweep is flat from 0.9 down
to 0.1 — 9 of the 11 misses have max probability `0.0000` across the whole polygon, so
recalibration cannot recover them.

First fine-tune (decoder+head, `bce 1.0`, 18 epochs) made it **worse**: test split 3/6 → 0/6,
all crowns 20/31 → 3/31, with loss falling while F1 collapsed 0.48 → 0.07. Cause: the 2 m
ring negatives land exactly on the model's crown-boundary overhang — 13 % of ring pixels are
predicted deadwood by the raw model, at 5.3:1 negative-dominated — which teaches suppression.
Open fix: leave an unlabelled gap between the polygon and the hard negatives, so boundary
disagreement is never penalised. See `docs/HANDOFF-deadtrees-finetune.md`.

## Spectral Analysis

Standing deadwood detection from the multi-year orthomosaic time series.
### Stage A — Align the time series

Reprojects every scene in `datafiles/OM_domAligned/` onto the `crown_mask.tif`
grid (5 cm, EPSG:32736). 

uv run python scripts/spectral_align.py --config configs/spectral/align.yaml

### Stage B1 — Spectral overview

Descriptive only: 

```
uv run python scripts/spectral_overview.py --config configs/spectral/overview.yaml
```
Setting `window.label_date` restricts the run to the half-open window
`(label_date - window_months, label_date]`.

Three classes are compared over the selected acquisitions. `deadwood` is the
`soff` field polygons — the only real ground truth. `living` is the *crown
model's* prediction, not the `son` polygons, because that is the surface a
classifier meets at inference time. `background` is bare ground, and it is not
optional: a leafless dead crown collapses spectrally toward the ground, so
without the ground curve the deadwood curve has nothing to be distinguished
from.

The pixel set is drawn once and reused at every date. Redrawing per acquisition
would make every step in a curve ambiguous between phenology and resampling.
The `soff` pixels are taken whole; the two reference classes are cut to
`sampling.max_pixels_per_class` by a seeded draw. Object-wise curves exist only
for the `soff` trees — for `living` the per-object spread would be the spread of
the crown segmenter, not of the phenology.

Measures, per pixel and per date. `Green`/`Red`/`RedEdge`/`NIR` are the
multispectral bands, `R`/`G`/`B` the RGB composite — two sensors, never mixed
inside one measure:

- `ndvi` = (NIR − Red) / (NIR + Red)
- `ndre` = (NIR − RedEdge) / (NIR + RedEdge) — same, on the red edge, which
  saturates later than red.
- `gndvi` = (NIR − Green) / (NIR + Green) — same, on green; tracks chlorophyll
  rather than leaf area.
- `nir_red_ratio` = NIR / Red — unnormalised, spreads the high end `ndvi`
  compresses. NaN where Red is 0.
- `NIR` — raw near-infrared. A collapsed NIR is the most direct deadwood
  signature there is, and every normalised difference hides it.
- `brightness` = (R + G + B) / 3 — visible reflectance; bare wood and ground are
  bright where leaves are dark.
- `green_red` = (G − R) / (G + R) — greenness from RGB alone, independent of the
  multispectral sensor.

Outputs in `out/spectral/overview/`:

- `sample_pixels.gpkg` — the drawn pixels as points at their centres, with
  `class` and `tree_id`. Not a polygonised mask: the mask says where a class
  *could* have been drawn, this says where it *was*, which for a seeded draw is
  the question worth asking. Load it over the orthomosaic in QGIS and filter by
  `class`.
- `overview_class.csv` — per class, date and measure: `median` with `q25`/`q75`
  and `n_valid_px`. The count separates a kink in a curve from a data hole.
- `overview_tree.csv` — the same per `soff` tree, so it is visible whether the
  class median stands for all eighteen or one tree is dragging it.
- `signature_class.csv` — mean reflectance per band, class and season. The
  time-series tables answer *when* it swings; this one answers *what it looks
  like*. April and October are transitional and are excluded here, though they
  still appear in the curves.
- `ts_<measure>.png` — class medians with their interquartile band. Per-tree
  curves are deliberately not drawn over them; that check lives in
  `overview_tree.csv`.
- `signature.png` — reflectance against band, one panel per season.
