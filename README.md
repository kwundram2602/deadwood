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
uv run python scripts/preprocess.py --config ./configs/preprocess/preprocess.yaml
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
