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

## Spectral Analysis

Standing deadwood detection from the multi-year orthomosaic time series.
### Stage A — Align the time series

Reprojects every scene in `datafiles/OM_domAligned/` onto the `crown_mask.tif`
grid (5 cm, EPSG:32736). 

uv run python scripts/spectral_align.py --config configs/spectral/align.yaml

### Stage B1 — Spectral overview

Descriptive only: no model, no threshold, no detections. It answers whether the
seasonal signal the project rests on is actually in the data, before anything is
trained on it.

```
uv run python scripts/spectral_overview.py --config configs/spectral/overview.yaml
```

Three classes are compared over every aligned acquisition. `deadwood` is the
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

Outputs in `out/spectral/overview/`:

- `overview_class.csv` — per class, date and measure: `median` with `q25`/`q75`
  and `n_valid_px`. The count separates a kink in a curve from a data hole.
- `overview_tree.csv` — the same per `soff` tree, so it is visible whether the
  class median stands for all eighteen or one tree is dragging it.
- `signature_class.csv` — mean reflectance per band, class and season. The
  time-series tables answer *when* it swings; this one answers *what it looks
  like*. April and October are transitional and are excluded here, though they
  still appear in the curves.
- `ts_<measure>.png` — class medians with their interquartile band, single
  `soff` trees thin behind them.
- `signature.png` — reflectance against band, one panel per season.
