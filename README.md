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