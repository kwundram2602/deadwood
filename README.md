


## Pipeline


### 1a)
```bash
uv run python explore_and_process/rasterize_crowns.py --config D:\EAGLE\InnoLab_DL\deadwood\configs\preprocess.yaml
```
### 1b)
```bash
uv run python explore_and_process/apply_dsm_mask.py --config D:\EAGLE\InnoLab_DL\deadwood\configs\preprocess.yaml
```

### Stage 2 — Tile full-res outputs into 512×512 patches

```bash
uv run python explore_and_process/tile_patches.py --config
```
### Run full pipeline (1 - 2 +  data split)
```
uv run python scripts/preprocess.py --config \configs/preprocess.yaml
```
### Training

##### Local training (from deadwood/)
uv run python scripts/train.py --config configs/train_config/crown_ms.yaml --working_dir .


### Evaluation

uv run python scripts/evaluate.py --config configs/train_config/user/crown_ms_bce_dice.yaml --working_dir .
