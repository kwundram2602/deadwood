


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
### Run full pipeline (1 - 2)
```
uv run python scripts/preprocess.py --config \configs/preprocess.yaml
```
### Stage 3 — Split patches into train / val / test

```bash
uv run python data_utils/data_split.py
```

### Training
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
##### Local training (from deadwood/)
uv run python scripts/train.py --config configs/train_config/crown_ms.yaml --working_dir .