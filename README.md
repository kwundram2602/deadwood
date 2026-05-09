


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
uv run python explore_and_process/tile_patches.py --image datafiles\process_out\images\20260313_Airport_Main_MAVICM3MFIXEDM3M_tile001_OM_shift_ms4.tif --mask datafiles\process_out\masks\crown_mask_final_lm_w400-500-700-800_ht1.6_gr_s3.0_and.tif --dsm datafiles\process_out\ndsm\dsm_ndsm_w400-500-700-800.tif --out   out/crown_ms_patches --size  512
```

### Stage 3 — Split patches into train / val / test

```bash
uv run python data_utils/data_split.py
```

### Training

##### Local training (from deadwood/)
uv run python scripts/train.py --config configs/crown_ms.yaml --working_dir .