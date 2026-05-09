


## Pipeline


### 1a)
```bash
uv run python explore_and_process/rasterize_crowns.py --crowns datafiles/crown_poly/2_crown_main_20260409_editLP.gpkg datafiles/crown_poly/2_crown_neighbour_20260409_editLP.gpkg --reference datafiles/raster/20260313/20260313_Airport_Main_MAVICM3MFIXEDM3M_tile001_OM_shift.tif --out_mask datafiles/process_out/masks/crown_mask.tif --out_image_dir datafiles/process_out/images
```
### 1b)
```bash
uv run python explore_and_process/apply_dsm_mask.py --mask datafiles/process_out/masks/crown_mask.tif --dsm datafiles/raster/20260313/20260313_Airport_Main_MAVICM3MFIXEDM3M_tile001_DSM_shift.tif --out datafiles/process_out/masks/crown_mask_final.tif --out_dsm datafiles/process_out/ndsm/dsm_ndsm.tif --method  both --gradient_sigma 1 --combine and --windows 150 350 700 --height_threshold 2.0 
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