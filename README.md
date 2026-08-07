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
dormant — the seasonal amplitude can.

**Nothing below has been run end to end on real data yet.** The time series is
not aligned on disk, no crown prediction exists, and `configs/spectral/analysis.yaml`
and `configs/spectral/classify.yaml` still carry `<predicted-mask-path>` /
`<ndsm-path>` placeholders for `crown_prediction` and `ndsm`. Before Stage B can
run, produce:

1. A crown prediction over the full AOI (`scripts/predict.py`, see Prediction
   above), and fill `sampling.crown_prediction` in `analysis.yaml`.
2. The matching nDSM under `datafiles/process_out/ndsm_in_m/`, and fill
   `paths.ndsm` in both `analysis.yaml` and `classify.yaml` — with the **same
   file**. Two nDSM variants exist (metres and normalized) and both sit on the
   reference grid, so nothing about the rasters themselves can tell them
   apart. Stage B records the nDSM's identity next to `samples.parquet`,
   Stage C copies it into the model directory as `ndsm_reference.json`, and
   `spectral_apply.py` refuses to run against a different one.
3. Stage A's aligned stacks, then a manual check: overlay one aligned date and
   `crown_mask.tif` in QGIS and confirm the crowns coincide before trusting
   anything downstream.

### Stage A — Align the time series

Reprojects every scene in `datafiles/OM_domAligned/` onto the `crown_mask.tif`
grid (5 cm, EPSG:32736) and reports the residual co-registration per date. Idempotent:
re-run it as new dates arrive.

```
uv run python scripts/spectral_align.py --config configs/spectral/align.yaml
```

`coreg.tiles` in `align.yaml` starts empty — pick stable, non-vegetated tiles in
QGIS first and list them as `[x, y]` map-coordinate pairs, or the co-registration
report is skipped with a warning. Then, before trusting the output, overlay one
aligned stack and the crown mask in QGIS — the crowns must coincide.

List 4–6 tiles, and pick them **fully inside the source footprint**. The shift
estimate fills NaN with the tile mean, and ~45% of a real aligned stack is NaN
outside the footprint: a large filled block abutting real data is a step edge
that can drag the phase-correlation peak. Tiles above `coreg.max_tile_nan_frac`
(2% by default) are rejected per date, and a date left with fewer than
`coreg.min_tiles` usable tiles gets no estimate at all and is flagged (and so
excluded downstream by `extract.exclude_flagged_dates`). The report says which:
`n_tiles`, `n_tiles_total`, `n_tiles_rejected_nan` and `status`. `spread_m` is
NaN rather than 0 when only one tile was used — one tile agrees with itself.

### Stage B — Sample and describe

Draws grouped, balanced samples from three pools (deadwood from the eroded `soff`
polygons, living from the crown prediction, background from the rest), extracts
every band and index for every date into `samples.parquet`, and writes the
separability report.

```
uv run python scripts/spectral_report.py --config configs/spectral/analysis.yaml
```

Two sampling details cost real ground truth and are worth knowing before reading
the numbers:

- The quality filter (`certaintyLP >= 50` and `coverage == 'nc'`) leaves only
  **7 of the 18** `soff` trees. The real `certaintyLP` distribution is
  0:9, 20:1, 50:3, 100:5.
- `sampling.erode_m` (0.10 m) shrinks each `soff` crown before rasterizing it as
  deadwood. The smallest real `soff` crown is 0.02 m² — 7 pixels at 5 cm — and
  erosion empties it completely, so only 17 of 18 trees actually contribute
  deadwood pixels. `build_pools` logs which `tree_id`s it dropped.

The key output is `seasonal_amplitude.png` (alongside `summary.csv`,
`coverage.csv`, `amplitude_population.csv` and `separability_jm.png` in the
timestamped `out/spectral/<stamp>/` run directory): if deadwood and living do
not separate there, the approach does not carry.

Two things to know before reading those two artefacts:

- **The amplitude is only computed for pixels observed on *every* date of the
  cycle.** A partial amplitude — max − min over whatever dates a pixel
  happened to be seen on — is not comparable between rows, and with ~45% NaN
  per stack and a different footprint per date it would mostly measure how
  often a pixel was observed. The plot's axis labels carry `n=<complete>/<all>`
  per class and `amplitude_population.csv` has the full counts, so the
  population is always stated. This is the same NaN-propagating definition the
  classifier feature `ndvi_amplitude` uses — Stage B and Stage C no longer
  disagree.
- **`summary.csv` has two AUC columns.** `auc_raw` is directional,
  P(deadwood > living): deadwood NDVI sits *below* living NDVI, so a
  well-separated date shows `auc_raw` near 0. `auc_sep` is the folded
  magnitude, `max(auc_raw, 1 − auc_raw)` — "how well separated", regardless of
  which class is on top — and it is what `best_date` selects the single-date
  baseline on. Read `auc_sep` for separation, `auc_raw` for direction.

### Stage C — Classify and map

Trains a RandomForest for every combination of three feature variants
(`full`, `reduced`, `baseline`) and two label sets — `filtered` (the quality
filter above) and `all` (every sampled row, since the filter costs 11 of 18
ground-truth trees) — validated grouped by tree (leave-one-tree-out over the
`soff` trees). `classify.primary_variant` / `classify.primary_label_set`
(`full` / `filtered` by default) picks which of the trained models is
persisted and used downstream. Then applies that model to the whole scene.

```
uv run python scripts/spectral_classify.py --config configs/spectral/classify.yaml
uv run python scripts/spectral_apply.py    --config configs/spectral/classify.yaml
```

`variant_comparison.csv` carries `n_samples`, `n_groups` and
`n_deadwood_groups` per row, and they are **not** the same across variants.
Rows missing a value on any of a variant's dates are dropped before fitting,
so the 1-date `baseline` keeps rows the 12-date `full` drops — and can keep
whole trees that `full` loses. Where those numbers differ between two rows,
part of the score difference is a population difference rather than a
feature-set difference; the run logs a warning naming the affected label set.
`train_variant` also logs the NaN drop per class, how many groups lost rows,
and names any group that lost all of them — with only 7 trees in the filtered
label set, losing one silently would matter.

Outputs: `deadwood_prob.tif`, `deadwood_class.tif`, `deadwood_objects.gpkg`
(paths set in `classify.yaml`'s `apply` block). Roughly 45% of a real aligned
stack is NaN — the area outside the source scene's footprint. In
`deadwood_class.tif` that shows up as the value **255** ("not evaluated", not
"no deadwood"); in `deadwood_prob.tif` it shows up as **NaN**. Don't read
either as a deadwood-absence signal outside the valid footprint.

Set `retrospect.enabled: true` and fill in `retrospect.cycles` to additionally
apply the model to earlier seasonal cycles and estimate when each object died.
Cycle keys must be zero-padded `"YYYY_YY"` labels (e.g. `"2023_24"`) — they are
sorted lexicographically and that is trusted to be chronological order — and
each cycle needs exactly as many dates as the training cycle, since the feature
vector has a fixed length. The result, `mortality_timing.csv`, carries
`first_dead_cycle_coverage` and `low_confidence` alongside `first_dead_cycle`:
a cycle where most of an object's footprint was nodata can otherwise look
"still alive" instead of "unobserved".

### Caveats

Only 18 deadwood trees exist in the field data, so every metric is validated
grouped by tree and the per-tree spread matters more than the mean. The
quality filter and the erosion step above both shrink that 18 further before
training ever sees it. The living labels come from the model's crown
prediction, not from field polygons, and can contain undetected deadwood.
