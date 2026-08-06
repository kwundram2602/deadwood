# Handoff — Deadwood Spectral Time Series

**Branch:** `feature/deadwood-spectral-timeseries` (27 commits ahead of `master`, head `d1e4954`)
**Written:** 2026-08-03
**Status:** code complete and reviewed; **nothing has run end to end on real data yet**

Read this before touching anything. The most expensive mistakes available here are
silent ones — wrong labels, wrong grid, wrong dates — and several of them were already
made once and fixed.

---

## 1. What the project does

Detect standing deadwood in a savanna AOI from a multi-year drone orthomosaic time
series, on top of an existing crown-segmentation pipeline.

**The hypothesis, in one sentence:** a living deciduous savanna tree is leafless in the
dry season too, so a single dry-season date cannot separate dead from dormant — but
across a season the living tree greens up and deadwood stays flat.

This is not a guess. Measured on the real field polygons (median NDVI):

| Date | deadwood | living | gap |
|---|---|---|---|
| 20250807 | 0.293 | 0.308 | **0.015** |
| 20250907 | 0.254 | 0.269 | **0.015** |
| 20251027 | 0.220 | 0.488 | 0.267 |
| **20251121** | 0.149 | 0.468 | **0.320** |

Dry season: indistinguishable. Green-up: a twentyfold gap. That is the entire
justification for a time series over a single flight.

**Second measured fact, which matters for feature importance:** the deadwood NDVI curve
tracks the *background* curve almost exactly (20251121: deadwood 0.149, background
0.145). A leafless dead crown lets you see the ground through it, and the ground greens
up. So deadwood shows a seasonal amplitude of ~0.22 that is *not the tree*. Expect
`ndsm` (height above ground) to carry more weight than a purely spectral view suggests
— it is what separates a standing dead trunk from a dry grass patch.

---

## 2. Architecture

New code lives in `deadwood_spectral/`, entrypoints in `scripts/spectral_*.py`, configs
in `configs/spectral/`. Three stages:

```
OM_domAligned/*.tif ──align──> timeseries/<date>_stack.tif   (5 cm, reference grid)
                        └─coreg──> coreg_report.csv           (flags bad dates)
crown_mask.tif ─┐
crown pred      ├─sampling──> 3 label pools ──extract──> samples.parquet
crown_poly.gpkg │                                            │
ndsm ───────────┘                          ┌─────────────────┤
                                     report.py          features.py ──> classify.py ──> model
                                           │                                  │
                                     out/spectral/                        apply.py ──> deadwood_prob.tif
                                                                                       deadwood_class.tif
                                                                                       deadwood_objects.gpkg
                                                                                       (+ retrospect.py, off by default)
```

| Module | Responsibility |
|---|---|
| `grid.py` | the one reference-grid contract; everything asserts against it |
| `indices.py` | pure spectral index maths, no I/O |
| `align.py` | reproject a scene onto the reference grid, idempotently |
| `coreg.py` | per-date residual shift on stable tiles → report |
| `labels.py` | per-object bookkeeping (`bincount` + `find_objects`) — see §6 |
| `sampling.py` | the three disjoint label pools + grouped, balanced draw |
| `extract.py` | sampled pixels → one table |
| `report.py` | descriptive plots + separability |
| `features.py` | table → fixed-length feature matrix |
| `classify.py` | RF, grouped CV, leave-one-tree-out, variants |
| `apply.py` | block-wise scene inference + object aggregation |
| `retrospect.py` | apply an existing model to earlier cycles (optional) |

---

## 3. Hard facts about the data

- **Reference grid:** `datafiles/process_out/masks/crown_mask.tif` — EPSG:32736,
  6459 × 6962 px, 5 cm GSD. Every raster in the pipeline must sit on it exactly.
- **Time series:** 56 aligned dates, `20230824` … `20260313`, in
  `datafiles/process_out/timeseries/`. 7 bands `('R','G','B','Green','Red','RedEdge','NIR')`,
  float32, band-interleaved, ~0.76 GB each, **42.4 GB total**. Verified: all 56 pass the
  grid contract, correct band count, descriptions and dtype.
- **~45% of every aligned stack is NaN** — the area outside the source footprint. The
  crown prediction independently reports 45.8% nodata. Same footprint. Your usable AOI
  is a little over half of the grid.
- **Ground truth:** 18 `soff` (standing dead) polygons against 80 `son`, in
  `datafiles/crown_poly/2_crown_main_20260409_editLP.gpkg`, surveyed March/April 2026.
- **The default quality filter leaves 7 of the 18.** `soff` `certaintyLP` distribution
  is `0:9, 20:1, 50:3, 100:5`; species coverage drops from 6 to 4. Survivors: 4136,
  4302, 4321, 4323, 4325, 4333, 4341.
- **`soff` crown areas span 0.02 to 13.86 m².** `erode_m=0.10` destroys the 0.02 m²
  polygon entirely (17 of 18 survive). `build_pools` logs which tree_ids it dropped —
  read that log line.
- **Field-data quirks the code already handles:** `coverage` contains `'nc '` with a
  trailing space; `crown_category` has one empty string; `soff` 4389 overlaps `son` 4336
  by 76.9% of the soff crown.

---

## 4. Configuration — all four blanks are filled

| Key | Value | Why |
|---|---|---|
| `paths.reference` | `crown_mask.tif` | grid definition only, not a crown source |
| `paths.crown_prediction` | the model's `*_pred_t0.9.tif` under `sample_exp/…/predict/` | the `living` pool must come from what the model predicts, not from `son` polygons |
| `paths.ndsm` | `ndsm_in_m/…/dsm_ndsm_dtm_raw_m.tif` | metres, **identical in both configs** |
| `coreg.tiles` | 6 computed candidates | see §5 |
| `classify.cycle.dates` | 12 dates, `20250417`…`20260313` | see §5 |

**Traps that are already documented inline in the configs but worth repeating:**

- Use the **binarised** prediction, not `*_prob.tif`. The probability raster's nodata is
  `-1.0` and `binarize_crown_mask` only recognises `255`, so a `-1` pixel would read as
  "valid but below threshold" — background rather than "not evaluated" — and 45.8% of
  the grid would flood the background pool with all-NaN samples.
- `paths.ndsm` must name the **same file** in `analysis.yaml` and `classify.yaml`. Two
  variants exist on disk, both on the reference grid, so the grid check cannot tell them
  apart. There is an nDSM identity guard (path/size/shape/window checksum) that travels
  samples → model → apply and fails loudly on a mismatch; it was verified against the
  two real files (checksums 24331 vs 36652, sizes 80114129 vs 79424268).

---

## 5. Two things that were computed, not guessed

**`coreg.tiles`** — six 512 px (25.6 m) squares, scored over 8 dates sampled across the
series at 1/16 resolution on: low mean NDVI (non-vegetated), low NIR standard deviation
across time (stable), full validity on every sampled date, ≥15 m from any crown polygon,
≥60 m from each other. NDVI 0.12–0.26, NIR sd 0.043–0.057.

They are **ground control only** — they never enter the spectral analysis, the sampling
or the classifier. The same map square is read from every date; any difference in where
its content sits is misregistration, not real change. That is why they must be
non-vegetated: vegetation changes for real.

*Caveat:* all six sit between northing 7238245 and 7238313 — the southern third of the
AOI — because the valid footprint plus the crown buffer leaves nothing usable further
north. East–west spread is good (columns 200 to 5698). A north–south rotation or scale
error would resolve poorly. Add a northern tile if a visual check turns one up.

**`classify.cycle.dates`** — 12 dates thinned from the 19 available in the 2025/26
cycle, by spacing picks evenly along the **arc length** of the living-crown NDVI curve
rather than along time. The information is in the transitions, not the plateaus: three
dates inside five weeks across the green-up, four dry-season dates dropped because they
repeat each other. Forced keeps: dry trough `20250907`, wet peak `20260226`, maximum
separation `20251121`, survey anchor `20260313`.

Feature widths that follow: `6N + 12 + 1`.

| variant | features |
|---|---|
| `full` | 85 |
| `reduced` | 13 |
| `baseline` | 7 |

`baseline` is the bar the time series must clear. If 7 features from one date match 85
from twelve, that is a valuable practical result — eleven fewer flights — not a failure.

---

## 6. What is verified and what is not

**Verified:** 322 tests pass (`uv run pytest tests/ --ignore=tests/test_metrics.py -q`),
ruff clean on every touched file, all 56 stacks pass the grid contract, both nDSM
candidates pass, the crown prediction binarises to 5,654,733 crown / 24,383,099 valid /
20,584,459 invalid pixels.

**Not verified:** every test uses synthetic rasters from 4×4 to 100×100 px. The only
real-data contact so far is a single-scene alignment smoke test and the NDVI curve in §1.
No stage has run end to end.

**Pre-existing repo defects — not from this branch, confirmed at base `3fd2460`, do not
try to fix them as part of this work:**
- `tests/test_metrics.py` fails to collect (imports a nonexistent `pixel_metrics` from
  `training.metrics`). Always run the suite with `--ignore=tests/test_metrics.py`.
- `ruff check .` reports errors in `utils/viz.py`, `scripts/train.py`,
  `tests/test_resample_image.py`.

---

## 7. Next steps, in order

```sh
# 1. Co-registration report (does NOT re-align; is_aligned skips all 56)
uv run python scripts/spectral_align.py --config configs/spectral/align.yaml

# 2. Stage B — the go/no-go
uv run python scripts/spectral_report.py --config configs/spectral/analysis.yaml

# 3. Stage C
uv run python scripts/spectral_classify.py --config configs/spectral/classify.yaml
uv run python scripts/spectral_apply.py    --config configs/spectral/classify.yaml
```

**After step 1**, read `coreg_report.csv`:
- `status` saying `min_tiles` means too many tiles were rejected as NaN-heavy — the
  tiles sit too close to the footprint edge, not a data problem.
- `spread_m` much smaller than `dx_m`/`dy_m` means six independent locations agree and
  the offset is real. `spread_m` comparable to the offset means the tiles are not stable
  enough and the number is worthless.
- `flagged` is set either when fewer than `min_tiles=3` tiles were usable (then
  dx/dy/spread are NaN — no estimate at all rather than a confident-looking one) or when
  `hypot(dx, dy) > max_shift_m = 0.15 m` (3 px; the deadwood crowns are ~1.7 m across
  and the signal lives at their edges).

**⚠ The interaction that will bite:** flagged dates are excluded from `samples.parquet`.
If a flagged date is one of the 12 in `classify.cycle.dates`, Stage C aborts with
`missing column ndvi_<date>`. That is deliberate loud failure — but it means you must
either fix the co-registration or edit the cycle list. The 12 chosen dates are not
immune.

**After step 2**, open `out/spectral/<timestamp>/seasonal_amplitude.png`. **This is the
decision point of the project.** If deadwood and living do not separate there, Stage C
rests on a hypothesis the data rejected, and the finding should be reported rather than
worked around. Expect the separability heatmap to be near zero in Aug/Sep and to light
up around 20251121. Note the §1 curve used *uneroded* polygons at 20 cm while Stage B
uses *eroded* ones at 5 cm — separation should improve, but absolute numbers will differ.

**Also after step 2:** the QGIS overlay check of one aligned stack against
`crown_mask.tif`. It is the only check that catches a self-consistent but wrongly
positioned georeference; the unit tests work on 4×4 rasters and structurally cannot.

---

## 8. Defects found and fixed — do not reintroduce these

All were found by review, all were in the plan rather than the implementation, and all
would have produced plausible-looking wrong answers rather than crashes.

| Where | Defect | Consequence avoided |
|---|---|---|
| `rasterize_crowns.py` | `read_scaled_bands` rescales a source's own extent instead of reprojecting | time-series stacks shifted ~40 px behind a correct-looking georeference header |
| `sampling.draw_samples` | burned **all** polygons into one attribution raster; `rasterize` is last-shape-wins | 77% of one deadwood tree's pixels would inherit a **living** tree's `group_id`, silently breaking the grouped-CV guarantee |
| `report.class_auc` | directional AUC where a symmetric separation measure was needed | `best_date` would have selected the **worst**-separated date as the baseline |
| `classify` | `n_estimators` never threaded through | CV metrics describing a different forest than the one shipped |
| `apply.predict_scene` | non-finite pixels fell through `argmax` to class 0 | ~45% of the scene rendered as confident "no deadwood" instead of nodata |
| `retrospect` | raw `argmax` again | same bug, in the retrospective rasters |
| `report` / `features` | pandas `skipna=True` vs numpy NaN-propagating amplitude | the hypothesis plot would mix 12-date and 2-date amplitudes, showing observation count rather than phenology |
| `apply`, `retrospect` | whole-scene boolean mask per connected component | ~89,000 components × 59 ms ≈ 88 min, hours at a realistic false-positive rate — fixed via `labels.py` |
| `align.py` | pixel-interleaved tiles + band-at-a-time writes | tile rewrite/append left dead space: 3.35 GB written for 0.78 GB of data, factor 4.3 |

**Invariants worth protecting:**
- Non-finite pixels are 255 in class rasters and NaN in probability rasters.
- Only `soff` polygons are burned for deadwood attribution.
- Every metric is grouped by `group_id`; no tree's pixels in both train and test.
- Column order is a correctness contract, persisted in `feature_names.json` and asserted
  at inference. A RandomForest silently accepts a reordered matrix.
- Tiling in `apply.py` is a memory device, **not** a smoothing device — overlapping
  tiles overwrite with identical values. This is the opposite of `scripts/predict.py`'s
  Hann-blended CNN tiling. Do not add blending.

---

## 9. Deferred minors

Recorded, reviewed, deliberately not fixed. Full list with rulings in
`.superpowers/sdd/2026-08-02-deadwood-spectral-timeseries/progress.md`. The two most
likely to matter:

- **`estimate_shift` fills NaN with the tile mean.** Real stacks are ~45% NaN. There is
  now a per-tile NaN-fraction rejection (`max_tile_nan_frac`, default 2%), but the fill
  itself remains. Mitigation is tile selection.
- **The scale test in `tests/test_spectral_apply.py` asserts wall-clock time.** A flake
  needs a 30× slowdown; the correctness half is covered deterministically by the
  geometry test. Replace with an allocation count if it ever flakes.

---

## 10. Conventions

- `uv` for everything: `uv run python …`, `uv run pytest …`, `uv run ruff check …`.
- ruff: line-length 100, `select = ["E","F","I","UP"]`, `ignore = ["E501"]`.
- Tests follow `tests/test_apply_dsm_mask.py`: `sys.path.insert(...)` before repo
  imports, `# noqa: E402`.
- **Commit messages carry no `Co-Authored-By`, `Contributor` or "Generated with"
  trailer.** This is a standing instruction from the repo owner.
- `docs/` is gitignored — never `git add -f` it.
- There is a long-standing uncommitted `.gitignore` change in the working tree
  (adding `datafiles/OM_domAligned`). Leave it alone unless asked.
- Background processes started with `nohup … &` are killed when a tool call returns in
  this environment. Run long jobs in the foreground or detach them outside the session.
