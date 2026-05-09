# Soft DSM Masking — Design Spec

**Date:** 2026-05-08  
**File:** `explore_and_process/apply_dsm_mask.py`

---

## Problem

`rasterize_crowns.py` produces a soft crown mask (Gaussian blur, values 0–1) with noData=255
for uncertain pixels. `apply_dsm_mask.py` currently hard-overwrites all DSM-detected ground
pixels with 0.0, destroying the soft boundary information from the rasterize step.

The `ground_conf` [0,1] array is already computed by both detection methods but is unused for
the actual mask modification.

---

## Pixel Classes (input mask)

| Value | Meaning |
|---|---|
| `0.0` | Confirmed non-crown (far from any polygon) |
| `0.05 … 1.0` | Soft crown probability (Gaussian blur of polygon raster) |
| `255.0` | noData — no polygon nearby, DSM not yet assessed |

---

## Design

### 1. Blending Logic

Replace the hard override (current line 274):

```python
# REMOVE
mask[ground_bin] = 0.0
```

With two soft rules:

```python
# Crown pixels (0–1): multiplicative dampening by DSM ground confidence
crown_pixels = (mask >= 0.0) & (mask < 255.0)
mask[crown_pixels] = mask[crown_pixels] * (1.0 - ground_conf[crown_pixels])

# noData pixels (255): resolve to ground only when DSM is very confident
nodata_pixels = mask == 255.0
resolve = nodata_pixels & (ground_conf > args.nodata_resolve_threshold)
mask[resolve] = 1.0 - ground_conf[resolve]
```

**Rationale for multiplicative blend:**  
A crown pixel with value 0.8 and `ground_conf=0.9` → `0.8 × 0.1 = 0.08`.
For clearly detected ground pixels (`ground_conf` near 1.0), this is effectively 0.
For ambiguous pixels near the height threshold, the existing crown label is partially
preserved — producing genuine soft transitions.

**Rationale for noData threshold:**  
noData pixels have no polygon information. Assigning them crown-like values based on
weak DSM evidence would introduce noise. Only high `ground_conf` (> threshold) provides
enough evidence to resolve them. The assigned value `1 - ground_conf` is near 0 for
high-confidence ground detections.

`ground_bin` (boolean) is no longer used for masking. It is retained for diagnostic
statistics only.

### 2. New CLI Parameter

```
--nodata_resolve_threshold  float  default=0.7
```

`ground_conf` must exceed this value for a noData pixel to be resolved.
Pixels below the threshold remain 255.

### 3. Confidence Output Files

**Directory renamed:** `gradient_crown_conf/` → `ground_confidence/`

All runs save individual method confidence TIFs (before combination) plus the combined result:

| File | Saved when | Content |
|---|---|---|
| `<mask_stem>_lm_conf.tif` | method includes `local_min` | raw `lm_conf` [0,1] |
| `<mask_stem>_gr_conf.tif` | method includes `gradient` | raw `gr_conf` [0,1] |
| `<mask_stem>_conf.tif` | always | combined `ground_conf` [0,1] |

For single-method runs, `_conf.tif` and the corresponding individual file are identical
in content but both are written for consistency.

### 4. Statistics Output

```
# Before (removed)
Ground pixels forced to 0: 1,234,567

# After (new)
Crown pixels dampened by DSM:    1,234,567  (multiplicative blend)
noData pixels resolved to ground:   89,012  (ground_conf > 0.70)
```

### 5. Combine Logic — No Change

The existing `combine()` function (fuzzy max/min) is unchanged. `max` for OR and `min`
for AND is appropriate for correlated signals (both methods read the same DSM). A
probabilistic combination (`1 - (1-a)(1-b)`) would overestimate confidence for correlated
inputs.

---

## What Does Not Change

- `detect_ground_local_min()` — unchanged
- `detect_ground_gradient()` — unchanged
- `combine()` — unchanged
- All CLI parameters except the addition of `--nodata_resolve_threshold`
- noData sentinel value (255) and its meaning throughout the pipeline
- nDSM output path and logic
