# Soft DSM Masking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard ground-pixel override in `apply_dsm_mask.py` with a soft multiplicative blend that respects the existing Gaussian-blur crown probabilities from `rasterize_crowns.py`.

**Architecture:** Extract the masking logic into a pure function `apply_soft_blend()` for testability, then call it from `main()`. Individual method confidences (`lm_conf`, `gr_conf`) are saved as separate TIFs in a renamed `ground_confidence/` directory. A new `--nodata_resolve_threshold` CLI parameter controls when noData pixels are resolved.

**Tech Stack:** Python, NumPy, rasterio — all already used in the file. Tests via `uv run pytest`.

---

## File Map

| Action | Path | What changes |
|---|---|---|
| Modify | `explore_and_process/apply_dsm_mask.py` | New function + wiring + arg + dir rename + stats |
| Create | `tests/test_apply_dsm_mask.py` | Unit tests for `apply_soft_blend()` |

---

### Task 1: Write failing tests for `apply_soft_blend()`

**Files:**
- Create: `tests/test_apply_dsm_mask.py`

- [ ] **Step 1.1: Create the test file**

```python
# tests/test_apply_dsm_mask.py
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from explore_and_process.apply_dsm_mask import apply_soft_blend


def _arr(*values):
    return np.array(values, dtype=np.float32)


def test_crown_pixel_multiplied_by_one_minus_conf():
    mask = _arr(0.8)
    conf = _arr(0.9)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.8 * 0.1, abs=1e-6)


def test_crown_pixel_zero_conf_unchanged():
    mask = _arr(0.8)
    conf = _arr(0.0)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.8, abs=1e-6)


def test_crown_pixel_full_conf_becomes_zero():
    mask = _arr(0.6)
    conf = _arr(1.0)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_nodata_resolved_when_conf_above_threshold():
    mask = _arr(255.0)
    conf = _arr(0.8)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(1.0 - 0.8, abs=1e-6)


def test_nodata_stays_255_when_conf_below_threshold():
    mask = _arr(255.0)
    conf = _arr(0.5)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(255.0, abs=1e-6)


def test_nodata_stays_255_when_conf_exactly_at_threshold():
    # threshold is strict: must be *greater than*, not equal
    mask = _arr(255.0)
    conf = _arr(0.7)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(255.0, abs=1e-6)


def test_existing_zero_crown_pixel_stays_zero():
    mask = _arr(0.0)
    conf = _arr(0.9)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_mixed_array():
    mask = _arr(0.8, 255.0, 255.0, 0.0)
    conf = _arr(0.5, 0.9,   0.4,   0.8)
    result = apply_soft_blend(mask, conf, nodata_resolve_threshold=0.7)
    assert result[0] == pytest.approx(0.8 * 0.5, abs=1e-6)   # crown dampened
    assert result[1] == pytest.approx(1.0 - 0.9, abs=1e-6)   # noData resolved
    assert result[2] == pytest.approx(255.0, abs=1e-6)        # noData stays
    assert result[3] == pytest.approx(0.0, abs=1e-6)          # zero stays zero
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```
cd D:\EAGLE\InnoLab_DL\deadwood
uv run pytest tests/test_apply_dsm_mask.py -v
```

Expected: `ImportError` or `AttributeError` — `apply_soft_blend` does not exist yet.

---

### Task 2: Implement `apply_soft_blend()` + wire into `main()`

**Files:**
- Modify: `explore_and_process/apply_dsm_mask.py`

- [ ] **Step 2.1: Add `apply_soft_blend()` as a module-level function**

Insert after the `combine()` function (after line 152), before `_save_diagnostic()`:

```python
def apply_soft_blend(
    mask: np.ndarray,
    ground_conf: np.ndarray,
    nodata_resolve_threshold: float,
) -> np.ndarray:
    """Soft-blend ground confidence into the crown mask.

    Crown pixels (0–1): multiplied by (1 - ground_conf).
    noData pixels (255): resolved to (1 - ground_conf) only when
      ground_conf > nodata_resolve_threshold, otherwise kept at 255.
    """
    result = mask.copy()
    crown = (mask >= 0.0) & (mask < 255.0)
    result[crown] = mask[crown] * (1.0 - ground_conf[crown])

    nodata = mask == 255.0
    resolve = nodata & (ground_conf > nodata_resolve_threshold)
    result[resolve] = 1.0 - ground_conf[resolve]
    return result
```

- [ ] **Step 2.2: Add `--nodata_resolve_threshold` CLI argument**

In the `argparse` block at the bottom of the file, add after the `--max_ndsm_height` argument:

```python
p.add_argument("--nodata_resolve_threshold", type=float, default=0.7,
               help="ground_conf must exceed this to resolve a noData pixel (default: 0.7)")
```

- [ ] **Step 2.3: Replace the hard override in `main()` with the soft blend**

Find this block (currently around line 273–280):

```python
# --- Apply ground mask to crown mask -------------------------------------
mask[ground_bin] = 0.0

n_crown  = int(np.sum((mask > 0) & (mask < 255)))
n_ground = int(np.sum(mask == 0.0))
n_nodata = int(np.sum(mask == 255.0))
print(f"\nGround pixels forced to 0: {int(ground_bin.sum()):,}")
print(f"Final  =>  Crown: {n_crown:,}  Ground: {n_ground:,}  noData: {n_nodata:,}")
```

Replace with:

```python
# --- Apply soft ground blend to crown mask --------------------------------
n_nodata_before = int(np.sum(mask == 255.0))
mask = apply_soft_blend(mask, ground_conf, args.nodata_resolve_threshold)

n_crown_dampened = int(np.sum((mask >= 0.0) & (mask < 255.0)))
n_nodata_resolved = n_nodata_before - int(np.sum(mask == 255.0))
n_crown  = int(np.sum((mask > 0) & (mask < 255)))
n_ground = int(np.sum(mask == 0.0))
n_nodata = int(np.sum(mask == 255.0))
print(f"\nCrown pixels dampened by DSM:     {n_crown_dampened:,}  (multiplicative blend)")
print(f"noData pixels resolved to ground: {n_nodata_resolved:,}  (ground_conf > {args.nodata_resolve_threshold:.2f})")
print(f"Final  =>  Crown: {n_crown:,}  Ground: {n_ground:,}  noData: {n_nodata:,}")
```

- [ ] **Step 2.4: Run tests — all should pass**

```
cd D:\EAGLE\InnoLab_DL\deadwood
uv run pytest tests/test_apply_dsm_mask.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 2.5: Commit**

```
git add explore_and_process/apply_dsm_mask.py tests/test_apply_dsm_mask.py
git commit -m "feat: soft DSM ground blend — multiplicative crown dampening + noData threshold"
```

---

### Task 3: Save individual confidence TIFs + rename output directory

**Files:**
- Modify: `explore_and_process/apply_dsm_mask.py`

- [ ] **Step 3.1: Rename `conf_dir` from `gradient_crown_conf` to `ground_confidence`**

Find in `main()`:

```python
conf_dir = os.path.join(process_out_dir, "gradient_crown_conf")
```

Replace with:

```python
conf_dir = os.path.join(process_out_dir, "ground_confidence")
```

- [ ] **Step 3.2: Add `_save_conf_tif()` helper**

Insert directly above the `main()` function:

```python
def _save_conf_tif(arr: np.ndarray, path: str, profile: dict) -> None:
    """Write a float32 confidence raster [0,1] to disk."""
    conf_profile = profile.copy()
    conf_profile.update(dtype="float32", count=1, nodata=None)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **conf_profile) as dst:
        dst.write(arr[np.newaxis])
    print(f"  Saved confidence: {path}")
```

- [ ] **Step 3.3: Save individual confidences after detection, before combination**

Find the combine section in `main()`:

```python
# --- Combine -------------------------------------------------------------
if args.method == "local_min":
    ground_bin, ground_conf = lm_bin, lm_conf
```

Insert **before** that block:

```python
# --- Save individual confidences -----------------------------------------
os.makedirs(conf_dir, exist_ok=True)
mask_stem = os.path.splitext(os.path.basename(args.out))[0]
if lm_conf is not None:
    _save_conf_tif(lm_conf, os.path.join(conf_dir, f"{mask_stem}_lm_conf.tif"), profile)
if gr_conf is not None:
    _save_conf_tif(gr_conf, os.path.join(conf_dir, f"{mask_stem}_gr_conf.tif"), profile)
```

- [ ] **Step 3.4: Replace the existing combined confidence save block**

Find the current confidence save block (around line 289–295):

```python
# --- Write confidence raster ---------------------------------------------
os.makedirs(conf_dir, exist_ok=True)
conf_profile = profile.copy()
conf_profile.update(dtype="float32", count=1, nodata=None)
with rasterio.open(conf_path, "w", **conf_profile) as dst:
    dst.write(ground_conf[np.newaxis])
print(f"Saved confidence: {conf_path}")
```

Replace with:

```python
# --- Write combined confidence raster ------------------------------------
_save_conf_tif(ground_conf, conf_path, profile)
```

- [ ] **Step 3.5: Run tests to confirm nothing broke**

```
cd D:\EAGLE\InnoLab_DL\deadwood
uv run pytest tests/test_apply_dsm_mask.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 3.6: Commit**

```
git add explore_and_process/apply_dsm_mask.py
git commit -m "feat: save individual lm/gr confidence TIFs to ground_confidence/"
```

---

## Self-Review

**Spec coverage:**
- [x] Multiplicative blend for crown pixels — Task 2
- [x] noData resolved via threshold — Task 2
- [x] `--nodata_resolve_threshold` CLI param (default 0.7) — Task 2
- [x] Individual `_lm_conf.tif` and `_gr_conf.tif` always saved — Task 3
- [x] Combined `_conf.tif` still saved — Task 3
- [x] Directory renamed `gradient_crown_conf` → `ground_confidence` — Task 3
- [x] Stats output updated — Task 2
- [x] `ground_bin` retained for internal use (not removed) — Task 2 (used only in `n_ground` stat via `mask == 0.0`)
- [x] `combine()` logic unchanged — not touched

**Placeholder scan:** No TBDs or vague steps found.

**Type consistency:** `apply_soft_blend(mask, ground_conf, nodata_resolve_threshold)` — same signature in test imports and implementation.
