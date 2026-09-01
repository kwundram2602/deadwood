"""Central noData sentinels for the crown mask.

The pipeline has to tell two fundamentally different kinds of "no label" apart:

``MASK_OUTSIDE`` (-1.0)
    Outside the recorded scene footprint — the drone never saw this ground, so
    the image bands carry no information either. Predictions there are
    meaningless and must be suppressed.

``MASK_UNLABELLED`` (255.0)
    Inside the footprint, but no crown polygon is near enough for the soft mask
    to make a statement. The imagery is perfectly valid; only the *label* is
    missing, so these pixels are excluded from the loss but are legitimate
    prediction targets.

Both are excluded from loss and metrics, which is why a single sentinel used to
be enough for training. It is not enough for inference: the model never gets a
gradient over the out-of-footprint region and will happily emit high crown
probabilities there unless that region is masked out explicitly.

The image side carries the same distinction as ``NaN``: rasterize_crowns writes
the scene stack with ``nodata=nan`` and leaves out-of-footprint pixels NaN, so
the footprint travels with the imagery instead of being re-guessed downstream
with an ``all bands == 0`` heuristic.

Sentinel values are chosen so that pre-existing code comparing against 255.0
keeps working: -1.0 falls outside every ``mask >= 0.0`` guard.
"""

import numpy as np

# Inside the footprint, no crown label available (soft-mask noData).
MASK_UNLABELLED: float = 255.0

# Outside the recorded scene footprint (true noData).
MASK_OUTSIDE: float = -1.0

# What a mask raster declares as its GDAL noData value. The out-of-footprint
# sentinel is the one a GIS should render transparent.
MASK_RASTER_NODATA: float = MASK_OUTSIDE


def valid_target(mask):
    """Boolean mask of pixels carrying a real label, i.e. a value in [0, 1].

    Works for numpy arrays and torch tensors alike (both support the operators),
    and excludes every sentinel by construction rather than by enumerating them.
    """
    return (mask >= 0.0) & (mask <= 1.0)


def footprint_from_stack(bands: np.ndarray) -> np.ndarray:
    """Scene footprint from a (C, H, W) image stack.

    A pixel is inside the footprint when every band has data. Stacks written by
    the current rasterize_crowns carry the footprint as NaN; older stacks
    predate that and encode it as an all-zero pixel, so fall back to the legacy
    heuristic when the stack contains no NaN at all.
    """
    nan_any = np.asarray(np.isnan(bands).any(axis=0))
    if nan_any.any():
        return ~nan_any
    return ~np.asarray(np.all(bands == 0.0, axis=0))
