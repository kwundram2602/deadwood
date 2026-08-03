"""Per-object bookkeeping for connected-component label rasters.

The reference grid is 6459 x 6962 px. A loop of the form

    for label_id in range(1, labels.max() + 1):
        mask = labels == label_id          # whole-scene boolean, 45 MB
        n_pixels = int(mask.sum())

costs ~59 ms per component on that grid regardless of how small the component
is, so it is O(n_components * scene). A class raster with only 0.2% scattered
stray pixels already yields ~89,000 components — about 88 minutes of pure
masking, and hours at a plausible RandomForest false-positive rate.

`label_boxes` replaces that with two single passes over the raster:
`np.bincount` for every component's pixel count and `scipy.ndimage.find_objects`
for every component's bounding box. Callers then work inside the bounding-box
slice, so each object's cost is proportional to its own size, not the scene's.
"""

import numpy as np
from scipy import ndimage


def label_boxes(labels: np.ndarray) -> tuple[np.ndarray, list[tuple[slice, slice] | None]]:
    """(counts, boxes) for a connected-component label raster.

    counts[i] is the number of pixels carrying label i (index 0 is the
    background count). boxes[i - 1] is the bounding-box slice tuple of label i,
    or None if that label is absent — the indexing convention
    `scipy.ndimage.find_objects` uses.
    """
    labels = np.asarray(labels)
    max_label = int(labels.max()) if labels.size else 0
    counts = np.bincount(labels.reshape(-1), minlength=max_label + 1)
    boxes = ndimage.find_objects(labels, max_label=max_label) if max_label else []
    return counts, list(boxes)


def label_count(counts: np.ndarray, label_id: int) -> int:
    """Pixel count of one label, 0 when the label is out of range."""
    if label_id < 0 or label_id >= counts.size:
        return 0
    return int(counts[label_id])


def label_box(boxes: list, label_id: int):
    """Bounding-box slice tuple of one label, None when absent."""
    if label_id < 1 or label_id > len(boxes):
        return None
    return boxes[label_id - 1]
