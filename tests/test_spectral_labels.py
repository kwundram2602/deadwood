import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deadwood_spectral.labels import label_box, label_boxes, label_count  # noqa: E402


def test_label_boxes_counts_and_bounds_every_label():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:3, 1:4] = 1          # 6 px
    labels[6:9, 5:7] = 2          # 6 px
    labels[9, 9] = 3              # 1 px

    counts, boxes = label_boxes(labels)

    assert label_count(counts, 1) == 6
    assert label_count(counts, 2) == 6
    assert label_count(counts, 3) == 1
    assert counts[0] == 100 - 13   # background
    assert label_box(boxes, 1) == (slice(1, 3), slice(1, 4))
    assert label_box(boxes, 2) == (slice(6, 9), slice(5, 7))
    # The box crops to exactly the object: the mask inside it is the object.
    box = label_box(boxes, 2)
    assert (labels[box] == 2).all()


def test_label_boxes_reports_absent_and_out_of_range_labels_as_missing():
    """A gap in the label sequence must be None, not a wrong box.

    find_objects returns one entry per label id up to max, with None for ids
    that carry no pixels; callers rely on that to skip them.
    """
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[0, 0] = 1
    labels[3, 3] = 3              # label 2 is absent

    counts, boxes = label_boxes(labels)

    assert label_box(boxes, 2) is None
    assert label_count(counts, 2) == 0
    assert label_box(boxes, 3) == (slice(3, 4), slice(3, 4))
    # Out of range in either direction is missing, not an IndexError.
    assert label_box(boxes, 99) is None
    assert label_box(boxes, 0) is None
    assert label_count(counts, 99) == 0


def test_label_boxes_on_an_empty_raster():
    counts, boxes = label_boxes(np.zeros((4, 4), dtype=np.int32))
    assert boxes == []
    assert label_count(counts, 1) == 0
