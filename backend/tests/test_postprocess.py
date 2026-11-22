"""Turning raw model output into annotations: pure functions, no model needed."""

import numpy as np

from app.ml.base import Detection
from app.ml.postprocess import clip_box, map_to_project_classes, mask_to_polygon, polygon_bbox


def test_clip_box_keeps_inside_boxes() -> None:
    assert clip_box((1, 2, 30, 40), 100, 50) == (1, 2, 30, 40)


def test_clip_box_cuts_at_the_image_border() -> None:
    assert clip_box((-5, 10, 120, 60), 100, 50) == (0, 10, 100, 50)


def test_clip_box_outside_the_image_is_none() -> None:
    assert clip_box((110, 10, 130, 20), 100, 50) is None


def test_map_classes_is_case_insensitive_and_counts_ignored() -> None:
    detections = [
        Detection("HELMET", (0, 0, 1, 1), 0.9),
        Detection("person", (0, 0, 1, 1), 0.8),
        Detection("person", (0, 0, 1, 1), 0.7),
    ]
    kept, ignored = map_to_project_classes(detections, ["helmet", "head"])
    assert [d.class_name for d in kept] == ["helmet"]
    assert ignored == {"person": 2}


def test_mask_to_polygon_outlines_a_rectangle() -> None:
    mask = np.zeros((50, 80), dtype=bool)
    mask[10:30, 20:60] = True  # rows 10..29, columns 20..59
    polygon = mask_to_polygon(mask)
    assert polygon is not None
    assert len(polygon) == 4  # a rectangle simplifies to its 4 corners
    assert polygon_bbox(polygon) == (20, 10, 59, 29)


def test_mask_to_polygon_takes_the_largest_region() -> None:
    mask = np.zeros((50, 80), dtype=bool)
    mask[0:3, 0:3] = True  # small speck
    mask[20:40, 40:70] = True  # the object
    polygon = mask_to_polygon(mask)
    assert polygon is not None
    assert polygon_bbox(polygon) == (40, 20, 69, 39)


def test_empty_mask_is_none() -> None:
    assert mask_to_polygon(np.zeros((10, 10), dtype=bool)) is None
