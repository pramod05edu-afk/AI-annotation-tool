"""Turning raw model output into annotations: pure functions, no model needed."""

import numpy as np
import pytest

from app.ml.base import Detection
from app.ml.postprocess import (
    clip_box,
    drop_overlapping,
    map_to_project_classes,
    mask_to_polygon,
    pick_box_at_point,
    polygon_bbox,
    prompt_text,
    split_classes,
)


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


# ---------- smart click ----------

HELMET = Detection("helmet", (10, 10, 30, 30), 0.9)
WEAK_HEAD = Detection("head", (12, 12, 32, 32), 0.05)
FAR_AWAY = Detection("helmet", (60, 60, 80, 80), 0.99)


def test_pick_box_takes_the_most_confident_box_under_the_click() -> None:
    assert pick_box_at_point([WEAK_HEAD, HELMET, FAR_AWAY], (20, 20)) == HELMET


def test_pick_box_ignores_boxes_that_do_not_contain_the_click() -> None:
    assert pick_box_at_point([HELMET, FAR_AWAY], (45, 45)) is None


def test_pick_box_skips_objects_that_are_already_labelled() -> None:
    # The helmet is labelled already. A near-copy of it is skipped too (IoU 0.68 = duplicate);
    # a clearly different, weaker box under the click is found instead.
    other = Detection("head", (5, 5, 40, 45), 0.04)  # IoU with the label: 400 / 1400 = 0.29
    found = pick_box_at_point([HELMET, WEAK_HEAD, other], (20, 20), exclude=[(10, 10, 30, 30)])
    assert found == other


def test_pick_box_keeps_boxes_that_only_slightly_overlap_a_label() -> None:
    neighbour_label = (25, 10, 45, 30)  # IoU with HELMET = 100 / 700, well below 0.5
    assert pick_box_at_point([HELMET], (20, 20), exclude=[neighbour_label]) == HELMET


@pytest.mark.parametrize(
    ("label", "prompt"),
    [
        ("traffic_light", "traffic light"),
        ("safety-vest", "safety vest"),
        ("  Cat  ", "Cat"),
        ("hard  hat", "hard hat"),
        ("_", "_"),  # nothing left: keep the label rather than send an empty prompt
    ],
)
def test_prompt_text_turns_identifiers_into_words(label: str, prompt: str) -> None:
    assert prompt_text(label) == prompt


def test_split_classes_sends_known_labels_to_the_specialist() -> None:
    assert split_classes(["Helmet", "vest", "head", "cone"], ["helmet", "head"]) == (
        ["Helmet", "head"],
        ["vest", "cone"],
    )
    assert split_classes(["cat"], ["helmet", "head"]) == ([], ["cat"])
    assert split_classes([], ["helmet"]) == ([], [])


def test_drop_overlapping_keeps_only_new_objects() -> None:
    detections = [
        Detection("x", (0, 0, 10, 10), 0.9),  # same as an existing box
        Detection("x", (1, 1, 11, 11), 0.8),  # IoU 0.68 with it: duplicate
        Detection("x", (5, 0, 15, 10), 0.7),  # IoU 0.33: a different object
        Detection("x", (50, 50, 60, 60), 0.6),
    ]
    kept = drop_overlapping(detections, [(0, 0, 10, 10)])
    assert [d.score for d in kept] == [0.7, 0.6]
    assert drop_overlapping(detections, []) == detections
