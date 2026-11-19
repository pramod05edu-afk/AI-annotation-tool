"""Tests for app.services.geometry — these define the expected behavior for exercises E3 and E4."""

import pytest

from app.services.geometry import coco_to_voc, iou, voc_to_coco, voc_to_yolo, yolo_to_voc

# The worked example from the module docstring: a 100x50 image, box from (10, 20) to (30, 40).
VOC = (10.0, 20.0, 30.0, 40.0)
COCO = (10.0, 20.0, 20.0, 20.0)
YOLO = (0.2, 0.6, 0.2, 0.4)
IMG_W, IMG_H = 100, 50


# ---------- E3: format conversions ----------


def test_voc_to_coco() -> None:
    assert voc_to_coco(VOC) == pytest.approx(COCO)


def test_coco_to_voc() -> None:
    assert coco_to_voc(COCO) == pytest.approx(VOC)


def test_voc_to_yolo() -> None:
    assert voc_to_yolo(VOC, IMG_W, IMG_H) == pytest.approx(YOLO)


def test_yolo_to_voc() -> None:
    assert yolo_to_voc(YOLO, IMG_W, IMG_H) == pytest.approx(VOC)


def test_full_image_box_is_centered_in_yolo() -> None:
    assert voc_to_yolo((0, 0, 416, 416), 416, 416) == pytest.approx((0.5, 0.5, 1.0, 1.0))


@pytest.mark.parametrize(
    "box",
    [(0, 0, 1, 1), (357, 116, 404, 175), (4, 146, 39, 184), (0, 0, 416, 415)],
)
def test_round_trips_preserve_the_box(box: tuple[float, float, float, float]) -> None:
    assert coco_to_voc(voc_to_coco(box)) == pytest.approx(box)
    assert yolo_to_voc(voc_to_yolo(box, 416, 415), 416, 415) == pytest.approx(box)


# ---------- E4: IoU ----------


def test_iou_identical_boxes_is_one() -> None:
    assert iou(VOC, VOC) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_touching_edges_is_zero() -> None:
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_half_overlap() -> None:
    # overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_box_inside_another() -> None:
    # small 4x4 = 16 fully inside big 10x10 = 100 -> 16 / 100
    assert iou((0, 0, 10, 10), (3, 3, 7, 7)) == pytest.approx(0.16)


def test_iou_is_symmetric() -> None:
    a, b = (0, 0, 10, 10), (4, 2, 12, 9)
    assert iou(a, b) == pytest.approx(iou(b, a))
