import pytest

from practice.iou import iou


def test_identical_boxes() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_no_overlap() -> None:
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_touching_edges_is_zero() -> None:
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_half_overlap() -> None:
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_box_inside_another() -> None:
    assert iou((0, 0, 10, 10), (3, 3, 7, 7)) == pytest.approx(0.16)


def test_symmetric() -> None:
    a, b = (0, 0, 10, 10), (4, 2, 12, 9)
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_zero_area_boxes_do_not_crash() -> None:
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0
