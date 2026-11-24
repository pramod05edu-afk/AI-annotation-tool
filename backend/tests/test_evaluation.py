"""The matching logic behind every number in the evaluation."""

import pytest

from app.services.evaluation import Counts, Labeled, counts_of, match, read_yolo_labels, size_bucket

HELMET = Labeled("helmet", (0, 0, 10, 10))


def test_perfect_prediction_is_a_true_positive() -> None:
    result = match([Labeled("helmet", (0, 0, 10, 10), 0.9)], [HELMET])
    assert counts_of(result) == Counts(tp=1, fp=0, fn=0)


def test_low_overlap_is_a_false_positive_and_a_miss() -> None:
    result = match([Labeled("helmet", (5, 0, 15, 10), 0.9)], [HELMET])  # IoU 1/3
    assert counts_of(result) == Counts(tp=0, fp=1, fn=1)


def test_duplicate_box_counts_once_and_the_confident_one_wins() -> None:
    weak = Labeled("helmet", (0, 0, 10, 10), 0.3)
    strong = Labeled("helmet", (0, 0, 10, 9), 0.9)
    result = match([weak, strong], [HELMET])
    assert counts_of(result) == Counts(tp=1, fp=1, fn=0)
    assert result.true_positives[0][0] is strong


def test_wrong_class_at_the_right_place_is_a_confusion() -> None:
    result = match([Labeled("head", (0, 0, 10, 10), 0.8)], [HELMET])
    assert counts_of(result) == Counts(tp=0, fp=1, fn=1)
    assert len(result.class_confusions) == 1


def test_counts_metrics() -> None:
    c = Counts(tp=8, fp=2, fn=8)
    assert c.precision == pytest.approx(0.8)
    assert c.recall == pytest.approx(0.5)
    assert c.f1 == pytest.approx(2 * 0.8 * 0.5 / 1.3)
    assert Counts().f1 == 0.0  # nothing at all: no division by zero


def test_size_buckets_follow_coco() -> None:
    assert size_bucket((0, 0, 20, 20)) == "small"  # 400 px < 32^2
    assert size_bucket((0, 0, 50, 50)) == "medium"
    assert size_bucket((0, 0, 100, 100)) == "large"


def test_read_yolo_labels_to_pixel_boxes() -> None:
    truths = read_yolo_labels("1 0.5 0.5 0.2 0.4\n", ["helmet", "head"], 100, 50)
    assert truths[0].class_name == "head"
    assert truths[0].box == pytest.approx((40, 15, 60, 35))
