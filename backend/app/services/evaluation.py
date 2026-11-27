"""Matching predictions (or human labels) to ground truth: the basis of all evaluation scripts.

A prediction is a TRUE POSITIVE if it overlaps a not-yet-matched ground-truth box of the same
class with IoU >= threshold (0.5, as in mAP@0.5). Predictions are matched in order of
confidence, so a confident correct box is never "stolen" by a weaker duplicate.
Leftover predictions are FALSE POSITIVES (false alarms); leftover ground truth boxes are
FALSE NEGATIVES (missed objects).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.services.geometry import Box, iou

# COCO size categories (object area in pixels of the original image).
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96


@dataclass(frozen=True)
class Labeled:
    class_name: str
    box: Box
    score: float = 1.0


@dataclass
class MatchResult:
    true_positives: list[tuple[Labeled, Labeled]] = field(default_factory=list)  # (pred, truth)
    false_positives: list[Labeled] = field(default_factory=list)
    false_negatives: list[Labeled] = field(default_factory=list)
    # Correct place, wrong class: a false positive AND a false negative at the same time.
    class_confusions: list[tuple[Labeled, Labeled]] = field(default_factory=list)


def size_bucket(box: Box) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area < SMALL_AREA:
        return "small"
    return "medium" if area < MEDIUM_AREA else "large"


def match(
    predictions: Sequence[Labeled], truths: Sequence[Labeled], iou_threshold: float = 0.5
) -> MatchResult:
    result = MatchResult()
    unmatched = list(truths)
    for pred in sorted(predictions, key=lambda p: p.score, reverse=True):
        best, best_iou = None, iou_threshold
        for truth in unmatched:
            if truth.class_name != pred.class_name:
                continue
            overlap = iou(pred.box, truth.box)
            if overlap >= best_iou:
                best, best_iou = truth, overlap
        if best is None:
            result.false_positives.append(pred)
        else:
            unmatched.remove(best)
            result.true_positives.append((pred, best))
    result.false_negatives = unmatched
    # Which misses are explained by a box at the right place with the wrong class?
    for pred in result.false_positives:
        for truth in result.false_negatives:
            if truth.class_name != pred.class_name and iou(pred.box, truth.box) >= iou_threshold:
                result.class_confusions.append((pred, truth))
                break
    return result


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp, self.fp, self.fn = self.tp + other.tp, self.fp + other.fp, self.fn + other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def counts_of(result: MatchResult) -> Counts:
    return Counts(
        len(result.true_positives), len(result.false_positives), len(result.false_negatives)
    )


def read_yolo_labels(text: str, classes: list[str], width: int, height: int) -> list[Labeled]:
    """Ground truth from a YOLO .txt file (normalized center format) in pixel VOC boxes."""
    truths = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        x_c, y_c, bw, bh = (
            float(cx) * width,
            float(cy) * height,
            float(w) * width,
            float(h) * height,
        )
        box = (x_c - bw / 2, y_c - bh / 2, x_c + bw / 2, y_c + bh / 2)
        truths.append(Labeled(classes[int(cls)], box))
    return truths


def average_precision(
    predictions: Mapping[str, Sequence[Labeled]],
    truths: Mapping[str, Sequence[Labeled]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """AP per class over a whole dataset (keys = image ids), e.g. AP50 with the default.

    Every prediction is matched as in `match`; sorting all of a class's predictions by
    confidence gives a precision-recall curve, and AP is the area under it with all-point
    interpolation (precision at recall r = the best precision at any recall >= r), as in
    PASCAL VOC 2010+. Predictions should come from a LOW confidence threshold, otherwise the
    curve is cut short. Classes without ground truth are left out (AP is undefined there).
    """
    hits: dict[str, list[tuple[float, bool]]] = {}
    n_truths: dict[str, int] = {}
    for image_id, image_truths in truths.items():
        for truth in image_truths:
            n_truths[truth.class_name] = n_truths.get(truth.class_name, 0) + 1
        result = match(predictions.get(image_id, []), image_truths, iou_threshold)
        for pred, _ in result.true_positives:
            hits.setdefault(pred.class_name, []).append((pred.score, True))
        for pred in result.false_positives:
            hits.setdefault(pred.class_name, []).append((pred.score, False))

    ap: dict[str, float] = {}
    for class_name, total in n_truths.items():
        ranked = sorted(hits.get(class_name, []), key=lambda h: h[0], reverse=True)
        precisions, recalls, tp = [], [], 0
        for rank, (_, is_tp) in enumerate(ranked, start=1):
            tp += is_tp
            precisions.append(tp / rank)
            recalls.append(tp / total)
        area, previous_recall = 0.0, 0.0
        for i, recall in enumerate(recalls):
            area += (recall - previous_recall) * max(precisions[i:])
            previous_recall = recall
        ap[class_name] = area
    return ap
