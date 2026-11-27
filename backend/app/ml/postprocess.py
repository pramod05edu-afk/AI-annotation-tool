"""Pure functions that turn raw model output into annotations (no model needed to test them)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import cv2
import numpy as np
import numpy.typing as npt

from app.ml.base import Box, Detection, Point
from app.services.geometry import iou

# Polygon simplification: drop points that deviate less than this fraction of the outline
# length. 0.2 % keeps the shape visually identical but turns ~1000 contour pixels into ~30 points.
SIMPLIFY_FRACTION = 0.002


def clip_box(box: Box, width: float, height: float) -> Box | None:
    """Keep a box inside the image; None if nothing of it is left."""
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))


def prompt_text(label: str) -> str:
    """A label as a text prompt: "traffic_light" / "traffic-light" -> "traffic light".

    Text encoders were trained on captions, so plain words work better than identifiers.
    """
    words = label.replace("_", " ").replace("-", " ").split()
    return " ".join(words) or label


def split_classes(
    project_classes: Sequence[str], specialist_classes: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Which project labels the fine-tuned specialist knows, and which the general model needs.

    Case-insensitive: a project label "Helmet" is the specialist's "helmet". Order is kept.
    """
    known = {name.lower() for name in specialist_classes}
    for_specialist = [c for c in project_classes if c.lower() in known]
    for_general = [c for c in project_classes if c.lower() not in known]
    return for_specialist, for_general


def drop_overlapping(
    detections: Iterable[Detection], existing: Iterable[Box], overlap_limit: float = 0.5
) -> list[Detection]:
    """Detections that do not duplicate an existing box (IoU below the limit)."""
    boxes = list(existing)
    return [d for d in detections if all(iou(d.bbox, b) < overlap_limit for b in boxes)]


def map_to_project_classes(
    detections: Iterable[Detection], project_classes: list[str]
) -> tuple[list[Detection], Counter[str]]:
    """Rename model classes to the project's spelling (case-insensitive match).

    Detections of classes the project does not use are dropped and counted, so the UI can
    say e.g. "3 'person' boxes ignored" instead of silently hiding them.
    """
    by_lower = {name.lower(): name for name in project_classes}
    kept: list[Detection] = []
    ignored: Counter[str] = Counter()
    for d in detections:
        target = by_lower.get(d.class_name.lower())
        if target is None:
            ignored[d.class_name] += 1
        else:
            kept.append(Detection(class_name=target, bbox=d.bbox, score=d.score))
    return kept, ignored


def mask_to_polygon(mask: npt.NDArray[np.bool_]) -> list[Point] | None:
    """Outline of the largest region of a binary mask, simplified to a few dozen points."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:  # a few stray pixels, not an object
        return None
    epsilon = max(1.0, SIMPLIFY_FRACTION * cv2.arcLength(largest, closed=True))
    simplified = cv2.approxPolyDP(largest, epsilon, closed=True).reshape(-1, 2)
    if len(simplified) < 3:
        return None
    return [(float(x), float(y)) for x, y in simplified]


def polygon_bbox(points: list[Point]) -> Box:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def pick_box_at_point(
    detections: Iterable[Detection],
    point: Point,
    exclude: Iterable[Box] = (),
    overlap_limit: float = 0.5,
) -> Detection | None:
    """Smart click: the most confident detection whose box contains the clicked point.

    `detections` should come from a LOW confidence threshold: the click itself is the
    evidence that an object is there, so a weak candidate the auto-suggest hid is fine now.
    Boxes that overlap an existing label (IoU >= overlap_limit) are skipped, so clicking a
    second time near a labelled object finds the NEXT object instead of a duplicate.
    """
    x, y = point
    existing = list(exclude)
    candidates = [
        d
        for d in detections
        if d.bbox[0] <= x <= d.bbox[2]
        and d.bbox[1] <= y <= d.bbox[3]
        and all(iou(d.bbox, box) < overlap_limit for box in existing)
    ]
    return max(candidates, key=lambda d: d.score, default=None)
