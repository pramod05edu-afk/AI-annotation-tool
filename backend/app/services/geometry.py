"""Bounding-box formats and overlap — used by dataset preparation, export and evaluation.

Three box formats appear in this project (all in pixels unless stated otherwise):

    VOC  (Pascal VOC XML, also our internal format):  (x_min, y_min, x_max, y_max)
    COCO (COCO JSON):                                  (x_min, y_min, width, height)
    YOLO (YOLO .txt), normalized to 0..1 by image size: (x_center, y_center, width, height)

Picture a 100x50 image with a box from (10, 20) to (30, 40):

    VOC  -> (10, 20, 30, 40)
    COCO -> (10, 20, 20, 20)
    YOLO -> (0.2, 0.6, 0.2, 0.4)     # center (20, 30) / (100, 50), size (20, 20) / (100, 50)

Tests: cd backend && uv run pytest tests/test_geometry.py -v
"""

Box = tuple[float, float, float, float]


def voc_to_coco(box: Box) -> Box:
    """(x_min, y_min, x_max, y_max) -> (x_min, y_min, width, height)."""
    x_min, y_min, x_max, y_max = box
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def coco_to_voc(box: Box) -> Box:
    """(x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)."""
    x_min, y_min, width, height = box
    return (x_min, y_min, x_min + width, y_min + height)


def voc_to_yolo(box: Box, img_w: float, img_h: float) -> Box:
    """(x_min, y_min, x_max, y_max) in pixels -> normalized (x_center, y_center, width, height)."""
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    # Dividing by the image size makes the values independent of resolution (0..1),
    # so labels stay valid when YOLO resizes images for training.
    return (x_center / img_w, y_center / img_h, (x_max - x_min) / img_w, (y_max - y_min) / img_h)


def yolo_to_voc(box: Box, img_w: float, img_h: float) -> Box:
    """Normalized (x_center, y_center, width, height) -> (x_min, y_min, x_max, y_max) in pixels."""
    x_center, y_center, width, height = box
    # Back to pixels first, then step half the size left/up and right/down from the center.
    x_center, width = x_center * img_w, width * img_w
    y_center, height = y_center * img_h, height * img_h
    return (
        x_center - width / 2,
        y_center - height / 2,
        x_center + width / 2,
        y_center + height / 2,
    )


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection over Union of two VOC boxes, between 0.0 (no overlap) and 1.0 (identical).

    IoU = area of overlap / area of union, where union = area_a + area_b - overlap.
    The overlap rectangle runs from the larger of the two minimums to the smaller
    of the two maximums; a negative width or height means there is no overlap.
    """
    overlap_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    overlap_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0  # disjoint, or only touching at an edge

    overlap = overlap_w * overlap_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0
