Box = tuple[float, float, float, float]


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection over Union of two (x_min, y_min, x_max, y_max) boxes."""
    overlap_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    overlap_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0  # disjoint, or only touching at an edge

    overlap = overlap_w * overlap_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0
