Box = tuple[float, float, float, float]


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection over Union of two (x_min, y_min, x_max, y_max) boxes."""
    # Step 1: give the coordinates readable names
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Step 2: the overlap rectangle's edges
    left = max(ax1, bx1)  # the larger of the two left edges
    right = min(ax2, bx2)  # the smaller of the two right edges
    top = max(ay1, by1)  # the larger of the two top edges
    bottom = min(ay2, by2)  # the smaller of the two bottom edges

    # Step 3: no overlap → IoU is 0
    if left >= right or top >= bottom:
        return 0.0

    # Step 4: areas
    intersection = (right - left) * (bottom - top)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    # Step 5: divide, without dividing by zero
    return intersection / union if union > 0 else 0.0
