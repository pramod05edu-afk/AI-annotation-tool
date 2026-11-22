"""Click-to-segment with MobileSAM (Segment Anything, with a ~60x smaller image encoder).

SAM works in two steps:
  1. image encoder: image -> embedding. Expensive (most of the time), once per image.
  2. prompt encoder + mask decoder: embedding + clicks -> mask. Cheap (tens of ms).
So the embedding of the current image is cached: the first click on an image is slower,
every further click (refining with + / - points) is fast.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics.models.sam import Predictor as SamPredictor

from app.ml.base import Point, Segmentation
from app.ml.postprocess import mask_to_polygon, polygon_bbox


class MobileSamSegmenter:
    def __init__(self, weights: Path) -> None:
        self.name = weights.stem  # "mobile_sam"
        self._predictor = SamPredictor(
            overrides={
                "model": str(weights),  # downloaded automatically if missing
                "task": "segment",
                "mode": "predict",
                "imgsz": 1024,  # SAM's fixed input size
                "conf": 0.0,  # return the mask even if SAM is unsure; the user judges it
                "device": "cpu",
                "save": False,
                "verbose": False,
            }
        )
        self._predictor.setup_model(verbose=False)
        self._current_key: str | None = None  # which image's embedding is cached
        self._lock = threading.Lock()

    def _encode(self, image_key: str, load_image: Callable[[], Image.Image]) -> None:
        """Run the image encoder unless this image's embedding is already cached."""
        if image_key == self._current_key:
            return
        rgb = np.asarray(load_image().convert("RGB"))
        self._predictor.set_image(rgb[:, :, ::-1].copy())  # Ultralytics expects BGR
        self._current_key = image_key

    def prepare(self, image_key: str, load_image: Callable[[], Image.Image]) -> None:
        with self._lock:  # a click arriving meanwhile waits instead of encoding twice
            self._encode(image_key, load_image)

    def segment(
        self,
        image_key: str,
        load_image: Callable[[], Image.Image],
        points: Sequence[Point],
        labels: Sequence[int],
    ) -> Segmentation | None:
        with self._lock:
            self._encode(image_key, load_image)
            # One object with several clicks: shape (1, n_points, 2) and labels (1, n_points).
            results = self._predictor(points=[list(points)], labels=[list(labels)])
        masks = results[0].masks
        boxes = results[0].boxes
        if masks is None or len(masks) == 0:
            return None
        mask = masks.data[0].cpu().numpy().astype(bool)
        polygon = mask_to_polygon(mask)
        if polygon is None:
            return None
        score = float(boxes.conf[0]) if boxes is not None and len(boxes) else 0.0
        return Segmentation(polygon=polygon, bbox=polygon_bbox(polygon), score=round(score, 3))

    def warm_up(self) -> None:
        """Run once at startup so the first user click does not pay the start-up cost."""
        blank = Image.new("RGB", (64, 64))
        self.segment("__warm_up__", lambda: blank, [(32.0, 32.0)], [1])
        self._current_key = None
