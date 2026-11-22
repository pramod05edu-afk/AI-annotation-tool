"""Object detector: a YOLO model loaded with Ultralytics.

Works with PyTorch weights (`best.pt`) and with an exported OpenVINO folder
(`..._openvino_model/`, much faster on Intel CPUs): Ultralytics picks the runtime from the path.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics.models import YOLO

from app.ml.base import Detection


class YoloDetector:
    def __init__(self, weights: Path, imgsz: int = 640) -> None:
        # "yolo11n_helmet_openvino_model" (folder) or "yolo11n_helmet.pt" -> "yolo11n_helmet"
        self.name = weights.name.removesuffix("_openvino_model").removesuffix(".pt")
        self.imgsz = imgsz  # must match the training size (the OpenVINO export has a fixed size)
        self._model = YOLO(str(weights), task="detect")
        self.classes = [str(n) for n in self._model.names.values()]
        # Ultralytics models are not safe to call from several threads at once; FastAPI runs
        # requests in a thread pool, so one detection runs at a time.
        self._lock = threading.Lock()

    def detect(self, image: Image.Image, conf: float) -> list[Detection]:
        with self._lock:
            # A PIL image is read as RGB (a numpy array would be assumed to be BGR).
            # Ultralytics' return type is a loose union; for one image it is always one Results.
            results: Any = self._model.predict(
                image, conf=conf, imgsz=self.imgsz, device="cpu", verbose=False
            )
        result = results[0]
        boxes, names = result.boxes, result.names
        return [
            Detection(
                class_name=str(names[int(cls)]),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                score=round(float(score), 3),
            )
            for (x1, y1, x2, y2), cls, score in zip(
                boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True
            )
        ]

    def warm_up(self) -> None:
        """The first prediction is slow (graph compilation); do it at startup, not for a user."""
        self.detect(Image.new("RGB", (self.imgsz, self.imgsz)), conf=0.5)
