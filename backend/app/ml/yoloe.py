"""General-purpose detector: YOLOE, an open-vocabulary YOLO (Wang et al., 2025).

A normal detector has one output per class it was trained on. YOLOE instead compares image
regions with TEXT: a text encoder (MobileCLIP2) turns each label name into a vector, and a box
is reported where a region's vector is close to a label's vector. So any project's own labels
("cat", "forklift", "pallet") work without training. It can also take an example BOX instead
of text (visual prompt): "find more objects like this one", for things that have no name a
text model would know.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from ultralytics.models import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from app.ml.base import Box, Detection
from app.ml.postprocess import prompt_text

_EMBEDDING_CACHE_SIZE = 32  # label sets (projects); one entry is a few KB


class YoloeDetector:
    def __init__(self, weights: Path, imgsz: int = 640) -> None:
        self.name = weights.stem  # "yoloe-26s-seg"
        self.imgsz = imgsz
        # Two copies: text prompts change the model's class head (set_classes), visual prompts
        # use a different predictor; keeping them apart avoids one mode disturbing the other.
        self._text_model = YOLOE(str(weights))
        self._visual_model = YOLOE(str(weights))
        # Encoding label names takes ~0.3 s; a project's labels rarely change, so cache them.
        self._embeddings: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        self._active: tuple[str, ...] | None = None
        self._lock = threading.Lock()  # Ultralytics models are not thread-safe

    def _use_classes(self, prompts: tuple[str, ...]) -> None:
        """Point the model at these label names (called with the lock held)."""
        if prompts == self._active:
            return
        embeddings = self._embeddings.get(prompts)
        if embeddings is None:
            embeddings = self._text_model.get_text_pe(list(prompts))  # type: ignore[no-untyped-call]
            self._embeddings[prompts] = embeddings
            while len(self._embeddings) > _EMBEDDING_CACHE_SIZE:
                self._embeddings.popitem(last=False)
        self._embeddings.move_to_end(prompts)
        self._text_model.set_classes(list(prompts), embeddings)
        self._active = prompts

    def detect_classes(
        self, image: Image.Image, classes: Sequence[str], conf: float
    ) -> list[Detection]:
        if not classes:
            return []
        prompts = tuple(prompt_text(name) for name in classes)
        with self._lock:
            self._use_classes(prompts)
            results: Any = self._text_model.predict(
                image, conf=conf, imgsz=self.imgsz, device="cpu", verbose=False
            )
        # Class index i belongs to classes[i]: report the project's spelling, not the prompt.
        return _detections(results[0], lambda index: classes[index])

    def find_similar(
        self, image: Image.Image, examples: Sequence[Box], conf: float
    ) -> list[Detection]:
        if not examples:
            return []
        prompts: dict[str, npt.NDArray[Any] | list[npt.NDArray[Any]]] = {
            "bboxes": np.array(examples, dtype=np.float32),
            "cls": np.zeros(len(examples), dtype=np.int64),  # all examples show ONE kind of object
        }
        with self._lock:
            results: Any = self._visual_model.predict(
                image,
                visual_prompts=prompts,
                predictor=YOLOEVPSegPredictor,
                conf=conf,
                imgsz=self.imgsz,
                device="cpu",
                verbose=False,
            )
        return _detections(results[0], lambda _: "object")  # the API assigns the user's label

    def warm_up(self) -> None:
        """Load the text encoder and run both modes once, at startup instead of for a user."""
        blank = Image.new("RGB", (self.imgsz, self.imgsz))
        self.detect_classes(blank, ["object"], conf=0.5)
        self.find_similar(blank, [(100.0, 100.0, 200.0, 200.0)], conf=0.5)


def _detections(result: Any, name_of: Any) -> list[Detection]:
    boxes = result.boxes
    return [
        Detection(
            class_name=name_of(int(cls)),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            score=round(float(score), 3),
        )
        for (x1, y1, x2, y2), cls, score in zip(
            boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True
        )
    ]
