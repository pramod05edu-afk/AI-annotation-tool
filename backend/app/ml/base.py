"""What the rest of the app needs from an AI model, independent of the library behind it.

The API only talks to these interfaces (Protocols). The real implementations (YOLO via
Ultralytics/OpenVINO, MobileSAM) live in yolo.py and sam.py; the tests use small fakes.
Swapping YOLO for another detector later means writing one new class, nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

Box = tuple[float, float, float, float]  # x_min, y_min, x_max, y_max in image pixels
Point = tuple[float, float]


@dataclass(frozen=True)
class Detection:
    class_name: str  # the MODEL's class name (mapped to project classes by the API)
    bbox: Box
    score: float


@dataclass(frozen=True)
class Segmentation:
    polygon: list[Point]
    bbox: Box
    score: float


class Detector(Protocol):
    name: str
    classes: list[str]

    def detect(self, image: Image.Image, conf: float) -> list[Detection]:
        """All objects with a confidence of at least `conf`."""
        ...


class Segmenter(Protocol):
    name: str

    def prepare(self, image_key: str, load_image: Callable[[], Image.Image]) -> None:
        """Encode the image now (slow), so the user's first click is fast."""
        ...

    def segment(
        self,
        image_key: str,
        load_image: Callable[[], Image.Image],
        points: Sequence[Point],
        labels: Sequence[int],
    ) -> Segmentation | None:
        """Outline the ONE object indicated by the clicks (label 1 = object, 0 = background).

        `image_key` identifies the image: the expensive image encoding is cached, so further
        clicks on the same image are fast; `load_image` is only called on a cache miss.
        Returns None when nothing usable was found.
        """
        ...
