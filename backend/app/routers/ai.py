"""AI assistance: box suggestions from the detectors, click-to-outline with the segmenter.

Two detectors work together: the fine-tuned specialist for the labels it was trained on, and
a general open-vocabulary model (YOLOE) for every other label, so any project gets help.

The routes are plain `def` (not `async def`): FastAPI then runs them in a thread pool, so a
slow model call (hundreds of ms on CPU) never blocks other requests.
"""

import io
import logging
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException, Response, status
from PIL import Image as PilImage
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_image_or_404
from app.ml.base import Detection
from app.ml.manager import ModelManager, ModelSlot, ModelState, get_models
from app.ml.postprocess import (
    clip_box,
    drop_overlapping,
    map_to_project_classes,
    pick_box_at_point,
    split_classes,
)
from app.models import Image
from app.schemas import (
    AiStatus,
    ClickBoxRequest,
    ClickBoxResult,
    FindSimilarRequest,
    FindSimilarResult,
    ModelSlotStatus,
    SegmentRequest,
    SegmentResult,
    Suggestion,
    SuggestRequest,
    SuggestResult,
)
from app.storage import Storage, get_storage

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ai"])

# Smart click uses ALL candidates, even weak ones: the user's click is the extra evidence.
CLICK_MIN_CONF = 0.01
_CLICK_CACHE_SIZE = 64  # images; one entry is a short list of boxes


@dataclass(frozen=True)
class _Found:
    detection: Detection
    model: str
    model_version_id: int | None


_CacheKey = tuple[int, tuple[str, ...], str | None, str | None]  # image, labels, models


class _DetectionCache:
    """Low-confidence detections per (image, labels, models): the first click on an image runs
    the detectors, further clicks on the same image are answered from memory."""

    def __init__(self, size: int) -> None:
        self._items: OrderedDict[_CacheKey, list[_Found]] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, key: _CacheKey) -> list[_Found] | None:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            return None

    def put(self, key: _CacheKey, value: list[_Found]) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)  # forget the least recently used image


_click_cache = _DetectionCache(_CLICK_CACHE_SIZE)


def _slot_status(slot: ModelSlot[object]) -> ModelSlotStatus:
    return ModelSlotStatus(
        state=slot.state.value,
        name=slot.name,
        model_version_id=slot.model_version_id,
        error=slot.error,
    )


def _image_loader(storage: Storage, image: Image) -> PilImage.Image:
    try:
        data = storage.get(image.storage_key)
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Image storage unavailable"
        ) from error
    return PilImage.open(io.BytesIO(data)).convert("RGB")


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


@router.get("/ai/status", response_model=AiStatus)
def ai_status(models: ModelManager = Depends(get_models)) -> AiStatus:
    """Which models are loaded; the UI enables the AI buttons from this."""
    return AiStatus(
        detector=_slot_status(models.detector),  # type: ignore[arg-type]
        general=_slot_status(models.general),  # type: ignore[arg-type]
        segmenter=_slot_status(models.segmenter),  # type: ignore[arg-type]
    )


def _detect_for_project(
    models: ModelManager,
    classes: list[str],
    load_image: Callable[[], PilImage.Image],
    conf: float,
) -> tuple[list[_Found], Counter[str], list[str]]:
    """Detect a project's labels: the fine-tuned specialist for the labels it was trained on,
    the general open-vocabulary model for all others.

    Returns the detections (each with the model that found it), the specialist's classes the
    project does not use (with counts), and the names of the models that ran.
    """
    specialist = models.detector.model if models.detector.state == ModelState.READY else None
    if specialist is not None:
        for_specialist, for_general = split_classes(classes, specialist.classes)
    else:  # specialist missing or still loading: the general model covers every label
        for_specialist, for_general = [], list(classes)
    # Fail fast (503) before the slow part if the general model is needed but not ready.
    general = models.general.require() if for_general else None

    image = load_image()
    found: list[_Found] = []
    ignored: Counter[str] = Counter()
    used: list[str] = []
    if specialist is not None and for_specialist:
        kept, ignored = map_to_project_classes(specialist.detect(image, conf), for_specialist)
        found += [_Found(d, specialist.name, models.detector.model_version_id) for d in kept]
        used.append(specialist.name)
    if general is not None:
        detections = general.detect_classes(image, for_general, conf)
        found += [_Found(d, general.name, models.general.model_version_id) for d in detections]
        used.append(general.name)
    return found, ignored, used


def _suggestion(found: _Found, image: Image, class_name: str | None = None) -> Suggestion | None:
    box = clip_box(found.detection.bbox, image.width, image.height)
    if box is None:
        return None
    return Suggestion(
        class_name=class_name or found.detection.class_name,
        bbox=box,
        score=found.detection.score,
        model=found.model,
        model_version_id=found.model_version_id,
    )


@router.post("/images/{image_id}/suggest", response_model=SuggestResult)
def suggest_boxes(
    image_id: int,
    body: SuggestRequest,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    models: ModelManager = Depends(get_models),
) -> SuggestResult:
    """Pre-label an image with boxes for the project's own labels."""
    image = get_image_or_404(db, image_id)
    start = time.perf_counter()
    found, ignored, used = _detect_for_project(
        models, image.project.classes, lambda: _image_loader(storage, image), body.conf
    )
    latency = _ms_since(start)

    suggestions = [s for f in found if (s := _suggestion(f, image)) is not None]
    logger.info(
        "suggest image=%s models=%s conf=%.2f boxes=%d ignored=%d latency=%.0fms",
        image_id, "+".join(used), body.conf, len(suggestions), sum(ignored.values()), latency,
    )  # fmt: skip
    return SuggestResult(
        model_version_id=suggestions[0].model_version_id if suggestions else None,
        models=used,
        latency_ms=latency,
        suggestions=suggestions,
        ignored=dict(ignored),
    )


@router.post("/images/{image_id}/click-box", response_model=ClickBoxResult)
def click_box(
    image_id: int,
    body: ClickBoxRequest,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    models: ModelManager = Depends(get_models),
) -> ClickBoxResult:
    """Smart click: the best box (with its label) under the clicked point."""
    image = get_image_or_404(db, image_id)
    classes = image.project.classes
    start = time.perf_counter()
    # The labels and the loaded models decide the result, so they are part of the cache key.
    key = (image.id, tuple(classes), models.detector.name, models.general.name)
    found = _click_cache.get(key)
    if found is None:
        found, _, _ = _detect_for_project(
            models, classes, lambda: _image_loader(storage, image), CLICK_MIN_CONF
        )
        _click_cache.put(key, found)
    by_detection = {id(f.detection): f for f in found}
    best = pick_box_at_point([f.detection for f in found], body.point, body.exclude)
    suggestion = _suggestion(by_detection[id(best)], image) if best is not None else None
    latency = _ms_since(start)
    logger.info(
        "click-box image=%s found=%s latency=%.0fms", image_id, suggestion is not None, latency
    )
    return ClickBoxResult(
        model_version_id=suggestion.model_version_id if suggestion else None,
        latency_ms=latency,
        suggestion=suggestion,
    )


@router.post("/images/{image_id}/find-similar", response_model=FindSimilarResult)
def find_similar(
    image_id: int,
    body: FindSimilarRequest,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    models: ModelManager = Depends(get_models),
) -> FindSimilarResult:
    """Find more objects like the example boxes, for labels that no model knows by name."""
    general = models.general.require()
    image = get_image_or_404(db, image_id)
    if body.class_name not in image.project.classes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"'{body.class_name}' is not a label of this project",
        )
    pil_image = _image_loader(storage, image)
    start = time.perf_counter()
    detections = general.find_similar(pil_image, body.examples, conf=body.conf)
    latency = _ms_since(start)

    # The examples and other labelled objects are found too: suggest only NEW objects.
    fresh = drop_overlapping(detections, [*body.exclude, *body.examples])
    version = models.general.model_version_id
    suggestions = [
        s
        for d in fresh
        if (s := _suggestion(_Found(d, general.name, version), image, body.class_name)) is not None
    ]
    logger.info(
        "find-similar image=%s examples=%d found=%d latency=%.0fms",
        image_id, len(body.examples), len(suggestions), latency,
    )  # fmt: skip
    return FindSimilarResult(model_version_id=version, latency_ms=latency, suggestions=suggestions)


@router.post("/images/{image_id}/segment/prepare", status_code=status.HTTP_204_NO_CONTENT)
def prepare_segmentation(
    image_id: int,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    models: ModelManager = Depends(get_models),
) -> Response:
    """Encode the image in advance (the slow part of SAM), called when an image is opened."""
    segmenter = models.segmenter.require()
    image = get_image_or_404(db, image_id)
    start = time.perf_counter()
    segmenter.prepare(str(image.id), lambda: _image_loader(storage, image))
    logger.info("segment/prepare image=%s latency=%.0fms", image_id, _ms_since(start))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/images/{image_id}/segment", response_model=SegmentResult)
def segment_object(
    image_id: int,
    body: SegmentRequest,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    models: ModelManager = Depends(get_models),
) -> SegmentResult:
    """Outline the object under the clicks (label 1 = object, 0 = not the object)."""
    segmenter = models.segmenter.require()
    image = get_image_or_404(db, image_id)
    points = [
        (min(max(x, 0.0), image.width - 1.0), min(max(y, 0.0), image.height - 1.0))
        for x, y in body.points
    ]

    start = time.perf_counter()
    result = segmenter.segment(
        str(image.id), lambda: _image_loader(storage, image), points, body.labels
    )
    latency = _ms_since(start)
    logger.info("segment image=%s points=%d latency=%.0fms", image_id, len(points), latency)
    if result is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "No object found at these points - try again"
        )
    return SegmentResult(
        model_version_id=models.segmenter.model_version_id,
        latency_ms=latency,
        points=[(round(x, 1), round(y, 1)) for x, y in result.polygon],
        bbox=result.bbox,
        score=result.score,
    )
