"""AI assistance: box suggestions from the detector, click-to-outline with the segmenter.

The routes are plain `def` (not `async def`): FastAPI then runs them in a thread pool, so a
slow model call (hundreds of ms on CPU) never blocks other requests.
"""

import io
import logging
import time

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException, Response, status
from PIL import Image as PilImage
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_image_or_404
from app.ml.manager import ModelManager, ModelSlot, get_models
from app.ml.postprocess import clip_box, map_to_project_classes
from app.models import Image
from app.schemas import (
    AiStatus,
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
        segmenter=_slot_status(models.segmenter),  # type: ignore[arg-type]
    )


@router.post("/images/{image_id}/suggest", response_model=SuggestResult)
def suggest_boxes(
    image_id: int,
    body: SuggestRequest,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    models: ModelManager = Depends(get_models),
) -> SuggestResult:
    """Pre-label an image: the detector's boxes, renamed to the project's class names."""
    detector = models.detector.require()
    image = get_image_or_404(db, image_id)
    pil_image = _image_loader(storage, image)

    start = time.perf_counter()
    detections = detector.detect(pil_image, conf=body.conf)
    latency = _ms_since(start)

    kept, ignored = map_to_project_classes(detections, image.project.classes)
    suggestions = []
    for d in kept:
        box = clip_box(d.bbox, image.width, image.height)
        if box is not None:
            suggestions.append(Suggestion(class_name=d.class_name, bbox=box, score=d.score))
    logger.info(
        "suggest image=%s model=%s conf=%.2f boxes=%d ignored=%d latency=%.0fms",
        image_id, detector.name, body.conf, len(suggestions), sum(ignored.values()), latency,
    )  # fmt: skip
    return SuggestResult(
        model_version_id=models.detector.model_version_id,
        latency_ms=latency,
        suggestions=suggestions,
        ignored=dict(ignored),
    )


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
