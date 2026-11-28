"""Images: upload (validated), list, stream the file, change status, delete."""

import io
import logging
import uuid
from pathlib import PurePath

import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image as PILImage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import get_image_or_404, get_project_or_404
from app.models import Annotation, Image, ImageStatus
from app.schemas import ImageRead, ImageStatusUpdate, UploadRejected, UploadResult
from app.storage import ObjectNotFoundError, Storage, get_storage

router = APIRouter(tags=["images"])
logger = logging.getLogger(__name__)

# Decided by the decoded file content, NOT by the extension or the browser's content type.
ALLOWED_FORMATS = {
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    "WEBP": ("image/webp", "webp"),
}
# Formats browsers cannot display (TIFF, e.g. from microscopes or scanners; BMP): accepted,
# but stored as PNG so the annotator, the models and the export all see a normal image.
CONVERTED_FORMATS = {"TIFF", "BMP"}
MAX_FILES_PER_REQUEST = 200


class ImageValidationError(Exception):
    """The uploaded bytes are not an acceptable image."""


def _format_of(data: bytes) -> str | None:
    """The image format PIL recognises in these bytes (None if it is no image at all)."""
    try:
        with PILImage.open(io.BytesIO(data)) as image:
            return image.format
    except (PILImage.DecompressionBombError, OSError, SyntaxError, ValueError):
        return None


def inspect_image(data: bytes) -> tuple[str, str, int, int]:
    """Return (content_type, extension, width, height) or raise ImageValidationError."""
    try:
        with PILImage.open(io.BytesIO(data)) as probe:
            probe.verify()  # detects truncated/corrupt files
        with PILImage.open(io.BytesIO(data)) as image:  # verify() closes the image, so reopen
            image_format, (width, height) = image.format or "", image.size
    except (PILImage.DecompressionBombError, OSError, SyntaxError, ValueError) as error:
        raise ImageValidationError("not a readable image file") from error
    if image_format not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"format {image_format or 'unknown'} not allowed (PNG, JPEG, WEBP, TIFF, BMP)"
        )
    content_type, extension = ALLOWED_FORMATS[image_format]
    return content_type, extension, width, height


def to_png(data: bytes) -> bytes:
    """A TIFF / BMP as PNG (first page only). High bit-depth greyscale (16-bit, float), common in
    microscopy and medical imaging, is stretched to 8 bit, so faint images stay visible."""
    try:
        with PILImage.open(io.BytesIO(data)) as source:
            source.seek(0)
            image: PILImage.Image = source
            if source.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                pixels = np.asarray(source, dtype=np.float64)
                low, high = float(pixels.min()), float(pixels.max())
                scaled = (pixels - low) / (high - low) * 255 if high > low else pixels * 0
                image = PILImage.fromarray(scaled.astype(np.uint8))  # 2-D uint8 = greyscale
            elif source.mode not in ("RGB", "RGBA", "L", "LA"):
                image = source.convert("RGB")
            out = io.BytesIO()
            image.save(out, format="PNG")
    except (OSError, SyntaxError, ValueError) as error:
        raise ImageValidationError("not a readable image file") from error
    return out.getvalue()


def _with_counts(db: Session, images: list[Image]) -> list[ImageRead]:
    ids = [i.id for i in images]
    counts: dict[int, int] = {}
    if ids:
        rows = db.execute(
            select(Annotation.image_id, func.count())
            .where(Annotation.image_id.in_(ids))
            .group_by(Annotation.image_id)
        )
        counts = {image_id: n for image_id, n in rows}
    return [
        ImageRead.model_validate(i).model_copy(update={"annotation_count": counts.get(i.id, 0)})
        for i in images
    ]


@router.post(
    "/projects/{project_id}/images",
    response_model=UploadResult,
    status_code=status.HTTP_201_CREATED,
)
def upload_images(
    project_id: int,
    files: list[UploadFile] = File(..., description="One or more image files"),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> UploadResult:
    get_project_or_404(db, project_id)
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"At most {MAX_FILES_PER_REQUEST} files per request"
        )
    max_bytes = get_settings().max_upload_mb * 1024 * 1024

    created: list[Image] = []
    stored_keys: list[str] = []
    rejected: list[UploadRejected] = []
    for upload in files:
        filename = PurePath(upload.filename or "unnamed").name[:255]  # drop any folder path
        data = upload.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            rejected.append(
                UploadRejected(
                    filename=filename, reason=f"larger than {get_settings().max_upload_mb} MB"
                )
            )
            continue
        try:
            if _format_of(data) in CONVERTED_FORMATS:
                data = to_png(data)
                filename = f"{PurePath(filename).stem}.png"[:255]
            content_type, extension, width, height = inspect_image(data)
        except ImageValidationError as error:
            rejected.append(UploadRejected(filename=filename, reason=str(error)))
            continue

        key = f"projects/{project_id}/images/{uuid.uuid4().hex}.{extension}"
        try:
            storage.put(key, data, content_type)
        except (BotoCoreError, ClientError):
            logger.exception("Storage upload failed for %s", filename)
            rejected.append(UploadRejected(filename=filename, reason="storage unavailable"))
            continue
        stored_keys.append(key)
        image = Image(
            project_id=project_id,
            filename=filename,
            storage_key=key,
            content_type=content_type,
            width=width,
            height=height,
        )
        db.add(image)
        created.append(image)

    try:
        db.commit()
    except Exception:
        db.rollback()
        for key in stored_keys:  # don't leave orphaned files behind
            storage.delete(key)
        raise
    return UploadResult(uploaded=_with_counts(db, created), rejected=rejected)


@router.get("/projects/{project_id}/images", response_model=list[ImageRead])
def list_images(
    project_id: int,
    status_filter: ImageStatus | None = Query(default=None, alias="status"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[ImageRead]:
    get_project_or_404(db, project_id)
    query = select(Image).where(Image.project_id == project_id)
    if status_filter is not None:
        query = query.where(Image.status == status_filter)
    images = list(db.scalars(query.order_by(Image.id).offset(offset).limit(limit)))
    return _with_counts(db, images)


@router.get("/images/{image_id}", response_model=ImageRead)
def get_image(image_id: int, db: Session = Depends(get_db)) -> ImageRead:
    return _with_counts(db, [get_image_or_404(db, image_id)])[0]


@router.get(
    "/images/{image_id}/file",
    response_class=StreamingResponse,
    responses={200: {"content": {"image/png": {}, "image/jpeg": {}, "image/webp": {}}}},
)
def get_image_file(
    image_id: int,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> StreamingResponse:
    image = get_image_or_404(db, image_id)
    try:
        chunks, size = storage.stream(image.storage_key)
    except ObjectNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image file missing in storage") from error
    headers = {"Cache-Control": "private, max-age=3600"}  # file never changes for this id
    if size is not None:
        headers["Content-Length"] = str(size)
    return StreamingResponse(chunks, media_type=image.content_type, headers=headers)


@router.patch("/images/{image_id}", response_model=ImageRead)
def update_image_status(
    image_id: int, body: ImageStatusUpdate, db: Session = Depends(get_db)
) -> ImageRead:
    image = get_image_or_404(db, image_id)
    image.status = body.status
    db.commit()
    return _with_counts(db, [image])[0]


@router.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(
    image_id: int,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> Response:
    image = get_image_or_404(db, image_id)
    key = image.storage_key
    db.delete(image)
    db.commit()
    try:
        storage.delete(key)
    except (BotoCoreError, ClientError):
        logger.exception("Could not delete stored file %s", key)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
