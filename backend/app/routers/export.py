"""Download a project as a COCO or YOLO dataset (zip with images)."""

import logging
import re
import tempfile
from collections.abc import Iterator
from enum import StrEnum
from typing import IO

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.deps import get_project_or_404
from app.models import Image, ImageStatus
from app.services.export import ExportItem, write_zip
from app.storage import Storage, get_storage

router = APIRouter(tags=["export"])
logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024
SPOOL_IN_MEMORY_BYTES = 50 * 1024 * 1024  # bigger exports are written to a temporary file


class ExportFormat(StrEnum):
    COCO = "coco"
    YOLO = "yolo"


class ExportScope(StrEnum):
    ALL = "all"  # every image (unlabeled ones become background images)
    DONE = "done"  # only images marked as done: the reviewed, final labels


def _chunks(file: IO[bytes]) -> Iterator[bytes]:
    try:
        while chunk := file.read(CHUNK_BYTES):
            yield chunk
    finally:
        file.close()


@router.get(
    "/projects/{project_id}/export",
    response_class=StreamingResponse,
    responses={200: {"content": {"application/zip": {}}}},
)
def export_project(
    project_id: int,
    export_format: ExportFormat = Query(default=ExportFormat.COCO, alias="format"),
    scope: ExportScope = Query(default=ExportScope.ALL),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> StreamingResponse:
    project = get_project_or_404(db, project_id)
    query = (
        select(Image)
        .where(Image.project_id == project_id)
        .options(selectinload(Image.annotations))  # all annotations in one extra query
        .order_by(Image.id)
    )
    if scope == ExportScope.DONE:
        query = query.where(Image.status == ImageStatus.DONE)
    images = list(db.scalars(query))
    if not images:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No images to export for this selection")
    items = [
        ExportItem(image=i, annotations=sorted(i.annotations, key=lambda a: a.id)) for i in images
    ]

    # Build the zip into a temporary file: memory stays small even for big projects.
    file = tempfile.SpooledTemporaryFile(max_size=SPOOL_IN_MEMORY_BYTES)  # noqa: SIM115
    try:
        write_zip(file, project, items, export_format.value, lambda i: storage.get(i.storage_key))
    except (BotoCoreError, ClientError) as error:
        file.close()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Image storage unavailable"
        ) from error
    size = file.tell()
    file.seek(0)

    slug = re.sub(r"[^a-z0-9]+", "-", project.name.lower()).strip("-") or "project"
    filename = f"{slug}_{export_format.value}_{scope.value}.zip"
    logger.info(
        "export project=%s format=%s images=%d bytes=%d",
        project_id,
        export_format,
        len(items),
        size,
    )
    return StreamingResponse(
        _chunks(file),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(size),
        },
    )
