"""Annotations of one image. The editor saves the image's complete set at once (autosave),
so the API is "get all" + "replace all" — simple and safe against partial updates."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_image_or_404
from app.models import Annotation, AnnotationType, Image, ImageStatus, ModelVersion
from app.schemas import AnnotationIn, AnnotationRead, BoxGeometry

router = APIRouter(prefix="/images/{image_id}/annotations", tags=["annotations"])

# Allow tiny rounding errors from the canvas (e.g. 416.0001 on a 416-px image).
EDGE_TOLERANCE_PX = 1.0


def to_read(annotation: Annotation) -> AnnotationRead:
    geometry: dict[str, Any] = {"type": annotation.type.value, **annotation.geometry}
    if annotation.type == AnnotationType.POLYGON:
        geometry.pop("bbox", None)
    return AnnotationRead.model_validate(
        {
            "id": annotation.id,
            "image_id": annotation.image_id,
            "class_name": annotation.class_name,
            "geometry": geometry,
            "source": annotation.source,
            "score": annotation.score,
            "model_version_id": annotation.model_version_id,
            "created_at": annotation.created_at,
        }
    )


def _check_and_store(item: AnnotationIn, index: int, image: Image) -> dict[str, Any]:
    """Validate coordinates against the image size; return the JSON to store."""
    w, h = image.width + EDGE_TOLERANCE_PX, image.height + EDGE_TOLERANCE_PX
    lo = -EDGE_TOLERANCE_PX
    if isinstance(item.geometry, BoxGeometry):
        x_min, y_min, x_max, y_max = item.geometry.bbox
        if x_min < lo or y_min < lo or x_max > w or y_max > h:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Annotation {index}: box outside the {image.width}x{image.height} image",
            )
        return {"bbox": list(item.geometry.bbox)}

    points = item.geometry.points
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    if min(xs) < lo or min(ys) < lo or max(xs) > w or max(ys) > h:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Annotation {index}: polygon outside the {image.width}x{image.height} image",
        )
    # Keep the bounding box too: export and evaluation need it constantly.
    return {"points": [list(p) for p in points], "bbox": [min(xs), min(ys), max(xs), max(ys)]}


@router.get("", response_model=list[AnnotationRead])
def list_annotations(image_id: int, db: Session = Depends(get_db)) -> list[AnnotationRead]:
    get_image_or_404(db, image_id)
    rows = db.scalars(
        select(Annotation).where(Annotation.image_id == image_id).order_by(Annotation.id)
    )
    return [to_read(a) for a in rows]


@router.put("", response_model=list[AnnotationRead])
def replace_annotations(
    image_id: int, items: list[AnnotationIn], db: Session = Depends(get_db)
) -> list[AnnotationRead]:
    image = get_image_or_404(db, image_id)
    allowed = set(image.project.classes)

    referenced_models = {i.model_version_id for i in items if i.model_version_id is not None}
    if referenced_models:
        found = set(
            db.scalars(select(ModelVersion.id).where(ModelVersion.id.in_(referenced_models)))
        )
        if missing := referenced_models - found:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Unknown model_version_id(s): {sorted(missing)}",
            )

    new_rows: list[Annotation] = []
    for index, item in enumerate(items):
        if item.class_name not in allowed:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Annotation {index}: class '{item.class_name}' is not one of "
                f"the project classes {sorted(allowed)}",
            )
        new_rows.append(
            Annotation(
                image_id=image_id,
                class_name=item.class_name,
                type=AnnotationType(item.geometry.type),
                geometry=_check_and_store(item, index, image),
                source=item.source,
                score=item.score,
                model_version_id=item.model_version_id,
            )
        )

    # Replace in one transaction: either all new annotations are saved, or nothing changes.
    db.execute(delete(Annotation).where(Annotation.image_id == image_id))
    db.add_all(new_rows)
    if image.status == ImageStatus.TODO and new_rows:
        image.status = ImageStatus.IN_PROGRESS  # started labeling this image
    db.commit()
    return [to_read(a) for a in new_rows]
