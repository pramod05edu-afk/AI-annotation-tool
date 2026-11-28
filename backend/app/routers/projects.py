"""Projects: a named set of images plus the list of label classes used to annotate them."""

import logging

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import case, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_project_or_404
from app.models import Annotation, Image, ImageStatus, Project
from app.schemas import ProjectCreate, ProjectRead, ProjectUpdate
from app.storage import Storage, get_storage

router = APIRouter(prefix="/projects", tags=["projects"])
logger = logging.getLogger(__name__)


def _with_counts(db: Session, projects: list[Project]) -> list[ProjectRead]:
    """Attach image counts per status, using one GROUP BY query for all projects."""
    ids = [p.id for p in projects]
    counts: dict[int, dict[ImageStatus, int]] = {pid: {} for pid in ids}
    if ids:
        rows = db.execute(
            select(Image.project_id, Image.status, func.count())
            .where(Image.project_id.in_(ids))
            .group_by(Image.project_id, Image.status)
        )
        for project_id, image_status, n in rows:
            counts[project_id][image_status] = n
    previews = _preview_image_ids(db, ids)
    return [
        ProjectRead.model_validate(p).model_copy(
            update={
                "status_counts": counts[p.id],
                "image_count": sum(counts[p.id].values()),
                "preview_image_ids": previews.get(p.id, []),
            }
        )
        for p in projects
    ]


def _preview_image_ids(
    db: Session, project_ids: list[int], per_project: int = 3
) -> dict[int, list[int]]:
    """The first few image ids of every project (card thumbnails), in ONE query.

    ROW_NUMBER() numbers the images inside each project; keeping rows 1..3 avoids one query
    per project (the "N+1 queries" problem).
    """
    if not project_ids:
        return {}
    position = func.row_number().over(partition_by=Image.project_id, order_by=Image.id)
    numbered = (
        select(Image.project_id, Image.id, position.label("position"))
        .where(Image.project_id.in_(project_ids))
        .subquery()
    )
    rows = db.execute(
        select(numbered.c.project_id, numbered.c.id)
        .where(numbered.c.position <= per_project)
        .order_by(numbered.c.project_id, numbered.c.id)
    )
    previews: dict[int, list[int]] = {}
    for project_id, image_id in rows:
        previews.setdefault(project_id, []).append(image_id)
    return previews


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)) -> ProjectRead:
    project = Project(name=body.name, classes=body.classes, annotation_mode=body.annotation_mode)
    db.add(project)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A project named '{body.name}' already exists"
        ) from error
    return _with_counts(db, [project])[0]


@router.get("", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectRead]:
    projects = list(db.scalars(select(Project).order_by(Project.created_at.desc())))
    return _with_counts(db, projects)


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: int, db: Session = Depends(get_db)) -> ProjectRead:
    return _with_counts(db, [get_project_or_404(db, project_id)])[0]


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(
    project_id: int, body: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectRead:
    project = get_project_or_404(db, project_id)
    if body.classes is not None:
        unknown = sorted(set(body.renames) - set(project.classes))
        if unknown:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, f"Cannot rename unknown labels: {unknown}"
            )
        # Refuse to remove a class that existing annotations still use (renamed ones move).
        removed = set(project.classes) - set(body.classes) - set(body.renames)
        if removed:
            in_use = db.scalars(
                select(Annotation.class_name)
                .join(Image)
                .where(Image.project_id == project_id, Annotation.class_name.in_(removed))
                .distinct()
            ).all()
            if in_use:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"Labels still used by boxes: {', '.join(sorted(in_use))}. Delete or relabel"
                    " those boxes first, or rename the label instead.",
                )
        renames = {old: new for old, new in body.renames.items() if old != new}
        if renames:
            # One statement with CASE, so even a swap (a -> b, b -> a) is applied correctly.
            db.execute(
                update(Annotation)
                .where(
                    Annotation.image_id.in_(select(Image.id).where(Image.project_id == project_id)),
                    Annotation.class_name.in_(renames),
                )
                .values(class_name=case(renames, value=Annotation.class_name))
            )
        project.classes = body.classes
    if body.name is not None:
        project.name = body.name
    if "annotation_mode" in body.model_fields_set:  # given, even as null: set or unlock it
        project.annotation_mode = body.annotation_mode
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A project named '{body.name}' already exists"
        ) from error
    return _with_counts(db, [project])[0]


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> Response:
    project = get_project_or_404(db, project_id)
    db.delete(project)  # images, annotations and logs follow via ON DELETE CASCADE
    db.commit()
    try:
        storage.delete_prefix(f"projects/{project_id}/")
    except (BotoCoreError, ClientError):
        # The database is the source of truth; leftover files are harmless and can be cleaned.
        logger.exception("Could not delete stored files of project %s", project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
