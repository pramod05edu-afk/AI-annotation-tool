"""Totals for the dashboard: how much has been labelled, and how much of it by the AI."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Annotation, Image, ImageStatus, Project
from app.schemas import StatsRead

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsRead)
def stats(db: Session = Depends(get_db)) -> StatsRead:
    by_source = {
        str(source.value): count
        for source, count in db.execute(
            select(Annotation.source, func.count()).group_by(Annotation.source)
        )
    }
    return StatsRead(
        projects=db.scalar(select(func.count()).select_from(Project)) or 0,
        images_total=db.scalar(select(func.count()).select_from(Image)) or 0,
        images_done=db.scalar(
            select(func.count()).select_from(Image).where(Image.status == ImageStatus.DONE)
        )
        or 0,
        labels_total=sum(by_source.values()),
        labels_by_source=by_source,
    )
