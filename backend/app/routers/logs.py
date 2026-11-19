"""Interaction logging for the manual-vs-assisted effort study.
The browser buffers events and sends them in batches (fewer requests while annotating)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Image, InteractionLog
from app.schemas import LogBatch, LogBatchResult

router = APIRouter(prefix="/logs", tags=["logs"])


@router.post("", response_model=LogBatchResult, status_code=status.HTTP_201_CREATED)
def record_events(batch: LogBatch, db: Session = Depends(get_db)) -> LogBatchResult:
    image_ids = {e.image_id for e in batch.events}
    found = set(db.scalars(select(Image.id).where(Image.id.in_(image_ids))))
    if missing := image_ids - found:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown image_id(s): {sorted(missing)}"
        )
    rows = []
    for event in batch.events:
        row = InteractionLog(
            image_id=event.image_id,
            session_id=event.session_id,
            mode=event.mode,
            event=event.event,
            payload=event.payload,
        )
        if event.ts is not None:
            row.ts = event.ts  # otherwise the database fills in now()
        rows.append(row)
    db.add_all(rows)
    db.commit()
    return LogBatchResult(inserted=len(rows))
