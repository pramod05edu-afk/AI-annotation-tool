"""Model registry: which trained models exist, and which one is active per task.
Every AI suggestion stores the model version that produced it, so pretrained and fine-tuned
models can be compared inside the app (and in the evaluation)."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ModelTask, ModelVersion
from app.schemas import ModelVersionCreate, ModelVersionRead

router = APIRouter(prefix="/models", tags=["models"])


def _get_or_404(db: Session, model_id: int) -> ModelVersion:
    model = db.get(ModelVersion, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model version {model_id} not found")
    return model


@router.get("", response_model=list[ModelVersionRead])
def list_models(
    task: ModelTask | None = Query(default=None), db: Session = Depends(get_db)
) -> list[ModelVersion]:
    query = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    if task is not None:
        query = query.where(ModelVersion.task == task)
    return list(db.scalars(query))


@router.post("", response_model=ModelVersionRead, status_code=status.HTTP_201_CREATED)
def register_model(body: ModelVersionCreate, db: Session = Depends(get_db)) -> ModelVersion:
    model = ModelVersion(**body.model_dump())
    db.add(model)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"A model named '{body.name}' already exists"
        ) from error
    return model


@router.post("/{model_id}/activate", response_model=ModelVersionRead)
def activate_model(model_id: int, db: Session = Depends(get_db)) -> ModelVersion:
    model = _get_or_404(db, model_id)
    # Deactivate the others first; the database also enforces "one active per task".
    db.execute(
        update(ModelVersion)
        .where(ModelVersion.task == model.task, ModelVersion.id != model.id)
        .values(is_active=False)
    )
    model.is_active = True
    db.commit()
    return model
