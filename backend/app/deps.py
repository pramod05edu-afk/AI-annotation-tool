"""Shared helpers for routers: load an object or answer 404."""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models import Image, Project


def get_project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Project {project_id} not found")
    return project


def get_image_or_404(db: Session, image_id: int) -> Image:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")
    return image
