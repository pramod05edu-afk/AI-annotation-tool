"""Database tables (SQLAlchemy 2.0 ORM models).

    projects ─┬─< images ─┬─< annotations >── model_versions
              │           └─< interaction_logs
              └ classes (JSON list of label names)

`─<` means "one to many". Deleting a project deletes its images, and deleting an image deletes
its annotations and logs (ON DELETE CASCADE).
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

EMBEDDING_DIM = 512  # OpenCLIP ViT-B/32 (stretch goal, Phase 8)


class ImageStatus(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class AnnotationType(StrEnum):
    BOX = "box"
    POLYGON = "polygon"


class AnnotationSource(StrEnum):
    MANUAL = "manual"  # drawn by the user
    AI = "ai"  # model suggestion accepted unchanged
    AI_EDITED = "ai_edited"  # model suggestion accepted after the user changed it


class ModelTask(StrEnum):
    DETECT = "detect"  # the fine-tuned specialist (fixed classes)
    DETECT_OPEN = "detect_open"  # the general open-vocabulary detector (any classes)
    SEGMENT = "segment"
    EMBED = "embed"


class AnnotationMode(StrEnum):
    MANUAL = "manual"
    ASSISTED = "assisted"


def _enum(enum_cls: type[StrEnum]) -> Enum:
    """Store enums as plain strings (VARCHAR) with their values, e.g. 'in_progress'."""
    return Enum(
        enum_cls,
        native_enum=False,
        length=20,
        values_callable=lambda e: [member.value for member in e],
        validate_strings=True,
        create_constraint=True,  # CHECK constraint: the database itself rejects bad values
    )


def _now() -> Any:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    classes: Mapped[list[str]] = mapped_column(JSONB)
    # Fixed annotation mode (effort study): the UI enforces it for every image of the project.
    # NULL = the annotator chooses freely.
    annotation_mode: Mapped[AnnotationMode | None] = mapped_column(
        _enum(AnnotationMode), nullable=True
    )
    created_at: Mapped[datetime] = _now()

    images: Mapped[list["Image"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))  # original name, for display/export
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)  # where the file lives
    content_type: Mapped[str] = mapped_column(String(50))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    status: Mapped[ImageStatus] = mapped_column(
        _enum(ImageStatus), default=ImageStatus.TODO, server_default=ImageStatus.TODO.value
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    project: Mapped[Project] = relationship(back_populates="images")
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True
    )


class ModelVersion(Base):
    """One trained model: which weights, for which task, how good it was, whether it is in use."""

    __tablename__ = "model_versions"
    __table_args__ = (
        # At most ONE active model per task (partial unique index, PostgreSQL feature).
        Index(
            "uq_model_versions_one_active_per_task",
            "task",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)  # e.g. "yolo11s-hardhat-v1"
    task: Mapped[ModelTask] = mapped_column(_enum(ModelTask))
    weights_uri: Mapped[str] = mapped_column(String(1000))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = _now()


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    class_name: Mapped[str] = mapped_column(String(100))
    type: Mapped[AnnotationType] = mapped_column(_enum(AnnotationType))
    # In ORIGINAL image pixels. box: {"bbox": [x_min, y_min, x_max, y_max]}
    # polygon: {"points": [[x, y], ...], "bbox": [...]} (bbox kept for fast export/eval)
    geometry: Mapped[dict[str, Any]] = mapped_column(JSONB)
    source: Mapped[AnnotationSource] = mapped_column(_enum(AnnotationSource))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # model confidence
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = _now()

    image: Mapped[Image] = relationship(back_populates="annotations")


class InteractionLog(Base):
    """User actions while annotating — the raw data for the manual-vs-assisted effort study."""

    __tablename__ = "interaction_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    mode: Mapped[AnnotationMode] = mapped_column(_enum(AnnotationMode))
    event: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    ts: Mapped[datetime] = _now()
