"""API request and response shapes (Pydantic). FastAPI validates every request against these
and returns a clear 422 error when something is wrong, before our code even runs."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.models import AnnotationMode, AnnotationSource, ImageStatus, ModelTask

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
ClassName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class ORMModel(BaseModel):
    """Base for responses built directly from database objects."""

    model_config = ConfigDict(from_attributes=True)


# ---------- Projects ----------


def _unique(classes: list[str]) -> list[str]:
    if len(set(classes)) != len(classes):
        raise ValueError("class names must be unique")
    return classes


class ProjectCreate(BaseModel):
    name: Name
    classes: list[ClassName] = Field(min_length=1, max_length=100)

    _check_classes = field_validator("classes")(_unique)


class ProjectUpdate(BaseModel):
    name: Name | None = None
    classes: list[ClassName] | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("classes")
    @classmethod
    def _check_classes(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else _unique(value)


class ProjectRead(ORMModel):
    id: int
    name: str
    classes: list[str]
    created_at: datetime
    image_count: int = 0
    status_counts: dict[ImageStatus, int] = Field(default_factory=dict)


# ---------- Images ----------


class ImageRead(ORMModel):
    id: int
    project_id: int
    filename: str
    content_type: str
    width: int
    height: int
    status: ImageStatus
    created_at: datetime
    updated_at: datetime
    annotation_count: int = 0


class ImageStatusUpdate(BaseModel):
    status: ImageStatus


class UploadRejected(BaseModel):
    filename: str
    reason: str


class UploadResult(BaseModel):
    uploaded: list[ImageRead]
    rejected: list[UploadRejected]


# ---------- Annotations ----------
# Coordinates are always in ORIGINAL image pixels (never screen pixels).


class BoxGeometry(BaseModel):
    type: Literal["box"] = "box"
    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max

    @field_validator("bbox")
    @classmethod
    def _min_before_max(
        cls, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = bbox
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("bbox must be [x_min, y_min, x_max, y_max] with min < max")
        return bbox


class PolygonGeometry(BaseModel):
    type: Literal["polygon"] = "polygon"
    points: list[tuple[float, float]] = Field(min_length=3, max_length=5000)


# "discriminator": the `type` field decides which of the two shapes to validate against.
Geometry = Annotated[BoxGeometry | PolygonGeometry, Field(discriminator="type")]


class AnnotationIn(BaseModel):
    class_name: ClassName
    geometry: Geometry
    source: AnnotationSource = AnnotationSource.MANUAL
    score: float | None = Field(default=None, ge=0, le=1)
    model_version_id: int | None = None


class AnnotationRead(BaseModel):
    id: int
    image_id: int
    class_name: str
    geometry: Geometry
    source: AnnotationSource
    score: float | None
    model_version_id: int | None
    created_at: datetime


# ---------- Model versions ----------


class ModelVersionCreate(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    task: ModelTask
    weights_uri: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    metrics: dict[str, Any] = Field(default_factory=dict)


class ModelVersionRead(ORMModel):
    id: int
    name: str
    task: ModelTask
    weights_uri: str
    metrics: dict[str, Any]
    is_active: bool
    created_at: datetime


# ---------- Interaction logs (effort study) ----------


class LogEventIn(BaseModel):
    image_id: int
    session_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    mode: AnnotationMode
    event: Annotated[str, StringConstraints(min_length=1, max_length=50)]
    payload: dict[str, Any] = Field(default_factory=dict)
    # The browser's own timestamp: more accurate for timing than when the batch arrives.
    ts: datetime | None = None


class LogBatch(BaseModel):
    events: list[LogEventIn] = Field(min_length=1, max_length=500)


class LogBatchResult(BaseModel):
    inserted: int
