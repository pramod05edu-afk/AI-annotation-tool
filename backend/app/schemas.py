"""API request and response shapes (Pydantic). FastAPI validates every request against these
and returns a clear 422 error when something is wrong, before our code even runs."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

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
    annotation_mode: AnnotationMode | None = None  # fixed mode for every image (effort study)

    _check_classes = field_validator("classes")(_unique)


class ProjectUpdate(BaseModel):
    name: Name | None = None
    # The new label list (its order sets the keyboard shortcuts 1-9).
    classes: list[ClassName] | None = Field(default=None, min_length=1, max_length=100)
    # Renamed labels, old name -> new name: existing annotations move to the new name.
    # Without this, a rename would look like "remove a label that is still in use".
    renames: dict[str, ClassName] = Field(default_factory=dict, max_length=100)
    annotation_mode: AnnotationMode | None = None  # send null explicitly to unlock the mode

    @field_validator("classes")
    @classmethod
    def _check_classes(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else _unique(value)

    @model_validator(mode="after")
    def _renames_fit_the_new_list(self) -> "ProjectUpdate":
        if not self.renames:
            return self
        if self.classes is None:
            raise ValueError("renames need the new 'classes' list")
        missing = sorted(set(self.renames.values()) - set(self.classes))
        if missing:
            raise ValueError(f"renamed labels must be in 'classes': {missing}")
        if len(set(self.renames.values())) != len(self.renames):
            raise ValueError("two labels cannot be renamed to the same name")
        return self


class ProjectRead(ORMModel):
    id: int
    name: str
    classes: list[str]
    annotation_mode: AnnotationMode | None = None
    created_at: datetime
    image_count: int = 0
    status_counts: dict[ImageStatus, int] = Field(default_factory=dict)
    preview_image_ids: list[int] = Field(default_factory=list)  # first images, for thumbnails


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


# ---------- Dashboard ----------


class StatsRead(BaseModel):
    projects: int
    images_total: int
    images_done: int
    labels_total: int
    labels_by_source: dict[str, int]


# ---------- AI assistance ----------


class ModelSlotStatus(BaseModel):
    state: Literal["disabled", "loading", "ready", "failed"]
    name: str | None = None
    model_version_id: int | None = None
    error: str | None = None


class AiStatus(BaseModel):
    detector: ModelSlotStatus  # the fine-tuned specialist (its own fixed classes)
    general: ModelSlotStatus  # the open-vocabulary detector (any label names)
    segmenter: ModelSlotStatus


class SuggestRequest(BaseModel):
    # Only boxes the model is at least this sure about. Lower = more boxes, more false alarms.
    conf: float = Field(default=0.25, ge=0.01, le=0.99)


class Suggestion(BaseModel):
    class_name: str
    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max (image pixels)
    score: float
    # Which model produced this box (one request can use the specialist AND the general model).
    model: str | None = None
    model_version_id: int | None = None


class SuggestResult(BaseModel):
    model_version_id: int | None  # the model of the first box; each box also names its own
    models: list[str] = Field(default_factory=list)  # the models that ran for this image
    latency_ms: float
    suggestions: list[Suggestion]
    # Model classes that are not project classes, with how many boxes were dropped.
    ignored: dict[str, int]


class SegmentRequest(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=1, max_length=20)
    labels: list[Literal[0, 1]]  # 1 = "this is the object", 0 = "this is NOT the object"

    @field_validator("labels")
    @classmethod
    def _labels_match_points(cls, labels: list[int], info: ValidationInfo) -> list[int]:
        points = info.data.get("points")
        if points is not None and len(points) != len(labels):
            raise ValueError("one label per point is required")
        if 1 not in labels:
            raise ValueError("at least one point must be on the object (label 1)")
        return labels


class SegmentResult(BaseModel):
    model_version_id: int | None
    latency_ms: float
    points: list[tuple[float, float]]  # polygon outline in image pixels
    bbox: tuple[float, float, float, float]
    score: float


class ClickBoxRequest(BaseModel):
    point: tuple[float, float]  # where the user clicked (image pixels)
    # Boxes already labelled on this image: candidates that duplicate them are skipped.
    exclude: list[tuple[float, float, float, float]] = Field(default_factory=list, max_length=1000)


class FindSimilarRequest(BaseModel):
    # One or more boxes the user drew around examples of ONE kind of object.
    examples: list[tuple[float, float, float, float]] = Field(min_length=1, max_length=10)
    class_name: str  # the project label to give the objects found
    conf: float = Field(default=0.25, ge=0.01, le=0.99)
    # Boxes already labelled on this image (including the examples): not suggested again.
    exclude: list[tuple[float, float, float, float]] = Field(default_factory=list, max_length=1000)


class FindSimilarResult(BaseModel):
    model_version_id: int | None
    latency_ms: float
    suggestions: list[Suggestion]


class ClickBoxResult(BaseModel):
    model_version_id: int | None
    latency_ms: float
    suggestion: Suggestion | None  # None: the detector sees no object at this point
