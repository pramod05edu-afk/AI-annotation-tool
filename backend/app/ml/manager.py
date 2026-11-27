"""Loads the AI models once, in the background, and records which version is serving.

Loading takes ~10-40 s on a laptop (PyTorch import, weights, warm-up). Doing it in a
background thread lets the API start immediately; until a model is ready the AI endpoints
answer 503 and the UI shows "AI loading…". Every loaded model is registered in the
`model_versions` table, so each AI annotation records exactly which model produced it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, TypeVar

from fastapi import HTTPException, status
from sqlalchemy import select, update

from app.config import Settings
from app.db import SessionLocal
from app.ml.base import Detector, OpenVocabDetector, Segmenter
from app.ml.download import ensure_model
from app.models import ModelTask, ModelVersion

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ModelState(StrEnum):
    DISABLED = "disabled"  # turned off in the settings (e.g. in tests)
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ModelSlot(Generic[T]):
    label: str  # for error messages: "detector" / "general detector" / "segmenter"
    state: ModelState = ModelState.DISABLED
    model: T | None = None
    name: str | None = None
    model_version_id: int | None = None
    error: str | None = None

    def require(self) -> T:
        """The model, or a clear 503 error if it cannot be used (yet)."""
        if self.state == ModelState.READY and self.model is not None:
            return self.model
        reason = {
            ModelState.LOADING: f"The {self.label} is still loading, try again in a few seconds",
            ModelState.FAILED: f"The {self.label} could not be loaded: {self.error}",
            ModelState.DISABLED: f"The {self.label} is disabled",
        }[self.state]
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, reason)


@dataclass
class ModelManager:
    detector: ModelSlot[Detector] = field(default_factory=lambda: ModelSlot("detector"))
    segmenter: ModelSlot[Segmenter] = field(default_factory=lambda: ModelSlot("segmenter"))
    # The general open-vocabulary detector: labels the fine-tuned specialist does not know.
    general: ModelSlot[OpenVocabDetector] = field(
        default_factory=lambda: ModelSlot("general detector")
    )

    def start_loading(self, settings: Settings) -> None:
        self.detector.state = self.segmenter.state = self.general.state = ModelState.LOADING
        threading.Thread(target=self._load_all, args=(settings,), daemon=True).start()

    def _load_all(self, settings: Settings) -> None:
        # Imported here: PyTorch/Ultralytics take seconds to import and tests never need them.
        from app.ml.sam import MobileSamSegmenter
        from app.ml.yolo import YoloDetector
        from app.ml.yoloe import YoloeDetector

        weights_dir = Path(settings.weights_dir)
        detector_path = weights_dir / settings.detector_weights

        def load_detector() -> YoloDetector:
            ensure_model(detector_path, settings.detector_url, settings.detector_sha256)
            if not detector_path.exists():
                raise FileNotFoundError(
                    f"{detector_path} not found - put the trained model there "
                    "(see docs/training.md) or set DETECTOR_WEIGHTS"
                )
            detector = YoloDetector(detector_path, imgsz=settings.detector_imgsz)
            detector.warm_up()
            return detector

        def load_segmenter() -> MobileSamSegmenter:
            segmenter = MobileSamSegmenter(weights_dir / settings.segmenter_weights)
            segmenter.warm_up()
            return segmenter

        def load_general() -> YoloeDetector:
            _prepare_text_encoder(weights_dir, settings.general_text_encoder)
            general = YoloeDetector(weights_dir / settings.general_weights)
            general.warm_up()
            return general

        # The specialist and SAM first (small, fast): the largest download comes last.
        self._load(
            self.detector, load_detector, ModelTask.DETECT, detector_path,
            # Read AFTER loading: on a fresh clone the file arrives with the model download.
            metrics=lambda: _detector_metrics(weights_dir, detector_path),
        )  # fmt: skip
        self._load(
            self.segmenter, load_segmenter, ModelTask.SEGMENT,
            weights_dir / settings.segmenter_weights, metrics=dict,
        )  # fmt: skip
        self._load(
            self.general, load_general, ModelTask.DETECT_OPEN,
            weights_dir / settings.general_weights, metrics=dict,
        )  # fmt: skip

    @staticmethod
    def _load(
        slot: ModelSlot[Any],
        loader: Any,
        task: ModelTask,
        weights: Path,
        metrics: Callable[[], dict[str, Any]],
    ) -> None:
        start = time.perf_counter()
        try:
            model = loader()
            slot.model_version_id = register_model_version(model.name, task, weights, metrics())
            slot.model, slot.name, slot.state = model, model.name, ModelState.READY
            seconds = time.perf_counter() - start
            logger.info("%s %s ready in %.1f s", slot.label, model.name, seconds)
        except Exception as error:  # any failure: keep the API running without this model
            slot.state, slot.error = ModelState.FAILED, str(error)
            logger.exception("Could not load the %s", slot.label)


def _prepare_text_encoder(weights_dir: Path, file_name: str) -> None:
    """Make YOLOE find its text encoder in our weights folder (downloading it once if missing).

    Ultralytics looks for it in the current directory and then in ITS weights folder, and
    downloads into the current directory: point its weights folder at ours, and download to
    an explicit path first, so the file lands in the mounted weights/ volume and survives
    container restarts.
    """
    from ultralytics.utils import SETTINGS
    from ultralytics.utils.downloads import attempt_download_asset

    if SETTINGS["weights_dir"] != str(weights_dir):
        SETTINGS.update(weights_dir=str(weights_dir))  # type: ignore[no-untyped-call]
    attempt_download_asset(str(weights_dir / file_name))


def _detector_metrics(weights_dir: Path, detector_path: Path) -> dict[str, Any]:
    """Test-set metrics written by the Colab notebook (detector_metrics.json), if present."""
    metrics_file = weights_dir / "detector_metrics.json"
    if not metrics_file.exists():
        return {}
    all_metrics: dict[str, Any] = json.loads(metrics_file.read_text())
    model_key = detector_path.name.split("_")[0]  # "yolo11n_helmet_openvino_model" -> "yolo11n"
    result: dict[str, Any] = all_metrics.get(model_key, {})
    return result


def register_model_version(
    name: str, task: ModelTask, weights: Path, metrics: dict[str, Any]
) -> int:
    """Find or create the registry row for this model and make it the active one for its task."""
    with SessionLocal() as db:
        row = db.scalar(select(ModelVersion).where(ModelVersion.name == name))
        if row is None:
            row = ModelVersion(name=name, task=task, weights_uri=str(weights), metrics=metrics)
            db.add(row)
            db.flush()
        elif metrics:
            row.metrics = metrics
        db.execute(
            update(ModelVersion)
            .where(ModelVersion.task == task, ModelVersion.id != row.id)
            .values(is_active=False)
        )
        db.flush()  # deactivate first: only one active model per task is allowed
        row.is_active = True
        db.commit()
        return row.id


# One manager for the whole process (the API routes receive it through get_models()).
models = ModelManager()


def get_models() -> ModelManager:
    return models
