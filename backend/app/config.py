"""Application settings, read from environment variables (or a local .env file).

Every value has a local-development default so the app starts without any configuration.
In Docker Compose the values are passed as environment variables.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Default = the Docker Compose database as seen from your PC (host port 5433).
    database_url: str = "postgresql+psycopg://annotator:annotator@localhost:5433/annotator"

    s3_endpoint_url: str = "http://localhost:8333"
    s3_access_key: str = "annotator"
    s3_secret_key: str = "annotator-secret"
    s3_bucket: str = "annotations"

    max_upload_mb: int = 20

    # ---- AI models ----
    load_models: bool = True  # tests turn this off
    # Locally: the repo's weights/ folder (git-ignored). In Docker: /app/weights (a mounted folder).
    weights_dir: str = str(Path(__file__).resolve().parents[2] / "weights")
    # The fine-tuned detector exported by eval/colab_finetune.ipynb (folder or .pt file).
    detector_weights: str = "yolo11n_helmet_openvino_model"
    detector_imgsz: int = 640  # must match the training / export size
    # Downloaded automatically from the Ultralytics releases on first start if missing.
    segmenter_weights: str = "mobile_sam.pt"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, created once and then reused."""
    return Settings()
