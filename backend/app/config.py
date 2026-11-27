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
    # If the detector is missing (fresh clone), it is downloaded from the GitHub Release and
    # checked against this SHA-256 hash. Set DETECTOR_URL to "" to disable the download.
    detector_url: str = (
        "https://github.com/pramod05edu-afk/AI-annotation-tool/releases/download/"
        "model-v1/yolo11n_helmet_openvino_model.zip"
    )
    detector_sha256: str = "36ff9006ed6258b708c2184f9a38e7ce046a119741e71e197b7283bed0a2f44d"
    # Downloaded automatically from the Ultralytics releases on first start if missing.
    segmenter_weights: str = "mobile_sam.pt"
    # General open-vocabulary detector for labels the specialist does not know (any project).
    # Also downloaded automatically, together with its text encoder (~250 MB, first start only).
    general_weights: str = "yoloe-26s-seg.pt"
    general_text_encoder: str = "mobileclip2_b.ts"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, created once and then reused."""
    return Settings()
