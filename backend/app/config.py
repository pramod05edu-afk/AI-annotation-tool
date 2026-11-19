"""Application settings, read from environment variables (or a local .env file).

Every value has a local-development default so the app starts without any configuration.
In Docker Compose the values are passed as environment variables.
"""

from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    """Return the settings, created once and then reused."""
    return Settings()
