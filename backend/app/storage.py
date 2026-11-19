"""Object storage for image files, using the S3 API (SeaweedFS locally, AWS S3 in the cloud).

Images live here, not in the database: databases are bad at large binary blobs,
and S3-style storage is how real ML pipelines keep their data.
"""

from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings, get_settings


class Storage:
    """Thin wrapper around an S3 client, bound to one bucket."""

    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        # Normal client: a few retries, because a failed upload is worse than a slow one.
        self.client: Any = self._make_client(
            settings, connect_timeout=2, read_timeout=30, total_max_attempts=3
        )
        # Health-check client: fail fast. Health checks must answer within seconds,
        # otherwise Docker marks the whole backend unhealthy when storage is down.
        self._probe: Any = self._make_client(
            settings, connect_timeout=1, read_timeout=2, total_max_attempts=1
        )

    @staticmethod
    def _make_client(
        settings: Settings, connect_timeout: float, read_timeout: float, total_max_attempts: int
    ) -> Any:
        return boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                retries={"total_max_attempts": total_max_attempts, "mode": "standard"},
            ),
        )

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist yet (called once at startup)."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def is_healthy(self) -> bool:
        """Return True if the storage server is reachable and the bucket exists."""
        try:
            self._probe.head_bucket(Bucket=self.bucket)
            return True
        except (BotoCoreError, ClientError):
            return False


@lru_cache
def get_storage() -> Storage:
    """FastAPI dependency: one shared storage client for the whole app."""
    return Storage(get_settings())
