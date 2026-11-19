"""Object storage for image files, using the S3 API (SeaweedFS locally, AWS S3 in the cloud).

Images live here, not in the database: databases are bad at large binary blobs,
and S3-style storage is how real ML pipelines keep their data.
"""

from collections.abc import Iterator
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

    # ---------- object operations ----------

    def put(self, key: str, data: bytes, content_type: str) -> None:
        """Store bytes under `key` (overwrites an existing object)."""
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        """Read a whole object into memory (fine for images of a few MB)."""
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        data: bytes = response["Body"].read()
        return data

    def stream(self, key: str) -> tuple[Iterator[bytes], int | None]:
        """Return (chunk iterator, size in bytes) so large files never load fully into memory.
        Raises ObjectNotFoundError if the key does not exist."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise ObjectNotFoundError(key) from error
            raise
        chunks: Iterator[bytes] = response["Body"].iter_chunks(chunk_size=64 * 1024)
        return chunks, response.get("ContentLength")

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object whose key starts with `prefix` (e.g. a whole project).
        Returns how many objects were deleted."""
        deleted = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:  # the S3 API accepts at most 1000 keys per call; one page is <= 1000
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})
                deleted += len(keys)
        return deleted


class ObjectNotFoundError(Exception):
    """The requested storage key does not exist."""


@lru_cache
def get_storage() -> Storage:
    """FastAPI dependency: one shared storage client for the whole app."""
    return Storage(get_settings())
