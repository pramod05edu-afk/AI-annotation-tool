"""Health check: is the API up, and can it reach the database and the image storage?"""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.db import check_database
from app.storage import Storage, get_storage

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    storage: bool


def database_status() -> bool:
    """Dependency wrapper so tests can replace the real database check."""
    return check_database()


@router.get("/health", response_model=HealthResponse)
def health(
    database_ok: bool = Depends(database_status),
    storage: Storage = Depends(get_storage),
) -> HealthResponse:
    storage_ok = storage.is_healthy()
    status: Literal["ok", "degraded"] = "ok" if database_ok and storage_ok else "degraded"
    return HealthResponse(status=status, database=database_ok, storage=storage_ok)
