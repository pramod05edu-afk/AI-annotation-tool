"""Tests for /api/health. The real database and storage are replaced with fakes."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.health import database_status
from app.storage import get_storage


class FakeStorage:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy

    def is_healthy(self) -> bool:
        return self.healthy


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Not used as a context manager, so the startup lifespan (bucket creation) does not run.
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("db_ok", "storage_ok", "expected"),
    [(True, True, "ok"), (False, True, "degraded"), (True, False, "degraded")],
)
def test_health_reports_each_dependency(
    client: TestClient, db_ok: bool, storage_ok: bool, expected: str
) -> None:
    app.dependency_overrides[database_status] = lambda: db_ok
    app.dependency_overrides[get_storage] = lambda: FakeStorage(storage_ok)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": expected, "database": db_ok, "storage": storage_ok}
