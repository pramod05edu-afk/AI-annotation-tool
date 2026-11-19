"""Shared test setup.

- A real PostgreSQL test database (`annotator_test`), created if missing, migrated with Alembic
  (down to base and up to head, so the migrations themselves are tested).
- Each test runs inside a transaction that is rolled back afterwards: tests never see each
  other's data, and nothing needs cleaning up.
- Object storage is replaced by an in-memory fake.

Locally: start the database first with `docker compose up -d postgres`. Without it, database
tests are skipped; in CI (REQUIRE_DB=1) they fail instead, so they can never silently skip.
"""

import io
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from alembic import command
from app.db import get_db
from app.main import app
from app.storage import ObjectNotFoundError, get_storage

BACKEND_DIR = Path(__file__).resolve().parents[1]
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://annotator:annotator@localhost:5433/annotator_test"
)


class FakeStorage:
    """In-memory stand-in for the S3 storage, with the same methods the routers use."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = (data, content_type)

    def get(self, key: str) -> bytes:
        return self.objects[key][0]

    def stream(self, key: str) -> tuple[Iterator[bytes], int | None]:
        if key not in self.objects:
            raise ObjectNotFoundError(key)
        data = self.objects[key][0]
        return iter([data]), len(data)

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self.objects if k.startswith(prefix)]
        for key in keys:
            del self.objects[key]
        return len(keys)

    def is_healthy(self) -> bool:
        return True

    def ensure_bucket(self) -> None:
        pass


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    url = make_url(TEST_DATABASE_URL)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url.database}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    except OperationalError:
        if os.environ.get("REQUIRE_DB"):
            raise
        pytest.skip(
            f"PostgreSQL not reachable at {url.host}:{url.port} — "
            "start it with: docker compose up -d postgres"
        )
    finally:
        admin.dispose()

    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", url.render_as_string(hide_password=False))
    command.downgrade(config, "base")  # proves every downgrade works
    command.upgrade(config, "head")

    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    connection = db_engine.connect()
    transaction = connection.begin()
    # Commits inside the app become SAVEPOINT releases; the outer transaction is rolled back.
    session = Session(
        bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
    )
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def client(db_session: Session, storage: FakeStorage) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_storage] = lambda: storage
    yield TestClient(app)
    app.dependency_overrides.clear()


def make_image(width: int = 64, height: int = 48, fmt: str = "PNG") -> bytes:
    """Create a small real image file in memory."""
    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height), color=(200, 120, 40)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def project(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/api/projects", json={"name": "Hard hats", "classes": ["helmet", "head"]}
    )
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


@pytest.fixture
def upload(client: TestClient) -> Callable[..., dict[str, object]]:
    """Upload one image to a project and return its JSON."""

    def _upload(
        project_id: int, width: int = 64, height: int = 48, name: str = "a.png"
    ) -> dict[str, object]:
        files = [("files", (name, make_image(width, height), "image/png"))]
        response = client.post(f"/api/projects/{project_id}/images", files=files)
        assert response.status_code == 201, response.text
        uploaded: list[dict[str, object]] = response.json()["uploaded"]
        assert len(uploaded) == 1
        return uploaded[0]

    return _upload
