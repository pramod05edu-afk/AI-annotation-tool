from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import InteractionLog

Upload = Callable[..., dict[str, Any]]


def _register(client: TestClient, name: str, task: str = "detect") -> dict[str, Any]:
    response = client.post(
        "/api/models",
        json={"name": name, "task": task, "weights_uri": f"weights/{name}.pt", "metrics": {}},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def test_register_and_list_models(client: TestClient) -> None:
    _register(client, "yolo11s-coco")
    _register(client, "mobilesam", task="segment")

    assert len(client.get("/api/models").json()) == 2
    detectors = client.get("/api/models", params={"task": "detect"}).json()
    assert [m["name"] for m in detectors] == ["yolo11s-coco"]
    assert detectors[0]["is_active"] is False


def test_duplicate_model_name_is_rejected(client: TestClient) -> None:
    _register(client, "yolo11s-coco")
    duplicate = client.post(
        "/api/models", json={"name": "yolo11s-coco", "task": "detect", "weights_uri": "x"}
    )
    assert duplicate.status_code == 409


def test_only_one_active_model_per_task(client: TestClient) -> None:
    pretrained = _register(client, "yolo11s-coco")
    finetuned = _register(client, "yolo11s-hardhat-v1")
    sam = _register(client, "mobilesam", task="segment")

    client.post(f"/api/models/{pretrained['id']}/activate")
    client.post(f"/api/models/{sam['id']}/activate")
    client.post(f"/api/models/{finetuned['id']}/activate")

    active = {m["name"] for m in client.get("/api/models").json() if m["is_active"]}
    # Activating the fine-tuned detector switched off the pretrained one, not the segmenter.
    assert active == {"yolo11s-hardhat-v1", "mobilesam"}


def test_log_batch_is_stored_with_client_timestamps(
    client: TestClient,
    project: dict[str, Any],
    upload: Upload,
    db_session: Session,
) -> None:
    image = upload(project["id"])
    events = [
        {
            "image_id": image["id"],
            "session_id": "s1",
            "mode": "assisted",
            "event": "image_opened",
            "ts": "2026-11-19T10:00:00+00:00",
        },
        {
            "image_id": image["id"],
            "session_id": "s1",
            "mode": "assisted",
            "event": "suggestion_accepted",
            "payload": {"score": 0.91},
            "ts": "2026-11-19T10:00:03.500+00:00",
        },
    ]
    response = client.post("/api/logs", json={"events": events})
    assert response.status_code == 201
    assert response.json() == {"inserted": 2}

    rows = db_session.scalars(select(InteractionLog).order_by(InteractionLog.ts)).all()
    assert [r.event for r in rows] == ["image_opened", "suggestion_accepted"]
    assert (rows[1].ts - rows[0].ts).total_seconds() == 3.5
    assert rows[1].payload == {"score": 0.91}


def test_logs_for_unknown_image_are_rejected(client: TestClient) -> None:
    event = {"image_id": 999999, "session_id": "s", "mode": "manual", "event": "image_opened"}
    response = client.post("/api/logs", json={"events": [event]})
    assert response.status_code == 422


def test_log_mode_must_be_manual_or_assisted(
    client: TestClient, project: dict[str, Any], upload: Upload
) -> None:
    image = upload(project["id"])
    event = {"image_id": image["id"], "session_id": "s", "mode": "auto", "event": "x"}
    assert client.post("/api/logs", json={"events": [event]}).status_code == 422
