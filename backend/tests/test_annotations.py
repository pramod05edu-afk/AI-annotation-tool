from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

Upload = Callable[..., dict[str, Any]]

BOX = {"class_name": "helmet", "geometry": {"type": "box", "bbox": [10, 20, 30, 40]}}
POLYGON = {
    "class_name": "head",
    "geometry": {"type": "polygon", "points": [[5, 5], [25, 5], [15, 30]]},
    "source": "ai_edited",
    "score": 0.87,
}


@pytest.fixture
def image(project: dict[str, Any], upload: Upload) -> dict[str, Any]:
    return upload(project["id"], width=100, height=50)


def test_save_and_load_box_and_polygon(client: TestClient, image: dict[str, Any]) -> None:
    saved = client.put(f"/api/images/{image['id']}/annotations", json=[BOX, POLYGON])
    assert saved.status_code == 200, saved.text

    loaded = client.get(f"/api/images/{image['id']}/annotations").json()
    assert [a["class_name"] for a in loaded] == ["helmet", "head"]
    assert loaded[0]["geometry"] == {"type": "box", "bbox": [10, 20, 30, 40]}
    assert loaded[0]["source"] == "manual"  # default
    assert loaded[1]["geometry"] == {"type": "polygon", "points": [[5, 5], [25, 5], [15, 30]]}
    assert (loaded[1]["source"], loaded[1]["score"]) == ("ai_edited", 0.87)


def test_saving_marks_image_in_progress(client: TestClient, image: dict[str, Any]) -> None:
    client.put(f"/api/images/{image['id']}/annotations", json=[BOX])
    assert client.get(f"/api/images/{image['id']}").json()["status"] == "in_progress"


def test_put_replaces_previous_annotations(client: TestClient, image: dict[str, Any]) -> None:
    client.put(f"/api/images/{image['id']}/annotations", json=[BOX, POLYGON])
    client.put(f"/api/images/{image['id']}/annotations", json=[POLYGON])
    assert len(client.get(f"/api/images/{image['id']}/annotations").json()) == 1

    client.put(f"/api/images/{image['id']}/annotations", json=[])
    assert client.get(f"/api/images/{image['id']}/annotations").json() == []


def test_unknown_class_is_rejected(client: TestClient, image: dict[str, Any]) -> None:
    bad = {**BOX, "class_name": "vest"}
    response = client.put(f"/api/images/{image['id']}/annotations", json=[BOX, bad])
    assert response.status_code == 422
    assert "Annotation 1" in response.json()["detail"]
    # Nothing was saved: the whole request is one transaction.
    assert client.get(f"/api/images/{image['id']}/annotations").json() == []


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "box", "bbox": [30, 20, 10, 40]},  # x_min > x_max
        {"type": "box", "bbox": [10, 20, 10, 40]},  # zero width
        {"type": "box", "bbox": [10, 20, 30]},  # wrong length
        {"type": "polygon", "points": [[1, 1], [2, 2]]},  # fewer than 3 points
        {"type": "circle", "center": [1, 1]},  # unknown type
    ],
)
def test_invalid_geometry_is_rejected(
    client: TestClient, image: dict[str, Any], geometry: dict[str, Any]
) -> None:
    item = {"class_name": "helmet", "geometry": geometry}
    assert client.put(f"/api/images/{image['id']}/annotations", json=[item]).status_code == 422


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "box", "bbox": [90, 10, 120, 20]},  # beyond the 100-px width
        {"type": "polygon", "points": [[5, 5], [25, 5], [15, 80]]},  # beyond the 50-px height
    ],
)
def test_geometry_outside_image_is_rejected(
    client: TestClient, image: dict[str, Any], geometry: dict[str, Any]
) -> None:
    item = {"class_name": "helmet", "geometry": geometry}
    response = client.put(f"/api/images/{image['id']}/annotations", json=[item])
    assert response.status_code == 422
    assert "100x50" in response.json()["detail"]


def test_tiny_rounding_past_the_edge_is_allowed(client: TestClient, image: dict[str, Any]) -> None:
    item = {"class_name": "helmet", "geometry": {"type": "box", "bbox": [-0.4, 0, 100.3, 50]}}
    assert client.put(f"/api/images/{image['id']}/annotations", json=[item]).status_code == 200


def test_model_version_reference_must_exist(client: TestClient, image: dict[str, Any]) -> None:
    unknown = {**BOX, "source": "ai", "score": 0.9, "model_version_id": 999999}
    assert client.put(f"/api/images/{image['id']}/annotations", json=[unknown]).status_code == 422

    model = client.post(
        "/api/models",
        json={"name": "yolo11s-coco", "task": "detect", "weights_uri": "yolo11s.pt"},
    ).json()
    known = {**unknown, "model_version_id": model["id"]}
    saved = client.put(f"/api/images/{image['id']}/annotations", json=[known])
    assert saved.status_code == 200
    assert saved.json()[0]["model_version_id"] == model["id"]
