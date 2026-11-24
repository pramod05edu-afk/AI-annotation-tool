from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import FakeStorage

Upload = Callable[..., dict[str, Any]]


def test_create_and_get_project(client: TestClient, project: dict[str, Any]) -> None:
    assert project["name"] == "Hard hats"
    assert project["classes"] == ["helmet", "head"]
    assert project["image_count"] == 0

    response = client.get(f"/api/projects/{project['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == project["id"]


def test_list_projects_includes_status_counts(
    client: TestClient, project: dict[str, Any], upload: Upload
) -> None:
    first = upload(project["id"])
    upload(project["id"])
    client.patch(f"/api/images/{first['id']}", json={"status": "done"})

    projects = client.get("/api/projects").json()
    assert len(projects) == 1
    assert projects[0]["image_count"] == 2
    assert projects[0]["status_counts"] == {"todo": 1, "done": 1}


def test_duplicate_project_name_is_rejected(client: TestClient, project: dict[str, Any]) -> None:
    response = client.post("/api/projects", json={"name": "Hard hats", "classes": ["x"]})
    assert response.status_code == 409


def test_project_needs_unique_non_empty_classes(client: TestClient) -> None:
    assert client.post("/api/projects", json={"name": "A", "classes": []}).status_code == 422
    assert (
        client.post("/api/projects", json={"name": "B", "classes": ["a", "a"]}).status_code == 422
    )
    assert client.post("/api/projects", json={"name": " ", "classes": ["a"]}).status_code == 422


def test_rename_and_add_class(client: TestClient, project: dict[str, Any]) -> None:
    response = client.patch(
        f"/api/projects/{project['id']}",
        json={"name": "PPE", "classes": ["helmet", "head", "vest"]},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "PPE"
    assert response.json()["classes"] == ["helmet", "head", "vest"]


def test_cannot_remove_class_still_in_use(
    client: TestClient, project: dict[str, Any], upload: Upload
) -> None:
    image = upload(project["id"])
    box = {"class_name": "head", "geometry": {"type": "box", "bbox": [1, 1, 10, 10]}}
    client.put(f"/api/images/{image['id']}/annotations", json=[box])

    in_use = client.patch(f"/api/projects/{project['id']}", json={"classes": ["helmet"]})
    assert in_use.status_code == 409
    assert "head" in in_use.json()["detail"]

    unused = client.patch(f"/api/projects/{project['id']}", json={"classes": ["head"]})
    assert unused.status_code == 200


def test_delete_project_removes_images_and_files(
    client: TestClient, project: dict[str, Any], upload: Upload, storage: FakeStorage
) -> None:
    image = upload(project["id"])
    assert len(storage.objects) == 1

    assert client.delete(f"/api/projects/{project['id']}").status_code == 204
    assert client.get(f"/api/projects/{project['id']}").status_code == 404
    assert client.get(f"/api/images/{image['id']}").status_code == 404
    assert storage.objects == {}


def test_unknown_project_is_404(client: TestClient) -> None:
    assert client.get("/api/projects/999999").status_code == 404
    assert client.delete("/api/projects/999999").status_code == 404


def test_annotation_mode_can_be_fixed_and_unlocked(client: TestClient) -> None:
    created = client.post(
        "/api/projects",
        json={"name": "Study", "classes": ["helmet"], "annotation_mode": "manual"},
    ).json()
    assert created["annotation_mode"] == "manual"

    # Renaming without mentioning the mode keeps it.
    renamed = client.patch(f"/api/projects/{created['id']}", json={"name": "Study 2"}).json()
    assert renamed["annotation_mode"] == "manual"

    # An explicit null unlocks it.
    unlocked = client.patch(f"/api/projects/{created['id']}", json={"annotation_mode": None})
    assert unlocked.json()["annotation_mode"] is None


def test_annotation_mode_rejects_unknown_values(client: TestClient) -> None:
    response = client.post(
        "/api/projects", json={"name": "Bad", "classes": ["a"], "annotation_mode": "auto"}
    )
    assert response.status_code == 422
