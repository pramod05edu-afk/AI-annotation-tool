from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from tests.conftest import FakeStorage, make_image

Upload = Callable[..., dict[str, Any]]


def test_upload_accepts_images_and_rejects_others(
    client: TestClient, project: dict[str, Any], storage: FakeStorage
) -> None:
    files = [
        ("files", ("worker.png", make_image(100, 80, "PNG"), "image/png")),
        ("files", ("site.jpg", make_image(120, 90, "JPEG"), "image/jpeg")),
        ("files", ("notes.txt", b"definitely not an image", "text/plain")),
    ]
    response = client.post(f"/api/projects/{project['id']}/images", files=files)

    assert response.status_code == 201
    body = response.json()
    uploaded = {i["filename"]: i for i in body["uploaded"]}
    assert set(uploaded) == {"worker.png", "site.jpg"}
    assert (uploaded["worker.png"]["width"], uploaded["worker.png"]["height"]) == (100, 80)
    assert uploaded["site.jpg"]["content_type"] == "image/jpeg"
    assert uploaded["worker.png"]["status"] == "todo"
    assert body["rejected"] == [{"filename": "notes.txt", "reason": "not a readable image file"}]
    assert len(storage.objects) == 2


def test_file_type_comes_from_content_not_extension(
    client: TestClient, project: dict[str, Any]
) -> None:
    # A PNG renamed to .jpg is still stored and served as PNG.
    files = [("files", ("lying.jpg", make_image(), "image/jpeg"))]
    image = client.post(f"/api/projects/{project['id']}/images", files=files).json()["uploaded"][0]
    assert image["content_type"] == "image/png"


def test_folder_paths_are_stripped_from_filenames(
    client: TestClient, project: dict[str, Any]
) -> None:
    files = [("files", ("../../etc/evil.png", make_image(), "image/png"))]
    image = client.post(f"/api/projects/{project['id']}/images", files=files).json()["uploaded"][0]
    assert image["filename"] == "evil.png"


def test_too_large_file_is_rejected(
    client: TestClient, project: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    files = [("files", ("big.png", make_image(), "image/png"))]
    body = client.post(f"/api/projects/{project['id']}/images", files=files).json()
    assert body["uploaded"] == []
    assert "larger than" in body["rejected"][0]["reason"]


def test_list_images_with_status_filter_and_annotation_count(
    client: TestClient, project: dict[str, Any], upload: Upload
) -> None:
    first, second = upload(project["id"]), upload(project["id"])
    client.patch(f"/api/images/{second['id']}", json={"status": "done"})
    box = {"class_name": "helmet", "geometry": {"type": "box", "bbox": [1, 1, 5, 5]}}
    client.put(f"/api/images/{first['id']}/annotations", json=[box, box])

    all_images = client.get(f"/api/projects/{project['id']}/images").json()
    assert [i["id"] for i in all_images] == [first["id"], second["id"]]
    assert all_images[0]["annotation_count"] == 2

    done = client.get(f"/api/projects/{project['id']}/images", params={"status": "done"}).json()
    assert [i["id"] for i in done] == [second["id"]]


def test_image_file_is_streamed_back(
    client: TestClient, project: dict[str, Any], upload: Upload, storage: FakeStorage
) -> None:
    image = upload(project["id"])
    response = client.get(f"/api/images/{image['id']}/file")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == next(iter(storage.objects.values()))[0]


def test_missing_file_in_storage_is_404(
    client: TestClient, project: dict[str, Any], upload: Upload, storage: FakeStorage
) -> None:
    image = upload(project["id"])
    storage.objects.clear()
    assert client.get(f"/api/images/{image['id']}/file").status_code == 404


def test_invalid_status_is_rejected(
    client: TestClient, project: dict[str, Any], upload: Upload
) -> None:
    image = upload(project["id"])
    assert (
        client.patch(f"/api/images/{image['id']}", json={"status": "finished"}).status_code == 422
    )


def test_delete_image_removes_file(
    client: TestClient, project: dict[str, Any], upload: Upload, storage: FakeStorage
) -> None:
    image = upload(project["id"])
    assert client.delete(f"/api/images/{image['id']}").status_code == 204
    assert client.get(f"/api/images/{image['id']}").status_code == 404
    assert storage.objects == {}
