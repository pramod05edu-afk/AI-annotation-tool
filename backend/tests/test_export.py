"""Dataset export: round trips (export -> read back -> same coordinates) and format validity."""

import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pycocotools.coco import COCO

from app.services.export import polygon_area, unique_file_names
from app.services.geometry import yolo_to_voc

BOX = [10.0, 5.0, 40.0, 25.0]  # x_min, y_min, x_max, y_max on a 64 x 48 image
TRIANGLE = [[20.0, 30.0], [50.0, 30.0], [35.0, 45.0]]


@pytest.fixture
def labeled(
    client: TestClient, project: dict[str, object], upload: Callable[..., dict[str, object]]
) -> dict[str, int]:
    """One image with a box (helmet) and a polygon (head), one image without labels."""
    labeled_image = upload(project["id"], width=64, height=48, name="site.png")
    empty_image = upload(project["id"], width=64, height=48, name="empty.png")
    response = client.put(
        f"/api/images/{labeled_image['id']}/annotations",
        json=[
            {"class_name": "helmet", "geometry": {"type": "box", "bbox": BOX}, "source": "ai"},
            {"class_name": "head", "geometry": {"type": "polygon", "points": TRIANGLE}},
        ],
    )
    assert response.status_code == 200, response.text
    return {
        "project": int(project["id"]),  # type: ignore[call-overload]
        "labeled": int(labeled_image["id"]),  # type: ignore[call-overload]
        "empty": int(empty_image["id"]),  # type: ignore[call-overload]
    }


def download(client: TestClient, project_id: int, query: str) -> zipfile.ZipFile:
    response = client.get(f"/api/projects/{project_id}/export?{query}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers["content-disposition"]
    return zipfile.ZipFile(io.BytesIO(response.content))


def test_coco_round_trip(client: TestClient, labeled: dict[str, int], tmp_path: Path) -> None:
    zf = download(client, labeled["project"], "format=coco")
    assert sorted(zf.namelist()) == ["annotations.json", "images/empty.png", "images/site.png"]
    coco = json.loads(zf.read("annotations.json"))

    assert [c["name"] for c in coco["categories"]] == ["helmet", "head"]
    assert [c["id"] for c in coco["categories"]] == [1, 2]  # COCO ids start at 1
    box, polygon = coco["annotations"]
    assert box["bbox"] == [10, 5, 30, 20]  # [x, y, width, height]
    assert box["area"] == 600
    assert box["category_id"] == 1
    assert box["attributes"]["source"] == "ai"
    assert polygon["category_id"] == 2
    assert polygon["segmentation"] == [[20, 30, 50, 30, 35, 45]]
    assert polygon["bbox"] == [20, 30, 30, 15]
    assert polygon["area"] == 225  # triangle: 30 * 15 / 2

    # The official COCO API must be able to load and index the file.
    path = tmp_path / "annotations.json"
    path.write_bytes(zf.read("annotations.json"))
    api = COCO(str(path))
    assert len(api.getImgIds()) == 2
    assert len(api.getAnnIds(imgIds=[labeled["labeled"]])) == 2


def test_yolo_round_trip(client: TestClient, labeled: dict[str, int]) -> None:
    zf = download(client, labeled["project"], "format=yolo")
    names = set(zf.namelist())
    assert {"data.yaml", "images/site.png", "labels/site.txt", "labels/empty.txt"} <= names
    assert "0: helmet" in zf.read("data.yaml").decode()
    assert zf.read("labels/empty.txt") == b""  # unlabeled image = background example

    lines = zf.read("labels/site.txt").decode().splitlines()
    assert len(lines) == 2
    for line, (class_id, expected) in zip(
        lines, [(0, BOX), (1, [20.0, 30.0, 50.0, 45.0])], strict=True
    ):  # the polygon is exported as its bounding box
        cls, *values = line.split()
        assert int(cls) == class_id
        assert all(0 <= float(v) <= 1 for v in values)
        restored = yolo_to_voc(tuple(float(v) for v in values), 64, 48)  # type: ignore[arg-type]
        assert restored == pytest.approx(expected, abs=1e-3)


def test_scope_done_exports_only_done_images(client: TestClient, labeled: dict[str, int]) -> None:
    client.patch(f"/api/images/{labeled['labeled']}", json={"status": "done"})
    zf = download(client, labeled["project"], "format=coco&scope=done")
    coco = json.loads(zf.read("annotations.json"))
    assert [i["id"] for i in coco["images"]] == [labeled["labeled"]]


def test_scope_done_without_done_images_is_404(client: TestClient, labeled: dict[str, int]) -> None:
    response = client.get(f"/api/projects/{labeled['project']}/export?scope=done")
    assert response.status_code == 404


def test_unknown_format_is_422(client: TestClient, labeled: dict[str, int]) -> None:
    response = client.get(f"/api/projects/{labeled['project']}/export?format=voc")
    assert response.status_code == 422


def test_duplicate_file_names_get_unique_names() -> None:
    class FakeImage:
        def __init__(self, image_id: int, filename: str) -> None:
            self.id, self.filename = image_id, filename

    names = unique_file_names([FakeImage(1, "a.png"), FakeImage(2, "A.png"), FakeImage(3, "b.png")])  # type: ignore[list-item]
    assert names == {1: "a.png", 2: "2_A.png", 3: "b.png"}


def test_polygon_area_of_a_square() -> None:
    assert polygon_area([[0, 0], [10, 0], [10, 10], [0, 10]]) == 100
