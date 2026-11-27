"""AI endpoints, tested with small fake models.

The fakes implement the same interfaces as the real YOLO / YOLOE / MobileSAM wrappers, so
these tests check OUR logic (class mapping, model routing, clipping, error handling, request
validation) quickly and without PyTorch. The real models are measured by the evaluation scripts.
"""

from collections.abc import Callable, Iterator, Sequence

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ml.base import Box, Detection, Point, Segmentation
from app.ml.manager import ModelManager, ModelState, get_models


class FakeDetector:
    name = "fake-yolo"
    classes = ["Helmet", "head", "person"]

    def __init__(self) -> None:
        self.last_conf: float | None = None

    def detect(self, image: Image.Image, conf: float) -> list[Detection]:
        self.last_conf = conf
        return [
            Detection("Helmet", (10, 5, 30, 20), 0.91),  # capital H: mapped to "helmet"
            Detection("head", (50, 30, 80, 60), 0.75),  # sticks out of the 64x48 image: clipped
            Detection("person", (0, 0, 64, 48), 0.88),  # not a project class: ignored
            Detection("head", (70, 50, 90, 60), 0.6),  # completely outside: dropped
        ]


class FakeGeneral:
    """Stands in for YOLOE: one box per requested label, so tests see which labels it got."""

    name = "fake-yoloe"

    def __init__(self) -> None:
        self.asked: list[list[str]] = []
        self.examples: list[list[Box]] = []

    def detect_classes(
        self, image: Image.Image, classes: Sequence[str], conf: float
    ) -> list[Detection]:
        self.asked.append(list(classes))
        return [
            Detection(name, (2 + 10 * i, 30, 10 + 10 * i, 40), 0.5)
            for i, name in enumerate(classes)
        ]

    def find_similar(
        self, image: Image.Image, examples: Sequence[Box], conf: float
    ) -> list[Detection]:
        self.examples.append(list(examples))
        return [
            Detection("object", (10, 5, 30, 20), 0.9),  # the example itself: not suggested
            Detection("object", (40, 5, 60, 20), 0.7),  # a new one
            Detection("object", (41, 25, 61, 40), 0.6),  # duplicates an existing label
        ]


class FakeSegmenter:
    name = "fake-sam"

    def __init__(self) -> None:
        self.prepared: list[str] = []
        self.found = True

    def prepare(self, image_key: str, load_image: Callable[[], Image.Image]) -> None:
        load_image()
        self.prepared.append(image_key)

    def segment(
        self,
        image_key: str,
        load_image: Callable[[], Image.Image],
        points: Sequence[Point],
        labels: Sequence[int],
    ) -> Segmentation | None:
        if not self.found:
            return None
        square = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
        return Segmentation(polygon=square, bbox=(10, 10, 20, 20), score=0.9)


@pytest.fixture
def models(client: TestClient) -> Iterator[ModelManager]:
    manager = ModelManager()
    manager.detector.model, manager.detector.state = FakeDetector(), ModelState.READY
    manager.segmenter.model, manager.segmenter.state = FakeSegmenter(), ModelState.READY
    manager.general.model, manager.general.state = FakeGeneral(), ModelState.READY
    manager.detector.name, manager.general.name = "fake-yolo", "fake-yoloe"
    manager.detector.model_version_id, manager.general.model_version_id = 1, 2
    app.dependency_overrides[get_models] = lambda: manager
    yield manager


@pytest.fixture
def image_id(project: dict[str, object], upload: Callable[..., dict[str, object]]) -> int:
    image_id = upload(project["id"], width=64, height=48)["id"]
    assert isinstance(image_id, int)
    return image_id


def test_status_reports_model_states(client: TestClient, models: ModelManager) -> None:
    models.segmenter.state, models.segmenter.error = ModelState.FAILED, "weights missing"
    body = client.get("/api/ai/status").json()
    assert body["detector"]["state"] == "ready"
    assert body["general"]["state"] == "ready"
    assert body["segmenter"] == {
        "state": "failed",
        "name": None,
        "model_version_id": None,
        "error": "weights missing",
    }


def test_suggest_maps_classes_clips_and_ignores(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    response = client.post(f"/api/images/{image_id}/suggest", json={"conf": 0.4})
    assert response.status_code == 200, response.text
    body = response.json()
    spec = {"model": "fake-yolo", "model_version_id": 1}
    assert body["suggestions"] == [
        {"class_name": "helmet", "bbox": [10, 5, 30, 20], "score": 0.91, **spec},
        {"class_name": "head", "bbox": [50, 30, 64, 48], "score": 0.75, **spec},
    ]
    assert body["ignored"] == {"person": 1}
    # Both labels are known to the specialist: the general model is not needed.
    assert body["models"] == ["fake-yolo"]
    assert models.general.model.asked == []  # type: ignore[union-attr]
    assert models.detector.model.last_conf == 0.4  # type: ignore[union-attr]


def test_suggestions_can_be_saved_as_ai_annotations(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    suggestion = client.post(f"/api/images/{image_id}/suggest", json={}).json()["suggestions"][0]
    response = client.put(
        f"/api/images/{image_id}/annotations",
        json=[
            {
                "class_name": suggestion["class_name"],
                "geometry": {"type": "box", "bbox": suggestion["bbox"]},
                "source": "ai",
                "score": suggestion["score"],
            }
        ],
    )
    assert response.status_code == 200, response.text
    assert response.json()[0]["source"] == "ai"


@pytest.mark.parametrize("conf", [0, 1, -0.5])
def test_suggest_rejects_invalid_confidence(
    client: TestClient, models: ModelManager, image_id: int, conf: float
) -> None:
    response = client.post(f"/api/images/{image_id}/suggest", json={"conf": conf})
    assert response.status_code == 422


def test_suggest_while_loading_is_503(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    models.detector.state = models.general.state = ModelState.LOADING
    response = client.post(f"/api/images/{image_id}/suggest", json={})
    assert response.status_code == 503
    assert "still loading" in response.json()["detail"]


def test_suggest_uses_the_general_model_while_the_specialist_loads(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    models.detector.state = ModelState.LOADING
    body = client.post(f"/api/images/{image_id}/suggest", json={}).json()
    assert body["models"] == ["fake-yoloe"]
    assert models.general.model.asked == [["helmet", "head"]]  # type: ignore[union-attr]


def test_suggest_sends_unknown_labels_to_the_general_model(
    client: TestClient, models: ModelManager, upload: Callable[..., dict[str, object]]
) -> None:
    project = client.post(
        "/api/projects", json={"name": "Mixed", "classes": ["helmet", "Safety vest", "cone"]}
    ).json()
    image_id = upload(project["id"], width=64, height=48)["id"]
    body = client.post(f"/api/images/{image_id}/suggest", json={}).json()
    assert body["models"] == ["fake-yolo", "fake-yoloe"]
    # Only the labels the specialist does not know go to the general model, spelled as given.
    assert models.general.model.asked == [["Safety vest", "cone"]]  # type: ignore[union-attr]
    by_model = {(s["class_name"], s["model"], s["model_version_id"]) for s in body["suggestions"]}
    assert by_model == {
        ("helmet", "fake-yolo", 1),
        ("Safety vest", "fake-yoloe", 2),
        ("cone", "fake-yoloe", 2),
    }


def test_suggest_for_a_new_domain_uses_only_the_general_model(
    client: TestClient, models: ModelManager, upload: Callable[..., dict[str, object]]
) -> None:
    project = client.post("/api/projects", json={"name": "Pets", "classes": ["cat", "dog"]}).json()
    image_id = upload(project["id"], width=64, height=48)["id"]
    body = client.post(f"/api/images/{image_id}/suggest", json={}).json()
    assert body["models"] == ["fake-yoloe"]
    assert [s["class_name"] for s in body["suggestions"]] == ["cat", "dog"]
    assert body["ignored"] == {}
    assert models.detector.model.last_conf is None  # type: ignore[union-attr]


def test_find_similar_suggests_only_new_objects(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    response = client.post(
        f"/api/images/{image_id}/find-similar",
        json={
            "examples": [[10, 5, 30, 20]],
            "class_name": "helmet",
            "exclude": [[10, 5, 30, 20], [40, 25, 60, 40]],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["suggestions"] == [
        {
            "class_name": "helmet",
            "bbox": [40, 5, 60, 20],
            "score": 0.7,
            "model": "fake-yoloe",
            "model_version_id": 2,
        }
    ]
    assert models.general.model.examples == [[(10, 5, 30, 20)]]  # type: ignore[union-attr]


def test_find_similar_rejects_unknown_labels(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    response = client.post(
        f"/api/images/{image_id}/find-similar",
        json={"examples": [[10, 5, 30, 20]], "class_name": "giraffe"},
    )
    assert response.status_code == 422
    assert "not a label" in response.json()["detail"]


@pytest.mark.parametrize("examples", [[], [[1, 2, 3, 4]] * 11])
def test_find_similar_needs_one_to_ten_examples(
    client: TestClient, models: ModelManager, image_id: int, examples: list[list[int]]
) -> None:
    response = client.post(
        f"/api/images/{image_id}/find-similar", json={"examples": examples, "class_name": "helmet"}
    )
    assert response.status_code == 422


def test_find_similar_without_general_model_is_503(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    models.general.state = ModelState.FAILED
    models.general.error = "download failed"
    response = client.post(
        f"/api/images/{image_id}/find-similar",
        json={"examples": [[10, 5, 30, 20]], "class_name": "helmet"},
    )
    assert response.status_code == 503
    assert "download failed" in response.json()["detail"]


def test_suggest_unknown_image_is_404(client: TestClient, models: ModelManager) -> None:
    assert client.post("/api/images/999999/suggest", json={}).status_code == 404


def test_segment_returns_polygon(client: TestClient, models: ModelManager, image_id: int) -> None:
    response = client.post(
        f"/api/images/{image_id}/segment", json={"points": [[15, 15], [40, 40]], "labels": [1, 0]}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["points"] == [[10, 10], [20, 10], [20, 20], [10, 20]]
    assert body["bbox"] == [10, 10, 20, 20]


def test_segment_nothing_found_is_422(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    models.segmenter.model.found = False  # type: ignore[union-attr]
    response = client.post(
        f"/api/images/{image_id}/segment", json={"points": [[15, 15]], "labels": [1]}
    )
    assert response.status_code == 422
    assert "No object found" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"points": [[1, 1]], "labels": [1, 0]},  # label count does not match
        {"points": [[1, 1]], "labels": [0]},  # no positive click
        {"points": [], "labels": []},  # no clicks at all
        {"points": [[1, 1]], "labels": [2]},  # labels must be 0 or 1
    ],
)
def test_segment_validates_clicks(
    client: TestClient, models: ModelManager, image_id: int, body: dict[str, object]
) -> None:
    assert client.post(f"/api/images/{image_id}/segment", json=body).status_code == 422


def test_prepare_encodes_the_image(client: TestClient, models: ModelManager, image_id: int) -> None:
    response = client.post(f"/api/images/{image_id}/segment/prepare")
    assert response.status_code == 204
    assert models.segmenter.model.prepared == [str(image_id)]  # type: ignore[union-attr]


def test_click_box_returns_the_box_under_the_click(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    response = client.post(f"/api/images/{image_id}/click-box", json={"point": [20, 10]})
    assert response.status_code == 200, response.text
    assert response.json()["suggestion"] == {
        "class_name": "helmet",
        "bbox": [10, 5, 30, 20],
        "score": 0.91,
        "model": "fake-yolo",
        "model_version_id": 1,
    }
    # Smart click asks the detector for weak candidates too.
    assert models.detector.model.last_conf == 0.01  # type: ignore[union-attr]


def test_click_box_skips_labelled_objects_and_reports_nothing_found(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    response = client.post(
        f"/api/images/{image_id}/click-box",
        json={"point": [20, 10], "exclude": [[10, 5, 30, 20]]},
    )
    assert response.status_code == 200
    assert response.json()["suggestion"] is None  # "person" under the click is not a class


def test_click_box_caches_detections_per_image(
    client: TestClient, models: ModelManager, image_id: int
) -> None:
    detector = models.detector.model
    calls = 0
    original = detector.detect  # type: ignore[union-attr]

    def counting_detect(image: Image.Image, conf: float) -> list[Detection]:
        nonlocal calls
        calls += 1
        return original(image, conf)

    detector.detect = counting_detect  # type: ignore[union-attr, method-assign]
    for _ in range(3):
        client.post(f"/api/images/{image_id}/click-box", json={"point": [20, 10]})
    assert calls <= 1  # 0 if an earlier test already cached this image id
