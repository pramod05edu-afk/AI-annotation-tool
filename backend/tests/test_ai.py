"""AI endpoints, tested with small fake models.

The fakes implement the same interfaces as the real YOLO / MobileSAM wrappers, so these tests
check OUR logic (class mapping, clipping, error handling, request validation) quickly and
without PyTorch. The real models are checked by hand and in the evaluation (Phase 7).
"""

from collections.abc import Callable, Iterator, Sequence

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ml.base import Detection, Point, Segmentation
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
    assert body["suggestions"] == [
        {"class_name": "helmet", "bbox": [10, 5, 30, 20], "score": 0.91},
        {"class_name": "head", "bbox": [50, 30, 64, 48], "score": 0.75},
    ]
    assert body["ignored"] == {"person": 1}
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
    models.detector.state = ModelState.LOADING
    response = client.post(f"/api/images/{image_id}/suggest", json={})
    assert response.status_code == 503
    assert "still loading" in response.json()["detail"]


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
