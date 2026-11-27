"""Evaluate the general open-vocabulary detector (YOLOE): does the AI help with ANY labels?

Three questions, all answered on this laptop's CPU through the app's own model classes:

1. New domain, new labels: 80 everyday COCO classes the fine-tuned model never saw (coco128,
   https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip). The label
   names are the only input: zero-shot. Compared with YOLO-World v2 (the runner-up).
2. Specialist vs. generalist on the helmet TEST split: what fine-tuning buys.
3. Find similar (visual prompt) on the helmet test split: one example box per image and class,
   the model looks for the OTHER objects of that class. Shows whether an example fixes what a
   label name cannot express (here a "helmet" box covers the whole head, not only the hat).

Metrics: AP50 per class (mean over classes with ground truth), precision / recall / F1 at the
app's default confidence 0.25, and the median / p95 model time per image.

Usage (from backend/; run with nothing else busy on the computer, for fair timings):
    uv run python scripts/evaluate_open_vocab.py --coco <coco128 folder> \
        --helmet D:/datasets/hardhat-prepared
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from app.ml.base import Detection
from app.ml.manager import _prepare_text_encoder
from app.ml.postprocess import drop_overlapping
from app.ml.yolo import YoloDetector
from app.ml.yoloe import YoloeDetector
from app.services.evaluation import (
    Counts,
    Labeled,
    average_precision,
    counts_of,
    match,
    read_yolo_labels,
)

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights"
OUT = REPO / "eval" / "results" / "open_vocab_eval.json"
APP_CONF = 0.25  # the app's default "min. confidence"
LOW_CONF = 0.01  # for AP: the whole precision-recall curve
HELMET_CLASSES = ["helmet", "head"]
# COCO's 80 class names in label-index order (from Ultralytics' coco128.yaml).
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]  # fmt: skip

Detect = Callable[[Image.Image, float], list[Detection]]


def load_split(
    images_dir: Path, labels_dir: Path, classes: list[str]
) -> list[tuple[Path, list[Labeled]]]:
    items = []
    for image_path in sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ):
        with Image.open(image_path) as im:
            width, height = im.size
        label_file = labels_dir / f"{image_path.stem}.txt"
        text = label_file.read_text() if label_file.exists() else ""
        items.append((image_path, read_yolo_labels(text, classes, width, height)))
    return items


def to_labeled(detections: list[Detection]) -> list[Labeled]:
    return [Labeled(d.class_name, d.bbox, d.score) for d in detections]


def evaluate(detect: Detect, split: list[tuple[Path, list[Labeled]]]) -> dict[str, Any]:
    """AP50 from one low-confidence run; P/R/F1 at the app default from the same boxes."""
    predictions: dict[str, list[Labeled]] = {}
    truths: dict[str, list[Labeled]] = {}
    times: list[float] = []
    at_app_conf = Counts()
    for image_path, image_truths in split:
        image = Image.open(image_path).convert("RGB")
        start = time.perf_counter()
        preds = to_labeled(detect(image, LOW_CONF))
        times.append((time.perf_counter() - start) * 1000)
        predictions[image_path.name], truths[image_path.name] = preds, image_truths
        confident = [p for p in preds if p.score >= APP_CONF]
        at_app_conf.add(counts_of(match(confident, image_truths)))
    ap = average_precision(predictions, truths)
    ranked = sorted(ap.items(), key=lambda kv: kv[1])
    warm = times[3:] or times  # the first calls include warm-up
    return {
        "images": len(split),
        "objects": sum(len(t) for t in truths.values()),
        "classes_with_ground_truth": len(ap),
        "map50": round(statistics.mean(ap.values()), 4) if ap else 0.0,
        "ap50_per_class": {k: round(v, 4) for k, v in sorted(ap.items())},
        "weakest_classes": [(k, round(v, 3)) for k, v in ranked[:5]],
        "strongest_classes": [(k, round(v, 3)) for k, v in ranked[::-1][:5]],
        f"at_conf_{APP_CONF}": at_app_conf.as_dict(),
        "median_ms": round(statistics.median(warm), 1),
        "p95_ms": round(sorted(warm)[int(0.95 * (len(warm) - 1))], 1),
    }


def evaluate_find_similar(
    general: YoloeDetector, split: list[tuple[Path, list[Labeled]]]
) -> dict[str, Any]:
    """Per image and class with >= 2 objects: the largest one is the example, the others are
    what find-similar should return. Same matching as everywhere else (IoU >= 0.5)."""
    per_class: dict[str, Counts] = {}
    times: list[float] = []
    for image_path, image_truths in split:
        image: Image.Image | None = None
        for class_name in sorted({t.class_name for t in image_truths}):
            objects = [t for t in image_truths if t.class_name == class_name]
            if len(objects) < 2:
                continue
            example = max(objects, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
            targets = [t for t in objects if t is not example]
            image = image or Image.open(image_path).convert("RGB")
            start = time.perf_counter()
            found = general.find_similar(image, [example.box], conf=APP_CONF)
            times.append((time.perf_counter() - start) * 1000)
            # As in the app: what overlaps the example is the example itself, not a new object.
            new = drop_overlapping(found, [example.box])
            preds = [Labeled(class_name, d.bbox, d.score) for d in new]
            per_class.setdefault(class_name, Counts()).add(counts_of(match(preds, targets)))
    total = Counts()
    for counts in per_class.values():
        total.add(counts)
    warm = times[3:] or times
    return {
        "prompts": len(times),
        "per_class": {k: v.as_dict() for k, v in sorted(per_class.items())},
        "all": total.as_dict(),
        "median_ms": round(statistics.median(warm), 1) if warm else None,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coco", type=Path, required=True, help="coco128 folder")
    parser.add_argument("--helmet", type=Path, required=True, help="prepared helmet dataset")
    parser.add_argument("--no-world", action="store_true", help="skip the YOLO-World baseline")
    args = parser.parse_args()

    coco = load_split(
        args.coco / "images" / "train2017", args.coco / "labels" / "train2017", COCO_CLASSES
    )
    helmet = load_split(
        args.helmet / "test" / "images", args.helmet / "test" / "labels", HELMET_CLASSES
    )
    print(f"coco128: {len(coco)} images · helmet test: {len(helmet)} images", flush=True)

    _prepare_text_encoder(WEIGHTS, "mobileclip2_b.ts")
    general = YoloeDetector(WEIGHTS / "yoloe-26s-seg.pt")
    general.warm_up()
    specialist = YoloDetector(WEIGHTS / "yolo11n_helmet_openvino_model")
    specialist.warm_up()

    results: dict[str, Any] = {
        "settings": {"app_conf": APP_CONF, "low_conf_for_ap": LOW_CONF, "iou": 0.5, "imgsz": 640},
        "models": {"general": general.name, "specialist": specialist.name},
    }

    def step(name: str, run: Callable[[], dict[str, Any]]) -> None:
        start = time.perf_counter()
        results[name] = run()
        summary = {k: v for k, v in results[name].items() if not isinstance(v, dict | list)}
        print(f"{name} ({time.perf_counter() - start:.0f} s): {summary}", flush=True)

    step(
        "coco128_yoloe",
        lambda: evaluate(lambda im, c: general.detect_classes(im, COCO_CLASSES, c), coco),
    )
    step("helmet_specialist", lambda: evaluate(specialist.detect, helmet))
    step(
        "helmet_yoloe_text",
        lambda: evaluate(lambda im, c: general.detect_classes(im, HELMET_CLASSES, c), helmet),
    )
    step("helmet_yoloe_find_similar", lambda: evaluate_find_similar(general, helmet))

    if not args.no_world:
        from ultralytics.models import YOLOWorld

        world = YOLOWorld(str(WEIGHTS / "yolov8s-worldv2.pt"))

        def world_detect(classes: list[str]) -> Detect:
            world.set_classes(classes)

            def detect(image: Image.Image, conf: float) -> list[Detection]:
                results: Any = world.predict(
                    image, conf=conf, imgsz=640, device="cpu", verbose=False
                )
                result = results[0]
                return [
                    Detection(classes[int(c)], (b[0], b[1], b[2], b[3]), float(s))
                    for b, c, s in zip(
                        result.boxes.xyxy.tolist(),
                        result.boxes.cls.tolist(),
                        result.boxes.conf.tolist(),
                        strict=True,
                    )
                ]

            return detect

        step("coco128_yolo_world", lambda: evaluate(world_detect(COCO_CLASSES), coco))
        step("helmet_yolo_world_text", lambda: evaluate(world_detect(HELMET_CLASSES), helmet))

    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
