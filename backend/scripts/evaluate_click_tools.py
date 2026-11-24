"""Which click tool turns a click into the right box? MobileSAM vs. detector-guided smart click.

For a random sample of ground-truth objects from the TEST split, simulate a user clicking
the object, and compare the resulting box with the ground-truth box (IoU). The dataset has
boxes, not masks, so SAM is judged by the bounding box of its outline. Methods:

  sam_1_click    MobileSAM, one click in the centre of the object
  sam_2_clicks   MobileSAM, clicks in the upper and the lower third. In this dataset a
                 "helmet" box covers the whole head; one click on the hard hat makes SAM
                 outline only the hat, a second click on the face adds the face.
  smart_click    the fine-tuned detector's most confident box (conf >= 0.01) that contains
                 the click, as in the app (POST /api/images/{id}/click-box)

Also reported: for objects the automatic suggestions MISS (conf 0.25), how many smart click
recovers with one click; and the latencies.

Usage (from backend/, nothing else busy on the computer):
    uv run python scripts/evaluate_click_tools.py --data D:/datasets/hardhat-prepared
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from app.ml.postprocess import pick_box_at_point
from app.ml.sam import MobileSamSegmenter
from app.ml.yolo import YoloDetector
from app.services.evaluation import Labeled, read_yolo_labels, size_bucket
from app.services.geometry import iou

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights"
CLASSES = ["helmet", "head"]
METHODS = ["sam_1_click", "sam_2_clicks", "smart_click"]
APP_CONF = 0.25  # the app's automatic suggestions
CLICK_CONF = 0.01  # smart click


def summary(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean_iou": round(statistics.mean(values), 4),
        "median_iou": round(statistics.median(values), 4),
        "share_iou_50": round(sum(v >= 0.5 for v in values) / len(values), 4),
        "share_iou_75": round(sum(v >= 0.75 for v in values) / len(values), 4),
    }


def p95(values: list[float]) -> float:
    return round(sorted(values)[int(0.95 * (len(values) - 1))], 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--objects", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    objects: list[tuple[str, Labeled]] = []
    for labels in sorted((args.data / "test" / "labels").glob("*.txt")):
        size = Image.open(args.data / "test" / "images" / f"{labels.stem}.png").size
        objects += [(labels.stem, t) for t in read_yolo_labels(labels.read_text(), CLASSES, *size)]
    sample = random.Random(args.seed).sample(objects, min(args.objects, len(objects)))
    by_image: dict[str, list[Labeled]] = defaultdict(list)
    for name, truth in sample:
        by_image[name].append(truth)

    segmenter = MobileSamSegmenter(WEIGHTS / "mobile_sam.pt")
    segmenter.warm_up()
    detector = YoloDetector(WEIGHTS / "yolo11n_helmet_openvino_model")
    detector.warm_up()

    ious: dict[str, list[float]] = defaultdict(list)
    groups: dict[str, list[float]] = defaultdict(list)
    smart_class_ok: list[bool] = []
    missed_by_suggest = recovered = 0
    timing: dict[str, list[float]] = defaultdict(list)

    for i, (name, truths) in enumerate(by_image.items(), 1):
        image = Image.open(args.data / "test" / "images" / f"{name}.png").convert("RGB")
        start = time.perf_counter()
        segmenter.prepare(name, lambda image=image: image)
        timing["sam_encode_ms"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        detections = detector.detect(image, conf=CLICK_CONF)
        timing["detector_ms"].append((time.perf_counter() - start) * 1000)

        for truth in truths:
            x1, y1, x2, y2 = truth.box
            cx, h = (x1 + x2) / 2, y2 - y1
            results: dict[str, Any] = {}
            for method, points in {
                "sam_1_click": [(cx, y1 + h / 2)],
                "sam_2_clicks": [(cx, y1 + h / 3), (cx, y1 + 2 * h / 3)],
            }.items():
                start = time.perf_counter()
                seg = segmenter.segment(name, lambda image=image: image, points, [1] * len(points))
                timing["sam_click_ms"].append((time.perf_counter() - start) * 1000)
                results[method] = seg.bbox if seg else None
            pick = pick_box_at_point(detections, (cx, y1 + h / 2))
            results["smart_click"] = pick.bbox if pick else None
            smart_class_ok.append(pick is not None and pick.class_name == truth.class_name)

            for method in METHODS:
                overlap = iou(results[method], truth.box) if results[method] else 0.0
                ious[method].append(overlap)
                groups[f"{method}/{truth.class_name}"].append(overlap)
                groups[f"{method}/{size_bucket(truth.box)}"].append(overlap)

            # Would the automatic suggestions have found this object already?
            found_by_suggest = any(
                d.score >= APP_CONF
                and d.class_name == truth.class_name
                and iou(d.bbox, truth.box) >= 0.5
                for d in detections
            )
            if not found_by_suggest:
                missed_by_suggest += 1
                recovered += ious["smart_click"][-1] >= 0.5 and smart_class_ok[-1]
        if i % 25 == 0:
            print(f"  {i}/{len(by_image)} images")

    report = {
        "method": "simulated click(s) per truth box; IoU of the resulting box vs. the truth box",
        "objects": len(sample),
        "images": len(by_image),
        **{method: summary(ious[method]) for method in METHODS},
        "smart_click_correct_class": round(sum(smart_class_ok) / len(smart_class_ok), 4),
        "objects_missed_by_auto_suggest": missed_by_suggest,
        "recovered_by_one_smart_click": recovered,
        "by_group": {key: summary(values) for key, values in sorted(groups.items())},
        "latency_ms": {
            "sam_encode_median": round(statistics.median(timing["sam_encode_ms"]), 1),
            "sam_click_median": round(statistics.median(timing["sam_click_ms"]), 1),
            "sam_click_p95": p95(timing["sam_click_ms"]),
            "detector_per_image_median": round(statistics.median(timing["detector_ms"]), 1),
            "detector_per_image_p95": p95(timing["detector_ms"]),
        },
    }
    out = REPO / "eval" / "results" / "click_tools_eval.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for method in METHODS:
        print(f"{method:13s} {report[method]}")
    print(f"smart click: correct class {report['smart_click_correct_class']}")
    print(f"smart click: recovered {recovered} of {missed_by_suggest} objects auto-suggest missed")
    print("latency", report["latency_ms"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
