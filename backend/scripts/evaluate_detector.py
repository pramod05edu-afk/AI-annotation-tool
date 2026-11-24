"""Evaluate the deployed detector on the TEST split, on this laptop's CPU.

1. mAP of the exported OpenVINO model (Ultralytics val): must match the Colab result,
   otherwise the export lost accuracy.
2. Quality at the app's default confidence (0.25), as a user sees it: precision / recall / F1
   per class and recall by object size (small < 32x32 px, medium < 96x96 px, large).
3. Failure cases: the test images with the most errors, drawn into eval/results/failures/
   (green = correct, red = false alarm, orange = missed, magenta = wrong class).
4. CPU speed: OpenVINO vs. PyTorch, pure model time per image (pre + inference + post).

Usage (from backend/; run with nothing else busy on the computer, for fair timings):
    uv run python scripts/evaluate_detector.py --data D:/datasets/hardhat-prepared
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from ultralytics.models import YOLO

from app.services.evaluation import (
    Counts,
    Labeled,
    MatchResult,
    counts_of,
    match,
    read_yolo_labels,
    size_bucket,
)

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights"
OUT = REPO / "eval" / "results"
COLORS = {"tp": "#1a9f4b", "fp": "#e02424", "fn": "#ff8c00", "confused": "#d63fd6"}


def predict(model: YOLO, image: Image.Image, conf: float) -> tuple[list[Labeled], dict[str, float]]:
    result: Any = model.predict(image, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
    names = result.names
    preds = [
        Labeled(names[int(c)], (x1, y1, x2, y2), float(s))
        for (x1, y1, x2, y2), c, s in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            strict=True,
        )
    ]
    return preds, dict(result.speed)


def map_on_test(model_path: Path, data_yaml: Path) -> dict[str, float]:
    metrics: Any = YOLO(str(model_path), task="detect").val(
        data=str(data_yaml), split="test", imgsz=640, batch=1, device="cpu",
        plots=False, verbose=False, project=str(OUT / "_tmp"), name="val", exist_ok=True,
    )  # fmt: skip
    return {
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
    }


def draw_failures(image: Image.Image, result: MatchResult, path: Path) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    confused_preds = {id(p) for p, _ in result.class_confusions}
    for pred, _ in result.true_positives:
        draw.rectangle(pred.box, outline=COLORS["tp"], width=2)
    for pred in result.false_positives:
        color = COLORS["confused"] if id(pred) in confused_preds else COLORS["fp"]
        draw.rectangle(pred.box, outline=color, width=3)
        draw.text(
            (pred.box[0] + 2, pred.box[1] + 1), f"{pred.class_name} {pred.score:.2f}", fill=color
        )
    for truth in result.false_negatives:
        draw.rectangle(truth.box, outline=COLORS["fn"], width=3)
        draw.text(
            (truth.box[0] + 2, truth.box[3] - 11), f"missed {truth.class_name}", fill=COLORS["fn"]
        )
    canvas.save(path, quality=90)


def benchmark(model_path: Path, images: list[Image.Image], conf: float) -> dict[str, float]:
    model = YOLO(str(model_path), task="detect")
    for image in images[:5]:  # warm-up: first calls include one-time setup
        predict(model, image, conf)
    totals, inference = [], []
    for image in images:
        _, speed = predict(model, image, conf)
        totals.append(speed["preprocess"] + speed["inference"] + speed["postprocess"])
        inference.append(speed["inference"])
    totals.sort()
    return {
        "images": len(images),
        "total_ms_mean": round(statistics.mean(totals), 1),
        "total_ms_median": round(statistics.median(totals), 1),
        "total_ms_p95": round(totals[int(0.95 * (len(totals) - 1))], 1),
        "inference_ms_median": round(statistics.median(inference), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="prepared dataset (with test/)")
    parser.add_argument("--model", type=Path, default=WEIGHTS / "yolo11n_helmet_openvino_model")
    parser.add_argument(
        "--pt", type=Path, default=WEIGHTS / "yolo11n_helmet.pt", help="PyTorch weights"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="the app's default confidence")
    parser.add_argument("--bench", type=int, default=100, help="images for the speed benchmark")
    parser.add_argument("--failures", type=int, default=8, help="failure-case images to draw")
    parser.add_argument(
        "--limit", type=int, default=0, help="only the first N test images (testing)"
    )
    args = parser.parse_args()

    test_images = sorted((args.data / "test" / "images").glob("*.png"))
    if args.limit:
        test_images = test_images[: args.limit]
    model = YOLO(str(args.model), task="detect")
    classes = [str(n) for n in model.names.values()]
    report: dict[str, Any] = {
        "model": args.model.name,
        "conf": args.conf,
        "test_images": len(test_images),
    }

    # 1. mAP of the deployed (exported) model
    if not args.limit:
        print("1/4 mAP on the test split (Ultralytics val, CPU)…")
        report["map_openvino_cpu"] = map_on_test(args.model, args.data / "data.yaml")
        print("   ", report["map_openvino_cpu"])

    # 2. Quality at the app threshold, per image
    print(f"2/4 Matching predictions to ground truth on {len(test_images)} images…")
    total, per_class = Counts(), defaultdict(Counts)
    by_size: dict[str, Counter[str]] = defaultdict(Counter)  # size -> found / total
    confusions: Counter[str] = Counter()
    per_image = []
    for path in test_images:
        image = Image.open(path)
        truths = read_yolo_labels(
            (args.data / "test" / "labels" / f"{path.stem}.txt").read_text(), classes, *image.size
        )
        preds, _ = predict(model, image, args.conf)
        result = match(preds, truths)
        total.add(counts_of(result))
        for cls in classes:
            per_class[cls].add(
                counts_of(
                    match(
                        [p for p in preds if p.class_name == cls],
                        [t for t in truths if t.class_name == cls],
                    )
                )
            )
        for _, truth in result.true_positives:
            by_size[size_bucket(truth.box)]["found"] += 1
        for truth in truths:
            by_size[size_bucket(truth.box)]["total"] += 1
        for pred, truth in result.class_confusions:
            confusions[f"{truth.class_name} predicted as {pred.class_name}"] += 1
        errors = len(result.false_positives) + len(result.false_negatives)
        per_image.append((errors, path, result))

    report["at_app_threshold"] = {
        "all": total.as_dict(),
        **{cls: c.as_dict() for cls, c in per_class.items()},
    }
    report["recall_by_size"] = {
        size: {
            "objects": c["total"],
            "recall": round(c["found"] / c["total"], 4) if c["total"] else None,
        }
        for size in ("small", "medium", "large")
        if (c := by_size[size])
    }
    report["class_confusions"] = dict(confusions)
    print("    all:", total.as_dict())
    print("    recall by size:", report["recall_by_size"])

    # 3. Failure cases
    print(f"3/4 Drawing the {args.failures} images with the most errors…")
    failures_dir = OUT / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(per_image, key=lambda x: (-x[0], x[1].name))[: args.failures]
    report["failure_cases"] = []
    for errors, path, result in worst:
        draw_failures(Image.open(path), result, failures_dir / f"{path.stem}.jpg")
        report["failure_cases"].append(
            {
                "image": path.stem,
                "errors": errors,
                "false_positives": len(result.false_positives),
                "missed": len(result.false_negatives),
                "missed_small": sum(size_bucket(t.box) == "small" for t in result.false_negatives),
                "class_confusions": len(result.class_confusions),
            }
        )

    # 4. CPU speed: OpenVINO vs PyTorch
    print(f"4/4 CPU benchmark on {args.bench} images…")
    bench_images = [Image.open(p).convert("RGB") for p in test_images[: args.bench]]
    import torch  # noqa: PLC0415 - only to report the thread count

    report["cpu"] = {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()}
    report["speed"] = {"openvino": benchmark(args.model, bench_images, args.conf)}
    if args.pt.exists():
        report["speed"]["pytorch"] = benchmark(args.pt, bench_images, args.conf)
    else:
        print(f"    (no {args.pt.name}: PyTorch comparison skipped)")
    print("   ", report["speed"])

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / ("detector_eval_sample.json" if args.limit else "detector_eval.json")
    out_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_file} and {failures_dir}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Done in {time.perf_counter() - start:.0f} s")
