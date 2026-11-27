"""Quality of the delivered dataset: the reviewed labels vs. the AI's labels alone.

The 100 "export" images (never used in training) were labelled in the app in assisted mode
and reviewed by a human. Both the final labels and the detector's own output (at the app's
default confidence 0.25) are compared with the public dataset's original labels
(precision / recall / F1 at IoU >= 0.5). The difference shows what the human review added.

Usage (from backend/, with the app's database running):
    uv run python scripts/evaluate_final_dataset.py --data D:/datasets/hardhat-prepared
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image as PilImage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml.yolo import YoloDetector
from app.models import Annotation, Image, Project
from app.services.evaluation import Counts, Labeled, counts_of, match, read_yolo_labels

REPO = Path(__file__).resolve().parents[2]
PROJECT = "Final dataset (helmet compliance)"
CLASSES = ["helmet", "head"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="prepared dataset (export/)")
    parser.add_argument("--conf", type=float, default=0.25, help="the app's default confidence")
    args = parser.parse_args()
    split = args.data / "export"

    detector = YoloDetector(REPO / "weights" / "yolo11n_helmet_openvino_model")
    final, ai_only = Counts(), Counts()
    sources: Counter[str] = Counter()
    with Session(create_engine(get_settings().database_url)) as db:
        project = db.scalar(select(Project).where(Project.name == PROJECT))
        if project is None:
            raise SystemExit(f"Project '{PROJECT}' not found")
        images = list(db.scalars(select(Image).where(Image.project_id == project.id)))
        for image in images:
            truths = read_yolo_labels(
                (split / "labels" / f"{Path(image.filename).stem}.txt").read_text(),
                CLASSES,
                image.width,
                image.height,
            )
            annotations = list(
                db.scalars(select(Annotation).where(Annotation.image_id == image.id))
            )
            sources.update(a.source.value for a in annotations)
            labels = [Labeled(a.class_name, tuple(a.geometry["bbox"])) for a in annotations]
            final.add(counts_of(match(labels, truths)))
            detections = detector.detect(
                PilImage.open(split / "images" / image.filename).convert("RGB"), conf=args.conf
            )
            predictions = [Labeled(d.class_name, d.bbox, d.score) for d in detections]
            ai_only.add(counts_of(match(predictions, truths)))

    report = {
        "images": len(images),
        "labels": sum(sources.values()),
        "labels_by_source": dict(sources),
        "vs_original_labels_iou_50": {
            "ai_only_conf_025": ai_only.as_dict(),
            "final_reviewed": final.as_dict(),
        },
    }
    out = REPO / "eval" / "results" / "final_dataset_quality.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
