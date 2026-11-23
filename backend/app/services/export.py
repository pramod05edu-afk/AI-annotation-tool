"""Export a project as a standard dataset: COCO (one JSON file) or YOLO (one .txt per image).

Both exports are zip files that contain the images, so the dataset can be used directly for
training (e.g. with Ultralytics) or loaded by other tools (CVAT, FiftyOne, pycocotools).

    COCO zip                          YOLO zip
    ├── annotations.json              ├── data.yaml        class names
    └── images/                       ├── images/
                                      └── labels/          one .txt per image

Boxes: COCO uses [x, y, width, height] in pixels; YOLO uses "class x_center y_center width
height", normalized to 0..1. Polygons stay polygons in COCO ("segmentation"); YOLO detection
labels only hold boxes, so a polygon is exported as its bounding box.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePath
from typing import IO, Any

from app.models import Annotation, AnnotationType, Image, Project
from app.services.geometry import Box, voc_to_coco, voc_to_yolo


@dataclass
class ExportItem:
    image: Image
    annotations: list[Annotation] = field(default_factory=list)


def unique_file_names(images: Iterable[Image]) -> dict[int, str]:
    """Original file names where possible; prefix the id where two images share a name."""
    names: dict[int, str] = {}
    used: set[str] = set()
    for image in images:
        name = PurePath(image.filename).name or f"image_{image.id}"
        if name.lower() in used:
            name = f"{image.id}_{name}"
        used.add(name.lower())
        names[image.id] = name
    return names


def _bbox(annotation: Annotation) -> Box:
    x1, y1, x2, y2 = annotation.geometry["bbox"]
    return (float(x1), float(y1), float(x2), float(y2))


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace formula: area of a simple polygon from its corner points."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def build_coco(
    project: Project, items: list[ExportItem], file_names: dict[int, str]
) -> dict[str, Any]:
    """The COCO dictionary (category ids start at 1, as in the official COCO files)."""
    category_ids = {name: i for i, name in enumerate(project.classes, start=1)}
    images, annotations = [], []
    annotation_id = 1
    for item in items:
        image = item.image
        images.append(
            {
                "id": image.id,
                "file_name": file_names[image.id],
                "width": image.width,
                "height": image.height,
            }
        )
        for a in item.annotations:
            x, y, w, h = voc_to_coco(_bbox(a))
            entry: dict[str, Any] = {
                "id": annotation_id,
                "image_id": image.id,
                "category_id": category_ids[a.class_name],
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "segmentation": [],
                # Not part of the COCO standard, but allowed: where the label came from.
                "attributes": {"source": a.source.value, "score": a.score},
            }
            if a.type == AnnotationType.POLYGON:
                points = a.geometry["points"]
                entry["segmentation"] = [[round(v, 2) for point in points for v in point]]
                entry["area"] = round(polygon_area(points), 2)
            annotations.append(entry)
            annotation_id += 1
    return {
        "info": {
            "description": f"{project.name} - exported from the AI-assisted annotation tool",
            "date_created": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name, "supercategory": "object"}
            for name, cid in category_ids.items()
        ],
    }


def yolo_label_lines(item: ExportItem, class_ids: dict[str, int]) -> list[str]:
    """One "class cx cy w h" line per annotation, normalized to 0..1 (6 decimals)."""
    lines = []
    for a in item.annotations:
        cx, cy, w, h = voc_to_yolo(_bbox(a), item.image.width, item.image.height)
        lines.append(f"{class_ids[a.class_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def yolo_data_yaml(project: Project) -> str:
    names = "".join(f"  {i}: {name}\n" for i, name in enumerate(project.classes))
    return (
        f"# {project.name} - exported from the AI-assisted annotation tool\n"
        "# Every image is in one folder: split into train / val before training.\n"
        "train: images\nval: images\n"
        f"names:\n{names}"
    )


def write_zip(
    out: IO[bytes],
    project: Project,
    items: list[ExportItem],
    export_format: str,
    read_image: Callable[[Image], bytes],
) -> None:
    """Write the dataset zip into `out` (a file object)."""
    file_names = unique_file_names(item.image for item in items)
    # Images are already compressed (PNG/JPEG): storing them is as small and much faster.
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        if export_format == "coco":
            coco = build_coco(project, items, file_names)
            zf.writestr("annotations.json", json.dumps(coco, indent=1), zipfile.ZIP_DEFLATED)
        else:
            class_ids = {name: i for i, name in enumerate(project.classes)}
            zf.writestr("data.yaml", yolo_data_yaml(project))
            for item in items:
                stem = PurePath(file_names[item.image.id]).stem
                lines = yolo_label_lines(item, class_ids)
                # An empty file = "no objects": a valid background image for training.
                zf.writestr(f"labels/{stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
        for item in items:
            zf.writestr(f"images/{file_names[item.image.id]}", read_image(item.image))
