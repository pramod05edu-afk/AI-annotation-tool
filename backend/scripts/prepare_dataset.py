"""Prepare the Safety Helmet dataset for training, evaluation and the effort study.

Steps:
  1. Parse the Pascal VOC labels; keep the classes `helmet` and `head`
     (`person` is dropped: it is labeled in only 158 of 5,000 images).
  2. Group near-duplicate images with a perceptual hash (dHash), so copies of the
     same photo never end up in both the training and the test set (data leakage).
  3. Split the groups reproducibly (fixed seed) into train / val / test / export.
  4. Write each split as YOLO (.txt labels + data.yaml) and COCO (one JSON per split),
     plus eval/splits.json (the file lists, committed to git for reproducibility).

Usage (from the backend/ folder):
    uv run python scripts/prepare_dataset.py --src D:/datasets/hard-hat-detection \
        --out D:/datasets/hardhat-prepared            # add --dry-run to only print statistics
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.services.geometry import Box, voc_to_coco, voc_to_yolo

CLASSES = ["helmet", "head"]
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}
SPLIT_ORDER = ("export", "test", "val")  # filled first, in this order; the rest is train
ALL_SPLITS = ("train", "val", "test", "export")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Sample:
    name: str  # file stem, e.g. "hard_hat_workers0"
    width: int
    height: int
    boxes: list[tuple[int, Box]]  # (class id, VOC box in pixels)
    hash: int = 0


# ---------- 1. Labels ----------


def parse_voc(xml_path: Path) -> tuple[str, list[tuple[int, Box]], Counter[str]]:
    """Return (file stem, kept boxes, count of dropped boxes by reason)."""
    root = ET.parse(xml_path).getroot()
    dropped: Counter[str] = Counter()
    boxes: list[tuple[int, Box]] = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_IDS:
            dropped[f"class:{name}"] += 1
            continue
        bb = obj.find("bndbox")
        assert bb is not None, f"{xml_path.name}: object without bndbox"
        x1, y1, x2, y2 = (float(bb.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((CLASS_IDS[name], (x1, y1, x2, y2)))
    return xml_path.stem, boxes, dropped


def clip_boxes(boxes: list[tuple[int, Box]], w: int, h: int) -> list[tuple[int, Box]]:
    """Clip boxes to the real image size and drop boxes smaller than 1 pixel."""
    kept = []
    for cls, (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            kept.append((cls, (x1, y1, x2, y2)))
    return kept


# ---------- 2. Near-duplicate grouping ----------


def dhash(image: Image.Image, size: int = 16) -> int:
    """Difference hash: shrink to (size+1) x size grayscale, compare each pixel with its
    right neighbour -> size*size bits. Similar images give hashes with few differing bits."""
    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = small.tobytes()
    bits = 0
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (px[offset + col] > px[offset + col + 1])
    return bits


def group_near_duplicates(samples: list[Sample], max_distance: int) -> list[list[Sample]]:
    """Union-find over all pairs whose hashes differ in at most `max_distance` bits."""
    parent = list(range(len(samples)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [s.hash for s in samples]
    for i in range(len(hashes)):
        hi = hashes[i]
        for j in range(i + 1, len(hashes)):
            if (hi ^ hashes[j]).bit_count() <= max_distance:
                parent[find(i)] = find(j)

    groups: dict[int, list[Sample]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(find(i), []).append(s)
    return sorted(groups.values(), key=lambda g: g[0].name)


# ---------- 3. Splitting ----------


def split_groups(
    groups: list[list[Sample]], targets: dict[str, int], seed: int
) -> dict[str, list[Sample]]:
    """Shuffle groups with a fixed seed; fill export, test, val up to their targets; rest -> train.
    A whole group always goes to one split."""
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    splits: dict[str, list[Sample]] = {name: [] for name in ALL_SPLITS}
    for group in shuffled:
        target = next((s for s in SPLIT_ORDER if len(splits[s]) < targets[s]), "train")
        splits[target].extend(group)
    for samples in splits.values():
        samples.sort(key=lambda s: s.name)
    return splits


def pick_effort_study(test: list[Sample], n: int, seed: int) -> list[str]:
    """Choose test images with 3-15 objects: enough work to time, not overwhelming."""
    candidates = [s.name for s in test if 3 <= len(s.boxes) <= 15]
    return sorted(random.Random(seed).sample(candidates, min(n, len(candidates))))


# ---------- 4. Writing ----------


def to_coco(samples: list[Sample]) -> dict[str, object]:
    images, annotations = [], []
    ann_id = 1
    for img_id, s in enumerate(samples, start=1):
        images.append(
            {"id": img_id, "file_name": f"{s.name}.png", "width": s.width, "height": s.height}
        )
        for cls, box in s.boxes:
            x, y, w, h = voc_to_coco(box)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,  # COCO category ids start at 1
                    "bbox": [round(v, 2) for v in (x, y, w, h)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    categories = [{"id": i + 1, "name": n, "supercategory": "ppe"} for i, n in enumerate(CLASSES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def write_outputs(
    src: Path,
    out: Path,
    splits: dict[str, list[Sample]],
    effort: list[str],
    meta: dict[str, object],
) -> None:
    for split, samples in splits.items():
        img_dir, lbl_dir = out / split / "images", out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for s in samples:
            shutil.copy2(src / "images" / f"{s.name}.png", img_dir / f"{s.name}.png")
            lines = [
                f"{cls} " + " ".join(f"{v:.6f}" for v in voc_to_yolo(box, s.width, s.height))
                for cls, box in s.boxes
            ]
            (lbl_dir / f"{s.name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        coco_path = out / split / "annotations_coco.json"
        coco_path.write_text(json.dumps(to_coco(samples)))

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"# Generated by prepare_dataset.py\npath: {out.as_posix()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\nnames:\n{names}\n"
    )

    splits_file = REPO_ROOT / "eval" / "splits.json"
    splits_file.parent.mkdir(exist_ok=True)
    splits_file.write_text(
        json.dumps(
            {
                **meta,
                "splits": {k: [s.name for s in v] for k, v in splits.items()},
                "effort_study": effort,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nWrote {out} and {splits_file.relative_to(REPO_ROOT)}")


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src", type=Path, required=True, help="folder with images/ and annotations/"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output folder (must be new or empty)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=600)
    parser.add_argument("--export", type=int, default=100)
    parser.add_argument("--effort", type=int, default=60, help="test images for the timing study")
    parser.add_argument("--max-hash-distance", type=int, default=10, help="of 256 dHash bits")
    parser.add_argument("--dry-run", action="store_true", help="print statistics, write nothing")
    args = parser.parse_args()

    if not args.dry_run and args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"Output folder {args.out} is not empty — choose a new folder or empty it first.")

    xml_files = sorted((args.src / "annotations").glob("*.xml"))
    print(f"Reading {len(xml_files)} label files and hashing images ...")
    samples: list[Sample] = []
    dropped: Counter[str] = Counter()
    size_mismatch = 0
    for xml_path in xml_files:
        name, boxes, dropped_here = parse_voc(xml_path)
        dropped.update(dropped_here)
        with Image.open(args.src / "images" / f"{name}.png") as im:
            w, h = im.size
            image_hash = dhash(im)
        root = ET.parse(xml_path).getroot()
        if (int(root.findtext("size/width") or 0), int(root.findtext("size/height") or 0)) != (
            w,
            h,
        ):
            size_mismatch += 1
        kept = clip_boxes(boxes, w, h)
        dropped["too_small_after_clip"] += len(boxes) - len(kept)
        samples.append(Sample(name, w, h, kept, image_hash))

    groups = group_near_duplicates(samples, args.max_hash_distance)
    multi = [g for g in groups if len(g) > 1]
    print(f"Dropped boxes: {dict(dropped)}  |  XML/real size mismatches: {size_mismatch}")
    print(
        f"Near-duplicate groups (distance <= {args.max_hash_distance}): {len(multi)} groups, "
        f"{sum(len(g) for g in multi)} images  |  e.g. {[[s.name for s in g] for g in multi[:3]]}"
    )

    targets = {"export": args.export, "test": args.test, "val": args.val}
    splits = split_groups(groups, targets, args.seed)
    effort = pick_effort_study(splits["test"], args.effort, args.seed)

    header = ("split", "images", "helmet", "head", "obj/img", "no-label")
    print("\n{:8s} {:>7s} {:>8s} {:>7s} {:>8s} {:>9s}".format(*header))
    for split in ALL_SPLITS:
        ss = splits[split]
        counts = Counter(cls for s in ss for cls, _ in s.boxes)
        n_obj = sum(counts.values())
        empty = sum(1 for s in ss if not s.boxes)
        print(
            f"{split:8s} {len(ss):7d} {counts[0]:8d} {counts[1]:7d} "
            f"{n_obj / max(len(ss), 1):8.1f} {empty:9d}"
        )
    print(f"effort-study images (from test): {len(effort)}")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return
    meta = {
        "dataset": "Safety Helmet Detection (Kaggle andrewmvd/hard-hat-detection, CC0)",
        "classes": CLASSES,
        "seed": args.seed,
        "max_hash_distance": args.max_hash_distance,
    }
    write_outputs(args.src, args.out, splits, effort, meta)


if __name__ == "__main__":
    main()
