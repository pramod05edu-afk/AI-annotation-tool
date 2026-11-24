"""Analyse the effort study: is AI-assisted labelling faster, and is it as accurate?

Reads the interaction log and the final labels of the two effort-study projects from the
database (see docs/effort-study.md), and compares per image:

  effort   time per image, time per object, interactions per image
  quality  final labels vs. the dataset's ground truth (precision / recall / F1, IoU >= 0.5)
  AI share how many final labels the AI produced unchanged / edited, how many were manual

Statistics: mean +- standard deviation and median per mode; Mann-Whitney U test (two
independent groups of images, no assumption of normally distributed times) with the
rank-biserial correlation as effect size.

Usage (from backend/, with the database running):
    uv run python scripts/analyze_effort.py --data D:/datasets/hardhat-prepared
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.models import Annotation, Image, InteractionLog, Project  # noqa: E402
from app.services.effort import Event, rank_biserial, summarize_image  # noqa: E402
from app.services.evaluation import (  # noqa: E402
    Counts,
    Labeled,
    counts_of,
    match,
    read_yolo_labels,
    size_bucket,
)
from app.services.geometry import iou  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SETS_FILE = REPO / "eval" / "effort_study_sets.json"
OUT = REPO / "eval" / "results"
MODES = {"A": "manual", "B": "assisted"}
CLASSES = ["helmet", "head"]


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def compare(manual: list[float], assisted: list[float]) -> dict[str, Any]:
    if len(manual) < 2 or len(assisted) < 2:
        return {"note": "not enough finished images for a test"}
    test = mannwhitneyu(manual, assisted, alternative="two-sided")
    return {
        "mann_whitney_u": float(test.statistic),
        "p_value": float(test.pvalue),
        "rank_biserial": round(rank_biserial(float(test.statistic), len(manual), len(assisted)), 3),
        "mean_reduction_percent": round(
            100 * (1 - statistics.mean(assisted) / statistics.mean(manual)), 1
        ),
        "median_reduction_percent": round(
            100 * (1 - statistics.median(assisted) / statistics.median(manual)), 1
        ),
    }


def label_box(a: Annotation) -> Labeled:
    x1, y1, x2, y2 = a.geometry["bbox"]  # polygons also store their bounding box
    return Labeled(a.class_name, (x1, y1, x2, y2))


def analyse_project(db: Session, name: str, mode: str, labels_dir: Path) -> dict[str, Any]:
    project = db.scalar(select(Project).where(Project.name == name))
    if project is None:
        raise SystemExit(f"Project '{name}' not found - run scripts/seed_effort_study.py first")
    images = list(
        db.scalars(select(Image).where(Image.project_id == project.id).order_by(Image.id))
    )
    per_image, quality, lenient = [], Counts(), Counts()
    # Diagnostics: WHY labels miss the ground truth (absent vs. present but too loose).
    diag: Counter[str] = Counter()
    small_found = small_total = 0
    final_sources: Counter[str] = Counter()
    wrong_mode = 0
    for image in images:
        logs = db.scalars(select(InteractionLog).where(InteractionLog.image_id == image.id))
        effort = summarize_image(
            (Event(r.ts, r.event, r.mode.value, r.payload) for r in logs), mode
        )
        wrong_mode += effort.wrong_mode_events
        if not effort.done:
            continue
        truths = read_yolo_labels(
            (labels_dir / f"{Path(image.filename).stem}.txt").read_text(),
            CLASSES,
            image.width,
            image.height,
        )
        labels = [
            label_box(a)
            for a in db.scalars(select(Annotation).where(Annotation.image_id == image.id))
        ]
        counts = counts_of(match(labels, truths))
        quality.add(counts)
        lenient.add(counts_of(match(labels, truths, iou_threshold=0.3)))
        for truth in truths:
            best = max((iou(lab.box, truth.box) for lab in labels), default=0.0)
            diag[
                "no_label_near_object"
                if best < 0.1
                else "loose_label"
                if best < 0.5
                else "good_label"
            ] += 1
            if size_bucket(truth.box) == "small":
                small_total += 1
                small_found += best >= 0.5
        final_sources.update(effort.final_by_source)
        per_image.append(
            {
                "image": image.filename,
                "objects": len(truths),
                "seconds": round(effort.seconds, 2),
                "seconds_per_object": round(effort.seconds / max(len(truths), 1), 2),
                "interactions": effort.interactions,
                "events": effort.by_event,
                "f1": round(counts.f1, 4),
            }
        )
    return {
        "project": name,
        "mode": mode,
        "images_total": len(images),
        "images_done": len(per_image),
        "wrong_mode_events": wrong_mode,
        "per_image": per_image,
        "quality_vs_ground_truth": quality.as_dict(),
        # IoU >= 0.3: is the object FOUND at all, even if the box is not tight?
        "quality_lenient_iou_30": lenient.as_dict(),
        "objects_by_label_fit": dict(diag),  # good (IoU>=0.5) / loose (0.1-0.5) / none (<0.1)
        "recall_small_objects": round(small_found / small_total, 4) if small_total else None,
        "final_labels_by_source": dict(final_sources),
    }


def figures(results: dict[str, Any], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    modes = ["manual", "assisted"]
    data = {m: results[m]["per_image"] for m in modes}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, title in zip(
        axes,
        ["seconds", "seconds_per_object", "interactions"],
        ["Time per image (s)", "Time per object (s)", "Interactions per image"],
        strict=True,
    ):
        ax.boxplot(
            [[row[key] for row in data[m]] for m in modes], tick_labels=modes, showmeans=True
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Effort study: manual vs. AI-assisted labelling"
        " (box = middle 50 %, line = median, triangle = mean)"
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "effort_boxplots.png", dpi=150)
    plt.close(fig)


def fit(project: dict[str, Any]) -> str:
    f = project["objects_by_label_fit"]
    return (
        f"{f.get('good_label', 0)} / {f.get('loose_label', 0)} / {f.get('no_label_near_object', 0)}"
    )


def markdown(results: dict[str, Any]) -> str:
    m, a, c = results["manual"], results["assisted"], results["comparison"]
    lines = [
        "# Effort study results",
        "",
        f"Finished images: manual {m['images_done']}/{m['images_total']}, "
        f"assisted {a['images_done']}/{a['images_total']}.",
        "",
        "| Metric | Manual | Assisted | Reduction | p (Mann-Whitney) | Effect size r |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in [
        ("seconds", "Time per image (s)"),
        ("seconds_per_object", "Time per object (s)"),
        ("interactions", "Interactions per image"),
    ]:
        dm, da, cmp_ = m["summary"][key], a["summary"][key], c.get(key, {})
        cell = lambda d: f"{d['mean']} ± {d['std']} (median {d['median']})" if d.get("n") else "-"  # noqa: E731
        lines.append(
            f"| {label} | {cell(dm)} | {cell(da)} | {cmp_.get('mean_reduction_percent', '-')} % | "
            f"{cmp_.get('p_value', float('nan')):.4f} | {cmp_.get('rank_biserial', '-')} |"
        )
    qm, qa = m["quality_vs_ground_truth"], a["quality_vs_ground_truth"]
    lines += [
        "",
        "| Label quality vs. ground truth (IoU >= 0.5) | Manual | Assisted |",
        "|---|---|---|",
        f"| Precision | {qm['precision']} | {qa['precision']} |",
        f"| Recall | {qm['recall']} | {qa['recall']} |",
        f"| F1 | {qm['f1']} | {qa['f1']} |",
        f"| Recall at lenient IoU >= 0.3 | {m['quality_lenient_iou_30']['recall']} | "
        f"{a['quality_lenient_iou_30']['recall']} |",
        f"| Recall on small objects (< 32x32 px) | {m['recall_small_objects']} | "
        f"{a['recall_small_objects']} |",
        f"| Objects: good / loose / no label | {fit(m)} | {fit(a)} |",
        "",
        f"Final assisted labels by source: {a['final_labels_by_source']}",
        "",
        "![Effort boxplots](figures/effort_boxplots.png)",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # Windows console
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="prepared dataset (test/labels)")
    args = parser.parse_args()

    sets = json.loads(SETS_FILE.read_text())
    engine = create_engine(get_settings().database_url)
    results: dict[str, Any] = {}
    with Session(engine) as db:
        for key, mode in MODES.items():
            results[mode] = analyse_project(
                db, sets["projects"][key], mode, args.data / "test" / "labels"
            )
    for mode in MODES.values():
        rows = results[mode]["per_image"]
        results[mode]["summary"] = {
            key: describe([row[key] for row in rows])
            for key in ("seconds", "seconds_per_object", "interactions")
        }
    results["comparison"] = {
        key: compare(
            [row[key] for row in results["manual"]["per_image"]],
            [row[key] for row in results["assisted"]["per_image"]],
        )
        for key in ("seconds", "seconds_per_object", "interactions")
    }
    for mode in MODES.values():
        if results[mode]["wrong_mode_events"]:
            print(
                f"WARNING: {results[mode]['wrong_mode_events']} events in the {mode} project "
                "were logged in the other mode"
            )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "effort_study.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    if results["manual"]["per_image"] and results["assisted"]["per_image"]:
        figures(results, OUT / "figures")
    report = markdown(results)
    (OUT / "effort_study.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
