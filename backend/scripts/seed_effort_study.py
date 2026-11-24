"""Create the two projects for the effort study (manual vs. AI-assisted annotation).

The 60 effort-study images (chosen by prepare_dataset.py from the TEST split: never used in
training, 3-15 objects each) are divided into two sets of 30 with the same difficulty:
images are sorted by their number of objects and dealt out A, B, B, A, A, B, B, A, ...
so both sets get the same mix of easy and crowded scenes.

    Set A -> project "Effort study A (manual)"    label in MANUAL mode
    Set B -> project "Effort study B (assisted)"  label in ASSISTED mode

The split is saved to eval/effort_study_sets.json (committed), so the analysis is reproducible.
The app must be running (docker compose up). See docs/effort-study.md for the protocol.

Usage (from the backend/ folder):
    uv run python scripts/seed_effort_study.py --images D:/datasets/hardhat-prepared/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_FILE = REPO_ROOT / "eval" / "splits.json"
SETS_FILE = REPO_ROOT / "eval" / "effort_study_sets.json"
CLASSES = ["helmet", "head"]
PROJECTS = {"A": "Effort study A (manual)", "B": "Effort study B (assisted)"}
MODES = {"A": "manual", "B": "assisted"}


def object_count(labels_dir: Path, name: str) -> int:
    return sum(1 for line in (labels_dir / f"{name}.txt").read_text().splitlines() if line.strip())


def balanced_sets(names: list[str], counts: dict[str, int]) -> dict[str, list[str]]:
    """Deal images sorted by difficulty in the pattern A B B A: equal difficulty in A and B."""
    ordered = sorted(names, key=lambda n: (-counts[n], n))
    sets: dict[str, list[str]] = {"A": [], "B": []}
    for i, name in enumerate(ordered):
        sets["AB"[0 if i % 4 in (0, 3) else 1]].append(name)
    return sets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", type=Path, required=True, help="the prepared TEST split folder")
    parser.add_argument("--api", default="http://localhost:3000", help="URL of the running app")
    args = parser.parse_args()

    names: list[str] = json.loads(SPLITS_FILE.read_text())["effort_study"]
    counts = {n: object_count(args.images / "labels", n) for n in names}
    sets = balanced_sets(names, counts)

    with httpx.Client(base_url=args.api, timeout=60) as client:
        existing = {p["name"] for p in client.get("/api/projects").raise_for_status().json()}
        if clash := existing & set(PROJECTS.values()):
            print(f"error: {sorted(clash)} already exist - delete them in the app first")
            return 1
        for key, project_name in PROJECTS.items():
            project = (
                client.post(
                    "/api/projects",
                    # The app enforces this mode for every image of the project.
                    json={"name": project_name, "classes": CLASSES, "annotation_mode": MODES[key]},
                )
                .raise_for_status()
                .json()
            )
            for name in sets[key]:  # one by one: the upload order is the annotation order
                image = args.images / "images" / f"{name}.png"
                with image.open("rb") as f:
                    response = (
                        client.post(
                            f"/api/projects/{project['id']}/images",
                            files=[("files", (image.name, f, "image/png"))],
                        )
                        .raise_for_status()
                        .json()
                    )
                if response["rejected"]:
                    print(f"error: {image.name} rejected: {response['rejected']}")
                    return 1
            objects = sum(counts[n] for n in sets[key])
            print(
                f"{project_name}: {len(sets[key])} images, {objects} objects (id {project['id']})"
            )

    SETS_FILE.write_text(
        json.dumps({"projects": PROJECTS, "sets": sets, "objects": counts}, indent=2) + "\n"
    )
    print(f"Wrote {SETS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
