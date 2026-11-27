"""Effort per image from the interaction log (effort study: manual vs. AI-assisted).

Time counts only while the image is open: from each `image_opened` to the next
`image_closed` / `image_done`, summed, until the first `image_done`. Looking at an image again
after it was finished (a review) is not counted.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# What counts as one "interaction" (a deliberate action on the labels).
INTERACTIONS = {
    "annotation_created",
    "annotation_edited",
    "annotation_deleted",
    "class_changed",
    "undo",
    "magic_click",
    "smart_click",
    "find_similar",
}


@dataclass(frozen=True)
class Event:
    ts: datetime
    event: str
    mode: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageEffort:
    done: bool = False
    seconds: float = 0.0
    interactions: int = 0
    by_event: dict[str, int] = field(default_factory=dict)
    final_by_source: dict[str, int] = field(default_factory=dict)  # from the image_done event
    wrong_mode_events: int = 0  # events logged in the other mode (protocol check)


def summarize_image(events: Iterable[Event], expected_mode: str) -> ImageEffort:
    result = ImageEffort()
    opened_at: datetime | None = None
    for e in sorted(events, key=lambda e: e.ts):
        if result.done:
            break  # anything after the first "done" is a review, not labelling effort
        if e.mode != expected_mode and e.event != "mode_changed":
            result.wrong_mode_events += 1
        if e.event == "image_opened":
            opened_at = e.ts
        elif e.event in ("image_closed", "image_done"):
            if opened_at is not None:
                result.seconds += (e.ts - opened_at).total_seconds()
                opened_at = None
            if e.event == "image_done":
                result.done = True
                result.final_by_source = dict(e.payload.get("by_source", {}))
        elif e.event in INTERACTIONS or e.event == "suggest":
            result.by_event[e.event] = result.by_event.get(e.event, 0) + 1
            # A smart click that adds a box logs "smart_click" AND "annotation_created":
            # one user action, so the second event is not counted again.
            follows_smart_click = (
                e.event == "annotation_created" and e.payload.get("tool") == "smart"
            )
            result.interactions += e.event in INTERACTIONS and not follows_smart_click
    return result


def rank_biserial(u_statistic: float, n1: int, n2: int) -> float:
    """Effect size of a Mann-Whitney U test, from -1 to 1.

    The probability that a random value of group 1 is larger than one of group 2, minus the
    reverse. 0 = no difference; 1 = every value of group 1 is larger.
    """
    return 2 * u_statistic / (n1 * n2) - 1
