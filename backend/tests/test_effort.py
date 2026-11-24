"""Effort per image: the time and interaction counts behind the effort study."""

from datetime import datetime, timedelta

import pytest

from app.services.effort import Event, rank_biserial, summarize_image

T0 = datetime(2026, 11, 23, 10, 0, 0)


def at(seconds: float, event: str, mode: str = "manual", **payload: object) -> Event:
    return Event(T0 + timedelta(seconds=seconds), event, mode, payload)


def test_time_is_open_until_done() -> None:
    effort = summarize_image(
        [
            at(0, "image_opened"),
            at(5, "annotation_created"),
            at(9, "annotation_edited"),
            at(20, "image_done", by_source={"manual": 1}),
            at(20.1, "image_closed"),
        ],
        "manual",
    )
    assert effort.done
    assert effort.seconds == pytest.approx(20)
    assert effort.interactions == 2
    assert effort.final_by_source == {"manual": 1}


def test_time_away_from_the_image_is_not_counted() -> None:
    effort = summarize_image(
        [
            at(0, "image_opened"),
            at(10, "image_closed"),  # left the image (a break)
            at(300, "image_opened"),
            at(305, "image_done"),
        ],
        "manual",
    )
    assert effort.seconds == pytest.approx(15)


def test_review_after_done_is_ignored() -> None:
    effort = summarize_image(
        [
            at(0, "image_opened"),
            at(10, "image_done"),
            at(50, "image_opened"),
            at(60, "annotation_deleted"),
            at(70, "image_closed"),
        ],
        "manual",
    )
    assert effort.seconds == pytest.approx(10)
    assert effort.interactions == 0


def test_unfinished_image_is_not_done() -> None:
    assert not summarize_image([at(0, "image_opened"), at(8, "image_closed")], "manual").done


def test_suggest_is_counted_separately_and_wrong_mode_is_detected() -> None:
    effort = summarize_image(
        [
            at(0, "image_opened", "manual"),
            at(1, "suggest", "manual"),
            at(5, "image_done", "manual"),
        ],
        "assisted",
    )
    assert effort.by_event == {"suggest": 1}
    assert effort.interactions == 0
    assert effort.wrong_mode_events == 3


def test_rank_biserial_extremes() -> None:
    assert rank_biserial(u_statistic=100, n1=10, n2=10) == 1  # group 1 always larger
    assert rank_biserial(u_statistic=50, n1=10, n2=10) == 0  # no difference


def test_smart_click_counts_as_one_interaction() -> None:
    effort = summarize_image(
        [
            at(0, "image_opened", "assisted"),
            at(1, "smart_click", "assisted", found=True),
            at(1.1, "annotation_created", "assisted", tool="smart", source="ai"),
            at(3, "image_done", "assisted"),
        ],
        "assisted",
    )
    assert effort.interactions == 1
