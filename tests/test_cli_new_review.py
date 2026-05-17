"""CLI --new review screen + numbered-edit loop.

Pin the structure of the review table that ``python launch.py --new``
shows between tier-1 collection and quest launch. We don't try to
drive the full interactive loop (that needs a real TTY); we just
assert the helpers produce the right shape so a future edit can't
silently shuffle rows or drop the advanced section.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from core.interview import questions_for_tier
from launch import _build_review_rows, _print_review


def _stub_tier_qs():
    return (
        questions_for_tier(2, "cli"),
        questions_for_tier(3, "cli"),
    )


def test_review_rows_default_view_is_tier2_only() -> None:
    """Without ``show_advanced``, the review screen lists only the
    6 tier-2 rows. Tier-3 fields stay hidden behind the toggle."""
    t2, t3 = _stub_tier_qs()
    derived = {q.id: q.default for q in t2}
    advanced = {q.id: q.default for q in t3}
    rows = _build_review_rows(derived, advanced, t2, t3, show_advanced=False)

    assert len(rows) == len(t2), f"expected {len(t2)} tier-2 rows, got {len(rows)}"
    assert all(r["tier"] == 2 for r in rows)
    assert [r["id"] for r in rows] == [q.id for q in t2]


def test_review_rows_expanded_view_appends_tier3() -> None:
    """With ``show_advanced``, tier-3 rows append after tier-2 — the
    numbered list grows but the tier-2 numbers don't shift."""
    t2, t3 = _stub_tier_qs()
    derived = {q.id: q.default for q in t2}
    advanced = {q.id: (q.default if q.default is not None else "") for q in t3}
    rows = _build_review_rows(derived, advanced, t2, t3, show_advanced=True)

    assert len(rows) == len(t2) + len(t3)
    # First block is still tier-2 in the same order.
    assert [r["id"] for r in rows[: len(t2)]] == [q.id for q in t2]
    # Second block is tier-3.
    assert [r["id"] for r in rows[len(t2):]] == [q.id for q in t3]


def test_review_rows_each_carry_their_question_object() -> None:
    """The edit handler re-prompts via the carried Question, so each
    row must reference its source question (not just an id string)."""
    t2, t3 = _stub_tier_qs()
    rows = _build_review_rows(
        {q.id: q.default for q in t2},
        {q.id: (q.default if q.default is not None else "") for q in t3},
        t2, t3, show_advanced=True,
    )
    for r in rows:
        assert r["question"] is not None
        assert r["question"].id == r["id"]


def test_print_review_groups_advanced_under_subheading() -> None:
    """When advanced is expanded, the printed table prints a visible
    sub-heading right before the tier-3 rows so the user can tell
    where the basic block ends and the advanced block begins."""
    t2, t3 = _stub_tier_qs()
    rows = _build_review_rows(
        {q.id: q.default for q in t2},
        {q.id: (q.default if q.default is not None else "") for q in t3},
        t2, t3, show_advanced=True,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_review(rows, show_advanced=True)
    text = buf.getvalue()

    assert "Review before launch" in text
    assert "Advanced" in text
    # The advanced subheading must appear AFTER all tier-2 row labels
    # (we check the position of the first tier-3 label vs the heading).
    first_t3_label = t3[0].label
    advanced_pos = text.find("Advanced")
    t3_pos = text.find(first_t3_label)
    assert advanced_pos < t3_pos, (
        "advanced subheading should print before the first tier-3 row"
    )


def test_print_review_hides_advanced_subheading_when_collapsed() -> None:
    """When advanced is collapsed, the user sees a hint instead of
    the Advanced subheading + tier-3 rows."""
    t2, t3 = _stub_tier_qs()
    rows = _build_review_rows(
        {q.id: q.default for q in t2},
        {q.id: (q.default if q.default is not None else "") for q in t3},
        t2, t3, show_advanced=False,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_review(rows, show_advanced=False)
    text = buf.getvalue()

    assert "show advanced" in text.lower(), (
        "collapsed view should hint that 'a' expands the advanced block"
    )
    # No tier-3 row labels appear yet.
    for q in t3:
        assert q.label not in text, f"tier-3 label {q.label!r} leaked into collapsed view"
