"""Pin the structure of the tier-1 → review tier-2/3 flow in
``web/static/interview.html`` so a future edit can't quietly drop
the review section, the advanced toggle, or the per-row [edit]
buttons.

These are content-level assertions on the static HTML + JS. The
flow is exercised end-to-end in the browser; here we just guarantee
the surface area exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
INTERVIEW_HTML = STATIC / "interview.html"


@pytest.fixture(scope="module")
def html() -> str:
    return INTERVIEW_HTML.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "needle",
    [
        # JS hooks
        "renderReview",
        "deriveTier2",
        "deriveTier3",
        "editReviewField",
        "saveReviewField",
        "cancelReviewEdit",
        "toggleAdvanced",
        "backToForm",
        "launchFromReview",
        # DOM hooks the JS keys off
        'id="review-section"',
        'id="interview-form"',
    ],
)
def test_tier_flow_function_or_hook_present(html: str, needle: str) -> None:
    """Every JS function name + DOM id the tier flow uses must
    appear in the static file. Catches a half-revert that removes
    the review section while leaving the rest intact."""
    assert needle in html, f"interview.html lost the {needle!r} hook"


def test_tier1_filter_excludes_non_tier1_questions(html: str) -> None:
    """The new flow only renders tier-1 questions in the initial
    form. The filter must reference q.tier === 1; if a future edit
    drops the predicate, the form regresses to showing all 14
    questions on one page."""
    assert "q.tier === 1" in html, (
        "interview.html must filter questions by tier===1 to keep the "
        "review-step flow"
    )


def test_continue_to_review_button_replaces_launch_button(html: str) -> None:
    """The form submit button is "Continue to Review", not "Launch
    Quest" — the launch happens at the bottom of the review page
    after the user confirms the derived defaults."""
    assert "Continue to Review" in html
    # Launch button still exists, but inside renderReview() (NOT as
    # the form-submit button). Easiest way to assert: the button
    # type for the in-review launcher is type="button", not
    # type="submit".
    assert "onclick=\"launchFromReview()\"" in html


def test_review_renders_three_cards_tier1_tier2_advanced(html: str) -> None:
    """The review section shows three cards: tier-1 summary,
    tier-2 derived defaults, and a collapsible Advanced block."""
    assert "Your answers" in html, "tier-1 summary card title missing"
    assert "Defaults (auto-derived)" in html, "tier-2 card title missing"
    # Advanced card has a button toggling visibility; its label is
    # the literal string "Advanced".
    assert ">Advanced<" in html or "'Advanced'" in html or '"Advanced"' in html, (
        "advanced toggle label missing"
    )


def test_per_row_edit_buttons_present(html: str) -> None:
    """Each derived-value row carries an inline Edit button that
    calls editReviewField(id). The button's label is "Edit" (plus
    an icon)."""
    assert "editReviewField" in html
    # Two Edit buttons: one for the tier-1 summary card ("Edit"
    # back to form), one per derived row.
    # We just need the JS hook + the visible label.
    assert ">Edit<" in html or 'aria-label="Edit"' in html


def test_axon_status_probed_for_knowledge_enabled_default(html: str) -> None:
    """The knowledge_enabled tier-2 default is auto-probed from
    /api/axon/status — the JS must hit that endpoint, otherwise
    every quest defaults to knowledge_enabled=false regardless of
    the sidecar state."""
    assert "/api/axon/status" in html


def test_prose_formats_set_drives_no_simulation_default(html: str) -> None:
    """no_simulation defaults to True for prose paper formats
    (essay / report / policy_brief / whitepaper). The JS keeps a
    small PROSE_FORMATS set mirroring core/interview.py."""
    assert "PROSE_FORMATS" in html
    for fmt in ("essay", "report", "policy_brief", "whitepaper"):
        assert f"'{fmt}'" in html, f"prose format {fmt!r} missing from PROSE_FORMATS set"
