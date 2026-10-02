"""CLI --new review screen: four plain cards, internal names only under Advanced.

``python launch.py --new`` shows, between the three first questions and the launch, four cards (how strictly it is
checked and what it costs; the model and whether it is ready; data, sources and pauses; what you get), built from
``core.interview.REVIEW_CARDS``, the same cards the web page and VS Code show. Each card's advanced rows are folded
under "Advanced" until the person types 'a'; the byline's four fields are one folded row. We don't drive the full
interactive loop here (tests/test_interview_e2e_cli.py does); we pin the structure.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from core.interview import REVIEW_CARDS, derive_tier2, derive_tier3
from launch import _build_review_rows, _print_review

_ANSWERS = {"topic": "Heat flow through a cooling fin", "result_use": "research", "provider": "openai",
            "provider_model": "gpt-5", "second_reviewer_model": "gpt-5-mini"}


def _values() -> dict:
    derived = derive_tier2(_ANSWERS)
    return {**_ANSWERS, **derived, **derive_tier3({**_ANSWERS, **derived})}


def _shown(show_advanced: bool) -> str:
    values = _values()
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_review(_build_review_rows(values, show_advanced=show_advanced), show_advanced=show_advanced, values=values)
    return buf.getvalue()


def test_the_default_view_is_the_four_cards_shown_rows_in_card_order() -> None:
    rows = _build_review_rows(_values(), show_advanced=False)
    assert not any(r["advanced"] for r in rows)
    order = [c["id"] for c in REVIEW_CARDS]
    assert [order.index(str(r["card"])) for r in rows] == sorted(order.index(str(r["card"])) for r in rows)
    expected = [qid for c in REVIEW_CARDS for qid in c["shown"]
                if qid not in ("affiliation", "contact_email", "url")]
    assert [r["id"] for r in rows] == ["byline" if q == "author" else q for q in expected]


def test_advanced_rows_follow_without_renumbering_the_shown_ones() -> None:
    folded = _build_review_rows(_values(), show_advanced=False)
    opened = _build_review_rows(_values(), show_advanced=True)
    assert [r["id"] for r in opened[: len(folded)]] == [r["id"] for r in folded]
    assert [r["id"] for r in opened[len(folded):]] == [qid for c in REVIEW_CARDS for qid in c["advanced"]]


def test_each_row_carries_its_question_object() -> None:
    for r in _build_review_rows(_values(), show_advanced=True):
        if r["id"] == "byline":
            assert r["question"] is None
        else:
            assert r["question"].id == r["id"]


def test_the_cards_say_how_strict_what_it_costs_and_whether_the_model_is_ready() -> None:
    text = _shown(False)
    for card in REVIEW_CARDS:
        assert card["title"] in text
    assert "Cost: about 50-90 model calls" in text
    assert "Ready?" in text
    assert "Paper byline (optional)" in text and "asked once before the first paper" in text


def test_internal_names_only_under_advanced() -> None:
    folded = _shown(False)
    for raw in ("paper_md", "paper_pdf", "methodologist", "devil_advocate", "Axon", "(unset)", "Title (short slug)"):
        assert raw not in folded, f"{raw!r} on the review screen a first-time user sees"
    assert "method, statistics, devil's advocate" in folded
    assert "show advanced" in folded.lower()
    assert "Advanced" not in folded.replace("show advanced", "")
    opened = _shown(True)
    assert "══ Advanced ══" in opened
    assert opened.index("══ Advanced ══") < opened.index("Saved-library passages per quest")


def test_exploring_says_fewer_calls_and_a_preliminary_draft() -> None:
    values = {**_values(), "result_use": "explore"}
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_review(_build_review_rows(values, show_advanced=False), show_advanced=False, values=values)
    text = buf.getvalue()
    assert "preliminary draft" in text and "fewer model calls" in text
    assert "A different model for one reviewer" not in text
