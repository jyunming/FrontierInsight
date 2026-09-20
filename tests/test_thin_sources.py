"""The writer is told which retrieved works hold only a title or a short blurb.

A stored SIR quest cited a title-only record (Andersson and Britton, 57 characters
of text) for "stochastic extinction" and a book blurb (Keeling and Rohani, 812
characters) for "the effective R0 drops more rapidly": claims those records cannot
show. The prior-work block the writer reads now says ``[title only]`` or ``[short
blurb only]`` on such an entry, and the write and edit prompts each say, once, that
it may be cited only to say the work exists or for what its own title states. The
entry stays in the list.

Over the 2,321 sources of the 131 stored quests the marks fall on 513 title-only
records and 62 books that have a blurb and no full text; none of the 575 has an
abstract or full text.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from core.engine import (
    _THIN_TEXT_CHARS,
    _foundational_write_block,
    _format_lit_from_state,
    _thin_source,
)
from tests.test_write_patch import CLAIM, DRAFT, NEW_CLAIM, TOPIC, _edits, _engine

REPO = Path(__file__).resolve().parent.parent
SENTENCE = (
    "cite it only to say the work exists or for what its own title states, "
    "never for a specific finding, number or mechanism"
)

ABSTRACT = (
    "We study the probability that an epidemic started by one infective dies out before it "
    "reaches a macroscopic size, and show that it equals the extinction probability of the "
    "embedded branching process, 1/R0 when R0 exceeds one. The result holds for any "
    "infectious-period distribution and is checked against exact stochastic simulation."
)
# The stored Keeling and Rohani record: a title, then a review's opening lines.
BLURB = (
    "By Matthew James Keeling and Pejman Rohani. Princeton, NJ: Princeton University Press, 2008. "
    "408 pp., Illustrated. $65.00 (hardcover). Mathematical modeling of infectious diseases has "
    "progressed dramatically over the past 3 decades and continues to flourish at the nexus of "
    "mathematics, epidemiology, and infectious diseases research."
)


def _src(title: str, doi: str, text: str | None = None, **meta: Any) -> dict[str, Any]:
    """A stored source: ``content`` is the title, a blank line, then ``text``."""
    content = title if text is None else f"{title}\n\n{text}"
    return {"content": content, "metadata": {
        "source": "openalex", "title": title, "doi": doi, "authors": ["A. Author"], "year": 2000,
        "venue": "J. Tests", "content_quality": "snippet_only", **meta}}


ALPHA = _src("Alpha", "10.1/a", ABSTRACT, work_type="article")
TITLE_ONLY = _src("Stochastic Epidemic Models and Their Statistical Analysis", "10.1/b", work_type="book",
                  foundational="suggested as a foundational work")
BOOK_BLURB = _src("Modeling Infectious Diseases in Humans and Animals", "10.1/c", BLURB, work_type="book",
                  foundational="suggested as a foundational work")
DELTA = _src("Delta", "10.1/d", ABSTRACT + " " + ABSTRACT, work_type="article")
FULL_TEXT = _src("Epsilon", "10.1/e", "short", work_type="article", content_quality="full_text",
                 fetched_full_text=True)
LIT = [ALPHA, TITLE_ONLY, BOOK_BLURB, DELTA, FULL_TEXT]


def _state(lit: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"topic": TOPIC, "literature": LIT if lit is None else lit}


def _entry_lines(block: str) -> dict[str, str]:
    """The first line of each entry of a prior-work block, by label."""
    return {m.group(1): m.group(0) for m in re.finditer(r"^\[(\w+)\] .*$", block, re.MULTILINE)}


# --- what counts as thin ---------------------------------------------------------------

@pytest.mark.parametrize(("record", "expected"), [
    (TITLE_ONLY, "title"),  # the text is the title
    (_src("T", "10.1/x", "a few words"), "title"),  # the title and 11 characters
    (_src("T", "10.1/x", None, source="crossref"), "title"),
    (BOOK_BLURB, "blurb"),  # a book with a blurb and no full text
    (ALPHA, None),  # an abstract
    (DELTA, None),
    (FULL_TEXT, None),  # full text, however short the stored text
    (_src("A Book", "10.1/x", BLURB, work_type="book", content_quality="full_text", fetched_full_text=True), None),
    (_src("T", "10.1/x", None, content_quality="full_text"), None),
    (_src("A Page", "10.1/x", "x" * 433, source="web_search", kind="web_page"), None),  # a search snippet
])
def test_what_is_thin(record: dict[str, Any], expected: str | None) -> None:
    assert _thin_source(record["metadata"], record["content"]) == expected


def test_the_cut_is_the_first_length_no_title_only_record_reaches() -> None:
    assert _THIN_TEXT_CHARS == 100
    just_under = _src("T", "10.1/x", "x" * (_THIN_TEXT_CHARS - 1))
    at_cut = _src("T", "10.1/x", "x" * _THIN_TEXT_CHARS)
    assert _thin_source(just_under["metadata"], just_under["content"]) == "title"
    assert _thin_source(at_cut["metadata"], at_cut["content"]) is None
    assert _thin_source({"title": "T"}, "") == "title"  # nothing stored at all


# --- the writer's block ------------------------------------------------------------------

def test_the_writers_block_marks_the_thin_entries_and_only_them() -> None:
    lines = _entry_lines(_format_lit_from_state(_state(), mark_thin=True))
    assert lines["2"].endswith("Stochastic Epidemic Models and Their Statistical Analysis [title only]")
    assert lines["3"].endswith("Modeling Infectious Diseases in Humans and Animals [short blurb only]")
    for label in ("1", "4", "5"):
        assert "only]" not in lines[label], lines[label]


def test_a_marked_entry_stays_in_the_block_and_nothing_else_about_it_changes() -> None:
    plain = _format_lit_from_state(_state())
    marked = _format_lit_from_state(_state(), mark_thin=True)
    assert "only]" not in plain, "the other prompts that read the literature are unchanged"
    assert marked != plain
    assert marked.replace(" [title only]", "").replace(" [short blurb only]", "") == plain
    assert "Modeling Infectious Diseases in Humans and Animals" in marked and BLURB in marked


def test_a_block_with_no_thin_source_is_the_same_with_or_without_the_marks() -> None:
    lit = [ALPHA, DELTA, FULL_TEXT]
    assert _format_lit_from_state(_state(lit), mark_thin=True) == _format_lit_from_state(_state(lit))


def test_the_foundational_list_carries_the_same_marks() -> None:
    block = _foundational_write_block(LIT)
    lines = {line.split("]")[0].lstrip("- ["): line for line in block.splitlines() if line.startswith("- [")}
    assert lines["2"].endswith("[title only] (book)")
    assert lines["3"].endswith("[short blurb only] (book)")


# --- the prompts -------------------------------------------------------------------------

def test_the_instruction_is_one_sentence_in_the_write_and_the_edit_prompt() -> None:
    for name in ("write.md", "write_patch.md"):
        text = " ".join((REPO / "agents" / name).read_text(encoding="utf-8").split())
        assert text.count(SENTENCE) == 1, name
        assert "`[title only]` or `[short blurb only]`" in text, name


def test_the_write_prompt_marks_the_entries(tmp_path: Path) -> None:
    eng, client, _ = _engine(tmp_path, [DRAFT])
    asyncio.run(eng._node_write({"topic": TOPIC, "title": "t", "iteration": 0, "literature": LIT,
                                 "design": {}, "analysis": {}, "figures": []}))
    prompt = client.calls[0][1]
    # The prior-work block's own entries (the foundational list marks these works too).
    entries = _entry_lines(prompt)
    assert entries["2"].endswith("Statistical Analysis [title only]")
    assert entries["3"].endswith("Humans and Animals [short blurb only]")
    assert " ".join(prompt.split()).count(SENTENCE) == 1
    assert "[1] A. Author (2000). Alpha\n" in prompt, "an entry with an abstract is as it was"
    # Every entry is still there, cited by the writer as before.
    for title in ("Alpha", "Delta", "Epsilon", "Stochastic Epidemic Models", "Modeling Infectious"):
        assert title in prompt


def test_the_edit_prompt_marks_the_entries_too(tmp_path: Path) -> None:
    eng, client, _ = _engine(tmp_path, [DRAFT, _edits((CLAIM, NEW_CLAIM))])
    state = {"topic": TOPIC, "title": "t", "iteration": 0, "literature": LIT, "design": {}, "analysis": {}}
    out = asyncio.run(eng._node_write(state))
    revise = {
        **state, "literature": out["literature"], "paper_md": out["paper_md"], "paper_basis": out["paper_basis"],
        "iteration": 1,
        "review": {"verdict": "revise", "must_flag_hits": ["unsupported_claim"]},
        "claim_grounding": {"claims": [{"claim": "Extinction is certain below R0 = 1.2.", "basis": "unsupported",
                                        "evidence": "No source says so."}],
                            "unsupported": ["Extinction is certain below R0 = 1.2."]},
    }
    asyncio.run(eng._node_write(revise))
    assert client.nodes == ["write", "write.patch"]
    prompt = client.calls[1][1]
    entries = _entry_lines(prompt)
    assert entries["2"].endswith("Statistical Analysis [title only]")
    assert entries["3"].endswith("Humans and Animals [short blurb only]")
    assert " ".join(prompt.split()).count(SENTENCE) == 1

