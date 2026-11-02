"""Pin the anti-fabrication and persona-override rules in the
write prompts so they don't get accidentally weakened by a future
edit.

These are CONTENT checks, not LLM behavior checks — we just assert
the text contains the rules we added after a real EUV-stochastics
quest produced fabricated "Placeholder, A. & Example, B." citations,
fabricated ``https://frontierinsight.internal/...`` URLs, and full
IMRAD sections inside an essay-format paper.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.write_personas import fill

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
WRITE_MD = AGENTS_DIR / "write.md"
ESSAY_PERSONA = AGENTS_DIR / "write_persona_essay.md"
POLICY_PERSONA = AGENTS_DIR / "write_persona_policy_brief.md"


@pytest.fixture(scope="module")
def write_md_text() -> str:
    return WRITE_MD.read_text(encoding="utf-8")


# ---- Forbidden placeholder author names ----


@pytest.mark.parametrize(
    "token",
    [
        "Placeholder",
        "Author unspecified",
        "Date unspecified",
        "Venue unspecified",
        # Stand-in surnames the LLM has been observed to copy verbatim
        # from example citations:
        "Smith, J.",
        "Doe, J.",
        "Lee, M.",
    ],
)
def test_write_md_forbids_placeholder_author_token(
    write_md_text: str, token: str,
) -> None:
    """write.md must list ``token`` in the forbidden placeholder
    section. If a future edit removes this line, the writer LLM
    starts inventing citations again."""
    assert token in write_md_text, (
        f"write.md should explicitly forbid the placeholder token "
        f"{token!r} — see the 'Forbidden placeholder words' section"
    )


# ---- Forbidden URL/DOI fabrication ----


@pytest.mark.parametrize(
    "token",
    [
        "frontierinsight.internal",
        "example.com",
        "10.xxxx/xxxxx",
        "No URL or DOI fabrication",
    ],
)
def test_write_md_forbids_url_or_doi_fabrication(
    write_md_text: str, token: str,
) -> None:
    """The writer must not invent a URL or DOI to "complete" a
    citation. Pin the explicit forbidden examples + the section
    title."""
    assert token in write_md_text, (
        f"write.md should explicitly forbid {token!r} as a fabricated "
        f"URL/DOI pattern"
    )


# ---- Forbidden stub labels (engine renumbers but if it didn't, the
#      writer should still skip them) ----


@pytest.mark.parametrize(
    "token",
    [
        "Item-N",
        "Reference N",
        "Source N",
    ],
)
def test_write_md_forbids_stub_labels(
    write_md_text: str, token: str,
) -> None:
    """When the literature formatter emits a stub label, the writer
    must skip the entry — not promote the label into an author name."""
    assert token in write_md_text


# ---- The engine writes the source lists, so there is no format to copy ----


def test_write_md_has_no_reference_entries_to_copy(
    write_md_text: str,
) -> None:
    """Earlier versions of write.md showed example citations
    (``Smith, J. & Lee, M. (2021)``, then ``<Author 1 lastname>``
    shapes); the LLM copied them into actual papers. The engine now
    writes References and Further reading itself, so the prompt tells
    the writer not to write them and shows no entry format at all."""
    assert "Do not write a `## References` or a `## Further reading` section." in write_md_text
    assert "<Author 1 lastname>" not in write_md_text
    assert "1. Smith, J. & Lee, M." not in write_md_text


# ---- Persona-driven format overrides forbid IMRAD ----


@pytest.mark.parametrize(
    "forbidden_heading",
    [
        "## Methods",
        "## Results",
        "## Discussion",
        "## Limitations",
    ],
)
def test_essay_persona_forbids_imrad_heading(forbidden_heading: str) -> None:
    """The essay persona must explicitly forbid IMRAD headings — a
    fleet run with gemini-3.1-flash-lite-preview otherwise defaults
    to IMRAD on any experimental topic and ignores the essay shape."""
    # the forbidden list is filled in from core/write_personas.py, as the writer is shown it
    txt = fill("essay", ESSAY_PERSONA.read_text(encoding="utf-8"))
    assert "Forbidden headings" in txt, (
        "write_persona_essay.md must have an explicit 'Forbidden headings' "
        "section so the LLM can't bury IMRAD inside an essay"
    )
    assert forbidden_heading in txt, (
        f"write_persona_essay.md must name {forbidden_heading!r} as forbidden"
    )


@pytest.mark.parametrize(
    "forbidden_heading",
    [
        "## Methods",
        "## Results",
        "## Discussion",
        "## Executive Summary",
        "## Findings",
    ],
)
def test_policy_brief_persona_forbids_imrad_and_summary_headings(
    forbidden_heading: str,
) -> None:
    """policy_brief is strictly Issue / Context / Recommendation; every
    other H2 (IMRAD or Exec Summary / Findings) is forbidden."""
    # the forbidden list is filled in from core/write_personas.py, as the writer is shown it
    txt = fill("policy_brief", POLICY_PERSONA.read_text(encoding="utf-8"))
    assert "Forbidden headings" in txt
    assert forbidden_heading in txt, (
        f"write_persona_policy_brief.md must name {forbidden_heading!r} as forbidden"
    )


def test_policy_brief_persona_pins_three_acts() -> None:
    """The brief must use exactly three H2s: Issue / Context /
    Recommendation. Pin those literal strings."""
    txt = POLICY_PERSONA.read_text(encoding="utf-8")
    for required in ("## Issue", "## Context", "## Recommendation"):
        assert required in txt, (
            f"write_persona_policy_brief.md must prescribe {required!r}"
        )


def test_write_md_has_no_free_figure_sentence_rule(write_md_text: str) -> None:
    """The writer describes a figure only by placing a mark from its list; FI writes the sentence."""
    assert "{{figure:" in write_md_text
    assert "never write a sentence of your own about what such a figure shows" in write_md_text.lower()
    assert "result marks" in write_md_text
    # a figure with no marks list (no call record) keeps the instruction it always had
    assert "state the concrete takeaway the figure supports" in write_md_text
    assert "describe in prose what it would have shown" in write_md_text
    assert "is mentioned by its number only" not in write_md_text


def test_write_md_states_the_one_form_kept_for_a_checked_figure_first(write_md_text: str) -> None:
    """The checked-figure rule comes first and names the two kept forms; the old "Figure 1 shows that" is for unchecked figures."""
    first = write_md_text.index('1. A figure whose entry has a "Marks FI has checked')
    other = write_md_text.index("2. Any other figure")
    rule = write_md_text[first:other]
    assert "only statements you make about it are these two" in rule
    assert 'ends with "(Figure N)"' in rule and "FI removes every other sentence that names that figure" in rule
    assert "Figure 1 shows that the error falls" not in rule and "Figure 1 shows that the error falls" in write_md_text[other:]


def test_slides_and_poster_prompts_say_figure_statements_are_fis_words() -> None:
    root = Path(__file__).resolve().parent.parent / "agents"
    for name in ("slides.md", "poster.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "word for word" in text and "Never write a statement of your own about what such a figure shows" in text, name


def test_checked_figure_rule_names_the_drawn_quantity_and_the_mark_standing_alone() -> None:
    root = Path(__file__).resolve().parent.parent
    write = (root / "agents" / "write.md").read_text(encoding="utf-8")
    assert "is a quantity that figure draws" in write and "as a sentence on its own" in write
    slides = (root / "agents" / "slides.md").read_text(encoding="utf-8")
    assert "**Figure N shows what is plotted" not in slides
    assert "copied word for word" in slides[slides.index("## Figures"):]
    docs = (root / "docs" / "capabilities-reference.md").read_text(encoding="utf-8")
    assert "the paper places none of a figure's marks" in docs and "the shortest sentence" not in docs

