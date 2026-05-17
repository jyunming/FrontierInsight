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
        "Prior work",
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


# ---- Reference format examples are placeholders, not look-alike authors ----


def test_write_md_reference_examples_are_angle_bracket_placeholders(
    write_md_text: str,
) -> None:
    """Earlier versions of write.md showed real-looking example
    citations like ``Smith, J. & Lee, M. (2021)``; the LLM copied
    those verbatim into actual papers. The current version must
    use ``<...>`` placeholder syntax instead."""
    # The new placeholder shape we want to see in the examples block.
    assert "<Author 1 lastname>" in write_md_text, (
        "reference format examples should use <Author 1 lastname>-style "
        "placeholders, not concrete-looking names"
    )
    # The old leak-prone style must be gone from the EXAMPLE lines.
    # We can't assert "Smith, J." doesn't appear at all — the
    # forbidden-list section legitimately names it. So we just check
    # the examples block doesn't start a citation with "Smith, J. &":
    assert "1. Smith, J. & Lee, M." not in write_md_text, (
        "reference format examples should not seed copyable concrete names"
    )


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
    txt = ESSAY_PERSONA.read_text(encoding="utf-8")
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
    txt = POLICY_PERSONA.read_text(encoding="utf-8")
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
