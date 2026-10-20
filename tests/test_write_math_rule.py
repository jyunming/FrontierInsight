"""The writer puts math in LaTeX between dollar signs and leaves headings unnumbered.

The rule lives in the three write prompts and in the review's rewrite prompt
(``write_patch.md``). Each prompt is a ``string.Template``, so a literal dollar
sign must be written ``$$``; this test substitutes every placeholder with dummy
text and checks that the rule comes out with single dollar signs.
"""
from __future__ import annotations

import string
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
PROMPTS = ("write.md", "write_patch.md", "write_trim.md")


class _Any(dict):
    """Mapping that answers every placeholder with dummy text."""

    def __missing__(self, key: str) -> str:
        return f"DUMMY_{key}"


def _render(name: str) -> str:
    text = (AGENTS_DIR / name).read_text(encoding="utf-8")
    return string.Template(text).substitute(_Any())


@pytest.mark.parametrize("name", PROMPTS)
def test_math_rule_survives_substitution(name: str) -> None:
    rendered = _render(name)
    # Single dollar signs after substitution, not the doubled template form.
    assert "$\\sigma = 0.6$" in rendered
    assert "$k_1 \\approx 0.28$" in rendered
    assert "$$" not in rendered
    assert "Write every formula and symbol in LaTeX math between dollar signs" in rendered


@pytest.mark.parametrize("name", PROMPTS)
def test_headings_rule_is_present(name: str) -> None:
    rendered = _render(name)
    assert "Do not number the section headings; FI numbers them." in rendered


@pytest.mark.parametrize("name", PROMPTS)
def test_unicode_math_ban_is_present(name: str) -> None:
    rendered = _render(name)
    assert "do not use Unicode math symbols" in rendered
