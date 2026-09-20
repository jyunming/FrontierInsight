"""What the prompts ask of a figure: at most three panels in one row, each on a slide of its own.

The effect on a model is not measured here (nothing in these tests calls one); only the
tick-label floor of the slide check enforces it. These pin what each prompt says.
"""

from __future__ import annotations

import string
from pathlib import Path

import pytest

AGENTS = Path(__file__).resolve().parent.parent / "agents"


def _prompt(name: str) -> str:
    return (AGENTS / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["implement.md", "implement_body.md"])
def test_the_implement_prompts_cap_a_figure_at_three_panels_side_by_side(name: str) -> None:
    text = _prompt(name)
    assert "at most 3 panels, side by side in one row" in text
    assert "`plt.subplots(1, n)` with n <= 3" in text
    # A wider need is several figures, one file each, not one bigger grid.
    assert "SEVERAL figures of at most 3 panels each, one file per figure" in text


def test_the_outline_and_the_design_plan_figures_of_at_most_three_panels() -> None:
    outline = _prompt("implement_outline.md")
    assert "at most 3 panels, side by side in one row" in outline
    assert "SEVERAL figures of at most 3 panels each" in outline
    design = _prompt("design.md")
    assert "at most 3 panels side by side in one row" in design and "figures_planned" in design


def test_the_slides_prompt_gives_every_figure_a_slide_of_its_own_and_the_discussion_the_next() -> None:
    text = _prompt("slides.md")
    assert "One figure per slide" in text and "ONE bold lead sentence" in text and "then the figure alone" in text
    assert "The discussion goes on the NEXT slide" in text
    # The old ways of sharing a slide with text are gone: a figure in a side pane, and the
    # sizes the theme ignores.
    assert "![bg right" not in text.replace("never use `![bg ...]`", "") and "right:40%" not in text
    assert "![w:800]" not in text and "![h:380]" not in text
    assert "never use `![bg ...]`" in text
    # A figure slide does not count toward the deck's length.
    assert "plus one slide of its own for each figure" in text


def test_the_slides_prompt_still_fills_in_as_a_template() -> None:
    """It is a string.Template: a stray ``$`` in the new text would break every deck."""
    filled = string.Template(_prompt("slides.md")).substitute(paper_md="PAPER", figure_list="FIGURES")
    assert "PAPER" in filled and "FIGURES" in filled
