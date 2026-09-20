"""The writer's "Figure N." comes off a caption before a renderer that numbers
figures prints it (``generation/_figure_captions.py``)."""

from __future__ import annotations

import pytest

from generation._figure_captions import blank_lines_around_figures, numbers_off_figure_captions, without_number


@pytest.mark.parametrize("caption", [
    "**Figure 1.** Log-log plot of trajectory error versus time step $h$.",
    "**Figure 1:** Log-log plot of trajectory error versus time step $h$.",
    "*Fig. 1.* Log-log plot of trajectory error versus time step $h$.",
    "**Figure 1** — Log-log plot of trajectory error versus time step $h$.",
    "Figure 1. Log-log plot of trajectory error versus time step $h$.",
    "Fig 1: Log-log plot of trajectory error versus time step $h$.",
])
def test_the_writers_number_comes_off_the_caption(caption: str) -> None:
    assert without_number(caption) == "Log-log plot of trajectory error versus time step $h$."


@pytest.mark.parametrize("caption", [
    "Figure 2 shows the drift of each integrator.",   # a sentence, not a number
    "Energy decay against the analytical reference.",
    "**Figure 3.**",                                   # nothing else to print
])
def test_a_caption_without_a_leading_number_stays_as_written(caption: str) -> None:
    assert without_number(caption) == caption


def test_only_figure_lines_outside_code_change() -> None:
    md = "\n".join([
        "## Results",
        "",
        "![**Figure 1.** Error versus step, as in [3].](figures/error.png)",
        "",
        "![**Figure 2.** Energy decay.](figures/energy.png){width=80%}",
        "",
        "Figure 1 shows the error; an inline ![**Figure 9.** icon](figures/i.png) stays.",
        "",
        "```markdown",
        "![**Figure 4.** An example in a code block.](figures/x.png)",
        "```",
    ])
    assert numbers_off_figure_captions(md).split("\n") == [
        "## Results",
        "",
        "![Error versus step, as in [3].](figures/error.png)",
        "",
        "![Energy decay.](figures/energy.png){width=80%}",
        "",
        "Figure 1 shows the error; an inline ![**Figure 9.** icon](figures/i.png) stays.",
        "",
        "```markdown",
        "![**Figure 4.** An example in a code block.](figures/x.png)",
        "```",
    ]


# --- a figure is a paragraph of its own ----------------------------------------------------

def test_figures_written_one_under_the_other_are_paragraphs_of_their_own() -> None:
    """Three in a row, as a stored quest wrote them, printed with no caption at all: pandoc read them
    as one paragraph of three inline images."""
    md = (
        "Text.\n![**Figure 1.** One.](figures/a.png)\n![**Figure 2.** Two.](figures/b.png)\n"
        "![Three.](figures/c.png)\nMore text.\n"
    )
    assert blank_lines_around_figures(md) == (
        "Text.\n\n![**Figure 1.** One.](figures/a.png)\n\n![**Figure 2.** Two.](figures/b.png)\n\n"
        "![Three.](figures/c.png)\n\nMore text.\n"
    )


def test_a_figure_that_is_already_a_paragraph_and_a_code_block_are_left_alone() -> None:
    spaced = "Text.\n\n![A.](figures/a.png)\n\nMore.\n"
    assert blank_lines_around_figures(spaced) == spaced
    once = blank_lines_around_figures("a\n![A.](x.png)\nb")
    assert once == "a\n\n![A.](x.png)\n\nb" and blank_lines_around_figures(once) == once
    fenced = "```\n![A.](x.png)\n![B.](y.png)\n```\n"
    assert blank_lines_around_figures(fenced) == fenced
    # An image inside a sentence is not a figure line.
    inline = "See ![icon](i.png) here.\nNext line.\n"
    assert blank_lines_around_figures(inline) == inline
    assert blank_lines_around_figures("") == "" and blank_lines_around_figures("![A.](x.png)") == "![A.](x.png)"
