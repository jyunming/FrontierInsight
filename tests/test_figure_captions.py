"""The writer's "Figure N." comes off a caption before a renderer that numbers
figures prints it (``generation/_figure_captions.py``)."""

from __future__ import annotations

import pytest

from generation._figure_captions import numbers_off_figure_captions, without_number


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
