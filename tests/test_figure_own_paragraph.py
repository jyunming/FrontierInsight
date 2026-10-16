"""A figure that shares its paragraph with text is moved into a paragraph of its own.

Pandoc numbers and captions an image, and sets it as a float at the text width, only when the image is a paragraph by
itself. The live s6 kinetics paper put each figure after a bold run-in label on the same line
("**Spread.** ![**Figure 1.** ...](figures/x.png)"), so paper.pdf had no captions and the third figure ran off the page.
"""

from __future__ import annotations

import re

from generation._figure_captions import figures_in_own_paragraph

CAP1 = "![**Figure 1.** Bias (left), SD (centre) and coverage (right) of Ea against noise.](figures/fig1_Ea_vs_noise.png)"
CAP2 = "![**Figure 2.** Bias of Ea against n for the three estimators.](figures/fig2_methods_vs_npoints.png)"
CAP3 = "![**Figure 3.** Left: scatter of ln A against Ea. Right: Pearson correlation [3].](figures/fig3_Ea_lnA_correlation.png)"

# The three paragraphs of the live paper, as they stood (blank lines between, as the writer left them).
LIVE = (
    "Results are in the table.\n\n\n\n"
    f"**Spread.** {CAP1}\n\n\n\n"
    "Figure 1 shows that the SD of Ea grows in proportion to sigma.\n\n"
    f"**Methods.** {CAP2}\n\n"
    "Figure 2 compares the estimators.\n\n"
    f"**Correlation.** {CAP3}\n\n"
    "Figure 3 shows the correlation.\n"
)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def test_live_paragraphs_get_their_figure_alone_and_keep_the_caption():
    out, moved = figures_in_own_paragraph(LIVE)
    assert moved == 3
    paragraphs = _paragraphs(out)
    for cap in (CAP1, CAP2, CAP3):
        assert cap in paragraphs  # alone in its paragraph, caption untouched
    for p in paragraphs:
        if "![" in p:
            assert p.strip().startswith("![") and p.strip().endswith(")")


def test_run_in_label_goes_to_the_text_that_follows_the_figure():
    out, _ = figures_in_own_paragraph(LIVE)
    paragraphs = _paragraphs(out)
    i = paragraphs.index(CAP1)
    assert paragraphs[i + 1].startswith("**Spread.** Figure 1 shows that the SD")
    assert "**Spread.**" not in paragraphs[i - 1]


def test_text_with_the_figure_stays_before_it():
    md = f"The fit is shown below. {CAP1} It is tight.\n\nNext paragraph.\n"
    out, moved = figures_in_own_paragraph(md)
    assert moved == 1
    assert _paragraphs(out) == ["The fit is shown below. It is tight.", CAP1, "Next paragraph."]


def test_figures_one_under_the_other_each_get_a_paragraph():
    md = f"{CAP1}\n{CAP2}\n"
    out, moved = figures_in_own_paragraph(md)
    assert moved == 2
    assert _paragraphs(out) == [CAP1, CAP2]


def test_already_alone_is_unchanged_and_idempotent():
    md = f"Text.\n\n{CAP1}\n\nMore text.\n"
    assert figures_in_own_paragraph(md) == (md, 0)
    once, _ = figures_in_own_paragraph(LIVE)
    assert figures_in_own_paragraph(once) == (once, 0)


def test_code_fence_list_and_table_are_left_alone():
    md = (
        f"```\n{CAP1} in code\n\n{CAP2}\n```\n\n"
        f"- item {CAP1}\n\n"
        f"| a | {CAP2} |\n|---|---|\n"
    )
    assert figures_in_own_paragraph(md) == (md, 0)


def test_non_figure_image_is_left_alone():
    md = "Logo ![x](http://example.com/a.png) here.\n"
    assert figures_in_own_paragraph(md) == (md, 0)


def test_label_with_no_prose_after_it_stays_before_the_figure():
    md = f"**Spread.** {CAP1}\n\n## Next section\n"
    out, _ = figures_in_own_paragraph(md)
    assert _paragraphs(out) == ["**Spread.**", CAP1, "## Next section"]


def test_figure_check_still_finds_the_captions():
    from core.engine import _embedded_figure_names

    out, _ = figures_in_own_paragraph(LIVE)
    assert _embedded_figure_names(out) == _embedded_figure_names(LIVE) == {
        "fig1_Ea_vs_noise.png", "fig2_methods_vs_npoints.png", "fig3_Ea_lnA_correlation.png"}
