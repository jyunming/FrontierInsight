"""Native .pptx slide rendering (``generation/_pptx_slides.py``).

Replaces a `pandoc slides.md -o slides.pptx` shell-out that (a) needed pandoc
on PATH and (b) produced an unthemed deck. These cover the Marp subset the
deck prompt instructs the model to emit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from generation._pptx_slides import (
    _estimate_lines,
    parse_marp,
    render_marp_to_pptx,
)

DECK = """---
marp: true
theme: fi
---

<!-- _class: lead -->

# The main thesis

## A short kicker

---

### EVIDENCE

## Global error separates the methods

- **RK4** reaches 2.1e-6
- *Verlet* holds 4.8e-4
  - nested detail

> A pull quote.

---

## Chart slide

![bg right:42% fit](figures/x.png)

- point one

---

<!-- _class: lead -->

# Closing line
"""


def test_frontmatter_is_not_a_slide() -> None:
    """The leading `---\nmarp: true\n---` block must not become slide 1."""
    slides = parse_marp(DECK)
    assert len(slides) == 4
    assert slides[0].h1 == "The main thesis"


def test_lead_directive_and_headings() -> None:
    s = parse_marp(DECK)
    assert s[0].lead is True and s[0].h2 == "A short kicker"
    assert s[1].lead is False
    assert s[1].h3 == "EVIDENCE"
    assert s[1].h2 == "Global error separates the methods"
    assert s[3].lead is True


def test_bullets_nesting_and_quote() -> None:
    s = parse_marp(DECK)[1]
    assert [lvl for lvl, _ in s.bullets] == [0, 0, 1]
    assert s.bullets[0][1] == "**RK4** reaches 2.1e-6"
    assert s.quote == "A pull quote."


def test_bg_right_image_parsed_with_percentage() -> None:
    s = parse_marp(DECK)[2]
    assert s.image == "figures/x.png"
    assert s.image_mode == "bg_right"
    assert s.image_pct == 42
    # An image-only line contributes no body text.
    assert s.paras == []
    assert len(s.bullets) == 1


def test_plain_image_is_block_mode() -> None:
    s = parse_marp("## T\n\n![w:900](figures/y.png)\n")[0]
    assert s.image_mode == "block"


def test_directives_other_than_lead_are_dropped() -> None:
    s = parse_marp("<!-- paginate: false -->\n\n## T\n\n- a\n")[0]
    assert s.h2 == "T" and len(s.bullets) == 1


def test_estimate_lines_detects_wrapping() -> None:
    """Reserving too little height is what made titles overlap the bullets."""
    assert _estimate_lines("Short", 11.5, 30) == 1
    long_title = "Global error separates the integrators by two orders of magnitude"
    assert _estimate_lines(long_title, 11.5, 30, wide_factor=0.52) >= 2
    # Same title in a narrow two-column body wraps further.
    assert (_estimate_lines(long_title, 6.9, 30, wide_factor=0.52)
            > _estimate_lines(long_title, 11.5, 30, wide_factor=0.52))


def test_a_long_source_list_shrinks_to_fit_the_slide(tmp_path: Path) -> None:
    """A twelve-entry References slide at full size ran inches past the slide
    bottom; the body now shrinks until the wrap estimate fits, and a short
    slide keeps full size."""
    entry = (
        "Robert I. McLachlan, G. Quispel, Nicolas Robidoux (1999). Geometric "
        "integration using discrete gradients. Philosophical Transactions of "
        "the Royal Society A DOI: 10.1098/rsta.1999.0363"
    )
    deck = (
        "---\nmarp: true\ntheme: fi\n---\n\n## Short\n\n- one point\n\n---\n\n"
        "## References\n\n"
        + "\n".join(f"- [{i}] {entry}" for i in range(1, 13)) + "\n"
    )
    md = tmp_path / "slides.md"
    md.write_text(deck, encoding="utf-8")
    out = tmp_path / "slides.pptx"
    assert render_marp_to_pptx(md, out) is True

    from pptx import Presentation
    prs = Presentation(str(out))

    def sizes(slide, needle: str) -> set[float]:
        return {
            r.font.size.pt
            for sh in slide.shapes if sh.has_text_frame
            for p in sh.text_frame.paragraphs if needle in p.text
            for r in p.runs if r.font.size is not None
        }

    assert max(sizes(prs.slides[0], "one point")) == 16.5
    long_list = sizes(prs.slides[1], "DOI")
    assert long_list and max(long_list) < 16.5


def test_renders_a_real_pptx(tmp_path: Path) -> None:
    md = tmp_path / "slides.md"
    md.write_text(DECK, encoding="utf-8")
    out = tmp_path / "slides.pptx"
    assert render_marp_to_pptx(md, out) is True
    assert out.is_file() and out.stat().st_size > 5000

    from pptx import Presentation
    prs = Presentation(str(out))
    assert len(prs.slides) == 4
    # 16:9 at the same geometry the HTML deck uses.
    assert round(prs.slide_width / 914400, 2) == 13.33
    texts = [
        sh.text_frame.text
        for sh in prs.slides[1].shapes if sh.has_text_frame
    ]
    joined = " ".join(texts)
    assert "Global error separates the methods" in joined
    assert "EVIDENCE" in joined
    assert "RK4" in joined          # bold markers stripped, content kept
    assert "**" not in joined       # markup must not leak into the deck


def test_render_never_raises_on_bad_input(tmp_path: Path) -> None:
    """Slides are secondary: a render failure must not fail the quest."""
    missing = tmp_path / "nope.md"
    assert render_marp_to_pptx(missing, tmp_path / "o.pptx") is False


def test_empty_deck_returns_false(tmp_path: Path) -> None:
    md = tmp_path / "slides.md"
    md.write_text("---\nmarp: true\n---\n", encoding="utf-8")
    assert render_marp_to_pptx(md, tmp_path / "o.pptx") is False


# Slide 5 of the validation quest's deck: a lead line, three bullets (one
# wrapping) and a figure under them. The figure used to sit a fixed 1.6 in
# under the body top, on the third bullet.
FIGURE_DECK = """---
marp: true
theme: fi
---

## Numerical schemes successfully capture exponential decay

**Both RK4 and Velocity-Verlet accurately track the system's dissipative energy loss.**

- The system follows an analytical decay curve for the energy of the damped oscillator.
- Figure 2 shows that both stable methods avoid the catastrophic energy growth seen in Forward Euler.
- The aggregate relative energy error remains low for both methods over the simulation interval.
- Velocity-Verlet keeps its error bounded while Forward Euler grows without limit.

![w:800](figures/energy.png)

---

## A figure alone

![w:800](figures/energy.png)

---

## Many points above a figure

- Point one with enough words to fill most of a line on this slide here.
- Point two with enough words to fill most of a line on this slide here.
- Point three with enough words to fill most of a line on this slide here.
- Point four with enough words to fill most of a line on this slide here.
- Point five with enough words to fill most of a line on this slide here.
- Point six with enough words to fill most of a line on this slide here.
- Point seven with enough words to fill most of a line on this slide here.

![w:800](figures/energy.png)
"""


def _figure_deck(tmp_path: Path) -> Path:
    from PIL import Image, ImageDraw

    figures = tmp_path / "figures"
    figures.mkdir()
    chart = Image.new("RGB", (1500, 900), "white")
    ImageDraw.Draw(chart).rectangle((40, 40, 1460, 860), outline="black", width=8)
    chart.save(figures / "energy.png")
    md = tmp_path / "slides.md"
    md.write_text(FIGURE_DECK, encoding="utf-8")
    out = tmp_path / "slides.pptx"
    assert render_marp_to_pptx(md, out, figures_dir=figures) is True
    return out


def test_a_figure_goes_under_the_text_it_follows(tmp_path: Path) -> None:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(str(_figure_deck(tmp_path)))
    slide = prs.slides[0]
    picture = next(sh for sh in slide.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE)
    body = next(sh for sh in slide.shapes if sh.has_text_frame and "aggregate relative" in sh.text_frame.text)
    assert picture.top >= body.top + body.height
    assert picture.top + picture.height <= prs.slide_height - int(0.85 * 914400)
    assert picture.height >= int(2.0 * 914400), "the figure stays readable"
    # With no text above it the figure starts where the body would.
    alone = next(sh for sh in prs.slides[1].shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE)
    assert alone.top < picture.top
    # Seven points leave no room at full size: the text shrinks, and the
    # figure keeps its minimum height above the footer.
    crowded = next(sh for sh in prs.slides[2].shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE)
    assert crowded.top + crowded.height <= prs.slide_height - int(0.85 * 914400)
    assert crowded.height >= int(2.0 * 914400)


@pytest.mark.slow
def test_the_exported_deck_has_no_figure_over_its_text(tmp_path: Path) -> None:
    from generation._office_pdf import find_libreoffice, pptx_to_pdf
    from generation._pdf_measure import measure_pdf, slides_report

    if find_libreoffice() is None:
        pytest.skip("LibreOffice is not installed")
    pdf, reason = pptx_to_pdf(_figure_deck(tmp_path), tmp_path / "export")
    assert pdf is not None, reason
    report = slides_report(measure_pdf(pdf))
    assert [f for f in report["findings"] if f["check"] in ("overlap", "overflow")] == []
