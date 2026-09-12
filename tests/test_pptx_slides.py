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
