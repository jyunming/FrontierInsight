"""A slide figure takes only the room left under the slide's text.

Rendered with the real Marp CLI and the deck theme, then measured. The 48vh
cap alone let a figure under five bullets run 42 pt past the bottom of the
slide, and a caption written on the line under the figure went with it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from generation._marp import find_marp
from generation._pdf_measure import measure_pdf, slides_report
from generation.slides import THEME_PATH, _figures_as_own_paragraphs

DECK = """---
marp: true
theme: fi
paginate: true
---

## A crowded figure slide

**The bold lead line of the slide sits here.**

- First bullet with enough words to take most of a line on the slide.
- Second bullet with enough words to take most of a line on the slide.
- Third bullet with enough words to take most of a line on the slide.
- Fourth bullet with enough words to take most of a line on the slide.
- Fifth bullet with enough words to take most of a line on the slide.
![w:800](figures/chart.png)
**Figure 1:** Caption written on the line under the figure.

---

## A short figure slide

- One bullet.

![w:800](figures/chart.png)

Takeaway under a figure that has room.
"""


def _render(tmp_path: Path) -> Path:
    from PIL import Image, ImageDraw

    (tmp_path / "figures").mkdir()
    chart = Image.new("RGB", (1600, 1000), "white")
    ImageDraw.Draw(chart).rectangle((80, 80, 1520, 920), outline="black", width=6)
    chart.save(tmp_path / "figures" / "chart.png")
    (tmp_path / "slides.md").write_text(_figures_as_own_paragraphs(DECK), encoding="utf-8")
    subprocess.run(
        [find_marp(), "slides.md", "--allow-local-files", "--theme", str(THEME_PATH), "-o", "slides.pdf"],
        cwd=tmp_path, capture_output=True, timeout=300, stdin=subprocess.DEVNULL, check=True,
    )
    return tmp_path / "slides.pdf"


def _figure_and_text(page):
    figure = next(im for im in page.images if im[2] - im[0] < 0.9 * page.width)
    # PDF boxes run up from the bottom edge: (x0, bottom, x1, top).
    above = [line for line in page.lines if line.box[1] > figure[3]]
    below = [line for line in page.lines if line.box[3] < figure[1] and "FRONTIER" not in line.text and line.text.strip() != str(page.number)]
    return figure, above, below


@pytest.mark.slow
@pytest.mark.skipif(find_marp() is None, reason="the Marp CLI is not installed")
def test_a_figure_shrinks_into_the_room_under_its_text_and_keeps_its_caption_on_the_slide(tmp_path: Path) -> None:
    doc = measure_pdf(_render(tmp_path))
    assert [f for f in slides_report(doc)["findings"] if f["check"] == "overflow"] == []

    crowded, short = doc.pages
    figure, _above, below = _figure_and_text(crowded)
    assert figure[1] >= 0, "the figure stays on the slide"
    caption = next(line for line in below if line.text.startswith("Figure 1:"))
    assert caption.visible and caption.box[1] >= 0

    # Under a short text the figure keeps its usual size and spacing: the
    # figure's paragraph does not stretch down the slide.
    figure, above, below = _figure_and_text(short)
    assert figure[3] - figure[1] > 240, "a figure with room is not shrunk"
    gap_above = min(line.box[1] for line in above) - figure[3]
    gap_below = figure[1] - max(line.box[3] for line in below)
    assert gap_above < 22 and gap_below < 22, (gap_above, gap_below)
