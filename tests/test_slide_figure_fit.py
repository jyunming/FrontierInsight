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


FIGURE_SLIDES = """---
marp: true
theme: fi
paginate: true
---

## A figure on a slide of its own

**The lead line of the slide.**

![](figures/square.png)

---

## A figure that shares its slide with a bullet

- One bullet.

![](figures/square.png)

---

## A wide figure on a slide of its own

**The lead line of the slide.**

![](figures/row.png)
"""


def _render_figure_slides(tmp_path: Path) -> Path:
    from PIL import Image

    (tmp_path / "figures").mkdir()
    Image.new("RGB", (1300, 1000), "white").save(tmp_path / "figures" / "square.png", dpi=(100, 100))
    Image.new("RGB", (1500, 450), "white").save(tmp_path / "figures" / "row.png", dpi=(100, 100))
    (tmp_path / "slides.md").write_text(_figures_as_own_paragraphs(FIGURE_SLIDES), encoding="utf-8")
    subprocess.run(
        [find_marp(), "slides.md", "--allow-local-files", "--theme", str(THEME_PATH), "-o", "slides.pdf"],
        cwd=tmp_path, capture_output=True, timeout=300, stdin=subprocess.DEVNULL, check=True,
    )
    return tmp_path / "slides.pdf"


@pytest.mark.slow
@pytest.mark.skipif(find_marp() is None, reason="the Marp CLI is not installed")
def test_a_figure_on_a_slide_of_its_own_is_drawn_as_large_as_the_slide_allows(tmp_path: Path) -> None:
    """The 48vh cap is for a figure that shares its slide with text. On a slide of its
    own (a title, a lead line, the figure) a square figure used to leave about a fifth of
    the slide unused, and a wide one is as wide as the text column."""
    doc = measure_pdf(_render_figure_slides(tmp_path))
    alone, shared, row = (
        next(image for image in page.images if image[2] - image[0] < 0.9 * page.width and image[3] - image[1] > 24)
        for page in doc.pages
    )
    height = lambda box: box[3] - box[1]  # noqa: E731
    assert height(shared) <= 0.48 * 540 + 2, "with a bullet the cap still applies"
    assert height(alone) >= 0.60 * 540, "alone it takes the room under the lead line"
    assert alone[1] >= 50 and shared[1] >= 50 and row[1] >= 50, "and stays above the footer"
    assert row[2] - row[0] >= 0.85 * 960, "a wide figure is as wide as the text column"
    assert [f for f in slides_report(doc)["findings"] if f["check"] in ("overflow", "overlap", "figure_gap")] == []


# The six references of a stored quest, whose four most cited sources filled the References
# slide at the size it was first set at.
REFERENCES = [
    {"n": 1, "authors": ["P. van den Driessche"], "year": 2017, "venue": "Infectious Disease Modelling",
     "title": "Reproduction numbers of infectious disease models", "doi": "10.1016/j.idm.2017.06.002"},
    {"n": 2, "authors": ["Robin N. Thompson", "Christopher A. Gilligan", "Nik J. Cunniffe"], "year": 2016,
     "title": "Detecting Presymptomatic Infection Is Necessary to Forecast Major Epidemics in the Earliest Stages "
              "of Infectious Disease Outbreaks", "venue": "PLoS Computational Biology", "doi": "10.1371/journal.pcbi.1004836"},
    {"n": 3, "authors": ["Linda J. S. Allen"], "year": 2017, "venue": "Infectious Disease Modelling",
     "title": "A primer on stochastic epidemic models: Formulation, numerical simulation, and analysis",
     "doi": "10.1016/j.idm.2017.03.001"},
    {"n": 4, "authors": ["Linda J. S. Allen", "Glenn Lahodny"], "year": 2012, "venue": "Journal of Biological Dynamics",
     "title": "Extinction thresholds in deterministic and stochastic epidemic models", "doi": "10.1080/17513758.2012.665502"},
    {"n": 5, "authors": ["Daniel T. Gillespie"], "year": 1977, "venue": "The Journal of Physical Chemistry",
     "title": "Exact stochastic simulation of coupled chemical reactions", "doi": "10.1021/j100540a008"},
    {"n": 6, "authors": ["Matthew Hartfield", "Samuel Alizon"], "year": 2013, "venue": "PLoS Pathogens",
     "title": "Introducing the Outbreak Threshold in Epidemiology", "doi": "10.1371/journal.ppat.1003277"},
]


# Six shorter ones (about 110 characters each): the budget lists all six, and they still fit.
SHORT_REFERENCES = [
    {"n": n, "authors": ["A. Author", "B. Author"], "year": 2000 + n, "venue": "Journal of Dynamics",
     "title": f"A study of extinction thresholds number {n}", "doi": f"10.1000/j.{n}"}
    for n in range(1, 7)
]


@pytest.mark.slow
@pytest.mark.skipif(find_marp() is None, reason="the Marp CLI is not installed")
@pytest.mark.parametrize("refs,paper,listed_count", [
    (REFERENCES, "Cites [1] [2] [4] [5] and [1] [2].", 4),
    (SHORT_REFERENCES, "Cites [1] [2] [3] [4] [5] [6].", 6),
])
def test_the_references_slide_is_set_at_18_pt_and_its_last_entry_stays_on_the_slide(
    tmp_path: Path, refs: list, paper: str, listed_count: int,
) -> None:
    """The list was set at 0.6em (11 pt). At 0.96em the budget in core/engine.py lists as
    many sources as fit: for a stored quest's six references, its four most cited."""
    from core.engine import render_references_marp_slide

    slide = render_references_marp_slide(refs, paper_md=paper)
    listed = [line for line in slide.splitlines() if line.startswith("- [")]
    assert len(listed) == listed_count
    assert ("more sources in the paper" in slide) == (listed_count < len(refs))
    front_matter = "---\nmarp: true\ntheme: fi\npaginate: true\n---\n\n"
    (tmp_path / "slides.md").write_text(front_matter + slide.removeprefix("---\n") + "\n", encoding="utf-8")
    subprocess.run(
        [find_marp(), "slides.md", "--allow-local-files", "--theme", str(THEME_PATH), "-o", "slides.pdf"],
        cwd=tmp_path, capture_output=True, timeout=300, stdin=subprocess.DEVNULL, check=True,
    )
    doc = measure_pdf(tmp_path / "slides.pdf")
    assert len(doc.pages) == 1
    assert [f for f in slides_report(doc)["findings"] if f["check"] == "overflow"] == []
    # The text layer reports the size in the CSS pixels Chromium prints: 0.96em of the
    # slide's 25 px is 24, and 1280 px are printed as 960 pt, so 18 pt on the page.
    body = [line for line in doc.pages[0].lines if line.visible and 20 < line.size < 30]
    assert {line.size for line in body} == {24.0} and len(body) >= 10
    assert min(line.box[1] for line in body) >= 0, "the last entry is on the page"
