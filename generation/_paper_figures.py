"""The paper's figures from their paper copies.

Each figure an experiment saves into ``figures/`` also gets a paper copy, ``figures/paper/<stem>.pdf``, drawn at the
width of a one-column page's text with text at the body's size (``core/plot_style.py``). The LaTeX source of
``paper.pdf`` uses that vector copy in place of the PNG, at its own width as a share of the line width, so a point of
text in the figure prints as a point on the page. ``paper.md`` itself, the HTML render (which cannot show a PDF image)
and the slides keep the PNGs. A two-column paper keeps the PNGs too: the paper copy is drawn for one column of text
6.3 in wide, and scaled into a 3.5 in column its text would print at half its size.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.plot_style import PAPER_DIRNAME, PAPER_FULL_WIDTH_IN, PAPER_PANEL_MIN_SHARE, RECORDS_DIRNAME
from generation._pdf_measure import PaperFigure, copy_panel_share, measure_pdf

# The target of a Markdown image (or link) to a figure file in figures/, with an optional {attributes} block after it.
_FIGURE_TARGET = re.compile(r"\]\(figures/([^)\s/]+)\.(png|jpe?g|svg)\)(\{[^}]*\})?", re.IGNORECASE)
_MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]")


def _width_in(pdf: Path) -> float | None:
    """The page width of a one-page PDF matplotlib wrote, in inches; ``None`` when it cannot be read."""
    try:
        match = _MEDIABOX.search(pdf.read_bytes())
    except OSError:
        return None
    if not match:
        return None
    try:
        x0, _y0, x1, _y1 = (float(v) for v in match.groups())
    except ValueError:
        return None
    return (x1 - x0) / 72 if x1 > x0 else None


def paper_figure(root: Path, stem: str) -> PaperFigure | None:
    """What FI recorded of the paper copy of ``figures/<stem>``, from the figure's record
    (``.fi/figure_records/<stem>.json``); ``None`` when there is none or it lacks a size or a panel."""
    try:
        record = json.loads((Path(root) / ".fi" / RECORDS_DIRNAME / f"{stem}.json").read_text(encoding="utf-8"))
        paper = record["paper"]
        drawn = (float(paper["drawn_in"][0]), float(paper["drawn_in"][1]))
        places = tuple(tuple(float(v) for v in axes["place"]) for axes in record["axes"])
        if not places or any(len(place) != 4 for place in places):
            return None
        return PaperFigure(stem, float(paper["width_in"]), float(paper["height_in"]), drawn, places)  # type: ignore[arg-type]
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def paper_figures(root: Path) -> list[PaperFigure]:
    """The recorded paper copies of every figure of the quest."""
    found = []
    records = Path(root) / ".fi" / RECORDS_DIRNAME
    try:
        stems = sorted(p.stem for p in records.glob("*.json") if ".seed" not in p.name)
    except OSError:
        return []
    for stem in stems:
        figure = paper_figure(root, stem)
        if figure is not None:
            found.append(figure)
    return found


def copy_is_squeezed(pdf: Path, figure: PaperFigure | None) -> float | None:
    """The share of its drawn size that the smallest panel of the copy ``pdf`` keeps, measured on the file, when that
    is under ``PAPER_PANEL_MIN_SHARE``; ``None`` when the copy is fine or cannot be measured against a record."""
    if figure is None:
        return None
    doc = measure_pdf(pdf, max_pages=1)
    if doc is None or not doc.pages:
        return None
    share = copy_panel_share(doc.pages[0], figure)
    return share if share is not None and share < PAPER_PANEL_MIN_SHARE else None


#: The most of the page's text height a paper copy is set at: a third, so a tall figure leaves room for its caption
#: and the text around it.
PAPER_HEIGHT_SHARE = 33


def prefer_paper_figures(markdown: str, root: Path, *, two_column: bool = False) -> tuple[str, int]:
    """``(markdown, n)``: each image of ``figures/<stem>.png`` (or .jpg, .svg) whose paper copy
    ``figures/paper/<stem>.pdf`` exists under ``root``, beside the figure itself, points at that copy instead, with
    ``{width=NN% height=33%}``: its share of the line width that the copy was drawn at, and at most a third of the
    page's text height (the templates keep a figure's aspect ratio, so a tall figure is scaled down to that height and
    a wide one is set at its width). An attribute block the writer gave is kept as it is. ``n`` is how many were
    swapped. A two-column paper is returned unchanged."""
    if two_column:
        return markdown, 0
    swapped = 0

    def swap(match: re.Match[str]) -> str:
        nonlocal swapped
        stem, suffix, attributes = match.group(1), match.group(2), match.group(3)
        pdf = Path(root) / "figures" / PAPER_DIRNAME / f"{stem}.pdf"
        # Only the copy of a figure that is there: a figure FI took out of figures/ is not shown through a copy left
        # behind.
        if not pdf.is_file() or not (Path(root) / "figures" / f"{stem}.{suffix}").is_file():
            return match.group(0)
        # A copy whose panels were drawn as slivers (measured on the file against the figure's record) is not used:
        # the figure is set from its PNG.
        if copy_is_squeezed(pdf, paper_figure(root, stem)) is not None:
            return match.group(0)
        if not attributes:
            width = _width_in(pdf)
            share = min(100, max(10, round(100 * width / PAPER_FULL_WIDTH_IN))) if width else 100
            attributes = f"{{width={share}% height={PAPER_HEIGHT_SHARE}%}}"
        swapped += 1
        return f"](figures/{PAPER_DIRNAME}/{stem}.pdf){attributes}"

    return _FIGURE_TARGET.sub(swap, markdown), swapped
