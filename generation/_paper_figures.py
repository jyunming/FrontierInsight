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


def recorded_panel_share(root: Path, stem: str) -> float | None:
    """The share of its drawn size the smallest panel of the paper copy of ``figures/<stem>`` kept, as the bootstrap
    measured it on matplotlib's own axes and wrote it into the figure's record (``.fi/figure_records/<stem>.json``);
    ``None`` when the record has none (a figure from before it was measured)."""
    try:
        record = json.loads((Path(root) / ".fi" / RECORDS_DIRNAME / f"{stem}.json").read_text(encoding="utf-8"))
        share = record["paper"]["panel_share"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return float(share) if isinstance(share, (int, float)) and not isinstance(share, bool) else None


_ANY_IMAGE = re.compile(r"!\[[^\n]*?\]\(([^)\s]+)\)")


def placed_copies(root: Path) -> list[tuple[int, str, float | None]]:
    """The paper copies the paper's source (``paper_pdf_source.md``, which FI wrote) points at, in order, each as
    ``(figure number, stem, recorded panel share)``; the number is the figure's place among the images of the paper.
    Empty when the paper has no such source."""
    try:
        source = (Path(root) / "paper_pdf_source.md").read_text(encoding="utf-8")
    except OSError:
        return []
    found = []
    for number, match in enumerate(_ANY_IMAGE.finditer(source), start=1):
        target = re.fullmatch(rf"figures/{PAPER_DIRNAME}/([^/]+)\.pdf", match.group(1))
        if target:
            found.append((number, target.group(1), recorded_panel_share(root, target.group(1))))
    return found


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
        # A copy whose smallest panel the bootstrap recorded under the share a copy keeps is not used: the figure is
        # set from its PNG.
        share = recorded_panel_share(root, stem)
        if share is not None and share < PAPER_PANEL_MIN_SHARE:
            return match.group(0)
        if not attributes:
            width = _width_in(pdf)
            share = min(100, max(10, round(100 * width / PAPER_FULL_WIDTH_IN))) if width else 100
            attributes = f"{{width={share}% height={PAPER_HEIGHT_SHARE}%}}"
        swapped += 1
        return f"](figures/{PAPER_DIRNAME}/{stem}.pdf){attributes}"

    return _FIGURE_TARGET.sub(swap, markdown), swapped
