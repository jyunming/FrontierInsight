"""Figures out of a PDF, each with its caption, as images: no language model, only the page's own layout.

A paper's key numbers often live only in a plot. :func:`find` locates every captioned figure ("Figure 3.", "Fig. 3:",
"FIG. 3 ...") and renders the region the figure occupies to a PNG:

- A page with a text layer: the caption's line is found in the text, and the figure is the drawing above it -- the
  images and vector paths the page draws there, grown to take in the panels beside and above one another and the
  axis labels inside them, and stopped at the next caption above (a table or another figure). A figure whose caption
  sits above it (rare for figures) or that is not drawn above its caption is skipped, never guessed.
- A scanned page (a page image with no text layer, read by OCR in core/pdf_text.py): the caption is found in the OCR
  lines, and the figure is the region between it and the nearest line of body text above.

The images are saved for people (``data/literature/figures/``) and can be shown to a model that reads images.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Render scale: 1 is 72 dpi, so 144 dpi, enough to read a plot's tick labels.
SCALE = 2.0
#: At most this many figures from one PDF (a thesis or a book is not rendered whole).
MAX_FIGURES = 40
#: A caption's first drawing must end within this many points above it (a caption far from any drawing is a sentence
#: in the text that happens to start with "Figure 3.", not a caption).
_MAX_GAP = 60.0
#: Captions longer than this are cut (a caption that runs into the body text by a layout accident stays bounded).
_MAX_CAPTION = 1500

_CAPTION = re.compile(r"^\s*(?:Fig\.?|Figure|FIG\.?|FIGURE)\s*(\d+[A-Za-z]?)\s*(?:[.:|]|\s+(?=[A-Z(]))")
_ANY_CAPTION = re.compile(
    r"^\s*(?:Fig\.?|Figure|FIG\.?|FIGURE|Table|TABLE|Tab\.?)\s*[\dIVX]+[A-Za-z]?\s*(?:[.:|]|\s+(?=[A-Z(]))"
)

Box = tuple[float, float, float, float]  # (left, bottom, right, top) in PDF points, y up


@dataclass
class Figure:
    """One captioned figure of a PDF."""

    number: str       # "3", "3a"
    page: int         # 1-based
    caption: str      # the whole caption, "Figure 3. ..." included
    png: bytes
    scanned: bool     # found on a scanned page, by OCR

    def file_name(self, stem: str) -> str:
        return f"{stem}_fig{self.number}_p{self.page}.png"


def _near(a: Box, b: Box, gap: float) -> bool:
    return not (a[0] > b[2] + gap or b[0] > a[2] + gap or a[1] > b[3] + gap or b[1] > a[3] + gap)


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _text_rects(textpage: Any) -> list[tuple[Box, str]]:
    out = []
    for i in range(textpage.count_rects()):
        box = textpage.get_rect(i)
        text = (textpage.get_text_bounded(*box) or "").strip()
        if text:
            out.append((box, text))
    return out


def _graphics(page: Any, width: float, height: float) -> list[Box]:
    """Where the page draws: images, vector paths, shadings and embedded forms (top level, so in page coordinates)."""
    import pypdfium2.raw as pdfium_c

    kinds = (pdfium_c.FPDF_PAGEOBJ_IMAGE, pdfium_c.FPDF_PAGEOBJ_PATH, pdfium_c.FPDF_PAGEOBJ_SHADING,
             pdfium_c.FPDF_PAGEOBJ_FORM)
    boxes: list[Box] = []
    for obj in page.get_objects(max_depth=1):
        if obj.type not in kinds:
            continue
        box = tuple((getattr(obj, "get_bounds", None) or obj.get_pos)())
        if box[2] - box[0] > width * 0.9 and box[3] - box[1] > height * 0.9:
            continue  # a page-sized background or frame
        if (box[2] - box[0]) + (box[3] - box[1]) > 3:
            boxes.append(box)  # type: ignore[arg-type]
    return boxes


def _caption_line(rects: list[tuple[Box, str]], start: Box) -> Box:
    """The whole line a caption starts on: the runs to its right on the same row, joined while the gap is small."""
    line = start
    row = sorted((b for b, _ in rects if min(b[3], start[3]) - max(b[1], start[1]) > 0.5 * (start[3] - start[1])),
                 key=lambda b: b[0])
    for b in row:
        if b[0] >= line[0] and b[0] - line[2] < 12:
            line = _union(line, b)
    return line


def _caption_box(rects: list[tuple[Box, str]], first: Box) -> Box:
    """The caption paragraph: its first line, then each line just below that starts at its left edge and follows at the
    caption's own line pitch (top to top). A paragraph break (a wider gap) or body text set at another pitch ends it; a
    single-line caption followed by body text keeps only what is within a normal line gap."""
    box = first
    line = first
    height = max(first[3] - first[1], 4.0)
    pitch: float | None = None
    for _ in range(12):
        below = [b for b, _ in rects
                 if b[3] <= line[1] + height * 0.3 and line[1] - b[3] < height * 0.6 and abs(b[0] - first[0]) < 15]
        if not below:
            break
        nxt = _caption_line(rects, max(below, key=lambda b: b[3]))
        step = line[3] - nxt[3]
        if nxt[2] - nxt[0] < 5 or (pitch is not None and abs(step - pitch) > 0.25 * pitch):
            break
        pitch = step if pitch is None else pitch
        box = _union(box, nxt)
        line = nxt
    return box


def _figure_box(
    caption: Box, rects: list[tuple[Box, str]], graphics: list[Box], height: float, width: float,
) -> Box | None:
    """The drawing above a caption (see the module docstring), or None when there is none close above it. On a
    two-column page a figure in one column grows only within that column, so a plot in the next column never joins it."""
    left, _bottom, right, top = caption
    ceiling = height
    for b, text in rects:
        if b[1] > top + 2 and _ANY_CAPTION.match(text) and b[1] < ceiling and b[2] > left - 5 and b[0] < right:
            # Below that caption's last line (a table's caption above a figure can run over several lines).
            ceiling = min(b[1], max(top + 2, _caption_box(rects, _caption_line(rects, b))[1]))
    candidates = [(g[0], max(g[1], top), g[2], g[3]) for g in graphics if g[3] > top + 3 and g[3] <= ceiling + 1]
    candidates = [g for g in candidates if g[3] - g[1] > 1 or g[2] - g[0] > 1]
    over = [g for g in candidates if g[0] < right + 20 and g[2] > left - 20]
    if not over:
        return None
    seed = min(over, key=lambda g: g[1])
    if seed[1] - top > _MAX_GAP:
        return None
    box = seed
    for g in over:  # panels side by side resting on the same caption
        overlap = min(g[3], seed[3]) - max(g[1], seed[1])
        if g[1] - top <= _MAX_GAP and overlap > 0.5 * min(g[3] - g[1], seed[3] - seed[1]):
            box = _union(box, g)
    mid, slack = width / 2, width * 0.02
    spans_middle = (box[0] < mid - slack and box[2] > mid + slack) or (left < mid - slack and right > mid + slack)
    span = (0.0, width) if spans_middle else ((0.0, mid + slack) if box[2] <= mid + slack else (mid - slack, width))
    grown = True
    while grown:  # panels above, arrows and frames touching the drawing, within the figure's column
        grown = False
        for g in candidates:
            if g[0] < span[0] - 2 or g[2] > span[1] + 2:
                continue
            if _union(box, g) != box and _near(box, g, 14):
                box, grown = _union(box, g), True
    for b, _text in rects:  # axis labels, legends and panel titles at the drawing's edge
        if b[1] > top + 1 and b[3] <= ceiling and _near(box, b, 4) and (b[2] - b[0]) < 0.5 * (box[2] - box[0]):
            box = _union(box, b)
    return box


def _render(page: Any, box: Box, *, floor: float) -> bytes:
    width, height = page.get_size()
    left, bottom, right, top = box
    crop = (max(0.0, left - 3), max(0.0, max(bottom - 3, floor)), max(0.0, width - right - 3),
            max(0.0, height - top - 3))
    image = page.render(scale=SCALE, crop=crop).to_pil()
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _figures_on_text_page(page: Any, index: int, seen: set[str]) -> list[Figure]:
    width, height = page.get_size()
    textpage = page.get_textpage()
    try:
        rects = _text_rects(textpage)
        graphics = _graphics(page, width, height)
        out: list[Figure] = []
        for box, text in rects:
            m = _CAPTION.match(text)
            if not m or m.group(1) in seen:
                continue
            first = _caption_line(rects, box)
            figure = _figure_box(first, rects, graphics, height, width)
            if figure is None:
                continue
            cap = _caption_box(rects, first)
            caption = re.sub(r"\s+", " ", textpage.get_text_bounded(*cap) or "").strip()[:_MAX_CAPTION]
            seen.add(m.group(1))
            out.append(Figure(m.group(1), index + 1, caption, _render(page, figure, floor=box[3] + 0.5), False))
        return out
    finally:
        textpage.close()


def _caption_end(lines: list[tuple[Box, str]], index: int) -> float:
    """The bottom of the caption paragraph that starts at ``lines[index]`` (lines sorted top to bottom): each next
    line counts while it starts at the caption's left edge, sits just below, and is not another caption."""
    first = lines[index][0]
    last = first
    for b, t in lines[index + 1:]:
        if b[3] > last[1] + 2:  # beside the line above (the other column), not below it
            continue
        if last[1] - b[3] > (last[3] - last[1]) * 1.2 or abs(b[0] - first[0]) > 25 or _ANY_CAPTION.match(t):
            break
        last = b
    return last[1]


def _figures_on_scanned_page(page: Any, index: int, lines: list[tuple[Box, str]], seen: set[str]) -> list[Figure]:
    """``lines``: the page's OCR lines in PDF points (y up), from core/pdf_text.py."""
    width, height = page.get_size()
    out: list[Figure] = []
    for i, (box, text) in enumerate(lines):
        m = _CAPTION.match(text)
        if not m or m.group(1) in seen:
            continue
        full = box[0] < width * 0.45 and box[2] > width * 0.55
        col = (0.0, width) if full else ((0.0, width / 2) if box[2] <= width * 0.55 else (width / 2, width))
        top = box[3]
        # The figure reaches up to the nearest line of body text (as wide as most of a text column: a figure's own
        # labels and legends are short) or the last line of a caption above.
        body = 0.6 * min(col[1] - col[0], width / 2 - 40)
        ceiling = height - 36
        for j, (b, t) in enumerate(lines):
            if b[1] <= top or b[1] >= ceiling or b[2] < col[0] + 5 or b[0] > col[1] - 5:
                continue
            if _ANY_CAPTION.match(t):
                ceiling = _caption_end(lines, j)
            elif (b[2] - b[0]) > body:
                ceiling = b[1]
        if ceiling - top < 40:
            continue
        end = _caption_end(lines, i)
        caption = [t for b, t in lines if end <= b[1] and b[3] <= box[3] + 1 and abs(b[0] - box[0]) <= 25]
        seen.add(m.group(1))
        figure = (max(col[0], 18.0), top + 2, min(col[1], width - 18), ceiling - 2)
        out.append(Figure(m.group(1), index + 1, " ".join(caption)[:_MAX_CAPTION], _render(page, figure, floor=top + 1),
                          True))
    return out


def find(source: bytes | Path | str, *, ocr_lines: dict[int, list[tuple[Box, str]]] | None = None,
         max_figures: int = MAX_FIGURES) -> list[Figure]:
    """Every captioned figure of a PDF, in page order. Never raises: a PDF that cannot be read has none.

    ``ocr_lines`` maps a 1-based scanned page to its OCR lines (``PdfText.ocr_lines`` of core/pdf_text.py); a scanned
    page without them is skipped."""
    import pypdfium2 as pdfium

    try:
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        doc = pdfium.PdfDocument(data)
    except Exception:  # noqa: BLE001 -- unreadable: no figures
        return []
    out: list[Figure] = []
    seen: set[str] = set()
    try:
        for i in range(len(doc)):
            if len(out) >= max_figures:
                break
            page = doc[i]
            try:
                if ocr_lines and (i + 1) in ocr_lines:
                    out += _figures_on_scanned_page(page, i, ocr_lines[i + 1], seen)
                else:
                    out += _figures_on_text_page(page, i, seen)
            except Exception:  # noqa: BLE001 -- one odd page costs only its own figures
                continue
            finally:
                page.close()
    finally:
        doc.close()
    return out[:max_figures]
