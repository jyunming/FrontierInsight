"""Measurements of a rendered PDF, read with pypdfium2.

Font sizes, words, line lengths, content boxes, figures and column fill.
The visual check gives these numbers to the model together with the page
screenshots, so the model is asked only what a script cannot see. The
poster generator checks its layout against the published poster standards
with them. Nothing here calls an LLM.

Coordinates are PDF points with the origin at the bottom left of the page.
Every entry point fails open: an unreadable file or a missing pypdfium2
gives ``None`` (or ``[]``), never an exception, because a quest must not
stop over a measurement.
"""

from __future__ import annotations

import ctypes
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

Box = tuple[float, float, float, float]  # left, bottom, right, top

_PT_PER_CM = 72 / 2.54

# Published poster minimums in points. They hold at every sheet size:
# viewing distance, not paper size, decides what can be read.
POSTER_MIN_PT = {
    "title": 72.0,
    "heading": 36.0,
    "body": 24.0,
    "caption": 18.0,
    "references": 14.0,
}
POSTER_HEADING_RATIO = 1.4  # headings at least this many times the body size
POSTER_MAX_WORDS = 1000
POSTER_MAX_REFERENCES = 8
POSTER_MAX_CHARS_PER_LINE = 65
SLIDE_MIN_PT = 12.0
PAPER_MIN_PT = 7.0

_BOLD_WEIGHT = 600
_RULE_MAX_HEIGHT = 12.0  # a full-width path this thin is a rule, not a panel
_CAPTION_RE = re.compile(r"^(Figure|Fig\.?|Table)\s*\d+", re.IGNORECASE)
_REFERENCE_LABEL_RE = re.compile(r"\[(W?\d+)\]")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


@dataclass
class Line:
    text: str
    size: float  # most common font size of its visible characters
    bold: bool  # at least 80% of its visible characters are bold
    box: Box

    @property
    def visible(self) -> int:
        return sum(1 for c in self.text if not c.isspace())

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Page:
    number: int  # 1-based
    width: float
    height: float
    lines: list[Line]
    images: list[Box]
    rules: list[Box]  # thin filled paths spanning most of the page width


@dataclass
class Document:
    pages: list[Page]


def available() -> bool:
    try:
        import pypdfium2  # noqa: F401
    except Exception:
        return False
    return True


def measure_pdf(path: Path | str, *, max_pages: int | None = None) -> Document | None:
    """Read every page's text lines, images and rules. The file is read
    into memory first, so no handle stays open on it and a generator can
    overwrite the PDF straight away (Windows refuses while one is open)."""
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
    except Exception:
        return None
    try:
        pdf = pdfium.PdfDocument(Path(path).read_bytes())
    except Exception:
        return None
    try:
        count = len(pdf) if max_pages is None else min(len(pdf), max_pages)
        pages = []
        for index in range(count):
            page = pdf[index]
            try:
                pages.append(_measure_page(page, index + 1, raw))
            finally:
                page.close()
        return Document(pages)
    except Exception:
        return None
    finally:
        pdf.close()


def render_pages(
    path: Path | str,
    out_dir: Path,
    *,
    dpi: int = 90,
    stem: str = "page",
    max_pages: int | None = None,
) -> list[Path]:
    """Save page screenshots as ``<stem>-<n>.png``. Returns the files
    written, ``[]`` when the PDF cannot be rendered."""
    try:
        import pypdfium2 as pdfium
    except Exception:
        return []
    try:
        pdf = pdfium.PdfDocument(Path(path).read_bytes())
    except Exception:
        return []
    written: list[Path] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        count = len(pdf) if max_pages is None else min(len(pdf), max_pages)
        for index in range(count):
            page = pdf[index]
            try:
                image = page.render(scale=dpi / 72).to_pil()
            finally:
                page.close()
            target = out_dir / f"{stem}-{index + 1}.png"
            image.save(target)
            written.append(target)
    except Exception:
        return written
    finally:
        pdf.close()
    return written


# ---------------------------------------------------------------------------
# Reading a page


def _measure_page(page, number: int, raw) -> Page:
    width, height = (float(v) for v in page.get_size())
    textpage = page.get_textpage()
    try:
        lines = _read_lines(textpage, raw)
    finally:
        textpage.close()
    images = [
        _object_box(obj, raw)
        for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=5)
    ]
    rules = []
    for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_PATH], max_depth=5):
        box = _object_box(obj, raw)
        if box[2] - box[0] >= 0.8 * width and box[3] - box[1] <= _RULE_MAX_HEIGHT:
            rules.append(box)
    return Page(number, width, height, lines, images, rules)


def _read_lines(textpage, raw) -> list[Line]:
    """Split the page text into lines. PDFium marks the line breaks it
    detects with generated ``\\r\\n``; a large horizontal jump on the same
    row (an ``\\hfill`` gap, the next table cell) also starts a new line,
    so a line is one continuous run of text."""
    lines: list[Line] = []
    run: list[tuple[str, float, bool, Box | None]] = []
    last: tuple[float, Box] | None = None

    def flush() -> None:
        nonlocal run, last
        visible = [c for c in run if c[3] is not None]
        if visible:
            sizes = Counter(round(c[1], 1) for c in visible)
            boxes = [c[3] for c in visible]
            lines.append(Line(
                text=" ".join("".join(c[0] for c in run).split()),
                size=sizes.most_common(1)[0][0],
                bold=sum(1 for c in visible if c[2]) >= 0.8 * len(visible),
                box=(
                    min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes),
                ),
            ))
        run, last = [], None

    for index in range(textpage.count_chars()):
        code = raw.FPDFText_GetUnicode(textpage.raw, index)
        ch = chr(code) if code else ""
        if ch == "\n":
            flush()
            continue
        if ch in ("\r", ""):
            continue
        if ch.isspace():
            run.append((" ", 0.0, False, None))
            continue
        box = _box(textpage.get_charbox(index))
        if box[2] <= box[0] and box[3] <= box[1]:
            run.append((ch, 0.0, False, None))  # no glyph box: text only
            continue
        size = float(raw.FPDFText_GetFontSize(textpage.raw, index))
        if last is not None and _starts_new_line(last, size, box):
            flush()
        run.append((ch, size, _is_bold(textpage, index, raw), box))
        last = (size, box)
    flush()
    return lines


def _starts_new_line(last: tuple[float, Box], size: float, box: Box) -> bool:
    last_size, prev = last
    em = max(last_size, size, 1.0)
    same_row = min(prev[3], box[3]) > max(prev[1], box[1]) - 0.3 * em
    gap = box[0] - prev[2]
    return not same_row or gap > 2.5 * em or gap < -1.5 * em


def _is_bold(textpage, index: int, raw) -> bool:
    weight = raw.FPDFText_GetFontWeight(textpage.raw, index)
    if weight > 0:
        return weight >= _BOLD_WEIGHT
    # No weight in the font descriptor (the standard 14 fonts): go by name.
    buffer = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()
    raw.FPDFText_GetFontInfo(textpage.raw, index, buffer, 128, ctypes.byref(flags))
    return b"bold" in buffer.value.lower()


def _box(bounds) -> Box:
    left, bottom, right, top = (float(v) for v in bounds)
    return (left, bottom, right, top)


def _object_box(obj, raw) -> Box:
    # The raw call: ``PdfObject.get_bounds`` only exists from pypdfium2 5.
    left, bottom, right, top = (ctypes.c_float() for _ in range(4))
    raw.FPDFPageObj_GetBounds(
        obj.raw, ctypes.byref(left), ctypes.byref(bottom), ctypes.byref(right), ctypes.byref(top),
    )
    return (left.value, bottom.value, right.value, top.value)


# ---------------------------------------------------------------------------
# Helpers shared by the reports


def _cm(points: float) -> float:
    return round(points / _PT_PER_CM, 1)


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _size_mode(lines: Sequence[Line]) -> float | None:
    """Most common font size, weighted by visible characters."""
    counts: Counter = Counter()
    for line in lines:
        counts[line.size] += line.visible
    return counts.most_common(1)[0][0] if counts else None


def _smallest_readable_size(lines: Sequence[Line], min_chars: int = 20) -> float | None:
    """Smallest size carrying at least ``min_chars`` visible characters, so
    a stray superscript or page number does not count as small text."""
    counts: Counter = Counter()
    for line in lines:
        counts[line.size] += line.visible
    sizes = [size for size, n in counts.items() if n >= min_chars]
    return min(sizes) if sizes else None


def _percentile(values: Sequence[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def _finding(check: str, page: int, region: str, problem: str, severity: str = "medium", **extra) -> dict:
    return {
        "check": check,
        "page": page,
        "region": region,
        "problem": problem,
        "severity": severity,
        "source": "measure",
        **extra,
    }


def _outside(box: Box, page: Page) -> float:
    """How far (points) a box reaches past the page edge; 0 when inside."""
    return max(0.0, -box[0], -box[1], box[2] - page.width, box[3] - page.height)


def _overflow_findings(page: Page, *, region: str = "page") -> list[dict]:
    """Content reaching past the page edge. Even a figure 2 pt over is cut
    visibly (its frame and axis label lose their bottom edge), so any
    overhang past 1 pt counts; losing a tenth of the object is high.

    Cut-off text is one finding per page, however many lines ran over: a
    column that overflows by a page's worth is one problem to fix, not
    three hundred."""
    found = []
    cut = [
        (line, _outside(line.box, page)) for line in page.lines
        if line.visible and _outside(line.box, page) > 1.0
    ]
    if cut:
        worst = max(beyond for _line, beyond in cut)
        high = any(beyond > 0.25 * max(1.0, line.box[3] - line.box[1]) for line, beyond in cut)
        count = (
            f"{len(cut)} lines of text run past the page edge and are cut off"
            if len(cut) > 1 else "A line of text runs past the page edge and is cut off"
        )
        found.append(_finding(
            "overflow", page.number, region,
            f"{count} (up to {worst:.0f} pt), "
            f"starting with \"{cut[0][0].text[:60]}\".",
            "high" if high else "medium",
            lines_cut=len(cut),
            overhang_pt=round(worst, 1),
        ))
    for image in page.images:
        if image[2] - image[0] >= 0.9 * page.width or image[3] - image[1] >= 0.9 * page.height:
            # A full-bleed decoration (a slide's accent bar, a background)
            # is meant to run off the edge; LibreOffice exports the pptx
            # accent bar 3 pt wider than the slide on each side.
            continue
        beyond = _outside(image, page)
        if beyond > 1.0:
            found.append(_finding(
                "overflow", page.number, region,
                f"A figure runs {beyond:.0f} pt past the page edge and is cut off.",
                "high" if beyond > 0.10 * (image[3] - image[1]) else "medium",
            ))
    return found


def _figure_over_text_findings(page: Page, *, region: str = "page") -> list[dict]:
    """Text a figure is drawn over, one finding per figure. The pptx placed a
    figure over a slide's third bullet, and only the model's look at the
    screenshot noticed. A line the figure covers by more than half its height
    cannot be read: high."""
    found = []
    for image in page.images:
        if image[2] - image[0] >= 0.9 * page.width or image[3] - image[1] >= 0.9 * page.height:
            continue  # a full-bleed decoration sits behind everything
        covered = []
        for line in page.lines:
            if not line.visible:
                continue
            across = min(line.box[2], image[2]) - max(line.box[0], image[0])
            down = min(line.box[3], image[3]) - max(line.box[1], image[1])
            if across > 2.0 and down > 2.0:
                covered.append((line, down / max(1.0, line.box[3] - line.box[1])))
        if covered:
            count = f"{len(covered)} lines of text" if len(covered) > 1 else "a line of text"
            found.append(_finding(
                "overlap", page.number, region,
                f"A figure is drawn over {count}, starting with \"{covered[0][0].text[:60]}\".",
                "high" if any(share > 0.5 for _line, share in covered) else "medium",
                lines_covered=len(covered),
            ))
    return found


# A figure closer than this under a line of text reads as part of that line.
# The Marp deck leaves 17 pt; the pptx left 0.1 pt before its text estimate
# was corrected.
_FIGURE_GAP_MIN_PT = 6.0


def _figure_under_text_findings(page: Page, *, region: str = "page") -> list[dict]:
    """A figure that starts right under a line of text, one finding per
    figure. Nothing is drawn over the text, so the overlap check passes, yet
    the figure sits on the line. Low: it is the renderer's spacing, which a
    new version of the words would not change."""
    found = []
    for image in page.images:
        if image[2] - image[0] >= 0.9 * page.width or image[3] - image[1] >= 0.9 * page.height:
            continue
        if image[3] - image[1] < 24.0:
            continue  # a logo or an icon, not a figure
        close = []
        for line in page.lines:
            across = min(line.box[2], image[2]) - max(line.box[0], image[0])
            # PDF y runs up: the gap from the line's bottom down to the figure's top.
            gap = line.box[1] - image[3]
            if line.visible and across > 2.0 and -2.0 <= gap < _FIGURE_GAP_MIN_PT:
                close.append((gap, line))
        if close:
            gap, line = min(close, key=lambda item: item[0])
            found.append(_finding(
                "figure_gap", page.number, region,
                f"A figure starts {max(gap, 0.0):.0f} pt under \"{line.text[:60]}\", with no room between them.",
                "low",
                gap_pt=round(gap, 1),
            ))
    return found


# LaTeX a slide prints instead of typesetting: a $...$ span by pandoc's rule
# (a non-space inside each $, no digit right after the closing one, so "$5 to
# $10" is not one) or a bare math command.
_RAW_MATH_RE = re.compile(
    r"(?<!\\)\$(?=\S)[^$]*?[^\s\\]\$(?!\d)"
    r"|\\(?:times|frac|text|mathrm|sqrt|cdot|pm|leq?|geq?|approx|infty|exp|log|sum|int|partial"
    r"|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|pi)\b"
)


def _raw_math_findings(page: Page, *, region: str = "page") -> list[dict]:
    """One finding per page whose text layer shows LaTeX math as text. The
    pptx printed every `$h = 0.5$` of the validation deck, and only the
    model's look at the screenshots noticed."""
    shown = [line for line in page.lines if line.visible and _RAW_MATH_RE.search(line.text)]
    if not shown:
        return []
    count = f"{len(shown)} lines show" if len(shown) > 1 else "A line shows"
    return [_finding(
        "raw_markup", page.number, region,
        f"{count} LaTeX math as text instead of a formula, starting with \"{shown[0].text[:60]}\".",
        lines_with_latex=len(shown),
    )]


# ---------------------------------------------------------------------------
# Poster


def poster_report(
    doc: Document,
    *,
    columns: int | None = None,
    word_budget: int | None = None,
    header_terms: Sequence[str] = (),
) -> dict:
    """Measure page 1 of a poster against the published standards.

    Regions come from the template's full-width rules: the highest rule
    below the title closes the header, and any full-width rule below that
    opens the references band. ``columns`` defaults to 2 on a portrait
    sheet and 3 on a landscape one. ``header_terms`` (author, affiliation,
    contact) must each appear in the header text.
    """
    page = doc.pages[0]
    lines = [line for line in page.lines if line.visible]
    if columns is None:
        columns = 3 if page.width > page.height else 2

    title = max(
        (line for line in lines if line.visible >= 3),
        key=lambda line: line.size,
        default=None,
    )
    title_bottom = title.box[1] if title else page.height
    below_title = [rule for rule in page.rules if rule[3] <= title_bottom + 1]
    # A rule in the top half closes the header and one in the bottom half
    # opens the band, so a sheet with no header rule does not take the band
    # rule for the header's edge and call the whole sheet its header.
    middle = page.height / 2
    header_bottom = max(
        (rule[1] for rule in below_title if rule[1] >= middle), default=title_bottom,
    )
    band_rules = [rule for rule in below_title if rule[3] < min(middle, header_bottom - 1)]
    band_top = max((rule[3] for rule in band_rules), default=None)
    floor = band_top if band_top is not None else 0.0

    def centre_y(box: Box) -> float:
        return (box[1] + box[3]) / 2

    header = [line for line in lines if centre_y(line.box) >= header_bottom]
    in_band = [
        line for line in lines
        if band_top is not None and 0 <= centre_y(line.box) <= band_top
    ]
    body = [line for line in lines if floor < centre_y(line.box) < header_bottom]
    figures = [
        image for image in page.images
        if floor < centre_y(image) < header_bottom
        and _area(image) >= 0.01 * page.width * page.height
    ]

    # Columns: split the width the body content actually spans.
    items = [line.box for line in body] + figures
    left = min((box[0] for box in items), default=0.0)
    right = max((box[2] for box in items), default=page.width)
    slice_width = max(1.0, (right - left) / columns)
    bottoms = [header_bottom] * columns
    for box in items:
        k = min(columns - 1, max(0, int(((box[0] + box[2]) / 2 - left) / slice_width)))
        bottoms[k] = min(bottoms[k], box[1])

    captions = [line for line in body if _CAPTION_RE.match(line.text)]
    body_pt = _size_mode([
        line for line in body if not line.bold and line not in captions
    ])
    headings = [
        line for line in body
        if line.bold and line.visible <= 90 and line not in captions
        and body_pt is not None and line.size >= 0.95 * body_pt
    ]
    heading_pt = _size_mode(headings)
    caption_pt = _size_mode(captions)
    title_pt = max((line.size for line in header), default=title.size if title else None)
    # Column text that ran down into the band keeps its body size; the
    # reference list itself is set smaller.
    spilled = [
        line for line in in_band
        if body_pt is not None and line.size >= 0.95 * body_pt
    ]
    band = [line for line in in_band if line not in spilled]
    # A figure caption that slid down is set larger than the reference
    # list around it.
    references_pt = _size_mode(band)
    if references_pt is not None:
        spilled += [line for line in band if line.size > 1.1 * references_pt]
        band = [line for line in band if line.size <= 1.1 * references_pt]
    spilled_figures = [
        image for image in figures if band_top is not None and image[1] < band_top - 1
    ]
    references_pt = _size_mode(band)
    body_lengths = [
        len(line.text) for line in body
        if body_pt is not None and not line.bold and abs(line.size - body_pt) <= 0.05 * body_pt
        and len(line.text) >= 20
    ]
    chars_per_line = _percentile(body_lengths, 0.75)
    band_text = " ".join(line.text for line in band)
    reference_labels = sorted(set(_REFERENCE_LABEL_RE.findall(band_text)))
    words = sum(line.words for line in lines)
    content_bottom = min(
        [line.box[1] for line in lines] + [image[1] for image in page.images],
        default=0.0,
    )
    region_height = max(1.0, header_bottom - floor)
    figure_share = sum(_area(box) for box in figures) / max(1.0, region_height * (right - left))

    metrics = {
        "page_cm": [_cm(page.width), _cm(page.height)],
        "columns": columns,
        "title_pt": title_pt,
        "heading_pt": heading_pt,
        "body_pt": body_pt,
        "caption_pt": caption_pt,
        "references_pt": references_pt,
        "smallest_pt": _smallest_readable_size(lines),
        "words": words,
        "chars_per_line": chars_per_line,
        "column_bottoms_cm": [_cm(page.height - b) for b in bottoms],
        "column_gap_cm": _cm(max(bottoms) - min(bottoms)),
        "space_above_band_cm": _cm(min(bottoms) - floor),
        "empty_bottom_cm": _cm(max(0.0, content_bottom)),
        "figures": len(figures),
        "captions": len(captions),
        "figure_share": round(figure_share, 2),
        "references": len(reference_labels),
        # Region edges from the top of the sheet, for a layout that wants to
        # measure how much room its columns had.
        "header_bottom_cm": _cm(page.height - header_bottom),
        "band_top_cm": None if band_top is None else _cm(page.height - band_top),
    }

    findings = _overflow_findings(page)
    n = page.number
    if spilled or spilled_figures:
        what = []
        if spilled:
            what.append(f"{len(spilled)} lines of text" if len(spilled) > 1 else "a line of text")
        if spilled_figures:
            what.append(f"{len(spilled_figures)} figures" if len(spilled_figures) > 1 else "a figure")
        findings.append(_finding(
            "band_overlap", n, "references band",
            f"Column content runs down into the references band: {' and '.join(what)}.",
            "high",
        ))

    def too_small(check: str, key: str, value: float | None, region: str, what: str) -> None:
        floor_pt = POSTER_MIN_PT[key]
        if value is not None and value < floor_pt - 0.5:
            findings.append(_finding(
                check, n, region,
                f"{what} is {value:g} pt; posters need at least {floor_pt:g} pt.",
            ))

    too_small("title_font", "title", title_pt, "header", "The title")
    too_small("body_font", "body", body_pt, "columns", "Body text")
    too_small("caption_font", "caption", caption_pt, "columns", "Figure captions")
    too_small("references_font", "references", references_pt, "references band", "The reference list")
    if body_pt is not None:
        if heading_pt is None:
            findings.append(_finding(
                "headings", n, "columns",
                "No section headings were found in the columns.", "low",
            ))
        else:
            needed = max(POSTER_MIN_PT["heading"], POSTER_HEADING_RATIO * body_pt)
            if heading_pt < needed - 0.5:
                findings.append(_finding(
                    "heading_font", n, "columns",
                    f"Section headings are {heading_pt:g} pt against {body_pt:g} pt body text; "
                    f"they need at least {needed:.0f} pt ({POSTER_HEADING_RATIO:g}x body, "
                    f"and {POSTER_MIN_PT['heading']:g} pt).",
                ))
    if figures and len(captions) < len(figures):
        findings.append(_finding(
            "captions", n, "columns",
            f"{len(figures)} figures but {len(captions)} numbered captions (\"Figure N: ...\").",
        ))
    budget = word_budget or POSTER_MAX_WORDS
    if words > budget:
        findings.append(_finding(
            "word_count", n, "page",
            f"The poster has {words} words; the budget for this size is {budget}.",
        ))
    if chars_per_line is not None and chars_per_line > POSTER_MAX_CHARS_PER_LINE:
        findings.append(_finding(
            "line_length", n, "columns",
            f"Body lines hold about {chars_per_line} characters; keep them to "
            f"{POSTER_MAX_CHARS_PER_LINE} or fewer.",
        ))
    gap = max(bottoms) - min(bottoms)
    if gap > max(5 * _PT_PER_CM, 0.10 * region_height):
        short = bottoms.index(max(bottoms)) + 1
        findings.append(_finding(
            "column_balance", n, f"column {short}",
            f"Column {short} ends {_cm(gap)} cm above the longest column; "
            "the columns should end at about the same height.",
            "low" if gap <= 0.20 * region_height else "medium",
        ))
    empty = max(min(bottoms) - floor, content_bottom)
    if empty > 0.06 * page.height:
        findings.append(_finding(
            "empty_space", n, "page bottom",
            f"{_cm(empty)} cm of the sheet is left empty below the content.",
        ))
    if len(reference_labels) > POSTER_MAX_REFERENCES:
        findings.append(_finding(
            "references_count", n, "references band",
            f"The reference list has {len(reference_labels)} entries; a poster lists at most "
            f"{POSTER_MAX_REFERENCES}, only the ones it cites.",
        ))
    if _URL_RE.search(band_text):
        findings.append(_finding(
            "references_urls", n, "references band",
            "The reference list prints raw URLs; give author, year, title and DOI instead.",
        ))
    header_text = " ".join(" ".join(line.text for line in header).lower().split())
    for term in header_terms:
        wanted = " ".join((term or "").lower().split())
        if wanted and wanted not in header_text:
            findings.append(_finding(
                "header_info", n, "header",
                f"The header does not show \"{term}\".",
            ))
    return {"kind": "poster", "metrics": metrics, "findings": findings}


# ---------------------------------------------------------------------------
# Slides


def slides_report(doc: Document) -> dict:
    """Per-slide overflow and text too small to read when projected."""
    findings: list[dict] = []
    per_page = []
    for page in doc.pages:
        lines = [line for line in page.lines if line.visible]
        smallest = _smallest_readable_size(lines)
        per_page.append({
            "page": page.number,
            "body_pt": _size_mode(lines),
            "smallest_pt": smallest,
            "words": sum(line.words for line in lines),
            "figures": len(page.images),
        })
        findings += _overflow_findings(page, region="slide")
        findings += _figure_over_text_findings(page, region="slide")
        findings += _figure_under_text_findings(page, region="slide")
        findings += _raw_math_findings(page, region="slide")
        if smallest is not None and smallest < SLIDE_MIN_PT - 0.5:
            findings.append(_finding(
                "small_font", page.number, "slide",
                f"Text is set at {smallest:g} pt; slides need at least {SLIDE_MIN_PT:g} pt "
                "to be read from the back of a room.",
            ))
    return {
        "kind": "slides",
        "metrics": {"pages": len(doc.pages), "per_page": per_page},
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Paper


def paper_report(doc: Document) -> dict:
    """Text or figures wider than the text block (an overfull table or a
    figure LaTeX could not fit), content off the page, and text too small.

    The text block is found from the lines themselves: justified lines end
    at the same x, so the edges that many lines share are the margins. In
    a two-column layout both columns' edges are shared, and the outermost
    shared edges bound the block.
    """
    lines = [line for page in doc.pages for line in page.lines if line.visible]
    body_pt = _size_mode(lines)
    long_lines = [
        line for line in lines
        if body_pt is not None and abs(line.size - body_pt) <= 0.05 * body_pt
        and len(line.text) >= 40
    ]
    block = _shared_edges(long_lines)
    findings: list[dict] = []
    for page in doc.pages:
        findings += _overflow_findings(page)
        page_lines = [line for line in page.lines if line.visible]
        smallest = _smallest_readable_size(page_lines)
        if smallest is not None and smallest < PAPER_MIN_PT - 0.5:
            findings.append(_finding(
                "small_font", page.number, "page",
                f"Text is set at {smallest:g} pt, too small to read in print.",
            ))
        if block is None:
            continue
        left_edge, right_edge = block
        for line in page_lines:
            over = max(line.box[2] - right_edge, left_edge - line.box[0])
            if over > 3.0 and _outside(line.box, page) == 0:
                findings.append(_finding(
                    "overwide", page.number, "text block",
                    f"A line runs {over:.0f} pt into the margin: \"{line.text[:60]}\".",
                    object="text", overhang_pt=round(over, 1),
                ))
        for image in page.images:
            over = max(image[2] - right_edge, left_edge - image[0])
            if over > 3.0 and _outside(image, page) == 0:
                findings.append(_finding(
                    "overwide", page.number, "text block",
                    f"A figure runs {over:.0f} pt into the margin.",
                    object="figure", overhang_pt=round(over, 1),
                ))
    half_empty = 0
    for page in doc.pages[:-1]:
        empty = _empty_share(page)
        if empty >= HALF_EMPTY_SHARE:
            half_empty += 1
            findings.append(_finding(
                "half_empty_page", page.number, "page",
                f"{empty:.0%} of the page's text area is empty, and the paper goes on to the next page.",
            ))
    last_lines = _last_page_text_lines(doc)
    if len(doc.pages) >= 2 and last_lines <= 2:
        findings.append(_finding(
            "last_page_nearly_empty", doc.pages[-1].number, "last page",
            f"The last page holds only {last_lines} line(s) of text; the rest of it is empty.",
        ))
    return {
        "kind": "paper",
        "metrics": {
            "pages": len(doc.pages),
            "body_pt": body_pt,
            "text_block_pt": list(block) if block else None,
            "words": sum(line.words for line in lines),
            "last_page_lines": last_lines,
            "half_empty_pages": half_empty,
        },
        "findings": findings,
    }


# A page before the last with this much of its text area empty looks
# unfinished. The validation quest's paper had two: a figure alone on its page,
# centred, with about a fifth of the page blank above it and a fifth below.
HALF_EMPTY_SHARE = 0.35
# Empty stretches shorter than this share of the text area are the space
# between lines, paragraphs, headings and captions, not an empty page.
_EMPTY_STRETCH_MIN = 0.05


def _empty_share(page: Page) -> float:
    """How much of the page's text area is empty, as a share of that area's
    height: every empty stretch above, between and below what is on the page
    that is at least ``_EMPTY_STRETCH_MIN`` of the area. The top and bottom 8%
    of the page (running header, footer, page number, footer mark) are not
    part of the area."""
    top, bottom = 0.08 * page.height, 0.92 * page.height
    height = bottom - top
    boxes = [line.box for line in page.lines if line.visible] + list(page.images)
    spans = sorted((max(b[1], top), min(b[3], bottom)) for b in boxes if b[3] > top and b[1] < bottom)
    empty, reach = 0.0, top
    for start, end in [*spans, (bottom, bottom)]:
        if start - reach >= _EMPTY_STRETCH_MIN * height:
            empty += start - reach
        reach = max(reach, end)
    return empty / height


def _last_page_text_lines(doc: Document) -> int:
    """Lines of text on the last page. A running header or footer (a line in
    the top or bottom 8% of the page at a height that recurs on another page)
    and a bare page number there are not counted."""
    def in_margin(line: Line, page: Page) -> bool:
        band = 0.08 * page.height
        return line.box[3] < band or line.box[1] > page.height - band

    heights: Counter[int] = Counter()
    for page in doc.pages:
        heights.update({round(line.box[1]) for line in page.lines if line.visible and in_margin(line, page)})
    last = doc.pages[-1]
    count = 0
    for line in last.lines:
        if not line.visible:
            continue
        if in_margin(line, last) and (
            line.text.strip().isdigit()
            or any(heights[y] >= 2 for y in range(round(line.box[1]) - 1, round(line.box[1]) + 2))
        ):
            continue
        count += 1
    return count


def _shared_edges(lines: Sequence[Line]) -> tuple[float, float] | None:
    """Left and right edges at least a quarter of the lines share."""
    if len(lines) < 8:
        return None

    def shared(values: list[float]) -> list[float]:
        counts = Counter(round(v / 2) * 2 for v in values)
        return [edge for edge, n in counts.items() if n >= 0.25 * len(values)]

    lefts = shared([line.box[0] for line in lines])
    rights = shared([line.box[2] for line in lines])
    if not lefts or not rights:
        return None
    return (min(lefts) - 2.0, max(rights) + 2.0)
