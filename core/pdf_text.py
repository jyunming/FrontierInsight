"""Text out of a PDF: every page, in reading order, with scanned pages read by OCR.

Every PDF FI reads (a paper fetched during the literature step, a paper the person drops into ``inputs/papers/`` or
lists in ``knowledge.local_papers``) goes through :func:`extract`. What it replaced was pypdf's ``extract_text()``
alone, which split words mid-way ("p roperly"), returned nothing for a scanned page (most papers from before the
1990s are page images, the classics a quest most needs among them) and was then capped at 64 KB, so a long paper
lost its results and discussion.

- Text: PyMuPDF when the person has installed it (``pip install pymupdf``; not a dependency of FI, whose licence is
  Apache-2.0 while PyMuPDF's is AGPL-3.0), otherwise pypdfium2 (a dependency already). Words split by a line-end
  hyphen are joined, ligatures are spelt out, and (on the pypdfium2 path) a two-column page whose text was stored
  line by line across the columns is read column by column.
- Scanned pages (almost no text layer) are rendered and read by OCR: tesseract when its program is installed and
  ``pytesseract`` is importable, otherwise RapidOCR (``pip install rapidocr onnxruntime``; its default models, about
  31 MB, are inside the wheel, so it runs offline once installed). With neither, the page is counted as unread and the result says so;
  it is never silently empty. No language model is involved.
- ``max_bytes`` bounds the text (10 MB by default); a PDF cut there says at which page.
- Text drawn so a reader cannot see it (in white on a white page, invisibly, or at a size below
  :data:`HIDDEN_SIZE_PT`) is kept in the text like the rest and also listed in ``hidden_text``: it reaches the model
  all the same, and it is how instructions are planted in a paper for a model to read (core/source_text.py flags it).
  White or invisible text over a clearly darker fill or over any image is not counted: a label on a dark figure, a
  journal's banner, and the searchable layer a scan (or a browser printing its text as images) lays over the page
  image. Not found, then: white or invisible text placed over an image (even a plain white one), near-white text,
  text covered by a white box drawn after it, and (on the pypdfium2 path) text inside an included figure.
"""

from __future__ import annotations

import io
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Characters of text below which a page counts as scanned (a page number and a running head fit under this).
SCANNED_PAGE_CHARS = 40
#: Render scale for OCR: 1 is 72 dpi, so 144 dpi. Measured on an image-only copy of a real two-column paper:
#: 98.6% of its words recovered at about 7 s a page with RapidOCR; more resolution only costs time.
OCR_SCALE = 2.0

# Only a hyphen the typesetter added: pdfium marks one with U+FFFE, and U+00AD is the soft hyphen. A plain "-" at a
# line end may be a real compound ("well-known"), so it is kept.
_SOFT_BREAK = re.compile(r"[\u00ad\ufffe]\r?\n?(?=[A-Za-z])")
_SPACES = re.compile(r"[ \t\u00a0]+")


@dataclass
class PdfText:
    """What :func:`extract` recovered from one PDF."""

    text: str = ""
    pages: int = 0
    engine: str = ""                       # "pymupdf" | "pdfium" | "" (unreadable)
    ocr_engine: str = ""                   # "tesseract" | "rapidocr" | "" (none used)
    ocr_pages: list[int] = field(default_factory=list)       # 1-based pages read by OCR
    unread_pages: list[int] = field(default_factory=list)    # 1-based scanned pages no OCR engine could read
    truncated_at_page: int | None = None   # the page the size cap stopped at, or None
    ocr_out_of_time: bool = False          # OCR stopped at its deadline; the rest are in unread_pages
    error: str = ""
    #: Each OCR-read page's lines with their place, ``{page: [((left, bottom, right, top) in PDF points, text)]}``,
    #: for core/pdf_figures.py to find a scanned page's figures without reading the page twice.
    ocr_lines: dict[int, list[tuple[tuple[float, float, float, float], str]]] = field(default_factory=dict)
    #: Passages drawn so a reader cannot see them (white on the page, or tiny); see the module docstring.
    hidden_text: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One plain line for the log and the file header."""
        parts = [f"{self.pages} page(s) via {self.engine or 'nothing'}"]
        if self.ocr_pages:
            parts.append(f"{len(self.ocr_pages)} scanned page(s) read by OCR ({self.ocr_engine})")
        if self.unread_pages:
            why = ("OCR ran out of time" if self.ocr_out_of_time else
                   "no OCR engine: pip install rapidocr onnxruntime, or install tesseract and pip install pytesseract")
            parts.append(f"{len(self.unread_pages)} scanned page(s) not read ({why})")
        if self.truncated_at_page:
            parts.append(f"cut at page {self.truncated_at_page} by the size limit")
        if self.hidden_text:
            words = sum(len(t.split()) for t in self.hidden_text)
            parts.append(f"{words} word(s) drawn so a reader cannot see them (white on the page, or tiny)")
        if self.error:
            parts.append(f"error: {self.error}")
        return "; ".join(parts)


def clean(text: str) -> str:
    """Join words split by a line-end hyphen, spell out ligatures (NFKC), and tidy spaces; line breaks are kept."""
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _SOFT_BREAK.sub("", text)
    text = text.replace("\ufffe", "").replace("\u00ad", "")
    return "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n")).strip()


def _open(source: bytes | Path | str) -> tuple[str, Any]:
    """Open with PyMuPDF if it is installed, else pypdfium2. Returns ``(engine, document)``."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF: optional, AGPL, installed by the person

        return "pymupdf", fitz.open(stream=data, filetype="pdf")
    except ImportError:
        pass
    import pypdfium2 as pdfium

    return "pdfium", pdfium.PdfDocument(data)


def _page_text_pymupdf(page: Any) -> tuple[str, bool]:
    text = page.get_text("text") or ""
    has_images = bool(page.get_images(full=False))
    return text, has_images


def _page_text_pdfium(page: Any) -> tuple[str, bool]:
    import pypdfium2.raw as pdfium_c

    textpage = page.get_textpage()
    try:
        text = _pdfium_reading_order(page, textpage)
    finally:
        textpage.close()
    has_images = any(obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE for obj in page.get_objects(max_depth=1))
    return text, has_images


def _pdfium_reading_order(page: Any, textpage: Any) -> str:
    """The page's text in stream order, unless the stream alternates between two columns line by line; then the
    text runs are re-read column by column (full-width runs, such as a title, keep their place above)."""
    text = textpage.get_text_range() or ""
    width = page.get_width()
    n = textpage.count_rects()
    if n < 8 or width <= 0:
        return text
    rects = [textpage.get_rect(i) for i in range(n)]  # (left, bottom, right, top)
    mid = width / 2
    side = []
    for left, _bottom, right, _top in rects:
        if right <= mid + width * 0.02:
            side.append("L")
        elif left >= mid - width * 0.02:
            side.append("R")
        else:
            side.append("W")
    columns = [s for s in side if s != "W"]
    if len(columns) < 8 or "L" not in columns or "R" not in columns:
        return text
    switches = sum(1 for a, b in zip(columns, columns[1:]) if a != b)
    if switches <= 2:  # stored column by column already (what LaTeX writes): stream order is reading order
        return text
    order = sorted(range(n), key=lambda i: ({"W": 0, "L": 1, "R": 2}[side[i]], -rects[i][3], rects[i][0]))
    lines = []
    for i in order:
        left, bottom, right, top = rects[i]
        lines.append(textpage.get_text_bounded(left, bottom, right, top))
    return "\n".join(lines)


#: Text drawn below this size (points, with the page's scaling) counts as hidden. Real small print (a figure's axis
#: label, a copyright line) is 4 pt or more.
HIDDEN_SIZE_PT = 2.0
#: A colour with every channel at least this (0-255) counts as white.
_WHITE = 250
#: A fill counts as a backdrop that white or invisible text can sit on only when it is clearly darker than white: its
#: relative luminance (0 black, 1 white) below this, which leaves white text a contrast of at least 1.4. A page-sized
#: rectangle in 249 grey is a white page, and bright yellow, cyan or lime under white text hide it about as well as
#: white does; grey 200, orange and a brochure's leaf green count as dark (white on them is read).
_DARK = 0.7
#: At most this many hidden passages are kept per PDF (each cut to 200 characters).
_HIDDEN_MAX = 20


def _dark(r: float, g: float, b: float) -> bool:
    """Whether a colour (channels 0-255) is clearly darker than white: WCAG relative luminance below :data:`_DARK`."""
    def linear(c: float) -> float:
        c = max(0.0, min(255.0, float(c))) / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b) < _DARK


def _inside(box: tuple[float, float, float, float], boxes: list[tuple[float, float, float, float]]) -> bool:
    """Whether the middle of ``box`` lies within one of ``boxes`` (any corner order)."""
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return any(min(b[0], b[2]) <= x <= max(b[0], b[2]) and min(b[1], b[3]) <= y <= max(b[1], b[3]) for b in boxes)


def _is_hidden(*, invisible: bool, tiny: bool, white: bool, covered: bool) -> bool:
    """A run of text no reader sees: drawn invisibly, tiny, or white, and (unless tiny) not over an image or a
    coloured fill. An invisible layer over a page image is how a scan (and some browsers' print-to-PDF) makes the page
    searchable, so it does not count; an invisible run on a bare page does."""
    if invisible:
        return not covered
    return tiny or (white and not covered)


def _hidden_pymupdf(page: Any) -> list[str]:
    backdrops: list[tuple[float, float, float, float]] = []
    for d in page.get_drawings():
        fill = d.get("fill")
        opacity = d.get("fill_opacity")
        if fill and _dark(*(255 * c for c in fill[:3])) and (1.0 if opacity is None else float(opacity)) > 0:
            backdrops.append(tuple(d["rect"]))
    backdrops += [tuple(i["bbox"]) for i in page.get_image_info()]
    out: list[str] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = " ".join(str(span.get("text") or "").split())
                if not text:
                    continue
                color = int(span.get("color") or 0)
                if _is_hidden(invisible=span.get("alpha", 255) == 0,
                              tiny=float(span.get("size") or 12) < HIDDEN_SIZE_PT,
                              white=min((color >> 16) & 255, (color >> 8) & 255, color & 255) >= _WHITE,
                              covered=_inside(tuple(span["bbox"]), backdrops)):
                    out.append(text)
    return out


def _hidden_pdfium(page: Any) -> list[str]:
    """The pypdfium2 reading. Only the page's own objects are read (``max_depth=1``): an object inside a form (a
    figure included as one) has its size and place in the form's own coordinates, which would misread both."""
    import ctypes

    import pypdfium2.raw as pdfium_c

    def colour_of(obj: Any, *, stroke: bool = False) -> tuple[int, int, int, int] | None:
        r, g, b, a = (ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint())
        get = pdfium_c.FPDFPageObj_GetStrokeColor if stroke else pdfium_c.FPDFPageObj_GetFillColor
        if not get(obj.raw, r, g, b, a):
            return None
        return r.value, g.value, b.value, a.value

    # Mode 7 (text used only as a clip) is painted by what is drawn through it, often a gradient title: not counted.
    clip_only = pdfium_c.FPDF_TEXTRENDERMODE_CLIP
    stroked_modes = {pdfium_c.FPDF_TEXTRENDERMODE_STROKE, pdfium_c.FPDF_TEXTRENDERMODE_STROKE_CLIP}
    both_modes = {pdfium_c.FPDF_TEXTRENDERMODE_FILL_STROKE, pdfium_c.FPDF_TEXTRENDERMODE_FILL_STROKE_CLIP}
    page_area = abs(float(page.get_width()) * float(page.get_height())) or 1.0

    def area(box: tuple[float, float, float, float]) -> float:
        return abs((box[2] - box[0]) * (box[3] - box[1]))
    objects = list(page.get_objects(max_depth=1))

    backdrops: list[tuple[float, float, float, float]] = []
    for obj in objects:
        if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            backdrops.append(tuple(obj.get_bounds()))
        elif obj.type in (pdfium_c.FPDF_PAGEOBJ_PATH, pdfium_c.FPDF_PAGEOBJ_FORM):
            if obj.type == pdfium_c.FPDF_PAGEOBJ_FORM:
                # A form (an included figure): what it draws is not read here, so a figure-sized one counts as a
                # backdrop. One as large as the page (a stamped or wrapped page) would hide everything, so it counts
                # only when the page holds an image inside a form: a scan wrapped whole, with its text layer on top.
                if area(tuple(obj.get_bounds())) >= page_area / 2 and not any(
                        o.type == pdfium_c.FPDF_PAGEOBJ_IMAGE for o in page.get_objects(max_depth=2, form=obj)):
                    continue
                backdrops.append(tuple(obj.get_bounds()))
                continue
            mode, stroke = ctypes.c_int(), ctypes.c_int()
            colour = colour_of(obj)
            if (colour and _dark(*colour[:3]) and colour[3] > 0
                    and pdfium_c.FPDFPath_GetDrawMode(obj.raw, ctypes.byref(mode), ctypes.byref(stroke)) and mode.value):
                backdrops.append(tuple(obj.get_bounds()))
    out: list[str] = []
    textpage = None
    try:
        for obj in objects:
            if obj.type != pdfium_c.FPDF_PAGEOBJ_TEXT:
                continue
            mode = pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw)
            if mode == clip_only:
                continue
            colour = colour_of(obj, stroke=mode in stroked_modes)  # outlined text is seen by its outline
            size = ctypes.c_float()
            if colour is None or not pdfium_c.FPDFTextObj_GetFontSize(obj.raw, ctypes.byref(size)):
                continue
            white = min(colour[:3]) >= _WHITE and colour[3] > 0
            if white and mode in both_modes:  # filled and outlined: white only when the outline is white too
                outline = colour_of(obj, stroke=True)
                white = outline is None or (min(outline[:3]) >= _WHITE and outline[3] > 0)
            m = obj.get_matrix()
            drawn = size.value * (abs(m.a * m.d - m.b * m.c) ** 0.5 or 1.0)
            if not _is_hidden(invisible=mode == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE or colour[3] == 0,
                              tiny=drawn < HIDDEN_SIZE_PT, white=white,
                              covered=_inside(tuple(obj.get_bounds()), backdrops)):
                continue
            if textpage is None:
                textpage = page.get_textpage()
            n = pdfium_c.FPDFTextObj_GetText(obj.raw, textpage.raw, None, 0)
            if n <= 2:
                continue
            buf = (ctypes.c_ushort * (n // 2 + 1))()
            pdfium_c.FPDFTextObj_GetText(obj.raw, textpage.raw, buf, n)
            text = " ".join(bytes(buf)[:n - 2].decode("utf-16-le", errors="ignore").split())
            if text:
                out.append(text)
    finally:
        if textpage is not None:
            textpage.close()
    return out


def _hidden_text(engine: str, page: Any) -> list[str]:
    """What ``page`` draws so a reader cannot see it; ``[]`` when that cannot be read (never raises)."""
    try:
        return (_hidden_pymupdf if engine == "pymupdf" else _hidden_pdfium)(page)
    except Exception:  # noqa: BLE001 -- a check on top of the text; the text is what the PDF is read for
        return []


def _page_height(engine: str, page: Any) -> float:
    return float(page.rect.height) if engine == "pymupdf" else float(page.get_height())


def _render_png(engine: str, doc: Any, index: int) -> bytes:
    if engine == "pymupdf":
        return doc[index].get_pixmap(matrix=__import__("fitz").Matrix(OCR_SCALE, OCR_SCALE)).tobytes("png")
    image = doc[index].render(scale=OCR_SCALE).to_pil()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class _Ocr:
    """The OCR engine to use, found once per extraction: tesseract, else RapidOCR, else none."""

    def __init__(self) -> None:
        self.name = ""
        self._engine: Any = None
        try:
            import pytesseract  # type: ignore[import-not-found]

            if shutil.which("tesseract"):
                self.name, self._engine = "tesseract", pytesseract
                return
        except ImportError:
            pass
        try:
            from rapidocr import RapidOCR  # type: ignore[import-not-found]  # rapidocr >= 2 (default models ship in the wheel)

            self._engine, self.name = RapidOCR(), "rapidocr"
            return
        except Exception:  # noqa: BLE001 -- not installed, or its models could not be fetched
            pass
        # Not the older ``rapidocr_onnxruntime``: measured on an image-only copy of a real paper, its default model
        # dropped the spaces between English words ("presentaresiduallearningframework") at about 45 s a page, text a
        # quote check cannot match. Such a page is counted as unread instead.
        self.name = ""

    def read(self, png: bytes) -> tuple[str, list[tuple[Any, str]]]:
        """The page's text in reading order, and its lines as ``(corner points in pixels, text)``."""
        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGB")
        if self.name == "tesseract":
            return self._engine.image_to_string(image), _tesseract_lines(self._engine, image)
        if self.name == "rapidocr":
            result = self._engine(image)
            items = list(zip(result.boxes if result.boxes is not None else [], result.txts or ()))
            return ocr_lines(items, width=image.width), items
        return "", []


def _tesseract_lines(engine: Any, image: Any) -> list[tuple[Any, str]]:
    """tesseract's words grouped into its own lines, as ``(corner points in pixels, text)``."""
    try:
        data = engine.image_to_data(image, output_type=engine.Output.DICT)
    except Exception:  # noqa: BLE001 -- the text was read; only the places are missing
        return []
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data.get("text") or []):
        if str(word).strip():
            lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(i)
    out = []
    for idx in lines.values():
        x0 = min(data["left"][i] for i in idx)
        y0 = min(data["top"][i] for i in idx)
        x1 = max(data["left"][i] + data["width"][i] for i in idx)
        y1 = max(data["top"][i] + data["height"][i] for i in idx)
        out.append(([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], " ".join(str(data["text"][i]) for i in idx)))
    return out


def _to_points(items: list[tuple[Any, str]], page_height: float) -> list[tuple[tuple[float, float, float, float], str]]:
    """OCR lines from pixels of a page rendered at ``OCR_SCALE`` (y down) to PDF points (y up), top to bottom."""
    out = []
    for box, text in items:
        xs = [float(p[0]) / OCR_SCALE for p in box]
        ys = [float(p[1]) / OCR_SCALE for p in box]
        out.append(((min(xs), page_height - max(ys), max(xs), page_height - min(ys)), str(text)))
    return sorted(out, key=lambda item: (-item[0][3], item[0][0]))


def ocr_lines(items: list[tuple[Any, str]], *, width: float) -> str:
    """OCR boxes as text in reading order. On a two-column page (boxes to both sides of a gutter) the full-width
    boxes above the columns come first, then the left column, then the right, then any full-width box below; within
    each, boxes whose vertical centres are within half a line height share a line, read left to right."""
    boxes = []
    for box, text in items:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        boxes.append({"x0": min(xs), "x1": max(xs), "y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys), "t": text})
    mid, gap = width / 2, width * 0.02
    for b in boxes:
        b["side"] = "L" if b["x1"] <= mid + gap else "R" if b["x0"] >= mid - gap else "W"
    two_columns = sum(b["side"] == "L" for b in boxes) >= 5 and sum(b["side"] == "R" for b in boxes) >= 5
    if two_columns:
        # Full-width boxes (a title, a figure or equation across both columns) cut the page into bands; each is read
        # in place, and between two of them the left column is read before the right.
        groups = []
        band: list[dict[str, Any]] = []
        for b in sorted(boxes, key=lambda b: b["y"]):
            if b["side"] == "W":
                groups += [[x for x in band if x["side"] == "L"], [x for x in band if x["side"] == "R"], [b]]
                band = []
            else:
                band.append(b)
        groups += [[x for x in band if x["side"] == "L"], [x for x in band if x["side"] == "R"]]
    else:
        groups = [boxes]
    out: list[str] = []
    for group in groups:
        group.sort(key=lambda b: (b["y"], b["x0"]))
        lines: list[list[dict[str, Any]]] = []
        for b in group:
            if lines and abs(b["y"] - lines[-1][-1]["y"]) <= max(b["h"], lines[-1][-1]["h"]) / 2:
                lines[-1].append(b)
            else:
                lines.append([b])
        out.extend(" ".join(x["t"] for x in sorted(line, key=lambda x: x["x0"])) for line in lines)
    return "\n".join(out)


def extract(
    source: bytes | Path | str, *, max_bytes: int = 10 * 1024 * 1024, ocr: bool = True,
    ocr_deadline: float | None = None,
) -> PdfText:
    """Read every page of a PDF (see the module docstring). Never raises: a PDF that cannot be opened comes back
    with ``error`` set and no text.

    ``ocr_deadline`` (a ``time.monotonic()`` value) stops OCR mid-document: the scanned pages after it are counted as
    unread, so a 300-page scan cannot hold a batch far past its budget."""
    import time

    out = PdfText()
    try:
        engine, doc = _open(source)
    except Exception as e:  # noqa: BLE001 -- a broken or encrypted PDF is reported, not raised
        out.error = f"could not open the PDF ({type(e).__name__}: {str(e)[:120]})"
        return out
    out.engine = engine
    reader: _Ocr | None = None
    parts: list[str] = []
    size = 0
    try:
        out.pages = len(doc)
        for i in range(out.pages):
            page = doc[i]
            text, has_images = (_page_text_pymupdf if engine == "pymupdf" else _page_text_pdfium)(page)
            if len(out.hidden_text) < _HIDDEN_MAX:
                out.hidden_text += [t[:200] for t in _hidden_text(engine, page)][:_HIDDEN_MAX - len(out.hidden_text)]
            if len(text.strip()) < SCANNED_PAGE_CHARS and has_images:
                if ocr and reader is None:
                    reader = _Ocr()
                if ocr_deadline is not None and time.monotonic() >= ocr_deadline:
                    out.unread_pages.append(i + 1)
                    out.ocr_out_of_time = True
                elif reader is not None and reader.name:
                    try:
                        text, items = reader.read(_render_png(engine, doc, i))
                        out.ocr_lines[i + 1] = _to_points(items, _page_height(engine, page))
                        out.ocr_pages.append(i + 1)
                        out.ocr_engine = reader.name
                    except Exception:  # noqa: BLE001 -- one unreadable page is counted, the rest go on
                        out.unread_pages.append(i + 1)
                else:
                    out.unread_pages.append(i + 1)
            text = clean(text)
            if not text:
                continue
            size += len(text.encode("utf-8", errors="replace"))
            if size > max_bytes:
                out.truncated_at_page = i + 1
                break
            parts.append(text)
    except Exception as e:  # noqa: BLE001 -- what was read so far is kept
        out.error = f"stopped reading at page {len(parts) + 1} ({type(e).__name__})"
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass
    out.text = "\n\n".join(parts)
    if not out.text and not out.unread_pages and not out.error and out.pages:
        # No text layer and no images: the text is drawn as outlines (some print-ready files), or the pages are blank.
        # Said, so an empty result is never mistaken for a read one.
        out.error = "no text on any page and no page image to read (text drawn as outlines, or blank pages)"
    return out
