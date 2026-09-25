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

    def read(self, png: bytes) -> str:
        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGB")
        if self.name == "tesseract":
            return self._engine.image_to_string(image)
        if self.name == "rapidocr":
            result = self._engine(image)
            items = list(zip(result.boxes if result.boxes is not None else [], result.txts or ()))
            return ocr_lines(items, width=image.width)
        return ""


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
            if len(text.strip()) < SCANNED_PAGE_CHARS and has_images:
                if ocr and reader is None:
                    reader = _Ocr()
                if ocr_deadline is not None and time.monotonic() >= ocr_deadline:
                    out.unread_pages.append(i + 1)
                    out.ocr_out_of_time = True
                elif reader is not None and reader.name:
                    try:
                        text = reader.read(_render_png(engine, doc, i))
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
