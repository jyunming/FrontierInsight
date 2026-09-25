"""Reading a PDF whole (core/pdf_text.py) and what the literature step keeps of it (core/engine.py)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from core import pdf_text


def _text_pdf(path: Path, lines: list[str]) -> Path:
    """A PDF with a real text layer (matplotlib writes its text as text)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 11))
    for i, line in enumerate(lines):
        fig.text(0.08, 0.92 - i * 0.03, line, fontsize=11)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def _scanned_pdf(path: Path, lines: list[str]) -> Path:
    """A PDF whose page is only an image of text: what a scanner produces."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1275, 1650), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((100, 120 + i * 60), line, fill="black", font=font)
    image.save(path, "PDF", resolution=150)
    return path


def _ocr_available() -> bool:
    try:
        return bool(pdf_text._Ocr().name)
    except Exception:  # noqa: BLE001
        return False


def test_clean_joins_only_hyphens_the_typesetter_added() -> None:
    assert pdf_text.clean("In\ufffe\nstead of re\u00adsults") == "Instead of results"
    assert pdf_text.clean("a well-\nknown result") == "a well-\nknown result", "a real compound is kept"
    assert pdf_text.clean("the \ufb01rst  \ufb01t") == "the first fit"


def test_a_text_pdf_is_read_whole(tmp_path: Path) -> None:
    lines = [f"Line {i} of the stochastic epidemic paper, section results." for i in range(20)]
    result = pdf_text.extract(_text_pdf(tmp_path / "t.pdf", lines))
    assert not result.error and result.pages == 1 and not result.unread_pages
    assert "Line 0 of the stochastic epidemic paper" in result.text and "Line 19" in result.text


def test_a_scanned_page_without_ocr_is_counted_as_unread_never_silently_empty(tmp_path: Path) -> None:
    result = pdf_text.extract(_scanned_pdf(tmp_path / "s.pdf", ["A contribution to the"]), ocr=False)
    assert result.unread_pages == [1] and result.text == ""
    assert "1 scanned page(s) not read" in result.summary()


@pytest.mark.skipif(not _ocr_available(), reason="no OCR engine installed (tesseract or rapidocr)")
def test_a_scanned_page_is_read_by_ocr(tmp_path: Path) -> None:
    result = pdf_text.extract(_scanned_pdf(tmp_path / "s.pdf", [
        "A contribution to the mathematical", "theory of epidemics, by Kermack", "and McKendrick, 1927.",
    ]))
    assert result.ocr_pages == [1] and result.ocr_engine
    text = result.text.lower()
    assert "mathematical" in text and "epidemics" in text


def test_the_size_cap_says_where_it_stopped(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    path = tmp_path / "long.pdf"
    with PdfPages(path) as pages:
        for n in range(3):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.1, 0.5, f"Page {n} " + "word " * 60, fontsize=6)
            pages.savefig(fig)
            plt.close(fig)
    result = pdf_text.extract(path, max_bytes=250)  # each page holds about 170 characters
    assert result.truncated_at_page is not None and "cut at page" in result.summary()


def test_ocr_lines_read_a_two_column_page_column_by_column() -> None:
    def box(x0, x1, y, text):  # noqa: ANN001, ANN202
        return ([(x0, y - 5), (x1, y - 5), (x1, y + 5), (x0, y + 5)], text)

    items = [box(100, 900, 20, "Title across the page")]
    for i in range(6):
        items.append(box(50, 480, 100 + i * 20, f"left {i}"))
        items.append(box(520, 950, 100 + i * 20, f"right {i}"))
    lines = pdf_text.ocr_lines(items, width=1000).splitlines()
    assert lines[0] == "Title across the page"
    assert lines[1:7] == [f"left {i}" for i in range(6)] and lines[7:13] == [f"right {i}" for i in range(6)]


def test_a_scanned_pdf_fetched_is_set_aside_for_ocr_not_read_inside_the_fetch(tmp_path: Path) -> None:
    from core import knowledge

    body = _scanned_pdf(tmp_path / "s.pdf", ["Scanned classic"]).read_bytes()
    slot: list[bytes] = []
    token = knowledge._scanned_pdfs.set(slot)
    try:
        assert knowledge._pdf_bytes_to_text(body, cap=10_000) is None
    finally:
        knowledge._scanned_pdfs.reset(token)
    assert slot == [body]


def test_content_quality_tells_full_text_from_an_abstract_or_a_table_of_contents() -> None:
    from core.engine import _content_quality

    abstract = "The final size formula of Kermack and McKendrick holds for any infectious period. " * 3
    assert _content_quality(abstract, {}) == "snippet_only"
    again = f"{abstract}\n\n---FULL TEXT (fetched)---\n\n{abstract}"
    assert _content_quality(again, {"fetched_full_text": True}) == "abstract_only"
    toc = "\n".join(f"{i}.{j} Section heading number {j} {10 * i + j}" for i in range(1, 9) for j in range(1, 8))
    assert _content_quality(f"{abstract}\n\n---FULL TEXT (fetched)---\n\n{toc}", {"fetched_full_text": True}) == "preview_only"
    body = "We simulate the stochastic SIR model and report the final size. " * 200
    assert _content_quality(f"{abstract}\n\n---FULL TEXT (fetched)---\n\n{body}", {"fetched_full_text": True}) == "full_text"


def test_a_long_source_is_whole_on_disk_and_bounded_in_the_state(tmp_path: Path) -> None:
    from core.engine import _STATE_TEXT_CHARS, _item_content, _literature_entry

    text = "x" * (_STATE_TEXT_CHARS + 5000)
    entry = _literature_entry(tmp_path, text, {"doi": "10.1/x", "fetched_full_text": True}, quality="full_text")
    assert len(entry["content"]) == _STATE_TEXT_CHARS
    assert Path(entry["metadata"]["full_text_path"]).is_file()
    assert _item_content(entry) == text
    short = _literature_entry(tmp_path, "short text", {})
    assert "full_text_path" not in short["metadata"] and _item_content(short) == "short text"
