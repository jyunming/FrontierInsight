"""PDF measurements (``generation/_pdf_measure.py``).

The PDFs are built here with pypdfium2 itself, so each test controls the
exact font sizes and positions it measures and needs no LaTeX or browser.
"""

from __future__ import annotations

import ctypes

import pytest

pdfium = pytest.importorskip("pypdfium2")

from generation._pdf_measure import (  # noqa: E402
    measure_pdf,
    paper_report,
    poster_report,
    render_pages,
    slides_report,
)

A1 = (1684.0, 2384.0)
SLIDE = (960.0, 540.0)
A4 = (595.0, 842.0)


def _pdf(tmp_path, pages, name="doc.pdf"):
    """``pages`` is a list of ``(width, height, items)``. Items:
    ``("text", text, size, x, y, bold)``, ``("rule", left, bottom, right, top)``
    and ``("image", left, bottom, right, top)``."""
    from PIL import Image

    raw = pdfium.raw
    pdf = pdfium.PdfDocument.new()
    for width, height, items in pages:
        page = pdf.new_page(width, height)
        for item in items:
            if item[0] == "text":
                _, text, size, x, y, bold = item
                font = b"Helvetica-Bold" if bold else b"Helvetica"
                obj = raw.FPDFPageObj_NewTextObj(pdf.raw, font, ctypes.c_float(size))
                buf = ctypes.create_string_buffer((text + "\x00").encode("utf-16-le"))
                raw.FPDFText_SetText(obj, ctypes.cast(buf, ctypes.POINTER(raw.FPDF_WCHAR)))
                raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, y)
                raw.FPDFPage_InsertObject(page.raw, obj)
            elif item[0] == "rule":
                _, left, bottom, right, top = item
                obj = raw.FPDFPageObj_CreateNewRect(left, bottom, right - left, top - bottom)
                raw.FPDFPath_SetDrawMode(obj, raw.FPDF_FILLMODE_WINDING, 0)
                raw.FPDFPage_InsertObject(page.raw, obj)
            elif item[0] == "image":
                _, left, bottom, right, top = item
                image = pdfium.PdfImage.new(pdf)
                image.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (8, 8), (40, 90, 140))))
                image.set_matrix(pdfium.PdfMatrix().scale(right - left, top - bottom).translate(left, bottom))
                page.insert_obj(image)
        page.gen_content()
        page.close()
    path = tmp_path / name
    pdf.save(path)
    pdf.close()
    return path


def _column(x, top, bottom, size, text, step):
    items, y = [], top
    while y >= bottom:
        items.append(("text", text, size, x, y, False))
        y -= step
    return items


BODY_50 = "Body text that reads like a real poster sentence."  # 50 characters


def _good_poster_items():
    items = [
        ("rule", 60, 2270, 1624, 2272),
        ("text", "Symplectic methods keep energy bounded", 80, 60, 2170, True),
        ("text", "J. Researcher, Example Lab, j@example.org", 30, 60, 2110, False),
        ("rule", 60, 2090, 1624, 2096),
        ("text", "Background", 40, 60, 2010, True),
        ("text", "Results", 40, 880, 2010, True),
    ]
    items += _column(60, 1960, 1300, 26, BODY_50, 36)
    items += [
        ("image", 60, 720, 800, 1260),
        ("text", "Figure 1: Energy error over time", 20, 60, 680, False),
    ]
    items += _column(60, 630, 200, 26, BODY_50, 36)
    items += _column(880, 1960, 200, 26, BODY_50, 36)
    items += [
        ("rule", 60, 150, 1624, 152),
        ("text", "[1] E. Hairer (2006). Geometric Numerical Integration. Springer.", 16, 60, 120, False),
        ("text", "[2] L. Verlet (1967). Computer experiments. Phys. Rev.", 16, 60, 95, False),
    ]
    return items


def test_measure_pdf_reads_lines_sizes_bold_images_and_rules(tmp_path):
    path = _pdf(tmp_path, [(*A1, _good_poster_items())])
    doc = measure_pdf(path)
    assert doc is not None and len(doc.pages) == 1
    page = doc.pages[0]
    title = next(line for line in page.lines if line.text.startswith("Symplectic"))
    assert title.size == 80 and title.bold
    body = next(line for line in page.lines if line.text == BODY_50)
    assert body.size == 26 and not body.bold
    assert len(page.images) == 1
    assert len(page.rules) == 3


def test_a_poster_that_meets_the_standards_has_no_findings(tmp_path):
    path = _pdf(tmp_path, [(*A1, _good_poster_items())])
    report = poster_report(
        measure_pdf(path), header_terms=("J. Researcher", "Example Lab"),
    )
    assert report["findings"] == []
    m = report["metrics"]
    assert (m["title_pt"], m["heading_pt"], m["body_pt"], m["caption_pt"], m["references_pt"]) == (
        80, 40, 26, 20, 16,
    )
    assert m["columns"] == 2 and m["figures"] == 1 and m["captions"] == 1
    assert m["references"] == 2
    assert m["column_gap_cm"] < 2


def test_a_poster_like_the_old_one_is_flagged_on_every_standard(tmp_path):
    long_line = "Body text at the old size runs on for far too many characters on each line."
    long_line += " More."  # 80 characters
    items = [
        ("rule", 60, 2270, 1624, 2272),
        ("text", "Stability and Accuracy Trade-offs", 47, 60, 2170, True),
        ("rule", 60, 2130, 1624, 2136),
        ("text", "Background", 18.8, 60, 2060, True),
        ("text", "Results", 18.8, 880, 2060, True),
        ("image", 880, 1300, 1600, 2000),
    ]
    items += _column(60, 2020, 1300, 18.8, long_line, 26)  # left column stops early
    items += _column(880, 1260, 400, 18.8, long_line, 26)
    items.append(("rule", 60, 360, 1624, 362))
    refs = " ".join(f"[{i}] Author (2020) https://example.org/{i}" for i in range(1, 13))
    items.append(("text", refs[:140], 10.9, 60, 330, False))
    items.append(("text", refs[140:280], 10.9, 60, 315, False))
    items.append(("text", refs[280:], 10.9, 60, 300, False))
    path = _pdf(tmp_path, [(*A1, items)])
    report = poster_report(measure_pdf(path), header_terms=("J. Researcher",))
    checks = {f["check"] for f in report["findings"]}
    assert {
        "title_font", "body_font", "heading_font", "captions", "line_length",
        "column_balance", "empty_space", "references_count", "references_urls",
        "references_font", "header_info",
    } <= checks
    balance = next(f for f in report["findings"] if f["check"] == "column_balance")
    assert balance["region"] == "column 1"
    assert all(f["page"] == 1 and f["source"] == "measure" for f in report["findings"])


def test_poster_text_below_the_sheet_is_reported_as_cut_off(tmp_path):
    items = _good_poster_items() + [("text", BODY_50, 26, 880, -40, False)]
    path = _pdf(tmp_path, [(*A1, items)])
    report = poster_report(measure_pdf(path))
    cut = [f for f in report["findings"] if f["check"] == "overflow"]
    assert cut and cut[0]["severity"] == "high"


def test_a_column_that_runs_off_the_sheet_is_one_finding(tmp_path):
    items = _good_poster_items() + _column(880, -40, -400, 26, BODY_50, 36)
    report = poster_report(measure_pdf(_pdf(tmp_path, [(*A1, items)])))
    cut = [f for f in report["findings"] if f["check"] == "overflow"]
    assert len(cut) == 1
    assert cut[0]["lines_cut"] == 11 and cut[0]["severity"] == "high"


def test_a_sheet_without_a_header_rule_still_finds_its_columns_and_band(tmp_path):
    # Only the band rule is left. It must not be read as the header's lower
    # edge, which made the whole sheet "header" and the band the "body".
    items = [item for item in _good_poster_items() if not (item[0] == "rule" and item[2] > 1200)]
    report = poster_report(measure_pdf(_pdf(tmp_path, [(*A1, items)])))
    m = report["metrics"]
    assert (m["body_pt"], m["heading_pt"], m["references_pt"]) == (26, 40, 16)
    assert m["column_gap_cm"] < 2


def test_column_text_running_into_the_references_band_is_reported(tmp_path):
    items = _good_poster_items() + _column(880, 180, 60, 26, BODY_50, 36)
    report = poster_report(measure_pdf(_pdf(tmp_path, [(*A1, items)])))
    spill = [f for f in report["findings"] if f["check"] == "band_overlap"]
    assert len(spill) == 1 and spill[0]["severity"] == "high"
    # The spilled body lines are not mistaken for the reference list.
    assert report["metrics"]["references_pt"] == 16


def test_a_caption_running_into_the_references_band_is_reported(tmp_path):
    items = _good_poster_items() + [("text", "Figure 2: A caption that slid into the band", 20, 880, 120, False)]
    report = poster_report(measure_pdf(_pdf(tmp_path, [(*A1, items)])))
    assert any(f["check"] == "band_overlap" for f in report["findings"])
    assert report["metrics"]["references_pt"] == 16


def test_columns_a_little_uneven_are_a_low_finding(tmp_path):
    right_column = [i for i in _good_poster_items() if i[0] == "text" and i[3] == 880 and i[2] == 26]
    items = [i for i in _good_poster_items() if i not in right_column]
    items += _column(880, 1960, 470, 26, BODY_50, 36)  # ends about 9 cm above the left column
    report = poster_report(measure_pdf(_pdf(tmp_path, [(*A1, items)])))
    balance = [f for f in report["findings"] if f["check"] == "column_balance"]
    assert len(balance) == 1 and balance[0]["severity"] == "low"


def test_landscape_posters_are_measured_in_three_columns(tmp_path):
    width, height = 3456.0, 2592.0  # 48 x 36 in
    items = [
        ("rule", 60, 2480, 3396, 2482),
        ("text", "A landscape finding headline", 90, 60, 2380, True),
        ("rule", 60, 2340, 3396, 2346),
    ]
    for x in (60, 1200, 2340):
        items += _column(x, 2250, 300, 28, BODY_50, 40)
    items.append(("rule", 60, 250, 3396, 252))
    items.append(("text", "[1] E. Hairer (2006). Geometric Numerical Integration.", 18, 60, 220, False))
    path = _pdf(tmp_path, [(width, height, items)])
    report = poster_report(measure_pdf(path))
    assert report["metrics"]["columns"] == 3
    assert "column_balance" not in {f["check"] for f in report["findings"]}


def test_slides_report_finds_cut_off_text_and_tiny_text_on_the_right_slide(tmp_path):
    ok = [("text", "A slide title", 40, 60, 460, True), ("text", "A bullet that fits.", 25, 60, 380, False)]
    cut = ok + [("text", "This bullet fell off the bottom of the slide", 25, 60, -25, False)]
    tiny = ok + [("text", "A source list squeezed down to nine points", 9, 60, 100, False)]
    # Slide 5 of the gemma4 validation deck: the figure sat 2 pt below the
    # slide, and its frame and axis label lost their bottom edge.
    low_figure = ok + [("image", 297, -2, 663, 257)]
    # The pptx accent bar as LibreOffice exports it: 3 pt past both sides,
    # on purpose. Not a cut-off figure.
    ok.append(("image", -3.1, 527.9, 963.1, 541.4))
    path = _pdf(tmp_path, [(*SLIDE, ok), (*SLIDE, cut), (*SLIDE, tiny), (*SLIDE, low_figure)])
    report = slides_report(measure_pdf(path))
    by_check = {(f["check"], f["page"]) for f in report["findings"]}
    assert by_check == {("overflow", 2), ("small_font", 3), ("overflow", 4)}
    assert report["metrics"]["pages"] == 4


def test_paper_report_finds_a_line_running_into_the_margin(tmp_path):
    justified = "x" * 70  # the same text gives every line the same right edge
    page1 = _column(72, 760, 80, 10, justified, 14)
    page2 = _column(72, 760, 400, 10, justified, 14)
    page2.append(("text", "y" * 80, 10, 72, 380, False))
    page2.append(("image", 72, 100, 590, 300))
    path = _pdf(tmp_path, [(*A4, page1), (*A4, page2)])
    report = paper_report(measure_pdf(path))
    wide = [f for f in report["findings"] if f["check"] == "overwide"]
    assert {(f["page"], f["object"]) for f in wide} == {(2, "text"), (2, "figure")}
    assert all(f["overhang_pt"] > 3 for f in wide)


def test_paper_report_accepts_a_two_column_layout(tmp_path):
    text = "z" * 40
    items = _column(50, 760, 80, 10, text, 14) + _column(310, 760, 80, 10, text, 14)
    path = _pdf(tmp_path, [(*A4, items)])
    report = paper_report(measure_pdf(path))
    assert [f for f in report["findings"] if f["check"] == "overwide"] == []


def _footer(number):
    return [("text", "Frontier Insight", 8, 72, 40, False), ("text", str(number), 8, 500, 40, False)]


def test_a_last_page_holding_one_line_is_reported(tmp_path):
    # The validation quest's paper: page 6 held only the tail of a URL.
    text = "x" * 70
    page1 = _column(72, 760, 80, 10, text, 14) + _footer(1)
    page2 = [("text", "ebooks rst/3 Ordinary Differential Equations/02 Examples", 10, 72, 760, False)] + _footer(2)
    report = paper_report(measure_pdf(_pdf(tmp_path, [(*A4, page1), (*A4, page2)])))
    last = [f for f in report["findings"] if f["check"] == "last_page_nearly_empty"]
    assert [(f["page"], f["severity"]) for f in last] == [(2, "medium")]
    assert report["metrics"]["last_page_lines"] == 1


def test_a_last_page_with_real_text_is_not_reported(tmp_path):
    text = "x" * 70
    page1 = _column(72, 760, 80, 10, text, 14) + _footer(1)
    page2 = _column(72, 760, 634, 10, text, 14) + _footer(2)
    report = paper_report(measure_pdf(_pdf(tmp_path, [(*A4, page1), (*A4, page2)])))
    assert not [f for f in report["findings"] if f["check"] == "last_page_nearly_empty"]
    assert report["metrics"]["last_page_lines"] == 10


def test_a_one_page_paper_is_never_reported_as_a_nearly_empty_last_page(tmp_path):
    page = [("text", "A one-line note.", 10, 72, 760, False)] + _footer(1)
    report = paper_report(measure_pdf(_pdf(tmp_path, [(*A4, page)])))
    assert not [f for f in report["findings"] if f["check"] == "last_page_nearly_empty"]


def test_measure_pdf_fails_open_on_a_file_that_is_not_a_pdf(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_text("not a pdf", encoding="utf-8")
    assert measure_pdf(bad) is None
    assert measure_pdf(tmp_path / "missing.pdf") is None
    assert render_pages(bad, tmp_path / "shots") == []


def test_measuring_and_rendering_leave_the_file_free_to_replace(tmp_path):
    path = _pdf(tmp_path, [(*SLIDE, [("text", "Title", 40, 60, 460, True)])])
    assert measure_pdf(path) is not None
    shots = render_pages(path, tmp_path / "shots", dpi=36)
    assert [p.name for p in shots] == ["page-1.png"]
    from PIL import Image

    with Image.open(shots[0]) as shot:
        assert shot.size == (480, 270)
    path.write_bytes(b"replaced")  # Windows refuses this while a handle is open
