"""PDF measurements (``generation/_pdf_measure.py``).

The PDFs are built here with pypdfium2 itself, so each test controls the
exact font sizes and positions it measures and needs no LaTeX or browser.
"""

from __future__ import annotations

import ctypes

import pytest

pdfium = pytest.importorskip("pypdfium2")

from generation._pdf_measure import (  # noqa: E402
    SLIDE_FIGURE_TICK_MIN_PT,
    FigureSource,
    Line,
    Page,
    figure_tick_findings,
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
                _, left, bottom, right, top, *pixels = item
                image = pdfium.PdfImage.new(pdf)
                size = pixels[0] if pixels else (8, 8)
                image.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", size, (40, 90, 140))))
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


def test_slides_report_finds_a_figure_drawn_over_text(tmp_path):
    bullets = [
        ("text", "A slide title", 40, 60, 460, True),
        ("text", "First bullet above the figure", 25, 60, 380, False),
        ("text", "Second bullet above the figure", 25, 60, 340, False),
        ("text", "Third bullet the figure covers", 25, 60, 300, False),
    ]
    # The pptx deck of the validation quest: the figure's top edge sat on the
    # third bullet.
    covering = bullets + [("image", 40, 60, 700, 318)]
    below = bullets + [("image", 297, 40, 663, 285)]
    # A full-slide background behind the text is not a figure over it.
    background = bullets + [("image", 0, 0, 960, 540)]
    path = _pdf(tmp_path, [(*SLIDE, covering), (*SLIDE, below), (*SLIDE, background)])
    findings = [f for f in slides_report(measure_pdf(path))["findings"] if f["check"] == "overlap"]
    assert [(f["page"], f["severity"], f["lines_covered"]) for f in findings] == [(1, "high", 1)]
    assert "Third bullet the figure covers" in findings[0]["problem"]


def test_slides_report_finds_a_figure_that_starts_right_under_text(tmp_path):
    bullets = [
        ("text", "A slide title", 40, 60, 460, True),
        ("text", "First bullet above the figure", 25, 60, 380, False),
        ("text", "Last bullet right over the figure", 25, 60, 300, False),
    ]
    # The pptx deck of the validation quest: nothing overlapped, but the
    # figure's top edge was 0.1 pt under the last bullet.
    touching = bullets + [("image", 40, 60, 700, 292)]
    spaced = bullets + [("image", 40, 60, 700, 262)]
    # A small logo beside a line is not a figure.
    logo = bullets + [("image", 700, 282, 730, 292)]
    path = _pdf(tmp_path, [(*SLIDE, touching), (*SLIDE, spaced), (*SLIDE, logo)])
    findings = slides_report(measure_pdf(path))["findings"]
    gaps = [f for f in findings if f["check"] == "figure_gap"]
    assert [(f["page"], f["severity"]) for f in gaps] == [(1, "low")]
    assert "Last bullet right over the figure" in gaps[0]["problem"] and gaps[0]["gap_pt"] < 6
    assert [f for f in findings if f["check"] == "overlap"] == []


def test_slides_report_finds_latex_math_shown_as_text(tmp_path):
    title = [("text", "A slide title", 40, 60, 460, True)]
    # The validation quest's pptx printed its formulas as LaTeX.
    raw = title + [
        ("text", "Diverges completely at $h = 0.5$ and $h = 0.1$.", 25, 60, 380, False),
        ("text", "RK4 error is 5.297 \\times 10^{-9} m.", 25, 60, 340, False),
    ]
    # Prices by pandoc's rule are not math, and a typeset formula has no $.
    prices = title + [("text", "Licences cost $5-$10 per seat.", 25, 60, 380, False)]
    typeset = title + [("text", "RK4 error is 5.297 × 10−9 m.", 25, 60, 380, False)]
    path = _pdf(tmp_path, [(*SLIDE, raw), (*SLIDE, prices), (*SLIDE, typeset)])
    findings = [f for f in slides_report(measure_pdf(path))["findings"] if f["check"] == "raw_markup"]
    assert [(f["page"], f["lines_with_latex"]) for f in findings] == [(1, 2)]


# ---------------------------------------------------------------------------
# Tick labels of a figure on a slide

# A figure drawn 13.1 in wide at 100 dpi, whose ticks were drawn at 15 pt (the house
# style). Its pixels are what a placed image is matched to it by.
ROW = FigureSource("row.png", 1310, 410, 13.1, 15.0)
GRID = FigureSource("grid.png", 1310, 1010, 13.1, 15.0)


def _slide_page(images, *, bullets=True, title="Established outbreaks match theory", number=1):
    """A slide as ``measure_pdf`` reads it: a title, a lead line, optionally bullets
    beside or under the figure, the page number in the footer, and ``images``, each
    ``(box, pixel size)``."""
    lines = [
        Line(title, 27.4, True, (58.0, 430.0, 800.0, 460.0)),
        Line("Figure 3 shows the final size against the population.", 18.75, True, (58.0, 390.0, 700.0, 412.0)),
        Line(str(number), 15.0, False, (924.0, 25.0, 928.0, 32.0)),
    ]
    if bullets:
        lines += [
            Line("First point with its number.", 18.75, False, (58.0, 300.0, 440.0, 322.0)),
            Line("Second point with its number.", 18.75, False, (58.0, 260.0, 440.0, 282.0)),
        ]
    return Page(number, 960.0, 540.0, lines, [box for box, _px in images], [], [px for _box, px in images])


def test_measure_pdf_reads_the_pixel_size_of_each_image(tmp_path):
    path = _pdf(tmp_path, [(*SLIDE, [("image", 100, 100, 500, 220, (1310, 410)), ("image", 600, 100, 700, 200)])])
    assert measure_pdf(path).pages[0].image_px == [(1310, 410), (8, 8)]


def test_a_figure_beside_bullets_with_tick_labels_under_8_pt_is_sent_back_to_be_given_its_own_slide():
    """cb1 of the six stored quests: a 13.1 in figure in a 5.6 in side pane, so
    10 pt ticks came out at 4.3 pt. The finding says what to do about it."""
    assert SLIDE_FIGURE_TICK_MIN_PT == 8.0
    page = _slide_page([((500.0, 200.0, 903.0, 328.0), (1310, 410))])
    (finding,) = figure_tick_findings(page, [ROW])
    assert (finding["check"], finding["severity"], finding["page"]) == ("figure_ticks_small", "medium", 1)
    assert finding["figure"] == "row.png" and finding["alone"] is False
    assert finding["tick_pt"] == 6.4 and finding["scale"] == 0.427
    problem = finding["problem"]
    assert "row.png" in problem and "Established outbreaks match theory" in problem
    assert "5.6 in wide" in problem and "43%" in problem and "6.4 pt" in problem and "at least 8 pt" in problem
    assert "slide of its own" in problem and "no bullets" in problem and "next slide" in problem


def test_a_figure_that_is_large_enough_is_not_reported_and_its_tick_size_is_a_metric():
    wide = _slide_page([((57.0, 60.0, 903.0, 335.0), (1310, 410))], bullets=False)
    assert figure_tick_findings(wide, [ROW]) == []
    # The floor is 8 pt: a 15 pt tick drawn at 8.01/15 of the width it was drawn at is enough, at 7.99/15 it is not.
    def at(tick_pt):
        return _slide_page([((100.0, 60.0, 100.0 + 13.1 * 72 * tick_pt / 15, 200.0), (1310, 410))], bullets=False)

    assert figure_tick_findings(at(8.01), [ROW]) == []
    assert [f["check"] for f in figure_tick_findings(at(7.99), [ROW])] == ["figure_ticks_small"]

    from generation._pdf_measure import Document

    report = slides_report(Document([wide, _slide_page([], number=2)]), figures=[ROW])
    assert [e.get("figure_tick_pt") for e in report["metrics"]["per_page"]] == [[13.5], None]
    assert report["findings"] == []


def test_a_figure_that_already_has_its_slide_is_low_since_no_new_deck_changes_it():
    """A 3x3 grid is 10.1 in tall: alone on its slide it still reaches only 0.45 of
    its size. That is a figure with too many panels, not a slide with too much text."""
    page = _slide_page([((268.0, 53.0, 692.0, 380.0), (1310, 1010))], bullets=False)
    (finding,) = figure_tick_findings(page, [GRID])
    assert (finding["severity"], finding["alone"], finding["tick_pt"]) == ("low", True, 6.7)
    assert "already has a slide of its own" in finding["problem"] and "too many panels" in finding["problem"]
    assert "Give the figure a slide of its own" not in finding["problem"]


def test_a_caption_under_the_figure_means_it_shares_its_slide_but_the_page_number_does_not():
    box = (500.0, 200.0, 903.0, 328.0)
    caption = _slide_page([(box, (1310, 410))], bullets=False)
    caption.lines.append(Line("Figure 3: conditional final size.", 12.0, False, (58.0, 170.0, 500.0, 186.0)))
    assert figure_tick_findings(caption, [ROW])[0]["alone"] is False
    assert figure_tick_findings(_slide_page([(box, (1310, 410))], bullets=False), [ROW])[0]["alone"] is True


def test_an_image_no_figure_file_matches_is_left_out_and_so_is_one_two_files_disagree_on():
    small = ((500.0, 200.0, 903.0, 328.0), (1310, 410))
    assert figure_tick_findings(_slide_page([((500.0, 200.0, 903.0, 328.0), (999, 700))]), [ROW]) == []
    assert figure_tick_findings(_slide_page([((500.0, 200.0, 903.0, 328.0), None)]), [ROW]) == []
    # Two files of one pixel size but different tick sizes: naming either could be wrong.
    other = FigureSource("row_small.png", 1310, 410, 13.1, 8.0)
    assert figure_tick_findings(_slide_page([small]), [ROW, other]) == []
    # The same size and the same ticks: the finding stands, without a name.
    twin = FigureSource("row_twin.png", 1310, 410, 13.1, 15.0)
    (finding,) = figure_tick_findings(_slide_page([small]), [ROW, twin])
    assert finding["figure"] is None and "row.png" not in finding["problem"]


def test_a_resampled_figure_is_matched_by_its_proportions_and_a_bar_or_an_icon_is_not_a_figure():
    """LibreOffice reduces a picture to 300 dpi in its PDF: the pixels change and the
    proportions do not. Its accent bar and logo are images too."""
    resampled = _slide_page([((36.0, 46.0, 924.0, 324.0), (655, 205))], bullets=False)   # 92% of the width
    (finding,) = figure_tick_findings(resampled, [FigureSource("row.png", 1310, 410, 24.0, 15.0)])
    assert finding["figure"] == "row.png" and finding["tick_pt"] == 7.7 and finding["severity"] == "low"
    decorations = _slide_page([
        ((-3.0, 527.0, 963.0, 541.0), (966, 14)),      # the accent bar
        ((62.0, 464.0, 110.0, 473.0), (200, 39)),      # a logo
        ((0.0, 0.0, 960.0, 540.0), (1310, 410)),       # a picture covering the slide
    ], bullets=False)
    assert figure_tick_findings(decorations, [ROW, FigureSource("bar.png", 966, 14, 9.7, 15.0)]) == []


def test_the_slide_title_is_the_largest_text_even_when_the_pdf_gives_it_one_glyph_at_a_time():
    """The serif titles of the Marp deck come out of the PDF one glyph to a line."""
    glyphs = [
        Line(ch, 36.5, False, (58.0 + 16 * i + (12 if i > 3 else 0), 425.0, 70.0 + 16 * i + (12 if i > 3 else 0), 445.0))
        for i, ch in enumerate("Fig one")
        if ch != " "
    ]
    body = Line("A bullet under the title that is set smaller.", 18.75, False, (58.0, 300.0, 600.0, 322.0))
    page = Page(1, 960.0, 540.0, [*glyphs, body], [(500.0, 100.0, 903.0, 230.0)], [], [(1310, 410)])
    (finding,) = figure_tick_findings(page, [ROW])
    assert 'on the slide "Fig one"' in finding["problem"]


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


def test_a_figure_alone_on_a_half_blank_page_is_reported_but_not_a_page_with_text(tmp_path):
    text = "x" * 70
    full = _column(72, 780, 60, 10, text, 14)
    # The validation quest's float page: one figure, centred, blank above and below.
    alone = [("image", 72, 300, 520, 560), ("text", "Figure 1: Outbreak probability.", 9, 72, 285, False)]
    with_text = [("image", 72, 450, 520, 770)] + _column(72, 430, 60, 10, text, 14)
    short_last = _column(72, 780, 600, 10, text, 14)
    pages = [(*A4, full), (*A4, alone), (*A4, with_text), (*A4, short_last)]
    report = paper_report(measure_pdf(_pdf(tmp_path, pages)))
    half = [f for f in report["findings"] if f["check"] == "half_empty_page"]
    assert [(f["page"], f["severity"]) for f in half] == [(2, "medium")]
    assert "the paper goes on to the next page" in half[0]["problem"]
    assert report["metrics"]["half_empty_pages"] == 1, "the last page is judged by its own check"


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
