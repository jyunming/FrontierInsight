"""The screenshot + AI check of a finished PDF.

The model's findings must be tied to text visible on their page; a transport
that cannot send images leaves a measurements-only report; the report keeps
one entry per output and the check never raises.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Config
from core.provider import ImageInputUnsupported
from generation import _visual_check as vc
from generation._pdf_measure import Line, Page


def _doc(*pages: list[str]):
    return SimpleNamespace(pages=[
        Page(n, 595.0, 842.0, [Line(text, 10.0, False, (72.0, 700.0 - 14 * i, 500.0, 712.0 - 14 * i)) for i, text in enumerate(lines)], [], [])
        for n, lines in enumerate(pages, start=1)
    ])


def _finding(**overrides) -> dict:
    finding = {
        "page": 1, "region": "top of the page", "check": "raw_markup",
        "problem": "Raw LaTeX shows.", "severity": "high", "quote": "the textbf command shows",
    }
    finding.update(overrides)
    return finding


CHECKS = set(vc.checks_for("paper"))


def test_a_finding_is_kept_when_its_quote_is_on_its_page() -> None:
    doc = _doc(["Results: the textbf command shows as text."])
    kept, dropped = vc.grounded_findings([_finding()], doc, CHECKS)
    assert [f["check"] for f in kept] == ["raw_markup"] and dropped == []
    assert kept[0]["source"] == "model"


def test_a_url_quote_matches_the_text_layer_that_drew_its_underscores_as_spaces() -> None:
    # Paper page 6 of the validation quest: the model quoted the URL as drawn,
    # the text layer reads LaTeX's underscore rules as spaces and splits "ff".
    doc = _doc([], [], [], [], [], ["ebooks rst/3 Ordinary Dif ferential Equations/02 Examples/Harmonic Oscillator.html"])
    finding = _finding(
        page=6, check="empty_area",
        quote="ebooks_rst/3_Ordinary_Differential_Equations/02_Examples/Harmonic_Oscillator.html",
    )
    kept, dropped = vc.grounded_findings([finding], doc, CHECKS)
    assert len(kept) == 1 and dropped == []


def test_a_line_end_hyphen_in_the_text_layer_still_matches() -> None:
    doc = _doc(["the dissi-", "pative regime is stable"])
    kept, _ = vc.grounded_findings([_finding(quote="the dissipative regime")], doc, CHECKS)
    assert len(kept) == 1


def test_a_quote_on_the_next_page_is_still_placed() -> None:
    doc = _doc(["nothing here"], ["the textbf command shows"])
    kept, _ = vc.grounded_findings([_finding(page=1)], doc, CHECKS)
    assert len(kept) == 1


@pytest.mark.parametrize("finding,reason", [
    (_finding(quote="a sentence that appears nowhere"), "the quoted text is not on that page"),
    (_finding(check="looks_ugly"), "the check is not on the list"),
    (_finding(region=""), "no region"),
    (_finding(quote="textbf"), "the quote is too short to place"),
    (_finding(page=3), "the quoted text is not on that page"),
])
def test_findings_that_cannot_be_placed_are_dropped_with_the_reason(finding: dict, reason: str) -> None:
    doc = _doc(["Results: the textbf command shows as text."], ["other"], ["more"])
    kept, dropped = vc.grounded_findings([finding], doc, CHECKS)
    assert kept == [] and [d["dropped_because"] for d in dropped] == [reason]


def test_a_chinese_quote_without_spaces_can_be_placed() -> None:
    doc = _doc(["分子動力學積分器的能量守恆"])
    kept, _ = vc.grounded_findings([_finding(quote="動力學積分器")], doc, CHECKS)
    assert len(kept) == 1


def test_an_unknown_severity_becomes_medium_and_non_dicts_are_ignored() -> None:
    doc = _doc(["the textbf command shows"])
    kept, _ = vc.grounded_findings(["junk", _finding(severity="critical")], doc, CHECKS)
    assert [f["severity"] for f in kept] == ["medium"]


def _config(tmp_path: Path, *, ai: bool = True) -> Config:
    return Config.model_validate({"topic": "t", "title": "t", "output": {
        "output_dir": str(tmp_path / "out"), "visual_check_ai": ai}})


@pytest.mark.asyncio
async def test_by_default_no_model_is_asked_and_the_measurements_are_the_check(tmp_path: Path, monkeypatch) -> None:
    async def model(config, messages, supervisor):  # noqa: ANN001
        raise AssertionError("the AI check is off by default")

    monkeypatch.setattr(vc, "_ask", model)
    config = Config.model_validate({"topic": "t", "title": "t", "output": {"output_dir": str(tmp_path / "out")}})
    assert config.output.visual_check_ai is False
    report = await vc.check_pdf(config, "paper", _real_pdf(tmp_path), tmp_path)
    assert report["transport"] == "measurements only"
    assert "output.visual_check_ai" in report["reason"]
    assert "metrics" in report["measured"] and "pages_checked" not in report
    assert (tmp_path / ".fi" / "visual_check" / "paper" / "page-1.png").is_file(), "the screenshots are still saved"


def _real_pdf(tmp_path: Path, text: str = "Results: the textbf command shows as text.") -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 800), "white")
    ImageDraw.Draw(image).text((40, 40), text, fill="black")
    pdf = tmp_path / "paper.pdf"
    image.save(pdf, "PDF")
    return pdf


@pytest.mark.asyncio
async def test_a_transport_without_images_leaves_a_measurements_only_report(tmp_path: Path, monkeypatch) -> None:
    async def no_images(config, messages, supervisor, quest_root):  # noqa: ANN001
        raise ImageInputUnsupported("gemini_cli cannot send images to its model")

    monkeypatch.setattr(vc, "_ask", no_images)
    report = await vc.check_pdf(_config(tmp_path), "paper", _real_pdf(tmp_path), tmp_path)
    assert report["transport"] == "measurements only"
    assert "gemini_cli" in report["reason"]
    assert "metrics" in report["measured"] and report["findings"] == []
    saved = json.loads((tmp_path / ".fi" / "visual_check.json").read_text(encoding="utf-8"))
    assert saved["paper"]["transport"] == "measurements only"


@pytest.mark.asyncio
async def test_the_model_sees_every_page_and_its_grounded_findings_are_reported(tmp_path: Path, monkeypatch) -> None:
    sent: list = []

    async def model(config, messages, supervisor, quest_root):  # noqa: ANN001
        sent.extend(messages)
        return "```json\n" + json.dumps({"findings": [_finding(quote="appears nowhere at all")]}) + "\n```"

    monkeypatch.setattr(vc, "_ask", model)
    monkeypatch.setattr(vc, "measure_pdf", lambda path: _doc(["Results: the textbf command shows as text."]))
    report = await vc.check_pdf(_config(tmp_path), "paper", _real_pdf(tmp_path), tmp_path)
    parts = sent[0]["content"]
    assert parts[0]["type"] == "text" and "`table_overflow`" in parts[0]["text"]
    assert [p["type"] for p in parts[1:]] == ["image_url"]
    assert report["transport"] == "images" and report["pages_checked"] == 1
    assert report["findings"] == [] and report["dropped"][0]["dropped_because"] == "the quoted text is not on that page"
    assert (tmp_path / ".fi" / "visual_check" / "paper" / "page-1.png").is_file()


@pytest.mark.asyncio
async def test_reports_for_several_outputs_share_one_file_and_a_broken_pdf_never_raises(tmp_path: Path, monkeypatch) -> None:
    async def model(config, messages, supervisor, quest_root):  # noqa: ANN001
        return json.dumps({"findings": []})

    monkeypatch.setattr(vc, "_ask", model)
    await vc.check_pdf(_config(tmp_path), "paper", _real_pdf(tmp_path), tmp_path)
    broken = tmp_path / "poster.pdf"
    broken.write_bytes(b"not a pdf")
    report = await vc.check_pdf(_config(tmp_path), "poster", broken, tmp_path)
    assert report["transport"] == "none" and "could not be read" in report["error"]
    saved = json.loads((tmp_path / ".fi" / "visual_check.json").read_text(encoding="utf-8"))
    assert set(saved) == {"paper", "poster"}


# ---------------------------------------------------------------------------
# Redo


def _problem(check: str, severity: str = "high") -> dict:
    return {
        "check": check, "severity": severity, "page": 2, "region": "slide body",
        "problem": f"{check} on the slide", "quote": "some visible words",
    }


def _checked(*findings: dict, measured: tuple = ()) -> dict:
    return {"transport": "images", "measured": {"findings": list(measured)}, "findings": list(findings)}


def _script(monkeypatch, reports: list[dict]) -> list:
    seen = iter(reports)

    async def fake_check(config, kind, pdf, quest_root, *, supervisor=None):  # noqa: ANN001
        return next(seen)

    monkeypatch.setattr(vc, "check_pdf", fake_check)
    monkeypatch.setattr(vc, "_screenshots", lambda pdf, quest_root, kind: [])


def _deck(root: Path, version: str) -> Path:
    (root / "slides.pdf").write_text(f"pdf {version}", encoding="utf-8")
    (root / "slides.md").write_text(f"md {version}", encoding="utf-8")
    return root / "slides.pdf"


def _limit(redos: int) -> SimpleNamespace:
    return SimpleNamespace(output=SimpleNamespace(visual_check_max_redos=redos))


@pytest.mark.asyncio
async def test_slides_are_redone_with_the_findings_and_the_better_version_is_kept(tmp_path: Path, monkeypatch) -> None:
    _script(monkeypatch, [_checked(_problem("raw_markup")), _checked()])
    pdf = _deck(tmp_path, "first")
    feedback: list[str] = []

    async def regenerate(text: str) -> None:
        feedback.append(text)
        _deck(tmp_path, "second")

    report = await vc.check_and_redo(_limit(2), "slides", pdf, tmp_path, regenerate)
    assert len(feedback) == 1 and "raw_markup on the slide" in feedback[0] and "some visible words" in feedback[0]
    assert pdf.read_text(encoding="utf-8") == "pdf second"
    assert [(a["attempt"], a["kept"]) for a in report["attempts"]] == [(0, False), (1, True)]
    saved = json.loads((tmp_path / ".fi" / "visual_check.json").read_text(encoding="utf-8"))
    assert saved["slides"]["attempts"][1]["redo_for"] == ["raw_markup"]


@pytest.mark.asyncio
async def test_a_redo_that_checks_worse_is_put_back_file_for_file(tmp_path: Path, monkeypatch) -> None:
    _script(monkeypatch, [_checked(_problem("crowded_slide", "medium")), _checked(_problem("overlap"), _problem("cut_off_text"))])
    pdf = _deck(tmp_path, "first")

    async def regenerate(text: str) -> None:
        _deck(tmp_path, "worse")
        (tmp_path / "slides.pptx").write_text("new file", encoding="utf-8")

    report = await vc.check_and_redo(_limit(2), "slides", pdf, tmp_path, regenerate)
    assert pdf.read_text(encoding="utf-8") == "pdf first"
    assert (tmp_path / "slides.md").read_text(encoding="utf-8") == "md first"
    assert not (tmp_path / "slides.pptx").exists()
    assert [(a["attempt"], a["kept"]) for a in report["attempts"]] == [(0, True), (1, False)]
    assert report["findings"][0]["check"] == "crowded_slide"
    saved = json.loads((tmp_path / ".fi" / "visual_check.json").read_text(encoding="utf-8"))
    assert saved["slides"]["findings"][0]["check"] == "crowded_slide"


@pytest.mark.asyncio
async def test_redos_stop_at_the_configured_limit(tmp_path: Path, monkeypatch) -> None:
    _script(monkeypatch, [_checked(_problem("crowded_slide")), _checked(_problem("crowded_slide", "medium")), _checked()])
    pdf = _deck(tmp_path, "first")
    calls: list[str] = []

    async def regenerate(text: str) -> None:
        calls.append(text)

    report = await vc.check_and_redo(_limit(1), "slides", pdf, tmp_path, regenerate)
    assert len(calls) == 1 and len(report["attempts"]) == 2


@pytest.mark.asyncio
async def test_poster_layout_findings_never_ask_for_a_new_reply(tmp_path: Path, monkeypatch) -> None:
    layout = (_problem("empty_space"), _problem("column_balance"), _problem("overflow"))
    _script(monkeypatch, [_checked(_problem("reading_order"), measured=layout)])
    (tmp_path / "poster.pdf").write_text("pdf", encoding="utf-8")
    calls: list[str] = []

    async def regenerate(text: str) -> None:
        calls.append(text)

    report = await vc.check_and_redo(_limit(2), "poster", tmp_path / "poster.pdf", tmp_path, regenerate)
    assert calls == [] and report["attempts"] == [{"attempt": 0, "score": 12, "kept": True}]


@pytest.mark.asyncio
async def test_a_redo_that_fails_keeps_the_first_version(tmp_path: Path, monkeypatch) -> None:
    _script(monkeypatch, [_checked(_problem("raw_markup"))])
    pdf = _deck(tmp_path, "first")

    async def regenerate(text: str) -> None:
        _deck(tmp_path, "half written")
        raise RuntimeError("model timed out")

    report = await vc.check_and_redo(_limit(2), "slides", pdf, tmp_path, regenerate)
    assert pdf.read_text(encoding="utf-8") == "pdf first"
    assert "model timed out" in report["attempts"][1]["error"] and report["attempts"][0]["kept"] is True


def test_only_medium_and_high_findings_of_a_redo_check_count() -> None:
    report = _checked(_problem("raw_markup", "low"), _problem("reading_order"), measured=(_problem("small_font"),))
    assert [f["check"] for f in vc.redo_findings("slides", report)] == ["small_font"]
    assert vc.redo_findings("paper", report) == []
    paper = _checked(measured=(_problem("last_page_nearly_empty", "medium"), _problem("overwide")))
    assert [f["check"] for f in vc.redo_findings("paper", paper)] == ["last_page_nearly_empty"]


@pytest.mark.asyncio
async def test_a_paper_repair_that_checks_worse_puts_the_first_pdf_back(tmp_path: Path, monkeypatch) -> None:
    _script(monkeypatch, [
        _checked(measured=(_problem("last_page_nearly_empty", "medium"),)),
        _checked(_problem("overlap"), _problem("cut_off_text")),
    ])
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("pdf first", encoding="utf-8")

    async def repair(text: str) -> None:
        pdf.write_text("pdf taller", encoding="utf-8")

    report = await vc.check_and_redo(_limit(2), "paper", pdf, tmp_path, repair)
    assert pdf.read_text(encoding="utf-8") == "pdf first"
    assert [(a["attempt"], a["kept"]) for a in report["attempts"]] == [(0, True), (1, False)]


@pytest.mark.asyncio
async def test_the_paper_is_repaired_at_most_once(tmp_path: Path, monkeypatch) -> None:
    # The first repair helps but the last page is still nearly empty: a second
    # line would crowd the footer, so there is no second repair.
    _script(monkeypatch, [
        _checked(measured=(_problem("last_page_nearly_empty"), _problem("overwide"))),
        _checked(measured=(_problem("last_page_nearly_empty", "medium"),)),
        _checked(),
    ])
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("pdf first", encoding="utf-8")
    calls: list[str] = []

    async def repair(text: str) -> None:
        calls.append(text)

    report = await vc.check_and_redo(_limit(2), "paper", pdf, tmp_path, repair)
    assert len(calls) == 1 and [a["attempt"] for a in report["attempts"]] == [0, 1]


# ---------------------------------------------------------------------------
# Tick labels of the figures a slide shows


def _quest_with_figure(root: Path, *, record: dict | None, dpi: int | None = 100, size=(1310, 410), name="row") -> None:
    from PIL import Image

    (root / "figures").mkdir(exist_ok=True)
    Image.new("RGB", size, "white").save(root / "figures" / f"{name}.png", **({"dpi": (dpi, dpi)} if dpi else {}))
    if record is not None:
        (root / ".fi" / "figure_records").mkdir(parents=True, exist_ok=True)
        (root / ".fi" / "figure_records" / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")


def test_a_figure_source_knows_its_size_in_inches_and_where_its_tick_size_comes_from(tmp_path: Path) -> None:
    from core.plot_style import HOUSE_TICK_PT

    _quest_with_figure(tmp_path, record={"file": "own.png", "axes": [], "tick_pt": 8.0}, name="own")
    _quest_with_figure(tmp_path, record={"file": "older.png", "axes": []}, name="older")
    _quest_with_figure(tmp_path, record={"file": "bare.png", "axes": [], "tick_pt": None}, name="bare")
    _quest_with_figure(tmp_path, record=None, name="fetched")
    _quest_with_figure(tmp_path, record={"file": "no_dpi.png", "axes": []}, dpi=None, name="no_dpi")
    sources = {s.name: s for s in vc._figure_sources(tmp_path)}
    # The record's own size when it has one; a record from before the recorder wrote
    # it takes the house style's; no tick labels, no record and no dpi are left out.
    assert set(sources) == {"own.png", "older.png"}
    assert (sources["own.png"].tick_pt, sources["older.png"].tick_pt) == (8.0, HOUSE_TICK_PT)
    own = sources["own.png"]
    assert (own.width_px, own.height_px, round(own.width_in, 2)) == (1310, 410, 13.1)
    assert vc._figure_sources(tmp_path / "nothing") == []


def _slide_pdf(tmp_path: Path, *, image_width_pt: float = 403.0) -> Path:
    pdfium = pytest.importorskip("pypdfium2")
    import ctypes

    from PIL import Image

    raw = pdfium.raw
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(960.0, 540.0)
    for text, size, x, y in (
        ("Established outbreaks match theory", 27, 58, 450),
        ("A point beside the figure that is long enough.", 19, 58, 300),
    ):
        obj = raw.FPDFPageObj_NewTextObj(pdf.raw, b"Helvetica", ctypes.c_float(size))
        buf = ctypes.create_string_buffer((text + "\x00").encode("utf-16-le"))
        raw.FPDFText_SetText(obj, ctypes.cast(buf, ctypes.POINTER(raw.FPDF_WCHAR)))
        raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, y)
        raw.FPDFPage_InsertObject(page.raw, obj)
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (1310, 410), (40, 90, 140))))
    image.set_matrix(pdfium.PdfMatrix().scale(image_width_pt, image_width_pt * 410 / 1310).translate(500, 200))
    page.insert_obj(image)
    page.gen_content()
    page.close()
    path = tmp_path / "slides.pdf"
    pdf.save(path)
    pdf.close()
    return path


@pytest.mark.asyncio
async def test_the_slides_check_reads_each_figures_tick_size_from_its_file_and_asks_for_a_redo(tmp_path: Path) -> None:
    """The real measure of a real PDF, its image matched to the quest's figure file by
    its pixels: the finding is one the slides step is sent back for, with what to do."""
    _quest_with_figure(tmp_path, record={"file": "row.png", "axes": [], "tick_pt": 15.0})
    report = await vc.check_pdf(_config(tmp_path, ai=False), "slides", _slide_pdf(tmp_path), tmp_path)
    found = [f for f in report["measured"]["findings"] if f["check"] == "figure_ticks_small"]
    assert [(f["page"], f["severity"], f["figure"], f["tick_pt"]) for f in found] == [(1, "medium", "row.png", 6.4)]
    assert vc.redo_findings("slides", report) == found
    assert "slide of its own" in vc.feedback_text(found) and "no bullets" in vc.feedback_text(found)
    assert report["measured"]["metrics"]["per_page"][0]["figure_tick_pt"] == [6.4]
    # The same figure at 0.90 of the width it was drawn at is not reported.
    big = await vc.check_pdf(_config(tmp_path, ai=False), "slides", _slide_pdf(tmp_path, image_width_pt=846.0), tmp_path)
    assert [f for f in big["measured"]["findings"] if f["check"] == "figure_ticks_small"] == []


@pytest.mark.asyncio
async def test_only_the_slides_are_checked_for_figure_ticks_and_only_figures_the_recorder_saw(tmp_path: Path) -> None:
    pdf = _slide_pdf(tmp_path)
    _quest_with_figure(tmp_path, record={"file": "row.png", "axes": [], "tick_pt": 15.0})
    as_paper = await vc.check_pdf(_config(tmp_path, ai=False), "paper", pdf, tmp_path)
    assert [f for f in as_paper["measured"]["findings"] if f["check"] == "figure_ticks_small"] == []
    # A figure with no record (a fetched web figure, a web plot): its tick size is unknown.
    (tmp_path / ".fi" / "figure_records" / "row.json").unlink()
    as_slides = await vc.check_pdf(_config(tmp_path, ai=False), "slides", pdf, tmp_path)
    assert [f for f in as_slides["measured"]["findings"] if f["check"] == "figure_ticks_small"] == []


def test_a_figure_too_small_on_its_slide_is_a_redo_finding_only_when_a_new_deck_can_fix_it() -> None:
    shared = _problem("figure_ticks_small", "medium")
    alone = _problem("figure_ticks_small", "low")
    report = _checked(measured=(shared, alone))
    assert vc.redo_findings("slides", report) == [shared]
    assert vc.redo_findings("paper", report) == [] and vc.redo_findings("poster", report) == []


@pytest.mark.asyncio
async def test_a_deck_with_a_figure_under_the_floor_is_regenerated_with_what_to_do_and_no_more_than_the_cap(
    tmp_path: Path, monkeypatch,
) -> None:
    from generation._pdf_measure import _finding

    problem = _finding(
        "figure_ticks_small", 3, "slide",
        "The figure row.png is drawn 5.6 in wide. Give the figure a slide of its own: the slide title, one lead "
        "sentence and the figure, with no bullets, and put its discussion on the next slide.",
    )
    # Each new deck checks slightly better than the one before, so each is kept and the
    # loop runs to its cap: a deck that ignores the finding is redone twice, and no more.
    second = _finding("figure_ticks_small", 3, "slide", "a shorter one", "low")
    _script(monkeypatch, [_checked(measured=(problem, problem)), _checked(measured=(problem,)), _checked(measured=(second,)), _checked()])
    pdf = _deck(tmp_path, "first")
    feedback: list[str] = []

    async def regenerate(text: str) -> None:
        feedback.append(text)
        _deck(tmp_path, f"try {len(feedback)}")

    report = await vc.check_and_redo(_limit(2), "slides", pdf, tmp_path, regenerate)
    assert len(feedback) == 2 and "a slide of its own" in feedback[0]
    assert [a["attempt"] for a in report["attempts"]] == [0, 1, 2]
