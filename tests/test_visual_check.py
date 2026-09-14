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


def _config(tmp_path: Path) -> Config:
    return Config.model_validate({"topic": "t", "title": "t", "output": {"output_dir": str(tmp_path / "out")}})


def _real_pdf(tmp_path: Path, text: str = "Results: the textbf command shows as text.") -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 800), "white")
    ImageDraw.Draw(image).text((40, 40), text, fill="black")
    pdf = tmp_path / "paper.pdf"
    image.save(pdf, "PDF")
    return pdf


@pytest.mark.asyncio
async def test_a_transport_without_images_leaves_a_measurements_only_report(tmp_path: Path, monkeypatch) -> None:
    async def no_images(config, messages, supervisor):  # noqa: ANN001
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

    async def model(config, messages, supervisor):  # noqa: ANN001
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
    async def model(config, messages, supervisor):  # noqa: ANN001
        return json.dumps({"findings": []})

    monkeypatch.setattr(vc, "_ask", model)
    await vc.check_pdf(_config(tmp_path), "paper", _real_pdf(tmp_path), tmp_path)
    broken = tmp_path / "poster.pdf"
    broken.write_bytes(b"not a pdf")
    report = await vc.check_pdf(_config(tmp_path), "poster", broken, tmp_path)
    assert report["transport"] == "none" and "could not be read" in report["error"]
    saved = json.loads((tmp_path / ".fi" / "visual_check.json").read_text(encoding="utf-8"))
    assert set(saved) == {"paper", "poster"}
