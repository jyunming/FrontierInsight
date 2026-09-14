"""The visual check of slides.pptx, exported to PDF through LibreOffice, and
the brief per-output summary the web quest page shows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import Config
from generation import _visual_check as vc


def _config(tmp_path: Path) -> Config:
    return Config.model_validate({"topic": "t", "title": "t", "output": {"output_dir": str(tmp_path / "out")}})


def _exported_pdf(out_dir: Path) -> Path:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 540), "white")
    ImageDraw.Draw(image).text((40, 40), "Stable integrators capture energy decay", fill="black")
    pdf = out_dir / "slides.pptx.pdf"
    image.save(pdf, "PDF")
    return pdf


@pytest.mark.asyncio
async def test_without_libreoffice_the_pptx_is_reported_as_not_checked(tmp_path: Path, monkeypatch) -> None:
    async def never(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("no PDF, so no model call")

    monkeypatch.setattr(vc, "pptx_to_pdf", lambda pptx, out_dir: (None, "LibreOffice was not found"))
    monkeypatch.setattr(vc, "_ask", never)
    report = await vc.check_pptx(_config(tmp_path), tmp_path / "slides.pptx", tmp_path)
    assert report == {
        "kind": "slides_pptx", "pdf": "slides.pptx", "transport": "none", "reason": "LibreOffice was not found",
    }
    saved = json.loads((tmp_path / ".fi" / "visual_check.json").read_text(encoding="utf-8"))
    assert saved["slides_pptx"]["reason"] == "LibreOffice was not found"


@pytest.mark.asyncio
async def test_the_exported_pptx_is_checked_as_a_slide_deck_under_its_own_name(tmp_path: Path, monkeypatch) -> None:
    exports: list[tuple[Path, Path]] = []
    prompts: list[str] = []

    def export(pptx: Path, out_dir: Path):  # noqa: ANN202
        exports.append((pptx, out_dir))
        return _exported_pdf(out_dir), ""

    async def model(config, messages, supervisor, quest_root):  # noqa: ANN001
        prompts.append(messages[0]["content"][0]["text"])
        return json.dumps({"findings": []})

    monkeypatch.setattr(vc, "pptx_to_pdf", export)
    monkeypatch.setattr(vc, "_ask", model)
    report = await vc.check_pptx(_config(tmp_path), tmp_path / "slides.pptx", tmp_path)
    shots = tmp_path / ".fi" / "visual_check" / "slides_pptx"
    assert exports == [(tmp_path / "slides.pptx", shots)]
    assert report["kind"] == "slides_pptx" and report["pdf"] == "slides.pptx.pdf"
    assert report["transport"] == "images" and report["pages_checked"] == 1
    # The slides' checklist and measurements, the pptx's own name.
    assert "`slide_overflow`" in prompts[0] and "PowerPoint file" in prompts[0]
    assert set(report["measured"]["metrics"]) == {"pages", "per_page"}
    assert (shots / "page-1.png").is_file()
    saved = json.loads((tmp_path / ".fi" / "visual_check.json").read_text(encoding="utf-8"))
    assert set(saved) == {"slides_pptx"}


def test_the_summary_gives_each_output_its_label_counts_and_redos(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "visual_check.json").write_text(json.dumps({
        "slides": {
            "transport": "images",
            "findings": [{"check": "overlap"}],
            "measured": {"findings": [{"check": "overflow"}, {"check": "small_font"}]},
            "attempts": [{"attempt": 0, "kept": False}, {"attempt": 1, "kept": True}],
        },
        "slides_pptx": {"transport": "none", "reason": "LibreOffice was not found"},
    }), encoding="utf-8")
    assert vc.report_summary(tmp_path) == {
        "slides": {
            "label": "slides", "transport": "images", "problems_seen": 1, "problems_measured": 2,
            "redos": 1, "reason": None,
        },
        "slides_pptx": {
            "label": "slides.pptx", "transport": "none", "problems_seen": 0, "problems_measured": 0,
            "redos": 0, "reason": "LibreOffice was not found",
        },
    }


def test_no_report_or_a_broken_one_gives_no_summary(tmp_path: Path) -> None:
    assert vc.report_summary(tmp_path) is None
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "visual_check.json").write_text("{not json", encoding="utf-8")
    assert vc.report_summary(tmp_path) is None
