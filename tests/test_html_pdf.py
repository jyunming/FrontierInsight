"""Tests for the HTML/Chromium PDF fallback (generation/_html_pdf.py) and
its wiring into PaperGenerator._compile_pdf.

The unit tests are fully mocked (CI-safe). The end-to-end render test is
gated on pandoc + a Chromium-family browser being available, so it runs on
a developer box but skips cleanly on a headless CI runner."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.config import Config
from generation import _html_pdf
from generation import paper as paper_mod
from generation._html_pdf import (
    _split_title,
    find_html_browser,
    render_paper_html_pdf,
)
from generation.paper import PaperGenerator


# ---------------------------------------------------------------------------
# _split_title
# ---------------------------------------------------------------------------


def test_split_title_strips_first_heading() -> None:
    title, body = _split_title("# My Paper\n\nAbstract text\n\n## Intro\nx")
    assert title == "My Paper"
    assert "# My Paper" not in body
    assert "Abstract text" in body and "## Intro" in body


def test_split_title_no_heading_returns_body_unchanged() -> None:
    title, body = _split_title("no leading heading here")
    assert title == ""
    assert body == "no leading heading here"


# ---------------------------------------------------------------------------
# find_html_browser
# ---------------------------------------------------------------------------


def test_find_html_browser_finds_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _html_pdf.shutil, "which",
        lambda n: "/usr/bin/google-chrome" if n == "google-chrome" else None,
    )
    assert find_html_browser() == ("google-chrome", "/usr/bin/google-chrome")


def test_find_html_browser_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_html_pdf.shutil, "which", lambda _n: None)
    # No default-location file exists either.
    monkeypatch.setattr(_html_pdf.Path, "is_file", lambda _self: False)
    assert find_html_browser() is None


# ---------------------------------------------------------------------------
# _compile_pdf wiring
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path, *, fallback: bool) -> Config:
    return Config.model_validate({
        "topic": "t", "title": "t",
        "output": {
            "kinds": ["paper_md", "paper_pdf"],
            "output_dir": str(tmp_path / "outputs"),
            "html_pdf_fallback": fallback,
        },
    })


def test_compile_pdf_uses_html_fallback_when_no_latex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No LaTeX engine + fallback on + a browser present ⇒ paper.pdf is
    rendered via the HTML path instead of skipping."""
    gen = PaperGenerator(_cfg(tmp_path, fallback=True))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paper_md = tmp_path / "paper.md"
    paper_md.write_text("# T\n\nbody\n", encoding="utf-8")

    # pandoc present, but no LaTeX engine.
    monkeypatch.setattr(
        paper_mod.shutil, "which", lambda n: "pandoc" if n == "pandoc" else None)
    monkeypatch.setattr(gen, "_find_pdf_engine", lambda: None)
    monkeypatch.setattr(
        "generation._html_pdf.find_html_browser", lambda: ("msedge", "edge"))

    captured: dict[str, Path] = {}

    def fake_render(pmd, out_pdf, **_kw):  # noqa: ANN001
        captured["out_pdf"] = out_pdf
        out_pdf.write_bytes(b"%PDF-1.4 fake")
        return out_pdf, ""

    monkeypatch.setattr("generation._html_pdf.render_paper_html_pdf", fake_render)

    pdf, skip = gen._compile_pdf(paper_md, out_dir)
    assert skip is None
    assert pdf == out_dir / "paper.pdf"
    assert captured["out_pdf"] == out_dir / "paper.pdf"


def test_compile_pdf_skips_when_fallback_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``output.html_pdf_fallback: false`` ⇒ strict LaTeX-only; the HTML
    path is never consulted and a no_latex_engine skip is returned."""
    gen = PaperGenerator(_cfg(tmp_path, fallback=False))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paper_md = tmp_path / "paper.md"
    paper_md.write_text("# T\n\nbody\n", encoding="utf-8")

    monkeypatch.setattr(
        paper_mod.shutil, "which", lambda n: "pandoc" if n == "pandoc" else None)
    monkeypatch.setattr(gen, "_find_pdf_engine", lambda: None)

    def must_not_call():
        raise AssertionError("HTML fallback must not run when disabled")

    monkeypatch.setattr(
        "generation._html_pdf.find_html_browser", must_not_call)

    pdf, skip = gen._compile_pdf(paper_md, out_dir)
    assert pdf is None
    assert skip is not None and skip.code == "no_latex_engine"


def test_compile_pdf_briefing_style_uses_briefing_theme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``output.paper_style=briefing`` renders via the HTML backend with the
    briefing theme — even when a LaTeX engine IS available (the chosen look
    wins over the LaTeX path)."""
    cfg = Config.model_validate({
        "topic": "t", "title": "t",
        "output": {"kinds": ["paper_md", "paper_pdf"],
                   "output_dir": str(tmp_path / "outputs"),
                   "paper_style": "briefing"},
    })
    gen = PaperGenerator(cfg)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paper_md = tmp_path / "paper.md"
    paper_md.write_text("# T\n\nbody\n", encoding="utf-8")

    monkeypatch.setattr(
        paper_mod.shutil, "which", lambda n: "pandoc" if n == "pandoc" else None)
    # A LaTeX engine IS present — briefing must still win.
    monkeypatch.setattr(gen, "_find_pdf_engine", lambda: ("pdflatex", "/fake/pdflatex"))
    monkeypatch.setattr(
        "generation._html_pdf.find_html_browser", lambda: ("msedge", "edge"))

    captured: dict[str, object] = {}

    def fake_render(pmd, out_pdf, **kw):  # noqa: ANN001
        captured["css_path"] = kw.get("css_path")
        out_pdf.write_bytes(b"%PDF-1.4 fake")
        return out_pdf, ""

    monkeypatch.setattr("generation._html_pdf.render_paper_html_pdf", fake_render)

    pdf, skip = gen._compile_pdf(paper_md, out_dir)
    assert skip is None
    assert pdf == out_dir / "paper.pdf"
    css = captured["css_path"]
    assert css is not None and Path(css).name == "briefing.css", captured


@pytest.mark.skipif(
    shutil.which("pandoc") is None or find_html_browser() is None,
    reason="needs pandoc + a Chromium-family browser for a real render",
)
def test_render_paper_html_pdf_integration(tmp_path: Path) -> None:
    """Real end-to-end: pandoc + a headless browser produce a non-empty PDF
    with the title block + sections."""
    work = tmp_path / "out"
    work.mkdir()
    pmd = work / "paper.md"
    pmd.write_text(
        "# Probe Title\n\n## Abstract\nHello world.\n\n## Introduction\nBody.\n",
        encoding="utf-8",
    )
    pdf, detail = render_paper_html_pdf(
        pmd, work / "paper.pdf",
        pandoc_path=shutil.which("pandoc"), browser=find_html_browser(),
    )
    assert pdf is not None, detail
    assert pdf.is_file() and pdf.stat().st_size > 0
