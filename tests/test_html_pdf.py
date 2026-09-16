"""Tests for the HTML/Chromium PDF fallback (generation/_html_pdf.py) and
its wiring into PaperGenerator._compile_pdf.

The unit tests are fully mocked (CI-safe). The end-to-end render test is
gated on pandoc + a Chromium-family browser being available, so it runs on
a developer box but skips cleanly on a headless CI runner."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from core.config import Config
from generation import _html_pdf
from generation import paper as paper_mod
from generation._html_pdf import (
    _split_title,
    find_html_browser,
    raw_tex_outside_math,
    render_paper_html_pdf,
)
from generation._pandoc import HTML_MARKDOWN_READER
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


def test_html_render_puts_the_author_line_in_the_byline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each set author field becomes a pandoc ``author`` entry (one byline
    row); with none set the byline stays Frontier Insight."""
    seen: list[list[str]] = []

    def fake_run(cmd, **_kw):  # noqa: ANN001
        seen.append(list(cmd))
        raise OSError("stop after pandoc")

    monkeypatch.setattr(_html_pdf.subprocess, "run", fake_run)
    pmd = tmp_path / "paper.md"
    pmd.write_text("# T\n\nbody\n", encoding="utf-8")

    def authors(**kw) -> list[str]:  # noqa: ANN003
        seen.clear()
        render_paper_html_pdf(pmd, tmp_path / "paper.pdf", pandoc_path="pandoc",
                              browser=("msedge", "edge"), **kw)
        cmd = seen[0]
        return [cmd[i + 1] for i, a in enumerate(cmd) if a == "--metadata" and cmd[i + 1].startswith("author=")]

    assert authors() == ["author=Frontier Insight"]
    assert authors(author_line=("Jane Chen", "R&D Lab")) == ["author=Jane Chen", "author=R&D Lab"]


def test_the_html_render_takes_the_writers_figure_number_off_the_caption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both themes number figures with a CSS counter."""
    def fake_run(cmd, **_kw):  # noqa: ANN001
        raise OSError("stop after writing the body")

    monkeypatch.setattr(_html_pdf.subprocess, "run", fake_run)
    pmd = tmp_path / "paper.md"
    pmd.write_text("# T\n\n![**Figure 1.** Energy decay.](figures/energy.png)\n", encoding="utf-8")
    render_paper_html_pdf(pmd, tmp_path / "paper.pdf", pandoc_path="pandoc", browser=("msedge", "edge"))
    body = (tmp_path / "paper_html_body.md").read_text(encoding="utf-8")
    assert "![Energy decay.](figures/energy.png)" in body
    for theme in ("latexlike.css", "briefing.css"):
        css = (Path(_html_pdf.__file__).resolve().parents[1] / "templates" / "paper" / "_html" / theme).read_text(encoding="utf-8")
        assert 'figcaption::before { content: "Figure " counter(fig)' in css, theme


def test_compile_pdf_hands_the_author_line_to_the_html_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config.model_validate({
        "topic": "t", "title": "t",
        "output": {"kinds": ["paper_md", "paper_pdf"],
                   "output_dir": str(tmp_path / "outputs"),
                   "paper_style": "briefing",
                   "author": "Jane Chen", "contact_email": "jane@example.org"},
    })
    gen = PaperGenerator(cfg)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paper_md = tmp_path / "paper.md"
    paper_md.write_text("# T\n\nbody\n", encoding="utf-8")
    monkeypatch.setattr(
        paper_mod.shutil, "which", lambda n: "pandoc" if n == "pandoc" else None)
    monkeypatch.setattr(
        "generation._html_pdf.find_html_browser", lambda: ("msedge", "edge"))
    captured: dict[str, object] = {}

    def fake_render(pmd, out_pdf, **kw):  # noqa: ANN001
        captured.update(kw)
        out_pdf.write_bytes(b"%PDF-1.4 fake")
        return out_pdf, ""

    monkeypatch.setattr("generation._html_pdf.render_paper_html_pdf", fake_render)
    gen._compile_pdf(paper_md, out_dir)
    assert captured["author_line"] == ("Jane Chen", "jane@example.org")


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


# ---------------------------------------------------------------------------
# Math: what the browser path must not do to a formula
# ---------------------------------------------------------------------------


def _pandoc_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, md: str) -> list[str]:
    """The pandoc command the HTML render builds for ``md`` (stops there)."""
    seen: list[list[str]] = []

    def fake_run(cmd, **_kw):  # noqa: ANN001
        seen.append(list(cmd))
        raise OSError("stop after pandoc")

    monkeypatch.setattr(_html_pdf.subprocess, "run", fake_run)
    pmd = tmp_path / "paper.md"
    pmd.write_text(md, encoding="utf-8")
    render_paper_html_pdf(
        pmd, tmp_path / "paper.pdf", pandoc_path="pandoc", browser=("msedge", "edge"))
    return seen[0]


def test_the_html_render_reads_backslash_paren_as_math(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``tex_math_single_backslash`` pandoc reads ``\\(`` as an
    escaped parenthesis and the commands between the delimiters as raw TeX —
    which the HTML writer cannot emit and therefore DELETED. The delivered
    paper printed "zero when (R_0)", a whole sentence whose meaning had
    changed, and a table header as "(R_0) (N) () () ()"."""
    cmd = _pandoc_cmd(tmp_path, monkeypatch, "# T\n\nzero when \\(R_0\\leq1\\).\n")
    frm = [a for a in cmd if a.startswith("--from=")]
    assert frm, cmd
    assert "tex_math_single_backslash" in frm[0], frm


def test_the_html_render_keeps_the_tex_it_cannot_typeset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-raw_tex`` is the safety net: LaTeX the HTML writer can't express
    reaches the page as its own source instead of vanishing. A renderer that
    deletes a formula changes what the paper says; one that prints the source
    merely looks wrong, which the reader can see."""
    cmd = _pandoc_cmd(tmp_path, monkeypatch, "# T\n\nbody \\SI{3}{\\micro}\n")
    frm = [a for a in cmd if a.startswith("--from=")][0]
    assert frm.endswith("-raw_tex"), frm


def test_the_html_render_warns_about_tex_it_cannot_typeset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserved-but-not-typeset is still a defect, so it is named in
    run.log rather than passing silently."""
    with caplog.at_level(logging.WARNING, logger="frontier_insight.paper"):
        _pandoc_cmd(tmp_path, monkeypatch, "# T\n\nbody \\SI{3}{\\micro}\n")
    assert "cannot typeset" in caplog.text
    assert "\\SI" in caplog.text


def test_raw_tex_outside_math_ignores_real_math_and_code() -> None:
    """Commands INSIDE math are the ones that do render (as MathML), so
    reporting them would make the warning meaningless."""
    assert raw_tex_outside_math(r"a \(R_0\leq1\) b $\alpha$ c `\beta` d") == []
    assert raw_tex_outside_math(r"\[\frac{a}{b}\] and $$\sum x$$") == []
    assert raw_tex_outside_math(r"a unit \SI{3}{\micro} here") == [r"\SI", r"\micro"]
    # A markdown escape of one punctuation char is not a lost formula.
    assert raw_tex_outside_math(r"file\_name and AT\&T") == []


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="needs pandoc")
def test_pandoc_really_sets_backslash_math_as_mathml_and_keeps_stray_tex(
    tmp_path: Path,
) -> None:
    """End-to-end through real pandoc, on the exact shape the codex-written
    SIR papers used: the formula becomes MathML instead of the bare text
    "(R_0)", and a stray command stays visible instead of being dropped."""
    md = tmp_path / "t.md"
    md.write_text(
        "zero when \\(R_0\\leq1\\) and \\SI{3}{\\micro}.\n", encoding="utf-8")
    html = subprocess.run(
        [shutil.which("pandoc"), str(md), "-t", "html", "--mathml",
         f"--from={HTML_MARKDOWN_READER}"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    assert "<math" in html, html
    assert "(R_0)" not in html, html
    assert "\\SI{3}{\\micro}" in html, html
