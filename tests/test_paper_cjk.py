"""Papers with Chinese, Japanese or Korean text compile with XeLaTeX.

pdflatex stops at the first CJK character in the title, the body or the
author line. ``PaperGenerator._compile_pdf`` switches to XeLaTeX with a CJK
font when both are installed, and otherwise goes straight to the browser
render instead of running a compile that cannot succeed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Config
from generation import _cjk
from generation import paper as paper_mod
from generation._pandoc import find_pandoc
from generation.paper import PaperGenerator

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = sorted((REPO / "templates" / "paper").glob("*/template.tex"))


def _config(tmp_path: Path, **output) -> Config:
    return Config.model_validate({
        "topic": "t",
        "title": "t",
        "output": {
            "kinds": ["paper_md", "paper_pdf"],
            "output_dir": str(tmp_path / "outputs"),
            **output,
        },
    })


def _paper(tmp_path: Path, text: str) -> tuple[Path, Path]:
    md = tmp_path / "paper.md"
    md.write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    return md, out


def _fake_tools(monkeypatch, *, xelatex=("xelatex", "/fake/xelatex"), font="Microsoft JhengHei"):
    monkeypatch.setattr(paper_mod, "find_pandoc", lambda _root: "/fake/pandoc")
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: ("pdflatex", "/fake/pdflatex"))
    monkeypatch.setattr(_cjk, "find_xelatex", lambda engine: xelatex)
    monkeypatch.setattr(_cjk, "find_cjk_font", lambda text: font)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kw):
        calls.append(list(cmd))
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)
    return calls


def test_a_paper_with_chinese_text_compiles_with_xelatex_and_a_cjk_font(tmp_path, monkeypatch):
    calls = _fake_tools(monkeypatch)
    md, out = _paper(tmp_path, "# 分子動力學積分器\n\n## Introduction\n\nBody.\n")
    pdf, skip = PaperGenerator(_config(tmp_path))._compile_pdf(md, out)
    assert skip is None and pdf == out / "paper.pdf"
    (cmd,) = calls
    assert "--pdf-engine=/fake/xelatex" in cmd
    assert cmd[cmd.index("fi-cjk-font=Microsoft JhengHei") - 1] == "-V"


def test_a_chinese_author_line_alone_switches_the_engine(tmp_path, monkeypatch):
    calls = _fake_tools(monkeypatch)
    md, out = _paper(tmp_path, "# Probe Title\n\n## Introduction\n\nBody.\n")
    PaperGenerator(_config(tmp_path, author="陳建明"))._compile_pdf(md, out)
    assert "--pdf-engine=/fake/xelatex" in calls[0]


def test_a_latin_paper_keeps_pdflatex(tmp_path, monkeypatch):
    calls = _fake_tools(monkeypatch)
    md, out = _paper(tmp_path, "# Probe Title\n\n## Introduction\n\nBody.\n")
    PaperGenerator(_config(tmp_path, author="Jane Chen"))._compile_pdf(md, out)
    assert "--pdf-engine=/fake/pdflatex" in calls[0]
    assert not any(part.startswith("fi-cjk-font=") for part in calls[0])


@pytest.mark.parametrize(
    "xelatex, font, code",
    [
        (None, "Microsoft JhengHei", "cjk_no_xelatex"),
        (("xelatex", "/fake/xelatex"), None, "cjk_no_font"),
    ],
)
def test_without_xelatex_or_a_font_the_paper_goes_to_the_browser_render(
    tmp_path, monkeypatch, xelatex, font, code,
):
    calls = _fake_tools(monkeypatch, xelatex=xelatex, font=font)
    reasons: list[str] = []

    def no_browser(self, paper_md, out_dir, pandoc_exe, *, why):
        reasons.append(why)
        return None

    monkeypatch.setattr(PaperGenerator, "_try_html_pdf_fallback", no_browser)
    md, out = _paper(tmp_path, "# 分子動力學\n\nBody.\n")
    pdf, skip = PaperGenerator(_config(tmp_path))._compile_pdf(md, out)
    assert calls == []  # no pdflatex run that cannot succeed
    assert len(reasons) == 1 and "Chinese, Japanese or Korean" in reasons[0]
    assert pdf is None and skip.code == code


def test_the_browser_render_stands_in_when_no_cjk_font_is_installed(tmp_path, monkeypatch):
    _fake_tools(monkeypatch, font=None)
    md, out = _paper(tmp_path, "# 分子動力學\n\nBody.\n")
    rendered = out / "paper.pdf"

    def browser(self, paper_md, out_dir, pandoc_exe, *, why):
        rendered.write_bytes(b"%PDF-html\n")
        return rendered

    monkeypatch.setattr(PaperGenerator, "_try_html_pdf_fallback", browser)
    assert PaperGenerator(_config(tmp_path))._compile_pdf(md, out) == (rendered, None)


def _render_template(template: Path, tmp_path: Path, args: list[str]) -> str:
    pandoc = find_pandoc(REPO)
    if pandoc is None:
        pytest.skip("pandoc not available")
    src = tmp_path / "in.md"
    src.write_text("---\ntitle: T\n---\n\nBody.\n", encoding="utf-8")
    out = tmp_path / f"{template.parent.name}.tex"
    subprocess.run(
        [pandoc, str(src), "-o", str(out), "--standalone", "--template", str(template), *args],
        check=True, capture_output=True,
    )
    return " ".join(out.read_text(encoding="utf-8").split())


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.parent.name)
def test_every_paper_template_loads_xecjk_under_xelatex_only_with_a_font(template, tmp_path):
    with_font = _render_template(template, tmp_path, ["-V", "fi-cjk-font=Noto Sans CJK TC"])
    assert (
        r"\ifXeTeX \usepackage{fontspec} \usepackage{xeCJK} \setCJKmainfont{Noto Sans CJK TC}"
        in with_font
    )
    assert r"\else \usepackage[utf8]{inputenc} \fi" in with_font
    without = _render_template(template, tmp_path, [])
    assert "xeCJK" not in without
    assert r"\usepackage[utf8]{inputenc}" in without


@pytest.mark.slow
def test_a_real_xelatex_compile_prints_a_chinese_title_body_and_author(tmp_path):
    from generation._pdf_engine import find_pdf_engine
    from generation._pdf_measure import measure_pdf

    text = "# 分子動力學積分器\n\n## Introduction\n\nBody text on 能量守恆.\n"
    engine = find_pdf_engine(REPO)
    if (
        find_pandoc(REPO) is None or engine is None
        or _cjk.find_xelatex(engine) is None or _cjk.find_cjk_font(text) is None
    ):
        pytest.skip("needs pandoc, XeLaTeX and a CJK font")
    md, out = _paper(tmp_path, text)
    cfg = _config(tmp_path, author="陳建明", affiliation="國立台灣大學", html_pdf_fallback=False)
    pdf, skip = PaperGenerator(cfg)._compile_pdf(md, out)
    assert skip is None, skip.summary if skip else ""
    doc = measure_pdf(pdf)
    text_layer = " ".join(line.text for page in doc.pages for line in page.lines)
    for expected in ("分子動力學積分器", "陳建明", "國立台灣大學", "能量守恆"):
        assert expected in text_layer
