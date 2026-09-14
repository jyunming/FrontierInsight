"""A paper whose last page holds only a line or two is recompiled taller.

The visual check never rewrites the paper. It asks ``_compile_pdf`` for a
text area one line taller, then two, which pulls a stray last line (the tail
of a long reference URL, on the validation quest) back onto the page before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Config
from generation import paper as paper_mod
from generation.paper import PaperGenerator

REPO = Path(__file__).resolve().parent.parent
FORMATS = sorted(p.parent.name for p in (REPO / "templates" / "paper").glob("*/template.tex"))


def _config(tmp_path: Path, paper_format: str = "generic") -> Config:
    return Config.model_validate({
        "topic": "t",
        "title": "t",
        "output": {
            "kinds": ["paper_md", "paper_pdf"],
            "output_dir": str(tmp_path / "outputs"),
            "paper_format": paper_format,
            "html_pdf_fallback": False,
        },
    })


def _paper(tmp_path: Path, text: str) -> tuple[Path, Path]:
    md = tmp_path / "paper.md"
    md.write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    return md, out


def test_extra_lines_reach_the_template_only_when_asked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(paper_mod, "find_pandoc", lambda _root: "/fake/pandoc")
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: ("pdflatex", "/fake/pdflatex"))
    commands: list[list[str]] = []

    def fake_run(cmd, **_kw):  # noqa: ANN001
        commands.append(list(cmd))
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)
    md, out = _paper(tmp_path, "# Title\n\n## Introduction\n\nBody.\n")
    generator = PaperGenerator(_config(tmp_path))
    generator._compile_pdf(md, out)
    generator._compile_pdf(md, out, extra_lines=2)
    assert not any(arg.startswith("fi-extra-lines") for arg in commands[0])
    assert commands[1][commands[1].index("fi-extra-lines=2") - 1] == "-V"


@pytest.mark.slow
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_template_lowers_its_last_body_line_when_asked(tmp_path: Path, fmt: str) -> None:
    from generation._pandoc import find_pandoc
    from generation._pdf_engine import find_pdf_engine
    from generation._pdf_measure import measure_pdf

    if find_pandoc(REPO) is None or find_pdf_engine(REPO) is None:
        pytest.skip("needs pandoc and a LaTeX engine")
    filler = " ".join(f"Sentence {k} fills the page with ordinary body text." for k in range(900))
    md, _ = _paper(tmp_path, f"# Taller Text Area\n\n## Body\n\n{filler}\n")

    def is_footer(text: str) -> bool:
        return text.strip().isdigit() or text.strip() == "Frontier Insight"

    lowest: list[float] = []
    footer: list[float] = []
    footer_top = 0.0
    pitch = 0.0
    for extra in (0, 1):
        out = tmp_path / f"out_{extra}"
        out.mkdir()
        pdf, skip = PaperGenerator(_config(tmp_path, fmt))._compile_pdf(md, out, extra_lines=extra)
        assert skip is None, skip.summary if skip else ""
        pages = measure_pdf(pdf).pages
        # A full page of body text: report and whitepaper open with a title
        # page and a section-heading page, and the last page is short.
        page = max(pages[:-1], key=lambda p: len(p.lines))
        body = sorted(line.box[1] for line in page.lines if line.visible and not is_footer(line.text))
        marks = [line.box[1] for line in page.lines if line.visible and is_footer(line.text)]
        lowest.append(body[0])
        footer.append(min(marks))
        footer_top = max(marks)
        pitch = min(b - a for a, b in zip(body, body[1:]) if b - a > 1)
    # One more line of text sits lower on the page, by about one baseline.
    assert 6 < lowest[0] - lowest[1] < 20, lowest
    # The footer stays where it was, within a fraction of a line: essay's
    # one-and-a-half spacing stretches its lines after the hook has run, and
    # its footer shifts 2.5 pt of a 17.9 pt line.
    assert abs(footer[0] - footer[1]) < pitch / 4, (footer, pitch)
    # And it stays clear of the text that moved down towards it.
    assert lowest[1] - footer_top > pitch / 2, (lowest, footer_top, pitch)
