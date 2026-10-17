"""paper.pdf sets each figure from its paper copy (generation/_paper_figures.py), so figure text prints at the body's
size: the copy is drawn at the text width with 8-9 pt text, and is placed at that width."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Config
from core.plot_style import write_boot
from generation import paper as paper_mod
from generation._paper_figures import prefer_paper_figures
from generation.paper import PaperGenerator
from tests.paper_fakes import fake_pandoc

PDF_63 = b"%PDF-1.4\n1 0 obj << /Type /Page /MediaBox [ 0 0 453.6 144 ] >> endobj\n"
PDF_31 = b"%PDF-1.4\n1 0 obj << /Type /Page /MediaBox [0 0 223.2 186] >> endobj\n"


def _copies(root: Path) -> None:
    paper = root / "figures" / "paper"
    paper.mkdir(parents=True)
    (paper / "wide.pdf").write_bytes(PDF_63)
    (paper / "single.pdf").write_bytes(PDF_31)
    # The figures themselves: a copy is used only beside its figure.
    for name in ("wide.png", "single.png"):
        (root / "figures" / name).write_bytes(b"png")


def test_a_figure_with_a_paper_copy_points_at_it_at_its_share_of_the_line(tmp_path: Path) -> None:
    _copies(tmp_path)
    md = (
        "![**Figure 1.** Bias [3] and spread.](figures/wide.png)\n\n"
        "![Scatter.](figures/single.png)\n\n"
        "![No copy.](figures/other.png)\n\n"
        "![Sized by the writer.](figures/wide.png){width=80%}\n"
    )
    out, n = prefer_paper_figures(md, tmp_path)
    assert n == 3
    assert out == (
        "![**Figure 1.** Bias [3] and spread.](figures/paper/wide.pdf){width=100% height=33%}\n\n"
        "![Scatter.](figures/paper/single.pdf){width=49% height=33%}\n\n"
        "![No copy.](figures/other.png)\n\n"
        "![Sized by the writer.](figures/paper/wide.pdf){width=80%}\n"
    )
    # A tall copy is capped at a third of the text height (the templates keep its aspect ratio): each copy FI sizes
    # says so, and the writer's own attribute block is left as it is.
    assert out.count("height=33%") == 2
    assert prefer_paper_figures(md, tmp_path, two_column=True) == (md, 0)
    assert prefer_paper_figures(md, tmp_path / "nowhere") == (md, 0)


def _config(tmp_path: Path, **output) -> Config:
    return Config.model_validate({"topic": "t", "title": "t", "output": {
        "kinds": ["paper_md", "paper_pdf"], "output_dir": str(tmp_path / "outputs"), **output}})


def test_the_latex_source_uses_the_paper_copy_and_paper_md_keeps_the_png(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(paper_mod, "find_pandoc", lambda _root: "/fake/pandoc")
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: ("pdflatex", "/fake/pdflatex"))

    def fake_run(cmd, **_kw):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    fake_pandoc(monkeypatch, fake_run)
    _copies(tmp_path)
    md = tmp_path / "paper.md"
    text = "# A title\n\n## Results\n\nText.\n\n![**Figure 1.** Bias.](figures/wide.png)\n\nMore text.\n"
    md.write_text(text, encoding="utf-8")
    pdf, skip = PaperGenerator(_config(tmp_path))._compile_pdf(md, tmp_path)
    assert skip is None and pdf is not None
    source = (tmp_path / "paper_pdf_source.md").read_text(encoding="utf-8")
    assert "](figures/paper/wide.pdf){width=100% height=33%}" in source and "figures/wide.png" not in source
    assert md.read_text(encoding="utf-8") == text
    # A two-column format keeps the PNG.
    PaperGenerator(_config(tmp_path, paper_format="ieee_access"))._compile_pdf(md, tmp_path)
    assert "figures/wide.png" in (tmp_path / "paper_pdf_source.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not (shutil.which("pandoc") or paper_mod.find_pandoc(paper_mod.REPO_ROOT))
                    or not shutil.which("pdflatex"), reason="needs pandoc and pdflatex")
def test_a_real_paper_prints_the_figure_text_at_the_body_size(tmp_path: Path) -> None:
    """Drawn by a script at 14 x 4.2 in through the bootstrap, compiled by pandoc and pdflatex: the axis label prints
    at about 9 pt (the generic template's line is a little wider than the copy's 6.3 in)."""
    pytest.importorskip("matplotlib")
    from generation._pdf_measure import measure_pdf

    boot = write_boot(tmp_path / ".fi", "latex")
    (tmp_path / "figures").mkdir()
    script = tmp_path / "plot.py"
    script.write_text(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.2))\n"
        "a.plot([0, 1, 2], [0, 1, 4]); b.plot([0, 1, 2], [4, 1, 0])\n"
        "a.set_xlabel('Elapsed time (s)'); b.set_xlabel('Elapsed time (s)')\n"
        "fig.savefig('figures/wide.png', dpi=150, bbox_inches='tight')\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(boot)}
    env.pop("FI_FIGURE_RECORDS", None)
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=180)
    assert done.returncode == 0, done.stderr
    md = tmp_path / "paper.md"
    md.write_text("# A title\n\n## Results\n\nSome text of the body.\n\n"
                  "![**Figure 1.** Two curves.](figures/wide.png)\n\nMore text.\n", encoding="utf-8")
    pdf, skip = PaperGenerator(_config(tmp_path))._compile_pdf(md, tmp_path)
    assert skip is None and pdf is not None and pdf.is_file()
    doc = measure_pdf(pdf)
    labels = [line for page in doc.pages for line in page.lines if "Elapsed time" in line.text]
    assert labels, [line.text for page in doc.pages for line in page.lines]
    # The size the PDF states is the figure's own (9 pt) whatever the figure is scaled to on the page; how big it
    # prints is the height of the same letter on the page over its height in the figure's own PDF.
    scale = _letter_height(pdf, "E") / _letter_height(tmp_path / "figures" / "paper" / "wide.pdf", "E")
    assert 0.95 <= scale <= 1.05, scale
    assert all(8.5 <= 9.0 * scale <= 9.5 for _ in labels)


def _letter_height(path: Path, letter: str) -> float:
    """The median height (pt) of ``letter`` set at 9 pt in ``path``, as drawn on the page."""
    import statistics

    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    pdf = pdfium.PdfDocument(str(path))
    heights = []
    try:
        for index in range(len(pdf)):
            text = pdf[index].get_textpage()
            for i in range(text.count_chars()):
                if text.get_text_range(i, 1) == letter and abs(raw.FPDFText_GetFontSize(text.raw, i) - 9.0) < 0.01:
                    box = text.get_charbox(i)
                    heights.append(box[3] - box[1])
    finally:
        pdf.close()
    assert heights, f"no 9 pt {letter!r} in {path}"
    return statistics.median(heights)
