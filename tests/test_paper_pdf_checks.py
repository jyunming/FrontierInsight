"""The measured checks of ``paper_report`` that read a paper's pages with no model.

Each defect is built into a small PDF with pdflatex (a positive), and a clean
paper, built here and a stored one, gives none of the new findings (a negative):
a check that finds what is not there would send a good paper back for repair.

The stored clean paper is the PDF an ARIS run delivered
(``tests/fixtures/paper_latex/aris_kinetics_main.pdf``). Its figures are vector, with
5.5-6 pt labels, so the existing ``small_font`` check does report them; no other check
does. The review-mode line numbers down its margin are not text that ran into it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pypdfium2")

from generation import _pdf_measure as pm


REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "paper_latex"
ARIS = FIXTURES / "aris_kinetics_main.pdf"

NEW_CHECKS = {
    "bottom_overflow", "doubled_number", "raw_table", "raw_markup", "overlap",
    "orphan_table_row", "pages_over_limit", "producer_is_fallback",
}

HEAD = (
    "\\documentclass[11pt,a4paper]{article}\n"
    "\\usepackage[T1]{fontenc}\\usepackage{lmodern}\\usepackage{graphicx}\n"
    "\\usepackage{booktabs,longtable}\\usepackage[margin=1in]{geometry}\n"
    "\\begin{document}\n"
)
SENTENCE = (
    "The estimate of the activation energy is recovered within a few percent of the true "
    "value across the noise levels that were tried in the simulation study here. "
)


def _filler(paragraphs: int) -> str:
    return "".join(f"{SENTENCE * 4}\n\n" for _ in range(paragraphs))


def _build(tmp_path: Path, name: str, body: str) -> Path:
    """A PDF made by pdflatex from ``body``; skips without pdflatex (a host with only
    tectonic or XeLaTeX skips too, rather than failing to start it)."""
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("needs pdflatex")
    from PIL import Image

    Image.new("RGB", (800, 500), "white").save(tmp_path / "fig.png")
    tex = tmp_path / f"{name}.tex"
    tex.write_text(HEAD + body + "\n\\end{document}\n", encoding="utf-8")
    run = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=tmp_path, capture_output=True, creationflags=_NO_WINDOW,
    )
    assert run.returncode == 0, run.stdout[-600:]
    return tmp_path / f"{name}.pdf"


_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _report(pdf: Path, **kw) -> dict:
    return pm.paper_report(pm.measure_pdf(pdf), **kw)


def _kinds(report: dict) -> list[str]:
    return [f["check"] for f in report["findings"]]


# ---------------------------------------------------------------------------
# Positives: one defect each


def test_a_figure_in_the_bottom_margin_is_found(tmp_path: Path) -> None:
    body = (
        _filler(14) + "\\newpage\n"
        + "A page whose figure is drawn below the text area, on the page number.\n"
        + "\\raisebox{-\\textheight}[0pt][0pt]{\\raisebox{-2.5cm}[0pt][0pt]{\\includegraphics[height=4cm]{fig.png}}}\n"
        + "\\newpage\n" + _filler(2)
    )
    report = _report(_build(tmp_path, "bottom", body))
    found = [f for f in report["findings"] if f["check"] == "bottom_overflow"]
    assert found and found[0]["severity"] == "high", report["findings"]
    assert any("A figure reaches" in f["problem"] and "bottom margin" in f["problem"] for f in found), found


def test_a_heading_numbered_twice_is_found(tmp_path: Path) -> None:
    body = (
        "\\section*{1 1. Introduction}\n" + _filler(2)
        + "\\subsection*{3.3 3.3 Coverage}\n" + _filler(2)
        + "\\section*{Methods}\n" + _filler(1)
    )
    found = [f for f in _report(_build(tmp_path, "doubled", body))["findings"] if f["check"] == "doubled_number"]
    assert [f["problem"] for f in found] == [
        'A heading prints its number twice: "1 1. Introduction".',
        'A heading prints its number twice: "3.3 3.3 Coverage".',
    ]


def test_a_heading_numbered_once_is_not(tmp_path: Path) -> None:
    body = "\\section{Introduction}\n" + _filler(2) + "\\subsection{Aim}\n" + _filler(1)
    assert "doubled_number" not in _kinds(_report(_build(tmp_path, "once", body)))


def test_table_rows_printed_with_their_bars_are_found(tmp_path: Path) -> None:
    rows = "".join(
        f"\\noindent {m} | 0.{k}0 | 0.0400 / 0.0400 | 0.4710 | 0.1640 | 0.3070 |\\\\\n"
        for k, m in enumerate((200, 2000, 2000), start=1)
    )
    body = _filler(3) + rows + _filler(3)
    found = [f for f in _report(_build(tmp_path, "raw_table", body))["findings"] if f["check"] == "raw_table"]
    assert found and found[0]["severity"] == "high" and "| bars" in found[0]["problem"]


def test_a_real_table_is_not_raw(tmp_path: Path) -> None:
    table = (
        "\\begin{table}[t]\\centering\\caption{Estimates.}\\begin{tabular}{lll}\\toprule\n"
        "Setting & Estimate & Coverage \\\\\\midrule\n"
        + "".join(f"S{i} & 0.{i}1 & 0.9{i} \\\\\n" for i in range(1, 5)) + "\\bottomrule\\end{tabular}\\end{table}\n"
    )
    assert "raw_table" not in _kinds(_report(_build(tmp_path, "table", _filler(3) + table + _filler(2))))


def test_latex_math_shown_as_text_is_found(tmp_path: Path) -> None:
    body = _filler(2) + "\\noindent\\texttt{The fit gives \\$E\\_a = 50\\$ kJ and \\textbackslash times 10.}\n\n" + _filler(1)
    found = [f for f in _report(_build(tmp_path, "math", body))["findings"] if f["check"] == "raw_markup"]
    assert found and "LaTeX math as text" in found[0]["problem"]


def test_a_figure_drawn_over_text_is_found(tmp_path: Path) -> None:
    body = (
        _filler(2)
        + "\\noindent\\makebox[0pt][l]{\\raisebox{-3cm}[0pt][0pt]{\\includegraphics[width=7cm]{fig.png}}}\n"
        + SENTENCE * 12 + "\n\n" + _filler(1)
    )
    found = [f for f in _report(_build(tmp_path, "overlap", body))["findings"] if f["check"] == "overlap"]
    assert found and "figure is drawn over" in found[0]["problem"]


TABLE = (
    "\\begin{longtable}{lll}\n\\toprule\nSetting & Estimate & Coverage \\\\\n\\midrule\n\\endhead\n"
    + "".join(f"S{i} & 0.{i}1 & 0.9{i} \\\\\n" for i in range(1, 9))
    + "\\bottomrule\n\\end{longtable}\nText after the table.\n"
)


def _lines_before_a_table(count: int) -> str:
    return "".join(f"\\noindent Line {k} of the filler text that fills the page.\\\\\n" for k in range(count))


def test_the_end_of_a_table_alone_on_the_next_page_is_found(tmp_path: Path) -> None:
    # 40 lines then the table: the table's last row, and the repeated header above it, land alone on page 2.
    pdf = _build(tmp_path, "orphan", _lines_before_a_table(40) + TABLE)
    doc = pm.measure_pdf(pdf)
    assert len(doc.pages) == 2 and [l.text for l in doc.pages[1].lines][:2] == ["Setting Estimate Coverage", "S8"]
    found = [f for f in pm.paper_report(doc)["findings"] if f["check"] == "orphan_table_row"]
    assert found and found[0]["page"] == 2 and found[0]["rows"] == 1, found


def test_a_table_split_into_several_rows_per_page_is_not_an_orphan(tmp_path: Path) -> None:
    pdf = _build(tmp_path, "split", _lines_before_a_table(45) + TABLE)
    assert "orphan_table_row" not in _kinds(_report(pdf))


def test_a_table_that_fits_on_its_page_is_not_an_orphan(tmp_path: Path) -> None:
    pdf = _build(tmp_path, "whole", _lines_before_a_table(10) + TABLE)
    assert "orphan_table_row" not in _kinds(_report(pdf))


def test_more_pages_than_the_limit_is_found_and_every_page_counts(tmp_path: Path) -> None:
    pdf = _build(tmp_path, "long", _filler(40))
    doc = pm.measure_pdf(pdf)
    pages = len(doc.pages)
    assert pages >= 3
    found = [f for f in pm.paper_report(doc, page_limit=pages - 1)["findings"] if f["check"] == "pages_over_limit"]
    assert found and found[0]["severity"] == "high" and f"{pages} pages" in found[0]["problem"]
    assert "pages_over_limit" not in _kinds(pm.paper_report(doc, page_limit=pages))
    assert "pages_over_limit" not in _kinds(pm.paper_report(doc))


def test_a_pdf_a_browser_made_is_flagged_unless_the_style_is_meant_for_it() -> None:
    doc = pm.measure_pdf(ARIS)
    assert doc.producer and "pdfTeX" in doc.producer
    doc.producer = "Skia/PDF m155"
    found = [f for f in pm.paper_report(doc)["findings"] if f["check"] == "producer_is_fallback"]
    assert found and "not typeset" in found[0]["problem"]
    assert "producer_is_fallback" not in _kinds(pm.paper_report(doc, expect_latex=False))


# ---------------------------------------------------------------------------
# Negatives: a paper with none of these defects gives none of the findings


def test_a_clean_paper_built_here_gives_no_finding_at_all(tmp_path: Path) -> None:
    table = (
        "\\begin{table}[t]\\centering\\caption{Estimates.}\\begin{tabular}{lll}\\toprule\n"
        "Setting & Estimate & Coverage \\\\\\midrule\n"
        + "".join(f"S{i} & 0.{i}1 & 0.9{i} \\\\\n" for i in range(1, 5)) + "\\bottomrule\\end{tabular}\\end{table}\n"
    )
    figure = "\\begin{figure}[t]\\centering\\includegraphics[width=0.6\\linewidth]{fig.png}\\caption{A figure.}\\end{figure}\n"
    body = (
        "\\section{Introduction}\n" + _filler(8) + table + "\\section{Results}\n" + _filler(8)
        + figure + _filler(8) + "\\subsection{Notes}\n" + _filler(6)
    )
    doc = pm.measure_pdf(_build(tmp_path, "clean", body))
    assert len(doc.pages) >= 3
    report = pm.paper_report(doc, page_limit=len(doc.pages))
    assert report["findings"] == [], report["findings"]


def test_the_stored_clean_paper_gives_none_of_the_new_findings() -> None:
    doc = pm.measure_pdf(ARIS)
    report = pm.paper_report(doc, page_limit=len(doc.pages))
    assert not [f for f in report["findings"] if f["check"] in NEW_CHECKS], report["findings"]
    # The margin line numbers (000-215) are not text that ran into the margin ...
    assert "overwide" not in _kinds(report)
    # ... and the one finding left is the existing small-text check on its vector figures' 5.5-6 pt labels.
    assert set(_kinds(report)) <= {"small_font"}, report["findings"]


def test_the_pdf_of_a_paper_with_margin_line_numbers_has_them_ignored_by_the_edge_check() -> None:
    doc = pm.measure_pdf(ARIS)
    page = doc.pages[0]
    numbers = pm._line_number_ids(page)
    assert len(numbers) >= 50
    assert all(pm._LINE_NUMBER_RE.match(line.text) for line in page.lines if id(line) in numbers)


# ---------------------------------------------------------------------------
# What a reviewer of these checks found they reported wrongly, one case each


def _line(text: str, x0: float, y0: float, *, x1: float | None = None, size: float = 10.0, bold: bool = False) -> pm.Line:
    return pm.Line(text=text, size=size, bold=bold, box=(x0, y0, x1 if x1 is not None else x0 + 6.0 * len(text), y0 + size))


def _page(lines: list, number: int = 1) -> pm.Page:
    return pm.Page(number=number, width=612.0, height=792.0, lines=lines, images=[], rules=[])


def _prose(count: int, top: float = 700.0) -> list:
    return [_line(SENTENCE[:80], 72.0, top - 14.0 * k, x1=540.0) for k in range(count)]


def test_a_displayed_absolute_value_is_not_a_table_row() -> None:
    lines = _prose(4) + [
        _line("|x̂ − x| ≤ 0.1 for every setting", 200.0, 600.0),
        _line("|ŷ − y| ≤ 0.2 at the largest noise", 200.0, 580.0),
        _line("|z| < 3 in all runs", 200.0, 560.0),
    ]
    assert pm._raw_table_findings(_page(lines)) == []


def test_shell_and_r_dollars_in_a_listing_are_not_math_left_as_text() -> None:
    lines = [
        _line("cp $HOME/data.csv $USER/out", 72.0, 600.0),
        _line("total <- df$a + df$b", 72.0, 580.0),
    ]
    assert pm._raw_math_findings(_page(lines), paper=True) == []
    # ... while a dollar span with math markup in it still is.
    found = pm._raw_math_findings(_page([_line("The fit gives $E_a = 50$ kJ", 72.0, 600.0)]), paper=True)
    assert found and found[0]["check"] == "raw_markup"


def test_a_table_column_of_whole_numbers_inside_the_text_block_is_not_line_numbers() -> None:
    rows = [_line(str(10 * (k + 1)), 300.0, 500.0 - 14.0 * k) for k in range(8)]
    page = _page(_prose(12) + rows)
    assert pm._line_number_ids(page) == set()
    # The same numbers down the left margin, outside the text block, are line numbers.
    margin = [_line(str(k + 1), 40.0, 700.0 - 14.0 * k) for k in range(8)]
    page = _page(_prose(12) + margin)
    assert pm._line_number_ids(page) == {id(line) for line in margin}


def test_a_sentence_that_repeats_a_count_is_not_a_doubled_heading_number() -> None:
    doc = pm.Document(pages=[_page(_prose(6) + [
        _line("4 4 × 4 grids were run at each noise level, as the plan set out.", 72.0, 500.0, x1=540.0),
        _line("2 2 runs failed.", 72.0, 486.0),
    ])])
    assert pm._doubled_number_findings(doc, 10.0) == []
    doc = pm.Document(pages=[_page(_prose(6) + [_line("2 2. Methods", 72.0, 500.0, size=12.0, bold=True)])])
    assert len(pm._doubled_number_findings(doc, 10.0)) == 1


# ---------------------------------------------------------------------------
# Stored papers a browser drew (pandoc's HTML route): s1 kinetics and s1 fdr


S1_KINETICS = FIXTURES / "s1_kinetics_browser.pdf"
S1_FDR = FIXTURES / "s1_fdr_browser.pdf"


def test_the_stored_browser_kinetics_paper_shows_its_doubled_numbers_and_its_producer() -> None:
    report = pm.paper_report(pm.measure_pdf(S1_KINETICS))
    doubled = [f["problem"] for f in report["findings"] if f["check"] == "doubled_number"]
    assert len(doubled) == 9, doubled
    assert any('"3.3 3.3 Coverage' in p for p in doubled), doubled
    assert "producer_is_fallback" in _kinds(report)


def test_the_stored_browser_fdr_paper_shows_the_table_row_alone_on_page_3() -> None:
    report = pm.paper_report(pm.measure_pdf(S1_FDR))
    found = [f for f in report["findings"] if f["check"] == "orphan_table_row"]
    assert [(f["page"], f["rows"]) for f in found] == [(3, 1)], found


def test_a_running_header_on_every_page_is_not_a_table_header() -> None:
    # The whitepaper and policy brief templates print the title at the top of every page.
    def page(number: int, words: str) -> pm.Page:
        header = _line("Error control in multiple testing", 72.0, 740.0, size=8.0)
        body = [_line(f"{words} {k} " + SENTENCE[:70], 72.0, 700.0 - 14.0 * k, x1=540.0) for k in range(10)]
        return _page([header] + body, number=number)

    doc = pm.Document(pages=[page(1, "First"), page(2, "Second"), page(3, "Third")])
    heads = pm._head_line_ids(doc)
    assert len(heads) == 3
    assert pm._orphan_table_row_findings(doc, 10.0) == []
