r"""A paper's layout: a table fits its column, a URL breaks inside it, no column is pulled apart, and the
measurement of the PDF sees each of these when it goes wrong.

Neutral made-up content throughout. The PDFs that need pandoc and a LaTeX engine are skipped without them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.config import Config
from generation import _pdf_measure as pm
from generation import _visual_check as vc
from generation._pandoc import find_pandoc
from generation._pdf_engine import find_pdf_engine
from generation._tables import fit_tables_to_column
from generation.paper import PaperGenerator

pdfium = pytest.importorskip("pypdfium2")

from tests.test_pdf_measure import _column, _pdf  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "templates" / "paper"

WIDE = (
    "| Quantity | Baseline | Best result | Improvement over the baseline |\n"
    "|---|---|---|---|\n"
    "| at the first settings | 0 | 0 | 0 units |\n"
    "| at the same settings again | 0 | 0 | 0 units |\n"
)


def _dashes(markdown: str) -> list[int]:
    rule = next(line for line in markdown.split("\n") if re.fullmatch(r"\|[-:|]+\|", line))
    return [len(cell.strip(":")) for cell in rule.strip("|").split("|")]


# ---------------------------------------------------------------------------
# The table's widths


def test_a_wide_table_gets_widths_from_its_text() -> None:
    out = fit_tables_to_column(WIDE, two_column=True)
    dashes = _dashes(out)
    assert len(dashes) == 4 and dashes != [3, 3, 3, 3]
    assert max(dashes) > min(dashes), "the column with the most text gets the most room"
    assert sum(dashes) > 72, "a line longer than pandoc's --columns is what makes it wrap the cells"
    assert "```" not in out, "the longest words fit side by side, so no smaller type"


def test_the_longest_word_of_each_column_always_fits() -> None:
    out = fit_tables_to_column(WIDE, two_column=True)
    dashes = _dashes(out)
    usable = 46 - 2.4 * 4
    words = [len("Quantity"), len("Baseline"), len("result"), len("Improvement")]
    for share, word in zip(dashes, words):
        assert share / sum(dashes) * usable >= word - 0.5


def test_a_table_that_fits_is_left_as_written() -> None:
    small = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert fit_tables_to_column(small, two_column=True) == small
    assert fit_tables_to_column(small, two_column=False) == small


def test_the_same_table_needs_no_smaller_type_in_one_column() -> None:
    # 6.5 in holds twice what a column does: the same table fits at the body size.
    assert max(len(line) for line in WIDE.splitlines()) <= 72, "so pandoc would not wrap it by itself"
    out = fit_tables_to_column(WIDE, two_column=False)
    assert out == WIDE


def test_alignment_marks_and_code_are_kept() -> None:
    table = WIDE.replace("|---|---|---|---|", "|:---|---:|:---:|---|")
    out = fit_tables_to_column(table, two_column=True)
    rule = next(line for line in out.split("\n") if line.startswith("|:-"))
    cells = rule.strip("|").split("|")
    assert cells[0].startswith(":") and not cells[0].endswith(":")
    assert cells[1].endswith(":") and not cells[1].startswith(":")
    assert cells[2].startswith(":") and cells[2].endswith(":")
    fenced = "```\n" + WIDE + "```\n"
    assert fit_tables_to_column(fenced, two_column=True) == fenced


def test_words_that_cannot_sit_side_by_side_set_the_table_smaller() -> None:
    crowded = (
        "| Experimental condition | Measurement uncertainty | Reproducibility estimate | Calibration reference |"
        " Environmental influence |\n|---|---|---|---|---|\n| first | 1 | 2 | 3 | 4 |\n"
    )
    out = fit_tables_to_column(crowded, two_column=True)
    assert r"\begingroup\footnotesize" in out or r"\begingroup\scriptsize" in out
    assert out.rstrip().endswith("```") and r"\endgroup" in out
    assert fit_tables_to_column(WIDE, two_column=True).count("begingroup") == 0


def test_a_caption_stays_with_the_table_it_names() -> None:
    crowded = (
        "| Experimental condition | Measurement uncertainty | Reproducibility estimate | Calibration reference |"
        " Environmental influence |\n|---|---|---|---|---|\n| first | 1 | 2 | 3 | 4 |\n\nTable: What the rows are.\n\nAfter.\n"
    )
    out = fit_tables_to_column(crowded, two_column=True)
    assert out.index("Table: What the rows are.") < out.index(r"\endgroup") < out.index("After.")
    assert out.index(r"\begingroup") < out.index("| Experimental condition")


def test_a_redo_sets_every_table_a_size_smaller() -> None:
    out = fit_tables_to_column(WIDE, two_column=True, smaller=True)
    assert r"\begingroup\footnotesize" in out
    small = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert r"\footnotesize" in fit_tables_to_column(small, two_column=True, smaller=True)


# ---------------------------------------------------------------------------
# The templates


@pytest.mark.parametrize("fmt", sorted(p.parent.name for p in TEMPLATES.glob("*/template.tex")))
def test_every_template_breaks_a_url_anywhere(fmt: str) -> None:
    text = (TEMPLATES / fmt / "template.tex").read_text(encoding="utf-8")
    assert r"\usepackage{xurl}" in text
    assert text.index(r"\usepackage{hyperref}") < text.index(r"\usepackage{xurl}"), "xurl goes after hyperref"


def test_the_two_column_template_ends_columns_ragged_and_keeps_the_column_room() -> None:
    text = (TEMPLATES / "ieee_access" / "template.tex").read_text(encoding="utf-8")
    assert r"\raggedbottom" in text
    # longtable resets the room left in the column; the table's box must put it back.
    assert text.index(r"\FIcolroom=\@colroom") < text.index(r"\global\@colroom=\FIcolroom")


# ---------------------------------------------------------------------------
# The measurement


def _lines_of_text(x: float, top: float, count: int, text: str, step: float = 12.0):
    return _column(x, top, top - step * (count - 1), 10, text, step)


PROSE = "The quick brown fox jumps over the lazy dog"


def test_text_drawn_over_text_is_found(tmp_path: Path) -> None:
    items = _lines_of_text(54, 700, 30, PROSE) + _lines_of_text(320, 700, 30, PROSE)
    items.append(("text", "Baseline Best design Improvement over the baseline", 10, 180, 400, False))
    report = pm.paper_report(pm.measure_pdf(_pdf(tmp_path, [(612, 792, items)])))
    over = [f for f in report["findings"] if f["check"] == "text_overlap"]
    assert len(over) == 1 and over[0]["severity"] == "high"


def test_a_page_of_columns_has_no_overlap_and_no_gap(tmp_path: Path) -> None:
    items = _lines_of_text(54, 700, 40, PROSE) + _lines_of_text(320, 700, 40, PROSE)
    report = pm.paper_report(pm.measure_pdf(_pdf(tmp_path, [(612, 792, items)])))
    assert [f for f in report["findings"] if f["check"] in ("text_overlap", "column_gap")] == []


def test_paragraphs_pulled_apart_in_a_column_are_found(tmp_path: Path) -> None:
    left = (_lines_of_text(54, 700, 6, PROSE) + _lines_of_text(54, 590, 6, PROSE) + _lines_of_text(54, 480, 6, PROSE)
            + _lines_of_text(54, 370, 6, PROSE))
    right = _lines_of_text(320, 700, 40, PROSE)
    report = pm.paper_report(pm.measure_pdf(_pdf(tmp_path, [(612, 792, left + right)])))
    gaps = [f for f in report["findings"] if f["check"] == "column_gap"]
    assert len(gaps) == 1 and "left column" in gaps[0]["problem"], report["findings"]


def test_the_space_around_a_float_in_a_column_is_not_a_gap(tmp_path: Path) -> None:
    # paragraph, a 150 pt figure with its caption, paragraph: 70 pt either side of the figure is the float's own space.
    left = (_lines_of_text(54, 700, 6, PROSE) + [("image", 54, 480, 280, 630)]
            + [("text", "Figure 1: A made-up picture of nothing", 9, 54, 470, False)]
            + _lines_of_text(54, 380, 10, PROSE))
    right = _lines_of_text(320, 700, 40, PROSE)
    report = pm.paper_report(pm.measure_pdf(_pdf(tmp_path, [(612, 792, left + right)])))
    assert [f for f in report["findings"] if f["check"] == "column_gap"] == [], report["findings"]


def test_a_blank_stretch_under_the_last_line_of_a_column_is_not_a_gap(tmp_path: Path) -> None:
    left = _lines_of_text(54, 700, 14, PROSE)
    right = _lines_of_text(320, 700, 40, PROSE)
    report = pm.paper_report(pm.measure_pdf(_pdf(tmp_path, [(612, 792, left + right)])))
    assert [f for f in report["findings"] if f["check"] == "column_gap"] == []


def test_the_checks_a_recompile_repairs() -> None:
    checks = vc.REDO_CHECKS["paper"]
    assert {"overwide", "overflow", "text_overlap", "last_page_nearly_empty"} <= checks
    assert "column_gap" not in checks, "the template ends columns ragged: a recompile would change nothing"
    assert vc.paper_repairs(f"{vc.PAPER_FEEDBACK_TAG} last_page_nearly_empty\n- page 3") == (True, False)
    assert vc.paper_repairs(f"{vc.PAPER_FEEDBACK_TAG} text_overlap overwide\n- page 3") == (False, True)
    assert vc.paper_repairs("no tag") == (True, False), "untagged feedback is the old last-page repair"


# ---------------------------------------------------------------------------
# A real paper, two columns


def _paper() -> str:
    body = ("A paragraph of plain words about a made-up study, long enough to fill lines of a column and to be set "
            "in two columns next to its neighbour, saying nothing about any particular field of research. ") * 3
    url = "https://www.example.org/archive/publication/2718281828_A_Very_Long_Title_With_Underscores_And_Words_" \
          "That_Never_End/some-section-of-the-page/with-more-path-segments-than-fit-in-a-column/index.html"
    parts = ["# A made-up study", "", "## Abstract", "", "Short.", ""]
    for i in range(1, 6):
        parts += [f"## Section {i}", "", body, "", body, ""]
        if i == 3:
            parts += ["The comparison is in the table.", "", WIDE, "", body, ""]
    parts += ["## References", "", f"1. A. Author (2020). A title. {url}", "", f"- [W1] Another title. {url}", ""]
    return "\n".join(parts)


def _build(tmp_path: Path, fmt: str, monkeypatch=None, fit: bool = True) -> Path:
    if find_pandoc(REPO) is None or find_pdf_engine(REPO) is None:
        pytest.skip("needs pandoc and a LaTeX engine")
    import generation.paper as paper_mod

    if not fit and monkeypatch is not None:
        import generation._tables as tables

        monkeypatch.setattr(tables, "fit_tables_to_column", lambda md, **kw: md)
    md = tmp_path / "paper.md"
    md.write_text(_paper(), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    cfg = Config.model_validate({"topic": "x", "output": {"paper_format": fmt, "output_dir": str(tmp_path)}})
    pdf, skip = PaperGenerator(cfg)._compile_pdf(md, out)
    assert skip is None, skip.summary if skip else ""
    assert paper_mod  # the module under test is the one imported
    return pdf


BAD = {"overflow", "overwide", "text_overlap", "column_gap"}


@pytest.mark.parametrize("fmt", ["ieee_access", "generic", "neurips"])
def test_a_paper_with_a_wide_table_and_long_urls_has_no_layout_problem(tmp_path: Path, fmt: str) -> None:
    pdf = _build(tmp_path, fmt)
    report = pm.paper_report(pm.measure_pdf(pdf))
    assert [f["check"] + ": " + f["problem"] for f in report["findings"] if f["check"] in BAD] == []


def test_without_the_fit_the_same_paper_is_caught_by_the_measurement(tmp_path: Path, monkeypatch) -> None:
    pdf = _build(tmp_path, "ieee_access", monkeypatch, fit=False)
    found = {f["check"] for f in pm.paper_report(pm.measure_pdf(pdf))["findings"]}
    assert found & {"overflow", "overwide", "text_overlap"}, found
