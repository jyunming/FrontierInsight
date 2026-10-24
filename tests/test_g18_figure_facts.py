"""FI states a figure's panel layout and line styles from its own records (core/figure_facts.py), and a caption that
places a panel or a line style the figure does not have is sent back (core/paper_facts.py::figure_hits)."""

from __future__ import annotations

import json
from pathlib import Path

from core import figure_facts, paper_facts

SIDE = [0.1, 0.15, 0.35, 0.7]
SIDE_RIGHT = [0.55, 0.15, 0.35, 0.7]
TOP = [0.1, 0.55, 0.8, 0.35]
BOTTOM = [0.1, 0.1, 0.8, 0.35]


def _record(*places, titles=("", "")):
    return {"file": "fig_a.png", "axes": [
        {"title": titles[i] if i < len(titles) else "", "ylabel": "gain", "place": p, "series": []}
        for i, p in enumerate(places)]}


def _lines(*styles):
    return {"axes": [{"lines": [{"label": f"s{i}", "kind": "data", "x": [0, 1, 2], "y": [1, 2, 3],
                                 "linestyle": st, "marker": "None"} for i, st in enumerate(styles)]}]}


def _quest(tmp_path: Path, record, lines=None, stem="fig_a") -> Path:
    folder = tmp_path / ".fi" / "figure_records"
    folder.mkdir(parents=True)
    (folder / f"{stem}.json").write_text(json.dumps(record), encoding="utf-8")
    if lines is not None:
        (folder / f"{stem}.seed0.json").write_text(json.dumps(lines), encoding="utf-8")
    return tmp_path


def test_side_by_side_panels_are_not_upper_and_lower(tmp_path):
    """Old code: nothing compared a caption's "upper/lower panel" with the figure, so two panels drawn side by side were
    described as stacked in 4 of 4 graded papers. The record puts them in one row."""
    root = _quest(tmp_path, _record(SIDE, SIDE_RIGHT, titles=("Fit", "Residual")), _lines("-", "-"))
    assert figure_facts.layout(_record(SIDE, SIDE_RIGHT))["arrangement"] == "side_by_side"
    paper = "![Upper panel: the fit. Lower panel: the residual.](figures/fig_a.png)"
    (hit,) = paper_facts.figure_hits(paper, root)
    assert "upper panel" in hit and "side by side" in hit and "{{figure:fig_a:layout}}" in hit


def test_stacked_panels_allow_top_and_bottom_but_not_left(tmp_path):
    """Known positive: top and bottom of stacked panels pass; a left panel of a stacked pair does not."""
    root = _quest(tmp_path, _record(TOP, BOTTOM), _lines("-"))
    assert figure_facts.layout(_record(TOP, BOTTOM))["arrangement"] == "stacked"
    assert paper_facts.figure_hits("![Top panel: the fit. Bottom panel: the residual.](figures/fig_a.png)", root) == []
    assert len(paper_facts.figure_hits("![Left panel: the fit.](figures/fig_a.png)", root)) == 1


def test_twin_axis_is_one_panel_and_a_left_axis_is_not_a_panel(tmp_path):
    """A twin axis shares its panel's box; "left axis" names an axis, not a panel."""
    record = _record(SIDE, SIDE)
    assert figure_facts.layout(record)["panels"] == 1
    root = _quest(tmp_path, record, _lines("-"))
    assert paper_facts.figure_hits("![The gain on the left axis and the cost on the right axis.](figures/fig_a.png)",
                                   root) == []


def test_dotted_baseline_that_is_not_drawn_is_flagged(tmp_path):
    """Old code: a caption naming "dotted linearised baseline curves" passed although the figure draws solid lines only."""
    root = _quest(tmp_path, _record(SIDE), _lines("-", "-"))
    (hit,) = paper_facts.figure_hits("![Measured gain with dotted baseline curves.](figures/fig_a.png)", root)
    assert "dotted" in hit and "no dashed or dotted line is drawn" in hit


def test_dotted_line_that_is_drawn_passes(tmp_path):
    root = _quest(tmp_path, _record(SIDE), _lines("-", ":"))
    assert paper_facts.figure_hits("![Measured gain with a dotted baseline curve.](figures/fig_a.png)", root) == []
    assert "dotted" in figure_facts.lines_sentence(_lines("-", ":"))


def test_no_record_says_nothing(tmp_path):
    assert paper_facts.figure_hits("![Upper panel.](figures/other.png)", tmp_path) == []
    assert figure_facts.facts(tmp_path) == {}


def test_sentences(tmp_path):
    assert figure_facts.layout_sentence(_record(SIDE, SIDE_RIGHT, titles=("Fit", "Residual"))) == \
        'two panels side by side in one row (left: "Fit"; right: "Residual")'
    assert figure_facts.layout_sentence(_record(TOP, BOTTOM, titles=("A", "B"))) ==         'two panels stacked in one column (top: "A"; bottom: "B")'
    assert figure_facts.lines_sentence(_lines("-")) == "solid lines are drawn; no dashed or dotted line is drawn"


def test_placeholder_is_filled_marked_and_turned_back(tmp_path):
    root = _quest(tmp_path, _record(SIDE, SIDE_RIGHT, titles=("Fit", "Residual")), _lines("-"))
    out = paper_facts.render("The figure has {{figure:fig_a:layout}}.", root).text
    assert "two panels side by side in one row" in out and "<!-- fi:fact-figure-layout-fig_a -->" in out
    assert paper_facts.to_placeholders(out) == "The figure has {{figure:fig_a:layout}}."
    assert "side by side" in paper_facts.without_marks(out)


def test_placeholder_of_an_unknown_figure_reads_not_recorded(tmp_path):
    root = _quest(tmp_path, _record(SIDE), _lines("-"))
    rendered = paper_facts.render("See {{figure:nope:layout}}.", root)
    assert paper_facts.NOT_RECORDED in rendered.text and "figure:nope:layout" in rendered.unfilled


def test_writer_is_given_the_figure_facts(tmp_path):
    root = _quest(tmp_path, _record(SIDE, SIDE_RIGHT), _lines("-", "--"))
    section = paper_facts.writer_section(root)
    assert "{{figure:fig_a:layout}}" in section and "{{figure:fig_a:lines}}" in section
    assert "dashed lines" in section
