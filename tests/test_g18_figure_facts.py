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


def test_negated_captions_give_no_finding(tmp_path):
    """Old code: a caption saying "no dotted line is drawn" was flagged as naming a dotted line (2 of 2 live captions), and
    "not in the upper panel" would have been flagged the same way. Captions are no longer read against the record."""
    root = _quest(tmp_path, _record(SIDE, SIDE_RIGHT), _lines("-", "-"))
    for caption in ("Power against m; no dotted line is drawn, the bound is not shown.",
                    "Gap in the right panel; nothing is drawn in the upper panel and there is no lower panel."):
        assert not hasattr(paper_facts, "figure_hits")
        assert not hasattr(figure_facts, "caption_findings")
        assert paper_facts.check(f"![{caption}](figures/fig_a.png)", root) == []


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
