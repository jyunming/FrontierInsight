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
        {"title": titles[i] if i < len(titles) else "", "ylabel": (titles[i] if i < len(titles) else "") or "gain", "place": p, "series": []}
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
    # The layout and the line styles are in FI's caption of the figure now, not placeholders the writer types.
    assert "{{figure:" not in section
    assert "in the caption FI writes under it" in section


def test_a_grid_with_an_unused_place_states_the_panels_it_draws(tmp_path):
    """A 2 by 2 grid with three plots: the layout sentence says three panels and one blank place, not four panels."""
    places = [[0.1, 0.55, 0.35, 0.35], [0.55, 0.55, 0.35, 0.35], [0.1, 0.1, 0.35, 0.35], [0.55, 0.1, 0.35, 0.35]]
    record = _record(*places, titles=("A", "B", "C", ""))
    drawn = {"axes": [{"lines": [{"label": "s", "kind": "data", "x": [0, 1], "y": [1, 2]}]} for _ in range(3)]
             + [{"lines": [], "draws_other": False, "legend": False, "has_text": False}]}
    assert figure_facts.layout_sentence(record, drawn).startswith("three panels in a grid of 2 rows by 2 columns")
    assert figure_facts.layout_sentence(record, drawn).endswith("one more place on the figure is left blank")
    assert figure_facts.layout_sentence(record).startswith("four panels")  # without the line file nothing is known


def test_an_old_record_without_bar_lengths_says_error_bars_are_not_recorded(tmp_path):
    """A record made before the bar lengths were kept cannot say a series has no bars: it says they are not recorded."""
    from core.engine import _figure_record_note, _read_figure_records

    record = {"file": "fig_a.png", "axes": [{"title": "", "ylabel": "gain", "place": SIDE, "ylim": [0, 10], "series": [
        {"label": "stub", "shows": "no points"}]}]}
    lines = {"axes": [{"lines": [
        {"label": "A", "kind": "data", "x": [1, 2, 3], "y": [1, 2, 3], "linestyle": "None", "marker": "o"},
        {"label": "B", "kind": "data", "x": [1, 2, 3], "y": [1, 2, 3], "linestyle": "-", "marker": "None"}]}]}
    root = _quest(tmp_path, record, lines)
    note = _figure_record_note(_read_figure_records(root / ".fi" / "figure_records", ["fig_a.png"])["fig_a.png"])
    assert "A: markers only, no line joins them (3 points)" in note
    assert "B: a solid line, no markers (3 points)" in note
    assert "stub: a legend entry with no data" in note
    assert "error bars: not recorded" in note and "no error bars" not in note


def test_a_figure_tells_at_most_twelve_series_then_how_many_more(tmp_path):
    lines = {"axes": [{"bars_recorded": True, "error_bar_sets": 0, "lines": [
        {"label": f"s{i}", "kind": "data", "x": [1, 2, 3], "y": [1, 2, 3], "linestyle": "-", "marker": "None"}
        for i in range(15)]}]}
    record = {"file": "fig_a.png", "axes": [{"title": "", "ylabel": "gain", "place": SIDE, "series": []}]}
    root = _quest(tmp_path, record, lines)
    figure_facts.attach_series_drawn(record, root / ".fi" / "figure_records" / "fig_a.seed0.json")
    panel = record["axes"][0]
    assert panel["series_drawn"][:12] == [f"s{i}: a solid line, no markers (3 points)" for i in range(12)]
    assert len(panel["series_drawn"]) == 12 and panel["series_drawn_more"] == 3
    assert panel["series_drawn_tail"] == "no error bars are drawn"
    from core.engine import _figure_record_note

    note = _figure_record_note(record)
    assert "s11: a solid line, no markers (3 points); and 3 more; no error bars are drawn" in note


def test_a_panel_without_a_title_is_described_by_its_axes_never_as_if_an_axis_label_were_its_title():
    """Live fault: the layout sentence said `panel 1: "Realised rate"` where "Realised rate" was the y axis label and the
    panel's own title (what distinguishes it from the others) was not said."""
    record = {"file": "fig_a.png", "axes": [
        {"title": "Fraction of real effects = 0", "ylabel": "Realised rate", "xlabel": "tests", "place": SIDE, "series": []},
        {"title": "", "ylabel": "Realised rate", "xlabel": "tests", "place": SIDE_RIGHT, "series": []}]}
    assert figure_facts.layout_sentence(record) == (
        'two panels side by side in one row (left: "Fraction of real effects = 0"; '
        'right: y axis "Realised rate", x axis "tests")')
