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


def _geo(legend=None, bars=None, limits=None, places=(SIDE, SIDE_RIGHT)):
    rec = _record(*places, titles=("left", "right"))
    rec["axes"][0]["legend_box"] = legend
    rec["axes"][0]["errorbars"] = bars if bars is not None else []
    rec["axes"][0]["texts"] = []
    rec["axes"][0]["data_limits"] = limits or {"x": [0, 1], "y": [0, 10]}
    return rec


def test_legend_off_canvas_is_a_finding():
    got = figure_facts.geometry_findings(_geo(legend={"x0": 0.2, "y0": 0.2, "x1": 1.3, "y1": 0.4}))
    assert any("legend" in s and "outside the figure" in s for s in got)


def test_legend_over_other_panel_is_a_finding():
    got = figure_facts.geometry_findings(_geo(legend={"x0": 0.3, "y0": 0.2, "x1": 0.7, "y1": 0.4}))
    assert any("legend" in s and '"right"' in s for s in got)


def test_errorbar_outside_limits_is_a_finding():
    got = figure_facts.geometry_findings(_geo(bars=[{"line": "fit", "lo": [1.0, -2.0], "hi": [3.0, 4.0]}]))
    assert len(got) == 1 and "error bars" in got[0] and '"fit"' in got[0]


def test_clean_record_has_no_findings():
    rec = _geo(legend={"x0": 0.15, "y0": 0.2, "x1": 0.4, "y1": 0.4}, bars=[{"line": None, "lo": [1.0], "hi": [9.0]}])
    assert figure_facts.geometry_findings(rec) == []


def test_old_record_has_no_findings():
    assert figure_facts.geometry_findings(_record(SIDE, SIDE_RIGHT)) == []
    assert figure_facts.geometry_findings(None) == []


def test_errorbar_cut_off_on_a_small_scale_is_a_finding():
    rec = _geo(bars=[{"line": "fit", "lo": [1e-5], "hi": [2.5e-4]}], limits={"x": [0, 1], "y": [0, 2e-4]})
    assert any("error bars" in s for s in figure_facts.geometry_findings(rec))


def test_legend_overlapping_only_in_y_is_not_a_finding():
    # same height as the right panel but entirely left of it
    assert figure_facts.geometry_findings(_geo(legend={"x0": 0.12, "y0": 0.3, "x1": 0.3, "y1": 0.5})) == []


def test_legend_over_twin_or_inset_is_not_a_finding():
    legend = {"x0": 0.3, "y0": 0.3, "x1": 0.4, "y1": 0.4}
    twin = _geo(legend=legend, places=(SIDE, SIDE))
    inset = _geo(legend=legend, places=(SIDE, [0.2, 0.3, 0.15, 0.2]))
    assert figure_facts.geometry_findings(twin) == []
    assert figure_facts.geometry_findings(inset) == []


# --- marks with arguments: FI re-evaluates a claim against its call record, or leaves the sentence out ----------------


def _calls(raw: bool = False, panels: int = 1) -> dict:
    def panel(k: int) -> dict:
        return {"panel": k, "raw": raw if k == 1 else False, "raw_reason": "drawn by hand" if raw and k == 1 else None,
                "x": {"id": "size", "kind": "setting", "label": "size", "scale": "linear"},
                "y": {"what": "signal", "label": "signal", "scale": "linear"},
                "series": [{"id": "a", "label": "a", "kind": "line", "step": False, "x": [1, 2, 3], "y": [1.0, 2.0, 3.0],
                            "yerr": None, "merged_with": [], "x_checked": True}],
                "bands": [], "refs": [], "notes": [], "texts_raw": []}
    return {"version": 2, "fallback": False, "panels": [panel(k) for k in range(1, panels + 1)]}


def _with_calls(tmp_path: Path, calls: dict | None, stem: str = "fig1") -> Path:
    folder = tmp_path / ".fi" / "figure_records"
    folder.mkdir(parents=True, exist_ok=True)
    if calls is not None:
        (folder / f"{stem}.calls.json").write_text(json.dumps(calls), encoding="utf-8")
    return tmp_path


def test_figure_mark_with_args_fills_from_record(tmp_path):
    root = _with_calls(tmp_path, _calls())
    mark = "{{figure:fig1:trend @ series=a, panel=1}}"
    out = paper_facts.render(f"Results. {mark}", root)
    assert out.unfilled == ()
    assert "<!-- fi:fact-figure-trend-fig1-args" in out.text and "not recorded" not in out.text
    sentence = paper_facts.without_marks(out.text).removeprefix("Results. ")
    assert "increases" in sentence
    assert paper_facts.to_placeholders(out.text) == f"Results. {mark}"
    # a re-render of the filled text gives the same text
    assert paper_facts.render(out.text, root).text == out.text


def test_old_layout_mark_still_fills(tmp_path):
    root = _quest(tmp_path, _record(SIDE, SIDE_RIGHT, titles=("Fit", "Residual")), _lines("-"))
    out = paper_facts.render("Has {{figure:fig_a:layout}}.", root)
    assert out.unfilled == () and "<!-- fi:fact-figure-layout-fig_a -->" in out.text
    assert paper_facts.to_placeholders(out.text) == "Has {{figure:fig_a:layout}}."


def test_made_up_claim_renders_not_recorded_and_sentence_is_taken_out(tmp_path):
    root = _with_calls(tmp_path, _calls())
    out = paper_facts.render("Intro.\n\nThe curve {{figure:fig1:sparkle @ series=a}} here.\n\n{{figure:fig1:sparkle @ series=a}}\n", root)
    assert out.unfilled == ("figure:fig1:sparkle",) and "not recorded" in out.text
    text, taken = paper_facts.take_out_unfilled(out.text)
    assert "not recorded" not in text and "sparkle" not in text and "fi:fact-figure" not in text
    assert text.startswith("Intro.") and all(t["how"] in ("sentence removed", "removed") for t in taken)
    assert all(t["key"].startswith("figure:fig1:sparkle") and {"sentence", "how"} <= set(t) for t in taken)


def test_claim_on_raw_panel_is_taken_out(tmp_path):
    root = _with_calls(tmp_path, _calls(raw=True))
    out = paper_facts.render("Before. {{figure:fig1:trend @ series=a, panel=1}}\n\nNext.\n", root)
    assert out.unfilled == ("figure:fig1:trend",)
    text, taken = paper_facts.take_out_unfilled(out.text)
    assert text == "Before.\n\nNext.\n" and len(taken) == 1 and taken[0]["how"] == "sentence removed"


def test_mark_with_missing_call_record_is_taken_out(tmp_path):
    out = paper_facts.render("Before. {{figure:fig1:drawn @ series=a}}\n", tmp_path)
    assert out.unfilled == ("figure:fig1:drawn",)
    text, taken = paper_facts.take_out_unfilled(out.text)
    assert text.strip() == "Before." and len(taken) == 1


def test_malformed_args_fail_closed_without_raising(tmp_path):
    root = _with_calls(tmp_path, _calls())
    for args in ("series", "series=a, series=a", "=a", "panel=x, series=a", "series=zzz", "series=a, bogus=1", ""):
        out = paper_facts.render(f"A. {{{{figure:fig1:trend @ {args}}}}}\n", root)
        assert out.unfilled == ("figure:fig1:trend",), args
        text, taken = paper_facts.take_out_unfilled(out.text)
        assert "not recorded" not in text and len(taken) == 1


def test_one_disclosure_per_figure_not_per_sentence(tmp_path):
    root = _with_calls(tmp_path, _calls())
    paper = ("![Cap](figures/other.png)\n\n![Cap two](figures/fig1.png)\n\n"
             "A. {{figure:fig1:sparkle @ series=a}} B. {{figure:fig1:trend @ series=zzz}}\n\n"
             "{{figure:fig1:range @ series=a, axis=y}} C.\n")
    out = paper_facts.render(paper, root)
    text, taken = paper_facts.take_out_unfilled(out.text)
    assert len(taken) == 2 and {t["figure"] for t in taken} == {"2"}
    paper_facts.keep_check(root, [], text, dropped=taken)
    kept = paper_facts.kept_check(root)
    unmet = [u for u in kept["unmet_items"] if u["kind"] == paper_facts.FIGURE_LEFT_OUT]
    assert [u["text"] for u in unmet] == ["Figure 2: some statements about this figure could not be checked against "
                                          "what it draws and were left out."]
    items = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.FIGURE_LEFT_OUT]
    assert len(items) == 1 and items[0].sentence == unmet[0]["text"]


def test_taken_out_figure_sentence_keeps_bold_run_in_heading(tmp_path):
    root = _with_calls(tmp_path, _calls())
    out = paper_facts.render("**Growth.** The signal rises. {{figure:fig1:trend @ series=zzz}} Later text.\n", root)
    text, taken = paper_facts.take_out_unfilled(out.text)
    assert text == "**Growth.** The signal rises. Later text.\n" and taken[0]["how"] == "sentence removed"


def test_mid_sentence_figure_mark_takes_the_whole_running_sentence_out(tmp_path):
    root = _with_calls(tmp_path, _calls())
    out = paper_facts.render("Intro. The curve {{figure:fig1:sparkle @ series=a}} here. Next one.\n", root)
    text, taken = paper_facts.take_out_unfilled(out.text)
    assert text == "Intro. Next one.\n" and len(taken) == 1 and taken[0]["how"] == "sentence removed"


def test_stem_with_an_image_extension_is_taken_out_not_left_as_not_recorded(tmp_path):
    root = _with_calls(tmp_path, _calls())
    out = paper_facts.render("Before. {{figure:fig1.png:sparkle @ series=a}}\n\nSee {{figure:bad stem!:trend @ series=a}} now.\n", root)
    assert "fi:fact-figure_" not in out.text
    text, taken = paper_facts.take_out_unfilled(out.text)
    assert "not recorded" not in text and len(taken) == 2
    assert taken[0]["key"].startswith("figure:fig1:sparkle")
    # with the extension dropped the claim is found
    ok = paper_facts.render("{{figure:fig1.png:trend @ series=a}}", root)
    assert ok.unfilled == () and "increases" in paper_facts.without_marks(ok.text)


def test_empty_args_keep_the_stem_and_the_figure_number(tmp_path):
    root = _with_calls(tmp_path, _calls())
    out = paper_facts.render("![c](figures/fig1.png)\n\nA. {{figure:fig1:trend @ }}\n", root)
    text, taken = paper_facts.take_out_unfilled(out.text)
    assert text == "![c](figures/fig1.png)\n\nA.\n"
    assert taken[0]["key"] == "figure:fig1:trend" and taken[0]["figure"] == "1"
    assert paper_facts.to_placeholders(out.text).endswith("{{figure:fig1:trend @ }}\n")


def test_plan_reaches_the_claim(tmp_path, monkeypatch):
    seen = {}
    def fake(calls, claim, args, fmt, *, plan=None, **kw):
        seen["plan"] = plan
        return "Series b is not drawn.", args
    monkeypatch.setattr("core.figure_claims.evaluate", fake)
    root = _with_calls(tmp_path, _calls())
    out = paper_facts.render("{{figure:fig1:not_drawn @ series=b}}", root, plan={"series": ["a"]})
    assert seen["plan"] == {"series": ["a"]} and out.unfilled == ()


def test_limitations_carrying_the_figure_sentence_is_not_listed_unmet(tmp_path):
    root = _with_calls(tmp_path, _calls())
    out = paper_facts.render("![c](figures/fig1.png)\n\nA. {{figure:fig1:sparkle @ series=a}}\n", root)
    text, taken = paper_facts.take_out_unfilled(out.text)
    said = paper_facts.figure_left_out_sentence("1")
    paper_facts.keep_check(root, [], text + "\n" + said, dropped=taken)
    assert not [u for u in paper_facts.kept_check(root).get("unmet_items", []) if u["kind"] == paper_facts.FIGURE_LEFT_OUT]
