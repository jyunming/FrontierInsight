"""Two figure faults FI's checks let through, on a made-up study (a colony whose ``rate`` is read at ``load`` and ``size``):

1. a reference the plan gives for a result a figure draws, and the figure does not draw it
   (core/figure_text_check.py ``missing_references``, said in the paper's Limitations by core/paper_facts.py and in the
   quality report);
2. a sentence FI composes about one curve of a figure that draws one curve per level of a setting, which read as though that
   level were the only one drawn (core/figure_claims.py ``drawn``).
"""

from __future__ import annotations

import json
from pathlib import Path

from core import figure_claims as fc
from core import figure_text_check as ftc
from core import paper_facts, quality_report

PROTOCOL = {"metrics": [
    {"id": "rate", "label": "Growth rate",
     "reference": {"value": "rate_theory", "formula": "(1-load)*cap", "kind": "bound"}},
    {"id": "spread", "label": "Spread"},
    {"id": "level", "label": "Level", "reference": {"value": 0.5, "kind": "value"}},
    {"id": "share", "label": "Share", "reference": {"value": "share", "formula": "nominal_share", "kind": "value"}},
]}


def _series(sid, what, **kw):
    return {"id": sid, "label": sid, "kind": "line", "step": False, "x": [1, 2, 3], "y": [1.0, 2.0, 3.0], "what": what,
            "merged_with": [], "x_checked": True, **kw}


def _panel(series, refs=(), raw=False, n=1):
    return {"panel": n, "raw": raw, "raw_reason": None, "x": {"id": "load", "kind": "setting", "label": "Load", "scale": "linear"},
            "y": {"what": "rate", "label": "Growth rate", "scale": "linear"}, "series": list(series), "bands": [],
            "refs": list(refs), "notes": [], "texts_raw": []}


def _calls(*panels):
    return {"version": 2, "fallback": False, "panels": list(panels)}


def _line(ident, value, source="plan"):
    return {"id": ident, "axis": "y", "value": value, "label": ident, "source": source}


# ---- 1. a reference the plan gives and the figure does not draw ---------------------------------------------------------------

def test_a_plan_reference_the_figure_does_not_draw_is_found():
    got = ftc.missing_references(_calls(_panel([_series("rate@size:1", "rate")])), PROTOCOL)
    assert [(f["result"], f["formula"]) for f in got] == [("rate", "(1-load)*cap")]


def test_a_reference_drawn_from_the_plan_by_its_id_is_not_missing():
    assert ftc.missing_references(_calls(_panel([_series("rate@size:1", "rate")], [_line("rate", 0.4)])), PROTOCOL) == []
    assert ftc.missing_references(_calls(_panel([_series("rate@size:1", "rate")], [_line("rate_theory", 0.4)])), PROTOCOL) == []


def test_a_reference_curve_from_the_plan_or_one_that_carries_the_references_name_is_not_missing():
    curve = _series("rate", "other", kind="curve", source="plan")
    assert ftc.missing_references(_calls(_panel([_series("rate@size:1", "rate"), curve])), PROTOCOL) == []
    named = _series("rate_theory", "other")
    assert ftc.missing_references(_calls(_panel([_series("rate", "rate"), named])), PROTOCOL) == []


def test_the_results_own_curve_is_not_its_reference():
    # the measured series carries the result's id; only a line from the plan, or the reference's own name, is the reference
    assert ftc.missing_references(_calls(_panel([_series("rate", "rate")])), PROTOCOL)
    assert ftc.missing_references(_calls(_panel([_series("rate", "rate")], [_line("rate", 0.4, source="table")])), PROTOCOL)


def test_a_line_that_is_not_the_plans_does_not_count_as_its_reference():
    typed = _line("line-1", 0.4, source="stated in the figure")
    assert ftc.missing_references(_calls(_panel([_series("rate", "rate")], [typed])), PROTOCOL)


def test_a_reference_given_as_a_number_is_drawn_by_a_line_at_that_number():
    only = _calls(_panel([_series("level", "level")]))
    assert [f["result"] for f in ftc.missing_references(only, PROTOCOL)] == ["level"]
    drawn = _calls(_panel([_series("level", "level")], [_line("line-1", 0.5, source="stated in the figure")]))
    assert ftc.missing_references(drawn, PROTOCOL) == []


def test_a_level_the_plan_names_is_drawn_by_a_line_under_that_name():
    calls = _calls(_panel([_series("share", "share")], [_line("nominal_share", 0.95)]))
    assert ftc.missing_references(calls, PROTOCOL) == []
    assert [f["result"] for f in ftc.missing_references(_calls(_panel([_series("share", "share")])), PROTOCOL)] == ["share"]


def test_a_result_the_figure_does_not_draw_or_a_custom_drawing_has_nothing_missing():
    assert ftc.missing_references(_calls(_panel([_series("spread", "spread")])), PROTOCOL) == []
    assert ftc.missing_references(_calls(_panel([], raw=True)), PROTOCOL) == []
    assert ftc.missing_references(_calls(_panel([_series("rate", "rate")])), None) == []
    assert ftc.missing_references(_calls(_panel([_series("rate", "rate")])), {"metrics": [{"id": "rate"}]}) == []


def _quest(root: Path, calls: dict, paper: bool = True) -> Path:
    (root / ".fi" / "figure_records").mkdir(parents=True)
    (root / ".fi" / "figure_records" / "rate_vs_load.calls.json").write_text(json.dumps(calls), encoding="utf-8")
    if paper:
        (root / "paper").mkdir()
        (root / "paper" / "paper.md").write_text("![**Figure 1.** Growth rate.](figures/rate_vs_load.png)\n", encoding="utf-8")
    return root


def test_the_quest_check_names_the_figure_and_its_place_in_the_paper(tmp_path):
    root = _quest(tmp_path, _calls(_panel([_series("rate", "rate")])))
    got = ftc.quest_missing_references(root, PROTOCOL)
    assert [(f["file"], f["number"], f["result"]) for f in got] == [("rate_vs_load.png", 1, "rate")]
    assert got[0]["record"] == ".fi/figure_records/rate_vs_load.calls.json"


def test_before_the_paper_is_written_every_recorded_figure_that_is_still_there_is_read(tmp_path):
    root = _quest(tmp_path, _calls(_panel([_series("rate", "rate")])), paper=False)
    assert ftc.quest_missing_references(root, PROTOCOL) == []  # the figure's file was taken out: its old record says nothing
    (root / "figures").mkdir()
    (root / "figures" / "rate_vs_load.png").write_bytes(b"")
    got = ftc.quest_missing_references(root, PROTOCOL)
    assert [(f["file"], f["number"]) for f in got] == [("rate_vs_load.png", None)]


def test_the_paper_must_say_a_reference_the_plan_gives_is_not_drawn(tmp_path, monkeypatch):
    root = _quest(tmp_path, _calls(_panel([_series("rate", "rate")])))
    monkeypatch.setattr(paper_facts, "plan_protocol", lambda *a, **k: PROTOCOL)
    mine = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.FIGURE_REFERENCE]
    assert len(mine) == 1
    assert "Figure 1 draws no line for the plan's reference for Growth rate" in mine[0].sentence
    assert "(1-load)*cap" in mine[0].sentence
    drawn = _quest(tmp_path / "ok", _calls(_panel([_series("rate", "rate")], [_line("rate", 0.4)])))
    assert not [i for i in paper_facts.required_items(drawn) if i.kind == paper_facts.FIGURE_REFERENCE]


def _says(tmp_path, monkeypatch, calls):
    root = _quest(tmp_path, calls)
    monkeypatch.setattr(paper_facts, "plan_protocol", lambda *a, **k: PROTOCOL)
    mine = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.FIGURE_REFERENCE]
    rows = [p for p in quality_report._paper_problems(quality_report._Quest(root)) if "Growth rate" in p["text"]]
    assert len(mine) == 1 and len(rows) == 1
    return mine[0].sentence, rows[0]["text"]


def test_a_figure_with_no_other_line_says_it_draws_no_line_for_the_reference(tmp_path, monkeypatch):
    sentence, row = _says(tmp_path, monkeypatch, _calls(_panel([_series("rate", "rate")])))
    for text in (sentence, row):
        assert "draws no line for the plan's reference for Growth rate" in text
        assert "did not take from the plan" not in text


def test_a_figure_with_a_line_not_from_the_plan_says_fi_cannot_confirm_it(tmp_path, monkeypatch):
    typed = _line("line-1", 0.4, source="stated in the figure")
    computed = _series("curve1", "other", kind="curve", source="computed in the figure")
    for extra in (_panel([_series("rate", "rate")], [typed]), _panel([_series("rate", "rate"), computed])):
        sentence, row = _says(tmp_path / str(id(extra)), monkeypatch, _calls(extra))
        for text in (sentence, row):
            assert "draws a line that FI did not take from the plan, so FI cannot confirm it is the plan's reference for Growth rate" in text
            assert "draws no line" not in text


def test_the_quality_report_lists_the_reference_the_figure_does_not_draw(tmp_path, monkeypatch):
    root = _quest(tmp_path, _calls(_panel([_series("rate", "rate")])))
    monkeypatch.setattr(paper_facts, "plan_protocol", lambda *a, **k: PROTOCOL)
    rows = [p for p in quality_report._paper_problems(quality_report._Quest(root)) if "the plan's reference for" in p["text"]]
    assert len(rows) == 1 and rows[0]["text"].startswith("Figure 1 draws no line")
    assert rows[0]["status"] == "not fixed"


# ---- 2. a sentence about one curve of a figure that draws one curve per level -------------------------------------------------

def _level_series(value, kind="line"):
    return _series(f"rate@size:{value}", "rate", kind=kind, label=f"Size = {value}",
                   level={"setting": "size", "setting_label": "Size", "value": value, "text": str(value), "result": "rate",
                          "result_label": "Growth rate"})


def _evaluate(calls, series_id):
    return fc.evaluate(calls, "drawn", {"series": series_id})


def test_one_curve_of_five_levels_is_not_said_to_be_the_only_one_drawn():
    calls = _calls(_panel([_level_series(v) for v in (1, 2, 3, 4, 5)]))
    sentence = _evaluate(calls, "rate@size:1")[0]
    assert sentence.startswith("Size = 1 is drawn")
    assert "Size = 1 is drawn, one of 5 curves, one per Size" in sentence


def test_a_level_drawn_as_a_line_and_as_points_counts_once():
    series = [_level_series(1), _level_series(1, kind="points"), _level_series(2)]
    series[1]["id"] = "rate@size:1:points"
    assert "one of 2 curves, one per Size" in _evaluate(_calls(_panel(series)), "rate@size:1")[0]


def test_a_single_level_or_a_series_that_is_not_a_level_keeps_its_plain_sentence():
    assert _evaluate(_calls(_panel([_level_series(1)])), "rate@size:1")[0] == "Size = 1 is drawn"
    plain = _calls(_panel([_series("a", "rate"), _series("b", "rate")]))
    assert _evaluate(plain, "a")[0] == "a is drawn"
