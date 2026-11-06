"""Faults a review found in how FI names and checks the lines of a figure (neutral made-up study; no model is called):
a line left out that empties a panel keeps the figure and says so; a reference that changes along x is never drawn as
a line across x; a panel that draws only reference lines has them in its caption; the legend/caption check does not
take "at 30" for "at 3", does not count two merged series as two entries, and reads the caption's own "Marked:" list."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from core import frame_figure  # noqa: E402
from core.engine import _caption_from_calls  # noqa: E402
from core.figure_text_check import check_figure_text  # noqa: E402
from tests.test_analysis_figures import _calls_run, _drawn, frame5  # noqa: E402,F401
from tests.test_frame_figure import _Host  # noqa: E402
from tests.test_registered_curves import _host as _law_host  # noqa: E402


def test_a_line_left_out_that_empties_a_panel_keeps_the_figure(tmp_path, frame5, monkeypatch):  # noqa: F811
    monkeypatch.setenv(frame_figure.LAST_RUN_ENV, "1")

    def body(fig, table):
        _drawn(fig, table, 0)
        fig.ref(1, value=7.77)  # matches no reference of the plan: the only thing panel 2 would draw

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok", out
    assert out.get("removed") and out["removed"][0]["panel"] == 2
    record = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))
    assert "Panel 2 draws nothing." in _caption_from_calls("g.png", record, None, n_seeds=None, facts=None)


def test_a_figure_whose_every_panel_is_emptied_still_fails_and_keeps_what_was_left_out(tmp_path, frame5, monkeypatch):  # noqa: F811
    monkeypatch.setenv(frame_figure.LAST_RUN_ENV, "1")
    out = _calls_run(tmp_path, lambda fig, table: (fig.ref(0, value=7.77), fig.ref(1, value=7.77)))
    assert out["status"] == "failed", out
    assert len(out.get("removed") or []) == 2


def test_a_reference_that_changes_along_x_is_refused_on_the_x_axis():
    h = _law_host()
    h._panel(0)[1]["hold"] = {"step": 2}
    h.line(0, "fast", [1, 2, 3], [2.0, 4.0, 6.0], "growth")
    with pytest.raises(ValueError) as e:
        h.ref(0, "law", axis="x")
    assert "axis" in str(e.value)
    assert all(s["id"] != "law" for s in h.record()["panels"][0]["series"])


def test_a_panel_drawing_only_reference_lines_has_them_in_its_caption():
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.ref(1, "limit")
    h.finish(h.labels, {})
    caption = _caption_from_calls("f.png", h.record(), None, n_seeds=None, facts=None)
    assert "upper limit at 3" in caption, caption
    assert "Panel 2 draws nothing" not in caption


def test_a_caption_value_is_matched_whole_not_as_a_prefix():
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.ref(0, "limit")
    h.finish(h.labels, {})
    rec = h.record()
    texts = list(rec["legend"]["entries"])
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert check_figure_text(rec, texts, caption) == []
    wrong = caption.replace("at 3", "at 30")
    assert any(f["kind"] == "ref_value" for f in check_figure_text(rec, texts, wrong)), wrong


def test_two_merged_series_are_one_legend_entry_and_not_flagged():
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.line(0, "slow", [1, 2], [1.0, 2.0], "growth")  # the same points: FI merges the two into one entry
    h.finish(h.labels, {})
    rec = h.record()
    assert any(s["merged_with"] for s in rec["panels"][0]["series"]), rec["panels"][0]["series"]
    texts = list(rec["legend"]["entries"]) if rec.get("legend") else None
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert check_figure_text(rec, texts, caption) == [], (texts, caption)


def test_marked_inside_a_quoted_label_is_not_the_list():
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.ref(0, "limit")
    h.finish(h.labels, {})
    rec = h.record()
    texts = list(rec["legend"]["entries"])
    caption = "Panel \"Marked: early\" of growth. " + _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert check_figure_text(rec, texts, caption) == [], caption


def _two_panel_levels():
    import matplotlib.pyplot as plt

    from tests.test_figure_series_levels import _Grid, _Host as _LevelHost

    class _Two(_LevelHost):
        def __init__(self):
            super().__init__()
            self.fig, axes = plt.subplots(1, 2, squeeze=False)
            self.axes = list(axes[0])
            self.table = _Grid((10, 20))
            self.table.declared = ["growth", "spread"]
            self.labels = {**self.labels, "spread": "Spread"}

    return _Two()


def test_a_setting_value_drawn_alike_in_two_panels_is_one_legend_entry():
    h = _two_panel_levels()
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [1.0, 2.0, 3.0 + t], "growth")
        h.line(1, t, [1, 2, 3], [0.5, 0.7, 0.9 + t], "spread")
    h.finish(h.labels, {})
    rec = h.record()
    texts = list(rec["legend"]["entries"])
    assert len(texts) == len(set(texts)) == 2, texts
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert check_figure_text(rec, texts, caption) == [], (texts, caption)


def test_two_lines_that_share_a_name_but_look_different_keep_two_entries():
    h = _Host(identity=True)
    a = h.axes[0].plot([1, 2], [1, 2], color="red", marker="o")[0]
    b = h.axes[1].plot([1, 2], [2, 3], color="blue", marker="s")[0]
    keyed = {0: [(("fast",), a, "Fast colony")], 1: [(("slow",), b, "Fast colony")]}
    h._figure_legend(h.fig, [0, 1], keyed)
    texts = [t.get_text() for leg in h.fig.legends for t in leg.get_texts()]
    assert texts == ["Fast colony", "Fast colony"], texts


def test_two_lines_that_share_a_name_and_look_alike_are_one_entry():
    h = _Host(identity=True)
    a = h.axes[0].plot([1, 2], [1, 2], color="red", marker="o")[0]
    b = h.axes[1].plot([1, 2], [2, 3], color="red", marker="o")[0]
    keyed = {0: [(("fast",), a, "Fast colony")], 1: [(("slow",), b, "Fast colony")]}
    h._figure_legend(h.fig, [0, 1], keyed)
    texts = [t.get_text() for leg in h.fig.legends for t in leg.get_texts()]
    assert texts == ["Fast colony"], texts
    assert h._fc_legend["covers"] == [2], h._fc_legend


def test_two_results_at_one_value_in_one_panel_keep_two_colours_and_two_entries():
    h = _two_panel_levels()
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [1.0, 2.0, 3.0 + t], "growth")
        h.line(0, t, [1, 2, 3], [0.5, 0.7, 0.9 + t], "spread")
        h.line(1, t, [1, 2, 3], [0.4, 0.6, 0.8 + t], "spread")
    h.finish(h.labels, {})
    rec = h.record()
    first = h._fc_extra[0]["series"]
    looks = {h._look(a[0]) for a in first if a[0] is not None}
    assert len(looks) == 4, looks
    texts = list(rec["legend"]["entries"])
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert not [f for f in check_figure_text(rec, texts, caption) if f["kind"] == "legend_missing"], (texts, caption)


def test_a_plan_formula_of_a_result_is_not_named_as_the_result_itself():
    h = _law_host()
    h.table.declared = list(getattr(h.table, "declared", []) or []) + ["law"]  # the formula is the plan's for a result
    h._panel(0)[1]["hold"] = {"step": 2}
    h.line(0, "fast", [1, 2, 3], [2.0, 4.0, 6.0], "law")
    h.ref(0, "law")
    s = h.record()["panels"][0]["series"][-1]
    assert s["kind"] == "curve" and s["label"] == "Expected Scaling law (plan formula)", s
    h2 = _law_host()
    h2._panel(0)[1]["hold"] = {"step": 2}
    h2.line(0, "fast", [1, 2, 3], [2.0, 4.0, 6.0], "growth")
    h2.ref(0, "law")  # a reference that is not a result keeps the plan's own name
    assert h2.record()["panels"][0]["series"][-1]["label"] == "Scaling law"


def test_the_legend_names_a_plan_formula_line_apart_from_its_result():
    h = _law_host()
    h.table.declared = list(getattr(h.table, "declared", []) or []) + ["law"]
    h._panel(0)[1]["hold"] = {"step": 2}
    h.line(0, "fast", [1, 2, 3], [2.0, 4.0, 6.0], "law")
    h.references = {}
    h.table._references = {"law": {k: 3.0 for k in h.table._references["law"]}}  # one number at every setting: a flat line
    h.ref(0, "law")
    h.finish({**h.labels, "law": {"label": "Scaling law"}}, {})
    texts = list(h.record()["legend"]["entries"])
    assert "Expected Scaling law (plan formula)" in texts and len(set(texts)) == len(texts), texts


def test_a_plan_reference_whose_given_label_is_its_result_name_is_named_as_expected():
    h = _law_host()
    h.table.declared = list(getattr(h.table, "declared", []) or []) + ["law"]
    h.labels = {**h.labels, "law": "Scaling law"}
    h.references = {"law": {"value": 3.0, "label": "Scaling law"}}
    h.line(0, "fast", [1, 2, 3], [2.0, 4.0, 6.0], "law")
    h.ref(0, "law")
    assert h.record()["panels"][0]["refs"][-1]["label"] == "Expected Scaling law (plan formula)"


def test_one_table_line_per_panel_under_one_id_is_counted_once_per_name():
    calls = {"legend": {"scope": "figure", "entries": ["A", "Mean of A at 1", "Mean of A at 2"]},
             "panels": [{"panel": 1, "series": [{"id": "a", "kind": "line"}], "refs": [{"id": "refline1", "label": "Mean of A at 1", "value": 1.0}]},
                        {"panel": 2, "series": [{"id": "a", "kind": "line"}], "refs": [{"id": "refline1", "label": "Mean of A at 2", "value": 2.0}]}]}
    caption = "A against x. Marked: Mean of A at 1 at 1; Mean of A at 2 at 2."
    assert not [f for f in check_figure_text(calls, calls["legend"]["entries"], caption) if f["kind"].startswith("legend")]
