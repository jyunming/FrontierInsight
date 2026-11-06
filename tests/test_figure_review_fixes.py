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


def test_a_setting_value_drawn_in_two_panels_is_one_legend_entry_and_one_colour():
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

    h = _Two()
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [1.0, 2.0, 3.0 + t], "growth")
        h.line(1, t, [1, 2, 3], [0.5, 0.7, 0.9 + t], "spread")
    h.finish(h.labels, {})
    rec = h.record()
    texts = list(rec["legend"]["entries"])
    assert len(texts) == len(set(texts)) == 2, texts
    colour = {}
    for p in rec["panels"]:
        for s in p["series"]:
            colour.setdefault(s["level"]["text"], set()).add(str(s.get("color")))
    assert all(len(c) == 1 for c in colour.values()), colour
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert check_figure_text(rec, texts, caption) == [], (texts, caption)
