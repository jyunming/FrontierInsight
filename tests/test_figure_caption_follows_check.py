"""FI's check of a drawn series against its record never renames it (core/frame_figure.py ``_add_series``): a series FI
could not compare keeps the plan's name in the caption and on the axis; one FI compared and found not to be its record is
left out of the figure, and the quality line says which (core/quality_report.py). Made-up study: a colony whose ``growth``
is read at settings of ``size`` for each ``temp``."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import engine, frame_figure, quality_report  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _Grid:
    declared = ["growth"]

    def keys(self):
        return ["size=%d,temp=%s" % (s, t) for t in (10, 20) for s in (1, 2, 3)]

    def mean(self, name, key):
        size, temp = (float(part.split("=")[1]) for part in key.split(","))
        return temp + size - 1


class _Host(frame_figure.FigureCalls):
    def __init__(self):
        self.fig, axes = plt.subplots(1, 1, squeeze=False)
        self.axes = list(axes[0])
        self.table = _Grid()
        self.x = "size"
        self.series = "temp"
        self.fallback = False
        self.settings = ["size", "temp"]
        self.series_ids = []
        self.labels = {"growth": "Growth rate", "size": "Size", "temp": "Temperature (K)"}


def _caption(h):
    return engine._figure_caption_text("growth.png", None, n_seeds=None, calls=h.record())


def _draw(matching):
    h = _Host()
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0] if matching else [5e-10, 6e-10, 7e-10], "growth")
    h.finish({}, {})
    return h


def test_a_series_that_is_not_the_declared_result_is_left_out_with_the_reason():
    h = _draw(False)  # these are not the table's value of the result: another quantity
    rec = h.record()["panels"][0]
    assert rec["series"] == [] and not h.axes[0].lines
    assert rec["left_out"] == [
        {"id": "growth@temp:%d" % t, "kind": "curve", "declared": "growth", "label": "Growth rate",
         "name": "Growth rate at Temperature (K) = %d" % t, "setting": str(t), "setting_label": "Temperature (K)",
         "reason": "the drawn values are not FI's record of Growth rate at Temperature (K) = %d" % t} for t in (10, 20)]


def test_a_series_that_is_the_declared_result_keeps_its_label_in_caption_and_axis():
    h = _draw(True)
    assert "2 curves of Growth rate, one per Temperature (K)" in _caption(h)
    assert h.record()["panels"][0]["y"]["what"] == "growth"
    s = h.record()["panels"][0]["series"][0]
    assert "declared_as" not in s and s["level"]["result"] == "growth" and "left_out" not in h.record()["panels"][0]


def _quest(tmp_path, matching, host=None):
    root = tmp_path / "q"
    (root / ".fi" / "figure_records").mkdir(parents=True)
    (root / "paper").mkdir()
    h = host or _draw(matching)
    (root / ".fi" / "figure_records" / "g2.calls.json").write_text(json.dumps(h.record()), encoding="utf-8")
    (root / "paper" / "paper.md").write_text("![First.](figures/g1.png)\n\n![Second.](figures/g2.png)\n", encoding="utf-8")
    return root


def test_the_quality_report_lists_a_series_left_out_of_a_figure(tmp_path):
    found = quality_report._paper_problems(quality_report._Quest(_quest(tmp_path, False)))
    rows = [p for p in found if "declared as" in p["text"]]
    assert [p["text"] for p in rows] == ["Figure 2: a series declared as Growth rate at Temperature (K) = %d does not match "
                                         "FI's record of that result; it is left out of the figure." % t for t in (10, 20)]
    assert {r["status"] for r in rows} == {"left out"}


def test_the_quality_report_is_silent_when_the_series_matches(tmp_path):
    found = quality_report._paper_problems(quality_report._Quest(_quest(tmp_path, True)))
    assert not [p for p in found if "declared as" in p["text"]]


def test_two_declared_results_that_do_not_match_are_both_left_out():
    h = _Host()
    h.table.declared = ["growth", "spread"]
    h.labels = {**h.labels, "spread": "Spread"}
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [5e-10, 6e-10, 7e-10], "growth")
        h.line(0, t, [1, 2, 3], [8e-10, 9e-10, 9.5e-10], "spread")
    rec = h.record()["panels"][0]
    assert rec["series"] == []
    assert [(e["declared"], e["setting"]) for e in rec["left_out"]] == [
        ("growth", "10"), ("spread", "10"), ("growth", "20"), ("spread", "20")]


def test_a_series_that_could_not_be_compared_keeps_its_plan_name_and_is_not_said_to_differ(tmp_path):
    h = _Host()
    h.derived_ids = ["ratio"]  # a quantity the study computes: FI holds no table value of it to compare with
    h.labels = {**h.labels, "ratio": "Growth ratio"}
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [0.5, 0.6, 0.7], "ratio")
    h.finish(h.labels, {})
    assert "2 curves of Growth ratio, one per Temperature (K)" in _caption(h)
    assert h.axes[0].get_ylabel() == "Growth ratio"
    s = h.record()["panels"][0]["series"][0]
    assert s["declared_as"]["checked"] == "not compared" and s["what"] == "other"  # no claim reads its values
    found = quality_report._paper_problems(quality_report._Quest(_quest(tmp_path, False, h)))
    rows = [p for p in found if "declared as" in p["text"]]
    assert [p["text"] for p in rows] == ["Figure 2: a series declared as Growth ratio could not be checked against FI's "
                                         "record of that result; it is drawn under that name."]
    assert rows[0]["status"] == "disclosed"


def test_a_series_that_was_compared_and_differs_is_said_to_differ(tmp_path):
    from core import figure_text_check

    got = figure_text_check.quest_neutral_declared(_quest(tmp_path, False))
    assert [(f["label"], f["kind"], f["checked"], f["drawn"]) for f in got] == [
        ("Growth rate at Temperature (K) = %d" % t, "curve", "differs", False) for t in (10, 20)]


def test_a_matched_line_and_unmatched_points_of_one_level_keep_only_the_line():
    h = _Host()
    h.line(0, 10, [1, 2, 3], [10.0, 11.0, 12.0], "growth")  # the table's values
    assert h.points(0, 10, [1, 2, 3], [5e-10, 6e-10, 7e-10], "growth") == []  # not the table's values
    h.finish({}, {})
    entries = list(h.record()["legend"]["entries"]) if h.record().get("legend") else [t.get_text() for t in h.axes[0].get_legend().get_texts()]
    assert entries == ["Temperature (K) = 10"], entries
    assert [e["kind"] for e in h.record()["panels"][0]["series"]] == ["line"]


def test_a_matched_result_keeps_its_label_when_two_other_results_are_left_out():
    h = _Host()
    h.table.declared = ["growth", "spread", "rate"]
    h.labels = {**h.labels, "spread": "Spread", "rate": "Rate"}
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0], "spread")  # the table's values
        h.line(0, t, [1, 2, 3], [5e-10, 6e-10, 7e-10], "growth")
        h.line(0, t, [1, 2, 3], [8e-10, 9e-10, 9.5e-10], "rate")
    h.finish({}, {})
    text = _caption(h)
    assert "2 curves of Spread, one per Temperature (K)" in text, text
    assert "Growth rate" not in text and "Rate" not in text, text
    assert {e["declared"] for e in h.record()["panels"][0]["left_out"]} == {"growth", "rate"}


class _Grid3:
    """Growth at each size for each feed level and temperature: ``temp + 10 * feed + size``."""

    declared = ["growth"]

    def keys(self):
        return ["size=%d,feed=%d,temp=%d" % (s, f, t) for t in (10, 20) for f in (1, 2) for s in (1, 2, 3)]

    def mean(self, name, key):
        size, feed, temp = (float(part.split("=")[1]) for part in key.split(","))
        return temp + 10 * feed + size


class _TwoPanelHost(_Host):
    """Two panels, one per temperature (``fig.panel``), each with one curve per feed level."""

    def __init__(self):
        self.fig, axes = plt.subplots(1, 2, squeeze=False)
        self.axes = list(axes[0])
        self.table = _Grid3()
        self.x = "size"
        self.series = "feed"
        self.fallback = False
        self.settings = ["size", "feed", "temp"]
        self.series_ids = []
        self.labels = {"growth": "Growth rate", "size": "Size", "feed": "Feed level", "temp": "Temperature (K)"}


def test_a_level_left_out_of_two_panels_is_listed_once_per_panel_setting(tmp_path):
    h = _TwoPanelHost()
    for panel, temp in ((0, 10), (1, 20)):
        h._panel(panel)[1]["hold"] = {"temp": temp}  # what `fig.panel(panel, temp=temp)` records
        h.line(panel, 1, [1, 2, 3], [temp + 11.0, temp + 12.0, temp + 13.0], "growth")  # FI's record: drawn
        h.line(panel, 2, [1, 2, 3], [5e-10, 6e-10, 7e-10], "growth")  # not FI's record at feed level 2: left out
    h.points(0, 2, [1, 2, 3], [5e-10, 6e-10, 7e-10], "growth")  # the same level left out again in the same panel
    h.finish(h.labels, {})
    root = tmp_path / "q"
    (root / ".fi" / "figure_records").mkdir(parents=True)
    (root / "paper").mkdir()
    (root / ".fi" / "figure_records" / "g.calls.json").write_text(json.dumps(h.record()), encoding="utf-8")
    (root / "paper" / "paper.md").write_text("![A figure.](figures/g.png)\n", encoding="utf-8")
    found = quality_report._paper_problems(quality_report._Quest(root))
    assert [p["text"] for p in found if "declared as" in p["text"]] == [
        "Figure 1: a series declared as Growth rate at Feed level = 2 in the panel at Temperature (K) = %d does not match "
        "FI's record of that result; it is left out of the figure." % t for t in (10, 20)]


def test_a_curve_left_out_of_one_panel_of_two_names_its_panel(tmp_path):
    # Two panels that hold no setting; only panel 2 loses a curve. The row names panel 2, as it would if both lost one.
    h = _Host()
    h.fig, axes = plt.subplots(1, 2, squeeze=False)
    h.axes = list(axes[0])
    h.line(0, 10, [1, 2, 3], [10.0, 11.0, 12.0], "growth")  # FI's record at temperature 10: drawn
    h.line(1, 20, [1, 2, 3], [5e-10, 6e-10, 7e-10], "growth")  # not FI's record at temperature 20: left out
    h.finish({}, {})
    found = quality_report._paper_problems(quality_report._Quest(_quest(tmp_path, False, h)))
    assert [p["text"] for p in found if "declared as" in p["text"]] == [
        "Figure 2: a series declared as Growth rate at Temperature (K) = 20 in panel 2 does not match FI's record of that "
        "result; it is left out of the figure."]
