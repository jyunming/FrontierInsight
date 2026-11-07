"""A figure's caption and axis come from the same fact, the quantity FI's check found the series draws
(core/frame_figure.py ``_add_series``, the caption in core/engine.py ``_call_levels``, the quality line in
core/quality_report.py). Made-up study: a colony whose ``growth`` is read at settings of ``size`` for each ``temp``."""

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


def test_a_series_that_is_not_the_declared_result_is_described_as_a_computed_value():
    h = _draw(False)  # these are not the table's value of the result: another quantity
    text = _caption(h)
    assert "Growth rate" not in text and "computed value" in text
    assert h.axes[0].get_ylabel() == frame_figure.NEUTRAL_Y
    s = h.record()["panels"][0]["series"][0]
    assert s["declared_as"] == {"id": "growth", "label": "Growth rate"}
    assert s["level"]["declared_as"] == "growth" and s["level"]["result_label"] == frame_figure.NEUTRAL_Y


def test_a_series_that_is_the_declared_result_keeps_its_label_in_caption_and_axis():
    h = _draw(True)
    assert "2 curves of Growth rate, one per Temperature (K)" in _caption(h)
    assert h.record()["panels"][0]["y"]["what"] == "growth"
    s = h.record()["panels"][0]["series"][0]
    assert "declared_as" not in s and s["level"]["result"] == "growth"


def _quest(tmp_path, matching):
    root = tmp_path / "q"
    (root / ".fi" / "figure_records").mkdir(parents=True)
    (root / "paper").mkdir()
    h = _draw(matching)
    (root / ".fi" / "figure_records" / "g2.calls.json").write_text(json.dumps(h.record()), encoding="utf-8")
    (root / "paper" / "paper.md").write_text("![First.](figures/g1.png)\n\n![Second.](figures/g2.png)\n", encoding="utf-8")
    return root


def test_the_quality_report_lists_a_series_drawn_as_a_computed_value(tmp_path):
    found = quality_report._paper_problems(quality_report._Quest(_quest(tmp_path, False)))
    lines = [p["text"] for p in found if "declared as" in p["text"]]
    assert lines == ["Figure 2: a series declared as Growth rate does not match FI's record of that result; "
                     "it is drawn as a computed value."]


def test_the_quality_report_is_silent_when_the_series_matches(tmp_path):
    found = quality_report._paper_problems(quality_report._Quest(_quest(tmp_path, True)))
    assert not [p for p in found if "declared as" in p["text"]]


def test_two_declared_results_that_do_not_match_stay_two_numbered_groups_on_one_panel():
    h = _Host()
    h.table.declared = ["growth", "spread"]
    h.labels = {**h.labels, "spread": "Spread"}
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [5e-10, 6e-10, 7e-10], "growth")
        h.line(0, t, [1, 2, 3], [8e-10, 9e-10, 9.5e-10], "spread")
    h.finish({}, {})
    text = _caption(h)
    assert "Growth rate" not in text and "Spread" not in text
    assert "2 curves of computed value 1, one per Temperature (K)" in text
    assert "2 curves of computed value 2, one per Temperature (K)" in text
    assert len({e["label"] for e in h.record()["panels"][0]["series"]}) == 4  # no two curves share a legend name
