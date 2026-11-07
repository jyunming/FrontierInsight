"""A series that draws a quantity the plan derives is named by the plan's label when it is the number FI worked out for it
(core/frame_figure.py ``_derived_comparison``, fed by core/analysis_figures.py ``run_figures``); one that is not stays a computed
value. Made-up study: a colony whose ``growth`` is read at settings of ``size`` for each ``temp``, and a ``ratio`` derived from it."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import analysis_figures, engine, frame_figure  # noqa: E402


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


def _ratio(shift=0.0, names=("ratio",)):
    return {n: {"size=%d,temp=%d" % (s, t): 0.1 * s + t / 100.0 + shift + k for t in (10, 20) for s in (1, 2, 3)}
            for k, n in enumerate(names)}


class _Host(frame_figure.FigureCalls):
    def __init__(self, derived=None, ids=("ratio",)):
        self.fig, axes = plt.subplots(1, 1, squeeze=False)
        self.axes = list(axes[0])
        self.table = _Grid()
        self.x = "size"
        self.series = "temp"
        self.fallback = False
        self.settings = ["size", "temp"]
        self.series_ids = []
        self.derived_ids = list(ids)
        self.derived_values = derived if derived is not None else _ratio()
        self.labels = {"growth": "Growth rate", "size": "Size", "temp": "Temperature (K)", "ratio": "Growth ratio",
                       "spread": "Growth spread"}


def _caption(h):
    return engine._figure_caption_text("ratio.png", None, n_seeds=None, calls=h.record())


def _line(h, name, shift=0.0, level_of=None):
    for t in (10, 20):
        at = t if level_of is None else level_of[t]
        h.line(0, t, [1, 2, 3], [0.1 * s + at / 100.0 + shift for s in (1, 2, 3)], name)


def test_a_series_that_is_the_number_worked_out_for_a_derived_quantity_is_named_by_its_label():
    h = _Host()
    _line(h, "ratio")
    h.finish(h.labels, {})
    rec = h.record()["panels"][0]
    assert rec["y"]["label"] == "Growth ratio" == h.axes[0].get_ylabel()
    s = rec["series"][0]
    assert s["what"] == "other" and "declared_as" not in s and "neutral" not in s  # no value of the table is claimed for it
    assert s["named_as"]["id"] == "ratio" and s["named_as"]["checked"] == "matches"
    assert s["level"]["result_label"] == "Growth ratio"
    assert "2 curves of Growth ratio, one per Temperature (K)" in _caption(h)


def test_a_series_that_is_not_the_number_worked_out_stays_a_computed_value():
    h = _Host()
    _line(h, "ratio", shift=0.5)  # known positive: the check still fires
    h.finish(h.labels, {})
    rec = h.record()["panels"][0]
    assert rec["y"]["label"] == frame_figure.NEUTRAL_Y
    s = rec["series"][0]
    assert s["declared_as"] == {"id": "ratio", "label": "Growth ratio", "checked": "differs"} and "named_as" not in s
    assert "Growth ratio" not in _caption(h)


def test_a_curve_is_compared_at_its_own_level_of_the_setting():
    h = _Host()
    _line(h, "ratio", level_of={10: 20, 20: 10})  # each curve is the other level's numbers
    h.finish(h.labels, {})
    assert all(s["declared_as"]["checked"] == "differs" for s in h.record()["panels"][0]["series"])


def test_without_a_record_of_the_derived_values_the_series_could_not_be_compared():
    h = _Host(derived={})
    _line(h, "ratio")
    h.finish(h.labels, {})
    assert h.record()["panels"][0]["series"][0]["declared_as"]["checked"] == "not compared"
    assert h.axes[0].get_ylabel() == frame_figure.NEUTRAL_Y


def test_two_derived_quantities_on_one_panel_are_each_named():
    h = _Host(derived=_ratio(names=("ratio", "spread")), ids=("ratio", "spread"))
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [0.1 * s + t / 100.0 for s in (1, 2, 3)], "ratio")
        h.line(0, t, [1, 2, 3], [0.1 * s + t / 100.0 + 1 for s in (1, 2, 3)], "spread")
    h.finish(h.labels, {})
    rec = h.record()["panels"][0]
    assert rec["y"]["label"] == "Growth ratio / Growth spread"
    assert len({e["label"] for e in rec["series"]}) == 4
    assert {e["level"]["result_label"] for e in rec["series"]} == {"Growth ratio", "Growth spread"}


def test_a_table_result_is_checked_against_the_table_as_before():
    h = _Host()
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0], "growth")
    h.finish(h.labels, {})
    rec = h.record()["panels"][0]
    assert rec["y"]["what"] == "growth" and "names" not in rec["y"] and rec["y"]["label"] == "Growth rate"


def test_the_record_of_the_derived_values_given_to_the_figures_holds_the_numbers_worked_out_only():
    got = analysis_figures._derived_values({
        "a": {"status": "ok", "by_setting": {"x=1": 0.5, "x=2": None, "x=3": True}},
        "b": {"status": "failed", "by_setting": {"x=1": 1.0}},
        "c": {"status": "ok", "value": 3.0},
        "d": "text"})
    assert got == {"a": {"x=1": 0.5}}
    assert analysis_figures._derived_values(None) == {}


def test_run_figures_hands_the_derived_values_to_the_figure(monkeypatch):
    seen = []
    monkeypatch.setattr(analysis_figures, "_draw_one", lambda spec, body, table, out_dir, plan=None: seen.append(plan) or {})
    analysis_figures.run_figures([{"id": "f"}], {}, _Grid(), "out", {"r": {"status": "ok", "by_setting": {"x=1": 2.0}}})
    assert seen[0]["derived_values"] == {"r": {"x=1": 2.0}}
