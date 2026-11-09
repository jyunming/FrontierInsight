"""Figures FI draws or lays out from its own records: a default figure whose curves are levels of the series setting, bars that do
not hide each other, a legend that does not cover the data, and a nominal level of the plan drawn as a named line. Made-up study:
a colony whose ``growth`` is read at settings of ``size`` for each value of ``temp``."""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import analysis_figures as af  # noqa: E402
from core import frame_figure  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def frame5(tmp_path, monkeypatch):
    """The frame of the drawing calls beside the figure code, the way a quest has it."""
    from core import skeleton

    code = tmp_path / "code"
    code.mkdir()
    (code / "fi_frame.py").write_text(skeleton.RUNTIME, encoding="utf-8")
    (code / skeleton.FIGURE_CALLS_FILE).write_text(skeleton.FRAME_FIGURE_SOURCE, encoding="utf-8")
    for name in ("fi_frame", "fi_figure_calls"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(code))
    records = tmp_path / ".fi" / "figure_records"
    records.mkdir(parents=True)
    monkeypatch.setenv("FI_FIGURE_RECORDS", str(records))
    yield records
    for name in ("fi_frame", "fi_figure_calls"):
        sys.modules.pop(name, None)


class _Grid:
    """``growth`` at three sizes for each of the given temperatures."""

    declared = ["growth"]

    def __init__(self, temps):
        self.temps = list(temps)
        self.gave_values = set()
        self.last_error = {}

    def keys(self):
        return ["size=%d,temp=%s" % (s, t) for t in self.temps for s in (1, 2, 3)]

    def setting(self, key):
        return {"size": int(key.split(",")[0].split("=")[1]), "temp": int(key.split("temp=")[1])}

    def mean(self, ident, key):
        self.gave_values.add(ident)
        cell = self.setting(key)
        return 1.0 + 0.1 * cell["temp"] + cell["size"]

    def per_level(self, ident, key):
        return []

    def n(self, ident, key):
        return 10

    def interval(self, ident, key):
        return (self.mean(ident, key) - 0.5, self.mean(ident, key) + 0.5)

    def reset_reads(self):
        self.gave_values.clear()


# ---- R2: a default figure's curves are levels of the series setting --------------------------------------------------------------
def test_a_default_figure_over_many_settings_has_a_condensed_legend_and_levelled_lines(tmp_path, frame5):
    out = af.default_figure({"id": "g", "file": "g.png", "shows": ["growth"], "x": "size", "series": "temp"},
                            _Grid(range(1, 31)), str(tmp_path))
    assert out["status"] == "ok" and out["fallback"] is True
    record = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))
    series = record["panels"][0]["series"]
    assert len(series) == 30 and all(e.get("level") for e in series)
    assert len(record["legend"]["entries"]) <= frame_figure.LEGEND_LEVELS


# ---- R3a: bars of several series sit side by side ---------------------------------------------------------------------------------
class _Host(frame_figure.FigureCalls):
    def __init__(self, series_ids=("a", "b"), settings=("pi",)):
        self.fig, axes = plt.subplots(1, 1, squeeze=False)
        self.axes = list(axes[0])
        self.table = _Grid([1])
        self.x = "pi"
        self.series = None
        self.fallback = False
        self.settings = list(settings)
        self.series_ids = list(series_ids)
        self.labels = {}


def _extents(patches):
    return [(p.get_x(), p.get_x() + p.get_width()) for p in patches]


def test_bars_of_two_series_at_the_same_x_do_not_cover_each_other():
    h = _Host()
    first = h.bars(0, "a", [0.05, 0.2], [1.0, 2.0], "other")
    second = h.bars(0, "b", [0.05, 0.2], [2.0, 1.0], "other")
    h.finish({}, {})
    for (lo_a, hi_a), (lo_b, hi_b) in zip(_extents(first), _extents(second)):
        assert hi_a <= lo_b + 1e-12 or hi_b <= lo_a + 1e-12  # side by side
    assert all(abs(hi - lo - 0.4) < 1e-9 for lo, hi in _extents(first) + _extents(second))  # 0.8 of a slot over two series
    # the group stays centred on its slot: the values are evenly spaced categories, named by the x values
    assert abs((_extents(first)[0][0] + _extents(second)[0][1]) / 2 - 0.0) < 1e-9
    assert [t.get_text() for t in h.axes[0].get_xticklabels()] == ["0.05", "0.2"]


def test_bars_error_bars_follow_their_bar():
    h = _Host()
    first = h.bars(0, "a", [1, 2], [1.0, 2.0], "other", yerr=[0.1, 0.1])
    h.bars(0, "b", [1, 2], [2.0, 1.0], "other", yerr=[0.1, 0.1])
    h.finish({}, {})
    centres = [p.get_x() + p.get_width() / 2 for p in first]
    assert centres == pytest.approx([-0.2, 0.8])
    errs = [c for c in h.axes[0].containers if hasattr(c, "has_yerr")]
    assert len(errs) == 2
    cap_x = sorted(float(x) for cap in errs[0].lines[1] for x in cap.get_xdata())
    assert cap_x == pytest.approx(sorted(centres * 2))  # the caps of the first series sit on its (moved) bars
    stem_x = sorted(seg[0][0] for lc in errs[0].lines[2] for seg in lc.get_segments())
    assert stem_x == pytest.approx(sorted(centres))


# ---- R3b: a legend over the data is put outside the axes ----------------------------------------------------------------------------
def test_a_legend_over_the_data_is_put_outside_and_the_record_says_so(monkeypatch):
    h = _Host(series_ids=("a", "b", "c"))
    monkeypatch.setattr(frame_figure.FigureCalls, "_free_corner", lambda self, ax, rec, rows: "upper right")
    xs = [i / 10 for i in range(11)]
    for k, name in enumerate(("a", "b", "c")):
        h.line(0, name, xs, [0.9 + 0.05 * k] * 11, "other")  # lines that run through the corner the legend was given
    h.finish({}, {})
    legend = h.record()["legend"]
    assert legend["scope"] == "figure" and legend["adjusted"]["was"]["over_data"] > 0
    assert legend["adjusted"]["after"]["over_data"] == 0
    assert h.axes[0].get_legend() is None


# ---- R4: the plan's nominal level is a named reference --------------------------------------------------------------------------------
def test_a_nominal_level_the_plan_declares_is_a_reference_by_its_id_and_a_typed_constant_is_not(tmp_path, frame5, monkeypatch):
    monkeypatch.setenv(frame_figure.LAST_RUN_ENV, "1")  # no repair left: a line FI cannot name is left out and recorded
    spec = {"id": "g", "file": "g.png", "shows": ["growth"], "x": "size", "nominal": {"growth": 0.95}}
    plan = af._plan_for(spec, None, _Grid([1]))
    assert plan["references"]["growth_nominal"]["value"] == 0.95

    def body(fig, table):
        fig.line(0, "growth", [1, 2, 3], [2.0, 3.0, 4.0], "growth")
        fig.ref(0, "growth_nominal")
        fig.ref(0, value=0.95)  # equal to the plan's level, but a typed number is never bound by its value
        fig.ref(0, value=0.77)  # a constant the plan does not hold

    out = af.run_figures([spec], {"figure_g": body}, _Grid([1]), str(tmp_path))
    assert out[0]["status"] == "ok", out
    record = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))
    refs = record["panels"][0]["refs"]
    assert [r["id"] for r in refs] == ["growth_nominal"]
    assert [r["what"] for r in out[0].get("removed", [])] == ["a line at 0.95", "a line at 0.77"]


def test_every_result_with_a_declared_level_gives_the_figure_a_nominal_entry():
    import types

    from core import analysis_frame

    c = types.SimpleNamespace(results=[types.SimpleNamespace(id="growth", known_value={"level": 0.95}),
                                       types.SimpleNamespace(id="other", known_value={"level": True}),
                                       types.SimpleNamespace(id="plain", known_value=None)])
    assert analysis_frame._nominal_levels(c) == {"nominal": {"growth": 0.95}}
    assert analysis_frame._nominal_levels(c, ["plain"]) == {}  # a result the figure does not draw offers no level
    assert analysis_frame._nominal_levels(types.SimpleNamespace(results=[])) == {}


def test_each_result_keeps_its_own_nominal_line_with_its_own_name():
    spec = {"id": "g", "file": "g.png", "shows": ["growth", "other"], "x": "size", "nominal": {"growth": 0.95, "other": 0.95}}
    refs = af._plan_for(spec, None, _Grid([1]))["references"]
    assert {k: r["value"] for k, r in refs.items()} == {"growth_nominal": 0.95, "other_nominal": 0.95}  # the same number from two results stays two named lines
    assert "growth" in refs["growth_nominal"]["label"] and "other" in refs["other_nominal"]["label"]


# ---- the default figure with a setting and a compared value ------------------------------------------------------------------------
class _Both(_Grid):
    """``growth`` recorded for each of two arms inside every run, at two temperatures."""

    def per_level(self, ident, key):
        return ["a", "b"]

    def level_means(self, ident, key):
        self.gave_values.add(ident)
        t = self.setting(key)["temp"]
        return {"a": 1.0 + t, "b": 5.0 + t}

    def mean(self, ident, key):
        self.last_error[ident] = "UnknownResult: `%s` is recorded for each value" % ident
        raise LookupError("`%s` is recorded for each value of `mode` (a, b)" % ident)

    def compared(self):
        return ["mode"]


def test_a_default_figure_with_a_setting_and_compared_values_keeps_one_curve_for_each_pair(tmp_path, frame5):
    out = af.default_figure({"id": "g", "file": "g.png", "shows": ["growth"], "x": "size", "series": "temp"}, _Both([10, 20]), str(tmp_path))
    assert out["status"] == "ok", out
    series = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"][0]["series"]
    assert len(series) == 4 and len({e["id"] for e in series}) == 4
    assert len({tuple(e["y"]) for e in series}) == 4  # no curve joins points of two levels


# ---- the legend condenses for each result, and only ordered numbers ---------------------------------------------------------------------
class _Many(frame_figure.FigureCalls):
    def __init__(self, results, levels):
        self.fig, axes = plt.subplots(1, 1, squeeze=False)
        self.axes = list(axes[0])
        self.levels = list(levels)

        class T:
            declared = list(results)

            def keys(_self):
                return ["size=%d,m=%s" % (s, m) for m in levels for s in (1, 2)]

            def mean(_self, ident, key):
                return 1.0

        self.table = T()
        self.x, self.series, self.fallback = "size", "m", False
        self.settings, self.series_ids, self.labels = ["size", "m"], [], {}


def test_levels_are_condensed_for_each_result_on_its_own():
    h = _Many(["r%d" % i for i in range(5)], [1, 2, 3])  # 5 results x 3 levels = 15 curves, each result has only 3
    for r in h.table.declared:
        for m in (1, 2, 3):
            h.line(0, m, [1, 2], [1.0 + m + 10 * int(r[1:]), 2.0], r)
    h.finish({}, {})
    assert len(h.record()["legend"]["entries"]) == 15


def test_levels_that_are_names_keep_every_legend_entry():
    names = ["m%02d" % i for i in range(13)]
    h = _Many(["r"], names)
    for k, n in enumerate(names):
        h.line(0, n, [1, 2], [1.0 + k, 2.0], "r")
    h.finish({}, {})
    assert len(h.record()["legend"]["entries"]) == 13


# ---- bars over a setting are categories --------------------------------------------------------------------------------------------------
def test_sparse_bars_keep_a_visible_slot_and_the_axis_names_the_values():
    h = _Host()
    a = h.bars(0, "a", [1, 1.001, 100], [1.0, 2.0, 3.0], "other")
    b = h.bars(0, "b", [1, 1.001, 100], [3.0, 2.0, 1.0], "other")
    h.finish({}, {})
    lo, hi = h.axes[0].get_xlim()
    for p in list(a) + list(b):
        assert p.get_width() / (hi - lo) >= 0.8 / 3 / 3 - 1e-9  # each bar is a fair share of its slot, whatever the gaps between x values
    assert [t.get_text() for t in h.axes[0].get_xticklabels()] == ["1", "1.001", "100"]


def test_a_text_put_at_a_bar_moves_with_it_when_another_series_joins_the_slot():
    h = _Host()
    a = h.bars(0, "a", [1, 2], [1.0, 2.0], "other")
    labels = [h.axes[0].text(p.get_x() + p.get_width() / 2, p.get_height(), "v") for p in a]
    h.bars(0, "b", [1, 2], [2.0, 1.0], "other")
    for p, t in zip(a, labels):
        assert abs(t.get_position()[0] - (p.get_x() + p.get_width() / 2)) < 1e-9


def test_bars_over_a_setting_stay_a_drawing_FI_checked(tmp_path, frame5):
    def body(fig, table):
        fig.bars(0, "growth", [1, 2], [2.0, 3.0], "growth")

    out = af.run_figures([{"id": "g", "file": "g.png", "shows": ["growth"], "x": "size"}], {"figure_g": body}, _Grid([1]), str(tmp_path))
    record = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))
    assert out[0]["status"] == "ok" and [p["raw"] for p in record["panels"]] == [False]  # the axis' value names are FI's, not typed text
