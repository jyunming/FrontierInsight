"""A drawn series is named by its result only when every point is FI's own value at the one setting the call says.

The setting a point is checked at is fixed by the drawn x, the curve's level and the settings its panel holds (``fig.panel``);
every one of them must be in the table's key, once, at an equal value. When further settings still vary there, the series is
named only when every fitting key gives one and the same value. Equality is relative only, and a quantity FI derives per setting
is compared the same way. A reference line drawn at a table value is labelled with the plan's words for the setting."""
from __future__ import annotations

import json
import types

import matplotlib
import pytest

matplotlib.use("Agg")

from core import analysis_figures as af  # noqa: E402
from tests.test_analysis_figures import frame5  # noqa: E402,F401

@pytest.fixture(autouse=True)
def _table_code(tmp_path, monkeypatch, frame5):
    """The frame's table code beside the frame (``frame5`` writes the frame and the drawing calls)."""
    import sys

    from core import skeleton

    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "fi_table.py").write_text(skeleton.TABLE_SOURCE, encoding="utf-8")
    monkeypatch.delitem(sys.modules, "fi_table", raising=False)
    monkeypatch.syspath_prepend(str(code))
    yield
    sys.modules.pop("fi_table", None)


SIZES = [1.0, 2.0, 3.0]
RATES = [0.1, 0.2]


def _growth(size, rate):
    return size * 10.0 + rate * 100.0  # differs between rates


def _cells(extra=None):
    cells = []
    for s in SIZES:
        for r in RATES:
            g = _growth(s, r)
            cells.append({"cell": {"size": s, "rate": r},
                          "metrics": {"growth": [g - 1.0, g + 1.0], "flat": [s, s], "yield": [g / 2 - 1.0, g / 2 + 1.0],
                                      "zero": [0.0, 0.0], **((extra or {}).get((s, r), {}))}})
    return cells


def _table(cells=None, declared=("growth", "flat", "yield", "zero")):
    from fi_frame import Table

    return Table(cells or _cells(), list(declared))


def _key(s, r):
    return "size=%r,rate=%r" % (s, r)


SPREAD = {"status": "ok", "by_setting": {_key(s, r): s + r for s in SIZES for r in RATES}, "role": "result"}


def _run(tmp_path, body, series="fast", table=None, records=None, labels=None):
    ns = types.SimpleNamespace(figure_g=body, COMPARED={"speed": ["fast", "slow"]}, DERIVED={"spread": {"by": "derive_spread"}})
    spec = {"id": "g", "file": "g.png", "panels": 1, "x": "size", "series": series, "shows": ["growth"],
            "settings": {"rate": RATES, "size": SIZES}, "labels": labels or {}}
    args = (records,) if records is not None else ()
    out = af.run_figures([spec], ns, table or _table(), str(tmp_path / "figures"), *args)[0]
    assert out["status"] == "ok", out
    return out


def _series(frame5):
    return json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"][0]["series"]


def _whats(frame5):
    return [s["what"] for s in _series(frame5)]


def _at(table, rate, name="growth"):
    return [table.mean(name, _key(s, rate)) for s in SIZES]


# ---- the curve's level ----------------------------------------------------------------------------------------------------
def test_a_curve_drawn_with_another_levels_values_is_not_named(tmp_path, frame5):
    _run(tmp_path, lambda fig, t: fig.line(0, 0.1, SIZES, _at(t, 0.2), "growth"), series="rate")
    assert _whats(frame5) == ["other"]


def test_a_curve_drawn_with_its_own_levels_values_keeps_its_name(tmp_path, frame5):
    _run(tmp_path, lambda fig, t: (fig.line(0, 0.1, SIZES, _at(t, 0.1), "growth"), fig.line(0, 0.2, SIZES, _at(t, 0.2), "growth")),
         series="rate")
    assert _whats(frame5) == ["growth", "growth"]


# ---- the panel's held settings --------------------------------------------------------------------------------------------
def test_a_series_of_another_slice_than_its_panel_holds_is_not_named(tmp_path, frame5):
    def body(fig, t):
        fig.panel(0, rate=0.2)
        fig.line(0, "fast", SIZES, _at(t, 0.1), "growth")

    _run(tmp_path, body)
    assert _whats(frame5) == ["other"]


def test_a_series_of_the_slice_its_panel_holds_keeps_its_name(tmp_path, frame5):
    def body(fig, t):
        fig.panel(0, rate=0.2)
        fig.line(0, "fast", SIZES, _at(t, 0.2), "growth")

    _run(tmp_path, body)
    assert _whats(frame5) == ["growth"]


def test_a_slice_the_panel_neither_holds_nor_varies_is_not_named(tmp_path, frame5):
    _run(tmp_path, lambda fig, t: fig.line(0, "fast", SIZES, _at(t, 0.1), "growth"))
    assert _whats(frame5) == ["other"]


def test_values_equal_over_every_fitting_key_are_named_without_a_hold(tmp_path, frame5):
    _run(tmp_path, lambda fig, t: fig.line(0, "fast", SIZES, _at(t, 0.1, "flat"), "flat"))
    assert _whats(frame5) == ["flat"]


# ---- equality -------------------------------------------------------------------------------------------------------------
def test_a_drawn_near_zero_is_not_a_recorded_zero(tmp_path, frame5):
    def body(fig, t):
        fig.panel(0, rate=0.1)
        fig.line(0, "fast", SIZES, [9e-13, 0.0, 0.0], "zero")

    _run(tmp_path, body)
    assert _whats(frame5) == ["other"]


def test_two_points_at_one_x_are_not_compared(tmp_path, frame5):
    def body(fig, t):
        fig.panel(0, rate=0.1)
        y = _at(t, 0.1)
        fig.line(0, "fast", [1.0, 1.0, 2.0], [y[0], y[0], y[1]], "growth")

    _run(tmp_path, body)
    assert _whats(frame5) == ["other"]


def test_a_level_is_written_so_it_reads_back_as_its_value(tmp_path, frame5):
    near = 1.0000004
    cells = [{"cell": {"size": s, "rate": r}, "metrics": {"growth": [s * r, s * r]}} for s in SIZES for r in (1.0, near)]
    table = _table(cells, ("growth",))

    def body(fig, t):
        for r in (1.0, near):
            fig.line(0, r, SIZES, [t.mean("growth", _key(s, r)) for s in SIZES], "growth")

    _run(tmp_path, body, series="rate", table=table)
    labels = [s["label"] for s in _series(frame5)]
    assert [float(label.rsplit("= ", 1)[1]) for label in labels] == [1.0, near]


# ---- a result recorded for each value compared inside the runs ------------------------------------------------------------
def _per_value_cells():
    cells = []
    for s in SIZES:
        cells.append({"cell": {"size": s}, "metrics": {"flat": [s, s]}})
        for speed, k in (("fast", 1.0), ("slow", 3.0)):
            cells.append({"cell": {"size": s, "speed": speed}, "level": {"speed": speed}, "metrics": {"growth": [s * k, s * k]}})
    return cells


def test_a_series_drawn_with_another_compared_values_numbers_is_not_named(tmp_path, frame5):
    table = _table(_per_value_cells(), ("growth", "flat"))
    slow = [s * 3.0 for s in SIZES]
    _run(tmp_path, lambda fig, t: fig.line(0, "fast", SIZES, slow, "growth"), table=table)
    assert _whats(frame5) == ["other"]
    _run(tmp_path, lambda fig, t: fig.line(0, "slow", SIZES, slow, "growth"), table=_table(_per_value_cells(), ("growth", "flat")))
    assert _whats(frame5) == ["growth"]  # known negative: its own compared value


# ---- one result against another -------------------------------------------------------------------------------------------
def test_pairs_of_another_held_setting_are_not_named(tmp_path, frame5):
    def body(rate):
        def draw(fig, t):
            fig.panel(0, rate=0.1)
            fig.points(0, "fast", ("growth", _at(t, rate)), _at(t, rate, "yield"), "yield")
        return draw

    _run(tmp_path, body(0.2))
    assert _whats(frame5) == ["other"]
    _run(tmp_path, body(0.1))
    assert _whats(frame5) == ["yield"]  # known negative: the panel's own pairs


# ---- a quantity FI derives per setting ------------------------------------------------------------------------------------
def test_a_derived_quantity_is_read_from_the_table_and_named_when_it_is_fis_own(tmp_path, frame5):
    def body(fig, t):
        fig.panel(0, rate=0.1)
        fig.line(0, "fast", SIZES, [t.mean("spread", _key(s, 0.1)) for s in SIZES], "spread")

    _run(tmp_path, body, records={"spread": SPREAD})
    assert _whats(frame5) == ["spread"]


def test_a_derived_quantity_of_another_slice_is_not_named(tmp_path, frame5):
    def body(fig, t):
        fig.panel(0, rate=0.1)
        fig.line(0, "fast", SIZES, _at(t, 0.1), "growth")
        fig.line(0, "slow", SIZES, [s + 0.2 for s in SIZES], "spread")

    _run(tmp_path, body, records={"spread": SPREAD})
    assert _whats(frame5) == ["growth", "other"]


def test_without_the_derived_record_a_derived_quantity_stays_not_compared(tmp_path, frame5):
    """A frozen frame that calls the figure step without FI's derived record keeps today's behaviour, with no error."""
    def body(fig, t):
        fig.panel(0, rate=0.1)
        fig.line(0, "fast", SIZES, _at(t, 0.1), "growth")
        fig.line(0, "slow", SIZES, [s + 0.1 for s in SIZES], "spread")

    _run(tmp_path, body)
    s = _series(frame5)[1]
    assert s["what"] == "other" and s["declared_as"]["checked"] == "not compared"


# ---- the key itself -------------------------------------------------------------------------------------------------------
def test_a_key_naming_a_setting_twice_or_lacking_one_never_agrees():
    from core.frame_figure import FigureCalls

    class T:
        declared = ["growth"]

        def keys(self):
            return ["size=1,size=2,rate=0.1", "size=1", "size=1,rate=0.1"]

    host = FigureCalls()
    host.table = T()
    assert host._fitting_keys({"size": 1.0, "rate": 0.1}) == ["size=1,rate=0.1"]


# ---- a reference line at a table value ------------------------------------------------------------------------------------
def test_a_reference_at_a_table_value_is_labelled_with_the_plans_words(tmp_path, frame5):
    def body(fig, t):
        fig.panel(0, rate=0.1)
        fig.line(0, "fast", SIZES, _at(t, 0.1), "growth")
        fig.ref(0, at="growth", key=_key(2.0, 0.1))

    _run(tmp_path, body, labels={"size": "Colony size", "rate": "Feed rate", "growth": "Growth"})
    ref = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"][0]["refs"][0]
    assert ref["label"] == "Mean of Growth at Colony size = 2, Feed rate = 0.1"
