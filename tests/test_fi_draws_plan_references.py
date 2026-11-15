"""FI draws the plan's reference of a result a panel draws when the figure's body left it out
(core/frame_figure.py::FigureCalls.add_plan_references, called by core/analysis_figures.py before ``finish``). Left out,
the reference was only a Limitations sentence ("Figure 2 draws no line for the plan's reference ..."). FI draws it only
where it has the reference's value at every setting the panel shows, never beside a line of the body's own, and never one
``fig.ref`` refuses. Made-up study: a colony's ``growth`` read at ``size``. No model is called."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from core import figure_text_check as ftc  # noqa: E402
from tests.test_frame_figure import _Host, _Table  # noqa: E402

PROTOCOL = {"metrics": [{"id": "growth", "label": "Growth", "reference": {"value": 0.5, "formula": "0.5", "kind": "value"}}]}


class _TwoSizes(_Table):
    def __init__(self, references):
        self._references = references

    def keys(self):
        return ["size=1", "size=2"]


def _host(references):
    h = _Host()
    h.table = _TwoSizes(references)
    return h


def _panel(h):
    return h.record()["panels"][0]


def _drawn_as_growth(h):
    assert [s["what"] for s in _panel(h)["series"]] == ["growth"], "the line is the table's: FI names it the result"


def _missing(h):
    p = _panel(h)
    return ftc.missing_references({"panels": [{"panel": 1, "raw": False, "series": p["series"], "refs": p["refs"]}]}, PROTOCOL)


def test_a_reference_the_body_left_out_is_drawn_by_fi_and_the_check_finds_it_drawn():
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.5}})
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    _drawn_as_growth(h)
    assert _missing(h), "known positive: before FI draws it, the reference is reported missing"
    assert h.add_plan_references() == [(0, "growth")]
    refs = _panel(h)["refs"]
    assert [(r["id"], r["source"], r["value"], r.get("drawn_by")) for r in refs] == [("growth", "plan", 0.5, "FI")]
    assert _missing(h) == []


def test_a_reference_with_no_value_at_some_setting_the_panel_shows_is_not_drawn():
    # the plan's `when`: a reference that applies at size=1 only would read as a line over both sizes
    h = _host({"growth": {"size=1": 0.5}})
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    _drawn_as_growth(h)
    assert h.add_plan_references() == [] and _panel(h)["refs"] == []
    assert _missing(h), "the Limitations sentence stays"


def test_a_panel_with_a_line_of_the_bodys_own_is_left_as_it_is():
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.5}})
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    h.ref(0, at="growth", key="size=1")  # a line at a table value: it may be the reference, FI cannot tell
    before = list(_panel(h)["refs"])
    assert h.add_plan_references() == [] and _panel(h)["refs"] == before


def test_a_reference_the_body_drew_is_not_drawn_twice():
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.5}})
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    h.ref(0, "growth")
    assert h.add_plan_references() == []
    assert [r.get("drawn_by") for r in _panel(h)["refs"]] == [None]


def test_a_reference_that_changes_along_the_x_setting_is_drawn_as_fis_curve():
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.6}})
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    _drawn_as_growth(h)
    assert h.add_plan_references() == [(0, "growth")]
    curve = [s for s in _panel(h)["series"] if s.get("drawn_by") == "FI"]
    assert [(s["kind"], s["source"], s["y"]) for s in curve] == [("curve", "plan", [0.5, 0.6])]
    assert _missing(h) == []


class _SizeAndLoad(_Table):
    _references = {"growth": {"size=1,load=1": 0.5, "size=1,load=2": 0.7, "size=2,load=1": 0.6, "size=2,load=2": 0.8}}

    def keys(self):
        return list(self._references["growth"])


def test_a_reference_fig_ref_refuses_is_left_out():
    # it also changes with `load`, which the panel does not hold: fig.ref refuses it, so FI draws nothing
    h = _Host()
    h.table = _SizeAndLoad()
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    _drawn_as_growth(h)
    assert h.add_plan_references() == [] and _panel(h)["refs"] == []
    assert not any(s.get("drawn_by") for s in _panel(h)["series"])
    assert _missing(h), "known positive: the Limitations sentence stays"


def test_a_result_without_a_reference_and_a_figure_without_references_draw_nothing():
    h = _host({"other_result": {"size=1": 1.0, "size=2": 1.0}})
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    assert h.add_plan_references() == []
    h = _host({})
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    assert h.add_plan_references() == []


def _label_drawn(kinds):
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.5}})
    if kinds is not None:
        h.reference_kinds = kinds
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    assert h.add_plan_references() == [(0, "growth")]
    return [r["label"] for r in _panel(h)["refs"]]


def test_fis_line_is_named_by_the_plans_kind_of_reference():
    # an upper or a lower limit of a result is not its expected value: the line's name says what the plan says it is
    assert _label_drawn({"growth": "bound"}) == ["Upper bound of Growth (plan formula)"]
    assert _label_drawn({"growth": "lower_bound"}) == ["Lower bound of Growth (plan formula)"]
    assert _label_drawn({"growth": "value"}) == ["Expected Growth (plan formula)"]


def test_a_figure_without_kinds_keeps_the_expected_name():
    # a frame written before kinds were carried, or a reference the plan gives no kind: "Expected", as before
    assert _label_drawn(None) == ["Expected Growth (plan formula)"]
    assert _label_drawn({}) == ["Expected Growth (plan formula)"]


def test_a_curve_reference_is_named_by_its_kind_too():
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.6}})
    h.reference_kinds = {"growth": "lower_bound"}
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    assert h.add_plan_references() == [(0, "growth")]
    assert [s["label"] for s in _panel(h)["series"] if s.get("drawn_by") == "FI"] == ["Lower bound of Growth (plan formula)"]


def test_a_curve_is_never_drawn_past_the_x_the_result_is_drawn_at():
    # review finding: the result drawn at size=1 only; FI's curve through size=1 and 2 would reach a setting with no data
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.6}})
    h.line(0, "fast", [1], [2.5], "growth")
    lines_before = len(h.axes[0].lines)
    assert h.add_plan_references() == []
    assert not any(s.get("drawn_by") for s in _panel(h)["series"]) and len(h.axes[0].lines) == lines_before


def test_a_vertical_line_at_the_reference_is_not_its_line_on_the_y_axis():
    # review finding: `fig.ref(0, "growth", axis="x")` is not the reference on the result's axis, so FI still draws that one
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.5}})
    h.line(0, "fast", [1, 2], [2.5, 2.5], "growth")
    h.ref(0, "growth", axis="x")
    assert _missing(h), "known positive: the vertical line does not count as drawn"
    assert h.add_plan_references() == [(0, "growth")]
    assert _missing(h) == []


def test_a_gap_in_the_result_is_not_an_x_it_is_drawn_at():
    # second review: the result has no point at size=2 (a gap); FI's curve would still reach size=2
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.6}})
    h.line(0, "fast", [1, 2], [2.5, None], "growth")
    lines_before, lim_before = len(h.axes[0].lines), h.axes[0].dataLim.frozen()
    assert h.add_plan_references() == []
    # decided before drawing: nothing on the axes changed
    assert len(h.axes[0].lines) == lines_before and (h.axes[0].dataLim.frozen().get_points() == lim_before.get_points()).all()


def test_a_constant_reference_is_drawn_even_where_the_result_has_a_gap():
    h = _host({"growth": {"size=1": 0.5, "size=2": 0.5}})
    h.line(0, "fast", [1, 2], [2.5, None], "growth")
    assert h.add_plan_references() == [(0, "growth")]
