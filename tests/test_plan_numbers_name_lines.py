"""A constant a figure types is named by the plan's own number it equals: a fixed number of the plan (a significance level, a
true parameter) is a reference of its own, preferred over the several results that happen to share the value."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from core import analysis_figures  # noqa: E402
from tests.test_frame_figure import _Host  # noqa: E402


def _host(references):
    h = _Host(identity=True)
    h.references = references
    return h


def test_a_typed_constant_equal_to_one_fixed_number_of_the_plan_is_drawn_and_named_by_it():
    h = _host({"alpha": {"value": 0.05, "label": "alpha = 0.05", "fixed": True},
               "rate_b": {"value": 0.07, "label": "Rate b"}})
    h.line(0, "fast", [1, 2, 3], [0.1, 0.2, 0.3], "growth")
    h.ref(0, value=0.05)
    refs = h.record()["panels"][0]["refs"]
    assert [(r["id"], r["label"]) for r in refs] == [("alpha", "alpha = 0.05")], refs
    assert not h.record().get("removed")


def test_two_fixed_numbers_with_the_same_value_still_leave_the_line_unnamed():
    h = _host({"alpha": {"value": 0.05, "label": "alpha = 0.05", "fixed": True},
               "limit_b": {"value": 0.05, "label": "limit_b = 0.05", "fixed": True}})
    h.line(0, "fast", [1, 2, 3], [0.1, 0.2, 0.3], "growth")
    with pytest.raises(ValueError, match="several references"):
        h.ref(0, value=0.05)


def test_the_plan_s_thresholds_are_offered_to_the_figure_as_named_references_and_other_fixed_numbers_are_not():
    class _Table:
        _thresholds = {"alpha": 0.05, "limit_b": 2}
        _references = {}

        def keys(self):
            return []

    plan = analysis_figures._plan_for({}, {"FIXED": {"trials": 200}}, _Table())
    refs = plan["references"]
    assert refs["alpha"] == {"value": 0.05, "label": "alpha = 0.05", "fixed": True}
    assert refs["limit_b"]["label"] == "limit_b = 2"
    assert "trials" not in refs  # a count of the study is not a line


def test_a_setting_value_still_matches_its_grid_value_as_a_plain_number():
    # a panel's setting (pi1=0.0) must still match the plan's grid: the helper that unwraps a value keeps its own name
    assert analysis_figures._in_grid(0.0, [0.0, 0.05, 0.2]) == 0.0
    assert analysis_figures._in_grid("fast", ["fast", "slow"]) == "fast"
    assert analysis_figures._plain(0.05) == 0.05
