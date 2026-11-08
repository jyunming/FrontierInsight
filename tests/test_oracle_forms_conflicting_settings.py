"""A setting given two different numbers is left out of a design's fixed settings, never decided by read order."""
from __future__ import annotations

from core import oracle_forms as of


def _design(fixed=None, controls=None):
    return {"protocol": {"fixed": fixed or {}}, "variables": {"controls": controls or []}}


def test_same_value_in_plan_and_control_is_kept():
    assert of.fixed_settings_of_design(_design({"k": 100}, [{"name": "k", "value": 100}])) == {"k": 100.0}


def test_plan_and_control_disagree_name_left_out():
    assert "k" not in of.fixed_settings_of_design(_design({"k": 200}, [{"name": "k", "value": 100}]))


def test_control_listed_first_in_order_does_not_matter():
    got = of.fixed_settings_of_design(_design({"k": 100}, [{"name": "k", "value": 200}]))
    assert "k" not in got


def test_two_controls_disagree_name_left_out():
    got = of.fixed_settings_of_design(_design(None, [{"name": "k", "value": 100}, {"name": "k", "value": 200}]))
    assert "k" not in got


def test_clash_inside_plan_is_not_revived_by_a_control():
    design = {"protocol": {"fixed": {"k": 1}, "constants": {"k": 2}},
              "variables": {"controls": [{"name": "k", "value": 1}]}}
    assert "k" not in of.fixed_settings_of_design(design)


def test_unrelated_names_unaffected():
    got = of.fixed_settings_of_design(_design({"a": 1.0, "b": 2.0, "k": 5},
                                              [{"name": "b", "value": 2.0}, {"name": "c", "value": 3.0},
                                               {"name": "k", "value": 6}]))
    assert got == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_several_conflicts_only_agreeing_kept():
    got = of.fixed_settings_of_design(_design({"k": 100, "rate": 50, "n": 10},
                                              [{"name": "k", "value": 200}, {"name": "rate", "value": 50},
                                               {"name": "n", "value": 20}]))
    assert got == {"rate": 50.0}


def test_empty_or_malformed_design():
    for d in (None, {}, {"protocol": None}, {"variables": None}):
        assert of.fixed_settings_of_design(d) == {}


def test_formula_with_conflicting_name_stays_unresolved():
    fixed = of.fixed_settings_of_design(_design({"k": 200}, [{"name": "k", "value": 100}]))
    value, why = of.formula_value({"expected_formula": "k * 2", "case": {}}, fixed)
    assert value is None
    assert "k" in why
