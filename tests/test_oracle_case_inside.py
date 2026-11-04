"""A check's case in a search study is not given the whole lists the run compares inside it as if it named them."""
from __future__ import annotations

from core import in_run, optimise
from core.trial_runner import case_inside, run_case_trials  # noqa: F401

PROTOCOL = {"optimisation": {
    "fixed": {"gain": 2.0},
    "baseline": {"values": {"knob": 1.0}},
    "in_each_run": {"width": [10, 20, 30], "mode": ["a", "b"]},
}}


def _judgeable(case):
    inside = in_run.of_study(PROTOCOL)
    full = case_inside(inside, case)
    named = {k: v for k, v in case.items() if k in full}
    return not named or bool(in_run.levels_named(named, in_run.level_cells(full)))


def test_case_leaves_out_lists_it_does_not_name():
    got = optimise.complete_case(PROTOCOL, {"width": 20})
    assert got == {"gain": 2.0, "knob": 1.0, "width": 20}
    assert _judgeable(got)


def test_case_naming_nothing_inside_is_judgeable():
    got = optimise.complete_case(PROTOCOL, {"gain": 3.0})
    assert "width" not in got and "mode" not in got
    assert _judgeable(got)


def test_search_evaluation_keeps_each_list_whole():
    got = optimise.complete_case(PROTOCOL, {}, keep_inside=True)
    assert got["width"] == [10, 20, 30] and got["mode"] == ["a", "b"]


def test_case_naming_a_choice_the_study_lacks_is_still_refused():
    got = optimise.complete_case(PROTOCOL, {"mode": "c"})
    assert not _judgeable(got)
