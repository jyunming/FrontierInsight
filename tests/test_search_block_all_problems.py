"""A search block with several problems is refused with all of them in one reply, so the plan's model can fix them in one
answer instead of one per request (a live run was refused 12 times, each time for one problem, and stopped before any run).
The variables are made-up neutral names."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import optimise_search as osearch
from core import optimisation_plan as op
from core import plan
from tests.test_replay_misplaced_part import STATE, _setup, _text

RICH: dict[str, Any] = {
    "objective": {"quantity": " gain ", "direction": "Maximize", "unit": "dB"},
    "design_variables": [
        {"name": "a", "low": 0, "high": 1, "kind": "continuous"},
        {"name": "n", "low": 1, "high": 9, "kind": "integer"},
        {"name": "mode", "kind": "choice", "values": ["p", "q"]},
        {"name": "img", "shape": [2, 2], "low": 0, "high": 1},
    ],
    "fixed": {"gap": 3.5},
    "baseline": {"values": {"a": 0.5, "n": 3, "mode": "p", "img": [[0, 1], [1, 0]]}, "source": "the topic"},
    "constraints": [{"quantity": "loss", "limit": "< 2"}, {"quantity": "margin", "limit": "≥ 0.5", "unit": "dB"}],
    "numerical_settings": {"dx": {"search": 0.4, "finer": "smaller", "tolerance": 0.05}, "k": 8,
                           "cells": {"search": 10, "check": [20, 40], "finer": "larger"}},
    "evaluation_budget": {"starts": 2, "per_start": 30, "seconds_per_evaluation": 1.5},
    "runs_per_evaluation": 1, "search_method": " bounded_local ",
    "improvement_tolerance": {"value": 0.1, "mode": "Relative"}, "target": 3, "constraint_margin": 0.01,
    "grid": {"a": [0.2, 0.8], "family": ["u", "v"], "gap": [3.5, 4.5]},
    "in_each_run": {"angle": [0, 10, 20]},
}

#: What the reader gave for RICH before this change (taken from the code of main), byte for byte.
RICH_READ_BEFORE = (
    '{"objective": {"quantity": "gain", "direction": "maximise", "unit": "dB"}, "design_variables": [{"name": "a", "l'
    'ow": 0, "high": 1, "kind": "continuous"}, {"name": "n", "low": 1, "high": 9, "kind": "integer"}, {"name": "mode"'
    ', "kind": "choice", "values": ["p", "q"]}, {"name": "img", "shape": [2, 2], "low": 0, "high": 1}], "fixed": {"ga'
    'p": 3.5, "family": "u"}, "baseline": {"values": {"a": 0.5, "n": 3, "mode": "p", "img": [[0, 1], [1, 0]]}, "sourc'
    'e": "the topic"}, "constraints": [{"quantity": "loss", "limit": "<= 2"}, {"quantity": "margin", "limit": ">= 0.5'
    '", "unit": "dB"}], "numerical_settings": {"dx": {"search": 0.4, "finer": "smaller", "tolerance": 0.05}, "k": {"s'
    'earch": 8}, "cells": {"search": 10, "check": [20, 40], "finer": "larger"}}, "evaluation_budget": {"starts": 2, "'
    'per_start": 30, "seconds_per_evaluation": 1.5}, "runs_per_evaluation": 1, "search_method": "bounded_local", "imp'
    'rovement_tolerance": {"value": 0.1, "mode": "relative"}, "target": 3, "constraint_margin": 0.01, "grid": {"a": ['
    '0.2, 0.8], "family": ["u", "v"], "gap": [3.5, 4.5]}, "in_each_run": {"angle": [0, 10, 20]}}')


def _three_problems() -> dict[str, Any]:
    """A block that is readable but for three independent problems, in three different parts."""
    block = copy.deepcopy(RICH)
    block["constraints"] = [{"quantity": "d", "limit": "< other_setting"}]
    block["numerical_settings"] = {"m": {"search": 0}}
    block["grid"] = {"a": [0.2], "w": [1, 2]}
    block.pop("in_each_run")
    return block


def test_a_readable_block_is_read_exactly_as_before() -> None:
    fixed, why = op.normalize(RICH)
    assert why is None and json.dumps(fixed) == RICH_READ_BEFORE
    assert op.problems_of(RICH) == []


def test_three_independent_problems_are_all_given() -> None:
    fixed, why = op.normalize(_three_problems())
    assert fixed is None and why
    problems = op.split_problems(why)
    assert len(problems) == 3 and problems == op.problems_of(_three_problems())
    # in the order the block is read: the first is the one a reader of one problem at a time always gave
    assert problems[0].startswith("`protocol.optimisation.constraints` entry 1 (`d`)")
    assert problems[1].startswith("`protocol.optimisation.numerical_settings` `m`")
    assert problems[2].startswith("`protocol.optimisation.grid` scans `w`")
    assert op.first_problem(why) == problems[0] and why.startswith(problems[0])


def test_one_problem_gives_the_reason_it_always_gave() -> None:
    block = copy.deepcopy(RICH)
    block["grid"] = {"a": [0.2], "w": [1, 2]}
    block.pop("in_each_run")
    _, why = op.normalize(block)
    assert why == ("`protocol.optimisation.grid` scans `w`, which is neither a design variable nor a condition held fixed: "
                   "to compare the study across `w`, also list it under `protocol.optimisation.fixed` with the value the "
                   "search holds it at (a setting whose values are names needs no such entry)")
    assert op.split_problems(why) == [why]
    _, why = op.normalize({**RICH, "objective": {"quantity": "gain"}})
    assert why.startswith("`protocol.optimisation.objective.direction` must be exactly one of minimise, maximise")
    assert op.split_problems(why) == [why]


def test_a_part_that_cannot_be_read_skips_only_the_checks_that_need_it() -> None:
    # a design variable that cannot be read: its name is still known, so the baseline's value for it and a scan of it are
    # not called wrong, and everything else is still checked
    block = copy.deepcopy(RICH)
    block["design_variables"][1] = {"name": "n", "low": "x", "high": 9}
    block["constraints"] = [{"quantity": "d", "limit": "bad"}]
    problems = op.problems_of(block)
    assert len(problems) == 2
    assert "`n`" in problems[0] and problems[1].startswith("`protocol.optimisation.constraints` entry 1")
    # a design variable with no name: the checks by name cannot be made, the others still are
    block["design_variables"][1] = {"low": 1, "high": 9}
    block["grid"] = {"w": [1, 2]}
    block["baseline"] = {"values": {"zz": 1}, "source": "the topic"}
    block["constraints"] = [{"quantity": "d", "limit": "bad"}]
    problems = op.problems_of(block)
    assert any("constraints" in p for p in problems) and not any("baseline.values` names" in p for p in problems)
    assert not any(".grid`" in p for p in problems)
    # a scan that cannot be read: the parts that divide it wait for it, and say nothing of it
    block = copy.deepcopy(RICH)
    block["grid"] = {"w": [1, 2]}
    block["sub_studies"] = [{"name": "s", "vary": ["w"], "hold": {}}]
    problems = op.problems_of(block)
    assert len(problems) == 1 and ".grid` scans `w`" in problems[0]


def test_the_grid_problem_is_found_when_it_is_not_the_first() -> None:
    kept, notes = op.repair(_three_problems())
    assert kept is not None
    text = " ".join(notes)
    assert "limit on d" in text and "numerical setting `m`" in text and "the setting `w`" in text
    assert "w" not in kept["grid"] and "d" not in [c["quantity"] for c in kept.get("constraints") or []]
    assert op.normalize(kept)[0] is not None


def test_the_plain_gaps_name_every_part_the_reason_names() -> None:
    design = {"study_type": "find_best_design", "protocol": {"optimisation": copy.deepcopy(RICH)}}
    block = design["protocol"]["optimisation"]
    block["objective"] = {"quantity": "gain", "direction": "sideways"}
    _, why = op.normalize(block)
    assert len(op.split_problems(why)) == 1
    one = op.plain_gaps(design, why)
    assert op._PLAIN_PARTS["objective"] in one and op._PLAIN_PARTS["evaluation_budget"] not in one
    block["evaluation_budget"] = {"starts": 0, "per_start": 5}
    _, why = op.normalize(block)
    assert len(op.split_problems(why)) == 2
    both = op.plain_gaps(design, why)
    assert op._PLAIN_PARTS["objective"] in both and op._PLAIN_PARTS["evaluation_budget"] in both


def test_a_limit_between_two_settings_is_told_how_to_be_written() -> None:
    _, why = op.normalize({**RICH, "constraints": [{"quantity": "d", "limit": "< other_setting"}]})
    assert "cannot be read" in why and "difference" in why and "returns" in why and '">= 0"' in why


def test_a_setting_worked_out_from_the_design_is_told_what_to_do() -> None:
    for bad in ({"search": 0}, {"search": "ceil(m) + 5"}, 0):
        _, why = op.normalize({**RICH, "numerical_settings": {"m": bad}})
        assert "positive number" in why and "inside the simulation" in why and "plain number" in why
        assert "leave the entry out" in why


def test_the_advice_is_what_the_search_honours() -> None:
    """The simulation returns the difference of two settings as one quantity; a limit of ">= 0" on it is judged, and a
    design whose difference is negative is not feasible (the advice of the two messages above)."""
    block = {"objective": {"quantity": "f", "direction": "minimise"},
             "constraints": [{"quantity": "inner_minus_outer", "limit": ">= 0"}]}
    ok = osearch.judge(block, {"f": 1.0, "inner_minus_outer": 0.2})
    bad = osearch.judge(block, {"f": 1.0, "inner_minus_outer": -0.2})
    assert ok["feasible"] and not bad["feasible"] and bad["violation"] > 0


def test_each_problem_names_its_entry_as_written() -> None:
    from core.engine import _entries_as_written

    block = _three_problems()
    shown = _entries_as_written({"optimisation": block}, op.block_faults(block)[1])
    assert shown.startswith("the entries as written: ") and "other_setting" in shown and '"search": 0' in shown
    one = op.block_faults({**RICH, "constraints": [{"quantity": "d", "limit": "bad"}]})[1]
    one_shown = _entries_as_written({"optimisation": {**RICH, "constraints": [{"quantity": "d", "limit": "bad"}]}}, one)
    assert one_shown.startswith("the entry as written: ")
    assert _entries_as_written({}, one, you=True) == 'the entry as you wrote it: {"quantity": "d", "limit": "bad"}', (
        "a fault carries its entry as written, so it is quoted even where the protocol is not at hand")
    assert _entries_as_written({}, [], you=True) == ""


@pytest.mark.asyncio
async def test_the_model_is_sent_every_problem_in_the_next_request(tmp_path: Path) -> None:
    block = _three_problems()
    block.pop("fixed", None)
    block["design_variables"] = [{"name": "a", "low": 0, "high": 1, "kind": "continuous"}]
    block["baseline"] = {"values": {"a": 0.5}, "source": "the topic"}
    block.pop("in_each_run", None)
    block["grid"] = {"a": [0.2], "w": [1, 2]}

    def change(design: dict[str, Any]) -> dict[str, Any]:
        design.pop("optimisation", None)
        design["protocol"] = {**{k: v for k, v in design["protocol"].items() if k != "grid"}, "optimisation": block}
        return design

    reply = plan.edit_design_block(_text(), change)
    assert reply is not None
    eng, asked, _paused = _setup(tmp_path, [reply] * 6)
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    refused = [a for a in asked if "Your last answer was not used" in a]
    assert refused, "the second request quotes the refusal"
    tail = refused[0].split("Your last answer was not used. The reason, exactly:", 1)[1]
    assert "constraints` entry 1" in tail and "numerical_settings` `m`" in tail and ".grid` scans `w`" in tail
    record = json.loads((eng.fi_dir / "search_block_asked.json").read_text(encoding="utf-8"))
    assert record["problem"].count("And also:") == 2, "all three problems are kept to be sent, none cut off"
    assert plan.plan_path(eng.quest_root).is_file() and _text()
