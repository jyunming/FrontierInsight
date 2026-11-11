"""A search block refused for some problems is asked for again from the block the plan's model wrote, so what it got right
stays right; the reader names the problems of the scan and of the scan's parts together, every failing part and not the
first, and a bound written as `max` is told to be written as `limit`. (A live run alternated between two kinds of problem for
two series of four requests, each reply fixing what the last refusal named and losing what the reply before had right.)
Neutral made-up names."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import optimisation_plan as op
from core import plan
from tests.test_replay_misplaced_part import STATE, _setup, _text

BASE: dict[str, Any] = {
    "objective": {"quantity": "gain", "direction": "maximise"},
    "design_variables": [{"name": "a", "low": 0, "high": 1, "kind": "continuous"}],
    "baseline": {"values": {"a": 0.5}, "source": "the topic"},
    "evaluation_budget": {"starts": 2, "per_start": 10},
}

#: A readable block with parts, as the reader gave it before this change (taken from the code of the commit before), byte for byte.
PARTS_READ_BEFORE = (
    '{"objective": {"quantity": "gain", "direction": "maximise"}, "design_variables": [{"name": "a", "low'
    '": 0, "high": 1, "kind": "continuous"}], "baseline": {"values": {"a": 0.5}, "source": "the topic"}, '
    '"evaluation_budget": {"starts": 2, "per_start": 10}, "fixed": {"p": 10, "q": 20}, "grid": {"p": [10,'
    ' 20], "q": [20, 40]}, "in_each_run": {"angle": [0, 10]}, "sub_studies": [{"name": "stage p", "vary":'
    ' ["p"], "hold": {"q": 20}}, {"name": "stage q", "vary": ["q"], "hold": {"p": 10}}]}')


def _parts() -> dict[str, Any]:
    return {**BASE, "fixed": {"p": 10, "q": 20}, "grid": {"p": [10, 20], "q": [20, 40]},
            "in_each_run": {"angle": [0, 10]},
            "sub_studies": [{"name": "stage p", "vary": ["p"], "hold": {"q": 20}},
                            {"name": "stage q", "vary": ["q"], "hold": {"p": 10}}]}


def test_a_readable_block_with_parts_is_read_exactly_as_before() -> None:
    fixed, why = op.normalize(_parts())
    assert why is None and json.dumps(fixed) == PARTS_READ_BEFORE


def test_a_scan_setting_that_is_not_held_and_a_part_that_does_not_hold_are_given_together() -> None:
    block = _parts()
    block["fixed"] = {"p": 10}  # `q` is scanned, neither a design variable nor held fixed
    block["sub_studies"][0]["hold"] = {}  # and the part that varies `p` says nothing about `q`
    _fixed, why = op.normalize(block)
    problems = op.split_problems(why)
    assert any("scans `q`" in p for p in problems)
    assert any("`protocol.optimisation.sub_studies`" in p and "does not say the value it holds `q`" in p for p in problems)


def test_a_part_is_read_beside_a_grid_that_has_a_problem_with_only_the_unreadable_axis_left_out() -> None:
    block = _parts()
    block["grid"] = {**block["grid"], "r": [{"x": 1}]}  # an axis that cannot be read
    block["sub_studies"][0]["hold"] = {}
    block["sub_studies"].append({"name": "stage r", "vary": ["r"], "hold": {"p": 10, "q": 20}})
    _fixed, why = op.normalize(block)
    problems = op.split_problems(why)
    assert any("`protocol.optimisation.grid` scans `r`" in p for p in problems)
    assert any("does not say the value it holds `q`" in p for p in problems)
    # the axis that cannot be read is not the cause of a second, wrong problem in the parts
    assert not any("sub_studies" in p and "`r`" in p for p in problems)


def test_a_part_is_read_beside_a_problem_in_what_every_run_goes_through() -> None:
    block = _parts()
    block["in_each_run"] = {"angle": [0]}  # one value is a condition, not a list of cases
    block["sub_studies"][1]["hold"] = {}
    _fixed, why = op.normalize(block)
    problems = op.split_problems(why)
    assert any("in_each_run` `angle` must list two or more values" in p for p in problems)
    assert any("entry 2" in p and "does not say the value it holds `p`" in p for p in problems)


def test_every_failing_part_is_given_and_one_failing_part_is_given_as_before() -> None:
    grid = {"p": [1, 2], "q": [3, 4], "r": [5, 6]}
    subs = [{"name": "one", "vary": ["p"], "hold": {}}, {"name": "two", "vary": ["q"], "hold": {"r": 5}},
            {"name": "three", "vary": ["r"], "hold": {"p": 1, "q": 3}}]
    fixed, why = plan._sub_studies(subs, grid, {})
    assert fixed is None
    found = why.split(plan.ALSO)
    assert len(found) == 2 and "entry 1" in found[0] and "entry 2" in found[1]
    assert plan.ALSO == op.ALSO
    # one failing part: the one sentence it always was
    _fixed, one = plan._sub_studies(subs[:1] + subs[2:], grid, {})
    assert one is not None and plan.ALSO not in one and "entry 1" in one
    # all parts read: the parts, and the values none of them runs are still refused
    ok, none = plan._sub_studies([subs[2], {"name": "four", "vary": ["p", "q"], "hold": {"r": 5}}], grid, {})
    assert none is None and ok is not None and len(ok) == 2


def test_the_request_for_a_search_block_says_what_a_part_holds() -> None:
    text = op.ask_request("a topic", ["the block"])
    assert "Each `sub_studies` entry lists in `vary`" in text and "in `hold` a value" in text
    assert "every other setting of the scan that has more than one value" in text


def test_a_bound_written_as_max_is_told_to_be_written_as_limit() -> None:
    for key, bound in (("max", "<="), ("min", ">=")):
        _fixed, why = op.normalize({**BASE, "constraints": [{"quantity": "loss", key: 3}]})
        assert f'you wrote `{key}`' in why and f'`limit`: "{bound} 3"' in why
        assert "difference" not in why and "compares one quantity" not in why


def test_the_two_quantities_note_is_given_only_when_the_limit_names_another_quantity() -> None:
    _fixed, named = op.normalize({**BASE, "constraints": [{"quantity": "loss", "limit": "<= other_quantity"}]})
    assert "compares one quantity with another" in named
    _fixed, plain = op.normalize({**BASE, "constraints": [{"quantity": "loss", "limit": "about three"}]})
    assert "cannot be read" in plain and "compares one quantity" not in plain
    _fixed, absent = op.normalize({**BASE, "constraints": [{"quantity": "loss"}]})
    assert "has no `limit`" in absent and "compares one quantity" not in absent


def _reply(block: dict[str, Any]) -> str:
    def change(design: dict[str, Any]) -> dict[str, Any]:
        design.pop("optimisation", None)
        design["protocol"] = {**{k: v for k, v in design["protocol"].items() if k != "grid"}, "optimisation": block}
        return design

    out = plan.edit_design_block(_text(), change)
    assert out is not None
    return out


@pytest.mark.asyncio
async def test_each_request_carries_the_block_the_last_answer_wrote_and_the_refusal(tmp_path: Path) -> None:
    first = copy.deepcopy(BASE)
    first["objective"] = {"quantity": "first_marker", "direction": "maximise"}
    first["constraints"] = [{"quantity": "loss", "max": 3}]
    second = copy.deepcopy(BASE)
    second["objective"] = {"quantity": "second_marker", "direction": "maximise"}
    second["constraints"] = [{"quantity": "loss", "limit": "about three"}]
    eng, asked, _paused = _setup(tmp_path, [_reply(first), _reply(second), _reply(second)])
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    assert len(asked) >= 3
    assert "The block you wrote last time" not in asked[0] and "first_marker" not in asked[0]
    # the second request names only the faulted path, with the entry as written; the block is not sent back whole
    assert "`protocol.optimisation.constraints[1]`" in asked[1] and '"max": 3' in asked[1]
    assert "you wrote `max`" in asked[1] and "only these parts of it could not be read" in asked[1]
    assert "first_marker" not in asked[1].split("# What is asked")[1]
    # the third: the second answer's entry, not the first's
    assert "about three" in asked[2] and "first_marker" not in asked[2].split("# What is asked")[1]
    assert "second_marker" not in asked[2].split("# What is asked")[1]
    assert json.loads((eng.fi_dir / "search_refused.json").read_text(encoding="utf-8"))["block"]["objective"]["quantity"] == "second_marker"
    # kept on disk with the refusal, so a resume sees it too
    record = json.loads((eng.fi_dir / "search_block_asked.json").read_text(encoding="utf-8"))
    assert json.loads(record["refused_block"])["objective"]["quantity"] == "second_marker"
    assert eng._search_refused_block(record["sha"]) == record["refused_block"]
    assert eng._search_refused_block("another plan") == ""


@pytest.mark.asyncio
async def test_a_block_longer_than_the_bound_is_cut_and_the_ask_says_the_bound(tmp_path: Path) -> None:
    import core.engine as engine

    long = copy.deepcopy(BASE)
    long["constraints"] = [{"quantity": "loss", "max": 3}]
    long["objective"] = {"quantity": "long_marker " + "x" * 20000, "direction": "maximise"}
    eng, asked, _paused = _setup(tmp_path, [_reply(long), _reply(long)])
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    record = json.loads((eng.fi_dir / "search_block_asked.json").read_text(encoding="utf-8"))
    assert len(record["refused_block"]) <= engine._REFUSED_BLOCK_CHARS
    assert "long_marker" not in asked[1].split("# What is asked")[1], "the block is not sent back, only the faulted path"
    assert "`protocol.optimisation.constraints[1]`" in asked[1]
