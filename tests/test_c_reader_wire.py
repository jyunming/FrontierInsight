"""The engine reads a model's answer part by part: the audit's change (one ask naming only the faulted paths, the plan's own
of a part the study cannot do without put back at that part alone) and a search answer (the readable parts used, the
faulted ones left out alone, an objective or a limit that cannot be read still refused). Fake model only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import answer_reading as reading
from core import plan
from core.engine import _entries_as_written, _read_a_search_block
from core import optimisation_plan as optim
from tests.test_plan_optimisation import BLOCK, HEAT_SINK, _engine

TOPIC = "Find the fin spacing and thickness that minimise the base temperature of a heat sink"


def _design(**block_changes: Any) -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    design["protocol"]["optimisation"].update(block_changes)
    return design


def _amended(limit: Any, starts: int = 7) -> dict[str, Any]:
    """The plan's design with its first limit written in a form FI cannot read and, beside it, a readable budget change."""
    design = _design()
    block = design["protocol"]["optimisation"]
    block["constraints"][0] = {**block["constraints"][0], "limit": limit}
    block["evaluation_budget"] = {**block["evaluation_budget"], "starts": starts}
    return design


def _asks(eng: Any, answers: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []

    async def chat_json(prompt: str, *, node: str, want: tuple[str, ...], usable: Any = None, why_unusable: Any = None):
        seen.append(prompt)
        reply = answers[min(len(seen), len(answers)) - 1]
        return json.dumps(reply), reply

    eng._chat_json = chat_json
    return seen


# -- core.answer_reading.put_at ----------------------------------------------------------------------------------------

def test_put_at_places_a_value_at_a_path_and_leaves_the_original() -> None:
    answer = {"protocol": {"metrics": [{"id": "a"}, {"id": "b"}], "optimisation": {"budget": {"starts": 2}}}}
    out = reading.put_at(answer, "protocol.metrics[2]", {"id": "c"}, "protocol")
    assert out is None, "the root is the answer's first step: a path that does not start with it is not placed"
    out = reading.put_at(answer["protocol"], "protocol.metrics[2]", {"id": "c"}, "protocol")
    assert out["metrics"] == [{"id": "a"}, {"id": "c"}] and answer["protocol"]["metrics"][1] == {"id": "b"}
    assert reading.put_at(answer["protocol"], "protocol.optimisation.budget.starts", 9, "protocol")["optimisation"]["budget"] == {"starts": 9}
    assert reading.put_at(answer["protocol"], "protocol.metrics[3]", {}, "protocol") is None, "no third entry"
    assert reading.put_at(answer["protocol"], "protocol.nothing.here", 1, "protocol") is None


# -- the audit's change (Engine._hold_audit_repairs_to_the_schema, _ask_the_audit_to_correct_parts) ---------------------

def test_an_unreadable_limit_of_the_audit_keeps_the_plans_own_limit_and_the_readable_change(tmp_path: Path) -> None:
    """The audit's block has a limit FI cannot read and a readable budget change. Before: the limit was left out of the
    search (two limits left of three) with nothing said of the plan's own. Now: the plan's own limit stays at its place, the
    budget change is made, and the item names the path of the limit."""
    eng = _engine(tmp_path, [])
    design, not_applied = eng._hold_audit_repairs_to_the_schema(_design(), _amended("== 1"))
    block = design["protocol"]["optimisation"]
    assert [c["limit"] for c in block["constraints"]] == ["<= 120", "<= 25"], "the plan's own limit stays"
    assert block["evaluation_budget"]["starts"] == 7, "the readable change is made"
    assert [n["key"] for n in not_applied] == ["protocol.optimisation"] and not_applied[0].get("applied")
    assert [f["path"] for f in not_applied[0]["faults"]] == ["protocol.optimisation.constraints[1]"]


def test_known_positive_a_limit_that_reads_is_made_without_a_question(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design, not_applied = eng._hold_audit_repairs_to_the_schema(_design(), _amended("<= 150"))
    assert design["protocol"]["optimisation"]["constraints"][0]["limit"] == "<= 150" and not not_applied


def test_a_different_unreadable_value_where_the_plan_had_one_is_the_audits_own_fault(tmp_path: Path) -> None:
    """The plan's own limit is unreadable (`== 1`); the audit rewrites it as another unreadable one (`== 2`) beside a
    readable budget change: the new fault is the audit's, so it is named and asked about, and the plan's own stays."""
    eng = _engine(tmp_path, [])
    before = _amended("== 1", starts=4)
    design, not_applied = eng._hold_audit_repairs_to_the_schema(before, _amended("== 2", starts=7))
    assert design["protocol"]["optimisation"]["evaluation_budget"]["starts"] == 4, "no readable limit of the plan's to put back"
    assert [f["path"] for n in not_applied for f in n["faults"]] == ["protocol.optimisation.constraints[1]"]
    other_field = _amended("== 1", starts=7)
    other_field["protocol"]["optimisation"]["constraints"][0]["unit"] = "kg"  # the same unreadable limit, another field changed
    unchanged, none_applied = eng._hold_audit_repairs_to_the_schema(before, other_field)
    assert unchanged["protocol"]["optimisation"]["constraints"][0]["unit"] == "kg" and not none_applied
    unchanged, none_applied = eng._hold_audit_repairs_to_the_schema(before, _amended("== 1", starts=7))
    assert unchanged["protocol"]["optimisation"]["evaluation_budget"]["starts"] == 7 and not none_applied, (
        "a fault the audit did not touch is the plan's own, not the audit's")


@pytest.mark.asyncio
async def test_the_audit_is_asked_once_about_the_faulted_path_alone_and_its_answer_is_placed_there(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    before, amended = _design(), _amended("== 1")
    design, not_applied = eng._hold_audit_repairs_to_the_schema(copy.deepcopy(before), amended)
    asked = _asks(eng, [{"fixes": {"protocol.optimisation.constraints[1]": {"quantity": "mass_g", "limit": "<= 130", "unit": "g"}}}])
    design, not_applied = await eng._ask_the_audit_to_correct_parts(before, amended, design, not_applied, "design_self_critique")
    assert len(asked) == 1, "one ask"
    assert "`protocol.optimisation.constraints[1]`" in asked[0] and "== 1" in asked[0], "the path and the entry as written"
    assert "evaluation_budget" not in asked[0], "nothing that read is asked about"
    block = design["protocol"]["optimisation"]
    assert block["constraints"][0]["limit"] == "<= 130" and block["evaluation_budget"]["starts"] == 7
    assert not not_applied, "read in full now"
    record = json.loads((tmp_path / "out").rglob("audit_readings.json").__next__().read_text(encoding="utf-8"))
    assert record[0]["asked"] == ["protocol.optimisation.constraints[1]"] and record[0]["dropped"] == []


@pytest.mark.asyncio
async def test_known_positive_an_answer_that_does_not_fix_it_leaves_the_plans_own_and_is_recorded(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    before, amended = _design(), _amended("== 1")
    design, not_applied = eng._hold_audit_repairs_to_the_schema(copy.deepcopy(before), amended)
    asked = _asks(eng, [{"fixes": {"protocol.optimisation.constraints[1]": {"quantity": "mass_g", "limit": "== 2"}}}])
    design, not_applied = await eng._ask_the_audit_to_correct_parts(before, amended, design, not_applied, "design_self_critique")
    assert len(asked) == 1
    block = design["protocol"]["optimisation"]
    assert block["constraints"][0]["limit"] == "<= 120" and block["evaluation_budget"]["starts"] == 7
    assert not_applied and not_applied[0]["faults"][0]["path"] == "protocol.optimisation.constraints[1]"
    record = json.loads((tmp_path / "out").rglob("audit_readings.json").__next__().read_text(encoding="utf-8"))
    assert record[0]["dropped"] == ["protocol.optimisation.constraints[1]"]


@pytest.mark.asyncio
async def test_a_failed_ask_leaves_the_first_reading_and_still_records_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    before, amended = _design(), _amended("== 1")
    design, not_applied = eng._hold_audit_repairs_to_the_schema(copy.deepcopy(before), amended)

    async def broken(*_a: Any, **_k: Any):
        raise RuntimeError("the call failed")

    eng._chat_json = broken
    again, na = await eng._ask_the_audit_to_correct_parts(before, amended, design, not_applied, "design_self_critique")
    assert again == design and na == not_applied
    record = json.loads((tmp_path / "out").rglob("audit_readings.json").__next__().read_text(encoding="utf-8"))
    assert record[0]["answered"] is False


# -- a search answer (Engine._keep_only_the_search) and the entries as written -------------------------------------------

def _plan_text() -> str:
    """A plan that has no search block yet (the answer is the whole of it)."""
    design = _design()
    design["protocol"].pop("optimisation")
    return plan.render(TOPIC, {}, design)


def test_a_search_answer_with_an_unreadable_scan_axis_is_used_in_part(tmp_path: Path) -> None:
    """Before: the whole block was refused for one scan axis that cannot be read (and asked for again). Now: the block is
    used, the axis is left out alone and said in the plan."""
    eng = _engine(tmp_path, [])
    block = copy.deepcopy(BLOCK)
    block["grid"] = {"fin_spacing": [2.0, 3.0, 4.0], "planted_axis": [1, 2]}
    text = eng._keep_only_the_search(_plan_text(), {"study_type": "find_best_design", "protocol": {"optimisation": block}})
    kept = plan.raw_design_block(text)["protocol"]["optimisation"]
    assert kept["objective"] == BLOCK["objective"] and kept["evaluation_budget"] == BLOCK["evaluation_budget"]
    assert "planted_axis" not in (kept.get("grid") or {}) and "fin_spacing" in (kept.get("grid") or {})
    assert "planted_axis" in text, "the plan says what was left out"


@pytest.mark.parametrize("change", [
    lambda b: b.__setitem__("objective", "make it better"),
    lambda b: b["constraints"].__setitem__(0, "keep it small"),
    lambda b: b.__setitem__("baseline", {"values": {"fin_spacing": "wide"}, "source": "x"}),
])
def test_known_positive_what_a_search_cannot_do_without_is_still_refused(tmp_path: Path, change: Any) -> None:
    eng = _engine(tmp_path, [])
    block = copy.deepcopy(BLOCK)
    change(block)
    with pytest.raises(ValueError, match="cannot be read"):
        eng._keep_only_the_search(_plan_text(), {"study_type": "find_best_design", "protocol": {"optimisation": block}})


def test_known_positive_a_scans_parts_all_unreadable_are_not_dropped_into_a_larger_scan(tmp_path: Path) -> None:
    """Every part of the scan cannot be read: leaving them all out would run the whole grid, a different and larger scan,
    so the answer is asked again, as for an objective. One part of several that cannot be read is left out alone."""
    block = copy.deepcopy(BLOCK)
    block["grid"] = {"fin_spacing": [2.0, 3.0], "fin_count": [10, 20]}
    block["fixed"] = {**block["fixed"]}
    two = [{"name": "spacing", "vary": ["fin_spacing"], "hold": {"fin_count": 10}},
           {"name": "count", "vary": ["fin_count"], "hold": {"fin_spacing": 2.0}}]
    taken, left_out, stopping = _read_a_search_block({**block, "sub_studies": [two[0], two[1], "an unreadable part"]})
    assert taken is not None and [s["name"] for s in taken["sub_studies"]] == ["spacing", "count"] and left_out
    taken, _left_out, stopping = _read_a_search_block({**block, "sub_studies": ["one unreadable part", "another"]})
    assert taken is None and stopping


def test_a_block_that_reads_whole_is_taken_as_written() -> None:
    block, left_out, stopping = _read_a_search_block(copy.deepcopy(BLOCK))
    assert block == BLOCK and not left_out and not stopping


def test_the_entries_as_written_are_found_by_the_faults_paths() -> None:
    protocol = {"optimisation": {**copy.deepcopy(BLOCK), "constraints": [{"quantity": "mass_g", "limit": "<= 120"}, "keep it small"]}}
    faults = optim.block_faults(protocol["optimisation"])[1]
    assert faults and faults[0].path == "protocol.optimisation.constraints[2]"
    assert _entries_as_written(protocol, faults) == 'the entry as written: "keep it small"'
    assert _entries_as_written(protocol, faults, you=True) == 'the entry as you wrote it: "keep it small"'
    assert _entries_as_written(protocol, []) == ""
