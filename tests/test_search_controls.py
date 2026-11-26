"""A control of a search for the best design is judged against the search's own coarse scan (core/controls.py), and the
merge of a search block into a plan leaves out a control the scan does not have, with a note, instead of refusing the
whole block. Fake model only; the study is made up (a heat sink)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import controls as ctl
from core import plan
from core.engine import Engine  # noqa: F401
from tests.test_plan_optimisation import BLOCK, HEAT_SINK, _engine

METRIC = {"id": "base_rise", "kind": "mean", "estimand": "the base temperature rise", "unit": "a run"}
CONTROL = {"name": "still air", "setting": {"air_speed_m_s": 2.0}, "quantity": "base_rise", "expected": 10, "tolerance": 1}


def _block(**scan: Any) -> dict[str, Any]:
    block = copy.deepcopy(BLOCK)
    block["grid"] = scan or {"air_speed_m_s": [1.0, 2.0, 3.0]}
    return block


def _protocol(block: dict[str, Any], controls: list[Any]) -> dict[str, Any]:
    return {"optimisation": block, "metrics": [dict(METRIC)], "controls": controls}


def test_a_control_of_a_search_is_judged_against_the_searchs_scan() -> None:
    """Before: the control was judged against `protocol.grid`, which a search never has, so every control of a search
    plan was refused ("the grid is empty")."""
    protocol = _protocol(_block(), [CONTROL])
    assert ctl.entry_fault(1, CONTROL, protocol) is None
    assert ctl.of_protocol(protocol) == [{**CONTROL, "name": "still air", "tolerance_mode": "absolute"}]


def test_known_positive_a_control_naming_a_setting_the_scan_does_not_have_is_still_refused() -> None:
    protocol = _protocol(_block(), [{**CONTROL, "setting": {"no_such_setting": 1}}])
    why = ctl.entry_fault(1, protocol["controls"][0], protocol)
    assert why is not None and "`no_such_setting` is not a setting of `protocol.optimisation.grid` (air_speed_m_s)" in why
    wrong_value = {**CONTROL, "setting": {"air_speed_m_s": 9.0}}
    assert "not one of its values in `protocol.optimisation.grid`" in ctl.entry_fault(1, wrong_value, protocol)
    kept, notes = plan.repair_protocol(_protocol(_block(), [CONTROL, protocol["controls"][0]]))
    assert [c["name"] for c in kept["controls"]] == ["still air"] and len(notes) == 1


def test_a_measurements_control_is_still_judged_against_its_grid() -> None:
    protocol = {"grid": {"lid": ["on", "off"]}, "metrics": [dict(METRIC)]}
    assert ctl.entry_fault(1, {**CONTROL, "setting": {"lid": "on"}}, protocol) is None
    assert "`protocol.grid`" in ctl.entry_fault(1, CONTROL, protocol)


def _plan_without_a_block(controls: list[Any]) -> str:
    design = copy.deepcopy(HEAT_SINK)
    design["protocol"] = {"grid": {"air_speed_m_s": [1.0, 2.0, 3.0]}, "runs_per_setting": 3, "metrics": [dict(METRIC)],
                          "controls": controls}
    design["study_type"] = "find_best_design"
    return plan.render("Find the fin spacing that minimises the base temperature of a heat sink", {}, design)


def test_a_search_block_is_merged_when_a_control_names_a_setting_the_scan_leaves_out(tmp_path: Path) -> None:
    """Before: the merge removed `protocol.grid`, the strict reader found the control's setting in no grid and refused
    the whole plan with the block in it (asked again, refused again). Now the control is left out, with a note."""
    eng = _engine(tmp_path, [])
    before = _plan_without_a_block([{**CONTROL, "setting": {"air_speed_m_s": 2.0}}, ])
    block = _block(heat_load_W=[30, 40])  # the scan the answer writes has no `air_speed_m_s`
    block["fixed"] = {"heat_load_W": 40, "air_speed_m_s": 2.0}
    text = eng._keep_only_the_search(before, {"study_type": "find_best_design", "protocol": {"optimisation": block}})
    design = plan.parse(text).design
    assert design is not None and design["protocol"]["optimisation"]["grid"] == {"heat_load_W": [30, 40]}
    assert "controls" not in design["protocol"], "the control the scan cannot pick is left out"
    assert "the control 'still air' was left out of `protocol.controls`" in text, "and the plan says so"


def test_a_control_the_scan_has_is_kept_by_the_merge(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    before = _plan_without_a_block([CONTROL])
    text = eng._keep_only_the_search(before, {"study_type": "find_best_design", "protocol": {"optimisation": _block()}})
    design = plan.parse(text).design
    assert design is not None and [c["name"] for c in design["protocol"]["controls"]] == ["still air"]
    assert "was left out of `protocol.controls`" not in text


def _held_plan(controls: list[Any]) -> str:
    design = copy.deepcopy(HEAT_SINK)
    design["protocol"] = {"grid": {"air_speed_m_s": [1.0, 2.0, 3.0]}, "runs_per_setting": 3,
                          "metrics": [{**METRIC, "reference": {"formula": "10", "when": {"air_speed_m_s": 2.0}}}],
                          "controls": controls}
    design["study_type"] = "find_best_design"
    return plan.render("Find the fin spacing that minimises the base temperature of a heat sink", {}, design)


def test_what_the_scan_lacked_comes_back_when_the_scan_setting_is_asked_back(tmp_path: Path) -> None:
    """Before: the control (and the metric's reference) were left out at the first merge and stayed out when the plan's model
    gave the setting back in its next answer. Now they are held aside and judged again at each merge."""
    eng = _engine(tmp_path, [])
    path = plan.plan_path(eng.quest_root)
    first = eng._keep_only_the_search(_held_plan([CONTROL]), {"study_type": "find_best_design", "protocol": {
        "optimisation": {**_block(heat_load_W=[30, 40], air_speed_m_s=[1.0, 2.0, 3.0]), "fixed": {"heat_load_W": 40}}}})
    design = plan.parse(first).design
    assert "controls" not in design["protocol"] and "reference" not in design["protocol"]["metrics"][0]
    assert "the control 'still air' was left out" in first and "the reference value of `base_rise` was left out" in first
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(first, encoding="utf-8")
    eng._commit_held(first)  # the plan is written: what the merge left out is kept beside it
    second = eng._keep_only_the_search(first, {"study_type": "find_best_design", "protocol": {
        "optimisation": {**_block(heat_load_W=[30, 40], air_speed_m_s=[1.0, 2.0, 3.0]),
                         "fixed": {"heat_load_W": 40, "air_speed_m_s": 2.0}}}})
    design = plan.parse(second).design
    assert [c["name"] for c in design["protocol"]["controls"]] == ["still air"]
    assert design["protocol"]["metrics"][0]["reference"]["formula"] == "10"
    assert "the control 'still air' was left out" not in second and "the reference value of `base_rise` was left out" not in second


ANSWER = {"study_type": "find_best_design", "protocol": {"optimisation": {
    **_block(heat_load_W=[30, 40], air_speed_m_s=[1.0, 2.0, 3.0]), "fixed": {"heat_load_W": 40, "air_speed_m_s": 2.0}}}}
FIRST = {"study_type": "find_best_design", "protocol": {"optimisation": {
    **_block(heat_load_W=[30, 40], air_speed_m_s=[1.0, 2.0, 3.0]), "fixed": {"heat_load_W": 40}}}}


def _first_merge(eng: Any) -> str:
    first = eng._keep_only_the_search(_held_plan([CONTROL]), FIRST)
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(first, encoding="utf-8")
    eng._commit_held(first)
    return first


def test_a_control_a_person_removed_by_hand_is_not_put_back_by_a_later_merge(tmp_path: Path) -> None:
    """Before: the held control came back at the next merge whatever had happened to the plan since."""
    eng = _engine(tmp_path, [])
    first = _first_merge(eng)
    edited = first.replace("hypothesis:", "hypothesis: (reworded by a person)", 1)
    assert edited != first
    second = eng._keep_only_the_search(edited, ANSWER)
    assert "controls" not in plan.parse(second).design["protocol"]


@pytest.mark.asyncio
async def test_a_refused_plan_write_leaves_the_held_entries_as_they_were(tmp_path: Path) -> None:
    """Before: the held record was replaced while the merge was built, so a write the hand edit refused lost the control."""
    from core.engine_base import _PlanEditedMeanwhile

    eng = _engine(tmp_path, [])
    first = _first_merge(eng)
    held_file = eng.fi_dir / "held_beside_the_scan.json"
    before = held_file.read_bytes()
    path = plan.plan_path(eng.quest_root)

    async def answer(prompt: str) -> dict[str, Any]:
        path.write_text(first + chr(10) + "A person's note." + chr(10), encoding="utf-8")  # saved while the model was answering
        return ANSWER

    eng._ask_for_plan_parts = answer  # type: ignore[method-assign]
    with pytest.raises(_PlanEditedMeanwhile):
        await eng._complete_plan_parts("ask", path, expect=first, keep=eng._keep_only_the_search, note="n")
    assert held_file.read_bytes() == before


def test_a_plan_with_a_broken_part_beside_a_valid_block_is_still_refused(tmp_path: Path) -> None:
    """Before: the merge read the whole protocol leniently and left the broken threshold out without a refusal."""
    from core.engine_base import _PlanBeyondTheBlock

    eng = _engine(tmp_path, [])
    broken = plan.edit_design_block(_held_plan([CONTROL]), lambda d: {
        **d, "protocol": {**d["protocol"], "thresholds": {"limit_x": "not a number"}}})
    assert broken is not None
    with pytest.raises(_PlanBeyondTheBlock):
        eng._keep_only_the_search(broken, ANSWER)


def test_the_stop_the_search_check_wrote_is_removed_when_the_search_is_ready_and_no_other_stop_is(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    stuck = eng.quest_root / "needs" / "STUCK.json"
    stuck.parent.mkdir(parents=True, exist_ok=True)
    eng._search_ready = lambda state, design: {"status": "not_ready", "detail": "the plan lacks its search", "tried": []}  # type: ignore[method-assign]
    eng._judge_the_search_the_design_hands_on({"design": {}}, stuck)
    assert json.loads(stuck.read_text(encoding="utf-8"))["source"] == "search_ready"
    eng._search_ready = lambda state, design: {"status": "ready"}  # type: ignore[method-assign]
    eng._judge_the_search_the_design_hands_on({"design": {}}, stuck)
    assert not stuck.exists(), "the plan is complete now: its own earlier stop goes"
    stuck.write_text(json.dumps({"kind": "returned", "return_reason": "known_value_mismatch"}), encoding="utf-8")
    eng._judge_the_search_the_design_hands_on({"design": {}}, stuck)
    assert stuck.exists(), "a stop of any other kind is never removed"
