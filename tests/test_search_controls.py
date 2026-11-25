"""A control of a search for the best design is judged against the search's own coarse scan (core/controls.py), and the
merge of a search block into a plan leaves out a control the scan does not have, with a note, instead of refusing the
whole block. Fake model only; the study is made up (a heat sink)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

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
