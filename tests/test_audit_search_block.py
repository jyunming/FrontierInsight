"""The methodology audit may change how a search for the best design is run (the design to beat, the variables and their
ranges, the search and its settings) but not what a design is judged by (the objective's quantity, unit and direction, the
limits, the margin of "better", the target)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import audit_open_items as ao
from core import plan
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine
from tests.test_plan_optimisation import HEAT_SINK

NEW_BASELINE = {"values": {"fin_spacing": 2.0, "fin_thickness": 0.8, "fin_count": 20}, "source": "a denser standard design"}


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="heat sink search", title="hs", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=[])})()
    return eng


def _search_design() -> dict:
    design, why = plan.normalize_design(copy.deepcopy(HEAT_SINK))
    assert why is None
    return design


def _amend(**changes) -> dict:
    amended = copy.deepcopy(_search_design())
    amended["protocol"]["optimisation"].update(changes)
    return amended


def test_how_the_search_is_run_is_taken_from_the_audit(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    before = _search_design()
    amended = _amend(baseline=NEW_BASELINE, evaluation_budget={"starts": 6, "per_start": 80}, search_method="global_then_local")
    amended["protocol"]["optimisation"]["design_variables"][0]["high"] = 8.0
    got, changed = eng._keep_what_the_results_are_judged_by(before, amended)
    block = got["protocol"]["optimisation"]
    assert block["baseline"] == NEW_BASELINE and block["evaluation_budget"]["starts"] == 6
    assert block["search_method"] == "global_then_local" and block["design_variables"][0]["high"] == 8.0
    assert changed is False


def test_what_a_design_is_judged_by_is_put_back_and_reported() -> None:
    """Known positive: the checks still fire on a change to the objective, the limits, the margin or the target."""
    eng = _engine(Path("."))
    before = _search_design()
    judged = {
        "objective": {**before["protocol"]["optimisation"]["objective"], "direction": "maximise", "quantity": "mass_g"},
        "constraints": [{"quantity": "mass_g", "limit": "<= 400", "unit": "g"}],
        "improvement_tolerance": {"value": 10.0, "mode": "absolute"},
        "target": 300.0,
    }
    amended = _amend(baseline=NEW_BASELINE, **judged)
    got, changed = eng._keep_what_the_results_are_judged_by(before, amended)
    block, old = got["protocol"]["optimisation"], before["protocol"]["optimisation"]
    assert changed is True
    assert block["objective"] == old["objective"] and block["constraints"] == old["constraints"]
    assert "improvement_tolerance" not in block and "target" not in block
    assert block["baseline"] == NEW_BASELINE  # the part of the same change that is how the study is run stays


def test_a_new_description_of_the_objective_is_not_a_change_of_what_it_is() -> None:
    eng = _engine(Path("."))
    before = _search_design()
    amended = copy.deepcopy(before)
    amended["protocol"]["optimisation"]["objective"]["meaning"] = "the hottest point of the base plate, as simulated"
    got, changed = eng._keep_what_the_results_are_judged_by(before, amended)
    assert got["protocol"]["optimisation"]["objective"]["meaning"].startswith("the hottest point") and changed is False


def test_an_audit_cannot_turn_a_measurement_into_a_search() -> None:
    eng = _engine(Path("."))
    before = {"hypothesis": "h", "protocol": {"grid": {"dt": [0.1, 0.01]}}}
    amended = {"hypothesis": "h", "protocol": {"grid": {"dt": [0.1, 0.01]}, "optimisation": copy.deepcopy(HEAT_SINK["protocol"]["optimisation"])}}
    got, _changed = eng._keep_what_the_results_are_judged_by(before, amended)
    assert "optimisation" not in got["protocol"]


@pytest.mark.parametrize("path,expected", [
    ("protocol.optimisation.baseline", False), ("protocol.optimisation.evaluation_budget.starts", False),
    ("protocol.optimisation", False), ("protocol.optimisation.constraints", True),
    ("protocol.optimisation.improvement_tolerance.value", True), ("protocol.optimisation.objective", True),
    ("protocol.optimisation.objective.direction", True), ("protocol.optimisation.objective.meaning", False),
    ("protocol.thresholds", True), ("protocol.grid", False), ("method", False),
])
def test_which_paths_the_audit_may_not_change(path: str, expected: bool) -> None:
    assert _engine(Path("."))._audit_may_not_change(path) is expected


@pytest.mark.asyncio
async def test_the_audits_baseline_is_in_the_design_that_runs_and_a_changed_limit_is_an_open_item(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    design = _search_design()
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("heat sink search", {}, design, []), encoding="utf-8")
    plan.record_version(eng.quest_root, path.read_text(encoding="utf-8"), by="engine", note="test")
    design = plan.parse(path.read_text(encoding="utf-8")).design
    change = {"check": "weak_baseline", "where": "protocol.optimisation.baseline", "objection": "the baseline cannot reach the best case",
              "fix": "use a denser one",
              "changes": [{"path": "protocol.optimisation.baseline", "value": NEW_BASELINE},
                          {"path": "protocol.optimisation.constraints",
                           "value": [{"quantity": "mass_g", "limit": "<= 400", "unit": "g"}]}]}
    eng._client.chat.side_effect = [json.dumps({"objections_addressed": [change]}), json.dumps({"objections_addressed": []})]
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    block = runs["protocol"]["optimisation"]
    assert block["baseline"] == NEW_BASELINE
    assert block["constraints"] == design["protocol"]["optimisation"]["constraints"]
    record = ao.read(eng.quest_root)
    assert [x["path"] for x in record["open"]] == ["protocol.optimisation.constraints"]
    assert "judged by" in record["open"][0]["reason"]
    assert [x["path"] for x in record["applied"]] == ["protocol.optimisation.baseline"]


def _with(block_changes: dict) -> dict:
    design = _search_design()
    design["protocol"]["optimisation"].update(block_changes)
    return design


def test_the_conditions_the_cases_and_the_numerical_settings_are_judged_by_and_put_back() -> None:
    eng = _engine(Path("."))
    before = _with({"in_each_run": {"load_case": ["low", "nominal", "high"]}})
    amended = copy.deepcopy(before)
    block = amended["protocol"]["optimisation"]
    block["fixed"] = {"heat_load_W": 80, "air_speed_m_s": 2.0}
    block["in_each_run"] = {"load_case": ["low", "nominal"]}
    block["numerical_settings"] = {"mesh_size_mm": {"search": 0.5, "check": [0.45]}}
    block["baseline"] = NEW_BASELINE
    got, changed = eng._keep_what_the_results_are_judged_by(before, amended)
    kept, old = got["protocol"]["optimisation"], before["protocol"]["optimisation"]
    assert changed is True
    assert kept["fixed"] == old["fixed"] and kept["in_each_run"] == old["in_each_run"]
    assert kept["numerical_settings"] == old["numerical_settings"] and kept["baseline"] == NEW_BASELINE


def test_runs_of_the_check_are_compared_as_they_are_read_not_as_written() -> None:
    """No `check_runs` is written: the check makes `runs_per_evaluation` runs. Cutting that to 1 cuts the check's runs."""
    eng = _engine(Path("."))
    before = _with({"runs_per_evaluation": 8})
    cut = _with({"runs_per_evaluation": 1})
    got, changed = eng._keep_what_the_results_are_judged_by(before, cut)
    assert changed is True and got["protocol"]["optimisation"]["runs_per_evaluation"] == 8
    # the same runs of the check, written in a different place, are not a change
    same = _with({"runs_per_evaluation": 2, "check_runs": 8})
    got, changed = eng._keep_what_the_results_are_judged_by(before, same)
    assert changed is False and got["protocol"]["optimisation"]["runs_per_evaluation"] == 2
    # an explicit `check_runs` cut is still put back
    explicit = _with({"runs_per_evaluation": 8, "check_runs": 8})
    got, changed = eng._keep_what_the_results_are_judged_by(explicit, _with({"runs_per_evaluation": 8, "check_runs": 1}))
    assert changed is True and got["protocol"]["optimisation"]["check_runs"] == 8


@pytest.mark.asyncio
async def test_a_whole_block_proposal_is_kept_per_part(tmp_path: Path) -> None:
    """The audit replaces the whole block: the baseline is applied (and is not listed as open), the objective's direction
    is put back and the open item names that part and why, not the block."""
    eng = _engine(tmp_path)
    design = _search_design()
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("heat sink search", {}, design, []), encoding="utf-8")
    plan.record_version(eng.quest_root, path.read_text(encoding="utf-8"), by="engine", note="test")
    design = plan.parse(path.read_text(encoding="utf-8")).design
    block = copy.deepcopy(design["protocol"]["optimisation"])
    block["baseline"] = NEW_BASELINE
    block["objective"]["direction"] = "maximise"
    change = {"check": "weak_baseline", "where": "protocol.optimisation", "objection": "the baseline and the goal",
              "fix": "replace the block", "changes": [{"path": "protocol.optimisation", "value": block}]}
    eng._client.chat.side_effect = [json.dumps({"objections_addressed": [change]}), json.dumps({"objections_addressed": []})]
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs["protocol"]["optimisation"]["baseline"] == NEW_BASELINE
    assert runs["protocol"]["optimisation"]["objective"]["direction"] == "minimise"
    record = ao.read(eng.quest_root)
    assert [x["path"] for x in record["open"]] == ["protocol.optimisation.objective.direction"]
    assert "judged by" in record["open"][0]["reason"]
    assert "protocol.optimisation.baseline" in [x["path"] for x in record["applied"]]


def test_with_check_runs_written_only_it_is_put_back() -> None:
    """`runs_per_evaluation` is then the search's own replication: it may change, the check's runs may not."""
    eng = _engine(Path("."))
    before = _with({"runs_per_evaluation": 8, "check_runs": 8})
    got, changed = eng._keep_what_the_results_are_judged_by(before, _with({"runs_per_evaluation": 1, "check_runs": 1}))
    block = got["protocol"]["optimisation"]
    assert changed is True and block["runs_per_evaluation"] == 1 and block["check_runs"] == 8
    # check_runs absent: the default comes from runs_per_evaluation, so that is put back and an added check_runs removed
    got, changed = eng._keep_what_the_results_are_judged_by(_with({"runs_per_evaluation": 8}),
                                                            _with({"runs_per_evaluation": 1, "check_runs": 1}))
    block = got["protocol"]["optimisation"]
    assert changed is True and block["runs_per_evaluation"] == 8 and "check_runs" not in block


@pytest.mark.asyncio
async def test_a_proposal_that_is_not_a_search_block_is_kept_open_with_that_reason(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    design = _search_design()
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("heat sink search", {}, design, []), encoding="utf-8")
    plan.record_version(eng.quest_root, path.read_text(encoding="utf-8"), by="engine", note="test")
    design = plan.parse(path.read_text(encoding="utf-8")).design
    change = {"check": "weak_baseline", "where": "protocol.optimisation", "objection": "use another search",
              "fix": "replace it", "changes": [{"path": "protocol.optimisation", "value": "a gradient search"}]}
    eng._client.chat.side_effect = [json.dumps({"objections_addressed": [change]}), json.dumps({"objections_addressed": []})]
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs["protocol"]["optimisation"] == design["protocol"]["optimisation"]
    item = ao.read(eng.quest_root)["open"][0]
    assert item["path"] == "protocol.optimisation"
    assert "not a valid search block" in item["reason"] and "must be a mapping" in item["reason"]
    assert "kept as it was" in item["reason"]
