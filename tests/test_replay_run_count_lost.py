"""Replay of a real quest that ran one trial per setting where its plan said 2500, and then delivered a paper.

In tests/fixtures/replay/run_count_lost/ the plan's protocol (``plan_protocol.yaml``) has ``runs_per_setting: 2500``, the
redesign that came after the first results (``redesign_protocol.json``) wrote only a grid, metrics, a model and a topic map,
the protocol FI froze from it (``frozen_protocol_summary.json``) had no run count, and ``study.json`` and the engine's own
run took ``or 1``. The analysis script (``experiment.py``) hard-codes 1000 trials; its answer to one trial per setting is
``result_json.json``: no result for any quantity of the study, but a number in it (``required_trials_per_cell``), which was
enough for the engine to call it a result. No model is called."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from core import code_project as cp
from core import plan
from tests.test_code_review import DESIGN, _engine

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "run_count_lost"


def _load(name: str) -> Any:
    text = (FIXTURE / name).read_text(encoding="utf-8")
    return yaml.safe_load(text) if name.endswith(".yaml") else json.loads(text)


def _quest(tmp_path: Path):
    """A quest whose plan.md holds the real plan's protocol (2500 runs per setting)."""
    engine = _engine(tmp_path)
    protocol = _load("plan_protocol.yaml")
    text = plan.render(engine.config.topic, {}, {**DESIGN, "protocol": protocol})
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="model")
    return engine, protocol


def test_the_evidence_is_what_the_real_quest_held() -> None:
    plan_protocol, redesign = _load("plan_protocol.yaml"), _load("redesign_protocol.json")
    frozen = _load("frozen_protocol_summary.json")
    assert plan_protocol["runs_per_setting"] == 2500 and plan_protocol["oracles"]
    assert "runs_per_setting" not in redesign and "oracles" not in redesign
    assert frozen["protocol_keys"] == ["grid", "metrics", "model", "topic_map"]
    assert _load("study.json")["runs_per_setting"] == 1
    assert "invalid_trial_record" in json.dumps(_load("result_json.json"))


def test_before_the_fix_the_redesign_ran_one_trial_per_setting() -> None:
    redesign = _load("redesign_protocol.json")
    assert int(redesign.get("runs_per_setting") or 1) == 1  # what TrialsRunner.execute and study_of did with it


def test_the_redesign_keeps_the_plans_run_count_and_checks(tmp_path: Path) -> None:
    engine, plan_protocol = _quest(tmp_path)
    redesign = _load("redesign_protocol.json")
    design = {**DESIGN, "protocol": redesign}
    state = {"design": design, "iteration": 1}
    kept = engine._inherit_plan_parts(state, design, iteration=1)
    protocol = kept["protocol"]
    assert protocol["runs_per_setting"] == 2500 and protocol["precision"] == plan_protocol["precision"]
    assert [o["name"] for o in protocol["oracles"]] == [o["name"] for o in plan_protocol["oracles"]]
    assert [c["name"] for c in protocol["criteria"]] == [c["name"] for c in plan_protocol["criteria"]]  # (after the oracles they read)
    assert set(plan_protocol) - {"grid", "metrics", "model", "topic_map"} <= set(protocol)  # every other part came across
    # What the redesign wrote itself is untouched.
    assert protocol["grid"] == redesign["grid"] and protocol["metrics"] == redesign["metrics"]
    # The protocol FI would freeze is now the one with the run count.
    assert engine._draft_protocol({**state, "design": kept})["runs_per_setting"] == 2500
    record = json.loads((engine.fi_dir / "drafted_parts_asked.json").read_text(encoding="utf-8"))
    assert "runs_per_setting" in record["inherited"]


def test_a_part_the_redesign_wrote_is_never_replaced(tmp_path: Path) -> None:
    engine, _plan_protocol = _quest(tmp_path)
    redesign = {**_load("redesign_protocol.json"), "runs_per_setting": 40}
    design = {**DESIGN, "protocol": redesign}
    kept = engine._inherit_plan_parts({"design": design, "iteration": 1}, design, iteration=1)
    assert kept["protocol"]["runs_per_setting"] == 40


def test_a_search_for_the_best_design_is_not_given_the_plans_parts_it_does_not_have(tmp_path: Path) -> None:
    engine, _plan_protocol = _quest(tmp_path)
    design = {**DESIGN, "protocol": {"grid": {"x": [1, 2]}}}
    kept = engine._inherit_plan_parts({"design": design, "iteration": 1}, design, iteration=1)
    assert "optimisation" not in kept["protocol"] and "grid" in kept["protocol"]
    assert kept["protocol"]["grid"] == {"x": [1, 2]}


def test_a_study_file_that_differs_from_the_protocol_is_found_and_written_again(tmp_path: Path) -> None:
    engine, plan_protocol = _quest(tmp_path)
    code = engine.quest_root / "code"
    (code / "simulate.py").write_text("def run_trial(cell, trial, seed):\n    return {'x': 1.0}\n", encoding="utf-8")
    # FI wrote this file itself, from the protocol it had then: the redesign's, with no run count (so one run).
    cp.refresh(engine.quest_root, protocol={**plan_protocol, "runs_per_setting": 1})
    assert json.loads((code / "study.json").read_text(encoding="utf-8"))["runs_per_setting"] == _load("study.json")["runs_per_setting"]
    found = cp.study_differences(code, plan_protocol)
    assert found and "2500" in found[0]
    import asyncio

    state = {"design": {**DESIGN, "protocol": plan_protocol}, "iteration": 0, "deps": []}
    asyncio.run(engine._study_follows_protocol(state))
    assert json.loads((code / "study.json").read_text(encoding="utf-8"))["runs_per_setting"] == 2500
    assert cp.study_differences(code, plan_protocol) == []


@pytest.mark.parametrize("protocol", [None, {}, {"grid": {"a": [1, 2]}}])
def test_a_protocol_with_no_run_count_leaves_the_study_file_alone(tmp_path: Path, protocol: Any) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("def run_trial(cell, trial, seed):\n    return {'x': 1.0}\n", encoding="utf-8")
    (code / "study.json").write_text(json.dumps({"entry": "run_trial", "grid": {"a": [1, 2]}, "runs_per_setting": 1}),
                                     encoding="utf-8")
    assert cp.study_differences(code, protocol) == []


# --- an analysis that fixes its own number of runs -------------------------------------------------------------------

def _real_files() -> dict[str, str]:
    return {"experiment.py": (FIXTURE / "experiment.py").read_text(encoding="utf-8"),
            "simulate.py": "def run_trial(cell, trial, seed):\n    return {'x': 1.0}\n"}


def test_an_analysis_that_fixes_a_count_the_protocol_sets_otherwise_is_found() -> None:
    from core import grid_keys

    protocol = _load("plan_protocol.yaml")
    found = grid_keys.problems(_real_files(), protocol)
    counted = [r for r in found if r.planned is not None]
    assert [(r.script, r.keys, r.default, r.planned) for r in counted] == [("experiment.py", ("N_TRIALS",), "1000", 2500)]
    note = grid_keys.plain_note(counted, protocol)
    assert "1000" in note and "2500" in note
    assert "count them" in grid_keys.directive(counted, protocol)
    assert grid_keys.record(counted, protocol)["reads"][0]["protocol_runs"] == 2500


def test_before_the_fix_a_protocol_with_no_run_count_made_the_same_script_pass() -> None:
    from core import grid_keys

    frozen_as_it_was = {"grid": _load("plan_protocol.yaml")["grid"]}  # no runs_per_setting: nothing to compare with
    assert [r for r in grid_keys.problems(_real_files(), frozen_as_it_was) if r.planned is not None] == []


def test_a_count_equal_to_the_protocols_or_with_no_trial_contract_is_not_a_fault() -> None:
    from core import grid_keys

    protocol = {**_load("plan_protocol.yaml"), "runs_per_setting": 1000}
    assert [r for r in grid_keys.problems(_real_files(), protocol) if r.planned is not None] == []
    only_cells = {**_real_files(), "simulate.py": "def run_cell(cell):\n    return {'x': 1.0}\n"}
    assert [r for r in grid_keys.problems(only_cells, _load("plan_protocol.yaml")) if r.planned is not None] == []
    other_names = {"experiment.py": "N_BOOTSTRAP = 2000\nN_REPLICATES = 2000\nN_REPEATS = 5\nBATCH = 7\nrun_label = 3\n",
                   "simulate.py": _real_files()["simulate.py"]}
    assert [r for r in grid_keys.problems(other_names, _load("plan_protocol.yaml")) if r.planned is not None] == []


# --- a result that holds none of the study's quantities ends in the honest stop ------------------------------------------

def _ids() -> list[str]:
    return [m["id"] for m in _load("plan_protocol.yaml")["metrics"]]


def _stuck_state(retries: int = 0, iteration: int = 0, result: Any = None) -> dict[str, Any]:
    return {"design": {**DESIGN, "protocol": _load("plan_protocol.yaml")}, "iteration": iteration,
            "result_json": _load("result_json.json") if result is None else result,
            "exec_result": {"returncode": 0, "duration_s": 2.0}, "evidence_no_result_retries": retries}


def test_before_the_fix_the_failed_result_counted_as_a_result() -> None:
    from core import numeric_oracle

    # The gate's old test: any number in the result is a result. `required_trials_per_cell` is one.
    assert list(numeric_oracle.flatten_numbers(_load("result_json.json"), keep_zero=True))


def test_the_failed_result_goes_back_to_design_once_and_then_stops_without_a_paper(tmp_path: Path) -> None:
    engine, _ = _quest(tmp_path)
    first = engine._no_results_verdict(_stuck_state(retries=0, iteration=0))
    assert first is not None and first.get("redesign") is True and not first.get("stuck")
    second = engine._no_results_verdict(_stuck_state(retries=1, iteration=0))
    assert second is not None and second["stuck"] is True and second["stuck_reason"] == "no_numbers"
    assert "none of the study's quantities" in second["rationale"] and _ids()[0] in second["rationale"]
    # No second design left: the same stop at once.
    assert engine._no_results_verdict(_stuck_state(retries=0, iteration=1))["stuck"] is True


def test_the_stop_writes_the_plain_record_and_no_paper(tmp_path: Path) -> None:
    import asyncio

    engine, _ = _quest(tmp_path)
    (engine.quest_root / "paper.md").write_text("# a paper\n", encoding="utf-8")
    verdict = engine._no_results_verdict(_stuck_state(retries=1))
    state = {**_stuck_state(retries=1), "evidence_assessment": verdict}
    asyncio.run(engine._node_stuck_no_findings(state))
    record = json.loads((engine.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert "none of the study's quantities" in record["problem"] and "No paper was written" in record["say"]
    assert not (engine.quest_root / "paper.md").exists()


def _honest_results() -> list[dict[str, Any]]:
    a, b, c = _ids()[:3]
    return [
        # the quantities are there, under names that contain the plan's
        {"status": "ok", "by_cell": {"x": {f"mean_{a}": 0.8, f"{c}_ci95": [0.01, 0.07], b: 0.04}}},
        # it reports a problem, but the study's quantities are in it all the same
        {"errors": ["one cell was thin"], a: 0.8, b: 0.05, c: 0.04},
        # no word of failure: a result under other names is left to the checks that already read it
        {"required_trials_per_cell": 10, "rate_a": 0.5},
        # `errors` holding error statistics is not a failure
        {"errors": {"max_abs": 0.01}, "rate_a": 0.5},
    ]


@pytest.mark.parametrize("n", range(4))
def test_an_honest_result_is_not_stopped(tmp_path: Path, n: int) -> None:
    result = _honest_results()[n]
    engine, _ = _quest(tmp_path)
    assert engine._no_results_verdict(_stuck_state(retries=1, result=result)) is None


def test_a_count_used_only_to_resample_or_for_seeds_is_not_the_trial_count() -> None:
    from core import grid_keys

    sim = "def run_trial(cell, trial, seed):\n    return {'x': 1.0}\n"
    protocol = {"runs_per_setting": 1000}
    resample = ("import numpy as np\nN_TRIALS = 200\nrng = np.random.default_rng(0)\n"
                "def boot(values):\n    return [rng.choice(values, size=N_TRIALS).mean() for _ in range(N_TRIALS)]\n")
    assert grid_keys.problems({"experiment.py": resample, "simulate.py": sim}, protocol) == []
    checked = "N_TRIALS = 200\ndef check(records):\n    assert len(records) == N_TRIALS\n"
    assert len(grid_keys.problems({"experiment.py": checked, "simulate.py": sim}, protocol)) == 1
    looped = "N_RUNS = 200\ndef go(cell):\n    return [run_trial(cell, i, i) for i in range(N_RUNS)]\n" \
             "def go2(cell):\n    out = []\n    for i in range(N_RUNS):\n        out.append(run_trial(cell, i, i))\n"
    assert len(grid_keys.problems({"experiment.py": looped, "simulate.py": sim}, protocol)) == 1
    passed = "N_TRIALS = 200\nrunner(runs_per_setting=N_TRIALS)\n"
    assert len(grid_keys.problems({"experiment.py": passed, "simulate.py": sim}, protocol)) == 1
