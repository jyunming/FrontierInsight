"""A simulation whose output does not depend on its inputs is broken, not a finding (core/flat_output.py).

A search whose objective is the same at every design it tried, and a measurement whose every number is the same in every
setting of a grid that varies something, are sent back to be repaired as a simulation that does not work, in plain words
and with no expected value in them. Still flat after the repairs: the quest stops with no paper (needs/STUCK.json).
Neutral toy topics only (a heat sink, a cooling cup).
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import flat_output as flat
from core import optimisation_plan as op
from core import optimise, trial_runner
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from core.execution import SharedInterpreterExecutor
from tests.test_optimise_runner import LocalExecutor

# --- the pure readers ------------------------------------------------------------------------------------------------


def _row(n: int, fin: float, objective: Any, constraints: dict[str, Any] | None = None, status: str = "ok") -> dict[str, Any]:
    return {"event": "evaluation", "n": n, "status": status, "design": {"fin": fin, "gap": 2.0 * fin},
            "objective": objective, "constraints": constraints or {}}


def test_a_search_with_one_value_at_every_design_is_flat() -> None:
    rows = [_row(i, 1.0 + i, 0.0) for i in range(6)]
    why = flat.flat_search(rows, "peak_temperature")
    assert why is not None
    assert "does not change when `fin`, `gap` change" in why and "the simulation does not use them" in why
    assert "`peak_temperature` was the same at all 6 designs" in why
    assert "expected" not in why.lower()


def test_a_search_whose_objective_moves_is_not_flat() -> None:
    rows = [_row(i, 1.0 + i, 10.0 - i) for i in range(6)]
    assert flat.flat_search(rows) is None


def test_a_flat_objective_is_broken_even_while_a_limit_quantity_still_moves() -> None:
    rows = [_row(i, 1.0 + i, 3.0, {"mass": 5.0 + i}) for i in range(6)]
    why = flat.flat_search(rows, "peak")
    assert why is not None and "`peak` was the same at all 6 designs" in why and "mass" not in why
    # A moving objective is fine whatever the limits do.
    assert flat.flat_search([_row(i, 1.0 + i, 3.0 + i, {"mass": 5.0}) for i in range(6)]) is None


def test_two_evaluations_or_one_design_prove_nothing() -> None:
    assert flat.flat_search([_row(0, 1.0, 0.0), _row(1, 2.0, 0.0)]) is None
    assert flat.flat_search([_row(i, 1.0, 0.0) for i in range(5)]) is None
    assert flat.flat_search([_row(i, 1.0 + i, 0.0, status="failed") for i in range(5)]) is None


def test_numbers_that_differ_only_by_noise_are_the_same() -> None:
    rows = [_row(i, 1.0 + i, 5.0 + i * 1e-13) for i in range(5)]
    assert flat.flat_search(rows) is not None
    rows = [_row(i, 1.0 + i, 5.0 + i * 1e-3) for i in range(5)]
    assert flat.flat_search(rows) is None


def test_designs_that_differ_only_by_noise_are_one_design() -> None:
    noisy = [{"event": "evaluation", "n": i, "status": "ok", "design": {"fin": 2.0 + i * 1e-13, "gap": 4.0},
              "objective": 7.0 + i * 1e-4, "constraints": {}} for i in range(4)]
    assert flat.flat_search(noisy) is None, "one design, a tiny objective change: nothing to say"
    same_objective = [{**r, "objective": 7.0} for r in noisy]
    assert flat.flat_search(same_objective) is None, "noise-level differences are not distinct designs"
    distinct = [_row(i, 1.0 + i, 7.0) for i in range(3)]
    assert flat.flat_search(distinct) is not None


def _check(no_effect: list[str], sensitivity: dict[str, float] | None = None, finished: bool = True) -> dict[str, Any]:
    return {"finished": finished, "checks": {"neighbourhood": {
        "status": "failed", "no_effect": no_effect, "sensitivity": sensitivity or {n: 0.0 for n in no_effect}}}}


BLOCK = {"design_variables": [{"name": "fin", "low": 0, "high": 5, "kind": "continuous"},
                              {"name": "gap", "low": 0, "high": 5, "kind": "continuous"},
                              {"name": "material", "kind": "choice", "values": ["a", "b"]}]}


def test_the_check_finding_that_every_nudge_changes_nothing_is_flat() -> None:
    why = flat.flat_search_check(_check(["fin", "gap"]), BLOCK)
    assert why is not None and "does not change when `fin`, `gap` change" in why
    # One variable that does matter, or a check that did not finish: not this.
    assert flat.flat_search_check(_check(["fin"], {"fin": 0.0, "gap": 1.5}), BLOCK) is None
    assert flat.flat_search_check(_check(["fin"]), BLOCK) is None
    assert flat.flat_search_check(_check(["fin", "gap"], finished=False), BLOCK) is None
    assert flat.flat_search_check(_check([]), BLOCK) is None


def _cell(cell: dict[str, Any], *values: dict[str, Any]) -> Any:
    return SimpleNamespace(cell=cell, rows=[{"status": "ok", "values": v} for v in values])


def test_a_sweep_with_the_same_numbers_in_every_setting_is_flat() -> None:
    cells = [_cell({"flow": f}, {"t_out": 20.0, "drop": 0.0}, {"t_out": 20.0, "drop": 0.0}) for f in (1.0, 2.0, 3.0)]
    why = flat.flat_sweep(cells)
    assert why is not None and "does not change when `flow` change" in why and "`drop`, `t_out`" in why
    assert "the simulation does not use it" in flat.flat_sweep(cells).replace("them", "it") or "does not use" in why


def test_a_sweep_where_anything_varies_is_not_flat() -> None:
    cells = [_cell({"flow": f}, {"t_out": 20.0 - f, "drop": 0.0}) for f in (1.0, 2.0, 3.0)]
    assert flat.flat_sweep(cells) is None
    # A single setting, or a grid that varies nothing, has nothing to compare.
    assert flat.flat_sweep([_cell({"flow": 1.0}, {"t_out": 20.0})]) is None
    assert flat.flat_sweep([_cell({"flow": 1.0}, {"t_out": 20.0}), _cell({"flow": 1.0}, {"t_out": 20.0})]) is None
    # Words and flags are not measurements.
    assert flat.flat_sweep([_cell({"flow": f}, {"note": "ok"}) for f in (1.0, 2.0)]) is None


def test_trial_noise_that_differs_between_trials_is_not_flat() -> None:
    cells = [_cell({"flow": f}, {"t_out": 20.0 + 0.1}, {"t_out": 20.0 - 0.1}) for f in (1.0, 2.0, 3.0)]
    assert flat.flat_sweep(cells) is None


# --- the runners -----------------------------------------------------------------------------------------------------

ANALYSIS = '''\
import json, os
best = json.load(open(os.environ["FI_BEST_DESIGN"], encoding="utf-8"))
print("RESULT_JSON: " + json.dumps({"evaluated": best["evaluations"]["search"]}))
'''
FLAT_SIM = 'def run_cell(cell):\n    return {"peak": 0.0, "mass": 1.0}\n'
WORKING_SIM = ('def run_cell(cell):\n    return {"peak": (cell["fin"] - 3.0) ** 2 + 2.0 * (cell["gap"] - 1.0) ** 2 + 30.0,'
               ' "mass": cell["fin"]}\n')


def _block(**changes: Any) -> dict[str, Any]:
    block = {
        "objective": {"quantity": "peak", "direction": "minimise", "unit": "K", "meaning": "the peak temperature"},
        "design_variables": [{"name": "fin", "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"},
                             {"name": "gap", "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"}],
        "baseline": {"values": {"fin": 0.0, "gap": 0.0}, "source": "the design in use"},
        "numerical_settings": {"mesh": {"search": 0.4}},
        "evaluation_budget": {"starts": 1, "per_start": 8},
        "search_method": "bounded_local",
    }
    block.update(copy.deepcopy(changes))
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def _quest(tmp_path: Path, simulate: str) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    return root


def _search_runner(root: Path) -> optimise.OptimisationRunner:
    return optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol={"optimisation": _block()},
                                       simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")


@pytest.mark.asyncio
async def test_a_search_over_a_flat_simulation_is_sent_back_before_anything_is_checked_or_analysed(tmp_path: Path) -> None:
    root = _quest(tmp_path, FLAT_SIM)
    runner = _search_runner(root)
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    assert result.returncode == 1 and runner.failed_script == "simulate.py"
    assert runner.flat and "does not change when `fin`, `gap` change" in result.stderr
    assert "RESULT_JSON" not in result.stdout
    assert not (root / "needs" / "OPTIMUM_CHECK.json").exists(), "no check was spent on a result that is not usable"


@pytest.mark.asyncio
async def test_a_working_search_is_unchanged(tmp_path: Path) -> None:
    root = _quest(tmp_path, WORKING_SIM)
    runner = _search_runner(root)
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    assert result.returncode == 0, result.stderr[-1500:]
    assert runner.flat is None and runner.failed_script is None and "RESULT_JSON" in result.stdout
    assert (root / "needs" / "OPTIMUM_CHECK.json").is_file()


SWEEP_ANALYSIS = '''\
import json, os
data = json.load(open(os.environ["FI_TRIALS"], encoding="utf-8"))
print("RESULT_JSON: " + json.dumps({"cells": len(data["cells"])}))
'''


def _sweep_runner(root: Path) -> trial_runner.TrialsRunner:
    return trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), quest_root=root,
        protocol={"grid": {"flow": [1.0, 2.0, 3.0]}}, deterministic=True,
        simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")


def test_a_sweep_whose_numbers_never_change_is_sent_back_and_one_that_varies_is_not(tmp_path: Path) -> None:
    root = _quest(tmp_path, 'def run_cell(cell):\n    return {"t_out": 20.0, "drop": 0.0}\n')
    (root / "code" / "experiment.py").write_text(SWEEP_ANALYSIS, encoding="utf-8")
    runner = _sweep_runner(root)
    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    result = asyncio.run(runner.execute(cmd, cwd=root, timeout_s=60, env={"FI_REPLICATE_SEED": "0"}))
    assert result.returncode == 1 and runner.failed_script == "simulate.py" and runner.flat
    assert "does not change when `flow` change" in result.stderr and "RESULT_JSON" not in result.stdout

    (root / "code" / "simulate.py").write_text('def run_cell(cell):\n    return {"t_out": 20.0 - cell["flow"], "drop": 0.0}\n',
                                               encoding="utf-8")
    runner = _sweep_runner(root)
    result = asyncio.run(runner.execute(cmd, cwd=root, timeout_s=60, env={"FI_REPLICATE_SEED": "0"}))
    assert result.returncode == 0, result.stderr[-1500:]
    assert runner.flat is None and runner.failed_script is None and '"cells": 3' in result.stdout


# --- the stop --------------------------------------------------------------------------------------------------------


def _cfg(tmp_path: Path, **engine: Any) -> Config:
    return Config(
        topic="a neutral topic about a heat sink", title="flat", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False, auto_accept_on_pass=True, oracle_check="off", **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def test_the_gate_names_a_flat_simulation_apart_from_a_crash(tmp_path: Path) -> None:
    eng = Engine(_cfg(tmp_path))
    state = {"exec_result": {"returncode": 1, "flat_output": "the result does not change when `fin` change, so ..."},
             "iteration": 2, "result_json": {}}
    ruled = eng._no_results_verdict(state)
    assert ruled is not None and ruled["stuck"] is True and ruled["stuck_reason"] == "flat"
    assert "does not depend on what it is given" in ruled["rationale"]
    # With an iteration left it is first sent back to the design, like any run with no result.
    again = eng._no_results_verdict({**state, "iteration": 0})
    assert again is not None and again.get("redesign") is True


@pytest.mark.asyncio
async def test_the_stop_says_the_simulation_gives_the_same_result_whatever_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    eng = Engine(_cfg(tmp_path))
    symptom = "the result does not change when `fin`, `gap` change, so the simulation does not use them: `peak` was the same"
    out = await eng._node_stuck_no_findings({
        "evidence_assessment": {"stuck_reason": "flat"}, "evidence_no_result_retries": 1, "exec_reflect_iter": 2,
        "exec_result": {"returncode": 1, "flat_output": symptom}})
    record = out["stuck"]
    assert "gives the same result whatever it is given" in record["problem"]
    assert "repaired the script 2 times" in record["problem"] and "no number" not in record["problem"]
    assert any("does not change when `fin`, `gap` change" in t for t in record["tried"])
    assert any("asked the model to repair the simulation (2 repairs)" in t for t in record["tried"])
    assert "still did not depend on its inputs" in record["why_repairs_ended"]
    assert (eng.quest_root / "needs" / "STUCK.json").is_file()
    assert "No paper was written" in capsys.readouterr().out


# --- through the whole graph, with a fake model ---------------------------------------------------------------------

E2E_MODEL = {
    "summary": "a smooth bowl-shaped temperature peak(fin, gap) with a limit on the mass",
    "assumptions": ["the peak is exact at every mesh size"],
    "holds_for": "fin and gap between 0 and 6",
    "equations": [{"id": "E1", "formula": "peak = (fin - 3)^2 + 2 (gap - 1)^2 + 30", "role": "generates",
                   "source": "derivation", "derivation": "a quadratic bowl with its minimum 30 at fin = 3, gap = 1"}],
}
FLAT_E2E_SIM = 'def run_cell(cell):\n    return {"peak": 0.0, "mass": cell["fin"] * 0.0}  # E1\n'
WORKING_E2E_SIM = ('def run_cell(cell):\n    return {"peak": (cell["fin"] - 3.0) ** 2 + 2.0 * (cell["gap"] - 1.0) ** 2 + 30.0,'
                   ' "mass": cell["fin"]}  # E1\n')


async def _run_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repaired: str | None) -> tuple[Any, Any, list[str]]:
    from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

    block = _block(evaluation_budget={"starts": 1, "per_start": 8})
    seen: list[str] = []
    kinds: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        kinds.append(kind)
        if "does not change when" in prompt:
            seen.append(prompt)
            sim = repaired if repaired is not None else FLAT_E2E_SIM
            return json.dumps({"code": sim, "patch_summary": "computed the results from the design"})
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body.update(study_type="find_best_design", method="search fin and gap for the lowest peak",
                        protocol={"optimisation": block, "oracles": [], "model": E2E_MODEL})
            return json.dumps(body)
        if kind == "Implementation":
            return (f"```python\n# file: simulate.py\n{FLAT_E2E_SIM}\n```\n```python\n# file: experiment.py\n"
                    f"{ANALYSIS}\n```\nDEPS: matplotlib\n")
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(Config(
        topic="Find the fin and gap that give the lowest peak temperature of a toy heat sink", title="toy flat search",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False, oracle_check="off", exec_reflect_max_iterations=2),
        execution=ExecutionConfig(sandbox="venv", timeout_s=300, shared_interpreter=False, system_site_packages=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(review="off"),
    ))
    artifacts = await engine.run()
    return engine, artifacts, seen


@pytest.mark.asyncio
async def test_a_flat_simulation_is_repaired_then_stops_with_no_paper_when_it_stays_flat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    engine, artifacts, seen = await _run_graph(tmp_path, monkeypatch, repaired=None)
    assert seen, "the repair was asked, in words about the symptom"
    for prompt in seen:
        assert "the simulation does not use them" in prompt and "`peak` was the same at all" in prompt
        assert "expected" not in prompt.split("does not change when", 1)[1][:400].lower()
    record = json.loads((engine.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert "gives the same result whatever it is given" in record["problem"]
    assert artifacts.paper_md is None
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "the simulation does not use the design it is given" in log and "stopping without a paper" in log
    assert "No paper was written" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_flat_simulation_the_repair_fixes_goes_on_to_a_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, artifacts, seen = await _run_graph(tmp_path, monkeypatch, repaired=WORKING_E2E_SIM)
    assert len(seen) == 1, "one repair, and the run that followed was a real result"
    assert not (engine.quest_root / "needs" / "STUCK.json").exists()
    best = json.loads((engine.quest_root / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert best["best"]["objective"] < 40
    assert artifacts.paper_md is not None
