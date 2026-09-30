"""The search for the best design runs in the ENGINE, not in the experiment's script.

The simulation only says what one design gives (``run_cell(cell) -> dict``, or ``run_trial`` for a study with
randomness). FI decides which design to evaluate next, calls the simulation on it in a process of its own through the
same harness the trials use, counts every evaluation against the plan's budget, keeps infeasible and failed designs out
of the answer, and writes the only record of it (``raw/optimisation_ledger.jsonl`` and ``results/best_design.json``).
A script that writes those files itself is ignored: FI's own copy is what stands.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from core import optimisation_plan as op
from core import optimise
from core import optimise_search as osearch
from core.execution import ExecutionResult


# --- a toy simulation with a known optimum ---------------------------------------------------------------------------

BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [
        {"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"},
        {"name": "y", "low": -5.0, "high": 5.0, "kind": "continuous"},
    ],
    "fixed": {"scale": 1.0},
    "baseline": {"values": {"x": 0.0, "y": 0.0}, "source": "the design in use"},
    "numerical_settings": {"mesh": {"search": 0.5}},
    "evaluation_budget": {"starts": 2, "per_start": 40},
    "search_method": "bounded_local",
}


def _f(cell: dict[str, Any]) -> dict[str, float]:
    x, y = cell["x"], cell["y"]
    return {"f": cell["scale"] * ((x - 2) ** 2 + (y + 1) ** 2), "g": x + y, "mesh_seen": cell["mesh"]}


def _block(**changes: Any) -> dict[str, Any]:
    block = copy.deepcopy(BLOCK)
    block.update(changes)
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def _in_process(block: dict[str, Any], *, seed: int = 0, method: str | None = None, fail=None) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []

    def evaluate(cell: dict[str, Any]) -> dict[str, float]:
        calls.append(dict(cell))
        if fail is not None and fail(cell):
            raise RuntimeError("the solver diverged")
        return _f(cell)

    outcome = osearch.run_sync(block, evaluate, seed=seed, method=method)
    outcome["_calls"] = calls
    return outcome


# --- the search itself (in-process: fast) ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bounded_local", "global_then_local"])
def test_each_built_in_method_finds_the_known_optimum(method: str) -> None:
    out = _in_process(_block(search_method=method, evaluation_budget={"starts": 3, "per_start": 60}))
    best = out["best"]
    assert best is not None and out["method_used"] == method
    assert abs(best["design"]["x"] - 2) < 0.05 and abs(best["design"]["y"] + 1) < 0.05, best
    assert best["objective"] < 5e-3


def test_exhaustive_evaluates_every_combination_of_whole_number_variables() -> None:
    block = _block(design_variables=[{"name": "x", "low": -3, "high": 4, "kind": "integer"},
                                     {"name": "y", "low": -3, "high": 3, "kind": "integer"}],
                   baseline={"values": {"x": 0, "y": 0}, "source": "s"}, search_method="exhaustive",
                   evaluation_budget={"starts": 1, "per_start": 60})
    out = _in_process(block)
    assert out["method_used"] == "exhaustive" and out["evaluations"] == 8 * 7 and out["stopped_because"] == "finished"
    assert out["best"]["design"] == {"x": 2, "y": -1} and out["best"]["objective"] == 0.0


def test_integer_variables_are_always_whole_numbers() -> None:
    block = _block(design_variables=[{"name": "x", "low": -5.0, "high": 5.0},
                                     {"name": "y", "low": -5, "high": 5, "kind": "integer"}])
    out = _in_process(block)
    assert out["_calls"] and all(isinstance(c["y"], int) for c in out["_calls"])
    assert all(isinstance(r["design"]["y"], int) for r in out["rows"])
    assert out["best"]["design"]["y"] == -1


def test_no_design_outside_its_range_is_ever_evaluated() -> None:
    out = _in_process(_block(search_method="global_then_local"))
    for cell in out["_calls"]:
        assert -5 <= cell["x"] <= 5 and -5 <= cell["y"] <= 5


def test_every_cell_carries_the_fixed_conditions_and_the_search_settings() -> None:
    out = _in_process(_block())
    assert all(c["scale"] == 1.0 and c["mesh"] == 0.5 for c in out["_calls"])


def test_the_budget_is_a_hard_limit_and_the_baseline_is_evaluated_first() -> None:
    out = _in_process(_block(evaluation_budget={"starts": 2, "per_start": 7}))
    assert len(out["_calls"]) == out["evaluations"] <= 14
    assert out["rows"][0]["stage"] == "baseline" and out["rows"][0]["design"] == {"x": 0.0, "y": 0.0}
    assert out["baseline"]["objective"] == 5.0
    assert out["stopped_because"] == "budget" and out["budget_ran_out"] is True


def test_an_infeasible_design_is_recorded_but_never_chosen() -> None:
    # x + y >= 2 moves the optimum to the boundary point (2.5, -0.5), where f = 0.5.
    block = _block(constraints=[{"quantity": "g", "limit": ">= 2"}], evaluation_budget={"starts": 3, "per_start": 80},
                   baseline={"values": {"x": 2.0, "y": 1.0}, "source": "s"})
    out = _in_process(block)
    assert any(not r["feasible"] and r["status"] == "ok" for r in out["rows"]), "the search did meet infeasible designs"
    best = out["best"]
    assert best["feasible"] and best["constraints"]["g"] >= 2
    assert abs(best["objective"] - 0.5) < 0.05, best


def test_no_feasible_design_gives_no_best_and_says_so() -> None:
    block = _block(constraints=[{"quantity": "g", "limit": ">= 100"}], evaluation_budget={"starts": 1, "per_start": 10})
    out = _in_process(block)
    assert out["best"] is None and out["baseline"]["feasible"] is False
    record = osearch.best_design(out, block, seed=0)
    assert record["best"] is None and record["improvement"] is None
    assert "no design" in record["says"].lower()


def test_a_failed_evaluation_is_recorded_and_never_chosen() -> None:
    out = _in_process(_block(search_method="global_then_local"), fail=lambda c: c["x"] > 1.5)
    failed = [r for r in out["rows"] if r["status"] == "failed"]
    assert failed and all("diverged" in r["reason"] for r in failed)
    assert out["best"]["design"]["x"] <= 1.5


def test_the_same_seed_gives_the_same_search_and_another_seed_another() -> None:
    def strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows]

    block = _block(evaluation_budget={"starts": 3, "per_start": 15})
    assert strip(_in_process(block, seed=7)["rows"]) == strip(_in_process(block, seed=7)["rows"])
    assert strip(_in_process(block, seed=7)["rows"]) != strip(_in_process(block, seed=8)["rows"])


def test_maximise_and_the_improvement_over_the_baseline() -> None:
    block = _block(objective={"quantity": "neg", "direction": "maximise"})

    def evaluate(cell: dict[str, Any]) -> dict[str, float]:
        return {"neg": -((cell["x"] - 2) ** 2 + (cell["y"] + 1) ** 2)}

    out = osearch.run_sync(block, evaluate, seed=0)
    record = osearch.best_design(out, block, seed=0)
    assert record["baseline"]["objective"] == -5.0
    assert record["improvement"]["value"] > 4.9 and record["improvement"]["better"] is True
    assert record["checked_at_finer_settings"] is False
    assert record["check_settings"] == {"mesh": [0.25, 0.125]}, "what the later check will use is recorded now"


def test_a_missing_objective_is_a_failed_evaluation_with_the_reason() -> None:
    block = _block(evaluation_budget={"starts": 1, "per_start": 3})
    out = osearch.run_sync(block, lambda cell: {"other": 1.0}, seed=0)
    assert out["best"] is None and all(r["status"] == "failed" for r in out["rows"])
    assert "`f`" in out["rows"][0]["reason"]


@pytest.mark.parametrize("method", ["bounded_local", "global_then_local"])
def test_a_design_taken_into_the_search_space_and_back_is_not_evaluated_again(method: str) -> None:
    # 2.711 in [-7, 5.761] comes back from the unit cube as 2.7110000000000003: the same design, not a second evaluation.
    block = _block(design_variables=[{"name": "x", "low": -7.0, "high": 5.761}, {"name": "y", "low": -5.0, "high": 5.0}],
                   baseline={"values": {"x": 2.711, "y": 0.0}, "source": "s"}, search_method=method,
                   evaluation_budget={"starts": 2, "per_start": 20})
    out = _in_process(block)
    keys = [json.dumps(c["x"]) + json.dumps(c["y"]) for c in out["_calls"]]
    assert len(keys) == len(set(keys)), "no design evaluated twice"
    assert sum(1 for c in out["_calls"] if abs(c["x"] - 2.711) < 1e-9 and c["y"] == 0.0) == 1


def test_every_starting_point_is_in_the_record_and_budget_means_the_whole_budget() -> None:
    out = _in_process(_block(evaluation_budget={"starts": 3, "per_start": 10}))
    assert [s["start"] for s in out["start_log"]] == [0, 1, 2]
    assert sum(s["evaluations"] for s in out["start_log"]) == out["evaluations"]
    easy = _in_process(_block(evaluation_budget={"starts": 2, "per_start": 400}))
    assert easy["budget_ran_out"] is False and easy["stopped_because"] == "converged", easy["start_log"]


def test_a_limit_written_another_way_is_read_as_the_plan_reads_it_and_one_that_cannot_be_read_is_never_met() -> None:
    block = _block(constraints=[{"quantity": "g", "limit": "< 0.5"}])  # the plan's check makes it "<= 0.5"
    assert all(r["constraints"]["g"] <= 0.5 for r in _in_process(block)["rows"] if r["feasible"])
    raw = {**block, "constraints": [{"quantity": "g", "limit": "≤ 0.5"}]}
    assert osearch.judge(raw, {"f": 1.0, "g": 3.0})["feasible"] is False
    assert osearch.judge({**block, "constraints": [{"quantity": "g", "limit": "small"}]},
                         {"f": 1.0, "g": 0.0})["status"] == "failed"


def test_the_plans_threshold_for_better_is_applied_when_it_gives_one() -> None:
    block = _block(improvement_tolerance={"value": 10.0, "mode": "absolute"}, evaluation_budget={"starts": 1, "per_start": 30})
    record = osearch.best_design(_in_process(block), block, seed=0)
    assert record["improvement"]["better"] is True and record["improvement"]["beyond_threshold"] is False
    assert "less than the plan's threshold" in record["says"]
    plain = _block(evaluation_budget={"starts": 1, "per_start": 30})
    assert osearch.best_design(_in_process(plain), plain, seed=0)["improvement"]["beyond_threshold"] is None


def test_what_the_analysis_reports_is_set_beside_fis_own_numbers() -> None:
    block = _block(evaluation_budget={"starts": 1, "per_start": 20})
    record = osearch.best_design(osearch.run_sync(block, _f, seed=0), block, seed=0)
    stdout = "hello\nRESULT_JSON: " + json.dumps({"best_f": record["best"]["objective"], "baseline_f": 5.0,
                                                  "improvement": 9.9, "other": 1}) + "\n"
    out, differs = optimise.with_fi_record(stdout, record)
    assert differs == ["improvement = 9.9"]
    result = json.loads(out.splitlines()[-1][len("RESULT_JSON: "):])
    assert result["fi_search"]["best_objective"] == record["best"]["objective"]
    assert result["fi_search"]["not_in_fi_record"] == ["improvement = 9.9"] and result["other"] == 1


def test_a_range_bound_with_many_digits_is_never_left() -> None:
    low = 0.1234567890123456
    block = _block(design_variables=[{"name": "x", "low": low, "high": 0.5}, {"name": "y", "low": -5.0, "high": 5.0}],
                   baseline={"values": {"x": low, "y": 0.0}, "source": "s"}, evaluation_budget={"starts": 2, "per_start": 15})
    out = _in_process(block)
    assert all(low <= c["x"] <= 0.5 for c in out["_calls"])


def test_fis_own_numbers_do_not_hide_an_all_zero_result() -> None:
    from core.engine import _is_degenerate_result

    block = _block(evaluation_budget={"starts": 1, "per_start": 10})
    record = osearch.best_design(osearch.run_sync(block, _f, seed=0), block, seed=0)
    out, _ = optimise.with_fi_record('RESULT_JSON: {"a": 0.0, "b": 0.0}\n', record)
    assert _is_degenerate_result(json.loads(out.strip()[len("RESULT_JSON: "):])) is True


def test_a_simulation_that_imports_an_optimiser_is_named() -> None:
    found = optimise.searches_itself({"simulate.py": "from scipy.optimize import minimize, brentq\nimport optuna\n"
                                                     "import scipy.optimize as so\nso.differential_evolution(f, b)\n"})
    assert "simulate.py imports scipy.optimize.minimize" in found and "simulate.py imports optuna" in found
    assert any("differential_evolution" in f for f in found) and not any("brentq" in f for f in found)
    assert optimise.searches_itself({"simulate.py": "from scipy.optimize import brentq, fsolve\n"}) == []


def test_exhaustive_with_a_continuous_variable_says_what_it_used_instead() -> None:
    out = _in_process(_block(search_method="exhaustive"))
    assert out["method_used"] == "bounded_local" and out["method_requested"] == "exhaustive"
    assert any("exhaustive" in n for n in out["notes"])
    assert all(r["method"] == "bounded_local" for r in out["rows"]), "the baseline too, under the method that ran"


# --- through FI's processes and its record ---------------------------------------------------------------------------

SIMULATE = '''\
import os, json


def run_cell(cell):
    with open("count.txt", "a", encoding="utf-8") as fh:
        fh.write("1\\n")
    x, y = cell["x"], cell["y"]
    if cell.get("x", 0) > 4.5:
        raise RuntimeError("the solver diverged")
    return {"f": cell["scale"] * ((x - 2) ** 2 + (y + 1) ** 2), "g": x + y}
'''

# A simulation that tries to write FI's record itself, on every call.
CHEATING = SIMULATE.replace(
    "    x, y = cell[\"x\"], cell[\"y\"]\n",
    "    os.makedirs('raw', exist_ok=True); os.makedirs('results', exist_ok=True)\n"
    "    open('raw/optimisation_ledger.jsonl', 'a').write(json.dumps({'design': {'x': 2, 'y': -1}, 'objective': -99}) + '\\n')\n"
    "    open('results/best_design.json', 'w').write(json.dumps({'best': {'objective': -99}}))\n"
    "    x, y = cell[\"x\"], cell[\"y\"]\n",
)


class LocalExecutor:
    """Runs a command in this interpreter's environment, as the quest's executor would in its own."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> ExecutionResult:
        self.calls.append(list(cmd))
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, env=env)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            out, err = await proc.communicate()
            timed_out = True
        return ExecutionResult(proc.returncode if proc.returncode is not None else -1, out.decode(errors="replace"),
                               err.decode(errors="replace"), time.monotonic() - started, timed_out)


def _quest(tmp_path: Path, simulate: str = SIMULATE) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    return root


def _protocol(**changes: Any) -> dict[str, Any]:
    return {"optimisation": _block(**changes)}


def _ledger(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "raw" / "optimisation_ledger.jsonl").read_text(encoding="utf-8").splitlines()]


def _count(root: Path) -> int:
    path = root / "count.txt"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


async def _run(root: Path, protocol: dict[str, Any], **kw: Any) -> optimise.SearchRun:
    return await optimise.run_search(LocalExecutor(), sys.executable, root, "code/simulate.py", protocol,
                                     base_seed=kw.pop("base_seed", 0), timeout_s=kw.pop("timeout_s", 300), **kw)


@pytest.mark.asyncio
async def test_fi_runs_every_evaluation_in_its_own_process_within_the_budget(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    run = await _run(root, _protocol(evaluation_budget={"starts": 2, "per_start": 12}))
    rows = [r for r in _ledger(root) if r.get("event") == "evaluation"]
    assert _count(root) == len(rows) == run.record["evaluations"]["search"] <= 24
    assert rows[0]["stage"] == "baseline" and rows[0]["design"] == {"x": 0.0, "y": 0.0}
    best = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert best["baseline"]["objective"] == 5.0 and best["best"]["objective"] < 5.0
    assert best["evaluations"]["budget"] == 24 and best["checked_at_finer_settings"] is False
    for row in rows:
        assert {"n", "stage", "start", "design", "objective", "constraints", "feasible", "status", "method",
                "elapsed_s"} <= set(row)


@pytest.mark.asyncio
async def test_a_script_that_writes_the_record_itself_is_ignored(tmp_path: Path) -> None:
    root = _quest(tmp_path, CHEATING)
    run = await _run(root, _protocol(evaluation_budget={"starts": 1, "per_start": 8}))
    lines = _ledger(root)
    assert all(line.get("schema") == osearch.LEDGER_SCHEMA or line.get("event") == "evaluation" for line in lines), lines
    assert not any(line.get("objective") == -99 for line in lines)
    best = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert best["best"]["objective"] != -99 and best == run.record
    assert optimise.restore(root) is False, "files FI just wrote are FI's, byte for byte (no newline translation)"
    # The analysis runs after FI wrote the files, and may overwrite one of them: FI puts its own back.
    (root / "results" / "best_design.json").write_text(json.dumps({"best": {"objective": -99}}), encoding="utf-8")
    assert optimise.restore(root) is True
    assert json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8")) == run.record
    assert optimise.restore(root) is False, "nothing to put back the second time"
    # One that also rewrites FI's record of the files cannot make its version stand: the runner restores from memory.
    record_path = root / ".fi" / "optimisation" / "run.json"
    forged = json.loads(record_path.read_text(encoding="utf-8"))
    forged["best"]["text"] = json.dumps({"best": {"objective": -99}})
    record_path.write_text(json.dumps(forged), encoding="utf-8")
    (root / "results" / "best_design.json").write_text(forged["best"]["text"], encoding="utf-8")
    assert optimise.restore(root, run.files) is True
    assert json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8")) == run.record
    assert json.loads(record_path.read_text(encoding="utf-8"))["best"]["text"] != forged["best"]["text"]


@pytest.mark.asyncio
async def test_a_failed_evaluation_is_in_the_ledger_and_the_search_goes_on(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    await _run(root, _protocol(search_method="global_then_local", evaluation_budget={"starts": 2, "per_start": 15}))
    rows = [r for r in _ledger(root) if r.get("event") == "evaluation"]
    failed = [r for r in rows if r["status"] == "failed"]
    assert failed and all("diverged" in r["reason"] for r in failed)
    best = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert best["best"]["design"]["x"] <= 4.5 and best["evaluations"]["failed"] == len(failed)


@pytest.mark.asyncio
async def test_a_coarse_scan_runs_first_through_the_trial_runner_and_is_not_counted_in_the_budget(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    run = await _run(root, _protocol(grid={"x": [-4.0, 0.0, 4.0], "y": [-2.0, 2.0]}, search_method="global_then_local",
                                     evaluation_budget={"starts": 2, "per_start": 10}))
    rows = [r for r in _ledger(root) if r.get("event") == "evaluation"]
    scan = [r for r in rows if r["stage"] == "scan"]
    assert rows[0]["stage"] == "baseline", "the baseline first, always"
    assert len(scan) == 6 and rows[1:7] == scan, "then the scan"
    assert (root / "raw" / "trials.json").is_file(), "the scan is the trial runner's, so the analysis can plot it"
    assert run.record["evaluations"]["scan"] == 6 and run.record["evaluations"]["search"] <= 20
    assert _count(root) == len(rows)


@pytest.mark.asyncio
async def test_the_search_is_kept_while_the_simulation_and_the_plan_are_unchanged(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6})
    await _run(root, protocol)
    before = _count(root)
    again = await _run(root, protocol)
    assert _count(root) == before and again.reused is True, "an analysis repair does not spend the budget again"
    await _run(root, protocol, base_seed=3)
    assert _count(root) > before, "another seed is another search"


@pytest.mark.asyncio
async def test_the_time_limit_stops_the_search_and_says_so(tmp_path: Path) -> None:
    root = _quest(tmp_path, SIMULATE.replace("def run_cell(cell):\n", "def run_cell(cell):\n    import time; time.sleep(0.4)\n"))
    protocol = _protocol(evaluation_budget={"starts": 2, "per_start": 50})
    run = await _run(root, protocol, timeout_s=3)
    assert run.record["evaluations"]["stopped_because"] == "time"
    assert run.record["evaluations"]["search"] < 100
    assert not any(r["status"] == "failed" for r in run.rows), "an evaluation cut off by the clock is not a failed design"
    # Kept for the same limit (a repair of the analysis does not run it again)...
    before = _count(root)
    same = await _run(root, protocol, timeout_s=3)
    assert same.reused is True and _count(root) == before and same.record == run.record
    # ...but a longer limit runs it again.
    again = await _run(root, protocol, timeout_s=6)
    assert again.reused is False and _count(root) > before


@pytest.mark.asyncio
async def test_a_search_that_got_no_result_is_not_kept_for_the_next_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every evaluation fails for a reason outside the simulation's text (a package installed afterwards, say).
    root = _quest(tmp_path, SIMULATE.replace("def run_cell(cell):\n",
                                             "def run_cell(cell):\n    if not os.environ.get('TOY_READY'):\n"
                                             "        raise RuntimeError('not ready')\n"))
    monkeypatch.delenv("TOY_READY", raising=False)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 5})
    run = await _run(root, protocol)
    assert run.ok_evaluations == 0
    monkeypatch.setenv("TOY_READY", "1")
    again = await _run(root, protocol)
    assert again.reused is False and again.ok_evaluations > 0


@pytest.mark.asyncio
async def test_a_library_method_that_the_environment_lacks_falls_back_to_a_built_in_one(tmp_path: Path) -> None:
    lacking = next((lib for lib in ("optuna", "scipy") if subprocess.run([sys.executable, "-c", f"import {lib}"],
                                                                         capture_output=True).returncode), None)
    if lacking is None:
        pytest.skip("both scipy and optuna are installed here")
    method = "optuna:tpe" if lacking == "optuna" else "scipy:minimize"
    root = _quest(tmp_path)
    run = await _run(root, _protocol(search_method=method, evaluation_budget={"starts": 2, "per_start": 10}))
    assert run.record["method"]["requested"] == method and run.record["method"]["used"] == "bounded_local"
    assert f"{lacking} is not installed" in run.record["method"]["why"]
    assert not (root / ".venv").exists(), "FI never installs it"


@pytest.mark.asyncio
async def test_scipy_is_used_when_the_quests_environment_has_it(tmp_path: Path) -> None:
    if subprocess.run([sys.executable, "-c", "import scipy.optimize"], capture_output=True).returncode:
        pytest.skip("scipy is not installed in this environment")
    # Each step of the library's method runs in the quest's Python (scipy is imported there, not in FI): a small budget.
    root = _quest(tmp_path)
    run = await _run(root, _protocol(search_method="scipy:Powell", evaluation_budget={"starts": 1, "per_start": 30}))
    assert run.record["method"]["used"] == "scipy:Powell"
    rows = [r for r in _ledger(root) if r.get("event") == "evaluation"]
    assert _count(root) == len(rows) <= 30 and rows[0]["stage"] == "baseline"
    assert all(r["method"] == "scipy:Powell" for r in rows)
    best = run.record["best"]
    assert abs(best["design"]["x"] - 2) < 0.05 and abs(best["design"]["y"] + 1) < 0.05, best
    # The same method in this process (as code/run.py runs it) asks for the same designs.
    def toy(cell: dict[str, Any]) -> dict[str, float]:
        if cell["x"] > 4.5:
            raise RuntimeError("the solver diverged")
        return {"f": cell["scale"] * ((cell["x"] - 2) ** 2 + (cell["y"] + 1) ** 2), "g": cell["x"] + cell["y"]}

    block = _protocol(search_method="scipy:Powell", evaluation_budget={"starts": 1, "per_start": 30})["optimisation"]
    again = osearch.run_sync(block, toy, seed=0)
    assert [r["design"] for r in again["rows"]] == [r["design"] for r in rows]


@pytest.mark.asyncio
async def test_optuna_is_used_when_the_quests_environment_has_it(tmp_path: Path) -> None:
    if subprocess.run([sys.executable, "-c", "import optuna"], capture_output=True).returncode:
        pytest.skip("optuna is not installed in this environment")
    root = _quest(tmp_path)
    run = await _run(root, _protocol(search_method="optuna:tpe", evaluation_budget={"starts": 1, "per_start": 30}))
    assert run.record["method"]["used"] == "optuna:tpe"
    assert _count(root) == run.record["evaluations"]["search"] <= 30


def test_the_search_module_follows_the_plan_modules_rules() -> None:
    """The search module is copied into the quest and cannot import FI: its copies of two rules must stay FI's."""
    for block in (_block(), _block(search_method=None), _block(grid={"x": [0.0, 1.0]}, search_method=None),
                  _block(design_variables=[{"name": "x", "low": 0, "high": 3, "kind": "integer"}],
                         baseline={"values": {"x": 1}, "source": "s"}, search_method=None),
                  _block(numerical_settings={"n": {"search": 40, "finer": "larger"}, "h": {"search": 0.3, "check": [0.1]}})):
        block = {k: v for k, v in block.items() if v is not None}
        assert osearch.effective_method(block) == op.effective_method(block)
        for name, setting in (block.get("numerical_settings") or {}).items():
            assert osearch.check_settings(block)[name] == op.check_levels(setting)[0]


@pytest.mark.asyncio
async def test_a_search_without_a_simulation_to_call_is_a_failed_run_not_a_script_that_searches(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {\"best_f\": 0}')\n", encoding="utf-8")
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol=_protocol(),
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")
    (root / "code" / "simulate.py").write_text("def helper(cell):\n    return {}\n", encoding="utf-8")
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60)
    assert result.returncode == 1 and runner.failed_script == "simulate.py"
    assert "neither run_cell" in result.stderr and "RESULT_JSON" not in result.stdout
    (root / "code" / "simulate.py").unlink()
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60)
    assert result.returncode == 1 and "is missing" in result.stderr and "RESULT_JSON" not in result.stdout


@pytest.mark.asyncio
async def test_code_run_py_repeats_the_search_without_fi(tmp_path: Path) -> None:
    from core import code_project

    root = _quest(tmp_path)
    (root / "code" / "experiment.py").write_text(
        "import json, os\nbest = json.load(open(os.environ['FI_BEST_DESIGN']))\n"
        "print('RESULT_JSON: ' + json.dumps({'best_f': best['best']['objective']}))\n", encoding="utf-8")
    protocol = _protocol(grid={"x": [-2.0, 3.0]}, search_method="global_then_local",
                         evaluation_budget={"starts": 2, "per_start": 10})
    run = await _run(root, protocol)
    written = code_project.refresh(root, protocol=protocol)
    assert "fi_search.py" in written and "study.json" in written
    study = json.loads((root / "code" / "study.json").read_text(encoding="utf-8"))
    assert study["optimisation"] == protocol["optimisation"] and study["method"] == "global_then_local"
    assert "searches for the best design" in (root / "code" / "README.md").read_text(encoding="utf-8")
    done = subprocess.run([sys.executable, "run.py"], cwd=root / "code", capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    again = json.loads((root / "run_output" / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert again["best"]["design"] == run.record["best"]["design"]
    assert again["evaluations"]["search"] == run.record["evaluations"]["search"]
    theirs = [json.loads(line)["design"] for line in
              (root / "run_output" / "raw" / "optimisation_ledger.jsonl").read_text(encoding="utf-8").splitlines()[1:]]
    assert theirs == [r["design"] for r in _ledger(root) if r.get("event") == "evaluation"], "the same designs, in order"
    assert "RESULT_JSON" in done.stdout
    # A plan that is a measurement again leaves no search behind.
    code_project.refresh(root, protocol={"grid": {"x": [1.0]}})
    assert not (root / "code" / "fi_search.py").exists()


# --- through the real graph, with a fake model -------------------------------------------------------------------------

E2E_MODEL = {
    "summary": "a smooth bowl-shaped energy f(x, y) = (x - 2)^2 + (y + 1)^2 with a limit on x + y",
    "assumptions": ["the energy is exact at every mesh size"],
    "holds_for": "x and y between -5 and 5",
    "equations": [{"id": "E1", "formula": "f = (x - 2)^2 + (y + 1)^2", "role": "generates", "source": "derivation",
                   "derivation": "a quadratic bowl with its minimum f = 0 at x = 2, y = -1"}],
}
E2E_ORACLE = {"name": "energy of the baseline", "kind": "special_case", "check": "f at the baseline (0, 0) is 5",
              "expected": 5.0, "tolerance": 1e-9, "reference": "derivation: f(0, 0) = (0 - 2)^2 + (0 + 1)^2 = 5",
              "case": {"x": 0.0, "y": 0.0}, "measure": "f"}
E2E_SIMULATE = '''\
def run_cell(cell):
    with open("count.txt", "a", encoding="utf-8") as fh:
        fh.write("1\\n")
    x, y = cell["x"], cell["y"]
    f = cell["scale"] * ((x - 2) ** 2 + (y + 1) ** 2)  # E1
    return {"f": f, "g": x + y}
'''
# The analysis also tries to overwrite FI's record of the search: FI puts its own copy back.
E2E_ANALYSIS = '''\
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
best = json.load(open(os.environ["FI_BEST_DESIGN"], encoding="utf-8"))
rows = [json.loads(line) for line in open(os.environ["FI_OPTIMISATION"], encoding="utf-8") if line.strip()]
rows = [r for r in rows if r.get("event") == "evaluation"]
so_far, best_now = [], None
for r in rows:
    if r["feasible"]:
        best_now = r["objective"] if best_now is None else min(best_now, r["objective"])
    so_far.append(best_now if best_now is not None else float("nan"))
os.makedirs("figures", exist_ok=True)
plt.figure(); plt.plot(range(1, len(so_far) + 1), so_far); plt.xlabel("evaluation"); plt.ylabel("best f so far")
plt.savefig("figures/search_progress.png", dpi=72)
with open(os.environ["FI_BEST_DESIGN"], "w", encoding="utf-8") as fh:
    fh.write(json.dumps({"best": {"objective": -1}}))
print("RESULT_JSON: " + json.dumps({"best_f": best["best"]["objective"], "baseline_f": best["baseline"]["objective"],
                                    "improvement": best["improvement"]["value"], "evaluations": len(rows)}))
'''


@pytest.mark.asyncio
async def test_a_search_for_the_best_design_runs_through_execute_with_a_fake_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig,
    )
    from core.engine import Engine
    from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

    block = _block(constraints=[{"quantity": "g", "limit": "<= 10"}], evaluation_budget={"starts": 2, "per_start": 20})
    prompts: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body.update(study_type="find_best_design", method="search x and y for the lowest energy f",
                        protocol={"optimisation": block, "oracles": [E2E_ORACLE], "model": E2E_MODEL})
            return json.dumps(body)
        if kind == "Implementation":
            prompts.append(prompt)
            return (f"```python\n# file: simulate.py\n{E2E_SIMULATE}\n```\n```python\n# file: experiment.py\n"
                    f"{E2E_ANALYSIS}\n```\nDEPS: matplotlib\n")
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(Config(
        topic="Find the x and y that minimise the energy f of a toy bowl, keeping x + y at most 10", title="toy search",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=3,
                            pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=300, shared_interpreter=False, system_site_packages=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(review="off"),
    ))
    artifacts = await engine.run()
    root = engine.quest_root
    log = (root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert not (root / ".fi" / "pause.json").is_file() or "best_design_stage" not in (root / ".fi" / "pause.json").read_text(
        encoding="utf-8"), "a search the engine can run does not stop at the design"
    assert prompts and "FI runs the search" in prompts[-1], "the code-writing prompt carries the search's contract"
    best = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert best["schema"] == osearch.RECORD_SCHEMA, "FI's own copy stands, not the one the analysis wrote"
    assert abs(best["best"]["design"]["x"] - 2) < 0.1 and abs(best["best"]["design"]["y"] + 1) < 0.1, best["best"]
    assert best["baseline"]["objective"] == 5.0 and best["improvement"]["better"] is True
    assert best["evaluations"]["search"] <= 40 and best["checked_at_finer_settings"] is False
    rows = [json.loads(line) for line in (root / "raw" / "optimisation_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([r for r in rows if r.get("event") == "evaluation"]) == best["evaluations"]["search"]
    # The simulation's own count of its calls: the oracle's one case, then the search, once (no replicate seeds, no
    # second search).
    assert _count(root) == best["evaluations"]["search"] + 1
    result = artifacts.raw_state["result_json"]
    assert result["fi_search"]["best_objective"] == best["best"]["objective"], "FI's own numbers ride with the result"
    assert "[optimise] best design:" in log and "not checked at finer numerical settings" in log
    assert "changed FI's record of the search" in log
    oracle = json.loads((root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert oracle["status"] == "ok", oracle
    assert (root / "figures" / "search_progress.png").is_file()


def test_the_log_lines_say_plainly_that_the_finer_check_is_not_done() -> None:
    block = _block(evaluation_budget={"starts": 1, "per_start": 20})
    record = osearch.best_design(osearch.run_sync(block, _f, seed=0), block, seed=0)
    text = "\n".join(optimise.summary_lines(record))
    assert "best design" in text and "baseline" in text
    assert "not checked at finer numerical settings" in text
    assert math.isfinite(record["improvement"]["value"])
