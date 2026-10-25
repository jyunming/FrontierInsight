"""A study that does not fit the time allowed is run up to the limit, and the paper says what ran and what did not.

FI used to stop before running anything when its timing said the whole study would take longer than ``execution.timeout_s``.
Now it stops only when not even one setting (or one design of a search) fits; otherwise the study runs in the plan's order
up to the limit, a search keeps the time its check needs, and the sentence the paper and the quality report carry comes from
FI's records (``core.disclosure.cut_short_text``). The sizing itself is tested in ``tests/test_run_checks.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import disclosure, optimise, paper_facts, trial_runner
from core.execution import SharedInterpreterExecutor

REASON = trial_runner.OUT_OF_TIME


def _quest(tmp_path: Path, truncated: dict[str, Any] | None, ledger: list[dict[str, Any]] | None = None,
           best: dict[str, Any] | None = None) -> Path:
    root = tmp_path / "quest"
    (root / ".fi").mkdir(parents=True)
    if truncated is not None:
        (root / ".fi" / "run_estimate.json").write_text(json.dumps({"fits": False, "truncated": truncated}), encoding="utf-8")
    if ledger is not None:
        (root / "raw").mkdir()
        (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in ledger) + "\n", encoding="utf-8")
    if best is not None:
        (root / "results").mkdir()
        (root / "results" / "best_design.json").write_text(json.dumps(best), encoding="utf-8")
    return root


def _ledger(cells: dict[str, list[str]]) -> list[dict[str, Any]]:
    """``{cell: [ok | failed | late, ...]}`` as FI's trial ledger holds it (a trial the time limit cut off is ``failed``)."""
    rows: list[dict[str, Any]] = []
    for cell, statuses in cells.items():
        rows.append({"event": "planned", "cell": cell, "trials": len(statuses)})
        for t, what in enumerate(statuses):
            rows.append({"event": "trial", "cell": cell, "trial": t, "seed": 10 + t,
                         "status": "ok" if what == "ok" else "failed",
                         **({"reason": REASON} if what == "late" else {"reason": "diverged"} if what == "failed" else {})})
    return rows


GRID_CUT = {"kind": "grid", "estimate_s": 64800.0, "at_least": False, "limit_s": 28800.0, "settings": 4, "order": "plan"}


def test_a_study_cut_at_the_limit_is_described_from_the_trial_ledger(tmp_path: Path) -> None:
    root = _quest(tmp_path, GRID_CUT, _ledger({"a=1": ["ok", "ok"], "a=2": ["ok", "late"], "a=3": ["late", "late"],
                                              "a=4": ["failed", "ok"]}))
    said = disclosure.cut_short_text(root)
    assert said == ("FI estimated that the plan's study would take about 18 h, more than the 8 h allowed, so it ran the "
                    "plan's settings in the plan's order until the time limit; the results cover only what ran; of 4 settings, "
                    "2 ran all their trials, 1 ran some and 1 ran none (3 trials were not run).")
    # a trial that failed for another reason ran: it is not counted as cut off
    [item] = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.CUT_SHORT]
    assert item.name == paper_facts.CUT_SHORT_NAME and item.sentence == said
    assert said in disclosure.run_record(root)


def test_a_study_that_was_not_cut_says_nothing_of_a_cut(tmp_path: Path) -> None:
    root = _quest(tmp_path, None, _ledger({"a=1": ["ok", "ok"]}))
    assert disclosure.cut_short_text(root) == ""
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.CUT_SHORT]
    assert "18 h" not in disclosure.run_record(root)


def test_a_search_cut_at_the_limit_says_how_far_the_scan_and_the_search_got(tmp_path: Path) -> None:
    cut = {**GRID_CUT, "kind": "search", "estimate_s": 129600.0, "at_least": True}
    best = {"evaluations": {"search": 40, "budget": 6000, "stopped_because": "time"},
            "compared": {"levels": [{"evaluations": 40, "budget": 1200}, {"evaluations": 1200, "budget": 1200},
                                    {"evaluations": 0, "budget": 1200}, {"same_as": {"x": 1}, "evaluations": 0, "budget": 1200}]}}
    root = _quest(tmp_path, cut, _ledger({"s=1": ["ok"], "s=2": ["ok"], "s=3": ["late"]}), best)
    said = disclosure.cut_short_text(root)
    assert "would take at least 36 h, more than the 8 h allowed" in said
    assert "of 3 scan settings, 2 ran all their trials, 0 ran some and 1 ran none (1 trial was not run)" in said
    assert ("the search made 40 of the 6,000 evaluations the plan allows, and stopped at the time limit; of the 3 alternatives "
            "the plan compares, searched with their whole budget: 1, searched only in part: 1, not searched: 1.") in said


def test_a_cut_study_without_a_ledger_still_states_the_cut(tmp_path: Path) -> None:
    said = disclosure.cut_short_text(_quest(tmp_path, GRID_CUT))
    assert said.endswith("the results cover only what ran.") and "ran all their trials" not in said


SLOW_LAST = '''
import time

def run_trial(cell, trial_id, seed):
    if trial_id >= 1:
        time.sleep(60)
    return {"y": float(cell["a"])}
'''


def test_a_run_killed_at_the_time_limit_keeps_the_trials_that_finished(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SLOW_LAST, encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), sys.executable, root,
        "code/simulate.py", {"a": [1, 2]}, runs_per_setting=3, base_seed=0, deterministic=False, timeout_s=6, run_id="r1"))
    statuses = [[r["status"] for r in c.rows] for c in run.cells]
    assert statuses[0][0] == "ok", "the trial that finished before the kill is kept"
    assert set(statuses[0][1:]) == {"failed"}
    assert all(r["reason"] == REASON for c in run.cells for r in c.rows if r["status"] == "failed")
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert summary["cells"][0]["ok"] == 1 and summary["cells"][0]["metrics"]["y"]["count"] == 1


def test_the_search_keeps_the_time_its_check_needs_when_the_study_was_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {'f': 1.0}\n", encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    given: list[int] = []

    async def fake_search(*a: Any, timeout_s: int, **k: Any) -> Any:
        given.append(timeout_s)
        raise RuntimeError("stop here")

    monkeypatch.setattr(optimise, "run_search", fake_search)
    protocol = {"optimisation": {"design_variables": []}}
    runner = optimise.OptimisationRunner(
        SharedInterpreterExecutor(python_version="3.11"), quest_root=root, protocol=lambda: protocol,
        simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")
    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    with pytest.raises(RuntimeError):
        asyncio.run(runner.execute(cmd, cwd=root, timeout_s=1000, env={}))
    assert given == [1000], "a study that fits is searched for the whole limit"
    runner.check_reserve_s = 300.0
    with pytest.raises(RuntimeError):
        asyncio.run(runner.execute(cmd, cwd=root, timeout_s=1000, env={}))
    assert given == [1000, 700], "the scan and the search stop when the time kept for the check begins"


CUT_SIM = '''
import time

def run_trial(cell, trial_id, seed):
    if cell["n"] >= 2:
        time.sleep(60)
    return {"y": float(cell["n"]) + trial_id}
'''
CUT_ANALYSIS = '''
import json, os
data = json.load(open(os.environ["FI_TRIALS"]))
done = [c["key"] for c in data["cells"] if c.get("ok")]
print("RESULT_JSON: " + json.dumps({"settings_done": len(done), "settings_with_results": done}))
'''


@pytest.mark.asyncio
async def test_a_run_cut_at_the_limit_is_a_result_not_a_failed_run(tmp_path: Path) -> None:
    """The settings the limit cut off do not make the run a crash: no repair of the script, no stop, no 'no results'."""
    from tests.test_run_checks import _sizing

    eng, _model, _runner, _ex = await _sizing(tmp_path, timeout_s=6)
    code = eng.quest_root / "code"
    (code / "simulate.py").write_text(CUT_SIM, encoding="utf-8")
    (code / "experiment.py").write_text(CUT_ANALYSIS, encoding="utf-8")
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), quest_root=eng.quest_root,
        protocol={"grid": {"n": [1, 2, 4]}, "runs_per_setting": 3}, deterministic=False,
        simulate=code / "simulate.py", analysis=code / "experiment.py", log=eng._log)
    result = await runner.execute([sys.executable, str(code / "experiment.py")], cwd=eng.quest_root, timeout_s=6, env={})
    assert result.returncode == 0 and not result.timed_out, result.stderr
    out = json.loads(result.stdout.split("RESULT_JSON: ")[1])
    assert out == {"settings_done": 1, "settings_with_results": ["n=1"]}, "the setting that finished is analysed"
    exec_result = {"returncode": result.returncode, "timed_out": result.timed_out, "stdout_tail": result.stdout,
                   "stderr_tail": result.stderr, "duration_s": result.duration_s}
    state = {"exec_result": exec_result, "result_json": out, "iteration": 0, "topic": "t"}
    assert await eng._node_execute_reflect(state) == {}, "no repair of the script is asked for"
    assert eng._route_after_execute_reflect(state) != "stuck"
    assert eng._no_results_verdict(state) is None
