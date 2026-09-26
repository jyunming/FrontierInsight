"""FI runs the trials and writes their record (core/trial_runner.py): the experiment's code only says what a trial does."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from core import run_manifest, trial_runner
from core.execution import SharedInterpreterExecutor

SIM = '''
import random, warnings, sys

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    print("noise the simulation prints")               # must never pass for a result
    if cell["R0"] == 2.0 and trial_id == 1:
        raise ValueError("diverged")
    if trial_id == 2:
        warnings.warn("overflow encountered in exp", RuntimeWarning)
    return {"infected": cell["R0"] * 100 + rng.random(), "peak_day": 10.0 + trial_id}
'''

CRASH = '''
import os

def run_trial(cell, trial_id, seed):
    if trial_id == 1:
        os._exit(7)                                       # the process dies part-way through the cell
    return {"x": 1.0}
'''

DET = '''
def run_cell(cell):
    return {"energy_drift": cell["dt"] * 2}
'''


def _run(tmp_path: Path, source: str, grid: dict, *, runs: int = 3, deterministic: bool = False, timeout: int = 60):
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return root, asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"),
        sys.executable, root, "code/simulate.py", grid, runs_per_setting=runs, base_seed=0,
        deterministic=deterministic, timeout_s=timeout, run_id="r1",
    ))


def test_cells_and_seeds_are_fixed_by_the_protocol() -> None:
    grid = {"R0": [1.5, 2.0], "N": [100]}
    assert trial_runner.cells(grid) == [{"R0": 1.5, "N": 100}, {"R0": 2.0, "N": 100}]
    assert trial_runner.cell_key({"R0": 1.5, "N": 100}) == "R0=1.5,N=100"
    a = trial_runner.trial_seed(0, "R0=1.5,N=100", 0)
    assert a == trial_runner.trial_seed(0, "R0=1.5,N=100", 0)
    assert len({trial_runner.trial_seed(0, k, t) for k in ("R0=1.5,N=100", "R0=2.0,N=100") for t in range(50)}) == 100


def test_fi_writes_the_ledger_and_the_per_cell_summary(tmp_path: Path) -> None:
    root, run = _run(tmp_path, SIM, {"R0": [1.5, 2.0]})
    assert run.ok_trials == 5 and run.failed_trials == 1
    rows = trial_runner.read_ledger(root)
    assert len(rows) == 6 and {r["status"] for r in rows} == {"ok", "failed"}
    failed = next(r for r in rows if r["status"] == "failed")
    assert failed["cell"] == "R0=2.0" and "diverged" in failed["reason"]
    # The existing checker reads FI's rows, and finds the grid and counts complete.
    manifest, row_problems = run_manifest.manifest_from_ledger({"grid": {"R0": [1.5, 2.0]}, "runs_per_setting": 3}, rows, {})
    assert not row_problems and len(manifest["failed_trials"]) == 1
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    first = summary["cells"][0]
    assert first["key"] == "R0=1.5" and first["ok"] == 3 and first["metrics"]["infected"]["count"] == 3
    assert 150 < first["metrics"]["infected"]["total"] / 3 < 151
    assert "noise the simulation prints" in run.stderr() and "overflow encountered" not in json.dumps(summary)
    from core import numeric_warnings
    assert any("overflow encountered in exp" in str(w) for w in numeric_warnings.scan(run.stderr(), {})),         "a warning inside a trial reaches the numeric check"
    events = [json.loads(line) for line in run.ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events][:1] == ["planned"] and events[-1]["warnings"] in (0, 1)


def test_the_spec_file_is_gone_before_the_simulation_loads(tmp_path: Path) -> None:
    """The spec names the nonce; the harness deletes it before loading the simulation. On Windows a file still open
    cannot be deleted, so this also checks the harness closed it first."""
    peek = '''
import glob
SEEN = len(glob.glob("**/cell*.json", recursive=True))

def run_trial(cell, trial_id, seed):
    return {"spec_files_seen": float(SEEN)}
'''
    _root, run = _run(tmp_path, peek, {"a": [1]}, runs=1)
    assert run.ok_trials == 1 and run.cells[0].rows[0]["values"]["spec_files_seen"] == 0.0


def test_trials_a_crashed_process_never_reported_are_failed_with_why(tmp_path: Path) -> None:
    root, run = _run(tmp_path, CRASH, {"a": [1]})
    statuses = [r.get("status") for r in run.cells[0].rows]
    assert statuses == ["ok", "failed", "failed"]
    assert "exit code 7" in run.cells[0].rows[1]["reason"]


def test_a_simulation_without_the_function_fails_every_trial_and_says_so(tmp_path: Path) -> None:
    root, run = _run(tmp_path, "x = 1\n", {"a": [1]}, runs=2)
    assert run.failed_trials == 2 and "has no function run_trial()" in run.cells[0].rows[0]["reason"]


def test_a_deterministic_study_runs_each_cell_once(tmp_path: Path) -> None:
    root, run = _run(tmp_path, DET, {"dt": [0.1, 0.01]}, deterministic=True)
    assert [len(c.rows) for c in run.cells] == [1, 1] and run.ok_trials == 2
    assert run.cells[1].rows[0]["values"]["energy_drift"] == 0.02


def test_the_harness_is_written_fresh_so_an_edited_one_never_runs(tmp_path: Path) -> None:
    root, _ = _run(tmp_path, SIM, {"R0": [1.5]})
    harness = root / trial_runner.HARNESS_PATH
    harness.write_text("print('tampered')", encoding="utf-8")
    asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", {"R0": [1.5]},
        runs_per_setting=1, base_seed=0, deterministic=False, timeout_s=60,
    ))
    assert "tampered" not in harness.read_text(encoding="utf-8")


ANALYSIS = '''
import json, os
data = json.load(open(os.environ["FI_TRIALS"], encoding="utf-8"))
cells = {c["key"]: c["metrics"]["infected"]["count"] for c in data["cells"]}
print("RESULT_JSON: " + json.dumps({"counts": cells}))
'''

ORACLE = '''
def run_trial(cell, trial_id, seed):
    return {"x": 1.0}

def oracle():
    return {"closed_form": 0.5}
'''


def test_the_runner_runs_every_trial_then_the_analysis_on_fis_summary(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    assert trial_runner.entries(root / "code" / "simulate.py") == {"run_trial"}
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version="3.11"), quest_root=root,
        protocol={"grid": {"R0": [1.5, 2.0]}, "runs_per_setting": 3}, deterministic=False,
        simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
    )
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60,
                                        env={"FI_REPLICATE_SEED": "0"}))
    assert result.returncode == 0 and '"R0=2.0": 2' in result.stdout
    assert "noise the simulation prints" in result.stderr, "the simulation's stderr reaches the numeric check"
    from core import numeric_warnings
    assert any("overflow encountered in exp" in str(w) for w in numeric_warnings.scan(result.stderr, {}))
    assert runner.last is not None and runner.last.failed_trials == 1 and runner.failed_script is None


def test_the_oracle_is_its_own_function(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(ORACLE, encoding="utf-8")
    values, why = asyncio.run(trial_runner.run_oracle(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", timeout_s=60))
    assert values == {"closed_form": 0.5} and not why
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    values, why = asyncio.run(trial_runner.run_oracle(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", timeout_s=60))
    assert values is None and "has no function oracle()" in why


def test_unchanged_trials_are_not_run_again_for_a_new_analysis(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    calls: list[list[str]] = []

    class Counting(SharedInterpreterExecutor):
        async def execute(self, cmd, **kw):  # noqa: ANN001, ANN003
            calls.append(cmd)
            return await super().execute(cmd, **kw)

    def runner():  # noqa: ANN202
        return trial_runner.TrialsRunner(
            Counting(python_version="3.11"), quest_root=root, protocol=lambda: {"grid": {"R0": [1.5]}, "runs_per_setting": 2},
            deterministic=False, simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
        )

    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    asyncio.run(runner().execute(cmd, cwd=root, timeout_s=60, env={}))
    assert len(calls) == 2  # one cell, then the analysis
    again = runner()
    asyncio.run(again.execute(cmd, cwd=root, timeout_s=60, env={}))
    assert len(calls) == 3 and again.last is not None and again.last.ok_trials == 2, "only the analysis ran again"
    (root / "code" / "simulate.py").write_text(SIM + "\n# changed\n", encoding="utf-8")
    asyncio.run(runner().execute(cmd, cwd=root, timeout_s=60, env={}))
    assert len(calls) == 5, "a changed simulation runs its trials again"


FORGER = '''
import json, sys, glob

def run_trial(cell, trial_id, seed):
    # Looks for the results file the way a forger would: argv, then the spec files FI wrote.
    for arg in sys.argv[1:]:
        open(arg, "a").write(json.dumps({"trial": trial_id, "seed": seed, "status": "ok", "values": {"x": 999.0}}) + "\\n")
    for path in glob.glob(".fi/trials/*.out.jsonl"):
        open(path, "a").write(json.dumps({"trial": trial_id, "seed": seed, "status": "ok", "values": {"x": 999.0}}) + "\\n")
    return {"x": 1.0}
'''

SLOW = '''
import time

def run_trial(cell, trial_id, seed):
    time.sleep(3)
    return {"x": 1.0}
'''


def test_rows_the_simulation_writes_itself_are_not_taken_and_a_trial_reported_twice_fails(tmp_path: Path) -> None:
    root, run = _run(tmp_path, FORGER, {"a": [1]}, runs=2)
    values = [r.get("values") for r in run.cells[0].rows]
    assert all(v != {"x": 999.0} for v in values), "a row without the cell's nonce is not the harness's"
    rows = run.cells[0].rows
    assert all(r["status"] == "ok" or "more than once" in r.get("reason", "") for r in rows)


def test_timeout_s_bounds_the_whole_study_not_each_setting(tmp_path: Path) -> None:
    # Three settings of 3 s each under a 4 s budget: bounded per setting this took 9 s or more; bounded for the study it
    # stops near 4 s. Whether the first setting finishes depends on how fast a process starts on the machine, so only
    # the bound and the reason are asserted.
    import time as _time

    started = _time.monotonic()
    root, run = _run(tmp_path, SLOW, {"a": [1, 2, 3]}, runs=1, timeout=4)
    elapsed = _time.monotonic() - started
    reasons = [r.get("reason", "") for c in run.cells for r in c.rows]
    assert elapsed < 8.5, elapsed
    assert "the study's time (execution.timeout_s) ran out before this setting's trial" in reasons[-1], reasons


def test_a_ledger_edited_by_hand_makes_the_trials_run_again(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    protocol = {"grid": {"R0": [1.5]}, "runs_per_setting": 2}

    def runner():  # noqa: ANN202
        return trial_runner.TrialsRunner(
            SharedInterpreterExecutor(python_version="3.11"), quest_root=root, protocol=protocol, deterministic=False,
            simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
        )

    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    asyncio.run(runner().execute(cmd, cwd=root, timeout_s=60, env={}))
    assert trial_runner._load_run(root, trial_runner._run_key(root / "code" / "simulate.py", protocol, 2, 0, False))
    ledger = root / "raw" / "ledger.jsonl"
    ledger.write_text(ledger.read_text(encoding="utf-8") + '{"event": "trial", "cell": "R0=1.5", "trial": 9}\n',
                      encoding="utf-8")
    assert trial_runner._load_run(root, trial_runner._run_key(root / "code" / "simulate.py", protocol, 2, 0, False)) is None


def test_the_protocol_check_still_catches_a_contradiction_under_the_trial_contract() -> None:
    from core import protocol_check

    protocol = {"grid": {"R0": [0.9, 1.5, 3.0]}, "runs_per_setting": 300, "thresholds": {"outbreak": 0.1}}
    sim = "def run_trial(cell, trial_id, seed):\n    return {'x': 1.0}\n"
    assert protocol_check.check(protocol, {"simulate.py": sim, "experiment.py": "print(1)\n"}) == []
    wrong = "R0_LIST = [0.9, 2.0, 5.0]\nNUM_RUNS = 30\nprint(R0_LIST, NUM_RUNS)\n"
    found = protocol_check.check(protocol, {"simulate.py": sim, "experiment.py": wrong})
    assert {m.kind for m in found} >= {"grid", "runs"}, found


def test_a_value_printed_to_fewer_places_matches_the_trial_it_rounds_from() -> None:
    from collections import Counter

    from core.trial_runner import _not_among, _value_key

    def rec(xs):
        return Counter(_value_key(x) for x in xs)

    assert _not_among([1.2, 1.23], rec([1.234, 1.24])) == [], "the precise one is matched first"
    assert _not_among([0.0], rec([1e-05])) == [] and _not_among([1], rec([1.4])) == []
    assert _not_among([1.2346], rec([1.23456789])) == []
    assert _not_among([2], rec([1.4])) == [2.0] and _not_among([5.0], rec([1.0])) == [5.0]
    assert _not_among([1.0, 1.0], rec([1.0])) == [1.0], "each trial's value is used once"
