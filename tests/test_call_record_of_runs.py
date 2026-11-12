"""FI's record of the functions its checking runs saw called (core/call_record.py).

The record is made only in FI's checking runs (a check's run of the simulation, the small run, the analysis gate's small copy,
a finding's failing test), never in the study's own trials.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from core import call_record, trial_runner
from core.execution import SharedInterpreterExecutor

# A neutral study: the trial calls a helper that defines and calls an inner function; one function is only used for a large
# setting; one inner function is never called; one function is called by a name chosen at run time.
SIM = '''
import sys


def run_trial(cell, trial_id, seed):
    total = tally(cell["size"])
    if cell["size"] > 100:
        total += big_only(cell["size"])
    return {"total": total + dispatch("lookup_step")}


def tally(n):
    def square(x):
        return x * x

    def never_used(x):
        return -x

    return sum(square(i) for i in range(n))


def big_only(n):
    return n


def lookup_step():
    return 1


def dispatch(name):
    return dispatch.__globals__[name]()


class Box:
    def grow(self):
        return 1
'''


def _quest(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    for name, text in {"simulate.py": SIM, **(files or {})}.items():
        (root / "code" / name).write_text(text, encoding="utf-8")
    return root


def _executor() -> SharedInterpreterExecutor:
    return SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")


def _check_run(root: Path, size: int = 3) -> None:
    rows, _ = asyncio.run(trial_runner._run_batch(
        _executor(), sys.executable, root, "code/simulate.py", entry="run_trial", cell={"size": size},
        trials=[{"trial": 0, "seed": 1}], timeout_s=60, env=None, thresholds=None))
    assert rows and rows[0]["status"] == "ok", rows


def test_a_checking_run_notes_the_functions_called_including_nested_ones(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    _check_run(root)
    seen = call_record.load(root)
    assert seen is not None
    assert {("simulate.py", "run_trial"), ("simulate.py", "tally"), ("simulate.py", "tally.<locals>.square"),
            ("simulate.py", "lookup_step")} <= seen
    assert ("simulate.py", "tally.<locals>.never_used") not in seen and ("simulate.py", "big_only") not in seen
    assert ("simulate.py", "Box.grow") not in seen
    assert not list((root / ".fi" / "calls").glob("*.json"))  # each run's own file is moved into the one record


def test_the_study_s_own_trials_are_not_recorded(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", {"size": [3, 4]}, runs_per_setting=2, base_seed=0,
        deterministic=False, timeout_s=60, run_id="r1"))
    assert call_record.load(root) is None
    assert not (root / ".fi" / "call_record.py").exists() and not (root / ".fi" / "calls").exists()


def test_the_runner_keeps_the_programs_exit_code_and_arguments(tmp_path: Path) -> None:
    root = _quest(tmp_path, {"prog.py": "import sys\n\ndef f():\n    return 1\n\nf()\nsys.exit(int(sys.argv[1]))\n"})
    runner, out = call_record.prepare(root, "x")
    done = subprocess.run([sys.executable, runner, out.as_posix(), "code/prog.py", "5"], cwd=root, capture_output=True)
    assert done.returncode == 5
    call_record.absorb(root, root / out)
    assert ("prog.py", "f") in (call_record.load(root) or set())
