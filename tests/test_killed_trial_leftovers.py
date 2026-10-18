"""A trial FI stopped for its time did not finish, so a file it was writing is not trusted by the next trial.

The toy simulation caches a costly shared computation across trials behind a lock file it removes in a ``finally``;
a trial FI kills never reaches the ``finally``, so the lock stays and the next trial waits on it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core import outside_run
from core import run_estimate as est
from core import trial_runner as tr
from core.execution import SharedInterpreterExecutor

SIM = '''
import os, time

CACHE = os.path.join("data", "results", "cache")


def _shared(key, fn, wait_s=4.0):
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        return open(path).read()
    os.makedirs(CACHE, exist_ok=True)
    lock = path + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        t = time.time()
        while time.time() - t < wait_s:
            if os.path.exists(path):
                return open(path).read()
            time.sleep(0.2)
        raise RuntimeError("timed out waiting for shared computation")
    try:
        v = fn()
        open(path, "w").write(v)
    finally:
        os.remove(lock)
    return v


def _compute(n):
    def fn():
        time.sleep(30 if n == 1 else 0)
        return "7"
    return fn


def run_trial(cell, trial, seed):
    return {"y": float(_shared("kind_a", _compute(cell["n"])))}
'''

SLOW, QUICK = {"n": 1}, {"n": 2}


def _quest(base: Path) -> Path:
    root = base / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "table.csv").write_text("a\n1\n", encoding="utf-8")
    return root


def _time(root: Path, cell: dict, timeout_s: int):  # noqa: ANN202
    return asyncio.run(est._time_one(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                                     "code/simulate.py", cell, runs=1, deterministic=False, timeout_s=timeout_s,
                                     env=None, thresholds=None))


def _small(root: Path, cell: dict, timeout_s: int):  # noqa: ANN202
    return asyncio.run(tr._run_one(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                                   "code/simulate.py", entry="run_trial", cell=cell, seed=1, timeout_s=timeout_s,
                                   env=None, thresholds=None, label="the trial"))


def test_a_timed_trial_killed_mid_computation_leaves_nothing_the_next_trial_waits_on(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    probe, why = _time(root, SLOW, 3)
    assert probe is not None and probe.at_least and why == "", "the first trial ran out of its time"
    probe, why = _time(root, QUICK, 20)
    assert why == "" and probe is not None and not probe.at_least, f"the next trial must not wait on a leftover: {why}"
    assert (root / "data" / "table.csv").exists(), "what the quest had before is kept"


def test_a_small_run_trial_killed_mid_computation_leaves_nothing_the_next_trial_waits_on(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    values, why = _small(root, SLOW, 3)
    assert values is None and "out of time" in why
    values, why = _small(root, QUICK, 20)
    assert why == "" and values == {"y": 7.0}, why


def test_a_file_a_finished_trial_wrote_is_kept(tmp_path: Path) -> None:
    timed = _quest(tmp_path / "a")
    probe, why = _time(timed, QUICK, 20)
    assert why == "" and probe is not None
    assert (timed / "data" / "results" / "cache" / "kind_a.json").read_text() == "7"
    small = _quest(tmp_path / "b")
    values, why = _small(small, QUICK, 20)
    assert why == "" and values == {"y": 7.0}
    assert (small / "data" / "results" / "cache" / "kind_a.json").read_text() == "7"


def test_a_killed_trials_leftovers_are_found_by_what_was_not_there_before(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    before = outside_run.working_files(root)
    (root / "data" / "made.tmp").write_text("x")
    (root / "raw").mkdir()
    (root / "raw" / "ledger.jsonl").write_text("x")
    (root / ".fi").mkdir()
    (root / ".fi" / "run.log").write_text("x")
    gone = outside_run.remove_created_since(root, before)
    assert gone == ["data/made.tmp"]
    assert (root / "raw" / "ledger.jsonl").exists() and (root / ".fi" / "run.log").exists()
    assert (root / "data" / "table.csv").exists()


def test_a_leftover_fi_cannot_remove_is_said(tmp_path: Path, monkeypatch, caplog) -> None:  # noqa: ANN001
    """On Windows a handle of the stopped process can outlive it: the removal is tried again, and a file still held is
    named in the log rather than left behind without a trace."""
    import logging

    root = _quest(tmp_path)
    before = outside_run.working_files(root)
    (root / "data" / "held.lock").write_text("x")
    real_unlink = Path.unlink

    def held(self: Path, missing_ok: bool = False) -> None:
        if self.name == "held.lock":
            raise PermissionError("in use")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held)
    left: list[str] = []
    assert outside_run.remove_created_since(root, before, left) == [] and left == ["data/held.lock"]
    with caplog.at_level(logging.WARNING):
        outside_run.say_removed(logging.getLogger("t"), [], left)
    assert "could not remove (data/held.lock)" in caplog.text


class _Writes:
    """An executor whose run writes a file in its working folder and is (or is not) stopped for its time."""

    def __init__(self, timed_out: bool) -> None:
        self.timed_out = timed_out

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
        from core.execution import ExecutionResult
        (Path(cwd) / "data" / "x.lock").write_text("x")
        (Path(cwd) / ".fi").mkdir(exist_ok=True)
        (Path(cwd) / ".fi" / "out.jsonl").write_text("row\n")
        return ExecutionResult(returncode=-9 if self.timed_out else 0, stdout="", stderr="", duration_s=1.0,
                               timed_out=self.timed_out)


def test_a_copy_run_stopped_for_its_time_brings_back_only_what_fi_named(tmp_path: Path) -> None:
    for timed_out in (True, False):
        root = _quest(tmp_path / str(timed_out))
        (root / ".fi").mkdir()
        seen: list[str] = []

        class _Log:
            def warning(self, msg, *a):  # noqa: ANN001, ANN202
                seen.append(msg % a)

        ex = outside_run.OutsideExecutor(_Writes(timed_out), root, base=tmp_path / "s", log=_Log())
        asyncio.run(ex.execute(["python", ".fi/out.jsonl"], cwd=root, timeout_s=5))
        assert (root / ".fi" / "out.jsonl").exists(), "the result file FI named is kept either way"
        assert (root / "data" / "x.lock").exists() is (not timed_out)
        assert bool(seen) is timed_out and (not timed_out or "data/x.lock" in seen[0]), seen
