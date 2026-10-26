"""The run environment fails in the open: a process stopped from outside is a recorded fact, a lock it left does not
block the next run, and settings run together only as many as the memory holds."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import execution as ex
from core import outside_run
from core import run_marker as rm
from core import trial_runner as tr
from core.execution import ExecutionResult, SettingsGate, VenvExecutor
from core.outside_run import SHARED_CACHE


# ---------------------------------------------------------------- a command that ends with no error message

def _run_cmd(code: str, tmp_path: Path) -> ExecutionResult:
    return asyncio.run(VenvExecutor().execute([sys.executable, "-c", code], cwd=tmp_path, timeout_s=60))


def test_a_command_that_died_without_an_error_is_said_with_its_exit_code(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    with caplog.at_level(logging.WARNING, logger="frontier_insight.execution"):
        result = _run_cmd("import os; os._exit(7)", tmp_path)
    assert result.returncode == 7
    said = [r.getMessage() for r in caplog.records if "printed no error" in r.getMessage()]
    assert len(said) == 1 and "exit code 7" in said[0], said


def test_a_command_with_a_traceback_or_a_clean_end_is_not_reported_as_silent(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    with caplog.at_level(logging.WARNING, logger="frontier_insight.execution"):
        _run_cmd("raise ValueError('boom')", tmp_path)
        _run_cmd("pass", tmp_path)
    assert not [r for r in caplog.records if "printed no error" in r.getMessage()]


def test_the_most_memory_a_command_held_is_measured(tmp_path: Path) -> None:
    result = _run_cmd("import time; a = bytearray(300 * 1024 * 1024); a[::4096] = b'x' * len(a[::4096]); time.sleep(1.5)",
                      tmp_path)
    assert result.returncode == 0 and result.peak_mb is not None and result.peak_mb > 250, result.peak_mb


def test_plain_exit_names_a_kill_for_memory() -> None:
    assert "memory" in ex.plain_exit(-9) and "memory" in ex.plain_exit(137)
    assert ex.plain_exit(1) == ""


# ---------------------------------------------------------------- the marker: how an earlier session ended

def _dead_pid() -> tuple[int, float]:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid, 1.0e9  # a start time no live process has


def _write_record(root: Path, record: dict[str, Any]) -> None:
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    (root / rm.RECORD).write_text(json.dumps(record), encoding="utf-8")


def _cache_with(root: Path, *names: str) -> None:
    (root / SHARED_CACHE).mkdir(parents=True, exist_ok=True)
    for n in names:
        (root / SHARED_CACHE / n).write_text("x", encoding="utf-8")


def test_a_session_that_was_stopped_from_outside_is_found_said_and_its_locks_removed(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    pid, _ = _dead_pid()
    _write_record(tmp_path, {"pid": pid, "started": 1.0e9, "seen_at": 1.0e9 + 60, "ended": None})
    _cache_with(tmp_path, "design_ab12.json.lock", "table.npz.tmp123", "table.npz")
    marker = rm.RunMarker(tmp_path, beat_s=0.05)
    with caplog.at_level(logging.WARNING):
        found = marker.start()
    marker.finish("finished")
    assert found is not None and found["pid"] == pid
    assert any("ended without recording an end" in r.getMessage() for r in caplog.records)
    left = sorted(p.name for p in (tmp_path / SHARED_CACHE).iterdir())
    assert left == ["table.npz"], left  # the finished value stays; the lock and the temporary file go
    events = [json.loads(x)["event"] for x in (tmp_path / rm.HISTORY).read_text(encoding="utf-8").splitlines()]
    assert events == ["found_stopped", "started", "ended"]
    assert json.loads((tmp_path / rm.RECORD).read_text(encoding="utf-8"))["ended"] == "finished"


def test_a_session_that_ended_cleanly_leaves_everything_as_it_was(tmp_path: Path) -> None:
    marker = rm.RunMarker(tmp_path, beat_s=0.05)
    assert marker.start() is None
    marker.finish("finished")
    _cache_with(tmp_path, "keep.json.lock")  # a lock of some other, running work
    again = rm.RunMarker(tmp_path, beat_s=0.05)
    assert again.start() is None
    again.finish()
    assert (tmp_path / SHARED_CACHE / "keep.json.lock").is_file()


def test_a_record_of_a_process_still_running_is_not_touched(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    started = rm._started_at(os.getpid())
    assert started is not None
    _write_record(tmp_path, {"pid": os.getpid(), "started": started, "seen_at": started, "ended": None})
    _cache_with(tmp_path, "busy.json.lock")
    marker = rm.RunMarker(tmp_path, beat_s=0.05)
    # this very process is "the earlier one": it is running, so nothing is found and nothing removed
    assert rm.is_running(json.loads((tmp_path / rm.RECORD).read_text(encoding="utf-8")))
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        began = rm._started_at(other.pid)
        _write_record(tmp_path, {"pid": other.pid, "started": began, "seen_at": began, "ended": None})
        with caplog.at_level(logging.WARNING):
            found = marker.start()
        marker.finish()
    finally:
        other.kill()
        other.wait()
    assert found is None and (tmp_path / SHARED_CACHE / "busy.json.lock").is_file()
    assert any("already runs in another process" in r.getMessage() for r in caplog.records)


def test_the_beat_moves_the_sign_of_life(tmp_path: Path) -> None:
    import time

    marker = rm.RunMarker(tmp_path, beat_s=0.05)
    marker.start()
    first = json.loads((tmp_path / rm.RECORD).read_text(encoding="utf-8"))["seen_at"]
    time.sleep(0.4)
    later = json.loads((tmp_path / rm.RECORD).read_text(encoding="utf-8"))["seen_at"]
    marker.finish()
    assert later > first


# ---------------------------------------------------------------- the stale lock that hung a run for hours

LOCKING = '''
import os, time, json
CACHE = os.path.join("data", "results", "cache")

def run_cell(cell):
    os.makedirs(CACHE, exist_ok=True)
    value = os.path.join(CACHE, "shared.json")
    lock = value + ".lock"
    waited = 0.0
    while os.path.exists(lock) and not os.path.exists(value):
        time.sleep(0.1)
        waited += 0.1
        if waited > 4:
            raise RuntimeError("waited on a lock for ever")
    if not os.path.exists(value):
        json.dump({"v": 3}, open(value, "w"))
    return {"y": float(cell["n"]) * json.load(open(value))["v"]}
'''


def _quest(base: Path) -> Path:
    root = base / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(LOCKING, encoding="utf-8")
    (root / "data").mkdir()
    outside_run.prepare_shared_cache(root)  # the folder belongs to this code, as in a run already under way
    _cache_with(root, "shared.json.lock")  # left by a trial a stopped session was running
    return root


def _ok_trials(root: Path) -> int:
    n = 0
    for path in (root / ".fi" / "trials").glob("cell*.out.jsonl"):
        n += sum('"status": "ok"' in x for x in path.read_text(encoding="utf-8").splitlines())
    return n


def _study(root: Path) -> None:
    asyncio.run(tr.run_trials(ex.SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                              "code/simulate.py", {"n": [1]}, runs_per_setting=1, base_seed=1, deterministic=True,
                              timeout_s=60))


def test_a_lock_a_stopped_session_left_blocked_the_next_run_before_and_not_after(tmp_path: Path) -> None:
    blocked = _quest(tmp_path / "old")
    _study(blocked)  # nothing removes the lock: the code waits on it and gives up
    assert _ok_trials(blocked) == 0
    pid, _ = _dead_pid()
    freed = _quest(tmp_path / "new")
    _write_record(freed, {"pid": pid, "started": 1.0e9, "seen_at": 1.0e9, "ended": None})
    marker = rm.RunMarker(freed, beat_s=5)
    assert marker.start() is not None
    marker.finish()
    _study(freed)
    assert _ok_trials(freed) == 1


# ---------------------------------------------------------------- settings at once, by memory

def _gate(at_once: int, free: list[float], total: float = 64_000.0) -> SettingsGate:
    return SettingsGate(at_once, free_mb=lambda: free[0], total_mb=total, poll_s=0.01)


def test_with_the_memory_for_all_of_them_every_core_is_used() -> None:
    free = [60_000.0]

    async def go() -> int:
        gate = _gate(8, free)
        gate.RAMP_S = 0.0
        gate.note(1_000.0)

        async def one() -> None:
            async with gate.slot():
                await asyncio.sleep(0.2)

        await asyncio.gather(*(one() for _ in range(8)))
        return gate.most_at_once

    assert asyncio.run(go()) == 8


def test_settings_that_would_not_fit_in_memory_wait_for_one_to_finish(caplog) -> None:  # noqa: ANN001
    # 16 could run at once, each holds 3 GB, 12 GB are free beyond the reserve: at most 3 at a time
    total = 64_000.0
    free = [12_000.0 + 0.1 * total + 3_000.0]  # the room for (reserve + 12 GB)

    async def go() -> SettingsGate:
        gate = _gate(16, free, total)
        gate.RAMP_S = 0.0
        gate._log = logging.getLogger("t")
        gate.note(3_000.0)
        live = {"n": 0, "most": 0}

        async def one() -> None:
            async with gate.slot():
                live["n"] += 1
                live["most"] = max(live["most"], live["n"])
                free[0] -= 3_750.0  # what it takes while it runs
                await asyncio.sleep(0.15)
                free[0] += 3_750.0
                live["n"] -= 1

        await asyncio.gather(*(one() for _ in range(16)))
        gate.live_most = live["most"]  # type: ignore[attr-defined]
        return gate

    with caplog.at_level(logging.WARNING, logger="t"):
        gate = asyncio.run(go())
    assert gate.live_most < 16 and gate.live_most >= 1, gate.live_most  # type: ignore[attr-defined]
    assert gate.record()["held_back"] is True and gate.record()["largest_setting_mb"] == 3_000.0
    assert any("holding settings back" in r.getMessage() for r in caplog.records)


def test_one_setting_always_runs_even_when_no_memory_is_free() -> None:
    free = [10.0]

    async def go() -> int:
        gate = _gate(4, free)
        gate.note(5_000.0)
        done = []

        async def one(k: int) -> None:
            async with gate.slot():
                done.append(k)
                await asyncio.sleep(0.01)

        await asyncio.wait_for(asyncio.gather(*(one(k) for k in range(3))), timeout=20)
        return len(done)

    assert asyncio.run(go()) == 3


def test_nothing_is_measured_so_settings_start_a_little_apart_and_a_blind_gate_uses_the_cores() -> None:
    free = [60_000.0]

    async def go() -> tuple[list[float], int]:
        gate = _gate(4, free)
        gate.STAGGER = 0.3
        starts: list[float] = []
        t0 = asyncio.get_running_loop().time()

        async def one() -> None:
            async with gate.slot():
                starts.append(asyncio.get_running_loop().time() - t0)
                await asyncio.sleep(0.8)

        await asyncio.gather(*(one() for _ in range(3)))
        blind = _gate(4, free)
        blind.note(None)  # a run whose memory could not be measured
        t1 = asyncio.get_running_loop().time()
        seen: list[float] = []

        async def two() -> None:
            async with blind.slot():
                seen.append(asyncio.get_running_loop().time() - t1)
                await asyncio.sleep(0.3)

        await asyncio.gather(*(two() for _ in range(4)))
        return sorted(starts), sum(t < 0.2 for t in seen)

    starts, at_once_blind = asyncio.run(go())
    assert starts[1] - starts[0] >= 0.25 and starts[2] - starts[1] >= 0.25, starts
    assert at_once_blind == 4


def test_run_trials_records_how_many_ran_at_once(tmp_path: Path) -> None:
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {'y': float(cell['n'])}\n",
                                                encoding="utf-8")
    (root / "data").mkdir()
    asyncio.run(tr.run_trials(ex.SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                              "code/simulate.py", {"n": [1, 2, 3, 4]}, runs_per_setting=1, base_seed=1,
                              deterministic=True, timeout_s=120, at_once=2))
    record = json.loads((root / ".fi" / "trials" / "at_once.json").read_text(encoding="utf-8"))
    assert record["could_run_at_once"] == 2 and 1 <= record["most_run_at_once"] <= 2
    assert record["largest_setting_mb"] is None or record["largest_setting_mb"] > 0


# ---------------------------------------------------------------- the engine keeps the marker on every way a run ends

def _quest_engine(tmp_path: Path):  # noqa: ANN202
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine

    cfg = Config(
        topic="a neutral topic about a cooling cup", title="marker", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


def test_a_quest_run_finds_a_session_stopped_from_outside_and_records_how_it_ended(
        tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from core.engine import Engine

    eng = _quest_engine(tmp_path)
    pid, _ = _dead_pid()
    _write_record(eng.quest_root, {"pid": pid, "started": 1.0e9, "seen_at": 1.0e9 + 60, "ended": None})
    _cache_with(eng.quest_root, "design_ab12.json.lock")

    async def _boom(self) -> None:  # noqa: ANN001
        raise RuntimeError("a planted failure right after the start")

    monkeypatch.setattr(Engine, "_check_records_changed_outside", _boom)
    with pytest.raises(RuntimeError):
        asyncio.run(eng.run())
    # The quest's own log (run.log) is where a person reads it.
    assert "ended without recording an end" in (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert not (eng.quest_root / SHARED_CACHE / "design_ab12.json.lock").exists()
    record = json.loads((eng.quest_root / rm.RECORD).read_text(encoding="utf-8"))
    assert record["pid"] == os.getpid() and record["ended"] == "failed: RuntimeError"
