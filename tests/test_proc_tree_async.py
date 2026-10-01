"""A timed-out or cancelled experiment script or CLI call stops together with every process it started.

``core/execution.py`` (the experiment script) and ``core/provider.py`` (the CLI providers) used to kill only the
program they started. A script that started its own subprocess or ``multiprocessing`` worker, or a CLI that started a
helper, left those running after the timeout: still computing, holding files, and able to write results later. Here
every program FI starts launches helpers that each add a byte to a ``beat`` file every 0.1 s and, if still alive at a
set moment after the timeout, write a ``late`` marker. After the stop the beats must stop growing, the helpers' pids
must be gone, and no marker may ever appear.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from core.execution import VenvExecutor
from core.proc_tree import AsyncProcessTree, ProcessTree
from core.provider import _CLI_SPECS, ProxySupervisor, _CliTransientError, _ProxyHandle, _run_cli

# argv: folder, name, late_at (epoch seconds). Beats every 0.1 s; writes ``<name>.late`` if alive at ``late_at``.
_HELPER = textwrap.dedent(
    """
    import os, pathlib, sys, time

    def run(folder, name, late_at):
        folder = pathlib.Path(folder)
        (folder / (name + ".pid")).write_text(str(os.getpid()))
        end = time.monotonic() + 90  # a helper that survives a failed test does not run for ever
        with open(folder / (name + ".beat"), "ab", buffering=0) as beat:
            while time.monotonic() < end:
                beat.write(b".")
                if time.time() >= late_at:
                    (folder / (name + ".late")).write_text("still running after the stop")
                    return
                time.sleep(0.1)

    if __name__ == "__main__":
        run(sys.argv[1], sys.argv[2], float(sys.argv[3]))
    """
)

# The experiment script / fake CLI. argv: helper script, folder, late_at, mode.
# mode "stay": starts a subprocess helper and a multiprocessing worker, then waits silently.
# mode "exit": starts a subprocess helper that inherits its output pipe, and exits as soon as the helper runs: the
# caller is then still waiting on a pipe the helper holds, with the program itself already gone.
_PROGRAM = textwrap.dedent(
    """
    import multiprocessing, pathlib, subprocess, sys, time

    def worker(helper, folder, late_at):
        sys.path.insert(0, str(pathlib.Path(helper).parent))
        import helper as h
        h.run(folder, "mp", late_at)

    if __name__ == "__main__":
        helper, folder, late_at, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
        sys.stdin.close()
        subprocess.Popen([sys.executable, helper, folder, "sub", str(late_at)], stdin=subprocess.DEVNULL)
        if mode == "exit":
            end = time.monotonic() + 60
            while not (pathlib.Path(folder) / "sub.pid").exists() and time.monotonic() < end:
                time.sleep(0.05)
            (pathlib.Path(folder) / "program.pid").write_text(str(__import__("os").getpid()))
            print("started", flush=True)
            sys.exit(0)
        multiprocessing.Process(target=worker, args=(helper, folder, late_at)).start()
        print("started", flush=True)
        time.sleep(90)
    """
)

_NAMES = ("sub", "mp")


def _write_programs(tmp_path: Path) -> tuple[Path, Path, Path]:
    folder = tmp_path / "run"
    folder.mkdir()
    helper = tmp_path / "helper.py"
    helper.write_text(_HELPER, encoding="utf-8")
    program = tmp_path / "program.py"
    program.write_text(_PROGRAM, encoding="utf-8")
    return helper, program, folder


def _beats(folder: Path, name: str) -> int:
    beat = folder / f"{name}.beat"
    if not beat.exists():
        return 0
    with open(beat, "rb") as fh:
        return fh.seek(0, os.SEEK_END)


async def _wait_until_beating(folder: Path, names: tuple[str, ...] = _NAMES, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    for name in names:
        while _beats(folder, name) < 3:
            assert time.monotonic() < deadline, f"helper {name} never started beating"
            await asyncio.sleep(0.05)


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


async def _assert_nothing_left(folder: Path, late_at: float, names: tuple[str, ...] = _NAMES) -> None:
    missing = [name for name in names if not (folder / f"{name}.pid").exists()]
    assert not missing, f"helper(s) {missing} never started, so the test proves nothing about them"
    pids = {name: int((folder / f"{name}.pid").read_text()) for name in names}
    before = {name: _beats(folder, name) for name in names}
    await asyncio.sleep(1.5)  # fifteen beats of a helper that survived
    assert {name: _beats(folder, name) for name in names} == before, "a helper is still beating after the stop"
    assert not [name for name, pid in pids.items() if _alive(pid)], "a helper's process is still alive"
    # Past the moment a surviving helper would have written its marker.
    while time.time() < late_at + 1.0:
        await asyncio.sleep(0.1)
    assert not list(folder.glob("*.late")), "a helper wrote its result after the stop"


# --- the experiment script (core/execution.py) ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_timed_out_script_stops_with_its_subprocess_and_its_multiprocessing_worker(tmp_path: Path) -> None:
    helper, program, folder = _write_programs(tmp_path)
    timeout_s = 15
    late_at = time.time() + timeout_s + 3
    result = await VenvExecutor().execute(
        [sys.executable, str(program), str(helper), str(folder), str(late_at), "stay"], cwd=tmp_path, timeout_s=timeout_s,
    )
    assert result.timed_out is True
    assert result.returncode != 0
    assert result.duration_s < timeout_s + 30  # the stop did not wait on the helpers' open output pipes
    await _assert_nothing_left(folder, late_at)


@pytest.mark.asyncio
async def test_a_cancelled_script_stops_with_everything_it_started(tmp_path: Path) -> None:
    helper, program, folder = _write_programs(tmp_path)
    late_at = time.time() + 60
    task = asyncio.create_task(VenvExecutor().execute(
        [sys.executable, str(program), str(helper), str(folder), str(late_at), "stay"], cwd=tmp_path, timeout_s=300,
    ))
    await _wait_until_beating(folder)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_nothing_left(folder, time.time())


@pytest.mark.asyncio
async def test_a_cancelled_script_that_already_exited_still_stops_the_helper_holding_its_output(
    tmp_path: Path,
) -> None:
    """The script has exited; the call still waits on the output pipe its helper holds. Cancelling then must stop the
    helper too, although the script itself is no longer running (on POSIX nothing else would)."""
    helper, program, folder = _write_programs(tmp_path)
    late_at = time.time() + 60
    task = asyncio.create_task(VenvExecutor().execute(
        [sys.executable, str(program), str(helper), str(folder), str(late_at), "exit"], cwd=tmp_path, timeout_s=300,
    ))
    await _wait_until_beating(folder, ("sub",))
    deadline = time.monotonic() + 60
    pid_file = folder / "program.pid"
    while True:
        pid = pid_file.read_text().strip() if pid_file.exists() else ""
        if pid and not _alive(int(pid)):
            break
        assert time.monotonic() < deadline, "the script never exited"
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.3)  # let asyncio see the exit
    if os.name != "nt":
        # On POSIX the helper holds the inherited output pipe, so the call is still waiting on it. (On Windows the
        # helper may not inherit it; the call then returns and closing the job stops the helper.)
        assert not task.done()
    if not task.done():
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    await _assert_nothing_left(folder, time.time(), ("sub",))


@pytest.mark.asyncio
async def test_a_timed_out_script_that_already_exited_still_stops_the_helper_holding_its_output(
    tmp_path: Path,
) -> None:
    helper, program, folder = _write_programs(tmp_path)
    timeout_s = 8
    late_at = time.time() + timeout_s + 3
    await VenvExecutor().execute(
        [sys.executable, str(program), str(helper), str(folder), str(late_at), "exit"], cwd=tmp_path,
        timeout_s=timeout_s,
    )
    await _assert_nothing_left(folder, late_at, ("sub",))


@pytest.mark.asyncio
async def test_a_script_that_finishes_keeps_its_output_and_exit_code(tmp_path: Path) -> None:
    result = await VenvExecutor().execute(
        [sys.executable, "-c", "import sys; print('hello'); print('err', file=sys.stderr); sys.exit(3)"],
        cwd=tmp_path, timeout_s=60,
    )
    assert (result.returncode, result.stdout.strip(), result.stderr.strip(), result.timed_out) == (
        3, "hello", "err", False,
    )


# --- the CLI providers (core/provider.py) -----------------------------------------------------------------------------


def _fake_cli(tmp_path: Path, output_via: str) -> tuple[object, Path, float, float]:
    """A CLI provider spec whose binary is ``_PROGRAM``: it starts its helpers and then says nothing."""
    helper, program, folder = _write_programs(tmp_path)
    timeout_s = 12.0
    late_at = time.time() + timeout_s + 4
    spec = dataclasses.replace(
        _CLI_SPECS["claude_cli"], argv=(sys.executable, str(program), str(helper), str(folder), str(late_at), "stay"),
        output_via=output_via,
    )
    return spec, folder, timeout_s, late_at


@pytest.mark.asyncio
@pytest.mark.parametrize("output_via", ["stream_json", "stdout"])
async def test_a_timed_out_cli_stops_with_the_helpers_it_started(tmp_path: Path, output_via: str) -> None:
    spec, folder, timeout_s, late_at = _fake_cli(tmp_path, output_via)
    with pytest.raises(_CliTransientError, match="wall-clock"):
        await _run_cli(spec, "hi", timeout_s=timeout_s, inactivity_timeout_s=600)
    await _assert_nothing_left(folder, late_at)


@pytest.mark.asyncio
@pytest.mark.parametrize("output_via", ["stream_json", "stdout"])
async def test_a_cancelled_cli_call_stops_with_the_helpers_it_started(tmp_path: Path, output_via: str) -> None:
    spec, folder, _timeout_s, _late_at = _fake_cli(tmp_path, output_via)
    task = asyncio.create_task(_run_cli(spec, "hi", timeout_s=300, inactivity_timeout_s=600))
    await _wait_until_beating(folder)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_nothing_left(folder, time.time())


@pytest.mark.asyncio
async def test_a_cli_that_answers_still_returns_its_answer(tmp_path: Path) -> None:
    stream = tmp_path / "stream.jsonl"
    stream.write_text(json.dumps({"type": "result", "result": "the answer"}) + "\n", encoding="utf-8")
    script = tmp_path / "replay.py"
    script.write_text("import sys\nsys.stdout.write(open(sys.argv[1], encoding='utf-8').read())\n", encoding="utf-8")
    spec = dataclasses.replace(_CLI_SPECS["claude_cli"], argv=(sys.executable, str(script), str(stream)))
    assert await _run_cli(spec, "x", timeout_s=60) == "the answer"


# --- the provider proxies --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_releasing_a_proxy_stops_the_server_its_launcher_started(tmp_path: Path) -> None:
    helper, program, folder = _write_programs(tmp_path)
    late_at = time.time() + 60
    tree = ProcessTree([sys.executable, str(program), str(helper), str(folder), str(late_at), "stay"])
    sup = ProxySupervisor()
    sup._spawn = lambda key: _ProxyHandle(name=key, port=1, proc=tree.proc, tree=tree)  # type: ignore[method-assign]
    await sup.acquire("claude_code")
    await _wait_until_beating(folder)
    await sup.release("claude_code")
    assert tree.proc.returncode is not None
    await _assert_nothing_left(folder, time.time())


# --- stand-ins ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stand_in_process_gets_no_job_or_group_even_with_an_int_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test's mock with a made-up pid must never reach OpenProcess or killpg, where the pid could be a real,
    unrelated process."""
    from unittest.mock import AsyncMock, MagicMock

    proc = MagicMock()
    proc.pid = os.getpid()  # the worst case: a pid that exists
    proc.returncode = None
    proc.wait = AsyncMock(return_value=-9)
    from core import proc_tree

    reached: list[str] = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=proc))
    if os.name == "nt":
        monkeypatch.setattr(proc_tree, "_adopt_suspended", lambda *a: reached.append("OpenProcess"))
    else:
        monkeypatch.setattr(proc_tree.os, "killpg", lambda *a: reached.append("killpg"))
    tree = await AsyncProcessTree.start("whatever")
    assert tree._job is None  # noqa: SLF001
    assert await tree.kill() is True
    proc.kill.assert_called()
    await tree.aclose(aborted=True)
    assert reached == []
