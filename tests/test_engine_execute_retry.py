"""Test the engine's one-shot retry on suspicious-fast-fail in _node_execute.

Triggered when execute() returns rc != 0 with duration < 0.5 s and no stdout —
the "process never started" signature observed twice on Windows after a
fresh `pip install`. A second attempt usually succeeds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import Engine
from core.execution import ExecutionResult


_WARMUP_OK = ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.05)


def test_engine_quest_root_is_absolute(tmp_path: Path) -> None:
    """Regression: prior live runs failed with rc=2 fast-fail because
    `quest_root` was relative (`./outputs/<id>`), so `cwd=quest_root` +
    a relative argv path produced a duplicated `<quest_root>/<quest_root>/...`
    path. Engine.__init__ must `.resolve()` quest_root unconditionally."""
    cfg = Config(
        topic="abs-path test",
        title="abs",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        # Intentionally relative output_dir to reproduce the bug.
        output=OutputConfig(output_dir=Path("outputs_relative")),
    )
    eng = Engine(cfg)
    assert eng.quest_root.is_absolute(), (
        f"quest_root must be absolute; got {eng.quest_root!r}"
    )


def _mock_execute_router(real_script_result, warmup_result=_WARMUP_OK):
    """Route `executor.execute` mock calls: any invocation whose argv
    contains `-c` is treated as the warmup, anything else is the real
    experiment script. Returns the per-call AsyncMock to assign to
    `eng.executor.execute`. Use `.real_calls` on the returned mock to
    count just the real-script invocations.
    """
    async def fn(cmd, *_, **__):
        # cmd is a list like [str(py), "-c", "pass"] for warmup
        # or [str(py), <script_path>] for the real run.
        if len(cmd) >= 2 and cmd[1] == "-c":
            return warmup_result
        return real_script_result if not isinstance(real_script_result, list) else real_script_result.pop(0)
    return AsyncMock(side_effect=fn)


def _mk_engine(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="execute-retry test",
        title="retry",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    # Stub the LLM client (any node calling self._chat would otherwise fail).
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="{}")})()
    # Ensure quest_root + subdirs exist so _node_execute can write/list.
    for sub in ("code", "figures"):
        (eng.quest_root / sub).mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text("print('hi')", encoding="utf-8")
    return eng


def _count_real_calls(mock: AsyncMock) -> int:
    """Number of `executor.execute` calls that targeted the experiment
    script (excludes the warmup invocations)."""
    n = 0
    for call in mock.await_args_list:
        cmd = call.args[0]
        if len(cmd) < 2 or cmd[1] != "-c":
            n += 1
    return n


@pytest.mark.asyncio
async def test_execute_retries_once_on_suspicious_fast_fail(tmp_path: Path) -> None:
    eng = _mk_engine(tmp_path)
    fast_fail = ExecutionResult(returncode=2, stdout="", stderr="", duration_s=0.02)
    happy = ExecutionResult(
        returncode=0, stdout="RESULT_JSON: {}", stderr="", duration_s=1.5
    )
    eng.executor.install = AsyncMock(  # type: ignore[method-assign]
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    # Real script: first fast-fails, then succeeds. Warmup always OK.
    real_results = [fast_fail, happy]
    eng.executor.execute = _mock_execute_router(real_results)  # type: ignore[method-assign]

    update = await eng._node_execute({"deps": []})

    assert _count_real_calls(eng.executor.execute) == 2, "expected exactly one retry"
    assert update["exec_result"]["returncode"] == 0
    assert update["exec_result"]["duration_s"] == 1.5


@pytest.mark.asyncio
async def test_execute_does_not_retry_when_duration_is_normal(tmp_path: Path) -> None:
    """A normal failure (took real wall time, has stderr) is the script's
    own fault — don't waste a retry."""
    eng = _mk_engine(tmp_path)
    real_fail = ExecutionResult(
        returncode=1, stdout="", stderr="Traceback...", duration_s=2.0
    )
    eng.executor.install = AsyncMock(
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    eng.executor.execute = _mock_execute_router(real_fail)

    await eng._node_execute({"deps": []})

    assert _count_real_calls(eng.executor.execute) == 1, "must NOT retry a real failure"


@pytest.mark.asyncio
async def test_execute_does_not_retry_on_timeout(tmp_path: Path) -> None:
    """Timeouts are intentional limits — retrying just burns the wall clock."""
    eng = _mk_engine(tmp_path)
    timed_out = ExecutionResult(
        returncode=-1, stdout="", stderr="", duration_s=60.0, timed_out=True
    )
    eng.executor.install = AsyncMock(
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    eng.executor.execute = _mock_execute_router(timed_out)

    update = await eng._node_execute({"deps": []})

    assert _count_real_calls(eng.executor.execute) == 1
    assert update["exec_result"]["timed_out"] is True


@pytest.mark.asyncio
async def test_execute_retries_fast_fail_even_with_stderr(
    tmp_path: Path,
) -> None:
    """The Windows venv race sometimes writes a DLL-load message to
    stderr while still exiting in < 0.5 s and never reaching user code.
    The retry must fire on that signature. A truly deterministic script
    error (e.g., bona fide ImportError) will fail the same way on the
    second attempt, costing one extra short invocation but not changing
    the outcome — an acceptable trade for catching the transient case."""
    eng = _mk_engine(tmp_path)
    fast_fail_with_stderr = ExecutionResult(
        returncode=1,
        stdout="",
        stderr="ImportError: DLL load failed while importing _multiarray_umath",
        duration_s=0.18,
    )
    happy = ExecutionResult(
        returncode=0, stdout="RESULT_JSON: {}", stderr="", duration_s=1.2
    )
    eng.executor.install = AsyncMock(
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    eng.executor.execute = _mock_execute_router([fast_fail_with_stderr, happy])

    update = await eng._node_execute({"deps": []})

    # 2 real-script calls (fast-fail then retry succeeds).
    assert _count_real_calls(eng.executor.execute) == 2
    assert update["exec_result"]["returncode"] == 0


@pytest.mark.asyncio
async def test_execute_does_not_retry_when_duration_normal_with_stderr(
    tmp_path: Path,
) -> None:
    """A real wall-time failure (duration >> 0.5 s, real stderr) is the
    script's own fault and not a transient race — retry would just
    double the wait."""
    eng = _mk_engine(tmp_path)
    long_real_fail = ExecutionResult(
        returncode=1,
        stdout="",
        stderr="Traceback (most recent call last):\n  ...\nValueError: bad",
        duration_s=3.5,
    )
    eng.executor.install = AsyncMock(
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    eng.executor.execute = _mock_execute_router(long_real_fail)

    await eng._node_execute({"deps": []})

    assert _count_real_calls(eng.executor.execute) == 1


@pytest.mark.asyncio
async def test_execute_warmup_retries_on_fast_fail_then_runs_real_script(
    tmp_path: Path,
) -> None:
    """The venv warmup itself can hit the rc=2 fast-fail race. The
    warmup should retry once and then proceed to the real script even
    if the warmup eventually returns success or failure — the warmup's
    job is just to consume the race, not gate the experiment."""
    eng = _mk_engine(tmp_path)

    warmup_attempts: list[ExecutionResult] = [
        ExecutionResult(returncode=2, stdout="", stderr="", duration_s=0.02),
        ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.06),
    ]
    real = ExecutionResult(
        returncode=0, stdout="RESULT_JSON: {}", stderr="", duration_s=2.5
    )

    async def router(cmd, *_, **__):
        if len(cmd) >= 2 and cmd[1] == "-c":
            return warmup_attempts.pop(0)
        return real

    eng.executor.install = AsyncMock(
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    eng.executor.execute = AsyncMock(side_effect=router)

    update = await eng._node_execute({"deps": []})

    # 2 warmup calls (one retry) + 1 real script call.
    assert eng.executor.execute.await_count == 3
    assert _count_real_calls(eng.executor.execute) == 1
    assert update["exec_result"]["returncode"] == 0


@pytest.mark.asyncio
async def test_execute_warmup_skips_retry_when_warmup_succeeds_first(
    tmp_path: Path,
) -> None:
    """Happy path: warmup succeeds on first try, no retry needed; real
    script then runs."""
    eng = _mk_engine(tmp_path)
    eng.executor.install = AsyncMock(
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    happy_script = ExecutionResult(
        returncode=0, stdout="RESULT_JSON: {}", stderr="", duration_s=1.0
    )
    eng.executor.execute = _mock_execute_router(happy_script)

    await eng._node_execute({"deps": []})

    # 1 warmup + 1 real script — no retries anywhere.
    assert eng.executor.execute.await_count == 2
