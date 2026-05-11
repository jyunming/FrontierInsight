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
    eng.executor.execute = AsyncMock(side_effect=[fast_fail, happy])  # type: ignore[method-assign]

    update = await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 2, "expected exactly one retry"
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
    eng.executor.execute = AsyncMock(return_value=real_fail)

    await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 1, "must NOT retry a real failure"


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
    eng.executor.execute = AsyncMock(return_value=timed_out)

    update = await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 1
    assert update["exec_result"]["timed_out"] is True


@pytest.mark.asyncio
async def test_execute_does_not_retry_when_some_output_was_produced(
    tmp_path: Path,
) -> None:
    """If the script produced any stdout before failing, the interpreter
    clearly started — don't retry."""
    eng = _mk_engine(tmp_path)
    partial = ExecutionResult(
        returncode=1, stdout="loading...\nERROR\n", stderr="", duration_s=0.3
    )
    eng.executor.install = AsyncMock(
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    eng.executor.execute = AsyncMock(return_value=partial)

    await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_execute_does_not_retry_when_stderr_was_produced(
    tmp_path: Path,
) -> None:
    """A fast failure with stderr text (e.g., an ImportError traceback or
    a missing-DLL message) means the interpreter ran and the script is
    deterministically broken — retrying would just mask the real error."""
    eng = _mk_engine(tmp_path)
    deterministic_import_error = ExecutionResult(
        returncode=1,
        stdout="",
        stderr="ImportError: DLL load failed while importing _ctypes",
        duration_s=0.15,
    )
    eng.executor.install = AsyncMock(
        return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    )
    eng.executor.execute = AsyncMock(return_value=deterministic_import_error)

    await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 1
