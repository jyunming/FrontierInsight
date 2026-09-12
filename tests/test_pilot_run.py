"""Pilot pass — a cheap smoke test of the DESIGN before the full run.

execute_reflect already repairs a script that crashes. It cannot catch one
that runs fine and answers the wrong question (a sweep over the wrong range,
a resolution too coarse to show the effect), which otherwise costs the full
timeout to discover. The pilot's numbers are discarded; only its signal is
used, and it never fails a quest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine


def _er(returncode: int = 0, stdout: str = 'RESULT_JSON: {"rmse": 0.1}\n',
        stderr: str = ""):
    return type("ER", (), {
        "returncode": returncode, "duration_s": 0.1, "timed_out": False,
        "stdout": stdout, "stderr": stderr,
    })()


def _engine(tmp_path: Path, *, pilot: bool, timeout_s: int = 60) -> Engine:
    cfg = Config(
        topic="pilot smoke", title="pilot",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
            execute_replicates=1, pilot_run=pilot,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=timeout_s),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.executor.install = AsyncMock(  # type: ignore[method-assign]
        return_value=type("IR", (), {"returncode": 0, "stderr": ""})())
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text("# fake\n", encoding="utf-8")
    return eng


@pytest.mark.asyncio
async def test_pilot_off_runs_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pilot=False)
    eng.executor.execute = AsyncMock(return_value=_er())  # type: ignore[method-assign]
    await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_pilot_runs_first_with_the_env_var(tmp_path: Path) -> None:
    """The pilot is a separate, earlier invocation carrying FI_PILOT=1."""
    eng = _engine(tmp_path, pilot=True)
    envs: list[dict] = []

    async def rec(cmd, **kw):
        envs.append(dict(kw.get("env") or {}))
        return _er()

    eng.executor.execute = AsyncMock(side_effect=rec)  # type: ignore[method-assign]
    await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 2, "pilot + full run"
    assert envs[0].get("FI_PILOT") == "1"
    assert "FI_PILOT" not in envs[1], "the real run must not be a pilot"


@pytest.mark.asyncio
async def test_pilot_uses_a_shorter_timeout(tmp_path: Path) -> None:
    """A pilot that takes as long as the real run buys nothing."""
    eng = _engine(tmp_path, pilot=True, timeout_s=600)
    timeouts: list[float] = []

    async def rec(cmd, **kw):
        timeouts.append(kw.get("timeout_s"))
        return _er()

    eng.executor.execute = AsyncMock(side_effect=rec)  # type: ignore[method-assign]
    await eng._node_execute({"deps": []})
    assert timeouts[0] < timeouts[1]
    assert timeouts[0] == pytest.approx(120)     # 600 * 0.2


@pytest.mark.asyncio
async def test_pilot_results_are_discarded(tmp_path: Path) -> None:
    """The pilot is a smoke test, not a measurement: the reported result must
    come from the full run."""
    eng = _engine(tmp_path, pilot=True)
    responses = [
        _er(stdout='RESULT_JSON: {"rmse": 999.0}\n'),   # pilot
        _er(stdout='RESULT_JSON: {"rmse": 0.25}\n'),    # real
    ]
    eng.executor.execute = AsyncMock(side_effect=responses)  # type: ignore[method-assign]
    patch = await eng._node_execute({"deps": []})
    assert patch["result_json"] == {"rmse": 0.25}


@pytest.mark.asyncio
async def test_failing_pilot_does_not_stop_the_full_run(tmp_path: Path) -> None:
    """A bad pilot is a warning, not a gate — execute_reflect still gets its
    chance to repair a real fault with the traceback it needs."""
    eng = _engine(tmp_path, pilot=True)
    responses = [
        _er(returncode=1, stdout="", stderr="boom"),    # pilot crashes
        _er(stdout='RESULT_JSON: {"rmse": 0.25}\n'),    # real run fine
    ]
    eng.executor.execute = AsyncMock(side_effect=responses)  # type: ignore[method-assign]
    patch = await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 2
    assert patch["result_json"] == {"rmse": 0.25}


@pytest.mark.asyncio
async def test_pilot_exception_is_swallowed(tmp_path: Path) -> None:
    """A pilot that cannot even be launched must not abort the quest."""
    eng = _engine(tmp_path, pilot=True)
    calls = {"n": 0}

    async def flaky(cmd, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("could not spawn")
        return _er(stdout='RESULT_JSON: {"rmse": 0.3}\n')

    eng.executor.execute = AsyncMock(side_effect=flaky)  # type: ignore[method-assign]
    patch = await eng._node_execute({"deps": []})
    assert patch["result_json"] == {"rmse": 0.3}


@pytest.mark.asyncio
async def test_out_of_range_pilot_warns_about_the_design(tmp_path: Path) -> None:
    """The pilot's real value: a crash-free run whose numbers are impossible
    usually means the DESIGN is wrong (range, units), not the code.

    Captured off the engine's own logger rather than caplog — the engine logs
    through a quest-scoped logger that does not propagate to the root.
    """
    import logging

    eng = _engine(tmp_path, pilot=True)
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    eng._log.addHandler(_Capture())
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(stdout='RESULT_JSON: {"rmse": -5.0}\n'),
        _er(stdout='RESULT_JSON: {"rmse": 0.2}\n'),
    ])
    # Schema per core.plausibility.parse_assertions: `path`, `min`/`max`.
    design = {"result_assertions": [
        {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
    ]}
    await eng._node_execute({"deps": [], "design": design})
    blob = "\n".join(records)
    assert "pilot produced out-of-range" in blob
    assert "DESIGN is wrong" in blob
