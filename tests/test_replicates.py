"""Multi-seed replication tests.

Two paths to exercise:

  1. ``_aggregate_result_json_replicates`` — pure function: mean/std/
     min/max over numeric scalar fields, deliberate skipping of
     non-numeric / non-shared keys.
  2. ``_node_execute`` — replication loop: when
     ``engine.execute_replicates > 1`` AND the primary run succeeds,
     the engine fires N-1 more runs with fresh ``FI_REPLICATE_SEED``
     env vars and aggregates the per-seed ``RESULT_JSON`` outputs.

These tests use the in-process ``VenvExecutor`` against a fake
``experiment.py`` that reads the env var and emits a seed-dependent
``RESULT_JSON``. No real LLM calls.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _aggregate_result_json_replicates


# ---------------------------------------------------------------------------
# _aggregate_result_json_replicates — pure function
# ---------------------------------------------------------------------------


def test_aggregator_empty_input_returns_empty_dict() -> None:
    assert _aggregate_result_json_replicates([]) == {}


def test_aggregator_mean_and_std_over_numeric_keys() -> None:
    """The aggregator computes sample std (n-1 denominator) for any
    scalar numeric field present in EVERY replicate."""
    reps = [
        {"_seed": 0, "rmse": 0.10, "best_method": "RK4"},
        {"_seed": 1, "rmse": 0.20, "best_method": "RK4"},
        {"_seed": 2, "rmse": 0.30, "best_method": "RK2"},
    ]
    agg = _aggregate_result_json_replicates(reps)
    # rmse is scalar-numeric in all 3 → aggregated.
    assert "rmse" in agg
    assert agg["rmse"]["n"] == 3
    assert math.isclose(agg["rmse"]["mean"], 0.20, abs_tol=1e-9)
    assert math.isclose(agg["rmse"]["std"], 0.1, abs_tol=1e-9)
    assert agg["rmse"]["min"] == 0.10
    assert agg["rmse"]["max"] == 0.30
    # best_method is a string in all 3 → skipped.
    assert "best_method" not in agg


def test_aggregator_skips_missing_or_non_numeric_keys() -> None:
    """A key that's missing from any replicate is skipped (we can't
    honestly compute mean over a non-uniform set). A key that's
    numeric in one replicate but string in another is also skipped."""
    reps = [
        {"_seed": 0, "rmse": 0.1, "extra": 1.0},
        {"_seed": 1, "rmse": 0.2},  # no "extra"
        {"_seed": 2, "rmse": 0.3, "extra": "oops"},  # non-numeric
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert "rmse" in agg
    assert agg["rmse"]["n"] == 3
    # extra is mixed-type — skipped.
    assert "extra" not in agg


def test_aggregator_single_replicate_emits_zero_std() -> None:
    """n=1 → std=0 (sample std needs n-1 denominator; we degenerate
    to 0 so downstream code doesn't divide by zero)."""
    reps = [{"_seed": 0, "rmse": 0.42}]
    agg = _aggregate_result_json_replicates(reps)
    assert agg["rmse"]["n"] == 1
    assert agg["rmse"]["std"] == 0.0
    assert agg["rmse"]["mean"] == 0.42
    assert agg["rmse"]["min"] == agg["rmse"]["max"] == 0.42


def test_aggregator_skips_bool_typed_as_numeric() -> None:
    """Python ``True`` and ``False`` pass ``isinstance(_, int)`` because
    of bool's int inheritance. The aggregator explicitly filters bool
    so a ``converged: true`` field doesn't become a 0/1 'mean'."""
    reps = [
        {"_seed": 0, "converged": True},
        {"_seed": 1, "converged": False},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert "converged" not in agg


# ---------------------------------------------------------------------------
# _node_execute — replication loop integration
# ---------------------------------------------------------------------------


def _make_engine_with_replicates(tmp_path: Path, n: int) -> Engine:
    cfg = Config(
        topic="replicates smoke",
        title="replicates",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            execute_replicates=n,
            clarify_mode="off",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


@pytest.mark.asyncio
async def test_execute_runs_once_when_replicates_is_one(tmp_path: Path) -> None:
    """Default replicates=1 → exactly one execute call, no
    ``result_json_replicates`` field on the patch."""
    eng = _make_engine_with_replicates(tmp_path, 1)
    eng.executor.execute = AsyncMock(return_value=type(  # type: ignore[method-assign]
        "ER", (), {
            "returncode": 0, "duration_s": 0.1, "timed_out": False,
            "stdout": 'RESULT_JSON: {"rmse": 0.1}\n', "stderr": "",
        })())
    eng.executor.install = AsyncMock(return_value=type(  # type: ignore[method-assign]
        "IR", (), {"returncode": 0, "stderr": ""})())
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text("# fake\n", encoding="utf-8")

    patch = await eng._node_execute({"deps": []})
    assert "result_json_replicates" not in patch
    assert patch["result_json"] == {"rmse": 0.1}
    # Two execute calls: one warmup (which is skipped for no-deps) + one real.
    # No-deps path skips warmup, so just one execute call.
    assert eng.executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_execute_runs_n_times_with_replicate_env_var(tmp_path: Path) -> None:
    """When replicates=3, the engine runs the script once for the
    primary run THEN twice more with FI_REPLICATE_SEED=1, =2. Each
    seed's RESULT_JSON lands in result_json_replicates."""
    eng = _make_engine_with_replicates(tmp_path, 3)
    # Make each call return a seed-tagged RESULT_JSON. Cycle through
    # three responses; track env vars so we can assert seed plumbing.
    call_envs: list[dict[str, str] | None] = []
    responses = [
        'RESULT_JSON: {"rmse": 0.10}\n',
        'RESULT_JSON: {"rmse": 0.20}\n',
        'RESULT_JSON: {"rmse": 0.30}\n',
    ]
    idx = {"n": 0}

    async def fake_execute(*args, env=None, **kw):  # noqa: ANN001
        call_envs.append(env)
        stdout = responses[idx["n"]]
        idx["n"] += 1
        return type("ER", (), {
            "returncode": 0, "duration_s": 0.1, "timed_out": False,
            "stdout": stdout, "stderr": "",
        })()
    eng.executor.execute = fake_execute  # type: ignore[method-assign]
    eng.executor.install = AsyncMock(return_value=type(  # type: ignore[method-assign]
        "IR", (), {"returncode": 0, "stderr": ""})())
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text("# fake\n", encoding="utf-8")

    patch = await eng._node_execute({"deps": []})

    # 3 execute calls total (primary + 2 replicates). Warmup is skipped
    # because deps is empty.
    assert idx["n"] == 3
    # Primary call has no env override (uses inherited env).
    assert call_envs[0] is None
    # Replicate calls carry FI_REPLICATE_SEED=<i> ALONGSIDE the parent
    # environment (PATH/PYTHONPATH/LANG etc) — the engine merges with
    # os.environ rather than replacing it so the venv python.exe can
    # still find its DLLs/site-packages on Windows. Assert just the
    # seed value without pinning the whole merged dict.
    assert call_envs[1] is not None
    assert call_envs[1]["FI_REPLICATE_SEED"] == "1"
    assert "PATH" in call_envs[1]  # parent env propagated through
    assert call_envs[2] is not None
    assert call_envs[2]["FI_REPLICATE_SEED"] == "2"
    # All three RESULT_JSONs aggregated, tagged by seed.
    assert "result_json_replicates" in patch
    reps = patch["result_json_replicates"]
    assert len(reps) == 3
    assert [r["_seed"] for r in reps] == [0, 1, 2]
    assert [r["rmse"] for r in reps] == [0.10, 0.20, 0.30]


@pytest.mark.asyncio
async def test_execute_skips_replicates_when_primary_run_fails(
    tmp_path: Path,
) -> None:
    """If the primary run fails (rc!=0 or no RESULT_JSON), we don't
    waste compute on replicates — the script is broken, fix it first
    via the execute_reflect loop."""
    eng = _make_engine_with_replicates(tmp_path, 3)
    # First call returns rc!=0 — primary failure (subtly with no RESULT_JSON output).
    # Multiple calls because the retry-on-fast-fail kicks in (rc!=0, duration<0.5s).
    primary_fail = type("ER", (), {
        "returncode": 1, "duration_s": 0.6, "timed_out": False,
        "stdout": "Traceback...\n", "stderr": "boom",
    })()
    eng.executor.execute = AsyncMock(return_value=primary_fail)  # type: ignore[method-assign]
    eng.executor.install = AsyncMock(return_value=type(  # type: ignore[method-assign]
        "IR", (), {"returncode": 0, "stderr": ""})())
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text("# fake\n", encoding="utf-8")

    patch = await eng._node_execute({"deps": []})
    # Just the primary execute (no replicates fired). Duration > 0.5s
    # so the suspicious-fast-fail retry guard does NOT trigger either.
    assert eng.executor.execute.await_count == 1
    # No replicates field on the patch.
    assert "result_json_replicates" not in patch
    assert patch["exec_result"]["returncode"] == 1


@pytest.mark.asyncio
async def test_execute_replicate_failure_doesnt_block_remaining_seeds(
    tmp_path: Path,
) -> None:
    """A single replicate failure (e.g. seed 1's RNG hit a NaN) MUST
    NOT abort the rest — better to have 2 of 3 valid replicates than
    zero. The failed seed is just absent from the aggregate."""
    eng = _make_engine_with_replicates(tmp_path, 3)
    # Primary OK; seed 1 fails (rc=2, no RESULT_JSON); seed 2 OK.
    call_outcomes = [
        type("ER", (), {
            "returncode": 0, "duration_s": 0.6, "timed_out": False,
            "stdout": 'RESULT_JSON: {"rmse": 0.1}\n', "stderr": "",
        })(),
        type("ER", (), {
            "returncode": 2, "duration_s": 0.6, "timed_out": False,
            "stdout": "Traceback\n", "stderr": "NaN",
        })(),
        type("ER", (), {
            "returncode": 0, "duration_s": 0.6, "timed_out": False,
            "stdout": 'RESULT_JSON: {"rmse": 0.3}\n', "stderr": "",
        })(),
    ]
    idx = {"n": 0}

    async def fake_execute(*args, **kw):  # noqa: ANN001
        out = call_outcomes[idx["n"]]
        idx["n"] += 1
        return out
    eng.executor.execute = fake_execute  # type: ignore[method-assign]
    eng.executor.install = AsyncMock(return_value=type(  # type: ignore[method-assign]
        "IR", (), {"returncode": 0, "stderr": ""})())
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text("# fake\n", encoding="utf-8")

    patch = await eng._node_execute({"deps": []})
    # All 3 ran (primary + 2 replicates).
    assert idx["n"] == 3
    # Only the 2 successful seeds in the aggregate (seed 0 and seed 2).
    reps = patch["result_json_replicates"]
    assert [r["_seed"] for r in reps] == [0, 2]
