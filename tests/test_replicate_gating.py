"""Replication should cost what it is worth, and say what it found.

Three changes, all of them things a real quest exposed that the existing
flat-JSON fixtures could not:

* Replicates ran BEFORE the plausibility gate, so every futile repair
  iteration paid for three runs instead of one.
* A deterministic experiment was replicated in full. Honouring
  ``FI_REPLICATE_SEED`` is not the same as consuming randomness: the observed
  script seeded numpy and then integrated an ODE, so all three runs were byte
  identical and the only possible aggregate was ``std=0`` everywhere.
* The aggregator scanned top-level keys only, so a crossed design
  (``by_h`` x integrator) aggregated **nothing** -- reported as
  ``0 numeric keys``, which reads like a broken aggregator.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine, _aggregate_result_json_replicates


def _er(stdout: str, returncode: int = 0):
    return type("ER", (), {
        "returncode": returncode, "duration_s": 0.1, "timed_out": False,
        "stdout": stdout, "stderr": "",
    })()


def _rj(payload: str) -> str:
    return f"RESULT_JSON: {payload}\n"


def _engine(tmp_path: Path, *, replicates: int) -> Engine:
    cfg = Config(
        topic="replication", title="rep",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
            execute_replicates=replicates, pilot_run=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
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


# --- ordering: the gate before the error bars --------------------------------

@pytest.mark.asyncio
async def test_no_replicates_when_the_primary_result_is_implausible(
    tmp_path: Path,
) -> None:
    """A result about to be regenerated is not worth an error bar."""
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(return_value=_er(_rj('{"rmse": 999.0}')))  # type: ignore[method-assign]
    design = {"result_assertions": [
        {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
    ]}
    await eng._node_execute({"deps": [], "design": design})
    assert eng.executor.execute.await_count == 1, "replicated a rejected result"


@pytest.mark.asyncio
async def test_replicates_still_run_when_the_result_is_plausible(
    tmp_path: Path,
) -> None:
    """The gate must not become a blanket opt-out."""
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"rmse": 0.10}')),
        _er(_rj('{"rmse": 0.20}')),
        _er(_rj('{"rmse": 0.30}')),
    ])
    design = {"result_assertions": [
        {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
    ]}
    patch = await eng._node_execute({"deps": [], "design": design})
    assert eng.executor.execute.await_count == 3
    assert len(patch["result_json_replicates"]) == 3


@pytest.mark.asyncio
async def test_no_assertions_means_no_gating(tmp_path: Path) -> None:
    """Silence in the design means nothing was claimed, not that everything
    failed — replication proceeds as before."""
    eng = _engine(tmp_path, replicates=2)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"rmse": 0.1}')), _er(_rj('{"rmse": 0.2}')),
    ])
    await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 2


@pytest.mark.asyncio
async def test_replicates_run_when_no_repair_attempt_is_left(tmp_path: Path) -> None:
    """At the cap nothing regenerates the result: the paper is written from
    it, so it gets its error bars. Skipping them left a real SIR quest with one
    seed and no confidence intervals."""
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"rmse": 999.0}')),
        _er(_rj('{"rmse": 998.0}')),
        _er(_rj('{"rmse": 997.0}')),
    ])
    design = {"result_assertions": [
        {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
    ]}
    spent = eng.config.engine.exec_reflect_max_iterations
    patch = await eng._node_execute({"deps": [], "design": design, "exec_reflect_iter": spent})
    assert eng.executor.execute.await_count == 3
    assert len(patch["result_json_replicates"]) == 3
    assert patch["exec_patch_pending"] is False


# --- determinism: stop once two seeds agree ----------------------------------

@pytest.mark.asyncio
async def test_identical_seeds_stop_replication_early(tmp_path: Path) -> None:
    eng = _engine(tmp_path, replicates=5)
    eng.executor.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=_er(_rj('{"rmse": 0.25}')))
    patch = await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 2, "should stop after seeds 0 and 1"
    assert patch["result_json_deterministic"] is True


@pytest.mark.asyncio
async def test_differing_seeds_replicate_in_full(tmp_path: Path) -> None:
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"rmse": 0.10}')),
        _er(_rj('{"rmse": 0.11}')),
        _er(_rj('{"rmse": 0.12}')),
    ])
    patch = await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 3
    assert patch["result_json_deterministic"] is False


@pytest.mark.asyncio
async def test_determinism_is_judged_on_values_not_the_seed_tag(
    tmp_path: Path,
) -> None:
    """``_seed`` differs by construction; it must not mask agreement."""
    eng = _engine(tmp_path, replicates=4)
    eng.executor.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=_er(_rj('{"a": {"b": 1.0}}')))
    patch = await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 2
    seeds = [r["_seed"] for r in patch["result_json_replicates"]]
    assert seeds == [0, 1]


# --- aggregation: nested results ---------------------------------------------

def test_nested_results_aggregate_under_dotted_paths() -> None:
    reps = [
        {"_seed": 0, "by_h": {"0.5": {"RK4": {"err": 1.0}}}},
        {"_seed": 1, "by_h": {"0.5": {"RK4": {"err": 1.2}}}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert "by_h.0.5.RK4.err" in agg
    assert agg["by_h.0.5.RK4.err"]["mean"] == pytest.approx(1.1)
    assert agg["by_h.0.5.RK4.err"]["n"] == 2


def test_flat_and_nested_coexist() -> None:
    reps = [
        {"_seed": 0, "top": 5.0, "grp": {"inner": 1.0}},
        {"_seed": 1, "top": 7.0, "grp": {"inner": 3.0}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert set(agg) == {"top", "grp.inner"}
    assert agg["top"]["mean"] == pytest.approx(6.0)
    assert agg["grp.inner"]["mean"] == pytest.approx(2.0)


def test_a_path_missing_from_one_replicate_is_skipped() -> None:
    """Aggregating over a path only some seeds produced would silently change
    n between metrics."""
    reps = [
        {"_seed": 0, "grp": {"a": 1.0, "b": 2.0}},
        {"_seed": 1, "grp": {"a": 3.0}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert "grp.a" in agg
    assert "grp.b" not in agg


def test_non_numeric_leaves_are_ignored() -> None:
    reps = [
        {"_seed": 0, "grp": {"name": "RK4", "err": 1.0, "ok": True, "xs": [1, 2]}},
        {"_seed": 1, "grp": {"name": "RK4", "err": 2.0, "ok": True, "xs": [1, 2]}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert set(agg) == {"grp.err"}, "bools, strings and lists are not measurements"


def test_seed_key_is_not_aggregated_but_a_nested_seed_is_kept() -> None:
    """``_seed`` is synthetic at the top level only; a metric that happens to
    be called `_seed` deeper in the payload is the experiment's own data."""
    reps = [
        {"_seed": 0, "cfg": {"_seed": 11.0}},
        {"_seed": 1, "cfg": {"_seed": 11.0}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert "_seed" not in agg
    assert "cfg._seed" in agg
