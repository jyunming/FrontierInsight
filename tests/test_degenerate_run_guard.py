"""Tests for the degenerate-run guard.

A script can exit 0 yet still be broken — e.g. a lithography aerial-image
sim that reports ``contrast=NILS=CD=...=0`` because the grid missed the
diffraction orders. The guard treats an all-zero RESULT_JSON as a soft
failure: the execute-repair loop retries before a paper is written, and
the result is flagged for analyze/review if it stays degenerate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _is_degenerate_result


def _engine(tmp_path: Path, *, guard: bool = True, max_iters: int = 3) -> Engine:
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            clarify_mode="off", review_loop=False,
            degenerate_run_guard=guard,
            exec_reflect_max_iterations=max_iters,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


# --- _is_degenerate_result -------------------------------------------------


@pytest.mark.parametrize(
    "rj,expected",
    [
        ({"contrast": 0.0, "NILS": 0.0, "CD": 0}, True),   # all-zero, >=2 leaves
        ({"contrast": 0.0, "NILS": 0.3}, False),           # one non-zero
        ({"x": 0.0}, False),                               # single leaf — too little signal
        ({}, False),                                       # empty
        ({"a": [0.0, 0.0], "b": {"c": 0}}, True),          # nested all-zero
        ({"a": [0.0, 1e-6]}, False),                       # a small non-zero present
        ({"ok": True, "done": False}, False),              # bools aren't metrics
        ({"a": float("nan"), "b": 0.0}, False),            # NaN → not degenerate
        ({"name": "x", "note": "y"}, False),               # no numeric leaves
        ({"contrast": 0.0, "NILS": 0.0, "degenerate": False}, False),   # degenerate=False override
        ({"contrast": 0.0, "NILS": 0.0, "_degenerate": False}, False),  # _degenerate=False override
        ({"contrast": 0.0, "NILS": 0.0, "degenerate": "false"}, False), # degenerate="false" string override
    ],
)
def test_is_degenerate_result(rj: dict, expected: bool) -> None:
    assert _is_degenerate_result(rj) is expected


# --- routing ---------------------------------------------------------------


def test_route_degenerate_retries(tmp_path: Path) -> None:
    """rc=0 + all-zero metrics + budget left → loop back to repair."""
    eng = _engine(tmp_path, guard=True, max_iters=3)
    state = {
        "exec_result": {"returncode": 0},
        "result_json": {"contrast": 0.0, "NILS": 0.0},
        "exec_reflect_iter": 0,
    }
    assert eng._route_after_execute_reflect(state) == "retry"  # type: ignore[arg-type]


def test_route_non_degenerate_proceeds(tmp_path: Path) -> None:
    """A real (non-zero) result proceeds as before — no spurious retry."""
    eng = _engine(tmp_path, guard=True)
    state = {
        "exec_result": {"returncode": 0},
        "result_json": {"contrast": 0.42, "NILS": 1.1},
        "exec_reflect_iter": 0,
    }
    assert eng._route_after_execute_reflect(state) == "proceed"  # type: ignore[arg-type]


def test_route_degenerate_guard_off_proceeds(tmp_path: Path) -> None:
    """With the guard disabled, all-zero results are accepted as-is."""
    eng = _engine(tmp_path, guard=False)
    state = {
        "exec_result": {"returncode": 0},
        "result_json": {"contrast": 0.0, "NILS": 0.0},
        "exec_reflect_iter": 0,
    }
    assert eng._route_after_execute_reflect(state) == "proceed"  # type: ignore[arg-type]


def test_route_degenerate_at_max_iters_proceeds(tmp_path: Path) -> None:
    """The repair budget caps the degenerate retry — at the cap we
    proceed (analyze flags it) rather than looping forever."""
    eng = _engine(tmp_path, guard=True, max_iters=2)
    state = {
        "exec_result": {"returncode": 0},
        "result_json": {"contrast": 0.0, "NILS": 0.0},
        "exec_reflect_iter": 2,  # exhausted
    }
    assert eng._route_after_execute_reflect(state) == "proceed"  # type: ignore[arg-type]


# --- analyze flagging ------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_flags_degenerate_result(tmp_path: Path) -> None:
    """When the final result is still degenerate, analyze sets
    ``degenerate_result`` so write/review frame it as a failure note."""
    eng = _engine(tmp_path, guard=True)
    eng.quest_root = tmp_path  # type: ignore[attr-defined]

    async def fake_chat(prompt: str, *, node: str | None = None) -> str:
        return json.dumps({"summary": "s", "key_findings": [], "next_step": "publish"})

    eng._chat = fake_chat  # type: ignore[assignment]
    state = {
        "result_json": {"contrast": 0.0, "NILS": 0.0},
        "exec_result": {"returncode": 0},
        "figures": [],
        "design": {},
    }
    patch = await eng._node_analyze(state)  # type: ignore[arg-type]
    assert patch.get("degenerate_result") is True


@pytest.mark.asyncio
async def test_analyze_no_flag_on_real_result(tmp_path: Path) -> None:
    """A real (non-zero) result is not flagged."""
    eng = _engine(tmp_path, guard=True)
    eng.quest_root = tmp_path  # type: ignore[attr-defined]

    async def fake_chat(prompt: str, *, node: str | None = None) -> str:
        return json.dumps({"summary": "s", "key_findings": [], "next_step": "publish"})

    eng._chat = fake_chat  # type: ignore[assignment]
    state = {
        "result_json": {"contrast": 0.45, "NILS": 1.42},
        "exec_result": {"returncode": 0},
        "figures": [],
        "design": {},
    }
    patch = await eng._node_analyze(state)  # type: ignore[arg-type]
    assert "degenerate_result" not in patch
