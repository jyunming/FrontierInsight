"""Unit tests for the human-feedback gate (engine.human_feedback_gate).

When ``engine.human_feedback_gate == "after_review"`` the engine fires
a ``human_feedback`` node that pauses on ``interrupt()`` and routes
to either END (accept / reject) or design (refine). These tests pin:

- Router behaviour: gate "off" → existing revise/done routing.
- Router behaviour: gate "after_review" → human_feedback always picked.
- Node behaviour: payload normalisation (action defaulted, empty
  feedback on ``refine`` falls back to ``accept``).
- Iteration bump on refine; iteration unchanged on accept/reject.
- ``_route_after_human_feedback`` respects ``max_iterations``.
- ``human_review.json`` snapshot written before the interrupt.

The full LangGraph + interrupt round-trip is integration-tested
elsewhere; these tests poke the engine helpers directly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine


def _engine_with_gate(tmp_path: Path, gate: str) -> Engine:
    """Build an Engine with the human-feedback gate set, no real LLM."""
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            clarify_mode="off", review_loop=True, max_iterations=2,
            human_feedback_gate=gate,  # type: ignore[arg-type]
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    return Engine(cfg)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_route_after_review_gate_off_keeps_existing_revise_done(tmp_path: Path) -> None:
    """When the gate is "off", review routing is exactly today's
    behaviour: revise on a verdict mismatch, done otherwise."""
    eng = _engine_with_gate(tmp_path, "off")
    accept_state = {"review": {"verdict": "accept"}, "iteration": 0}
    revise_state = {"review": {"verdict": "revise"}, "iteration": 0}
    assert eng._route_after_review(accept_state) == "done"  # type: ignore[arg-type]
    assert eng._route_after_review(revise_state) == "revise"  # type: ignore[arg-type]


def test_route_after_review_gate_on_always_human_feedback(tmp_path: Path) -> None:
    """When the gate is "after_review", routing ALWAYS goes through
    human_feedback first — the LLM's verdict is treated as a
    recommendation, not the final say."""
    eng = _engine_with_gate(tmp_path, "after_review")
    for verdict in ("accept", "revise"):
        state = {"review": {"verdict": verdict}, "iteration": 0}
        assert eng._route_after_review(state) == "human_feedback"  # type: ignore[arg-type]


def test_route_after_human_feedback_refine_loops_to_design(tmp_path: Path) -> None:
    """``refine`` → ``revise`` (back to design) when iteration cap not exhausted."""
    eng = _engine_with_gate(tmp_path, "after_review")
    state = {
        "human_feedback": {"action": "refine", "feedback": "be more rigorous"},
        "iteration": 1,
    }
    assert eng._route_after_human_feedback(state) == "revise"  # type: ignore[arg-type]


def test_route_after_human_feedback_refine_respects_max_iterations(tmp_path: Path) -> None:
    """A user who clicks "refine" past max_iterations can't outrun the
    loop budget — the route falls to ``done`` so the quest finalises."""
    eng = _engine_with_gate(tmp_path, "after_review")
    state = {
        "human_feedback": {"action": "refine", "feedback": "x"},
        "iteration": 2,  # already at max_iterations=2
    }
    assert eng._route_after_human_feedback(state) == "done"  # type: ignore[arg-type]


@pytest.mark.parametrize("action", ["accept", "reject"])
def test_route_after_human_feedback_accept_reject_terminates(
    tmp_path: Path, action: str,
) -> None:
    """accept / reject always finalise — the engine doesn't try to
    distinguish the two in the router (both stop the revise loop)."""
    eng = _engine_with_gate(tmp_path, "after_review")
    state = {
        "human_feedback": {"action": action, "feedback": ""},
        "iteration": 0,
    }
    assert eng._route_after_human_feedback(state) == "done"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Node behaviour
# ---------------------------------------------------------------------------


def _drive_node(eng: Engine, state: dict, payload: dict) -> dict[str, Any]:
    """Run ``_node_human_feedback`` and stub the LangGraph ``interrupt()``
    to return ``payload`` synchronously, mimicking a resumed graph.
    Returns the QuestState patch the node produced."""
    import core.engine as engine_mod
    real_interrupt = engine_mod.interrupt

    def fake_interrupt(_value: Any) -> dict[str, Any]:
        return payload

    engine_mod.interrupt = fake_interrupt
    try:
        result = asyncio.run(eng._node_human_feedback(state))  # type: ignore[arg-type]
    finally:
        engine_mod.interrupt = real_interrupt
    return result  # type: ignore[return-value]


@pytest.fixture
def gated_engine(tmp_path: Path) -> Engine:
    eng = _engine_with_gate(tmp_path, "after_review")
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    return eng


def test_human_feedback_accept_no_iteration_bump(gated_engine: Engine) -> None:
    """accept locks the iteration counter so the route fires ``done``."""
    state = {"review": {"verdict": "accept", "score": 3}, "iteration": 0}
    patch = _drive_node(gated_engine, state, {"action": "accept"})
    assert patch["human_feedback"]["action"] == "accept"
    assert "iteration" not in patch  # no bump


def test_human_feedback_refine_bumps_iteration_and_keeps_feedback(
    gated_engine: Engine,
) -> None:
    """refine bumps iteration so the loop budget is consumed; feedback
    text is preserved verbatim for the design node to read."""
    state = {"review": {"verdict": "accept"}, "iteration": 1}
    patch = _drive_node(
        gated_engine, state,
        {"action": "refine", "feedback": "tighten the abstract"},
    )
    assert patch["human_feedback"]["action"] == "refine"
    assert patch["human_feedback"]["feedback"] == "tighten the abstract"
    assert patch["iteration"] == 2


def test_human_feedback_refine_with_empty_feedback_falls_back_to_accept(
    gated_engine: Engine,
) -> None:
    """A 0-char refinement is indistinguishable from approval — the
    node normalises to ``accept`` so the router doesn't loop forever
    on no-content refinement."""
    state = {"review": {"verdict": "accept"}, "iteration": 0}
    patch = _drive_node(
        gated_engine, state, {"action": "refine", "feedback": "   "},
    )
    assert patch["human_feedback"]["action"] == "accept"
    assert "iteration" not in patch


def test_human_feedback_malformed_payload_defaults_to_accept(
    gated_engine: Engine,
) -> None:
    """Defensive: callback returning ``None`` / wrong-typed dict /
    unknown action all normalise to ``accept`` so a buggy frontend
    can't dump the quest into an undefined state."""
    state = {"review": {"verdict": "accept"}, "iteration": 0}
    for bad in (None, "garbage", {"action": "delete-everything"}, 42):
        patch = _drive_node(gated_engine, state, bad)  # type: ignore[arg-type]
        assert patch["human_feedback"]["action"] == "accept"


def test_human_feedback_writes_snapshot_json(gated_engine: Engine) -> None:
    """Before the interrupt fires, ``human_review.json`` lands in
    ``<quest_root>/.fi/`` so a web UI can render the gate state without
    pulling from the LangGraph checkpoint."""
    state = {
        "review": {
            "verdict": "revise", "score": 2,
            "strengths": ["clear motivation"],
            "weaknesses": ["unsupported claim in §3"],
            "suggestions": ["add a control experiment"],
        },
        "iteration": 1,
    }
    _drive_node(gated_engine, state, {"action": "accept"})
    snap_path = gated_engine.fi_dir / "human_review.json"
    assert snap_path.is_file()
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    assert snap["verdict"] == "revise"
    assert snap["iteration"] == 1
    assert snap["score"] == 2
    assert "weaknesses" in snap and snap["weaknesses"] == ["unsupported claim in §3"]
