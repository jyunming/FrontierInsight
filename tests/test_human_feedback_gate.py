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


def test_human_feedback_reject_overwrites_review_verdict(
    gated_engine: Engine,
) -> None:
    """Reject is the documented "user said no" path. The node persists
    ``review.verdict = "rejected"`` so finalisation / cost-report /
    digest paths can see the rejection without consulting
    state.human_feedback. accept leaves verdict untouched."""
    state = {"review": {"verdict": "accept", "score": 3}, "iteration": 0}
    patch_reject = _drive_node(gated_engine, state, {"action": "reject"})
    assert patch_reject["human_feedback"]["action"] == "reject"
    assert patch_reject["review"]["verdict"] == "rejected"
    patch_accept = _drive_node(gated_engine, state, {"action": "accept"})
    assert "review" not in patch_accept


def test_human_feedback_snapshot_uses_state_paper_md_path(
    gated_engine: Engine, tmp_path: Path,
) -> None:
    """The snapshot's ``paper_md_path`` is sourced from
    ``state["paper_md"]`` when present so custom pipelines / future
    write-node relocations don't desynchronise the gate view from the
    real file path."""
    custom = str(tmp_path / "custom-location" / "paper.md")
    state = {
        "review": {"verdict": "accept"},
        "iteration": 0,
        "paper_md": custom,
    }
    _drive_node(gated_engine, state, {"action": "accept"})
    snap = json.loads(
        (gated_engine.fi_dir / "human_review.json").read_text(encoding="utf-8"),
    )
    assert snap["paper_md_path"] == custom


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


# ---------------------------------------------------------------------------
# Must-flag hits are non-bypassable, even when review_loop is False.
# ---------------------------------------------------------------------------


def test_route_must_flag_forces_revise_even_with_review_loop_off(
    tmp_path: Path,
) -> None:
    """The whole point of the must_flag mechanism: even when the
    operator disables ``review_loop`` (intending "ship after one
    pass, no auto-revise"), a methodology must-flag hit (circular
    eval, single-point eval, weak baseline, pseudo-units) MUST force
    a revise pass. Otherwise the design errors the must-flag rules
    exist to catch slip through silently."""
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            clarify_mode="off", review_loop=False, max_iterations=3,
            # Gate off so we're testing the must-flag short-circuit
            # specifically, not the gate.
            human_feedback_gate="off",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = Engine(cfg)
    flagged_state = {
        "review": {"verdict": "accept", "must_flag_hits": ["[methodologist] circular_evaluation"]},
        "iteration": 0,
    }
    assert eng._route_after_review(flagged_state) == "revise"  # type: ignore[arg-type]


def test_route_must_flag_respects_max_iterations(tmp_path: Path) -> None:
    """A must-flag hit at the iteration cap can't loop forever — the
    same ``max_iterations`` budget the verdict-driven loop respects
    applies. Otherwise an LLM stuck flagging the same issue every
    pass would prevent the quest from ever finalising."""
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            clarify_mode="off", review_loop=False, max_iterations=2,
            human_feedback_gate="off",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = Engine(cfg)
    at_cap = {
        "review": {"verdict": "accept", "must_flag_hits": ["[methodologist] single_point_eval"]},
        "iteration": 2,
    }
    # iteration == max_iterations: fall through to legacy routing
    # which (with review_loop=False) returns "done".
    assert eng._route_after_review(at_cap) == "done"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Accumulated feedback_history across refine rounds.
# ---------------------------------------------------------------------------


def test_human_feedback_refine_appends_to_history(
    gated_engine: Engine,
) -> None:
    """First refine seeds the history; a second refine appends a
    fresh entry rather than overwriting. The design node reads ALL
    entries so a later iteration honours every prior ask."""
    state = {
        "review": {"verdict": "accept"},
        "iteration": 0,
        "feedback_history": [],
    }
    patch1 = _drive_node(
        gated_engine, state,
        {"action": "refine", "feedback": "tighten the abstract"},
    )
    assert patch1["human_feedback"]["action"] == "refine"
    assert len(patch1["feedback_history"]) == 1
    assert patch1["feedback_history"][0]["text"] == "tighten the abstract"
    state2 = {
        "review": {"verdict": "revise"},
        "iteration": 1,
        "feedback_history": patch1["feedback_history"],
    }
    patch2 = _drive_node(
        gated_engine, state2,
        {"action": "refine", "feedback": "add a control experiment"},
    )
    assert len(patch2["feedback_history"]) == 2
    assert patch2["feedback_history"][1]["text"] == "add a control experiment"


# ---------------------------------------------------------------------------
# Auto-accept-on-pass behaviour (the engine.auto_accept_on_pass attr).
# ---------------------------------------------------------------------------


def test_auto_accept_on_pass_reads_config_default(tmp_path: Path) -> None:
    """``Engine.auto_accept_on_pass`` is sourced from
    ``config.engine.auto_accept_on_pass`` when no explicit constructor
    arg is given — that's how fixtures + YAML opt in without touching
    the Engine constructor signature."""
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(auto_accept_on_pass=True),
        execution=ExecutionConfig(),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    assert eng.auto_accept_on_pass is True


def test_auto_accept_on_pass_constructor_arg_overrides_config(
    tmp_path: Path,
) -> None:
    """Explicit constructor arg wins over the YAML/config value —
    this is what ``launch.py --auto-accept-on-pass`` uses to force
    on regardless of YAML."""
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(auto_accept_on_pass=False),
        execution=ExecutionConfig(),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg, auto_accept_on_pass=True)
    assert eng.auto_accept_on_pass is True


# ---------------------------------------------------------------------------
# Aggregator unions ``must_flag_hits`` across the panel.
# ---------------------------------------------------------------------------


def test_aggregator_unions_must_flag_hits_across_panel() -> None:
    """Each panel member can flag independently; the union (deduped,
    persona-attributed) lands on the aggregated review so the router
    sees every reviewer's must-flag concerns."""
    from core.engine import _aggregate_panel_reviews
    panel = [
        {"persona": "methodologist", "verdict": "revise", "score": 2,
         "strengths": [], "weaknesses": [], "suggestions": [], "blocking": "",
         "must_flag_hits": ["circular_evaluation"]},
        {"persona": "statistician", "verdict": "accept", "score": 4,
         "strengths": [], "weaknesses": [], "suggestions": [], "blocking": "",
         "must_flag_hits": []},
        {"persona": "devil_advocate", "verdict": "revise", "score": 2,
         "strengths": [], "weaknesses": [], "suggestions": [], "blocking": "",
         "must_flag_hits": ["single_point_eval", "circular_evaluation"]},
    ]
    agg = _aggregate_panel_reviews(panel)
    mfh = agg["must_flag_hits"]
    # Two distinct hits across three personas; circular_evaluation
    # was flagged twice but appears only once (deduped).
    assert len(mfh) == 2
    assert any("circular_evaluation" in s for s in mfh)
    assert any("single_point_eval" in s for s in mfh)
    # Each surviving entry is persona-attributed.
    assert all(s.startswith("[") and "]" in s for s in mfh)


def test_aggregator_empty_panel_returns_empty_must_flag_hits() -> None:
    """The empty-panel fallback path must still include the
    ``must_flag_hits: []`` key so downstream consumers can ``.get()``
    without a KeyError."""
    from core.engine import _aggregate_panel_reviews
    agg = _aggregate_panel_reviews([])
    assert "must_flag_hits" in agg
    assert agg["must_flag_hits"] == []
