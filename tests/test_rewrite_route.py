"""A review whose must-flags are all problems with the text sends the quest
back to ``write`` only, and the writer gets the review of its previous draft."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    PausesConfig, ProviderConfig,
)
from core.engine import Engine, _format_review_for_writer, _hits_need_only_a_rewrite

TOPIC = "Extinction times in a stochastic SIR model"


def _engine(tmp_path: Path, *, max_iterations: int = 2) -> Engine:
    eng = Engine(Config(
        topic=TOPIC, title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=max_iterations, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        pauses=PausesConfig(review="off"),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    return eng


# --- the route -----------------------------------------------------------------

@pytest.mark.parametrize("hits", [
    ["unsupported_claim"],
    ["[methodologist] unsupported_claim", "[statistician] figure_caption: Figure 2 names S(t)"],
    ["`unsupported_claims`", "figure_captions"],
])
def test_hits_about_the_text_rewrite_the_paper(tmp_path: Path, hits: list[str]) -> None:
    assert _hits_need_only_a_rewrite(hits)
    state = {"review": {"verdict": "revise", "must_flag_hits": hits}, "iteration": 1}
    assert _engine(tmp_path)._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


@pytest.mark.parametrize("hits", [
    ["unsupported_claim", "[methodologist] circular_evaluation"],
    ["[methodologist] single_point_eval"],
    ["unverified_number: x"],
    # Not a hit name: the reviewer is asked for the identifier.
    ["unsupported claim"],
])
def test_any_other_hit_runs_the_quest_again_from_design(tmp_path: Path, hits: list[str]) -> None:
    assert not _hits_need_only_a_rewrite(hits)
    state = {"review": {"verdict": "revise", "must_flag_hits": hits}, "iteration": 1}
    assert _engine(tmp_path)._route_after_review(state) == "revise"  # type: ignore[arg-type]


def test_no_rewrite_once_the_iterations_are_spent(tmp_path: Path) -> None:
    state = {"review": {"verdict": "revise", "must_flag_hits": ["unsupported_claim"]}, "iteration": 2}
    assert _engine(tmp_path)._route_after_review(state) == "done"  # type: ignore[arg-type]


def test_no_hits_is_not_a_rewrite() -> None:
    assert not _hits_need_only_a_rewrite([])


# --- the graph -----------------------------------------------------------------

def test_the_graph_runs_write_claim_check_and_review_again_and_nothing_else(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    visits: list[str] = []
    reviews = iter([
        {"review": {"verdict": "revise", "must_flag_hits": ["[methodologist] unsupported_claim"]},
         "iteration": 1},
        {"review": {"verdict": "accept", "must_flag_hits": []}},
    ])

    def node(name: str, patch: Any = None) -> Any:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            visits.append(name)
            return patch() if patch else {}
        return run

    for name in ("design", "implement_outline", "implement", "execute", "analyze", "write", "claim_check"):
        setattr(eng, f"_node_{name}", node(name))
    eng._node_review = node("review", lambda: next(reviews))  # type: ignore[method-assign]
    eng._route_after_evidence_gate = lambda state: "write"  # type: ignore[method-assign]
    graph = eng._build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "rewrite"}}
    graph.update_state(config, {"topic": TOPIC, "iteration": 0}, as_node="evidence_gate")
    asyncio.run(graph.ainvoke(None, config))
    assert visits == ["write", "claim_check", "review", "write", "claim_check", "review"]
    assert graph.get_state(config).values["iteration"] == 1


# --- the write prompt ----------------------------------------------------------

def test_a_first_draft_has_no_review() -> None:
    assert _format_review_for_writer({}) == "(none — first draft)"  # type: ignore[typeddict-item]


def test_the_writer_gets_the_review_the_unsupported_claims_and_the_users_feedback(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    seen: dict[str, str] = {}

    async def chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        seen[node] = prompt
        return "# Extinction Times\n\n## Abstract\nShort.\n"

    eng._chat = chat  # type: ignore[method-assign]
    asyncio.run(eng._node_write({"topic": TOPIC}))  # type: ignore[arg-type]
    assert seen["write"].rstrip().endswith("## Review of the previous draft\n(none — first draft)")

    state = {
        "topic": TOPIC,
        "review": {
            "verdict": "revise", "score": 2,
            "must_flag_hits": ["[methodologist] unsupported_claim"],
            "figure_caption_warnings": ["Figure 2's caption names S(t), which the figure draws flat"],
            "weaknesses": "The discussion repeats the results.",
            "suggestions": ["Say how many seeds the interval is over."],
        },
        "claim_grounding": {"unsupported": ["Extinction is certain below R0 = 1.2 [2]."]},
        "feedback_history": [{"iteration": 0, "text": "Shorter introduction."}, {"iteration": 1, "text": " "}],
    }
    asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    review = seen["write"].split("## Review of the previous draft\n", 1)[1]
    assert review.rstrip() == "\n".join([
        "Verdict: revise (score 2)",
        "Must fix:",
        "  - [methodologist] unsupported_claim",
        "Captions that describe what their figure does not show:",
        "  - Figure 2's caption names S(t), which the figure draws flat",
        "Weaknesses:",
        "  - The discussion repeats the results.",
        "Suggestions:",
        "  - Say how many seeds the interval is over.",
        "Claims that neither this study's results nor a cited source backs:",
        "  - Extinction is certain below R0 = 1.2 [2].",
        "The user's feedback (honour every round):",
        "  - (round 0) Shorter introduction.",
    ])
    assert "## A new draft of a reviewed paper" in seen["write"]
