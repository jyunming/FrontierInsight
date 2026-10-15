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
def test_an_objection_to_the_design_rewrites_the_paper_and_keeps_the_design(tmp_path: Path, hits: list[str]) -> None:
    assert not _hits_need_only_a_rewrite(hits)
    state = {"review": {"verdict": "revise", "must_flag_hits": hits}, "iteration": 1}
    assert _engine(tmp_path)._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


def test_an_objection_is_recorded_stated_in_the_paper_and_not_counted_again(tmp_path: Path) -> None:
    """The first review's objection goes into FI's record, which the limitations state; the next review raising it
    again is not counted, so it neither costs a round nor sends the paper back once more."""
    from core import disclosure

    eng = _engine(tmp_path, max_iterations=3)
    first = {"verdict": "revise", "must_flag_hits": ["[methodologist] single_point_eval"],
             "blocking": "Only one operating point is evaluated."}
    counted = eng._note_design_objections({"iteration": 0}, first)  # type: ignore[arg-type]
    assert counted == first["must_flag_hits"] and "objections_repeated" not in first
    assert disclosure.stated_objections(tmp_path) == {"single_point_eval"}
    text = disclosure.run_record(tmp_path)
    assert "A reviewer objected to the design" in text and '"Only one operating point is evaluated"' in text

    again = {"verdict": "accept", "must_flag_hits": ["[methodologist] single_point_eval"]}
    assert eng._note_design_objections({"iteration": 1}, again) == []  # type: ignore[arg-type]
    assert again["objections_repeated"] == ["single_point_eval"]
    assert eng._route_after_review({"review": again, "iteration": 1}) == "done"  # type: ignore[arg-type]

    eng.config.engine.max_iterations = 1  # no rewrite follows: nothing is recorded
    disclosure.clear_objections(tmp_path)
    eng._note_design_objections({"iteration": 0}, dict(first))  # type: ignore[arg-type]
    assert disclosure.stated_objections(tmp_path) == set()


def test_two_objections_written_as_prose_are_two_and_each_is_said_as_written(tmp_path: Path) -> None:
    from core import disclosure
    from core.engine import _objection_key

    assert _objection_key("[methodologist] circular_evaluation: the test set is the training set") == "circular_evaluation"
    assert _objection_key("pseudo_units") == "pseudo_unit" and _objection_key("`single_point_eval`") == "single_point_eval"
    assert _objection_key("The baseline is weak") != _objection_key("The sample is too small")
    eng = _engine(tmp_path, max_iterations=3)
    eng._note_design_objections({"iteration": 0}, {"verdict": "accept", "must_flag_hits": ["The baseline is weak"]})  # type: ignore[arg-type]
    second = {"verdict": "accept", "must_flag_hits": ["The sample is too small"]}
    assert eng._note_design_objections({"iteration": 1}, second) == ["The sample is too small"]  # type: ignore[arg-type]
    assert "objections_repeated" not in second
    text = disclosure.run_record(tmp_path)
    assert '"The baseline is weak"; "The sample is too small"' in text


def test_an_objection_is_recorded_when_a_record_rewrite_follows_outside_the_rounds(tmp_path: Path) -> None:
    from core import disclosure

    eng = _engine(tmp_path, max_iterations=1)  # no round left: only the rewrite for FI's records follows
    review = {"verdict": "accept",
              "must_flag_hits": ["[methodologist] circular_evaluation", "record_contradiction: the paper says validated"]}
    eng._note_design_objections({"iteration": 0}, review)  # type: ignore[arg-type]
    assert disclosure.stated_objections(tmp_path) == {"circular_evaluation"}
    assert eng._route_after_review({"review": review, "iteration": 1, "record_rewrites": 1}) == "rewrite"  # type: ignore[arg-type]


def test_fi_adds_the_unsupported_claim_hit_from_its_own_claim_check() -> None:
    from core.engine import _unsupported_claim_hit

    state = {"claim_grounding": {"unsupported": ["R0 is 3.1", "the method is the fastest"]}}
    [hit] = _unsupported_claim_hit(state, ["[methodologist] circular_evaluation"])  # type: ignore[arg-type]
    assert hit.startswith("unsupported_claim:") and "2 claim(s)" in hit
    assert _unsupported_claim_hit(state, ["unsupported_claims"]) == []  # type: ignore[arg-type]  # already named
    assert _unsupported_claim_hit({"claim_grounding": {"unsupported": []}}, []) == []  # type: ignore[arg-type]


def test_the_writer_is_told_an_objection_is_stated_not_answered() -> None:
    state = {"review": {"verdict": "revise", "blocking": "The baseline is weak.",
                        "must_flag_hits": ["[methodologist] circular_evaluation", "unsupported_claim"]}}
    text = _format_review_for_writer(state)  # type: ignore[arg-type]
    block = text.split("Objections to the design")[1]
    assert "circular_evaluation" in block and "The baseline is weak." in block
    assert "never write that the study answers them" in block


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
