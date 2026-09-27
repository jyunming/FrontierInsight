"""A person's refine goes to the writing step first, and to the design only when the writer says a point needs a new
experiment. A refine used to go to the design every time: a note about one fabricated sentence redesigned the study
and drifted its frozen protocol (the Kimi diagnostic campaign, 2026-09-27)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core import frozen_protocol
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import (
    Engine,
    _TEXT_ONLY_HITS,
    _format_review_for_writer,
    _hits_need_only_a_rewrite,
    _refine_round,
    _take_needs_experiment,
    _user_feedback_review_block,
)

NOTES = [{"iteration": 1, "text": "The Holm p-value in the discussion is not computed anywhere; remove it."}]


def _engine(tmp_path: Path) -> Engine:
    return Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(clarify_mode="off", review_loop=True, max_iterations=2,
                            human_feedback_gate="after_review"),  # type: ignore[arg-type]
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def _refined(**over: Any) -> dict[str, Any]:
    return {"topic": "t", "title": "t", "iteration": 2, "human_feedback": {"action": "refine", "feedback": NOTES[0]["text"]},
            "feedback_history": list(NOTES), **over}


def test_the_graph_sends_a_refine_to_the_writer_and_the_writer_decides() -> None:
    # The edges as built, not a picture of them.
    import inspect

    src = inspect.getsource(Engine._build_graph)
    assert '{"rewrite": "write", "revise": "design", "done": END}' in src
    assert '{"check": "claim_check", "redesign": "design"}' in src


def test_routes(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    assert eng._route_after_human_feedback(_refined()) == "rewrite"  # type: ignore[arg-type]
    # FI re-opening a finished quest (--reopen, --update) injects a refine with no notes: the design, as before.
    assert eng._route_after_human_feedback({"human_feedback": {"action": "refine"}}) == "revise"  # type: ignore[arg-type]
    assert eng._route_after_write({"refine_needs_experiment": []}) == "check"  # type: ignore[arg-type]
    assert eng._route_after_write({"refine_needs_experiment": ["rerun with 30 seeds"]}) == "redesign"  # type: ignore[arg-type]


def test_a_refine_is_answered_once() -> None:
    assert _refine_round(_refined())  # type: ignore[arg-type]
    assert not _refine_round(_refined(refine_written_for=1))  # type: ignore[arg-type]
    assert not _refine_round({"human_feedback": {"action": "accept"}, "feedback_history": NOTES})  # type: ignore[arg-type]


def test_the_writer_is_told_how_to_name_an_experiment_point() -> None:
    text = _format_review_for_writer(_refined(), refine_round=True)  # type: ignore[arg-type]
    assert "Holm p-value" in text and "NEEDS_EXPERIMENT:" in text
    assert "NEEDS_EXPERIMENT:" not in _format_review_for_writer(_refined())  # type: ignore[arg-type]


def test_needs_experiment_lines_are_taken_out_of_the_paper() -> None:
    paper, points = _take_needs_experiment("# P\n\nBody.\n\nNEEDS_EXPERIMENT: rerun the sweep with 30 seeds\n")
    assert points == ["rerun the sweep with 30 seeds"] and "NEEDS_EXPERIMENT" not in paper and "Body." in paper
    assert _take_needs_experiment("# P\n\nBody.\n") == ("# P\n\nBody.\n", [])


def test_the_review_reads_the_notes_and_an_unanswered_one_is_a_text_fix() -> None:
    block = _user_feedback_review_block(_refined())  # type: ignore[arg-type]
    assert "Holm p-value" in block and "user_feedback_unaddressed" in block
    assert _user_feedback_review_block({}) == ""  # type: ignore[arg-type]
    assert "user_feedback_unaddressed" in _TEXT_ONLY_HITS
    assert _hits_need_only_a_rewrite(["user_feedback_unaddressed: the Holm p-value is still there"])


def _write(eng: Engine, state: dict[str, Any], reply: str, prompts: list[str]) -> dict[str, Any]:
    async def fake_chat(prompt: str, **kw: Any) -> str:
        prompts.append(prompt)
        return reply

    eng._chat = fake_chat  # type: ignore[method-assign]
    (eng.quest_root / "paper").mkdir(parents=True, exist_ok=True)
    return asyncio.run(eng._node_write(state))  # type: ignore[arg-type]


def test_a_text_only_refine_rewrites_the_paper_and_leaves_the_protocol(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    frozen_protocol.freeze(eng.quest_root, {"grid": {"n": [1, 2]}, "runs_per_setting": 10}, approved_by="test",
                           source="plan.md")
    before = frozen_protocol.protocol_of(eng.quest_root)
    prompts: list[str] = []
    out = _write(eng, _refined(review={"verdict": "accept", "must_flag_hits": ["unsupported_claim: Holm"]}),
                 "# Paper\n\nThe comparison is described without a p-value.\n", prompts)
    assert "Holm p-value" in prompts[0], "the whole-paper writer read the notes (not the passage editor)"
    assert out["refine_needs_experiment"] == [] and out["refine_scope"] == "paper" and out["refine_written_for"] == 1
    assert eng._route_after_write({**_refined(), **out}) == "check"  # type: ignore[arg-type]
    assert frozen_protocol.protocol_of(eng.quest_root) == before


def test_a_refine_the_text_cannot_answer_goes_to_the_design(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    prompts: list[str] = []
    out = _write(eng, _refined(), "# Paper\n\nBody.\n\nNEEDS_EXPERIMENT: run a permutation test to get the p-value\n",
                 prompts)
    assert out["refine_needs_experiment"] == ["run a permutation test to get the p-value"]
    assert out["refine_scope"] == "experiment"
    assert "NEEDS_EXPERIMENT" not in Path(out["paper_md"]).read_text(encoding="utf-8")
    assert eng._route_after_write({**_refined(), **out}) == "redesign"  # type: ignore[arg-type]
    facts = eng._route_facts("write", {**_refined(), **out})  # type: ignore[arg-type]
    assert facts == {"refine_scope": "experiment", "needs_experiment": ["run a permutation test to get the p-value"]}
