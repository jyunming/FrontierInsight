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
    _take_refine_points,
    _user_feedback_review_block,
)

def _take_needs(markdown: str) -> tuple[str, list[str]]:
    paper, points = _take_refine_points(markdown)
    return paper, points["experiment"]


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
    # An older state's notes only in human_feedback (no history round): not a refine round, the writer is not asked.
    assert not _refine_round({"human_feedback": {"action": "refine", "feedback": "x"}})  # type: ignore[arg-type]


def test_the_writer_is_told_how_to_name_an_experiment_point() -> None:
    text = _format_review_for_writer(_refined(), refine_round=True)  # type: ignore[arg-type]
    assert "Holm p-value" in text and "NEEDS_EXPERIMENT:" in text and "`NEEDS_EXPERIMENT" not in text
    assert "NEEDS_EXPERIMENT:" not in _format_review_for_writer(_refined())  # type: ignore[arg-type]


def test_needs_experiment_lines_are_taken_out_of_the_paper() -> None:
    paper, points = _take_needs("# P\n\nBody.\n\nNEEDS_EXPERIMENT: rerun the sweep with 30 seeds\n")
    assert points == ["rerun the sweep with 30 seeds"] and "NEEDS_EXPERIMENT" not in paper and "Body." in paper
    assert _take_needs("# P\n\nBody.\n") == ("# P\n\nBody.\n", [])


def test_needs_experiment_lines_in_other_shapes_are_taken_too() -> None:
    reply = ("# P\n\nBody.\n\n`NEEDS_EXPERIMENT: run a permutation test`\n1. NEEDS_EXPERIMENT: more seeds\n"
             "2) **NEEDS_EXPERIMENT:** a finer grid**\n- NEEDS_EXPERIMENT: a new baseline\n")
    paper, points = _take_needs(reply)
    assert points == ["run a permutation test", "more seeds", "a finer grid", "a new baseline"]
    assert "NEEDS_EXPERIMENT" not in paper and paper.strip().endswith("Body.")


def test_a_fenced_reply_with_points_after_it_loses_its_fence() -> None:
    paper, points = _take_needs("```markdown\n# P\n\nBody.\n```\n\nNEEDS_EXPERIMENT: x\n")
    assert points == ["x"] and paper == "# P\n\nBody.\n"


def test_the_review_sees_only_the_notes_of_the_last_gate() -> None:
    history = [{"iteration": 1, "text": "old note, answered at an earlier gate"}, {"iteration": 2, "text": "new note"}]
    block = _user_feedback_review_block({"feedback_history": history, "feedback_rounds_from": 1})  # type: ignore[arg-type]
    assert "new note" in block and "old note" not in block
    assert "old note" not in _user_feedback_review_block({"feedback_history": history})  # type: ignore[arg-type]
    assert _user_feedback_review_block({"feedback_history": history, "feedback_rounds_from": 2}) == ""  # type: ignore[arg-type]


def test_a_note_still_unanswered_after_its_rewrite_goes_to_the_design(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    hit = "user_feedback_unaddressed: the Holm p-value is still there"
    state = _refined(review={"verdict": "revise", "must_flag_hits": [hit]}, iteration=0, refine_written_for=1)
    assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]
    assert eng._route_after_review({**state, "feedback_rewrite_for": 1}) == "revise"  # type: ignore[arg-type]


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
    # The rewrite after it, for the review's note, is not a refine round: its scope is empty, and it is marked as the
    # rewrite for that note.
    later = _write(eng, {**_refined(), **out, "review": {"verdict": "revise", "must_flag_hits": [
        "user_feedback_unaddressed: still there"]}}, "# Paper\n\nFixed.\n", prompts)
    assert later["refine_scope"] == "" and later["feedback_rewrite_for"] == 1 and "refine_written_for" not in later
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
    assert facts == {"refine_scope": "experiment", "needs_experiment": ["run a permutation test to get the p-value"],
                     "extend": [], "layout": []}


def test_notes_the_history_does_not_hold_go_to_the_design_rather_than_being_dropped() -> None:
    from core.engine import Engine

    eng = object.__new__(Engine)
    legacy = {"human_feedback": {"action": "refine", "feedback": "cite the 2019 study"}, "feedback_history": []}
    assert eng._route_after_human_feedback(legacy) == "revise"
    current = {**legacy, "feedback_history": [{"iteration": 1, "text": "cite the 2019 study"}]}
    assert eng._route_after_human_feedback(current) == "rewrite"
