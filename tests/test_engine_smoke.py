"""End-to-end engine test with a fake LLM, no real API calls.

Patches the LLMClient.chat method so each node receives canned, valid
JSON responses. Verifies the full graph runs through every node
including ``review`` (but NOT the revise→design loop — the test
config sets ``review_loop=False`` so the review verdict short-
circuits to ``done`` instead of looping back) and produces a
paper.md plus a figures/ directory on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import Engine


# Pre-baked code for the implement node — writes a figure and a RESULT_JSON line.
_FAKE_EXPERIMENT_CODE = """\
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
plt.figure()
plt.plot([0, 1, 2], [0, 1, 4])
plt.title('toy fake-LLM smoke test')
plt.savefig('figures/result.png', dpi=72)

print('RESULT_JSON: {\"score\": 0.987}')
"""


_FAKE_RESPONSES = {
    "clarify": json.dumps({
        "comparative_baseline": {
            "question": "What baseline?",
            "default": "trivial linear fit",
        },
        "empirical_vs_theoretical": {
            "question": "Empirical or theoretical?",
            "default": "empirical",
        },
        # ``simulatability`` is the routing signal the engine uses to
        # decide between the normal implement → execute path and the
        # no-simulation wait_for_data path. The legacy fallback maps
        # ``empirical_vs_theoretical == "empirical"`` to NO_SIMULATION
        # when ``simulatability`` is absent, which would pause this
        # smoke test at wait_for_data and never reach write. The
        # clarify-completion smoke tests want the full DAG, so we
        # pin ``simulatability: yes`` here — Python-script-can-answer.
        "simulatability": {
            "question": "Can a Python script answer this?",
            "default": "yes",
            "reason": "synthetic curve, no real-world data needed",
        },
        "success_metric": {
            "question": "What metric?",
            "default": "monotonicity of the curve",
        },
        "budget": {"question": "Time?", "default": "seconds on CPU"},
        "output_kinds": {"question": "Deliverables?", "default": ["paper_md"]},
    }),
    "ideate": json.dumps({
        "ideas": [{"title": "fake-idea", "summary": "x", "feasibility": "high", "novelty": "low"}],
        "chosen": {"title": "fake-idea", "rationale": "only candidate"},
    }),
    "ideate_reflect": json.dumps({
        "strongest_objection": "data may be trivial",
        "swap_to": "",
        "refined_rationale": "still the strongest pick given the constraints",
    }),
    "cross_check": json.dumps({
        "supporting": [{"index": 1, "why": "matches reported direction"}],
        "conflicting": [],
        "neutral": [],
        "summary": "literature broadly agrees",
    }),
    "design": json.dumps({
        "hypothesis": "fake hypothesis",
        "variables": {"independent": ["x"], "dependent": ["y"], "controls": []},
        "method": "plot y=x^2",
        "expected_outcome": "monotonic curve",
        "figures_planned": ["result.png"],
        "dependencies": ["matplotlib"],
    }),
    "implement": json.dumps({
        "code": _FAKE_EXPERIMENT_CODE,
        "deps": ["matplotlib"],
    }),
    "analyze": json.dumps({
        "summary": "Curve plotted; trivially monotonic.",
        "key_findings": ["y grows with x"],
        "claims_supported": [],
        "claims_unsupported": [],
        "limitations": ["toy data"],
    }),
    "write": "# fake-paper\n\nMethods. Results. ![r](figures/result.png)\n\n## References\n1. Smith 2020.\n",
    "claim_check": json.dumps({"claims": [], "summary": "no substantive claims found"}),
    "evidence_gate": json.dumps({"verdict": "sufficient", "rationale": "fake: evidence looks fine", "gaps": []}),
    "review": json.dumps({
        "verdict": "accept",
        "score": 4,
        "strengths": ["clear"],
        "weaknesses": [],
        "suggestions": [],
        "blocking": "",
    }),
}


def _classify(prompt: str) -> str:
    """Routes the canned response based on the leading prompt header."""
    head = prompt.lstrip().splitlines()[0]
    # Prompts whose first line is descriptive prose rather than a section
    # heading — match on a prefix substring.
    prefix = prompt[:200]
    if "scoping a research quest BEFORE the autonomous loop" in prefix:
        return "Clarify"
    if "Ideation Reflection" in head:
        return "IdeateReflect"
    if "Execute-Reflect" in head:
        return "ExecuteReflect"
    if "Cross-Paper Check" in head:
        return "CrossCheck"
    if "Claim Grounding" in head:
        return "ClaimCheck"
    if "Evidence Gate" in head:
        return "EvidenceGate"
    for tag in ("Ideation", "Experiment Design", "Implementation", "Analysis", "Writing", "Review"):
        if tag in head:
            return tag
    return "(unknown)"


def _fake_response_for(prompt: str) -> str:
    head = _classify(prompt)
    return {
        "Clarify": _FAKE_RESPONSES["clarify"],
        "Ideation": _FAKE_RESPONSES["ideate"],
        "IdeateReflect": _FAKE_RESPONSES["ideate_reflect"],
        "Experiment Design": _FAKE_RESPONSES["design"],
        "Implementation": _FAKE_RESPONSES["implement"],
        "ExecuteReflect": "{}",  # default: no patch needed for happy-path smokes
        "Analysis": _FAKE_RESPONSES["analyze"],
        "CrossCheck": _FAKE_RESPONSES["cross_check"],
        "Writing": _FAKE_RESPONSES["write"],
        "ClaimCheck": _FAKE_RESPONSES["claim_check"],
        "EvidenceGate": _FAKE_RESPONSES["evidence_gate"],
        "Review": _FAKE_RESPONSES["review"],
    }.get(head, "{}")


@pytest.fixture
def smoke_config(tmp_path: Path) -> Config:
    return Config(
        topic="smoke-test topic for the engine",
        title="engine-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            # The default human-feedback gate is "after_review"; in this
            # smoke fixture the fake review verdict is "accept" so
            # auto-accept-on-pass resolves the gate without a callback.
            auto_accept_on_pass=True,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        # Disable knowledge so we don't try to import axon during the test.
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.asyncio
async def test_engine_runs_with_fake_llm(smoke_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Engine(smoke_config)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert artifacts.paper_md.exists()
    assert artifacts.figures_dir is not None
    assert (artifacts.figures_dir / "result.png").exists()
    assert artifacts.raw_state.get("review", {}).get("verdict") == "accept"


@pytest.mark.asyncio
async def test_a_provider_message_written_as_the_paper_fails_the_quest(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three Sonnet quests ended rc=0 with a review "accept" while paper.md held
    only "You've hit your weekly limit ...". A paper that short is never a paper;
    the quest must fail and say what the writer returned."""
    engine = Engine(smoke_config)
    weekly = "You've hit your weekly limit · resets Sep 18, 10pm (Europe/Brussels)"

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Writing":
            return weekly
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    with pytest.raises(Exception) as exc:
        await engine.run()
    assert "cannot be a paper" in str(exc.value)
    assert "weekly limit" in str(exc.value)


@pytest.mark.asyncio
async def test_run_log_says_why_the_literature_search_found_nothing(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quest that "did not look for literature" left only "retrieved 0 docs"
    in its log — no way to tell a disabled search from a failed one."""
    engine = Engine(smoke_config)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    await engine.run()

    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[literature] searching sources=" in log
    assert "knowledge.enabled is false, so nothing was searched" in log


@pytest.mark.asyncio
async def test_dump_state_prints_a_real_checkpoint_as_text(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """state.sqlite is a binary file; on a machine nothing can be copied off
    it was unreadable. Reads a checkpoint a real run wrote."""
    from core.state_dump import dump_state

    engine = Engine(smoke_config)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    await engine.run()

    out = dump_state(engine.quest_root)
    assert f"quest: {engine.quest_id}" in out
    assert "literature" in out and "paper_md" in out
    assert "->  select_skills" in out
    assert out.rstrip().endswith("(end)")
    assert dump_state(engine.quest_root / ".fi" / "state.sqlite") == out


def test_dump_state_says_what_to_give_it_when_there_is_no_checkpoint(tmp_path: Path) -> None:
    from core.state_dump import dump_state

    with pytest.raises(FileNotFoundError, match="state.sqlite"):
        dump_state(tmp_path / "no-such-quest")


@pytest.mark.asyncio
async def test_run_log_names_the_interpreter_running_fi(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `pip install` into a different Python than the one running FI changes
    nothing, and nothing in run.log said which one that was. The user cannot
    send files off the machine, so the log itself must carry it."""
    import sys

    engine = Engine(smoke_config)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    await engine.run()

    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert f"[env] python={sys.executable}" in log


@pytest.mark.asyncio
async def test_run_clears_stale_quest_failed_at_start(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior run's ``quest_failed.md`` must be cleared at the START of a
    run — not only on the completion/pause paths. Otherwise an in-flight
    resume keeps the old diagnostic on disk for its whole duration and the
    dashboard shows the quest as 'failed' even while it's actively running
    past the node that broke. The first LLM-using node asserts the file is
    already gone, proving the clear happened before any node executed."""
    engine = Engine(smoke_config)
    # Plant a stale diagnostic as if a prior run of this quest had failed.
    engine.quest_root.mkdir(parents=True, exist_ok=True)
    stale = engine.quest_root / "quest_failed.md"
    stale.write_text("# prior failure breadcrumb\n", encoding="utf-8")

    seen: dict[str, bool | None] = {"present_on_first_call": None}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        if seen["present_on_first_call"] is None:
            seen["present_on_first_call"] = stale.exists()
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    await engine.run()

    assert seen["present_on_first_call"] is False, (
        "quest_failed.md should be cleared at run start, before the first node"
    )
    assert not stale.exists()


@pytest.mark.asyncio
async def test_run_clears_stale_pause_markers_but_keeps_answers(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume that continues past a pause must drop the 'needs you' DISPLAY
    markers at run start (so the dashboard doesn't show a ghost human-review /
    user-input pause for the whole run), but must KEEP the *answer* files —
    those are the input the resume is about to consume. Asserted at the first
    node call, before any node could re-write or consume them."""
    engine = Engine(smoke_config)
    engine.quest_root.mkdir(parents=True, exist_ok=True)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    display = [
        engine.quest_root / "NEXT_STEP.md",
        engine.fi_dir / "pause.json",
        engine.fi_dir / "human_review.json",
        engine.fi_dir / "clarify_questions.json",
    ]
    for p in display:
        p.write_text("stale\n", encoding="utf-8")
    answer = engine.fi_dir / "human_review_answer.json"
    answer.write_text('{"action": "accept", "feedback": ""}', encoding="utf-8")

    seen: dict[str, bool | None] = {"display_gone": None, "answer_kept": None}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        if seen["display_gone"] is None:
            seen["display_gone"] = not any(p.exists() for p in display)
            seen["answer_kept"] = answer.exists()
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    await engine.run()

    assert seen["display_gone"] is True, (
        "stale pause DISPLAY markers should be cleared at run start"
    )
    assert seen["answer_kept"] is True, (
        "answer files must survive the run-start clear — they're the input"
    )


@pytest.mark.asyncio
async def test_human_feedback_callback_timeout_pause_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wired human-feedback callback that never returns must not park
    the quest forever (the 2.5 h VSCode-QuickPick hang). After
    ``human_feedback_timeout_s`` the engine stops waiting, falls back to
    the headless pause-exit, and returns clean (rc=0) with the review
    snapshot left on disk for ``--resume``."""
    import asyncio

    cfg = Config(
        topic="smoke-test topic for human-feedback timeout",
        title="hf-timeout-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            human_feedback_gate="after_review",
            # auto_accept_on_pass stays False so the callback IS invoked
            # (proves we exercised the timeout path, not auto-accept).
            auto_accept_on_pass=False,
            human_feedback_timeout_s=0.2,  # tiny so the test is fast
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    callback_entered = {"v": False}

    async def hanging_cb(snapshot: dict) -> dict:
        callback_entered["v"] = True
        await asyncio.sleep(3600)  # never returns within the test window
        return {"action": "accept"}  # pragma: no cover

    # Bound the whole run so a regression (a real hang) fails the test
    # instead of hanging the suite. The bound is generous because the
    # pipeline does a real venv + pip install before reaching the gate;
    # once the gate is hit, the internal 0.2 s timeout pause-exits in <1 s.
    artifacts = await asyncio.wait_for(
        engine.run(human_feedback_callback=hanging_cb), timeout=240,
    )

    # The callback was entered → we hit the timeout fallback, not the
    # auto-accept short-circuit.
    assert callback_entered["v"] is True
    # Pause-exit leaves the snapshot on disk so the user can answer +
    # resume; it is NOT consumed on the timeout path.
    assert (engine.fi_dir / "human_review.json").is_file()
    # A partial artifact bundle comes back (paper was written before
    # review), and knowledge write-back is skipped (quest not accepted).
    assert artifacts is not None


@pytest.mark.asyncio
async def test_human_refine_loops_past_max_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an explicit human refine re-runs the design→…→review pass
    even with ``max_iterations=1`` and ``review_loop=False`` — the callback is
    invoked TWICE (refine, then accept), proving the refine wasn't silently
    dropped at the iteration cap (the bug a user hit clicking Refine)."""
    import asyncio

    cfg = Config(
        topic="smoke-test topic for human refine loop",
        title="hf-refine-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            human_feedback_gate="after_review", auto_accept_on_pass=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    calls: list[dict] = []

    async def cb(snapshot: dict) -> dict:
        calls.append(snapshot)
        # Refine once (forces a revision past the cap), then accept.
        if len(calls) == 1:
            return {"action": "refine", "feedback": "tighten the methods"}
        return {"action": "accept"}

    artifacts = await asyncio.wait_for(
        engine.run(human_feedback_callback=cb), timeout=300,
    )
    # The refine looped back to design and returned to the gate → 2 calls.
    # (Pre-fix it routed straight to done after the first refine → 1 call.)
    assert len(calls) == 2
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()


@pytest.mark.asyncio
async def test_reopen_finished_quest_re_runs_to_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run(reopen=True)`` on a FINISHED quest (checkpoint at terminal,
    next=()) re-enters the pipeline and lands back at the human-review gate —
    the callback fires again on the re-opened pass, proving a done quest can be
    continued for another review/refine."""
    import asyncio

    cfg = Config(
        topic="smoke-test topic for re-open",
        title="reopen-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            human_feedback_gate="after_review", auto_accept_on_pass=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    first: list[dict] = []

    async def accept_cb(snapshot: dict) -> dict:
        first.append(snapshot)
        return {"action": "accept"}

    await asyncio.wait_for(
        engine.run(human_feedback_callback=accept_cb), timeout=300,
    )
    assert len(first) == 1  # reviewed once → accept → terminal

    # Re-open the finished quest: it must re-run and hit the gate again.
    reopened: list[dict] = []

    async def accept_cb2(snapshot: dict) -> dict:
        reopened.append(snapshot)
        return {"action": "accept"}

    art = await asyncio.wait_for(
        engine.run(reopen=True, human_feedback_callback=accept_cb2), timeout=300,
    )
    assert len(reopened) == 1, "re-open did not re-run to the human-review gate"
    assert art.paper_md is not None and art.paper_md.exists()


@pytest.mark.asyncio
async def test_engine_runs_with_clarify_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: full DAG including the clarify node firing
    in auto mode. The agent generates the questionnaire, self-answers
    from each slot's default, and the answers flow through into the
    ideate/design/write prompts via `_format_clarify`. Verifies the
    quest still reaches a terminal artifact bundle."""
    cfg = Config(
        topic="smoke-test topic for clarify-auto integration",
        title="clarify-auto-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="auto",
            auto_accept_on_pass=True,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    artifacts = await engine.run()
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    # Clarify state landed on the final raw_state — both the questions
    # and the auto-derived answers.
    raw = artifacts.raw_state
    assert raw.get("clarify_done") is True
    assert "comparative_baseline" in raw.get("clarify_questions", {})
    answers = raw.get("clarify_answers", {})
    assert answers.get("empirical_vs_theoretical") == "empirical"
    assert answers.get("output_kinds") == ["paper_md"]


@pytest.mark.asyncio
async def test_engine_runs_with_clarify_interactive_via_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: full DAG with clarify_mode='interactive'.
    The engine pauses at the clarify node via `interrupt()`; the
    test-supplied callback simulates a user answering the questions;
    the graph resumes and runs to completion."""
    cfg = Config(
        topic="smoke-test topic for clarify-interactive integration",
        title="clarify-interactive-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="interactive",
            auto_accept_on_pass=True,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    seen_questions: list[dict] = []

    async def cb(questions: dict) -> dict:
        seen_questions.append(questions)
        # Override two defaults so we can verify the callback's answers
        # actually flow into state, not the agent's auto-defaults.
        # ``simulatability: yes`` is required to keep the engine on
        # the normal implement → execute path; without it the
        # legacy fallback on ``empirical_vs_theoretical == "empirical"``
        # routes to NO_SIMULATION and pauses at wait_for_data.
        return {
            "comparative_baseline": "user-chosen baseline",
            "empirical_vs_theoretical": "empirical",
            "simulatability": "yes",
            "success_metric": "user-chosen metric",
            "budget": "5 minutes",
            "output_kinds": ["paper_md"],
        }

    artifacts = await engine.run(clarify_callback=cb)

    assert len(seen_questions) == 1
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    answers = artifacts.raw_state.get("clarify_answers", {})
    assert answers["comparative_baseline"] == "user-chosen baseline"
    assert answers["success_metric"] == "user-chosen metric"


@pytest.mark.asyncio
async def test_await_with_heartbeat_returns_result_and_logs(smoke_config: Config) -> None:
    """The heartbeat wrapper returns the awaited result unchanged AND emits at
    least one '[execute] … still running' line to the engine logger while the
    coroutine is in flight, so a long silent subprocess keeps the dashboard's
    log-recency signal fresh."""
    import asyncio
    import logging as _logging

    engine = Engine(smoke_config)
    msgs: list[str] = []

    class _Cap(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            msgs.append(record.getMessage())

    handler = _Cap()
    engine._log.addHandler(handler)
    engine._log.setLevel(_logging.INFO)
    try:
        async def slow() -> str:
            await asyncio.sleep(0.25)
            return "RESULT"

        result = await engine._await_with_heartbeat(
            slow(), label="running experiment.py", interval_s=0.08,
        )
    finally:
        engine._log.removeHandler(handler)

    assert result == "RESULT"
    assert any("still running" in m for m in msgs), msgs


@pytest.mark.asyncio
async def test_after_literature_pause_then_resume_does_not_search_again(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first half of a quest is the literature; the second half designs and
    runs the experiment with it. ``pauses.supply: after_literature`` stops once
    the literature is saved, and ``--resume`` continues from select_skills — the
    search is not run a second time."""
    from core.state_dump import dump_state

    cfg = smoke_config.model_copy(update={
        "pauses": smoke_config.pauses.model_copy(update={"supply": "after_literature"}),
    })
    calls: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        calls.append(_classify(prompt))
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    first = Engine(cfg)
    await first.run()
    log = (first.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[supply] paused" in log and "(after_literature)" in log
    assert (first.fi_dir / "paused_at_after_literature.flag").is_file()
    assert (first.quest_root / "NEXT_STEP.md").is_file()
    stopped = dump_state(first.quest_root)
    assert "literature" in stopped
    assert "  design " not in stopped, "the design must not exist before the resume"
    assert not (first.quest_root / "paper" / "paper.md").exists()
    calls_before_resume = list(calls)

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()

    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    log = (second.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert log.count("[literature] searching sources=") == 1, "the search ran again"
    assert "Experiment Design" not in calls_before_resume
    assert "Experiment Design" in calls
    path = dump_state(second.quest_root)
    assert "->  pause_after_literature" in path and "->  select_skills" in path


# ---------------------------------------------------------------------------
# The plan step (plan.md), through the real graph and a real checkpoint
# ---------------------------------------------------------------------------


def _plan_fake_chat(calls: list[str], design_prompts: list[str] | None = None, plan_text: dict | None = None):
    """A fake model that answers the design prompt with a design plus the ``plan`` prose, as asked."""

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Experiment Design":
            if design_prompts is not None:
                design_prompts.append(prompt)
            body = json.loads(_FAKE_RESPONSES["design"])
            if "Also write the plan" in prompt:
                body["plan"] = {
                    "in_short": "A toy plot of y=x^2.",
                    "literature": [{"source": "Smith 2020", "says": "y grows with x"}],
                    "gap": "Nobody has plotted it here.",
                    "success_criteria": ["the curve is monotonic"],
                    "risks": ["too easy"],
                    "out_of_scope": ["anything real"],
                }
            return json.dumps(body)
        if plan_text is not None and "You are revising the plan" in prompt:
            return plan_text["reply"]
        return _fake_response_for(prompt)

    return fake_chat


@pytest.mark.asyncio
async def test_the_plan_is_written_and_the_quest_goes_on_unless_asked_to_stop(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import plan
    from core.state_dump import dump_state

    calls: list[str] = []
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _plan_fake_chat(calls, prompts))

    engine = Engine(smoke_config)
    artifacts = await engine.run()

    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    assert "A toy plot of y=x^2." in text and "**Smith 2020**: y grows with x" in text
    assert plan.parse(text).design["hypothesis"] == "fake hypothesis"
    # One design call in all, made by the plan step; the design step used the file and asked nothing.
    assert len(prompts) == 1 and "Also write the plan" in prompts[0]
    assert artifacts.raw_state["design"]["hypothesis"] == "fake hypothesis"
    history = json.loads((engine.quest_root / "needs" / "DESIGN_HISTORY.json").read_text(encoding="utf-8"))
    assert history[0]["plan_sha256"] == plan.sha256(text)
    path = dump_state(engine.quest_root)
    assert "->  plan" in path and "->  design" in path
    assert not (engine.fi_dir / "paused_at_plan.flag").exists()


@pytest.mark.asyncio
async def test_stop_for_the_plan_then_edit_then_resume_runs_what_was_edited(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole loop a person goes through: the quest stops with the plan written, they change the design
    block, and the quest that resumes runs their design, not the model's."""
    from core import plan
    from core.state_dump import dump_state

    cfg = smoke_config.model_copy(update={
        "pauses": smoke_config.pauses.model_copy(update={"plan": "ask"}),
    })
    calls: list[str] = []
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _plan_fake_chat(calls, prompts))

    first = Engine(cfg)
    await first.run()
    log = (first.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[plan] paused" in log
    # It is its own kind of stop: not mistaken for a clarify pause, which would leave questions for the
    # quest page to ask (found by running the real CLI, which said "paused for clarify").
    assert "[FI] paused for the plan" in log and "paused for clarify" not in log
    assert not (first.fi_dir / "clarify_questions.json").exists()
    descriptor = json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "plan" and descriptor["interaction"] == "supply"
    assert "plan.md" in (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    stopped = dump_state(first.quest_root)
    assert "  design " not in stopped, "the design must not exist before the resume"
    assert not (first.quest_root / "paper" / "paper.md").exists()

    path = plan.plan_path(first.quest_root)
    path.write_text(
        path.read_text(encoding="utf-8").replace("hypothesis: fake hypothesis", "hypothesis: the person's own hypothesis"),
        encoding="utf-8",
    )

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()

    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert artifacts.raw_state["design"]["hypothesis"] == "the person's own hypothesis"
    assert len(prompts) == 1, "the model was asked for the design once, before the person read it"
    assert [r["by"] for r in plan.history(second.quest_root)] == ["model", "user"]
    history = json.loads((second.quest_root / "needs" / "DESIGN_HISTORY.json").read_text(encoding="utf-8"))
    assert history[0]["plan_sha256"] == plan.sha256(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_plan_that_cannot_be_read_stops_the_resume_and_says_why(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import plan

    cfg = smoke_config.model_copy(update={
        "pauses": smoke_config.pauses.model_copy(update={"plan": "ask"}),
    })
    monkeypatch.setattr("core.engine.LLMClient.chat", _plan_fake_chat([]))
    first = Engine(cfg)
    await first.run()
    path = plan.plan_path(first.quest_root)
    path.write_text(
        path.read_text(encoding="utf-8").replace("hypothesis: fake hypothesis", "hypothesis: [fake"), encoding="utf-8",
    )

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()

    assert artifacts.paper_md is None or not artifacts.paper_md.exists(), "an unreadable plan must not run"
    log = (second.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "plan.md cannot be read" in log
    assert "not valid YAML" in (second.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")

    # Fixed, it goes on.
    path.write_text(
        path.read_text(encoding="utf-8").replace("hypothesis: [fake", "hypothesis: fixed by hand"), encoding="utf-8",
    )
    third = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await third.run()
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert artifacts.raw_state["design"]["hypothesis"] == "fixed by hand"


@pytest.mark.asyncio
async def test_asking_for_a_change_rewrites_the_plan_and_the_resume_runs_the_rewrite(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--revise-plan`` on a stopped quest: a fresh engine connects its own model client, rewrites the file, and
    the next resume runs the rewritten design."""
    from core import plan

    cfg = smoke_config.model_copy(update={
        "pauses": smoke_config.pauses.model_copy(update={"plan": "ask"}),
    })
    reply: dict[str, str] = {}
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _plan_fake_chat(calls, plan_text=reply))
    first = Engine(cfg)
    await first.run()
    path = plan.plan_path(first.quest_root)
    revised = path.read_text(encoding="utf-8").replace("hypothesis: fake hypothesis", "hypothesis: rewritten on request")
    reply["reply"] = revised

    reviser = Engine(cfg, resume_quest_id=first.quest_id)
    done = await reviser.revise_plan("state the hypothesis differently")

    assert done["version"] == 2
    assert [r["by"] for r in plan.history(first.quest_root)] == ["model", "request"]

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.raw_state["design"]["hypothesis"] == "rewritten on request"
