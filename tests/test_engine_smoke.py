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
