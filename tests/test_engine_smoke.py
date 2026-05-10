"""End-to-end engine test with a fake LLM, no real API calls.

Patches the LLMClient.chat method so each node receives canned, valid
JSON responses. Verifies the full graph runs (including the review loop)
and produces a paper.md plus a figures/ directory on disk.
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
    "ideate": json.dumps({
        "ideas": [{"title": "fake-idea", "summary": "x", "feasibility": "high", "novelty": "low"}],
        "chosen": {"title": "fake-idea", "rationale": "only candidate"},
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
    for tag in ("Ideation", "Experiment Design", "Implementation", "Analysis", "Writing", "Review"):
        if tag in head:
            return tag
    return "(unknown)"


def _fake_response_for(prompt: str) -> str:
    head = _classify(prompt)
    return {
        "Ideation": _FAKE_RESPONSES["ideate"],
        "Experiment Design": _FAKE_RESPONSES["design"],
        "Implementation": _FAKE_RESPONSES["implement"],
        "Analysis": _FAKE_RESPONSES["analyze"],
        "Writing": _FAKE_RESPONSES["write"],
        "Review": _FAKE_RESPONSES["review"],
    }.get(head, "{}")


@pytest.fixture
def smoke_config(tmp_path: Path) -> Config:
    return Config(
        topic="smoke-test topic for the engine",
        title="engine-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
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
