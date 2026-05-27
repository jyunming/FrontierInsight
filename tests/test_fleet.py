"""Fleet validation: two Engines run concurrently with a fake LLM.

Verifies (a) N parallel engines can coexist in one process, (b) each gets
a distinct quest_id and quest_root, (c) per-quest venvs do not collide,
(d) figures and paper.md are written for both quests independently.
"""

from __future__ import annotations

import asyncio
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


_FAKE_EXPERIMENT_CODE = """\
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
plt.figure()
plt.plot([0, 1, 2], [0, 1, 4])
plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: {"score": 0.5}')
"""


_FAKE_RESPONSES = {
    "Ideation": json.dumps({
        "ideas": [{"title": "fleet-fake", "summary": "x", "feasibility": "high", "novelty": "low"}],
        "chosen": {"title": "fleet-fake", "rationale": "only candidate"},
    }),
    "Experiment Design": json.dumps({
        "hypothesis": "fleet hypothesis",
        "variables": {"independent": ["x"], "dependent": ["y"], "controls": []},
        "method": "plot",
        "expected_outcome": "monotonic",
        "figures_planned": ["result.png"],
        "dependencies": ["matplotlib"],
    }),
    "Implementation": json.dumps({"code": _FAKE_EXPERIMENT_CODE, "deps": ["matplotlib"]}),
    "Analysis": json.dumps({
        "summary": "Curve plotted.",
        "key_findings": ["y grows with x"],
        "claims_supported": [],
        "claims_unsupported": [],
        "limitations": [],
    }),
    "Writing": "# fleet-fake\n\nResult.\n\n![r](figures/result.png)\n",
    "Review": json.dumps({
        "verdict": "accept",
        "score": 4,
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
        "blocking": "",
    }),
}


def _fake_response_for(prompt: str) -> str:
    head = prompt.lstrip().splitlines()[0]
    for tag, resp in _FAKE_RESPONSES.items():
        if tag in head:
            return resp
    return "{}"


def _make_cfg(tmp_path: Path, slug: str) -> Config:
    return Config(
        topic=f"fleet smoke {slug}",
        title=f"fleet-{slug}",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            # Default gate is "after_review"; auto-accept-on-pass
            # finalises the clean-verdict happy path without a callback.
            auto_accept_on_pass=True,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.asyncio
async def test_two_engines_run_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    cfg_a = _make_cfg(tmp_path, "alpha")
    cfg_b = _make_cfg(tmp_path, "bravo")
    engine_a = Engine(cfg_a)
    engine_b = Engine(cfg_b)

    assert engine_a.quest_id != engine_b.quest_id
    assert engine_a.quest_root != engine_b.quest_root

    art_a, art_b = await asyncio.gather(engine_a.run(), engine_b.run())

    for art in (art_a, art_b):
        assert art.paper_md is not None and art.paper_md.exists()
        assert art.figures_dir is not None
        assert (art.figures_dir / "result.png").exists()
        assert art.raw_state.get("review", {}).get("verdict") == "accept"

    # Distinct on-disk artifacts.
    assert art_a.quest_root != art_b.quest_root
    assert art_a.paper_md != art_b.paper_md
