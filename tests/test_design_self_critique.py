"""Design self-critique pass tests.

Validates the second-pass methodology audit appended to ``_node_design``:

  - When the LLM returns a critique with ``amended_design``, the
    amended version replaces the draft and ``design_objections``
    surfaces into state.
  - When the critique returns nothing useful (empty object, parse
    failure, missing ``amended_design``), the draft survives and the
    quest proceeds — failing the audit must never block the quest.
  - The critique LLM call is fired AFTER the draft call, in order,
    and gets the draft JSON as one of its template variables.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine


def _mk_cfg(tmp_path: Path) -> Config:
    return Config(
        topic="OPC under low k1",
        title="opc",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


_DRAFT_DESIGN = {
    "hypothesis": "model-based OPC reduces EPE more than rule-based",
    "variables": {
        "independent": ["correction_strategy"],
        "dependent": ["epe", "cd_error"],
        "controls": ["seed"],
    },
    "method": "compare three strategies on synthetic clips",
    "expected_outcome": "model-based wins on EPE",
    "figures_planned": ["comparison.png"],
    "dependencies": ["numpy", "matplotlib"],
}

_AMENDED_DESIGN = {
    **_DRAFT_DESIGN,
    "hypothesis": (
        "model-based OPC reduces EPE more than rule-based (with the "
        "model-based correction optimized against a calibration "
        "simulator distinct from the evaluator)"
    ),
    "method": (
        "compare three strategies on synthetic clips; use independent "
        "evaluator simulator with different Gaussian sigma so the "
        "evaluation is not circular"
    ),
}


@pytest.mark.asyncio
async def test_design_self_critique_amended_design_replaces_draft(
    tmp_path: Path,
) -> None:
    """When the critique pass returns a valid amended_design, the
    amended version lands in state['design'] (not the draft) and the
    objections-addressed list is preserved for inspection."""
    cfg = _mk_cfg(tmp_path)
    eng = Engine(cfg)

    objections = [
        {"check": "circular_evaluation",
         "objection": "evaluator and optimizer share simulator",
         "fix": "introduce independent evaluator with different sigma"},
    ]
    responses = [
        json.dumps(_DRAFT_DESIGN),
        json.dumps({
            "objections_addressed": objections,
            "amended_design": _AMENDED_DESIGN,
        }),
    ]
    eng._client = type(
        "Stub", (), {"chat": AsyncMock(side_effect=responses)},
    )()

    patch = await eng._node_design({"topic": "OPC", "iteration": 0})

    assert patch["design"] == _AMENDED_DESIGN, (
        "amended_design from the critique must replace the draft"
    )
    assert patch["design_objections"] == objections


@pytest.mark.asyncio
async def test_design_self_critique_falls_back_to_draft_on_parse_failure(
    tmp_path: Path,
) -> None:
    """If the critique LLM returns garbage (un-JSON, malformed), the
    draft design survives and the quest proceeds. The audit is
    advisory — its failure must NEVER block the engine, because the
    draft is still a valid design."""
    cfg = _mk_cfg(tmp_path)
    eng = Engine(cfg)

    responses = [
        json.dumps(_DRAFT_DESIGN),
        "not valid json at all <<<>>>",  # critique garbles its output
    ]
    eng._client = type(
        "Stub", (), {"chat": AsyncMock(side_effect=responses)},
    )()

    patch = await eng._node_design({"topic": "OPC", "iteration": 0})

    assert patch["design"] == _DRAFT_DESIGN, (
        "parse failure on the critique must NOT corrupt the design — "
        "draft must survive"
    )
    # No objections_addressed key when nothing parseable returned —
    # downstream consumers should not see a partial list.
    assert "design_objections" not in patch


@pytest.mark.asyncio
async def test_design_self_critique_empty_objections_keeps_draft(
    tmp_path: Path,
) -> None:
    """When the critique pass finds nothing to flag (empty
    objections_addressed array), the draft is left unchanged — the
    critique returns the draft back as amended_design but with no
    objections. Both shapes (no key vs amended==draft) are valid
    no-ops."""
    cfg = _mk_cfg(tmp_path)
    eng = Engine(cfg)

    responses = [
        json.dumps(_DRAFT_DESIGN),
        json.dumps({
            "objections_addressed": [],
            "amended_design": _DRAFT_DESIGN,
        }),
    ]
    eng._client = type(
        "Stub", (), {"chat": AsyncMock(side_effect=responses)},
    )()

    patch = await eng._node_design({"topic": "OPC", "iteration": 0})

    assert patch["design"] == _DRAFT_DESIGN
    assert patch["design_objections"] == []


@pytest.mark.asyncio
async def test_design_self_critique_makes_two_llm_calls_in_order(
    tmp_path: Path,
) -> None:
    """The critique must fire AFTER the draft call (not in parallel,
    not before) and the second call must receive the draft as one of
    its template substitutions. Pins the ordering contract because the
    critique semantically depends on having a draft to audit."""
    cfg = _mk_cfg(tmp_path)
    eng = Engine(cfg)

    seen_prompts: list[str] = []

    async def fake_chat(messages, **_kw):
        prompt = messages[-1]["content"]
        seen_prompts.append(prompt)
        if len(seen_prompts) == 1:
            return json.dumps(_DRAFT_DESIGN)
        return json.dumps({
            "objections_addressed": [],
            "amended_design": _DRAFT_DESIGN,
        })

    eng._client = type(
        "Stub", (), {"chat": AsyncMock(side_effect=fake_chat)},
    )()

    await eng._node_design({"topic": "OPC", "iteration": 0})

    assert len(seen_prompts) == 2, (
        f"expected exactly two LLM calls (draft + critique), got "
        f"{len(seen_prompts)}"
    )
    # The second prompt must carry the draft design verbatim — that's
    # how the critique knows what to audit.
    assert _DRAFT_DESIGN["hypothesis"] in seen_prompts[1], (
        "critique prompt must include the draft design as $draft_design"
    )
    # The second prompt must be the critique one (carries the
    # MUST-FIX checklist token "Circular evaluation").
    assert "Circular evaluation" in seen_prompts[1]
