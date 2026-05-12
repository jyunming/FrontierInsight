"""Phase M — ideate self-reflection tests.

After ideate produces ideas + chosen, an inline second LLM call (gated
by `engine.ideate_reflect`) may:
  - Confirm the original pick (swap_to == "")
  - Swap to a different idea from the brainstormed list
  - Refine the rationale on the kept pick

Verifies the inline call is gated, swapping works, unknown-title is
rejected gracefully, and exceptions in the critique path don't break
the node.
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


def _mk_cfg(tmp_path: Path, *, reflect: bool = True) -> Config:
    return Config(
        topic="ideate-reflect tests",
        title="ir",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            clarify_mode="off", ideate_reflect=reflect,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


_IDEAS = [
    {"title": "A: small replication", "summary": "redo a known result", "feasibility": "high", "novelty": "low"},
    {"title": "B: novel sweep", "summary": "scan a new parameter", "feasibility": "medium", "novelty": "high"},
]
_FIRST_IDEATE = json.dumps({
    "ideas": _IDEAS,
    "chosen": {"title": "A: small replication", "rationale": "safer pick"},
})


@pytest.mark.asyncio
async def test_ideate_reflect_disabled_skips_second_call(tmp_path: Path) -> None:
    """When `engine.ideate_reflect=False`, only the first ideate LLM
    call happens — no critique, no chosen_idea swap."""
    eng = Engine(_mk_cfg(tmp_path, reflect=False))
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]
    chat_mock = AsyncMock(return_value=_FIRST_IDEATE)
    eng._client = type("Stub", (), {"chat": chat_mock})()

    patch = await eng._node_ideate({"topic": "x"})

    assert chat_mock.await_count == 1  # ideate only, no reflect
    assert patch["chosen_idea"]["title"] == "A: small replication"
    assert "ideate_critique" not in patch


@pytest.mark.asyncio
async def test_ideate_reflect_keeps_pick_with_refined_rationale(tmp_path: Path) -> None:
    """Reflection returns `swap_to=""` → keep the original pick but
    overwrite its rationale with the refined version."""
    eng = Engine(_mk_cfg(tmp_path))
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]
    critique = json.dumps({
        "strongest_objection": "result may be trivial",
        "swap_to": "",
        "refined_rationale": "safer pick; we still get a calibrated baseline",
    })
    chat_mock = AsyncMock(side_effect=[_FIRST_IDEATE, critique])
    eng._client = type("Stub", (), {"chat": chat_mock})()

    patch = await eng._node_ideate({"topic": "x"})

    assert patch["chosen_idea"]["title"] == "A: small replication"
    assert "calibrated baseline" in patch["chosen_idea"]["rationale"]
    assert patch["ideate_critique"]["strongest_objection"]


@pytest.mark.asyncio
async def test_ideate_reflect_swaps_to_different_idea(tmp_path: Path) -> None:
    """Reflection returns `swap_to=<title>` matching an item in the
    brainstormed list → chosen_idea is replaced by that entry."""
    eng = Engine(_mk_cfg(tmp_path))
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]
    critique = json.dumps({
        "strongest_objection": "replication is too safe; we should aim higher",
        "swap_to": "B: novel sweep",
        "refined_rationale": "novelty wins given the time budget",
    })
    chat_mock = AsyncMock(side_effect=[_FIRST_IDEATE, critique])
    eng._client = type("Stub", (), {"chat": chat_mock})()

    patch = await eng._node_ideate({"topic": "x"})

    assert patch["chosen_idea"]["title"] == "B: novel sweep"
    assert "novelty wins" in patch["chosen_idea"]["rationale"]


@pytest.mark.asyncio
async def test_ideate_reflect_ignores_unknown_swap_title(tmp_path: Path) -> None:
    """If the LLM names a title that isn't in the brainstormed list
    (hallucination / typo), keep the original pick."""
    eng = Engine(_mk_cfg(tmp_path))
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]
    critique = json.dumps({
        "strongest_objection": "x",
        "swap_to": "Z: an idea that never existed",
        "refined_rationale": "",
    })
    chat_mock = AsyncMock(side_effect=[_FIRST_IDEATE, critique])
    eng._client = type("Stub", (), {"chat": chat_mock})()

    patch = await eng._node_ideate({"topic": "x"})

    assert patch["chosen_idea"]["title"] == "A: small replication"


@pytest.mark.asyncio
async def test_ideate_reflect_tolerates_critique_failure(tmp_path: Path) -> None:
    """If the critique LLM call raises, the original pick stands and
    the node still returns a usable state. Best-effort throughout."""
    eng = Engine(_mk_cfg(tmp_path))
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]

    # Build an instance whose `chat` is an instance attribute (not a
    # class attribute) so Python does NOT bind `self` into the call.
    async def chat(messages, **kw):  # noqa: ANN001
        head = messages[-1]["content"].splitlines()[0]
        if "Ideation Reflection" in head:
            raise RuntimeError("LLM transient failure")
        return _FIRST_IDEATE
    stub = type("Stub", (), {})()
    stub.chat = chat
    eng._client = stub

    patch = await eng._node_ideate({"topic": "x"})

    assert patch["chosen_idea"]["title"] == "A: small replication"
