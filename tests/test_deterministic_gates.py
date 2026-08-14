"""Deterministic gate/verdict routing (PR-7).

Gate, verdict, and classifier nodes must run at temperature 0 so their routing
decision (sufficient vs broaden, accept vs revise, supported vs not) is
reproducible run-to-run, instead of flipping and triggering wasted broaden
loops or spurious revise cycles. Generative nodes keep the 0.2 default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _temperature_for_node


def test_temperature_for_node_gates_are_zero():
    for gate in (
        "evidence_gate", "review", "claim_check", "cross_check",
        "relevance_guard",
    ):
        assert _temperature_for_node(gate) == 0.0, gate
    # Every per-persona panel call is a verdict too.
    assert _temperature_for_node("review_panel.methodologist") == 0.0
    assert _temperature_for_node("review_panel.skeptic") == 0.0


def test_temperature_for_node_generative_are_default():
    for gen in ("ideate", "write", "implement", "analyze", "design"):
        assert _temperature_for_node(gen) == 0.2, gen
    # Unknown / empty node falls back to the generative default.
    assert _temperature_for_node(None) == 0.2
    assert _temperature_for_node("") == 0.2


def _mk_cfg(tmp_path: Path) -> Config:
    return Config(
        topic="deterministic gate tests",
        title="dg",
        provider=ProviderConfig(name="openai", model="default"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


class _CapClient:
    """Captures the temperature passed per node."""

    def __init__(self) -> None:
        self.temps: dict[str, float] = {}
        self.last_model = None
        self.last_usage = None

    async def chat(self, messages, *, temperature=0.2, model=None, node=""):  # noqa: ANN001
        self.temps[node] = temperature
        return "{}"

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_chat_routes_gate_nodes_to_temperature_zero(tmp_path: Path):
    eng = Engine(_mk_cfg(tmp_path))
    cap = _CapClient()
    eng._client = cap

    for node in (
        "evidence_gate", "review", "review_panel.methodologist",
        "claim_check", "cross_check", "relevance_guard",
        "ideate", "write",
    ):
        await eng._chat("prompt", node=node)

    assert cap.temps["evidence_gate"] == 0.0
    assert cap.temps["review"] == 0.0
    assert cap.temps["review_panel.methodologist"] == 0.0
    assert cap.temps["claim_check"] == 0.0
    assert cap.temps["cross_check"] == 0.0
    assert cap.temps["relevance_guard"] == 0.0
    # Generative nodes are unchanged.
    assert cap.temps["ideate"] == 0.2
    assert cap.temps["write"] == 0.2


@pytest.mark.asyncio
async def test_chat_explicit_temperature_overrides_routing(tmp_path: Path):
    eng = Engine(_mk_cfg(tmp_path))
    cap = _CapClient()
    eng._client = cap
    # An explicit override wins even for a gate node.
    await eng._chat("prompt", node="evidence_gate", temperature=0.7)
    assert cap.temps["evidence_gate"] == 0.7
