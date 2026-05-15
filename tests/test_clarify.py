"""Phase I clarify-node tests.

Three modes:
  - ``off``         — node returns ``{"clarify_done": True}`` and no questions.
  - ``auto``        — LLM generates questions, agent self-answers from defaults.
  - ``interactive`` — LLM generates questions, node calls ``interrupt()``;
                      resuming with answers populates ``clarify_answers``.

These tests do NOT call real LLMs — they monkeypatch the LLM client.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine, _default_clarify_questions, _format_clarify,
)


def _mk_cfg(tmp_path: Path, clarify_mode: str) -> Config:
    return Config(
        topic="EUV-MOR SE-yield study",
        title="se-yield",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode=clarify_mode,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


# --- helper-function unit tests --------------------------------------------


def test_format_clarify_empty_when_no_answers() -> None:
    """No clarify_answers in state → marker string so the prompt still
    parses cleanly (no $clarify_block placeholder left unsubstituted)."""
    assert _format_clarify({}) == "(none — clarify mode is off)"
    assert _format_clarify({"clarify_answers": {}}) == "(none — clarify mode is off)"


def test_format_clarify_renders_bulleted_block() -> None:
    state = {
        "clarify_answers": {
            "comparative_baseline": "Inpria 2015 MOR",
            "empirical_vs_theoretical": "empirical",
            "success_metric": "F_SE ≈ 2.3 at 92 eV",
            "budget": "a few minutes on CPU",
            "output_kinds": ["paper_md", "slides"],
        }
    }
    out = _format_clarify(state)
    # Each labeled slot lands as a bullet; lists become comma-joined.
    assert "- **Comparative baseline**: Inpria 2015 MOR" in out
    assert "- **Empirical / theoretical**: empirical" in out
    assert "- **Time / compute budget**: a few minutes on CPU" in out
    assert "- **Desired output kinds**: paper_md, slides" in out


def test_format_clarify_skips_unknown_slots() -> None:
    """Unknown slot keys (forward-compat) don't crash and are silently
    dropped from the rendering."""
    out = _format_clarify(
        {"clarify_answers": {"comparative_baseline": "x", "future_slot": "y"}}
    )
    assert "Comparative baseline" in out
    assert "future_slot" not in out


def test_default_clarify_questions_has_all_seven_slots() -> None:
    q = _default_clarify_questions("any topic")
    assert set(q) == {
        "comparative_baseline", "empirical_vs_theoretical",
        "success_metric", "budget", "output_kinds",
        "study_depth", "paper_venue",
    }
    for slot, value in q.items():
        assert isinstance(value, dict)
        assert "question" in value and "default" in value
        # Non-empty default is the invariant the prompt enforces and the
        # auto-answer path relies on.
        assert value["default"] not in (None, "", [])
    # study_depth default is one of the three documented levels — it's
    # the value the write/review prompts grep for to calibrate length.
    assert q["study_depth"]["default"] in (
        "brief preprint", "journal-length", "comprehensive review",
    )
    # paper_venue default must be one of the templates the engine knows
    # about — `_apply_paper_venue_override` silently drops unknown values.
    assert q["paper_venue"]["default"] in (
        "generic", "neurips", "iclr", "ieee_access", "nature_mi",
    )


# --- _node_clarify mode='off' ------------------------------------------------


@pytest.mark.asyncio
async def test_clarify_off_skips_node(tmp_path: Path, monkeypatch) -> None:
    """clarify_mode='off' must not call the LLM and must produce no
    questions/answers in state — keeps fleet runs and existing tests fast."""
    cfg = _mk_cfg(tmp_path, "off")
    eng = Engine(cfg)

    chat_calls: list[str] = []
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        chat_calls.append(messages[-1]["content"][:30])
        return "{}"
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    patch = await eng._node_clarify({"topic": "x"})

    # Phase D1 / β change: _node_clarify now ALSO returns the resolved
    # ``no_simulation_resolved`` flag so the routing decision is
    # visible at the state level (not just inferred from YAML/clarify
    # downstream). With clarify_mode='off' and ``engine.no_simulation``
    # unset, the resolved flag is False.
    assert patch == {"clarify_done": True, "no_simulation_resolved": False}
    assert chat_calls == []  # LLM was never invoked


# --- _node_clarify mode='auto' ----------------------------------------------


_FAKE_QUESTIONS = {
    "comparative_baseline": {
        "question": "What baseline?",
        "default": "Inpria 2015 MOR cathodoluminescence",
    },
    "empirical_vs_theoretical": {
        "question": "Which kind of study?",
        "default": "empirical",
    },
    "success_metric": {
        "question": "What metric?",
        "default": "F_SE within ±10% of literature value",
    },
    "budget": {
        "question": "How long?",
        "default": "1 hour on CPU",
    },
    "output_kinds": {
        "question": "Which deliverables?",
        "default": ["paper_md"],
    },
}


@pytest.mark.asyncio
async def test_clarify_auto_fills_answers_from_defaults(
    tmp_path: Path,
) -> None:
    """clarify_mode='auto': agent generates the questionnaire AND the
    answers populate from each slot's `default` field. No human loop."""
    from unittest.mock import AsyncMock

    cfg = _mk_cfg(tmp_path, "auto")
    eng = Engine(cfg)
    eng._client = type(
        "Stub", (), {"chat": AsyncMock(return_value=json.dumps(_FAKE_QUESTIONS))},
    )()

    patch = await eng._node_clarify({"topic": "EUV-MOR"})

    assert patch["clarify_done"] is True
    assert patch["clarify_questions"] == _FAKE_QUESTIONS
    answers = patch["clarify_answers"]
    assert answers["empirical_vs_theoretical"] == "empirical"
    assert answers["budget"] == "1 hour on CPU"
    assert answers["output_kinds"] == ["paper_md"]


@pytest.mark.asyncio
async def test_clarify_auto_falls_back_on_unparseable_llm_output(
    tmp_path: Path,
) -> None:
    """If the LLM produces JSON that doesn't parse, we synthesize a
    minimal default questionnaire so downstream nodes get something."""
    from unittest.mock import AsyncMock

    cfg = _mk_cfg(tmp_path, "auto")
    eng = Engine(cfg)
    eng._client = type(
        "Stub", (), {"chat": AsyncMock(return_value="not json at all — sorry about that")},
    )()

    patch = await eng._node_clarify({"topic": "some topic"})

    # Even with garbage from the LLM, we still produce all 7 slots.
    assert set(patch["clarify_questions"]) == {
        "comparative_baseline", "empirical_vs_theoretical",
        "success_metric", "budget", "output_kinds", "study_depth",
        "paper_venue",
    }
    assert patch["clarify_done"] is True


# --- _node_clarify idempotency ----------------------------------------------


@pytest.mark.asyncio
async def test_clarify_passes_through_when_already_done(
    tmp_path: Path,
) -> None:
    """When ``clarify_done`` is True (e.g. resuming after a kill), the
    node short-circuits and does NOT re-prompt the LLM. Idempotency is
    what makes the SqliteSaver resume semantics work."""
    from unittest.mock import AsyncMock

    cfg = _mk_cfg(tmp_path, "auto")
    eng = Engine(cfg)
    chat_mock = AsyncMock(return_value="{}")
    eng._client = type("Stub", (), {"chat": chat_mock})()

    patch = await eng._node_clarify({"topic": "x", "clarify_done": True})

    assert patch == {}
    chat_mock.assert_not_awaited()


# --- _node_clarify mode='interactive' ---------------------------------------


@pytest.mark.asyncio
async def test_clarify_interactive_raises_interrupt(
    tmp_path: Path,
) -> None:
    """clarify_mode='interactive' calls LangGraph's ``interrupt()``.
    Outside of a checkpointed graph context this raises ``GraphInterrupt``
    (or its older alias) — which the CLI/GUI driver catches and
    surfaces to the user. We assert the exception class is raised."""
    from unittest.mock import AsyncMock

    cfg = _mk_cfg(tmp_path, "interactive")
    eng = Engine(cfg)
    eng._client = type(
        "Stub", (), {"chat": AsyncMock(return_value=json.dumps(_FAKE_QUESTIONS))},
    )()

    # `interrupt()` outside a checkpointed graph context raises a
    # plain RuntimeError ("Called get_config outside of a runnable
    # context"). Inside a compiled graph with an SqliteSaver it
    # raises GraphInterrupt — but our unit-level call here is outside
    # that context. Either case is acceptable; the assertion is that
    # the node DOES pause-by-exception (not silently returns).
    with pytest.raises((RuntimeError, BaseException)) as ei:
        await eng._node_clarify({"topic": "EUV-MOR"})
    name = type(ei.value).__name__
    msg = str(ei.value).lower()
    assert (
        "Interrupt" in name
        or "GraphInterrupt" in name
        or "runnable context" in msg
        or "interrupt" in msg
    )
