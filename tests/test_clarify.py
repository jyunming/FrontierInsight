"""Clarify-node tests.

Three modes:
  - ``off``         — node returns ``{"clarify_done": True,
                      "no_simulation_resolved": <yaml-flag>}`` and no
                      questions. Even with clarify off, the engine
                      still has to surface a routing decision for
                      ``_route_after_design``, hence the resolved flag.
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


def _mk_cfg(
    tmp_path: Path, clarify_mode: str,
    *, local_papers: list[Path] | None = None,
) -> Config:
    return Config(
        topic="EUV-MOR SE-yield study",
        title="se-yield",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode=clarify_mode,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(
            enabled=False,
            local_papers=list(local_papers or []),
        ),
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

    # _node_clarify also returns the resolved
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


_PROPOSAL_FOR_SEED = """\
## TL;DR
A short summary of the planned quest.

## Background and prior work
RK4 and Verlet are standard integrators.

## Hypothesis

> H: RK4 outperforms Euler on long-horizon energy drift.

## Experimental plan

- Numerical simulation across timesteps (~5-minute simulation on CPU).
- Monte Carlo over 100 trials.

## Success criteria

The primary metric is relative energy drift ranked monotonically.
"""


@pytest.mark.asyncio
async def test_clarify_auto_seeded_from_proposal_skips_llm_call(
    tmp_path: Path,
) -> None:
    """When ``knowledge.local_papers`` contains a ``*-proposal.md``,
    the clarify node parses the proposal directly and skips the LLM
    call entirely — saving one premium request per quest started
    from a ``/proposal``-generated YAML."""
    from unittest.mock import AsyncMock

    proposal = tmp_path / "1778800000-test-aabbcc-proposal.md"
    proposal.write_text(_PROPOSAL_FOR_SEED, encoding="utf-8")

    cfg = _mk_cfg(tmp_path, "auto", local_papers=[proposal])
    eng = Engine(cfg)
    chat_mock = AsyncMock(return_value="{}")
    eng._client = type("Stub", (), {"chat": chat_mock})()

    patch = await eng._node_clarify({"topic": "anything"})

    chat_mock.assert_not_awaited(), (
        "the proposal short-circuit must SKIP the clarify LLM call"
    )
    assert patch["clarify_done"] is True
    assert patch["clarify_answers"]["simulatability"] == "yes"
    # Bonus slot — downstream nodes can read the hypothesis verbatim.
    assert "RK4" in patch["clarify_answers"]["_proposal_hypothesis"]
    # The resolver consumed the seeded answers — simulatability=yes
    # means SIMULATE, not no-simulation.
    assert patch["no_simulation_resolved"] is False


@pytest.mark.asyncio
async def test_clarify_interactive_does_not_seed_from_proposal(
    tmp_path: Path,
) -> None:
    """``interactive`` mode's contract is "let the human confirm /
    override every slot". Silently skipping the prompt because a
    proposal MD happens to be pinned would surprise users — they
    pinned the proposal as context for retrieval, NOT as a license
    to bypass the clarify modal. The short-circuit is gated on
    ``mode == "auto"`` only; ``interactive`` still calls the LLM
    to generate questions, then ``interrupt()`` for the human."""
    from unittest.mock import AsyncMock

    proposal = tmp_path / "1778800000-test-aabbcc-proposal.md"
    proposal.write_text(_PROPOSAL_FOR_SEED, encoding="utf-8")

    cfg = _mk_cfg(tmp_path, "interactive", local_papers=[proposal])
    eng = Engine(cfg)
    chat_mock = AsyncMock(return_value=json.dumps(_FAKE_QUESTIONS))
    eng._client = type("Stub", (), {"chat": chat_mock})()

    try:
        # In interactive mode the engine eventually hits
        # ``interrupt()``. We don't care to drive a resume here —
        # we just need to assert chat WAS called (the short-circuit
        # MUST NOT engage).
        await eng._node_clarify({"topic": "anything"})
    except Exception:
        # ``interrupt()`` raises in some test runtimes; only the
        # pre-interrupt chat-call matters.
        pass
    chat_mock.assert_awaited(), (
        "interactive mode must still call the LLM to generate "
        "questions even when a proposal MD is pinned"
    )


@pytest.mark.asyncio
async def test_clarify_auto_falls_through_to_llm_when_no_proposal(
    tmp_path: Path,
) -> None:
    """A non-proposal local_paper (or no local_papers at all) leaves
    the normal LLM clarify path in place — no false short-circuit."""
    from unittest.mock import AsyncMock

    other = tmp_path / "just-some-paper.md"
    other.write_text("# Some random paper\n", encoding="utf-8")

    cfg = _mk_cfg(tmp_path, "auto", local_papers=[other])
    eng = Engine(cfg)
    chat_mock = AsyncMock(return_value=json.dumps(_FAKE_QUESTIONS))
    eng._client = type("Stub", (), {"chat": chat_mock})()

    await eng._node_clarify({"topic": "some topic"})
    chat_mock.assert_awaited_once(), (
        "non-proposal local_papers must NOT short-circuit the LLM call"
    )


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
