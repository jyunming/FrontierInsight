"""The setup questions are asked of the model once, and the person's answers are matched to the questions they saw.

A real quest (VS Code, kimi-k3) logged "[clarify] Checking the setup." twice and paid for two ``clarify`` model calls:
the answers resume the node from its first line (LangGraph re-runs a node after its ``interrupt()``), and the node
asked the model again before reaching the pause. The second call can return other questions -- other title
suggestions -- than the ones the person answered, so a title picked by number named a title nobody was shown.

The same quest answered "can a simulation answer this?" with "yes, but more than 100lines." and the engine did not
recognise it, then said "no signal from YAML" although the YAML said ``simulatability: "yes"``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

import launch
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

_TOPIC = "verlet euler integrator step size error comparison energy drift"


def _cfg(tmp_path: Path, **kw: Any) -> Config:
    return Config(
        topic=_TOPIC,
        provider=ProviderConfig(name="openai"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        pauses={"clarify": "ask"},
        **kw,
    )


class _Reached(Exception):
    pass


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake model whose every clarify call suggests different titles; the run stops after the setup questions."""
    from tests.test_engine_smoke import _classify, _fake_response_for

    seen: dict[str, Any] = {"clarify_calls": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001, ANN003
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Clarify":
            seen["clarify_calls"] += 1
            n = seen["clarify_calls"]
            body = json.loads(_fake_response_for(prompt))
            body["title"] = {
                "question": "What should this study be called?",
                "options": [f"Call {n} title A", f"Call {n} title B", f"Call {n} title C"],
                "default": f"Call {n} title A",
            }
            return json.dumps(body)
        return _fake_response_for(prompt)

    async def stop_here(self, state):  # noqa: ANN001
        seen["state"] = dict(state)
        raise _Reached

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    monkeypatch.setattr(Engine, "_node_select_skills", stop_here)
    return seen


async def test_answering_in_process_asks_the_model_once(tmp_path: Path, model) -> None:
    """The terminal / VS Code / web-page callback path: the answer resumes the node in the same process."""
    engine = Engine(_cfg(tmp_path))
    shown: list[dict] = []

    async def person(questions):  # noqa: ANN001
        shown.append(questions)
        return {"title": "2"}  # the second suggestion they were shown

    with pytest.raises(_Reached):
        await asyncio.wait_for(engine.run(clarify_callback=person), timeout=90)
    assert model["clarify_calls"] == 1, "the resume asked the model for the setup questions again"
    assert shown[0]["title"]["suggestions"][1] == "Call 1 title B"
    assert model["state"]["title"] == "Call 1 title B"
    assert model["state"]["clarify_questions"]["title"]["suggestions"][0] == "Call 1 title A"
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "no second model call" in log


async def test_answering_after_a_pause_asks_the_model_once(tmp_path: Path, model, monkeypatch) -> None:
    """The quest stops at the questions (no one there to answer); the answers are written into
    ``.fi/clarify_answer.json`` (the web page or by hand) and a resume in a new process takes them."""
    monkeypatch.delenv("FI_WEB_ANSWERS", raising=False)
    cfg = _cfg(tmp_path, engine=EngineConfig(clarify_overrides={"budget": "one afternoon"}))
    first = Engine(cfg)
    await asyncio.wait_for(first.run(), timeout=90)
    assert "state" not in model, "an unanswered 'ask' stops and waits"
    asked = json.loads((first.fi_dir / "clarify_questions.json").read_text(encoding="utf-8"))
    assert asked["title"]["suggestions"][2] == "Call 1 title C"
    assert asked["budget"]["default"] == "one afternoon", "the form shows the YAML's pinned answer"
    kept = json.loads((first.fi_dir / Engine._CLARIFY_ASKED).read_text(encoding="utf-8"))["questions"]
    assert kept["budget"]["default"] == "seconds on CPU", (
        "kept as the model wrote them, so a pin changed by --update while the quest waits is the one shown next")

    (first.fi_dir / "clarify_answer.json").write_text(json.dumps({"title": "3"}), encoding="utf-8")
    resumed = Engine(cfg, resume_quest_id=first.quest_id)
    with pytest.raises(_Reached):
        await asyncio.wait_for(resumed.run(clarify_callback=launch._pick_clarify_callback(cfg, resumed, False)),
                               timeout=90)
    assert model["clarify_calls"] == 1
    assert model["state"]["title"] == "Call 1 title C"


async def test_questions_kept_for_another_topic_are_not_reused(tmp_path: Path, model) -> None:
    engine = Engine(_cfg(tmp_path))
    engine._keep_asked_clarify_questions("a different topic", {"title": {"question": "?", "default": "x"}})
    assert engine._asked_clarify_questions(_TOPIC) is None
    assert engine._asked_clarify_questions("a different topic") == {"title": {"question": "?", "default": "x"}}


# --- "can a simulation answer this?" -----------------------------------------------------------------------------

def _engine(tmp_path: Path, **engine_kw: Any) -> Engine:
    return Engine(Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(**engine_kw),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def _resolve(engine: Engine, answers: dict) -> tuple[bool, list[str]]:
    captured: list[logging.LogRecord] = []
    sink = logging.Handler()
    sink.emit = captured.append  # type: ignore[assignment]
    engine._log.addHandler(sink)
    try:
        result = engine._resolve_no_simulation_from_clarify(answers)
    finally:
        engine._log.removeHandler(sink)
    return result, [r.getMessage() for r in captured]


def test_an_answer_in_the_persons_own_words_counts_by_its_first_word(tmp_path: Path) -> None:
    engine = _engine(tmp_path, clarify_overrides={"simulatability": "yes"})
    # The real quest's answer.
    no_sim, lines = _resolve(engine, {"simulatability": "yes, but more than 100lines.",
                                      "empirical_vs_theoretical": "empirical"})
    assert no_sim is False
    assert any("source=clarify_simulatability" in m and "decision=yes" in m for m in lines)
    assert not any("does not start with" in m for m in lines)
    assert _resolve(_engine(tmp_path), {"simulatability": "No - it needs field measurements"})[0] is True
    assert _resolve(_engine(tmp_path), {"simulatability": {"default": "Uncertain; maybe"}})[0] is False


def test_not_sure_is_not_read_as_no(tmp_path: Path) -> None:
    for answer in ("not sure", "No idea", "no clue really", "No preference."):
        no_sim, lines = _resolve(_engine(tmp_path), {"simulatability": answer})
        assert no_sim is False, answer
        assert any("does not start with" in m for m in lines), answer
    assert _resolve(_engine(tmp_path), {"simulatability": "No."})[0] is True
    assert _resolve(_engine(tmp_path), {"simulatability": "no strong need to simulate, it is a survey"})[0] is True
    no_sim, lines = _resolve(_engine(tmp_path), {"simulatability": "Yes sure, a small script"})
    assert no_sim is False and any("decision=yes" in m for m in lines)


def test_with_clarify_off_a_pinned_no_is_honoured(tmp_path: Path) -> None:
    engine = _engine(tmp_path, clarify_overrides={"simulatability": "no"})
    assert engine._resolve_modes({})["no_simulation_resolved"] is True


def test_the_yaml_answer_is_used_when_the_clarify_answer_leaves_it_open(tmp_path: Path) -> None:
    """The YAML (``engine.clarify_overrides``, what the interview writes) was never consulted: an unreadable answer went
    straight to the legacy check and the log said "no signal from YAML"."""
    pinned_no = _engine(tmp_path, clarify_overrides={"simulatability": "no"})
    for answers in ({"simulatability": "we will see"}, {"simulatability": ""}, {}, {"simulatability": None}):
        no_sim, lines = _resolve(pinned_no, answers)
        assert no_sim is True, answers
        assert any("source=yaml_clarify_overrides" in m for m in lines), (answers, lines)
    # PyYAML reads an unquoted `no` as False.
    assert _resolve(_engine(tmp_path, clarify_overrides={"simulatability": False}), {})[0] is True
    # The interview writes "yes" for every quest not set to no-simulation: it never outranks the topic judgment of
    # the legacy empirical check.
    pinned_yes = _engine(tmp_path, clarify_overrides={"simulatability": "yes"})
    assert _resolve(pinned_yes, {"simulatability": "hmm", "empirical_vs_theoretical": "empirical"})[0] is True
    assert _resolve(pinned_yes, {"simulatability": "hmm", "empirical_vs_theoretical": "theoretical"})[0] is False
    # The person's own yes / no still wins over the YAML.
    assert _resolve(pinned_yes, {"simulatability": "no"})[0] is True


def test_with_nothing_anywhere_the_log_does_not_blame_the_yaml(tmp_path: Path) -> None:
    no_sim, lines = _resolve(_engine(tmp_path), {})
    assert no_sim is False
    assert not any("no signal from YAML" in m for m in lines)
    assert any("source=default" in m for m in lines)
