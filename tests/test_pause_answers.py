"""What a pause does with the answer it gets back.

A problem noticed while the clarify pause for web-launched quests was built: a review answer with no decision in it (``{}``, ``None``, a dismissed prompt) either re-fired the review pause
  forever (LangGraph reads ``Command(resume={})`` as "resume nothing"), crashed the run, or was silently taken as
  "accept". It now stops the quest cleanly, leaving NEXT_STEP.md to say what is needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph

from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, QuestState

# --------------------------------------------------------------------------------------------------------------------
# A review answer with no decision stops the quest; it never loops, crashes or accepts
# --------------------------------------------------------------------------------------------------------------------


class _ReviewOnlyEngine(Engine):
    """The real run loop and the real review pause, on a one-node graph (nothing else runs)."""

    def _build_graph(self) -> StateGraph:
        g: StateGraph[QuestState] = StateGraph(QuestState)
        g.add_node("human_feedback", self._node_human_feedback)
        g.add_edge(START, "human_feedback")
        g.add_edge("human_feedback", END)
        return g


def _review_engine(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        pauses={"review": "ask"},
    )
    return _ReviewOnlyEngine(cfg)


@pytest.fixture
def no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001, ANN003
        raise AssertionError("no model call is expected here")

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)


async def _run_with(eng: Engine, callback: Any) -> Any:
    return await asyncio.wait_for(eng.run(human_feedback_callback=callback), timeout=120)


@pytest.mark.parametrize("answer", [
    {}, None, {"feedback": "looks fine"}, {"action": "maybe"}, {"action": ""}, "accept",
    {"action": "refine", "feedback": "  "},
])
async def test_a_review_answer_without_a_decision_pauses_once(tmp_path: Path, no_llm: None, answer: Any) -> None:
    calls: list[dict] = []

    async def callback(snapshot: dict) -> Any:
        calls.append(snapshot)
        if len(calls) > 3:
            raise RuntimeError("the review pause re-fired: the empty answer looped")
        return answer

    eng = _review_engine(tmp_path)
    await _run_with(eng, callback)
    assert len(calls) == 1, f"asked {len(calls)} times"
    # Stopped, not accepted: the snapshot and the plain-language next step are still there for the person.
    assert (eng.fi_dir / "human_review.json").is_file()
    next_step = (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "--accept" in next_step
    state = await _saved_state(eng)
    assert "human_feedback" not in state, f"recorded a decision nobody made: {state.get('human_feedback')}"


async def test_a_dismissed_vscode_review_prompt_pauses(tmp_path: Path, no_llm: None) -> None:
    from core.vscode_bridge import BridgeError

    async def dismissed(snapshot: dict) -> Any:
        raise BridgeError("user cancelled the human-review panel")

    eng = _review_engine(tmp_path)
    await _run_with(eng, dismissed)
    assert (eng.fi_dir / "human_review.json").is_file()
    assert "human_feedback" not in await _saved_state(eng)


async def test_a_real_decision_still_goes_through(tmp_path: Path, no_llm: None) -> None:
    async def reject(snapshot: dict) -> Any:
        return {"action": "reject", "feedback": ""}

    eng = _review_engine(tmp_path)
    await _run_with(eng, reject)
    assert (await _saved_state(eng))["human_feedback"]["action"] == "reject"


async def test_a_staged_answer_file_without_a_decision_is_not_taken_as_accept(
        tmp_path: Path, no_llm: None) -> None:
    eng = _review_engine(tmp_path)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / "human_review_answer.json").write_text(json.dumps({"action": ""}), encoding="utf-8")
    await asyncio.wait_for(eng.run(), timeout=120)
    assert "human_feedback" not in await _saved_state(eng)
    # Dropped, so the next resume does not read it again; the review itself is still waiting.
    assert not (eng.fi_dir / "human_review_answer.json").exists()
    assert (eng.fi_dir / "human_review.json").is_file()
    assert "--accept" in (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")


async def test_a_callback_without_a_decision_still_uses_a_staged_decision(tmp_path: Path, no_llm: None) -> None:
    async def nothing(snapshot: dict) -> Any:
        return {}

    eng = _review_engine(tmp_path)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / "human_review_answer.json").write_text(json.dumps({"action": "reject"}), encoding="utf-8")
    await _run_with(eng, nothing)
    assert (await _saved_state(eng))["human_feedback"]["action"] == "reject"


# --------------------------------------------------------------------------------------------------------------------
# The same rule on each interface's side
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stdin_error", [EOFError, KeyboardInterrupt])
def test_terminal_prompt_with_no_answer_is_no_decision(monkeypatch: pytest.MonkeyPatch, stdin_error: type) -> None:
    import launch

    def no_input(*_a: Any) -> str:
        raise stdin_error

    monkeypatch.setattr("builtins.input", no_input)
    assert asyncio.run(launch._cli_human_feedback_callback({"verdict": "accept"})) == {}


def test_terminal_refine_notes_with_no_answer_is_no_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    def replies(prompt: str = "") -> str:
        if "action" in prompt:
            return "refine"
        raise EOFError

    monkeypatch.setattr("builtins.input", replies)
    assert asyncio.run(launch._cli_human_feedback_callback({"verdict": "accept"})) == {}


def test_terminal_prompt_asks_again_after_a_typo(monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    replies = iter(["rejct", "reject"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(replies))
    assert asyncio.run(launch._cli_human_feedback_callback({"verdict": "accept"}))["action"] == "reject"


def test_resume_with_an_empty_refine_is_refused(tmp_path: Path) -> None:
    import argparse

    import launch

    args = argparse.Namespace(accept=False, reject=False, refine="  ", resume="q1")
    with pytest.raises(SystemExit) as exc:
        launch._apply_review_decision(args, tmp_path)
    assert exc.value.code == 2
    assert not (tmp_path / "q1" / ".fi" / "human_review_answer.json").exists()


def test_vscode_bridge_passes_a_missing_action_on_unchanged() -> None:
    """The bridge no longer turns a reply without an action into "accept"; the engine decides."""
    from core.engine import _review_decision
    from core.vscode_bridge import VSCodeBridgeClient

    async def go() -> Any:
        client = VSCodeBridgeClient(host="127.0.0.1", port=1)
        fut = asyncio.get_running_loop().create_future()
        client._pending[7] = fut
        client._dispatch({"type": "human_review_response", "id": 7})
        return await fut

    answer = asyncio.run(go())
    assert answer["action"] == ""
    assert not _review_decision(answer)


async def _saved_state(eng: Engine) -> dict[str, Any]:
    return await eng._checkpoint_values()
