"""The quest talks the topic over first by default, and gets a readable title from that talk."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, _clean_title

_QUESTIONS = {
    "want_to_see": {"question": "What do you want to see?", "default": "a plot of error against step size"},
    "title": {
        "question": "What should this study be called?",
        "options": ['"Step size and error in Verlet integration"', "Verlet vs Euler accuracy", "Integrator accuracy"],
        "default": "",
    },
    "simulatability": {"question": "Can Python answer it?", "default": "yes", "reason": "pure numerics"},
}
_RAW = "verlet euler integrator step size error comparison energy drift"


def _cfg(tmp_path: Path, **kw) -> Config:
    return Config(
        topic=_RAW,
        provider=ProviderConfig(name="openai"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        **kw,
    )


def _engine(tmp_path: Path, answerable: bool, **kw) -> Engine:
    eng = Engine(_cfg(tmp_path, **kw))
    eng._clarify_answerable = answerable
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=json.dumps(_QUESTIONS))})()
    return eng


def test_unset_clarify_means_neither_off_nor_a_fixed_choice(tmp_path: Path) -> None:
    assert _cfg(tmp_path).pauses.clarify is None


@pytest.mark.asyncio
async def test_with_someone_to_answer_the_quest_asks_first(tmp_path: Path) -> None:
    eng = _engine(tmp_path, answerable=True)
    with pytest.raises(Exception) as exc:  # the interrupt raised outside a running graph
        await eng._node_clarify({"topic": _RAW})
    assert "interrupt" in repr(exc.value).lower() or "runnable" in repr(exc.value).lower()
    eng._client.chat.assert_awaited()


@pytest.mark.asyncio
async def test_headless_run_answers_for_itself_and_does_not_wait(tmp_path: Path) -> None:
    eng = _engine(tmp_path, answerable=False)
    patch = await eng._node_clarify({"topic": _RAW})
    assert patch["clarify_done"] is True
    assert patch["clarify_answers"]["want_to_see"] == "a plot of error against step size"


@pytest.mark.asyncio
async def test_agent_picked_title_replaces_the_raw_topic(tmp_path: Path) -> None:
    eng = _engine(tmp_path, answerable=False)
    patch = await eng._node_clarify({"topic": _RAW, "title": "verlet-euler"})
    assert patch["title"] == "Step size and error in Verlet integration"
    assert not patch.get("title_confirmed"), "an agent pick is a suggestion, not something a person chose"
    assert "Suggestions:" in patch["clarify_questions"]["title"]["question"]
    assert "options" not in patch["clarify_questions"]["title"]


@pytest.mark.asyncio
async def test_a_title_set_in_the_yaml_always_wins(tmp_path: Path) -> None:
    eng = _engine(tmp_path, answerable=False, title="My own title")
    patch = await eng._node_clarify({"topic": _RAW})
    assert "title" not in patch


@pytest.mark.asyncio
async def test_explicit_off_still_skips_the_discussion(tmp_path: Path) -> None:
    eng = _engine(tmp_path, answerable=True, pauses={"clarify": "off"})
    patch = await eng._node_clarify({"topic": _RAW})
    assert patch["clarify_done"] is True
    eng._client.chat.assert_not_awaited()
    assert "title" not in patch


@pytest.mark.asyncio
async def test_resume_keeps_the_chosen_title(tmp_path: Path) -> None:
    eng = _engine(tmp_path, answerable=False)
    patch = await eng._node_clarify({"topic": _RAW, "title": "Chosen", "title_confirmed": True, "clarify_done": True})
    assert patch == {}


def test_clean_title_is_one_short_unquoted_line() -> None:
    assert _clean_title('  "A\n  title"  ') == "A title"
    assert _clean_title(None) == ""
    assert len(_clean_title("x" * 300)) == 120


def test_write_prompt_tells_the_writer_to_use_the_chosen_title() -> None:
    text = (Path(__file__).resolve().parents[1] / "agents" / "write.md").read_text(encoding="utf-8")
    assert "## Title\n$title" in text


async def test_a_callback_that_answers_nothing_never_loops_the_run(tmp_path: Path, monkeypatch) -> None:
    """Through the real graph: an empty answer means "use the defaults". LangGraph reads an empty resume as no
    resume at all, which once re-fired the clarify pause forever (CI hung for hours)."""
    import asyncio

    from tests.test_engine_smoke import _fake_response_for

    class _Reached(Exception):
        pass

    async def fake_chat(self, messages, **kw):  # noqa: ANN001, ANN003
        return _fake_response_for(messages[-1]["content"])

    async def stop_here(self, state):  # noqa: ANN001
        raise _Reached

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    monkeypatch.setattr(Engine, "_node_select_skills", stop_here)
    asked: list = []

    async def callback(questions):  # noqa: ANN001
        asked.append(questions)
        return {}

    engine = Engine(_cfg(tmp_path))
    with pytest.raises(_Reached):
        await asyncio.wait_for(engine.run(clarify_callback=callback), timeout=90)
    assert len(asked) == 1, "one question round, then the defaults are used"


def test_a_placeholder_or_numbered_title_is_not_taken_literally() -> None:
    assert _clean_title("<a short, plain title>") == ""
    qs = {"title": {"question": "Name?", "default": "", "options": ["A study of X", "Y in Z"]}}
    from core.engine import _spell_out_title_options
    _spell_out_title_options(qs)
    assert qs["title"]["suggestions"] == ["A study of X", "Y in Z"]


async def test_a_dismissed_prompt_or_a_yaml_title_or_an_unattended_start_never_stalls(
        tmp_path: Path, monkeypatch) -> None:
    import asyncio

    from core.vscode_bridge import BridgeError
    from tests.test_engine_smoke import _fake_response_for

    class _Reached(Exception):
        pass

    async def fake_chat(self, messages, **kw):  # noqa: ANN001, ANN003
        return _fake_response_for(messages[-1]["content"])

    async def stop_here(self, state):  # noqa: ANN001
        raise _Reached

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    monkeypatch.setattr(Engine, "_node_select_skills", stop_here)

    async def dismissed(questions):  # noqa: ANN001
        raise BridgeError("cancelled")

    with pytest.raises(_Reached):
        await asyncio.wait_for(Engine(_cfg(tmp_path)).run(clarify_callback=dismissed), timeout=90)

    async def never(questions):  # noqa: ANN001
        await asyncio.sleep(3600)

    eng = Engine(_cfg(tmp_path / "b"))
    eng.human_feedback_timeout_s = 1
    with pytest.raises(_Reached):
        await asyncio.wait_for(eng.run(clarify_callback=never), timeout=90)


def test_the_web_clarify_payload_carries_the_title_choices_and_the_page_offers_them(tmp_path: Path) -> None:
    """The title question keeps its candidates as a plain list in the payload the quest page reads, and the page
    turns them into choices on the answer box (the question text alone serves the terminal and VS Code)."""
    from fastapi.testclient import TestClient

    from core.engine import _spell_out_title_options
    from web.server import make_app

    qs = {"title": {"question": "Name?", "default": "", "options": ["A study of X", "Y in Z"]}}
    _spell_out_title_options(qs)
    assert "A study of X | Y in Z" in qs["title"]["question"]  # the terminal and VS Code still see them in the text
    out = tmp_path / "outputs"
    (out / "q1" / ".fi").mkdir(parents=True)
    (out / "q1" / ".fi" / "clarify_questions.json").write_text(json.dumps(qs), encoding="utf-8")
    got = TestClient(make_app(out)).get("/api/quests/q1/clarify").json()
    assert got["pending"] and got["questions"]["title"]["suggestions"] == ["A study of X", "Y in Z"]
    page = (Path(__file__).resolve().parent.parent / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert "val.suggestions" in page and "datalist" in page
