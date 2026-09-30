"""A quest the web page starts as its own process (the interview's launch) asks its setup questions on the quest page.

The run waits for the answers the page stages in ``.fi/clarify_answer.json``; if nobody answers in time it does what
``pauses.clarify`` says, exactly as a quest the web server runs in-process does.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import launch
from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from web.server import make_app

_ENV = "FI_WEB_ANSWERS"  # set by the web launcher (web/quest_launcher.py) for the run it starts
_RAW = "verlet euler integrator step size error comparison energy drift"


def _cfg(tmp_path: Path, **kw: Any) -> Config:
    return Config(
        topic=_RAW,
        provider=ProviderConfig(name="openai"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        **kw,
    )


class _Reached(Exception):
    pass


@pytest.fixture
def stop_after_clarify(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake model; the run stops at the step after the setup questions and keeps the state it reached."""
    from tests.test_engine_smoke import _fake_response_for

    seen: dict[str, Any] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001, ANN003
        return _fake_response_for(messages[-1]["content"])

    async def stop_here(self, state):  # noqa: ANN001
        seen["state"] = dict(state)
        raise _Reached

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    monkeypatch.setattr(Engine, "_node_select_skills", stop_here)
    return seen


async def _wait_for(path: Path, timeout: float = 60.0) -> None:
    for _ in range(int(timeout * 10)):
        if path.is_file():
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"{path} never appeared")


def test_the_interview_launch_tells_the_run_the_web_page_can_answer(tmp_path: Path, monkeypatch) -> None:
    from web.quest_launcher import QuestLauncher

    captured: dict[str, Any] = {}

    class _P:
        pid = 1

        def poll(self):  # noqa: ANN201
            return None

    def fake_popen(argv, **kw):  # noqa: ANN001, ANN003
        captured["env"] = kw.get("env", {})
        return _P()

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen)
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    QuestLauncher(repo_root=tmp_path, output_root=tmp_path / "out", python_path="python").launch(
        quest_id="q1", yaml_path=yaml)
    assert captured["env"].get(_ENV) == "1"


def test_only_a_web_launched_run_gets_the_page_callback(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    eng = Engine(cfg)
    monkeypatch.delenv(_ENV, raising=False)
    assert launch._pick_clarify_callback(cfg, eng, False) is None
    monkeypatch.setenv(_ENV, "1")
    assert launch._pick_clarify_callback(cfg, eng, False) is not None
    # A terminal --interactive run keeps its own prompt.
    assert launch._pick_clarify_callback(cfg, eng, True) is launch._cli_clarify_callback


async def test_web_launched_quest_asks_on_the_page_and_uses_the_answers(
        tmp_path: Path, monkeypatch, stop_after_clarify) -> None:
    monkeypatch.setenv(_ENV, "1")
    cfg = _cfg(tmp_path)
    engine = Engine(cfg)
    callback = launch._pick_clarify_callback(cfg, engine, False)
    task = asyncio.create_task(engine.run(clarify_callback=callback))
    questions_file = engine.fi_dir / "clarify_questions.json"
    await _wait_for(questions_file)

    app = make_app(tmp_path / "out")
    app.state.launcher.job_state = lambda qid: {"alive": qid == engine.quest_id}  # the child is still running
    client = TestClient(app)
    qid = engine.quest_id

    def page() -> tuple[dict, dict, dict]:
        status = client.get(f"/api/quests/{qid}").json()
        shown = client.get(f"/api/quests/{qid}/clarify").json()
        posted = client.post(f"/api/quests/{qid}/clarify",
                             json={"answers": {"title": "Energy drift of Verlet steps"}})
        assert posted.status_code == 200, posted.text
        return status, shown, posted.json()

    status, shown, posted = await asyncio.to_thread(page)
    assert status["pending_clarify"] is True
    assert shown["pending"] is True and "budget" in shown["questions"]
    # The running quest reads the answers itself: the page must not start a second run.
    assert posted["run_waiting"] is True

    with pytest.raises(_Reached):
        await asyncio.wait_for(task, timeout=90)
    state = stop_after_clarify["state"]
    assert state["title"] == "Energy drift of Verlet steps"
    assert state.get("title_confirmed") is True
    assert not questions_file.exists() and not (engine.fi_dir / "clarify_answer.json").exists()


async def test_nobody_answers_and_the_setting_is_unset_the_run_answers_for_itself(
        tmp_path: Path, monkeypatch, stop_after_clarify) -> None:
    monkeypatch.setenv(_ENV, "1")
    cfg = _cfg(tmp_path)
    engine = Engine(cfg)
    engine.human_feedback_timeout_s = 1
    with pytest.raises(_Reached):
        await asyncio.wait_for(engine.run(clarify_callback=launch._pick_clarify_callback(cfg, engine, False)),
                               timeout=90)
    assert stop_after_clarify["state"]["clarify_done"] is True
    assert not (engine.fi_dir / "clarify_questions.json").exists()


async def test_nobody_answers_under_ask_the_quest_waits_and_a_resume_uses_the_page_answers(
        tmp_path: Path, monkeypatch, stop_after_clarify) -> None:
    monkeypatch.setenv(_ENV, "1")
    cfg = _cfg(tmp_path, pauses={"clarify": "ask"})
    engine = Engine(cfg)
    engine.human_feedback_timeout_s = 1
    await asyncio.wait_for(engine.run(clarify_callback=launch._pick_clarify_callback(cfg, engine, False)), timeout=90)
    assert "state" not in stop_after_clarify, "an unanswered 'ask' stops and waits"
    assert (engine.fi_dir / "clarify_questions.json").is_file()

    client = TestClient(make_app(tmp_path / "out"))  # nothing running: the page stages the answer and resumes
    posted = await asyncio.to_thread(lambda: client.post(
        f"/api/quests/{engine.quest_id}/clarify", json={"answers": {"title": "Verlet drift, answered later"}}).json())
    assert posted["in_process_resolved"] is False and not posted.get("run_waiting")

    # The web Resume runs the quest again as a web child: it takes the staged answers without asking again.
    resumed = Engine(cfg, resume_quest_id=engine.quest_id)
    with pytest.raises(_Reached):
        await asyncio.wait_for(
            resumed.run(clarify_callback=launch._pick_clarify_callback(cfg, resumed, False)), timeout=90)
    assert stop_after_clarify["state"]["title"] == "Verlet drift, answered later"


async def test_the_page_callback_reads_a_staged_answer_without_asking_again(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    cfg = _cfg(tmp_path)
    eng = Engine(cfg)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / "clarify_answer.json").write_text(json.dumps({"title": "Staged"}), encoding="utf-8")
    callback = launch._pick_clarify_callback(cfg, eng, False)
    assert await asyncio.wait_for(callback({"title": {"question": "?"}}), timeout=5) == {"title": "Staged"}
    assert not (eng.fi_dir / "clarify_questions.json").exists()


def test_the_resume_button_also_lets_the_page_answer(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    qid = "1782000003-web-resume"
    fi = out / qid / ".fi"
    fi.mkdir(parents=True)
    (out / qid / "config.yaml").write_text("topic: x\n", encoding="utf-8")
    (fi / "state.sqlite").write_bytes(b"")
    app = make_app(out)
    seen: dict[str, Any] = {}

    def fake_launch_command(*, argv_tail, job_id, extra_env=None):  # noqa: ANN001
        seen["env"] = extra_env or {}

        class _E:
            pid = 7
        return _E()

    app.state.launcher.launch_command = fake_launch_command
    assert TestClient(app).post(f"/api/quests/{qid}/resume").status_code == 200
    assert seen["env"].get(_ENV) == "1"
