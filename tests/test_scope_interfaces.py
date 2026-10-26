"""The study picker in the terminal, the web page file channel and the quest-page endpoints."""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

import launch
from core import scope_cards
from web.server import make_app

CARDS = [
    {"question": "How does the filter width change the ripple?", "study_type": "measure"},
    {"question": "Which filter design gives the least ripple?", "study_type": "find_best_design"},
    {"question": "How does the ripple compare across three filters?", "study_type": "measure"},
]
LOG = logging.getLogger("test_scope_interfaces")


def _feed(monkeypatch, lines: list[str]) -> list[str]:
    asked: list[str] = []
    it = iter(lines)

    def fake_input(prompt: str = "") -> str:
        asked.append(prompt)
        return next(it)

    monkeypatch.setattr(builtins, "input", fake_input)
    return asked


def test_terminal_reads_a_card_number_and_asks_again_on_a_bad_one(monkeypatch) -> None:
    asked = _feed(monkeypatch, ["9", "2"])
    out = asyncio.run(launch._cli_clarify_callback(scope_cards.question(CARDS)))
    assert out == {"study": 2}
    assert len(asked) == 2
    assert scope_cards.read_answer(out, CARDS) == CARDS[1]


def test_terminal_e_n_edits_the_card(monkeypatch) -> None:
    _feed(monkeypatch, ["e 3", "How does the ripple change with three filters?"])
    out = asyncio.run(launch._cli_clarify_callback(scope_cards.question(CARDS)))
    assert out["study"]["question"] == "How does the ripple change with three filters?"
    assert out["study"]["study_type"] == "measure"
    assert scope_cards.read_answer(out, CARDS)["question"].startswith("How does the ripple change")


def test_terminal_empty_edit_keeps_the_card_and_a_sentence_is_returned(monkeypatch) -> None:
    _feed(monkeypatch, ["e 1", ""])
    out = asyncio.run(launch._cli_clarify_callback(scope_cards.question(CARDS)))
    assert out["study"]["question"] == CARDS[0]["question"]
    _feed(monkeypatch, ["Measure the ripple at ten widths"])
    out = asyncio.run(launch._cli_clarify_callback(scope_cards.question(CARDS)))
    assert out == {"study": "Measure the ripple at ten widths"}


def test_web_callback_for_the_study_uses_its_own_files(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    cb = launch._web_page_clarify_callback(fi, LOG, 0.0, name="scope")

    async def go() -> dict:
        task = asyncio.create_task(cb(scope_cards.question(CARDS)))
        for _ in range(50):
            await asyncio.sleep(0.05)
            if (fi / "scope_questions.json").is_file():
                break
        assert (fi / "scope_waiting.json").is_file()
        assert not (fi / "clarify_questions.json").exists()
        assert not (fi / "clarify_waiting.json").exists()
        (fi / "clarify_answer.json").write_text('{"x": 1}', encoding="utf-8")  # must be ignored
        (fi / "scope_answer.json").write_text('{"study": 2}', encoding="utf-8")
        return await asyncio.wait_for(task, 10)

    assert asyncio.run(go()) == {"study": 2}
    assert not (fi / "scope_waiting.json").exists()
    assert json.loads((fi / "scope_questions.json").read_text(encoding="utf-8"))["study"]["cards"]


def test_the_clarify_web_callback_keeps_its_files(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    cb = launch._web_page_clarify_callback(fi, LOG, 0.0)

    async def go() -> dict:
        task = asyncio.create_task(cb({"q": {"question": "?", "default": "a"}}))
        for _ in range(50):
            await asyncio.sleep(0.05)
            if (fi / "clarify_questions.json").is_file():
                break
        assert (fi / "clarify_waiting.json").is_file()
        (fi / "clarify_answer.json").write_text('{"q": "b"}', encoding="utf-8")
        return await asyncio.wait_for(task, 10)

    assert asyncio.run(go()) == {"q": "b"}
    assert not list(fi.glob("scope_*"))


def _quest(root: Path, qid: str = "qs") -> Path:
    (root / qid / ".fi").mkdir(parents=True)
    return root / qid


def test_server_scope_endpoint_shows_cards_and_stages_the_answer(tmp_path: Path) -> None:
    q = _quest(tmp_path)
    client = TestClient(make_app(tmp_path))
    assert client.get("/api/quests/qs/scope").json() == {"pending": False, "questions": None}
    (q / ".fi" / "scope_questions.json").write_text(json.dumps(scope_cards.question(CARDS)), encoding="utf-8")
    got = client.get("/api/quests/qs/scope").json()
    assert got["pending"] is True and len(got["questions"]["study"]["cards"]) == 3
    assert client.get("/api/quests/qs").json()["pending_scope"] is True
    r = client.post("/api/quests/qs/scope", json={"answers": {"study": 2}})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert json.loads((q / ".fi" / "scope_answer.json").read_text(encoding="utf-8")) == {"study": 2}
    assert not (q / ".fi" / "clarify_answer.json").exists()
    assert client.get("/api/quests/qs/scope").json()["pending"] is False


def test_server_scope_post_409_without_a_pending_study_and_400_without_answers(tmp_path: Path) -> None:
    q = _quest(tmp_path)
    client = TestClient(make_app(tmp_path))
    assert client.post("/api/quests/qs/scope", json={"answers": {"study": 1}}).status_code == 409
    (q / ".fi" / "scope_questions.json").write_text("{}", encoding="utf-8")
    assert client.post("/api/quests/qs/scope", json={"nope": 1}).status_code == 400
    assert not (q / ".fi" / "scope_answer.json").exists()


def test_web_scope_callback_never_reads_the_setup_answer(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    (fi / "clarify_answer.json").write_text('{"title": "set up"}', encoding="utf-8")
    cb = launch._web_page_clarify_callback(fi, LOG, 0.0, name="scope")

    async def go() -> dict:
        task = asyncio.create_task(cb(scope_cards.question(CARDS)))
        await asyncio.sleep(1.5)
        assert not task.done()  # the setup answer did not answer the study
        (fi / "scope_answer.json").write_text('{"study": 1}', encoding="utf-8")
        return await asyncio.wait_for(task, 10)

    assert asyncio.run(go()) == {"study": 1}


def test_a_study_choice_nobody_waits_on_any_more_leaves_the_page(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    cb = launch._web_page_clarify_callback(fi, LOG, 0.0, name="scope")

    async def go() -> None:
        await asyncio.wait_for(cb(scope_cards.question(CARDS)), 1.5)

    try:
        asyncio.run(go())
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("the wait should have run out")
    assert not (fi / "scope_questions.json").exists() and not (fi / "scope_waiting.json").exists()
    client = TestClient(make_app(tmp_path.parent))
    assert client.get(f"/api/quests/{tmp_path.name}").json()["pending_scope"] is False


def test_server_scope_post_refuses_an_answer_that_names_no_study_and_a_second_answer(tmp_path: Path) -> None:
    q = _quest(tmp_path)
    (q / ".fi" / "scope_questions.json").write_text(json.dumps(scope_cards.question(CARDS)), encoding="utf-8")
    client = TestClient(make_app(tmp_path))
    for bad in ({}, {"study": True}, {"study": 9}, {"study": ""}, {"study": {"study_type": "measure"}}):
        assert client.post("/api/quests/qs/scope", json={"answers": bad}).status_code == 400, bad
    assert not (q / ".fi" / "scope_answer.json").exists()
    assert client.post("/api/quests/qs/scope", json={"answers": {"study": "How fast does it cool?"}}).status_code == 200
    assert client.post("/api/quests/qs/scope", json={"answers": {"study": 1}}).status_code == 409
    assert json.loads((q / ".fi" / "scope_answer.json").read_text(encoding="utf-8")) == {"study": "How fast does it cool?"}
    assert not list((q / ".fi").glob("*.tmp"))


def test_server_scope_post_works_on_a_drive_without_links(tmp_path: Path, monkeypatch) -> None:
    import os

    def no_links(*a, **k):
        raise OSError("links are not supported here")

    monkeypatch.setattr(os, "link", no_links)
    q = _quest(tmp_path)
    (q / ".fi" / "scope_questions.json").write_text(json.dumps(scope_cards.question(CARDS)), encoding="utf-8")
    client = TestClient(make_app(tmp_path))
    assert client.post("/api/quests/qs/scope", json={"answers": {"study": 3}}).status_code == 200
    assert client.post("/api/quests/qs/scope", json={"answers": {"study": 1}}).status_code == 409
    assert json.loads((q / ".fi" / "scope_answer.json").read_text(encoding="utf-8")) == {"study": 3}
