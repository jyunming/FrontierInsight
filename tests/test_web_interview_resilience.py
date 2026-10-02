"""The web interview's server side survives a lost answer, a double click and a restart.

A Launch carries a key (``Idempotency-Key``, or ``submit_key`` in the body). The server answers a second submit with
the same key with the first answer and starts no second quest: whether the first answer was lost on the way (the quest
had started), the button was pressed twice, or the server was restarted in between (the answer is kept on disk). The
answers of an interview not launched yet are kept as they are typed (``/api/interview/draft/<id>``), removed once the
quest is launched. The browser half of this (the page's retry, its saved answers) is in
``tests/test_web_interview_browser.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from tests.test_web_interview import _model_ready, _ok_answers_payload
from web.server import make_app

KEY = "k-0123456789abcdef"


@pytest.fixture(autouse=True)
def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _model_ready(monkeypatch)  # Launch checks the chosen model first; here it passes with no call made
DRAFT = "d-0123456789abcdef"


def _app(output_root: Path, launches: list[str]):
    app = make_app(output_root)

    def fake_launch(*, quest_id: str, yaml_path: Path):
        launches.append(quest_id)
        return SimpleNamespace(quest_id=quest_id, pid=4242)

    app.state.launcher.launch = fake_launch
    return app


def test_a_retry_with_the_same_key_gets_the_same_quest_and_starts_no_second_one(tmp_path: Path) -> None:
    launches: list[str] = []
    client = TestClient(_app(tmp_path / "out", launches))
    payload = {**_ok_answers_payload(), "submit_key": KEY}
    first = client.post("/api/interview/submit?launch=true", json=payload)
    again = client.post("/api/interview/submit?launch=true", json=payload, headers={"Idempotency-Key": KEY})
    assert first.status_code == again.status_code == 200
    assert again.json()["quest_id"] == first.json()["quest_id"]
    assert again.headers.get("Idempotent-Replayed") == "true"
    assert len(launches) == 1
    assert len(list((tmp_path / "out" / "_drafts").glob("*.yaml"))) == 1, "a retry wrote a second config"


def test_the_key_in_the_header_alone_is_enough(tmp_path: Path) -> None:
    launches: list[str] = []
    client = TestClient(_app(tmp_path / "out", launches))
    for _ in range(3):
        res = client.post("/api/interview/submit?launch=true", json=_ok_answers_payload(),
                          headers={"Idempotency-Key": KEY})
        assert res.status_code == 200
    assert len(launches) == 1


def test_two_submits_at_the_same_moment_start_one_quest(tmp_path: Path) -> None:
    launches: list[str] = []
    app = _app(tmp_path / "out", launches)
    payload = {**_ok_answers_payload(), "submit_key": KEY}

    async def both():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fi.test") as client:
            return await asyncio.gather(*(client.post("/api/interview/submit?launch=true", json=payload)
                                          for _ in range(2)))

    a, b = asyncio.run(both())
    assert a.status_code == b.status_code == 200
    assert a.json()["quest_id"] == b.json()["quest_id"]
    assert len(launches) == 1


def test_the_answer_outlives_a_server_restart(tmp_path: Path) -> None:
    out = tmp_path / "out"
    launches: list[str] = []
    payload = {**_ok_answers_payload(), "submit_key": KEY}
    first = TestClient(_app(out, launches)).post("/api/interview/submit?launch=true", json=payload)
    # The answer was lost and the server restarted before the page tried again.
    again = TestClient(_app(out, launches)).post("/api/interview/submit?launch=true", json=payload)
    assert again.json()["quest_id"] == first.json()["quest_id"]
    assert len(launches) == 1


def test_different_keys_are_different_quests(tmp_path: Path) -> None:
    launches: list[str] = []
    client = TestClient(_app(tmp_path / "out", launches))
    a = client.post("/api/interview/submit?launch=true", json={**_ok_answers_payload(), "submit_key": KEY})
    b = client.post("/api/interview/submit?launch=true", json={**_ok_answers_payload(), "submit_key": KEY + "x"})
    assert a.json()["quest_id"] != b.json()["quest_id"]
    assert len(launches) == 2


def test_a_submit_that_failed_is_not_kept_so_the_retry_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from web.quest_launcher import QuestLauncherFull

    launches: list[str] = []
    app = _app(tmp_path / "out", launches)
    client = TestClient(app)
    real = app.state.launcher.launch

    def full(**_kw):
        raise QuestLauncherFull("at capacity (test)")

    app.state.launcher.launch = full
    payload = {**_ok_answers_payload(), "submit_key": KEY}
    assert client.post("/api/interview/submit?launch=true", json=payload).status_code == 503
    bad = {**payload, "output_kinds": "not a list"}
    assert client.post("/api/interview/submit?launch=true", json=bad).status_code == 400
    app.state.launcher.launch = real
    ok = client.post("/api/interview/submit?launch=true", json=payload)
    assert ok.status_code == 200 and ok.json()["launched"] is True
    assert len(launches) == 1


def test_a_submit_without_a_key_works_as_before(tmp_path: Path) -> None:
    launches: list[str] = []
    client = TestClient(_app(tmp_path / "out", launches))
    for _ in range(2):
        assert client.post("/api/interview/submit?launch=true", json=_ok_answers_payload()).status_code == 200
    assert len(launches) == 2


def test_a_key_that_is_not_a_plain_token_is_ignored(tmp_path: Path) -> None:
    launches: list[str] = []
    client = TestClient(_app(tmp_path / "out", launches))
    for _ in range(2):
        res = client.post("/api/interview/submit?launch=true",
                          json={**_ok_answers_payload(), "submit_key": "../../etc/passwd"})
        assert res.status_code == 200
    assert len(launches) == 2
    assert not list((tmp_path / "out").rglob("passwd*"))


# --- answers kept while they are typed ------------------------------------------------------------------------------


def test_answers_are_kept_and_given_back(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path / "out", []))
    assert client.get(f"/api/interview/draft/{DRAFT}").status_code == 404
    saved = {"id": DRAFT, "rev": 3, "stage": "form", "form": {"topic": "Heat flow in a fin"}}
    assert client.put(f"/api/interview/draft/{DRAFT}", json=saved).status_code == 200
    # A restarted server still has them.
    back = TestClient(_app(tmp_path / "out", [])).get(f"/api/interview/draft/{DRAFT}").json()
    assert back["form"] == {"topic": "Heat flow in a fin"} and back["rev"] == 3
    assert not list((tmp_path / "out" / "_drafts").glob("*.yaml")), "kept answers are not a proposal draft"
    assert TestClient(_app(tmp_path / "out", [])).get("/api/drafts").json()["drafts"] == []


def test_kept_answers_are_removed_once_the_quest_is_launched(tmp_path: Path) -> None:
    launches: list[str] = []
    client = TestClient(_app(tmp_path / "out", launches))
    client.put(f"/api/interview/draft/{DRAFT}", json={"id": DRAFT, "form": {"topic": "x"}})
    res = client.post("/api/interview/submit?launch=true",
                      json={**_ok_answers_payload(), "submit_key": KEY, "draft_id": DRAFT})
    assert res.status_code == 200
    assert client.get(f"/api/interview/draft/{DRAFT}").status_code == 410
    # A save the page sent just before Launch, arriving after it, does not bring them back.
    assert client.put(f"/api/interview/draft/{DRAFT}", json={"id": DRAFT, "form": {"topic": "x"}}).status_code == 409
    assert client.get(f"/api/interview/draft/{DRAFT}").status_code == 410


@pytest.mark.parametrize("bad_id", ["short", "has space here", "..%2F..%2Fx1234567", "a" * 65])
def test_a_draft_id_that_is_not_a_plain_token_is_refused(tmp_path: Path, bad_id: str) -> None:
    client = TestClient(_app(tmp_path / "out", []))
    res = client.put(f"/api/interview/draft/{bad_id}", json={"form": {}})
    assert res.status_code in (400, 404)


def test_kept_answers_must_be_a_small_json_object(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path / "out", []))
    assert client.put(f"/api/interview/draft/{DRAFT}", json=[1, 2]).status_code == 400
    assert client.put(f"/api/interview/draft/{DRAFT}", content=b"{not json").status_code == 400
    big = {"form": {"topic": "x" * (300 * 1024)}}
    assert client.put(f"/api/interview/draft/{DRAFT}", json=big).status_code == 413


def test_the_same_key_with_changed_answers_is_refused_not_dropped(tmp_path: Path) -> None:
    """The answer was lost, the person changed something, and pressed Launch again: the first quest is named, the
    change is not silently replaced by it."""
    launches: list[str] = []
    client = TestClient(_app(tmp_path / "out", launches))
    first = client.post("/api/interview/submit?launch=true", json={**_ok_answers_payload(), "submit_key": KEY})
    changed = client.post("/api/interview/submit?launch=true",
                          json={**_ok_answers_payload(), "provider_model": "gpt-5", "submit_key": KEY})
    assert changed.status_code == 409
    assert first.json()["quest_id"] in changed.json()["detail"]
    assert len(launches) == 1


def test_launched_answers_are_gone_for_a_later_visit(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path / "out", []))
    client.put(f"/api/interview/draft/{DRAFT}", json={"id": DRAFT, "form": {"topic": "x"}})
    client.post("/api/interview/submit?launch=true",
                json={**_ok_answers_payload(), "submit_key": KEY, "draft_id": DRAFT})
    assert client.get(f"/api/interview/draft/{DRAFT}").status_code == 410
