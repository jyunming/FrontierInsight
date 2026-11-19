"""The quest index on the web page: a quest in another folder opens by its (short) id, resumes where it is, and the
dashboard can list every quest FI has run on this computer. A temporary ``FI_HOME`` is used (tests/conftest.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

from core import quest_index
from web.server import make_app

QID = "1790003131-energy-drift-479b06"


def _study(tmp_path: Path, qid: str = QID, title: str = "Energy drift") -> Path:
    folder = tmp_path / "study_a"
    root = folder / "outputs" / qid
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "state.sqlite").write_bytes(b"")
    (root / "config.yaml").write_text(f"topic: t\ntitle: {title}\n", encoding="utf-8")
    (root / "paper").mkdir()
    (root / "paper" / "paper.md").write_text(f"# {title}\n\nBody.\n", encoding="utf-8")
    quest_index.register(root, working_folder=folder, title=title)
    return root


def _client(tmp_path: Path) -> TestClient:
    output_root = tmp_path / "study_b" / "outputs"
    output_root.mkdir(parents=True)
    return TestClient(make_app(output_root))


class _FakeProc:
    pid = 4242

    def poll(self):  # noqa: ANN201
        return None


def test_a_quest_in_another_folder_is_served_by_its_short_id(tmp_path: Path) -> None:
    _study(tmp_path)
    client = _client(tmp_path)
    got = client.get("/api/quests/479b06/paper")
    assert got.status_code == 200, got.text
    assert "Energy drift" in got.text
    assert client.get(f"/api/quests/{QID}/paper").status_code == 200


def test_the_quest_page_uses_the_full_id(tmp_path: Path) -> None:
    _study(tmp_path)
    page = _client(tmp_path).get("/quest/479b06")
    assert page.status_code == 200
    assert f'window.__fi_quest_id = "{QID}"' in page.text


def test_an_ambiguous_short_id_is_a_clear_error_listing_the_quests(tmp_path: Path) -> None:
    _study(tmp_path, "1790000001-first-c0ffee", "First")
    _study(tmp_path, "1790000002-second-c0ffee", "Second")
    got = _client(tmp_path).get("/api/quests/c0ffee/paper")
    assert got.status_code == 409
    assert "1790000001-first-c0ffee" in got.text and "1790000002-second-c0ffee" in got.text


def test_resume_runs_the_quest_in_its_own_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _study(tmp_path)
    captured: list[tuple[list[str], dict]] = []

    def fake_popen(argv, **kw):  # noqa: ANN001, ANN003
        captured.append((argv, kw))
        return _FakeProc()

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen)
    got = _client(tmp_path).post("/api/quests/479b06/resume")
    assert got.status_code == 200, got.text
    assert got.json()["quest_id"] == QID
    argv, kw = captured[-1]
    assert argv[argv.index("--resume") + 1] == QID
    assert Path(argv[argv.index("--config") + 1]) == root.resolve() / "config.yaml"
    assert Path(argv[argv.index("--output") + 1]) == root.resolve().parent
    assert argv.count("--output") == 1
    assert Path(kw["cwd"]) == (tmp_path / "study_a").resolve()


def test_the_dashboard_lists_quests_from_every_folder_with_their_short_id(tmp_path: Path) -> None:
    root = _study(tmp_path)
    client = _client(tmp_path)
    here = client.app.state.output_root / "1790000005-local-one-5a5a5a"  # type: ignore[attr-defined]
    (here / ".fi").mkdir(parents=True)
    listed = client.get("/api/quests").json()["quests"]
    # one list: this folder's quests first, then every quest FI has run elsewhere (a round gives each study its own
    # folder, and the page showed only the one it was started on)
    assert [(q["quest_id"], q["short_id"]) for q in listed] == [("1790000005-local-one-5a5a5a", "5a5a5a"), (QID, "479b06")]
    assert listed[1]["quest_root"] == str(root.resolve())


def test_an_unknown_id_still_404s(tmp_path: Path) -> None:
    got = _client(tmp_path).get("/api/quests/1790000000-nothing-000000/paper")
    assert got.status_code == 404


def test_a_quest_whose_process_runs_is_listed_running_however_long_since_its_last_log_line(tmp_path: Path) -> None:
    """A long search wrote run.log every quarter of an hour; the list called it stopped while its process ran. FI's own
    record of the quest's process decides, not the log's time."""
    import json
    import os
    import time

    psutil = pytest.importorskip("psutil")
    root = _study(tmp_path)
    log = root / ".fi" / "run.log"
    log.write_text("x", encoding="utf-8")
    old = time.time() - 3600
    os.utime(log, (old, old))
    me = {"pid": os.getpid(), "started": psutil.Process(os.getpid()).create_time(), "seen_at": time.time(), "ended": None}
    (root / ".fi" / "process.json").write_text(json.dumps(me), encoding="utf-8")
    client = _client(tmp_path)
    mine = [q for q in client.get("/api/quests").json()["quests"] if q["quest_id"] == QID]
    assert mine and mine[0]["verdict"] == "(running)"
    assert quest_index.status(root) == "running"
    # a process that ended, or a record naming no live process, is not running even with a fresh log line
    os.utime(log, None)
    (root / ".fi" / "process.json").write_text(json.dumps({**me, "ended": "finished"}), encoding="utf-8")
    assert quest_index.status(root) != "running"
    (root / ".fi" / "process.json").write_text(json.dumps({**me, "started": 1.0}), encoding="utf-8")
    assert quest_index.status(root) != "running"


def test_a_quest_running_from_a_terminal_shows_running_and_is_not_resumed_or_renamed_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The list said running while the quest's own page offered Resume: a second run of the same quest would have
    shared its files. Every route asks the same question."""
    import json
    import os
    import time

    psutil = pytest.importorskip("psutil")
    root = _study(tmp_path)
    me = {"pid": os.getpid(), "started": psutil.Process(os.getpid()).create_time(), "seen_at": time.time(), "ended": None}
    (root / ".fi" / "process.json").write_text(json.dumps(me), encoding="utf-8")
    launched: list[list[str]] = []
    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", lambda argv, **kw: launched.append(argv) or _FakeProc())
    client = _client(tmp_path)
    assert client.get(f"/api/quests/{QID}").json()["alive"] is True
    got = client.post(f"/api/quests/{QID}/resume")
    assert got.status_code == 409 and "still running" in got.text and not launched
    assert client.post(f"/api/quests/{QID}/title", json={"title": "New"}).status_code == 409
    # once its process has ended, it resumes
    (root / ".fi" / "process.json").write_text(json.dumps({**me, "ended": "stopped"}), encoding="utf-8")
    assert client.get(f"/api/quests/{QID}").json()["alive"] is False
    assert client.post(f"/api/quests/{QID}/resume").status_code == 200 and launched
