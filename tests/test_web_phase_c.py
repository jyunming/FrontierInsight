"""Tests for Phase C — resume + tectonic install + clarify-resume +
trash bin endpoints."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

from web.server import make_app


def _client(tmp_path: Path) -> TestClient:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    app = make_app(output_root)
    return TestClient(app)


class _FakeProc:
    pid = 9999
    def poll(self): return None


@pytest.fixture
def mock_subprocess(monkeypatch: pytest.MonkeyPatch):
    captured: list[list[str]] = []

    def fake_popen(argv, **_kwargs):
        captured.append(argv)
        return _FakeProc()

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen)
    return captured


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_requires_config_yaml(tmp_path: Path) -> None:
    """Resume needs the original config.yaml in the quest dir."""
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    quest_dir = output_root / "q1"
    (quest_dir / ".fi").mkdir(parents=True)
    (quest_dir / ".fi" / "state.sqlite").write_bytes(b"")
    # no config.yaml
    res = client.post("/api/quests/q1/resume")
    assert res.status_code == 400
    assert "config.yaml" in res.text.lower()


def test_resume_requires_checkpoint(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    quest_dir = output_root / "q2"
    quest_dir.mkdir()
    (quest_dir / "config.yaml").write_text("topic: x", encoding="utf-8")
    # no state.sqlite
    res = client.post("/api/quests/q2/resume")
    assert res.status_code == 400
    assert "checkpoint" in res.text.lower()


def test_resume_happy_path_spawns_subprocess(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    quest_dir = output_root / "q3"
    (quest_dir / ".fi").mkdir(parents=True)
    (quest_dir / ".fi" / "state.sqlite").write_bytes(b"")
    (quest_dir / "config.yaml").write_text("topic: x", encoding="utf-8")
    res = client.post("/api/quests/q3/resume")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["resumed"] is True
    assert body["quest_id"] == "q3"
    # Argv includes --resume <id> --config <yaml>
    argv = mock_subprocess[0]
    assert "--resume" in argv
    assert "q3" in argv
    assert "--config" in argv


def test_resume_rejects_path_traversal(tmp_path: Path) -> None:
    client = _client(tmp_path)
    for hostile in ("../somewhere", "a/b"):
        res = client.post(f"/api/quests/{hostile}/resume")
        assert res.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# Tectonic install
# ---------------------------------------------------------------------------


def test_install_tectonic_spawns_subprocess(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    client = _client(tmp_path)
    res = client.post("/api/system/install-tectonic")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["job_id"].startswith("tectonic-")
    assert "--install-tectonic" in mock_subprocess[0]


# ---------------------------------------------------------------------------
# Trash
# ---------------------------------------------------------------------------


def test_trash_moves_quest_dir(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    quest_dir = output_root / "q4"
    quest_dir.mkdir()
    (quest_dir / "marker.txt").write_text("hello", encoding="utf-8")
    res = client.delete("/api/quests/q4")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["trashed"] is True
    assert body["bin_id"].startswith("q4-")
    # Original gone, trash entry exists.
    assert not quest_dir.exists()
    trash_dir = output_root / "_trash"
    assert (trash_dir / body["bin_id"]).is_dir()
    assert (trash_dir / body["bin_id"] / "marker.txt").read_text(encoding="utf-8") == "hello"


def test_trash_404_when_quest_missing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.delete("/api/quests/never-existed")
    assert res.status_code == 404


def test_list_trash(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    quest_dir = output_root / "q5"
    quest_dir.mkdir()
    client.delete("/api/quests/q5")
    res = client.get("/api/trash")
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1
    assert items[0]["bin_id"].startswith("q5-")


def test_restore_moves_back(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    quest_dir = output_root / "q6"
    quest_dir.mkdir()
    (quest_dir / "marker.txt").write_text("x", encoding="utf-8")
    bin_id = client.delete("/api/quests/q6").json()["bin_id"]
    res = client.post(f"/api/trash/{bin_id}/restore")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["quest_id"] == "q6"
    assert (output_root / "q6" / "marker.txt").exists()


def test_restore_refuses_when_target_exists(tmp_path: Path) -> None:
    """If the user re-created a quest with the same id while one
    was in trash, restore must refuse rather than clobber."""
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    (output_root / "q7").mkdir()
    bin_id = client.delete("/api/quests/q7").json()["bin_id"]
    # Recreate the quest dir.
    (output_root / "q7").mkdir()
    res = client.post(f"/api/trash/{bin_id}/restore")
    assert res.status_code == 409


def test_purge_deletes_forever(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    (output_root / "q8").mkdir()
    bin_id = client.delete("/api/quests/q8").json()["bin_id"]
    res = client.delete(f"/api/trash/{bin_id}")
    assert res.status_code == 200
    assert not (output_root / "_trash" / bin_id).exists()


def test_purge_rejects_path_traversal(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.delete("/api/trash/..")
    assert res.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# Provider availability + pages
# ---------------------------------------------------------------------------


def test_provider_availability_endpoint(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.get("/api/providers/availability")
    assert res.status_code == 200
    payload = res.json()
    assert "providers" in payload
    assert len(payload["providers"]) >= 5  # at least openai, codex, claude_cli, etc.
    for p in payload["providers"]:
        assert "name" in p and "available" in p


def test_static_pages_render(tmp_path: Path) -> None:
    client = _client(tmp_path)
    for path in ("/trash", "/settings", "/about", "/compare"):
        res = client.get(path)
        # Compare doesn't have a static page yet — that's Phase E.
        # It'll return 500 with our placeholder message until then.
        # The other three exist.
        if path == "/compare":
            assert res.status_code in (200, 500)
        else:
            assert res.status_code == 200, f"{path} returned {res.status_code}"


# ---------------------------------------------------------------------------
# Phase E — fancy features
# ---------------------------------------------------------------------------


def test_labels_get_returns_empty_when_no_file(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    (output_root / "q-tag1" / ".fi").mkdir(parents=True)
    res = client.get("/api/quests/q-tag1/labels")
    assert res.status_code == 200
    assert res.json()["labels"] == []


def test_labels_put_then_get_round_trip(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    (output_root / "q-tag2" / ".fi").mkdir(parents=True)
    res = client.put("/api/quests/q-tag2/labels",
                     json={"labels": ["euv", "baseline", "deep-dive"]})
    assert res.status_code == 200
    res = client.get("/api/quests/q-tag2/labels")
    assert res.json()["labels"] == ["euv", "baseline", "deep-dive"]


def test_labels_strips_empty_strings(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    (output_root / "q-tag3" / ".fi").mkdir(parents=True)
    res = client.put("/api/quests/q-tag3/labels",
                     json={"labels": ["a", "", "  ", "b"]})
    assert res.json()["labels"] == ["a", "b"]


def test_cost_endpoint_returns_records(tmp_path: Path) -> None:
    """Engine writes cost.jsonl per chat call. The endpoint reads
    + parses + returns the lines."""
    import json as _json
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    fi_dir = output_root / "q-cost" / ".fi"
    fi_dir.mkdir(parents=True)
    (fi_dir / "cost.jsonl").write_text(
        "\n".join([
            _json.dumps({"ts": 1.0, "node": "ideate", "model": "gpt-4o",
                         "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                                   "total_tokens": 150},
                         "cost_usd": 0.0008}),
            _json.dumps({"ts": 2.0, "node": "write", "model": "gpt-4o",
                         "usage": None, "cost_usd": None}),
            "",  # blank line tolerated
        ]),
        encoding="utf-8",
    )
    res = client.get("/api/quests/q-cost/cost")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is True
    assert len(body["records"]) == 2


def test_cost_endpoint_when_no_file(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    (output_root / "q-nocost" / ".fi").mkdir(parents=True)
    res = client.get("/api/quests/q-nocost/cost")
    assert res.json()["available"] is False
    assert res.json()["records"] == []


def test_execute_edit_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without FI_WEB_ALLOW_EXEC_EDIT=1, the re-execute endpoint
    returns 403."""
    monkeypatch.delenv("FI_WEB_ALLOW_EXEC_EDIT", raising=False)
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    quest_dir = output_root / "q-code"
    quest_dir.mkdir()
    (quest_dir / "config.yaml").write_text("topic: x", encoding="utf-8")
    res = client.post("/api/quests/q-code/code/execute",
                      json={"code": "print('hi')"})
    assert res.status_code == 403
    assert "disabled" in res.text.lower()


def test_iterations_endpoint(tmp_path: Path) -> None:
    """Returns iterations as a list; minimal when only paper.md exists."""
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    paper_dir = output_root / "q-iter" / "paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper.md").write_text("# Current", encoding="utf-8")
    (paper_dir / "paper.iter-1.md").write_text("# v1", encoding="utf-8")
    res = client.get("/api/quests/q-iter/iterations")
    assert res.status_code == 200
    its = res.json()["iterations"]
    assert len(its) == 2
    assert its[0]["iter"] == 1
    assert "current" in its[1].get("label", "")


def test_files_endpoint(tmp_path: Path) -> None:
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    qr = output_root / "q-files"
    (qr / ".fi").mkdir(parents=True)
    (qr / ".fi" / "should-be-hidden.txt").write_text("x", encoding="utf-8")
    (qr / "paper").mkdir()
    (qr / "paper" / "paper.md").write_text("# t", encoding="utf-8")
    (qr / "figures").mkdir()
    (qr / "figures" / "f1.png").write_bytes(b"\x89PNG")
    res = client.get("/api/quests/q-files/files")
    paths = {f["path"] for f in res.json()["files"]}
    assert "paper/paper.md" in paths
    assert "figures/f1.png" in paths
    # .fi/ contents are hidden.
    assert not any(".fi" in p for p in paths)


def test_quest_zip_download(tmp_path: Path) -> None:
    import zipfile, io
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    qr = output_root / "q-zip"
    qr.mkdir()
    (qr / "paper").mkdir()
    (qr / "paper" / "paper.md").write_text("# t", encoding="utf-8")
    res = client.get("/api/quests/q-zip/download")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(res.content))
    assert "paper/paper.md" in z.namelist()
