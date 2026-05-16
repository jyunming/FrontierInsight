"""Tests for the /tools/* routes — the 8 CLI tools exposed in the
web UI (proposal / critique / digest / portfolio / summarize /
analyze / fleet / ingest).

Each route is tested with a mocked subprocess so the suite runs in
<1 s and doesn't actually spawn Python."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

from web.server import make_app
from web.tools_routes import TOOL_SPECS, TOOLS_BY_NAME, _build_argv


def _client(tmp_path: Path) -> TestClient:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    app = make_app(output_root)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Schema endpoint
# ---------------------------------------------------------------------------


def test_tools_schema_endpoint_returns_all_tools(tmp_path: Path) -> None:
    res = _client(tmp_path).get("/api/tools/schema")
    assert res.status_code == 200
    payload = res.json()
    tools = payload["tools"]
    names = {t["name"] for t in tools}
    expected = {"proposal", "critique", "digest", "portfolio",
                "summarize", "analyze", "fleet", "ingest"}
    assert names == expected


def test_tool_specs_each_have_cli_flag_and_fields() -> None:
    """Sanity: every spec must declare a CLI flag; tools that take
    inputs must declare fields."""
    for spec in TOOL_SPECS:
        assert spec.cli_flag.startswith("--"), spec.name
        # portfolio is the only zero-input tool.
        if spec.name == "portfolio":
            assert spec.fields == ()
        else:
            assert len(spec.fields) >= 1, spec.name


# ---------------------------------------------------------------------------
# /tools/<name> HTML page
# ---------------------------------------------------------------------------


def test_tool_page_renders_with_injected_name(tmp_path: Path) -> None:
    res = _client(tmp_path).get("/tools/proposal")
    assert res.status_code == 200
    assert "__fi_tool_name" in res.text
    assert "proposal" in res.text


def test_tool_page_unknown_404(tmp_path: Path) -> None:
    res = _client(tmp_path).get("/tools/made-up")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# _build_argv — pure argv-shape unit tests
# ---------------------------------------------------------------------------


def test_build_argv_proposal_requires_topic(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["proposal"]
    with pytest.raises(ValueError, match="topic"):
        _build_argv(spec, {}, [], tmp_path)


def test_build_argv_proposal_happy_path(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["proposal"]
    argv = _build_argv(spec, {"topic": "  hello world  "}, [], tmp_path)
    assert argv == ["--proposal", "hello world"]


def test_build_argv_digest_default_window(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["digest"]
    argv = _build_argv(spec, {}, [], tmp_path)
    assert argv == ["--digest", "--days", "7"]


def test_build_argv_digest_custom_window(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["digest"]
    argv = _build_argv(spec, {"days": "14"}, [], tmp_path)
    assert argv == ["--digest", "--days", "14"]


def test_build_argv_portfolio_no_inputs(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["portfolio"]
    argv = _build_argv(spec, {}, [], tmp_path)
    assert argv == ["--portfolio"]


def test_build_argv_critique_requires_quest_id(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["critique"]
    with pytest.raises(ValueError, match="quest_id"):
        _build_argv(spec, {}, [], tmp_path)


def test_build_argv_critique_happy_path(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["critique"]
    argv = _build_argv(spec, {"quest_id": "1234-x-abc"}, [], tmp_path)
    assert argv == ["--critique", "1234-x-abc"]


def test_build_argv_analyze_needs_path_or_uploads(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["analyze"]
    with pytest.raises(ValueError, match="path"):
        _build_argv(spec, {"topic": "test"}, [], tmp_path)


def test_build_argv_analyze_needs_topic(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["analyze"]
    with pytest.raises(ValueError, match="topic"):
        _build_argv(spec, {"path": "/data"}, [], tmp_path)


def test_build_argv_analyze_happy_path(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["analyze"]
    argv = _build_argv(
        spec, {"path": "/data/x", "topic": "Compare"}, [], tmp_path,
    )
    assert argv == ["--analyze", "/data/x", "--analyze-topic", "Compare"]


def test_build_argv_summarize_uses_upload_folder_when_provided(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["summarize"]
    upload = tmp_path / "_uploads" / "summarize-1234"
    upload.mkdir(parents=True)
    (upload / "a.md").write_text("a", encoding="utf-8")
    argv = _build_argv(spec, {}, [upload / "a.md"], tmp_path)
    assert "--summarize" in argv
    assert str(upload) in argv


def test_build_argv_fleet_concatenates_yamls(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["fleet"]
    payload = {"yaml_paths": "/a.yaml\n/b.yaml\n"}
    argv = _build_argv(spec, payload, [], tmp_path)
    assert argv == ["--fleet", "/a.yaml", "/b.yaml"]


def test_build_argv_fleet_with_uploads(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["fleet"]
    upload1 = tmp_path / "u1.yaml"
    upload1.write_text("topic: x", encoding="utf-8")
    argv = _build_argv(spec, {"yaml_paths": ""}, [upload1], tmp_path)
    assert argv == ["--fleet", str(upload1)]


def test_build_argv_fleet_requires_at_least_one(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["fleet"]
    with pytest.raises(ValueError, match="at least one"):
        _build_argv(spec, {}, [], tmp_path)


def test_build_argv_ingest_concatenates_paths(tmp_path: Path) -> None:
    spec = TOOLS_BY_NAME["ingest"]
    argv = _build_argv(
        spec, {"paths": "/p1.pdf\n/p2.pdf\n"}, [], tmp_path,
    )
    assert argv == ["--ingest", "/p1.pdf", "/p2.pdf"]


# ---------------------------------------------------------------------------
# POST /api/tools/<name> — end-to-end with mocked subprocess
# ---------------------------------------------------------------------------


class _FakeProc:
    pid = 4242
    def poll(self): return None


@pytest.fixture
def mock_subprocess(monkeypatch: pytest.MonkeyPatch):
    captured: list[list[str]] = []

    def fake_popen(argv, **_kwargs):
        captured.append(argv)
        return _FakeProc()

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen)
    return captured


def test_post_proposal_spawns_subprocess(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    client = _client(tmp_path)
    res = client.post(
        "/api/tools/proposal",
        json={"topic": "Compare RK4 vs Verlet"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["tool"] == "proposal"
    assert body["job_id"].startswith("proposal-")
    assert mock_subprocess  # subprocess.Popen was called
    argv = mock_subprocess[0]
    assert "--proposal" in argv
    assert "Compare RK4 vs Verlet" in argv


def test_post_digest_with_custom_days(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    client = _client(tmp_path)
    res = client.post("/api/tools/digest", json={"days": 14})
    assert res.status_code == 200
    argv = mock_subprocess[0]
    assert "--days" in argv
    assert "14" in argv


def test_post_portfolio_no_inputs_required(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    client = _client(tmp_path)
    res = client.post("/api/tools/portfolio", json={})
    assert res.status_code == 200
    assert mock_subprocess[0].count("--portfolio") == 1


def test_post_unknown_tool_returns_404(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    client = _client(tmp_path)
    res = client.post("/api/tools/no-such-tool", json={})
    assert res.status_code == 404


def test_post_missing_required_field_returns_400(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    """Proposal needs a topic; empty body must 400 with a clear message."""
    client = _client(tmp_path)
    res = client.post("/api/tools/proposal", json={})
    assert res.status_code == 400
    assert "topic" in res.text


def test_post_503_when_launcher_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web.quest_launcher import QuestLauncherFull
    client = _client(tmp_path)

    def always_full(**_kwargs):
        raise QuestLauncherFull("at capacity (test)")

    monkeypatch.setattr(
        client.app.state.launcher, "launch_command", always_full,
    )
    res = client.post("/api/tools/portfolio", json={})
    assert res.status_code == 503
    assert res.headers.get("Retry-After") == "30"
