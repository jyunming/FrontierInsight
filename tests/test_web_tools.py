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


# ---------------------------------------------------------------------------
# vscode_extension provider + bridge guard
# ---------------------------------------------------------------------------


def test_tools_schema_reports_no_bridge_by_default(tmp_path: Path) -> None:
    """Without a VSCode bridge port, the schema must advertise the
    provider as unavailable so the UI doesn't surface it."""
    res = _client(tmp_path).get("/api/tools/schema")
    body = res.json()
    assert body["vscode_bridge_available"] is False
    # The vscode_extension model list is always included so the UI
    # can render it the moment a bridge becomes available without
    # re-fetching schema.
    assert isinstance(body.get("vscode_extension_models"), list)
    # Source-of-truth for these labels is the canonical ensemble trio
    # in core/interview.py — pin against that helper rather than a
    # hardcoded value so refreshing the trio doesn't drift this test.
    from core.interview import ensemble_model_trio
    expected_trio = set(ensemble_model_trio("vscode_extension"))
    actual = {m["value"] for m in body["vscode_extension_models"]}
    assert actual == expected_trio, (actual, expected_trio)


def test_tools_schema_reports_bridge_when_port_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the bridge port is configured AND something is actually
    listening on it, the schema must advertise the provider as
    available. The "actually listening" check is the new TCP probe —
    without it, a stale port would silently advertise vscode_extension
    and the spawned subprocess would crash on first LLM call."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    app = make_app(output_root, vscode_bridge_port=37001)

    async def _yes(*_args, **_kwargs):
        return True
    monkeypatch.setattr("web._bridge_probe.is_bridge_listening", _yes)
    monkeypatch.setattr("web._bridge_probe.is_socket_listening", _yes)

    client = TestClient(app)
    body = client.get("/api/tools/schema").json()
    assert body["vscode_bridge_available"] is True


def test_tools_schema_hides_bridge_when_port_set_but_nothing_listens(
    tmp_path: Path,
) -> None:
    """Port number alone is no longer enough — the probe must confirm
    a listener. This guards against a stale or wrong default port
    misleading the UI into offering a path that the engine would then
    crash on. (No monkeypatch here: port 37001 is unlikely to be
    listening in the test environment.)"""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    app = make_app(output_root, vscode_bridge_port=37001)
    client = TestClient(app)
    body = client.get("/api/tools/schema").json()
    assert body["vscode_bridge_available"] is False


def test_tools_schema_lights_up_when_socket_path_listens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new IPC transport: when ``vscode_bridge_socket`` is set
    AND the socket is alive, vscode_extension surfaces in the UI
    without any TCP port being involved. This is the default path
    for --serve started near a VSCode session."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    app = make_app(output_root, vscode_bridge_socket="/tmp/fake.sock")

    async def _yes(*_args, **_kwargs):
        return True
    monkeypatch.setattr("web._bridge_probe.is_bridge_listening", _yes)
    monkeypatch.setattr("web._bridge_probe.is_socket_listening", _yes)

    client = TestClient(app)
    body = client.get("/api/tools/schema").json()
    assert body["vscode_bridge_available"] is True


def test_post_tool_with_vscode_extension_requires_bridge(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    """The interview already 400s when a user picks vscode_extension
    without a live bridge; the tools endpoint must do the same so
    the spawned subprocess doesn't crash mid-run with a cryptic
    bridge error."""
    client = _client(tmp_path)  # no bridge_port
    res = client.post(
        "/api/tools/proposal",
        json={"topic": "x", "provider": "vscode_extension"},
    )
    assert res.status_code == 400
    assert "vscode_extension" in res.text
    assert "bridge" in res.text.lower()
    assert not mock_subprocess  # nothing spawned


def test_post_tool_with_vscode_extension_and_bridge_succeeds(
    tmp_path: Path, mock_subprocess: list[list[str]],
) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    app = make_app(output_root, vscode_bridge_port=37001)
    client = TestClient(app)
    res = client.post(
        "/api/tools/proposal",
        json={"topic": "test topic", "provider": "vscode_extension"},
    )
    assert res.status_code == 200, res.text
    argv = mock_subprocess[0]
    # The per-tool --<tool>-provider flag must carry vscode_extension
    # through to the subprocess, otherwise launch.py would fall back
    # to its own default.
    assert "--proposal-provider" in argv
    assert "vscode_extension" in argv


# ---------------------------------------------------------------------------
# Ensemble profile → CLI flag mapping
# ---------------------------------------------------------------------------


def test_build_argv_proposal_off_profile_emits_no_ensemble(tmp_path: Path) -> None:
    """'off' is the documented default — the spawned subprocess
    should run the cheap single-call path with zero ensemble flags."""
    spec = TOOLS_BY_NAME["proposal"]
    argv = _build_argv(
        spec,
        {"topic": "t", "provider": "openai", "ensemble_profile": "off"},
        [], tmp_path,
    )
    assert "--proposal-ensemble" not in argv


def test_build_argv_proposal_full_profile_expands_to_trio(tmp_path: Path) -> None:
    """Picking 'full' must expand the picked provider's curated trio
    into the --proposal-ensemble CSV the CLI expects."""
    spec = TOOLS_BY_NAME["proposal"]
    argv = _build_argv(
        spec,
        {"topic": "t", "provider": "openai", "ensemble_profile": "full"},
        [], tmp_path,
    )
    assert "--proposal-ensemble" in argv
    idx = argv.index("--proposal-ensemble")
    csv = argv[idx + 1]
    # The 3-model openai trio (see ensemble_model_trios) — exact
    # values are pinned in the interview schema test, not here.
    parts = csv.split(",")
    assert len(parts) == 3
    # Proposal-side merge defaults to tournament (the launch.py CLI
    # default for --proposal-ensemble-merge) — must be in the tail.
    assert "--proposal-ensemble-merge" in argv
    assert "tournament" in argv


def test_build_argv_critique_full_profile_expands_to_trio_with_synthesize(tmp_path: Path) -> None:
    """Critique's documented merge default is `synthesize` (different
    from proposal's `tournament`). The web mapping must preserve that
    parity so behaviour doesn't drift between CLI and serve users."""
    spec = TOOLS_BY_NAME["critique"]
    argv = _build_argv(
        spec,
        {"quest_id": "abc", "provider": "claude_cli", "ensemble_profile": "full"},
        [], tmp_path,
    )
    assert "--critique-ensemble" in argv
    assert "--critique-ensemble-merge" in argv
    assert "synthesize" in argv


def test_build_argv_unknown_profile_treated_as_off(tmp_path: Path) -> None:
    """An unknown profile name must NOT silently fan out — fall back
    to the single-call path (same as 'off')."""
    spec = TOOLS_BY_NAME["proposal"]
    argv = _build_argv(
        spec,
        {"topic": "t", "provider": "openai", "ensemble_profile": "bogus"},
        [], tmp_path,
    )
    assert "--proposal-ensemble" not in argv


def test_build_argv_ensemble_only_wired_for_proposal_and_critique(tmp_path: Path) -> None:
    """The other 4 LLM tools have no --<tool>-ensemble flag in
    launch.py yet — surfacing the profile would silently discard it,
    so the argv builder must drop it entirely for these tools."""
    for name in ("digest", "portfolio", "summarize", "analyze"):
        spec = TOOLS_BY_NAME[name]
        payload = {
            "provider": "openai", "ensemble_profile": "full",
            # Tool-specific required fields
            "days": 7, "folder": str(tmp_path), "path": str(tmp_path),
            "topic": "t", "kind": "auto",
        }
        try:
            argv = _build_argv(spec, payload, [], tmp_path)
        except ValueError:
            continue  # missing-arg path is fine; we're checking flag absence
        assert f"--{name}-ensemble" not in argv, name


# ---------------------------------------------------------------------------
# /api/provider/models — dynamic discovery + cache
# ---------------------------------------------------------------------------


def test_provider_models_endpoint_returns_static_marker_for_unknown_provider(
    tmp_path: Path,
) -> None:
    """An empty / unknown provider must NOT 500 — answer
    ``{source: "static"}`` so the UI keeps the schema fallback."""
    res = _client(tmp_path).get("/api/provider/models?provider=")
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "static"
    assert body["models"] == []


def test_provider_models_endpoint_returns_dynamic_when_discovery_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: discovery returns a list → endpoint reports
    source="dynamic" and the models flow through to the UI."""
    from web.tools_routes import _provider_models_cache_clear
    _provider_models_cache_clear()

    async def fake_discover(provider: str, **_kwargs):
        assert provider == "openai"
        return [
            {"value": "gpt-5", "label": "gpt-5", "description": ""},
            {"value": "gpt-4o", "label": "gpt-4o", "description": ""},
        ]
    monkeypatch.setattr(
        "core.provider_models_discover.discover", fake_discover,
    )
    body = _client(tmp_path).get("/api/provider/models?provider=openai").json()
    assert body["source"] == "dynamic"
    assert {m["value"] for m in body["models"]} == {"gpt-5", "gpt-4o"}


def test_provider_models_endpoint_falls_back_to_static_on_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery returns None → endpoint reports source="static" and
    the UI keeps the schema's curated list. This is the path CLI
    providers and offline Ollama hit."""
    from web.tools_routes import _provider_models_cache_clear
    _provider_models_cache_clear()

    async def fake_discover(*_args, **_kwargs):
        return None
    monkeypatch.setattr(
        "core.provider_models_discover.discover", fake_discover,
    )
    body = _client(tmp_path).get("/api/provider/models?provider=claude_cli").json()
    assert body["source"] == "static"
    assert body["models"] == []


def test_provider_models_endpoint_caches_results_within_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling the endpoint twice in a row must only invoke discovery
    once — the picker re-fetches on every provider change, so an
    uncached endpoint would hammer the upstream."""
    from web.tools_routes import _provider_models_cache_clear
    _provider_models_cache_clear()
    calls = {"n": 0}

    async def fake_discover(provider: str, **_kwargs):
        calls["n"] += 1
        return [{"value": "gpt-5", "label": "gpt-5", "description": ""}]
    monkeypatch.setattr(
        "core.provider_models_discover.discover", fake_discover,
    )
    client = _client(tmp_path)
    first = client.get("/api/provider/models?provider=openai").json()
    second = client.get("/api/provider/models?provider=openai").json()
    assert calls["n"] == 1
    assert first == second
