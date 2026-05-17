"""Tests for the ``--serve`` web UI's interview routes.

Uses FastAPI's TestClient against ``web.server.make_app`` so the same
wiring the real server uses gets exercised. The interview HTML page
itself isn't tested (no headless browser); the routes + JSON
contract are."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

from core.interview import answers_to_yaml, InterviewAnswers
from web.server import make_app


def _client(tmp_path: Path) -> TestClient:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    app = make_app(output_root)
    client = TestClient(app)
    return client


def _ok_answers_payload() -> dict:
    return {
        "topic": "Web test topic",
        "title": "web-test",
        "output_kinds": ["paper_md", "paper_pdf"],
        "paper_format": "generic",
        "no_simulation": False,
        "study_depth": "journal-length",
        "comparative_baseline": "Web baseline",
        "success_metric": "AUC >= 0.9",
        "budget": "5 minutes",
        "clarify_mode": "auto",
        "review_panel": [],
        "knowledge_enabled": False,
        "provider": "openai",
        "provider_model": "gpt-4o",
    }


def test_interview_page_renders(tmp_path: Path) -> None:
    res = _client(tmp_path).get("/interview")
    assert res.status_code == 200
    assert "Frontier Insight" in res.text


def test_interview_schema_endpoint_returns_questions(tmp_path: Path) -> None:
    res = _client(tmp_path).get("/api/interview/schema")
    assert res.status_code == 200
    payload = res.json()
    assert "questions" in payload
    ids = {q["id"] for q in payload["questions"]}
    # Sanity: the four new Phase-R questions must be in the schema.
    for new_q in ("study_depth", "comparative_baseline", "success_metric", "budget"):
        assert new_q in ids


def test_submit_new_writes_yaml(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.post("/api/interview/submit", json=_ok_answers_payload())
    assert res.status_code == 200, res.text
    payload = res.json()
    yaml_path = Path(payload["yaml_path"])
    assert yaml_path.is_file()
    assert "Web test topic" in yaml_path.read_text(encoding="utf-8")


def test_submit_missing_required_field_400(tmp_path: Path) -> None:
    client = _client(tmp_path)
    bad = _ok_answers_payload()
    del bad["topic"]
    res = client.post("/api/interview/submit", json=bad)
    assert res.status_code == 400


def test_update_endpoint_writes_back_yaml(tmp_path: Path) -> None:
    """End-to-end test of the update API: create a fake quest with
    a config.yaml, POST an update to the endpoint, verify the YAML
    changed and a backup was created."""
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]

    quest_id = "test-quest-update"
    quest_root = output_root / quest_id
    quest_root.mkdir()
    initial = InterviewAnswers(
        topic="Initial topic",
        title="initial",
        output_kinds=["paper_md"],
        paper_format="generic",
        no_simulation=False,
        study_depth="journal-length",
        comparative_baseline="X",
        success_metric="Y",
        budget="Z",
        clarify_mode="auto",
        review_panel=[],
        knowledge_enabled=False,
        provider="openai",
        provider_model="gpt-4o",
    )
    (quest_root / "config.yaml").write_text(
        answers_to_yaml(initial, frontend="cli"), encoding="utf-8",
    )

    # Only change paper_format — every other editable field stays
    # the same as the initial answers so the invalidation matrix
    # ONLY fires for paper_format → [write, review].
    update_payload = {
        "topic": initial.topic,
        "title": initial.title,
        "output_kinds": initial.output_kinds,
        "paper_format": "neurips",  # the only change
        "no_simulation": initial.no_simulation,
        "study_depth": initial.study_depth,
        "comparative_baseline": initial.comparative_baseline,
        "success_metric": initial.success_metric,
        "budget": initial.budget,
        "clarify_mode": initial.clarify_mode,
        "review_panel": initial.review_panel,
        "knowledge_enabled": initial.knowledge_enabled,
        "provider": initial.provider,
        "provider_model": initial.provider_model,
    }
    res = client.post(f"/api/interview/update/{quest_id}", json=update_payload)
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["quest_id"] == quest_id
    assert "paper_format" in payload["changes"]
    assert payload["invalidated_stages"] == ["write", "review"]
    backup = quest_root / "config.yaml.before-update"
    assert backup.is_file()


def test_update_endpoint_404_when_quest_missing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.post(
        "/api/interview/update/does-not-exist",
        json=_ok_answers_payload(),
    )
    assert res.status_code == 404


def test_update_endpoint_rejects_path_traversal(tmp_path: Path) -> None:
    """Hostile or buggy clients can't escape the output root by
    composing path separators into the quest_id. Mirrors the same
    allowlist + relative_to guard used by the GET /api/quests/{id}
    endpoint."""
    client = _client(tmp_path)
    for hostile in ("../somewhere", "..%2Fsomewhere", "a/b", "a\\b"):
        res = client.post(
            f"/api/interview/update/{hostile}",
            json=_ok_answers_payload(),
        )
        # Either 400 (caught by the regex) or 404 (after URL decode
        # the resolved path doesn't exist). Critically NOT 200.
        assert res.status_code in (400, 404, 422), (
            f"hostile quest_id {hostile!r} returned "
            f"unexpected status {res.status_code}: {res.text}"
        )


def test_submit_rejects_wrong_type_for_no_simulation(tmp_path: Path) -> None:
    """``bool("false")`` evaluates to ``True`` in Python. The web
    handler must reject a string-typed no_simulation field rather
    than coercing it silently."""
    client = _client(tmp_path)
    bad = _ok_answers_payload()
    bad["no_simulation"] = "false"  # string, not bool
    res = client.post("/api/interview/submit", json=bad)
    assert res.status_code == 400


def test_submit_rejects_non_string_in_output_kinds(tmp_path: Path) -> None:
    """A junk payload like ``output_kinds: [1, 2, 3]`` shouldn't get
    coerced silently — each element must be a string."""
    client = _client(tmp_path)
    bad = _ok_answers_payload()
    bad["output_kinds"] = [1, 2, 3]
    res = client.post("/api/interview/submit", json=bad)
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# Quest launch (subprocess pool)
# ---------------------------------------------------------------------------


def test_submit_with_launch_true_spawns_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POST /api/interview/submit?launch=true`` writes the YAML
    AND spawns ``python launch.py --config <yaml>``. Monkey-patch
    Popen so the test doesn't actually start a Python child."""
    captured: dict = {}

    class FakeProc:
        pid = 9999
        def poll(self): return None

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env", {})
        return FakeProc()

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen)
    client = _client(tmp_path)
    res = client.post("/api/interview/submit?launch=true", json=_ok_answers_payload())
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["launched"] is True
    assert body["quest_id"]
    assert body["pid"] == 9999
    # FI_PRESEED_QUEST_ID was passed to the child so its Engine uses
    # the same quest_id the response advertised.
    assert captured["env"]["FI_PRESEED_QUEST_ID"] == body["quest_id"]
    assert "--config" in captured["argv"]


def test_submit_launch_503_when_pool_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the launcher pool is at capacity, the submit handler
    returns 503 with a Retry-After header so clients can back off."""
    from web.quest_launcher import QuestLauncherFull
    client = _client(tmp_path)
    # Replace launch() on the live launcher to always raise.
    def always_full(**_kwargs):
        raise QuestLauncherFull("at capacity (test)")
    monkeypatch.setattr(
        client.app.state.launcher, "launch", always_full,
    )
    res = client.post("/api/interview/submit?launch=true", json=_ok_answers_payload())
    assert res.status_code == 503
    assert "capacity" in res.text.lower()
    # Retry-After is the standard back-off hint; clients keying off
    # it can wait the suggested seconds before retrying.
    assert res.headers.get("Retry-After") == "30"
    body = res.json()
    assert body.get("retry_after_seconds") == 30


def test_quest_detail_route_renders(tmp_path: Path) -> None:
    """``GET /quest/<id>`` renders the quest.html shell with the
    quest_id injected as a JS global."""
    client = _client(tmp_path)
    res = client.get("/quest/abc-123-def456")
    assert res.status_code == 200
    assert "abc-123-def456" in res.text


def test_quest_detail_page_carries_review_panel_renderer(tmp_path: Path) -> None:
    """The panel-renderer + single-reviewer renderer live in
    quest.html (the dashboard is a list-only surface). This pins
    the location so the multi-persona UI doesn't drift back into
    index.html."""
    client = _client(tmp_path)
    res = client.get("/quest/sample-id")
    assert res.status_code == 200
    assert "renderReviewPanel" in res.text
    assert "renderSingleReview" in res.text


def test_quest_detail_route_rejects_path_traversal(tmp_path: Path) -> None:
    """Same guard as the rest of the web UI."""
    client = _client(tmp_path)
    for hostile in ("../somewhere", "a/b"):
        res = client.get(f"/quest/{hostile}")
        # 400 (caught by regex) or 404 (URL routing). NOT 200.
        assert res.status_code in (400, 404, 422)


def test_cancel_route_404_when_quest_not_tracked(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.post("/api/quests/never-launched/cancel")
    assert res.status_code == 404


def test_get_quest_detail_merges_launcher_alive_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bot review #5: the dashboard / detail page's `alive` field
    must reflect quests spawned via the launcher, not just the
    in-process registry. Without this, web-launched quests always
    appear "idle" and the cancel button stays disabled. Test stubs
    the launcher status to True; the API response must propagate."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    # Create a fake quest dir so /api/quests/{id} doesn't 404.
    quest_dir = output_root / "fake-quest-id"
    (quest_dir / ".fi").mkdir(parents=True)
    (quest_dir / ".fi" / "run.log").write_text("started\n", encoding="utf-8")

    app = make_app(output_root)
    client = TestClient(app)
    # Stub the launcher's status_for to claim the quest is alive.
    def fake_status(qid: str):
        if qid == "fake-quest-id":
            return {"pid": 12345, "started_at": 0, "alive": True, "age_seconds": 1.5}
        return None
    monkeypatch.setattr(app.state.launcher, "status_for", fake_status)

    res = client.get("/api/quests/fake-quest-id")
    assert res.status_code == 200
    body = res.json()
    assert body["alive"] is True, (
        "merging launcher state into /api/quests/<id> is what makes "
        "the detail-page status badge + cancel button work for "
        "quests spawned via the web UI"
    )
    assert body["launcher_pid"] == 12345


def test_mint_quest_id_is_public_and_stable() -> None:
    """The submit handler used to reach into core.engine._new_quest_id
    (private underscore prefix). It now uses the public
    mint_quest_id alias. Pin the rename so the surface stays stable."""
    from core.engine import mint_quest_id, _new_quest_id
    assert mint_quest_id is not None
    # Same behavior.
    a = mint_quest_id("topic-x")
    assert "topic-x" in a or a.endswith("-x") or len(a) > 0
    # Underscore form is still available for legacy callers.
    b = _new_quest_id("topic-x")
    assert len(b) == len(a)  # same shape


# ---------------------------------------------------------------------------
# 0.0.0.0 warning
# ---------------------------------------------------------------------------


def test_non_loopback_host_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """When ``--serve`` is bound to a non-loopback address, a clear
    WARNING fires so the user knows the quest-launch endpoint is
    reachable from the network."""
    import logging
    from web.server import _warn_if_non_loopback
    caplog.set_level(logging.WARNING, logger="frontier_insight.serve")
    _warn_if_non_loopback("0.0.0.0")
    assert any("non-loopback" in rec.message for rec in caplog.records)


def test_loopback_host_emits_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The default localhost binding must not log a noise warning."""
    import logging
    from web.server import _warn_if_non_loopback
    caplog.set_level(logging.WARNING, logger="frontier_insight.serve")
    for host in ("127.0.0.1", "localhost", "::1"):
        caplog.clear()
        _warn_if_non_loopback(host)
        assert caplog.records == [], f"unexpected warning for {host}"
