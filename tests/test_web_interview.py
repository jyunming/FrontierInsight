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
