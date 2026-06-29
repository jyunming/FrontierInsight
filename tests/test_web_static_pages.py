"""Tests for the web UI's static / utility endpoints: resume,
tectonic install, clarify-resume, trash bin, knowledge info."""

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

    # rerun=true uses --rerun (re-open a finished quest) instead of --resume.
    res2 = client.post("/api/quests/q3/resume?rerun=true")
    assert res2.status_code == 200, res2.text
    argv2 = mock_subprocess[1]
    assert "--rerun" in argv2 and "--resume" not in argv2
    assert "q3" in argv2


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock both presence probes (tools/tectonic + PATH) so the
    endpoint actually spawns instead of short-circuiting with
    already_present=True. Without this, the test was flaky
    depending on whether prior smoke runs left tools/tectonic.exe
    on disk."""
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda name: None)
    real_is_file = Path.is_file
    def patched_is_file(self):
        if self.name.startswith("tectonic"):
            return False
        return real_is_file(self)
    monkeypatch.setattr(Path, "is_file", patched_is_file)
    client = _client(tmp_path)
    res = client.post("/api/system/install-tectonic")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body.get("spawned") is True, body
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
    """The operational --serve UI exposes /trash, /settings, /compare.
    /about deliberately is NOT here — the landing page lives in
    marketing/index.html for separate deploy (GitHub Pages etc.)."""
    client = _client(tmp_path)
    for path in ("/trash", "/settings", "/compare"):
        res = client.get(path)
        assert res.status_code == 200, f"{path} returned {res.status_code}"
    # /about should NOT be served from --serve.
    res = client.get("/about")
    assert res.status_code == 404


def test_dashboard_quest_card_escapes_label_to_prevent_xss() -> None:
    """The status pill on each quest card renders `${s.label}`,
    where `s.label = String(q.verdict || 'idle')`. The verdict
    is engine-generated text (LLM review output), so a hostile
    LLM could embed HTML/JS. The renderer MUST escape it.
    Caught by Copilot bot review on PR #103."""
    from pathlib import Path as _P
    page = _P(__file__).resolve().parent.parent / "web" / "static" / "index.html"
    text = page.read_text(encoding="utf-8")
    # The pill's label substitution must go through escapeHtml.
    assert "${escapeHtml(s.label)}" in text, (
        "quest card pill label must escape s.label to defend against "
        "LLM-generated verdicts containing HTML/script"
    )
    # And the raw-html slot (pillIcon) must still be inline so the
    # animated dot for running quests renders.
    assert "${s.pillIcon}" in text, "pillIcon stays raw — it's a trusted span literal"


def test_dashboard_surfaces_loadquests_failure_instead_of_silent() -> None:
    """loadQuests() previously swallowed every error. When the server
    was down, the page just looked empty. After the bot's review we
    surface a banner + auto-retry."""
    from pathlib import Path as _P
    page = _P(__file__).resolve().parent.parent / "web" / "static" / "index.html"
    text = page.read_text(encoding="utf-8")
    # The /* silent */ tombstone is gone…
    assert "/* silent */" not in text, (
        "the silent-swallow comment marks dead UX; replace with a banner"
    )
    # …replaced with a console.warn + an inline retry.
    assert "console.warn('[fi] /api/quests failed:'" in text
    assert "setTimeout(loadQuests, 5000)" in text


def test_marketing_landing_page_is_self_contained() -> None:
    """marketing/index.html is intended to be deployed separately
    (GitHub Pages, Netlify, etc.) as a marketing page. It must
    have no external dependencies — no /static/* refs, no external
    URLs except links to GitHub. Single-file deploy."""
    from pathlib import Path as _P
    page = _P(__file__).resolve().parent.parent / "marketing" / "index.html"
    assert page.is_file(), "marketing/index.html must exist"
    text = page.read_text(encoding="utf-8")
    # No links to /static/* or /api/* — those are --serve-only paths.
    assert "/static/" not in text, "marketing page must not reference /static/* (deploys separately)"
    assert "/api/" not in text, "marketing page must not reference /api/* (no FastAPI server on the marketing host)"
    # Has the inline SVG logo (not a /static/logo.svg ref).
    assert "<svg" in text and "viewBox=" in text


def test_marketing_page_respects_prefers_reduced_motion() -> None:
    """Accessibility contract: the hero terminal's fade-in animation
    and the status pulse can trigger motion sensitivity. PR 103
    review flagged this; the fix adds a ``prefers-reduced-motion``
    block that disables the animations. Pin it here so a future edit
    can't quietly drop the rule.

    Strong check: parse the actual ``@media (prefers-reduced-motion:
    reduce) {...}`` block and assert that both animated selectors
    declare ``animation: none`` inside it. A weaker "string appears
    somewhere in file" check would pass even if the media query was
    deleted but the selectors remained, which would silently regress
    the contract."""
    import re as _re
    from pathlib import Path as _P
    page = _P(__file__).resolve().parent.parent / "marketing" / "index.html"
    text = page.read_text(encoding="utf-8")
    # Find the @media (prefers-reduced-motion: reduce) header, then
    # capture its body by walking brace depth. Regex .*? stops at the
    # FIRST inner ``}`` (each nested rule has one), which would mis-
    # parse a block with multiple sub-selectors — walking the braces
    # explicitly handles arbitrary nesting.
    header = _re.search(
        r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)\s*\{",
        text,
    )
    assert header, (
        "marketing page must honour OS-level Reduce Motion preference — "
        "no @media (prefers-reduced-motion: reduce) block found"
    )
    depth = 1
    i = header.end()
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    assert depth == 0, (
        "reduced-motion @media block has unbalanced braces"
    )
    block = text[header.end():i - 1]
    # Both animated selectors must explicitly turn animation off
    # inside the block; a static fallback alone isn't enough because
    # the per-row ``animation-delay: 880ms`` style attrs would keep
    # the elements invisible if we only set opacity.
    for selector in (".terminal-line", ".pulse-dot"):
        assert selector in block, (
            f"reduced-motion media query must reference {selector}"
        )
    # ``animation: none`` (with optional ``!important``) must appear
    # for the terminal-line + pulse-dot rules. We don't enforce ordering
    # — just that the override is in the block.
    assert _re.search(r"animation\s*:\s*none", block), (
        "reduced-motion block must neutralise animation via "
        "'animation: none' on the affected selectors"
    )
    # Static fallback: terminal lines must be opaque under reduced
    # motion (otherwise the staged animation-delay leaves them hidden).
    assert _re.search(r"opacity\s*:\s*1", block), (
        "reduced-motion block must set opacity: 1 on terminal-line so "
        "the per-line animation-delay doesn't leave them invisible"
    )


def test_tools_dropdown_has_aria_semantics() -> None:
    """Accessibility contract for the dashboard header's Tools menu
    (PR 103 review). The toggle button must declare ``aria-haspopup``,
    ``aria-controls``, and a synchronised ``aria-expanded`` so screen
    readers can announce the menu state. The handler in header.js
    flips ``aria-expanded`` in ``setToolsOpen``."""
    from pathlib import Path as _P
    page = _P(__file__).resolve().parent.parent / "web" / "static" / "header.js"
    text = page.read_text(encoding="utf-8")
    assert 'aria-haspopup="menu"' in text
    assert 'aria-controls="fi-tools-menu"' in text
    # The handler must toggle aria-expanded (start false, set true on open).
    assert 'aria-expanded' in text
    assert "setAttribute('aria-expanded'" in text
    # Escape returns focus to the toggle — required for keyboard navs.
    assert "fi-tools-toggle')?.focus()" in text


# ---------------------------------------------------------------------------
# Labels / fancy features
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


# ---------------------------------------------------------------------------
# Knowledge endpoint + dir-existence guards
# ---------------------------------------------------------------------------


def test_labels_refuses_when_quest_doesnt_exist(tmp_path: Path) -> None:
    """PUT /labels for a nonexistent quest must NOT auto-create
    <output_root>/<id>/.fi/ (that would let _scan_quests surface
    a phantom quest). Returns 404 when no .fi/ exists."""
    client = _client(tmp_path)
    res = client.put("/api/quests/never-existed/labels",
                     json={"labels": ["x"]})
    assert res.status_code == 404


def test_trash_refuses_when_quest_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving the quest dir while the subprocess is still writing
    into it causes engine I/O errors. Refuse with 409 + a hint to
    cancel first."""
    client = _client(tmp_path)
    output_root = client.app.state.output_root  # type: ignore[attr-defined]
    (output_root / "q-alive").mkdir()
    monkeypatch.setattr(
        client.app.state.launcher, "status_for",
        lambda qid: {"alive": True, "pid": 1, "started_at": 0, "age_seconds": 1},
    )
    res = client.delete("/api/quests/q-alive")
    assert res.status_code == 409
    assert "still running" in res.text.lower()


def test_knowledge_info_endpoint_returns_payload(tmp_path: Path) -> None:
    """/api/knowledge/info surfaces AxonStore location. When Axon
    isn't installed (the typical local dev path on CI), returns
    available=False with a clear reason."""
    client = _client(tmp_path)
    res = client.get("/api/knowledge/info")
    assert res.status_code == 200
    body = res.json()
    assert "available" in body
    if not body["available"]:
        assert "reason" in body


def test_md_lite_blocks_javascript_url() -> None:
    """Bot comment: md_lite.js had no scheme validation, so
    `[click](javascript:alert(1))` rendered as a clickable script.
    SAFE_SCHEME_RE now restricts to http/https/mailto/#/relative."""
    md_lite = (Path(__file__).resolve().parent.parent / "web" / "static" / "md_lite.js").read_text(encoding="utf-8")
    # The whitelist regex must exist + javascript: must NOT match it.
    assert "SAFE_SCHEME_RE" in md_lite
    assert "javascript" not in md_lite or "SAFE_SCHEME_RE" in md_lite
    # The renderer should fall back to esc(text) on rejected URLs.
    assert "safeUrl" in md_lite


# ---------------------------------------------------------------------------
# User-reported bugfixes: knowledge_info hard-import + tectonic idempotence
# ---------------------------------------------------------------------------


def test_knowledge_info_works_when_axon_is_available(tmp_path: Path) -> None:
    """The endpoint must work whether the Axon module imported
    successfully or not. Earlier version unconditionally imported
    _AXON_IMPORT_ERROR, which only exists on the failure branch."""
    client = _client(tmp_path)
    res = client.get("/api/knowledge/info")
    assert res.status_code == 200
    body = res.json()
    # Either Axon is up or it isn't — neither must 500.
    assert "available" in body


def test_tectonic_install_idempotent_when_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User reported: every browser session 'Install tectonic'
    spawns a new installer even though tectonic is already on disk.
    Fix: probe before spawning; return already_present=True without
    re-running."""
    client = _client(tmp_path)
    # Fake `shutil.which("tectonic")` returning a real path so the
    # endpoint sees tectonic as installed. The status endpoint
    # falls through to PATH when no repo-local binary exists.
    import shutil as _shutil
    monkeypatch.setattr(
        _shutil, "which",
        lambda name: "/fake/tectonic" if name == "tectonic" else None,
    )
    res = client.post("/api/system/install-tectonic")
    assert res.status_code == 200
    body = res.json()
    assert body["already_present"] is True
    assert body["spawned"] is False


def test_tectonic_status_endpoint_reports_not_installed(tmp_path: Path) -> None:
    """When no tectonic on disk + not on PATH, status is False so
    the UI shows the Install button enabled."""
    client = _client(tmp_path)
    # Default shutil.which won't find a binary named "tectonic" on
    # most CI images. If a runner DOES have tectonic, the test
    # gracefully accepts either outcome.
    res = client.get("/api/system/tectonic")
    assert res.status_code == 200
    body = res.json()
    assert "installed" in body


# ---------------------------------------------------------------------------
# /api/quests/{id}/file URL-cache-buster tolerance
# ---------------------------------------------------------------------------


def _make_quest_with_file(output_root: Path, quest_id: str, rel: str, body: str) -> Path:
    """Build a minimal valid quest under output_root and return its dir."""
    qd = output_root / quest_id
    (qd / ".fi").mkdir(parents=True)
    (qd / ".fi" / "state.sqlite").write_bytes(b"")
    target = qd / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return qd


def test_get_quest_file_strips_preventcache_suffix(tmp_path: Path) -> None:
    """VS Code Live Server (and similar) naively appends
    ?preventCache=<unix-ms> without checking for an existing query
    string, producing ?path=paper.md?preventCache=1779035013387.
    FastAPI reads the ``path`` value as the literal
    ``paper.md?preventCache=...``. The handler must strip the suffix
    so the file still resolves."""
    client = _client(tmp_path)
    output_root: Path = client.app.state.output_root  # type: ignore[attr-defined]
    _make_quest_with_file(output_root, "q-cache", "paper/paper.md", "real body")

    # Baseline: clean URL works.
    clean = client.get("/api/quests/q-cache/file", params={"path": "paper/paper.md"})
    assert clean.status_code == 200
    assert clean.text == "real body"

    # The pathological case the user reported: path value carries a
    # spurious ?preventCache= suffix (no encoding, just literal '?').
    dirty = client.get(
        "/api/quests/q-cache/file",
        params={"path": "paper/paper.md?preventCache=1779035013387"},
    )
    assert dirty.status_code == 200, (
        f"expected 200 with stripped suffix, got {dirty.status_code} {dirty.text}"
    )
    assert dirty.text == "real body"


def test_get_quest_file_strips_fragment_suffix(tmp_path: Path) -> None:
    """The same defensive strip should also handle a stray ``#anchor``
    appended to the path value — same class of misbehaving client."""
    client = _client(tmp_path)
    output_root: Path = client.app.state.output_root  # type: ignore[attr-defined]
    _make_quest_with_file(output_root, "q-frag", "paper/paper.md", "real body")

    res = client.get(
        "/api/quests/q-frag/file",
        params={"path": "paper/paper.md#anchor-from-some-link"},
    )
    assert res.status_code == 200
    assert res.text == "real body"


def test_get_quest_file_404_when_real_file_truly_missing(tmp_path: Path) -> None:
    """The strip mustn't accidentally make 404s disappear: a request
    for a file that doesn't exist still 404s, even with a cache-buster
    appended."""
    client = _client(tmp_path)
    output_root: Path = client.app.state.output_root  # type: ignore[attr-defined]
    _make_quest_with_file(output_root, "q-404", "paper/paper.md", "x")

    res = client.get(
        "/api/quests/q-404/file",
        params={"path": "paper/missing.md?preventCache=12345"},
    )
    assert res.status_code == 404


def test_get_quest_file_serves_real_filename(tmp_path: Path) -> None:
    """The download must carry the file's real name, not the URL's last
    segment. Without a Content-Disposition filename the browser saves the
    response as ``file`` (the route segment) with no extension — the bug
    the user hit. Default disposition is ``inline`` (preview in a tab);
    ``?download=1`` forces ``attachment`` (save). Both keep the real name."""
    client = _client(tmp_path)
    output_root: Path = client.app.state.output_root  # type: ignore[attr-defined]
    _make_quest_with_file(output_root, "q-dl", "paper/paper.md", "real body")

    # Default: inline preview, real filename present.
    res = client.get("/api/quests/q-dl/file", params={"path": "paper/paper.md"})
    assert res.status_code == 200
    cd = res.headers.get("content-disposition", "")
    assert "inline" in cd
    assert "paper.md" in cd  # NOT "file"

    # ?download=1: attachment (save), still the real filename.
    res2 = client.get(
        "/api/quests/q-dl/file",
        params={"path": "paper/paper.md", "download": "1"},
    )
    assert res2.status_code == 200
    cd2 = res2.headers.get("content-disposition", "")
    assert "attachment" in cd2
    assert "paper.md" in cd2


def test_wanted_papers_and_upload_endpoints(tmp_path: Path) -> None:
    """The papers panel endpoints: GET wanted-papers returns the manifest +
    dropped list; POST papers saves sanitised PDFs into inputs/papers/ and
    rejects path-traversal / wrong-extension uploads."""
    pytest.importorskip("multipart")  # FastAPI form/file routes need python-multipart
    import io
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    qid = "1782000000-test-quest-abc"
    needs = output_root / qid / "needs"
    needs.mkdir(parents=True)
    (needs / "WANTED_PAPERS.md").write_text(
        "# Papers to download\n\n## 1. SPIE overlay\n- **Get it:** https://doi.org/10.1117/x\n",
        encoding="utf-8",
    )
    client = TestClient(make_app(output_root))

    r = client.get(f"/api/quests/{qid}/wanted-papers")
    assert r.status_code == 200
    body = r.json()
    assert body["has_needs"] is True
    assert "SPIE overlay" in body["wanted_markdown"]
    assert body["dropped"] == []

    files = [
        ("files", ("my paper.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")),
        ("files", ("../evil.exe", io.BytesIO(b"x"), "application/octet-stream")),
    ]
    r2 = client.post(f"/api/quests/{qid}/papers", files=files)
    assert r2.status_code == 200
    assert r2.json()["saved"] == ["my_paper.pdf"]            # sanitised
    assert "../evil.exe" in r2.json()["skipped"]             # traversal / .exe rejected
    on_disk = sorted(p.name for p in (output_root / qid / "inputs" / "papers").iterdir())
    assert on_disk == ["my_paper.pdf"]
    assert client.get(f"/api/quests/{qid}/wanted-papers").json()["dropped"] == ["my_paper.pdf"]


def test_next_step_endpoint_surfaces_paused_quest(tmp_path: Path) -> None:
    """The unified Action-needed endpoint: NEXT_STEP.md present → waiting,
    with a headline parsed from its first heading; absent → not waiting."""
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    qid = "1782000001-test-quest-pause"
    quest = output_root / qid
    (quest / ".fi").mkdir(parents=True)
    client = TestClient(make_app(output_root))

    # No NEXT_STEP.md yet → not waiting.
    r0 = client.get(f"/api/quests/{qid}/next-step")
    assert r0.status_code == 200
    assert r0.json()["waiting"] is False

    # A SUPPLY pause wrote NEXT_STEP.md → waiting, headline parsed, interaction
    # classified from the verb the unified core stamped in.
    (quest / "NEXT_STEP.md").write_text(
        "# Action needed — download 2 paywalled paper(s)\n\n"
        "Quest **q** is paused and waiting for you (**SUPPLY**).\n\n"
        "## What to do\n1. Drop the PDFs into `inputs/papers/`.\n",
        encoding="utf-8",
    )
    r1 = client.get(f"/api/quests/{qid}/next-step")
    body = r1.json()
    assert body["waiting"] is True
    assert body["headline"] == "Action needed — download 2 paywalled paper(s)"
    assert body["interaction"] == "supply"
    assert "inputs/papers/" in body["markdown"]

    # The structured pause.json descriptor is authoritative for kind +
    # upload_targets (what the banner's inline upload posts to).
    import json as _json
    (quest / ".fi" / "pause.json").write_text(
        _json.dumps({"kind": "papers", "interaction": "supply",
                     "headline": "download 2 paywalled paper(s)",
                     "upload_targets": ["papers"]}),
        encoding="utf-8",
    )
    body2 = client.get(f"/api/quests/{qid}/next-step").json()
    assert body2["kind"] == "papers"
    assert body2["upload_targets"] == ["papers"]

    # An ANSWER pause (e.g. clarify/review answered via files + resume) is
    # classified from the NEXT_STEP.md verb even with no in-process registry.
    (quest / ".fi" / "pause.json").unlink()
    (quest / "NEXT_STEP.md").write_text(
        "# Action needed — confirm the research setup\n\n"
        "Quest **q** is paused and waiting for you (**ANSWER**).\n",
        encoding="utf-8",
    )
    assert client.get(f"/api/quests/{qid}/next-step").json()["interaction"] == "answer"


def test_paused_quest_is_needs_you_not_complete(tmp_path: Path) -> None:
    """A quest paused at human-review wrote a summary on its clean pause-exit,
    but the dashboard must show 'needs_you', not 'complete' — and the detail
    endpoint exposes the pending action + has_config."""
    import json as _json
    from web.server import _quest_pending
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    qid = "1782000010-test-quest-review"
    quest = output_root / qid
    (quest / ".fi").mkdir(parents=True)
    # Pause-exit wrote a summary AND a pending human-review snapshot (no answer).
    (quest / "frontier_insight_summary.json").write_text(
        _json.dumps({"provider": "openai"}), encoding="utf-8")
    (quest / ".fi" / "human_review.json").write_text(
        _json.dumps({"verdict": "revise", "score": 2}), encoding="utf-8")
    (quest / "config.yaml").write_text("topic: t\n", encoding="utf-8")

    assert _quest_pending(quest) == "review"
    client = TestClient(make_app(output_root))

    # Dashboard list: needs_you, not complete.
    rec = next(q for q in client.get("/api/quests").json()["quests"]
               if q["quest_id"] == qid)
    assert rec["verdict"] == "needs_you" and rec["pending"] == "review"

    # Detail endpoint: pending_action + has_config.
    detail = client.get(f"/api/quests/{qid}").json()
    assert detail["pending_action"] == "review"
    assert detail["has_config"] is True

    # Once answered, it's no longer pending → complete.
    (quest / ".fi" / "human_review_answer.json").write_text("{}", encoding="utf-8")
    assert _quest_pending(quest) is None
    rec2 = next(q for q in client.get("/api/quests").json()["quests"]
                if q["quest_id"] == qid)
    assert rec2["verdict"] == "complete"


def test_quest_pending_signals(tmp_path: Path) -> None:
    """_quest_pending recognises the unified pause markers + the clarify
    snapshot, and reports None when nothing is pending."""
    from web.server import _quest_pending
    import json as _json
    q = tmp_path / "q"
    (q / ".fi").mkdir(parents=True)
    assert _quest_pending(q) is None
    # pause.json wins and carries the kind.
    (q / ".fi" / "pause.json").write_text(_json.dumps({"kind": "supply"}), encoding="utf-8")
    assert _quest_pending(q) == "supply"
    (q / ".fi" / "pause.json").unlink()
    # NEXT_STEP.md → generic paused.
    (q / "NEXT_STEP.md").write_text("# Action needed", encoding="utf-8")
    assert _quest_pending(q) == "paused"
    (q / "NEXT_STEP.md").unlink()
    # clarify questions without an answer → clarify.
    (q / ".fi" / "clarify_questions.json").write_text("{}", encoding="utf-8")
    assert _quest_pending(q) == "clarify"
    (q / ".fi" / "clarify_answer.json").write_text("{}", encoding="utf-8")
    assert _quest_pending(q) is None


def test_clarify_disk_fallback_endpoints(tmp_path: Path) -> None:
    """A subprocess clarify pause has no in-process future, so the web reads the
    questions from .fi/clarify_questions.json and writes the answers to
    .fi/clarify_answer.json for the next --resume to consume."""
    import json as _json
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    qid = "1782000002-test-quest-clarify"
    fi = output_root / qid / ".fi"
    fi.mkdir(parents=True)
    client = TestClient(make_app(output_root))

    # No questions on disk → not pending.
    assert client.get(f"/api/quests/{qid}/clarify").json()["pending"] is False

    (fi / "clarify_questions.json").write_text(
        _json.dumps({"success_metric": {"prompt": "How to measure success?",
                                        "default": "accuracy"}}),
        encoding="utf-8",
    )
    r = client.get(f"/api/quests/{qid}/clarify").json()
    assert r["pending"] is True and r["source"] == "disk"
    assert "success_metric" in r["questions"]
    # get_quest also reports the pause from disk.
    assert client.get(f"/api/quests/{qid}").json()["pending_clarify"] is True

    # Submitting writes the answer file (no in-process future for a subprocess).
    p = client.post(f"/api/quests/{qid}/clarify",
                    json={"answers": {"success_metric": "F1"}})
    assert p.status_code == 200 and p.json()["in_process_resolved"] is False
    assert _json.loads((fi / "clarify_answer.json").read_text())["success_metric"] == "F1"
    # With the answer staged, the gate is no longer pending.
    assert client.get(f"/api/quests/{qid}/clarify").json()["pending"] is False


def test_generalized_upload_targets(tmp_path: Path) -> None:
    """POST /upload routes files to inputs/papers, inputs/data, or data/ by
    target, filtering by that target's extensions."""
    pytest.importorskip("multipart")
    import io
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    qid = "1782000003-test-quest-upload"
    (output_root / qid / ".fi").mkdir(parents=True)
    client = TestClient(make_app(output_root))

    # data → inputs/data, accepting .csv but not .pdf.
    files = [
        ("files", ("samples.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")),
        ("files", ("paper.pdf", io.BytesIO(b"%PDF"), "application/pdf")),
    ]
    r = client.post(f"/api/quests/{qid}/upload", data={"target": "data"}, files=files)
    assert r.status_code == 200
    assert r.json()["saved"] == ["samples.csv"]
    assert "paper.pdf" in r.json()["skipped"]
    assert (output_root / qid / "inputs" / "data" / "samples.csv").is_file()

    # root_data → data/ (the no-sim data pause).
    r2 = client.post(f"/api/quests/{qid}/upload", data={"target": "root_data"},
                     files=[("files", ("d.json", io.BytesIO(b"{}"), "application/json"))])
    assert r2.json()["saved"] == ["d.json"]
    assert (output_root / qid / "data" / "d.json").is_file()

    # Unknown target → 400.
    assert client.post(f"/api/quests/{qid}/upload", data={"target": "nope"},
                       files=[("files", ("x.csv", io.BytesIO(b"x"), "text/csv"))]).status_code == 400
