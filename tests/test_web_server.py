"""FastAPI server smoke tests.

Uses FastAPI's TestClient (no real network). The server reads quest
state from disk and routes clarify-resume via an in-process registry;
both are testable without spawning real Engine.run() background tasks.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.server import (
    _QuestRegistry, _scan_quests, _read_log_tail, _current_node_from_log,
    make_app,
)


# --- helpers ---------------------------------------------------------------


def _mk_quest_dir(root: Path, qid: str, *, log_body: str = "", summary: dict | None = None,
                  paper_md: str = "", figure_names: list[str] | None = None) -> Path:
    """Build a fake on-disk quest directory in the layout the server expects."""
    q = root / qid
    (q / ".fi").mkdir(parents=True, exist_ok=True)
    (q / "paper").mkdir(parents=True, exist_ok=True)
    (q / "figures").mkdir(parents=True, exist_ok=True)
    (q / ".fi" / "run.log").write_text(log_body, encoding="utf-8")
    if summary is not None:
        (q / "frontier_insight_summary.json").write_text(
            json.dumps(summary), encoding="utf-8",
        )
    if paper_md:
        (q / "paper" / "paper.md").write_text(paper_md, encoding="utf-8")
    for name in (figure_names or []):
        # Tiny valid PNG file (1x1 transparent pixel).
        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c6300010000000500017a5e6a780000000049454e44"
            "ae426082"
        )
        (q / "figures" / name).write_bytes(png_bytes)
    return q


# --- _scan_quests / _read_log_tail / _current_node_from_log ----------------


def test_scan_quests_returns_empty_for_missing_root(tmp_path: Path) -> None:
    assert _scan_quests(tmp_path / "does-not-exist") == []


def test_scan_quests_finds_quest_dirs_with_dot_fi(tmp_path: Path) -> None:
    _mk_quest_dir(tmp_path, "1700000000-foo-aaaa11", log_body="hello\n")
    _mk_quest_dir(
        tmp_path, "1700000100-bar-bbbb22",
        summary={"quest_id": "1700000100-bar-bbbb22", "provider": "openai"},
        paper_md="# title\n",
    )
    # Dirs without a `.fi/` subdir are skipped.
    (tmp_path / "not-a-quest").mkdir()
    (tmp_path / "not-a-quest" / "stuff.txt").write_text("noise", encoding="utf-8")

    quests = _scan_quests(tmp_path)
    ids = {q["quest_id"] for q in quests}
    assert ids == {"1700000000-foo-aaaa11", "1700000100-bar-bbbb22"}
    by_id = {q["quest_id"]: q for q in quests}
    assert by_id["1700000100-bar-bbbb22"]["verdict"] == "complete"
    assert by_id["1700000100-bar-bbbb22"]["provider"] == "openai"
    assert by_id["1700000100-bar-bbbb22"]["has_paper"] is True
    assert by_id["1700000000-foo-aaaa11"]["verdict"] == "(running)"


def test_read_log_tail_returns_recent_lines(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text("\n".join(f"line {i}" for i in range(50)) + "\n", encoding="utf-8")
    assert _read_log_tail(log, n=5) == ["line 45", "line 46", "line 47", "line 48", "line 49"]


def test_read_log_tail_handles_missing_file(tmp_path: Path) -> None:
    assert _read_log_tail(tmp_path / "nope.log", n=10) == []


def test_current_node_extracted_from_log_lines() -> None:
    lines = [
        "[1234-foo-aa] [ideate] topic=...",
        "[1234-foo-aa] [literature] retrieved 0 docs",
        "[1234-foo-aa] [execute] pip install ['matplotlib']",
    ]
    assert _current_node_from_log(lines) == "execute"
    # Empty log → unknown sentinel.
    assert _current_node_from_log([]) == "(unknown)"


# --- _QuestRegistry --------------------------------------------------------


@pytest.mark.asyncio
async def test_quest_registry_resolves_clarify_future() -> None:
    """The registry connects POST /clarify to the engine's clarify
    callback. We simulate the engine awaiting the future; the resolver
    sets the result and the future unblocks."""
    reg = _QuestRegistry()
    fut = reg.register_clarify("q1", {"comparative_baseline": {"q": "?", "default": "x"}})
    assert reg.pending_clarify("q1") is not None

    answers = {"comparative_baseline": "user-chosen"}
    ok = reg.resolve_clarify("q1", answers)
    assert ok is True
    assert reg.pending_clarify("q1") is None
    assert await fut == answers


def test_quest_registry_resolve_unknown_quest_returns_false() -> None:
    reg = _QuestRegistry()
    assert reg.resolve_clarify("does-not-exist", {"x": 1}) is False


# --- FastAPI app endpoint smoke -------------------------------------------


def test_app_list_quests_endpoint_returns_quest_dirs(tmp_path: Path) -> None:
    _mk_quest_dir(tmp_path, "q1", paper_md="x")
    _mk_quest_dir(tmp_path, "q2")
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/quests")
    assert r.status_code == 200
    body = r.json()
    ids = {q["quest_id"] for q in body["quests"]}
    assert ids == {"q1", "q2"}


def test_jobs_endpoint_discovers_cli_quest_via_run_log(tmp_path: Path) -> None:
    """A quest started directly via `python launch.py` writes .fi/run.log
    but no .fi/launch.log. It must still appear in the Jobs tab — the
    frontend has to be consistent regardless of how the quest was started."""
    # CLI-style quest: run.log only, no launch.log.
    _mk_quest_dir(
        tmp_path, "1700000000-cli-quest-aaaa11",
        log_body="[1700000000-cli-quest-aaaa11] [write] authoring paper.md\n",
    )
    app = make_app(tmp_path)
    client = TestClient(app)

    r = client.get("/api/jobs")
    assert r.status_code == 200
    jobs = {j["job_id"]: j for j in r.json()["jobs"]}
    assert "1700000000-cli-quest-aaaa11" in jobs
    # Recent run.log + no summary.json → inferred running (alive), even
    # though the launcher never spawned it.
    assert jobs["1700000000-cli-quest-aaaa11"]["alive"] is True

    # And its log is readable via the job-detail endpoint (run.log fallback).
    r2 = client.get("/api/jobs/1700000000-cli-quest-aaaa11")
    assert r2.status_code == 200
    assert any("authoring paper.md" in ln for ln in r2.json().get("log_tail", []))
    assert r2.json()["alive"] is True


def test_jobs_endpoint_finished_cli_quest_not_alive(tmp_path: Path) -> None:
    """A finished CLI quest (has a summary.json) shows up but NOT as alive."""
    _mk_quest_dir(
        tmp_path, "1700000000-done-bbbb22",
        log_body="[1700000000-done-bbbb22] [review] done\n",
        summary={"quest_id": "1700000000-done-bbbb22", "provider": "openai"},
    )
    app = make_app(tmp_path)
    client = TestClient(app)
    jobs = {j["job_id"]: j for j in client.get("/api/jobs").json()["jobs"]}
    assert jobs["1700000000-done-bbbb22"]["alive"] is False


def test_app_quest_detail_endpoint(tmp_path: Path) -> None:
    _mk_quest_dir(
        tmp_path, "q-detail",
        log_body="[q-detail] [ideate] topic=...\n[q-detail] [literature] retrieved 0 docs\n",
        paper_md="# Hello\n\nFake paper body.\n",
        figure_names=["result.png", "loss.png"],
        summary={"quest_id": "q-detail", "provider": "openai"},
    )
    app = make_app(tmp_path)
    client = TestClient(app)

    r = client.get("/api/quests/q-detail")
    assert r.status_code == 200
    body = r.json()
    assert body["quest_id"] == "q-detail"
    assert body["current_node"] == "literature"
    assert "Hello" in body["paper_preview"]
    assert set(body["figures"]) == {"result.png", "loss.png"}
    assert body["summary"]["provider"] == "openai"
    assert body["alive"] is False  # no real Engine task registered
    assert body["pending_clarify"] is False


def test_app_quest_detail_404_for_unknown_quest(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    client = TestClient(app)
    assert client.get("/api/quests/does-not-exist").status_code == 404


def test_app_log_endpoint_returns_tail(tmp_path: Path) -> None:
    _mk_quest_dir(tmp_path, "ql", log_body="alpha\nbeta\ngamma\n")
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/quests/ql/log?n=2")
    assert r.status_code == 200
    assert r.json() == {"lines": ["beta", "gamma"]}


def test_app_figure_endpoint_serves_png_and_rejects_path_traversal(tmp_path: Path) -> None:
    _mk_quest_dir(tmp_path, "qfig", figure_names=["plot.png"])
    app = make_app(tmp_path)
    client = TestClient(app)

    r = client.get("/api/quests/qfig/figure/plot.png")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")

    # Reaching the handler with `..` or `\` triggers the explicit
    # in-route guard (400). Encoded slashes are normalized to `/` by
    # the router and don't match the {name} segment at all (404) —
    # which is also defense in depth. Both are acceptable rejections.
    r2 = client.get("/api/quests/qfig/figure/sub%2Fdir.png")
    assert r2.status_code in {400, 404}
    r3 = client.get("/api/quests/qfig/figure/..%5Cwindows%5Csystem32")
    # %5C → \ inside the name segment, reaches handler, hits the guard.
    assert r3.status_code == 400


def test_app_jobs_endpoint_lists_quests_and_tool_jobs(tmp_path: Path) -> None:
    """``/api/jobs`` reads launch logs from three layouts:
      - per-quest:    ``<quest_id>/.fi/launch.log``
      - per-tool-job: ``_jobs/<job_id>/launch.log``
      - legacy flat:  ``_logs/<job_id>.log`` (preserved for old sessions)
    All three must surface so the dashboard's "Jobs" tab doesn't lose
    visibility into either the new layout or pre-migration files."""
    # New layout — per-quest.
    qd = tmp_path / "quest-aaa"
    (qd / ".fi").mkdir(parents=True)
    (qd / ".fi" / "launch.log").write_text("hello\n", encoding="utf-8")
    # New layout — per-tool-job.
    jd = tmp_path / "_jobs" / "tool-bbb"
    jd.mkdir(parents=True)
    (jd / "launch.log").write_text("bye\n", encoding="utf-8")
    # Legacy flat layout.
    ld = tmp_path / "_logs"
    ld.mkdir()
    (ld / "legacy-ccc.log").write_text("legacy\n", encoding="utf-8")

    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/jobs")
    assert r.status_code == 200
    ids = {j["job_id"] for j in r.json()["jobs"]}
    assert ids == {"quest-aaa", "tool-bbb", "legacy-ccc"}


def test_app_jobs_detail_finds_log_in_each_layout(tmp_path: Path) -> None:
    """``/api/jobs/{job_id}`` probes new layouts first, falls back to
    legacy. Verifies all three are reachable from the same endpoint."""
    qd = tmp_path / "q-new"
    (qd / ".fi").mkdir(parents=True)
    (qd / ".fi" / "launch.log").write_text("quest-log-body\n", encoding="utf-8")
    jd = tmp_path / "_jobs" / "j-new"
    jd.mkdir(parents=True)
    (jd / "launch.log").write_text("job-log-body\n", encoding="utf-8")
    (tmp_path / "_logs").mkdir()
    (tmp_path / "_logs" / "j-legacy.log").write_text("legacy-body\n", encoding="utf-8")

    app = make_app(tmp_path)
    client = TestClient(app)
    for jid, expected in (
        ("q-new", "quest-log-body"),
        ("j-new", "job-log-body"),
        ("j-legacy", "legacy-body"),
    ):
        r = client.get(f"/api/jobs/{jid}")
        assert r.status_code == 200, (jid, r.text)
        body = r.json()
        assert any(expected in line for line in body["log_tail"]), (jid, body)


def test_app_paper_endpoint_serves_markdown(tmp_path: Path) -> None:
    _mk_quest_dir(tmp_path, "qpaper", paper_md="# title\nbody\n")
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/quests/qpaper/paper")
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/markdown; charset=utf-8"


def test_app_paper_endpoint_404_when_paper_missing(tmp_path: Path) -> None:
    _mk_quest_dir(tmp_path, "qnopaper")
    app = make_app(tmp_path)
    client = TestClient(app)
    assert client.get("/api/quests/qnopaper/paper").status_code == 404


def test_app_clarify_get_returns_no_pending_when_nothing_registered(tmp_path: Path) -> None:
    _mk_quest_dir(tmp_path, "qc")
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/quests/qc/clarify")
    assert r.status_code == 200
    assert r.json() == {"pending": False, "questions": None}


def test_app_clarify_post_resolves_pending(tmp_path: Path) -> None:
    """Pre-populate the registry with a pending clarify future, then
    POST answers and assert the registry resolves it. Mirrors the live
    flow: Engine awaits → user submits in GUI → graph resumes."""
    app = make_app(tmp_path)
    reg = app.state.registry
    fut = reg.register_clarify("qc", {"x": {"q": "?", "default": "a"}})

    client = TestClient(app)
    r = client.post(
        "/api/quests/qc/clarify",
        json={"answers": {"x": "chosen"}},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    # The future has been set.
    assert fut.done()
    assert fut.result() == {"x": "chosen"}


def test_app_clarify_post_409_when_no_pending(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.post("/api/quests/q/clarify", json={"answers": {"a": "b"}})
    assert r.status_code == 409


def test_app_clarify_post_400_on_missing_answers_key(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    reg = app.state.registry
    reg.register_clarify("qc", {"x": {"q": "?", "default": "a"}})
    client = TestClient(app)
    r = client.post("/api/quests/qc/clarify", json={"wrong_key": {}})
    assert r.status_code == 400


def test_app_start_400_on_invalid_yaml(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.post("/api/quests/start", json={"yaml": "this is: not valid: yaml: ::"})
    assert r.status_code == 400
    r2 = client.post("/api/quests/start", json={})
    assert r2.status_code == 400


# --- human-review endpoints ----------------------------------------------


def test_human_review_get_returns_not_pending_when_nothing_registered(
    tmp_path: Path,
) -> None:
    _mk_quest_dir(tmp_path, "qhr")
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/quests/qhr/human-review")
    assert r.status_code == 200
    body = r.json()
    assert body["pending"] is False
    assert body["snapshot"] is None


def test_human_review_get_reads_disk_snapshot_when_no_in_process_future(
    tmp_path: Path,
) -> None:
    """Subprocess-launched quests don't register an in-process future;
    instead the engine wrote ``human_review.json`` to disk. The
    endpoint reads it so the dashboard can show the gate state."""
    qid = "qhrdisk"
    q = _mk_quest_dir(tmp_path, qid)
    snap = {"verdict": "revise", "score": 2, "iteration": 0,
            "must_flag_hits": ["[methodologist] circular_evaluation"]}
    (q / ".fi" / "human_review.json").write_text(
        json.dumps(snap), encoding="utf-8",
    )
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/quests/{qid}/human-review")
    assert r.status_code == 200
    body = r.json()
    assert body["pending"] is True
    assert body["source"] == "disk"
    assert body["snapshot"]["verdict"] == "revise"
    assert "circular_evaluation" in body["snapshot"]["must_flag_hits"][0]


def test_human_review_get_treats_resolved_answer_file_as_not_pending(
    tmp_path: Path,
) -> None:
    """When the user has already posted an answer (so
    ``human_review_answer.json`` exists alongside the snapshot), the
    UI should not show the banner again — the resume is queued."""
    qid = "qhrresolved"
    q = _mk_quest_dir(tmp_path, qid)
    (q / ".fi" / "human_review.json").write_text(
        json.dumps({"verdict": "accept"}), encoding="utf-8",
    )
    (q / ".fi" / "human_review_answer.json").write_text(
        json.dumps({"action": "accept", "feedback": ""}), encoding="utf-8",
    )
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/quests/{qid}/human-review")
    body = r.json()
    assert body["pending"] is False


def test_human_review_post_resolves_in_process_future(
    tmp_path: Path,
) -> None:
    """In-process spawn path: registry has a pending future; POST
    fills it; ``in_process_resolved`` is True."""
    qid = "qhrinproc"
    _mk_quest_dir(tmp_path, qid)
    app = make_app(tmp_path)
    reg = app.state.registry
    fut = reg.register_human_review(qid, {"verdict": "accept"})

    client = TestClient(app)
    r = client.post(
        f"/api/quests/{qid}/human-review",
        json={"action": "accept", "feedback": ""},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["in_process_resolved"] is True
    assert fut.done()
    assert fut.result() == {"action": "accept", "feedback": ""}


def test_human_review_post_writes_disk_answer_for_subprocess_resume(
    tmp_path: Path,
) -> None:
    """Subprocess path: no in-process future, but the POST should
    still land the answer on disk so a later ``--resume`` consumes
    it. ``in_process_resolved`` is False, request still succeeds."""
    qid = "qhrsub"
    q = _mk_quest_dir(tmp_path, qid)
    (q / ".fi" / "human_review.json").write_text(
        json.dumps({"verdict": "accept"}), encoding="utf-8",
    )
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.post(
        f"/api/quests/{qid}/human-review",
        json={"action": "refine", "feedback": "tighten methods"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["in_process_resolved"] is False
    answer_path = q / ".fi" / "human_review_answer.json"
    assert answer_path.is_file()
    data = json.loads(answer_path.read_text(encoding="utf-8"))
    assert data == {"action": "refine", "feedback": "tighten methods"}


def test_human_review_post_400_on_bad_action(tmp_path: Path) -> None:
    _mk_quest_dir(tmp_path, "qbad")
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.post(
        "/api/quests/qbad/human-review",
        json={"action": "delete-everything"},
    )
    assert r.status_code == 400


def test_human_review_post_400_on_refine_without_feedback(
    tmp_path: Path,
) -> None:
    _mk_quest_dir(tmp_path, "qref")
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.post(
        "/api/quests/qref/human-review",
        json={"action": "refine", "feedback": "   "},
    )
    assert r.status_code == 400


def test_app_index_html_loads(tmp_path: Path) -> None:
    """The root `/` returns the HTMX shell from web/static/index.html."""
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Frontier Insight" in r.text
    assert "quest-list" in r.text  # quest cards land here
    # The dashboard delegates per-quest detail (paper preview,
    # review panel rendering, log tail) to the /quest/<id> page.
    # The dashboard itself just shows a list + a "+ New Quest" CTA
    # so it stays simple. The renderReviewPanel function lives in
    # the quest detail page; the test there pins it.
    assert "New Quest" in r.text, "dashboard must surface the New Quest CTA"


def test_app_detail_endpoint_surfaces_review_panel_when_recorded(
    tmp_path: Path,
) -> None:
    """When the in-process registry has recorded a final_state
    with `review_panel`, the detail endpoint exposes both `review`
    (moderator output) and `review_panel` (per-persona). When no
    panel was recorded, both fields are null."""
    _mk_quest_dir(tmp_path, "q-panel", paper_md="# t\n")
    app = make_app(tmp_path)
    reg = app.state.registry
    reg.record_final_state("q-panel", {
        "review": {"verdict": "revise", "score": 3, "agreement": "split",
                   "weaknesses": ["w1"], "rationale": "stat persona objected"},
        "review_panel": [
            {"persona": "methodologist", "verdict": "accept", "score": 4,
             "strengths": ["clear design"], "weaknesses": [],
             "suggestions": [], "blocking": ""},
            {"persona": "statistician", "verdict": "revise", "score": 2,
             "strengths": [], "weaknesses": ["no CIs"],
             "suggestions": ["bootstrap"], "blocking": ""},
        ],
    })
    client = TestClient(app)
    r = client.get("/api/quests/q-panel")
    assert r.status_code == 200
    body = r.json()
    assert body["review"]["agreement"] == "split"
    assert len(body["review_panel"]) == 2
    assert body["review_panel"][0]["persona"] == "methodologist"

    # A quest with no recorded final_state has null fields (the GUI
    # falls back to the legacy single-reviewer render path).
    _mk_quest_dir(tmp_path, "q-no-panel")
    r = client.get("/api/quests/q-no-panel")
    body = r.json()
    assert body["review"] is None
    assert body["review_panel"] is None
