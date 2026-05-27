"""FastAPI status server for Frontier Insight.

Endpoints
---------
* ``GET  /``                              — vanilla-JS single-page app
  served from ``static/index.html``.
* ``GET  /api/quests``                    — every quest dir under
  the configured root, current node + verdict + age.
* ``GET  /api/quests/{id}``               — quest detail (state JSON,
  figures, paper preview).
* ``GET  /api/quests/{id}/log``           — log tail (most recent N lines).
* ``GET  /api/quests/{id}/log/stream``    — SSE log tail (live).
* ``GET  /api/quests/{id}/clarify``       — pending questions, if any.
* ``POST /api/quests/{id}/clarify``       — submit answers; resumes graph.
* ``POST /api/quests/start``              — start a new quest from a posted
  YAML body. Returns the new quest_id.

The server is intentionally state-light: each quest's state lives on
disk (``<quest_root>/.fi/{run.log, state.sqlite}`` + the
``frontier_insight_summary.json`` index). The server reads those for
status; for ``clarify`` resume it routes the answers into the live
``Engine.run()`` task via an in-process registry of clarify futures.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from core.config import Config
from core.engine import Engine
from core.provider import ProxySupervisor


# ---- per-process state ----------------------------------------------------


class _QuestRegistry:
    """In-process registry of live quest tasks and their pending
    clarify futures. Each quest run gets one entry from start until
    its task either finishes or errors. The clarify panel POSTs
    answers; the registry resolves the matching future so the
    Engine's clarify callback returns and the graph proceeds.

    Also tracks the in-memory `Engine` reference so the
    detail endpoint can surface the most-recent `review_panel`
    snapshot from `final_state` (when the quest has finished) or
    from the SqliteSaver checkpoint (mid-quest). For now we just
    snapshot the latest QuestArtifacts.raw_state when the driver
    task completes — that's what the GUI renders."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._clarify_questions: dict[str, dict[str, Any]] = {}
        self._clarify_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._configs: dict[str, Config] = {}
        self._final_states: dict[str, dict[str, Any]] = {}
        # Mirror of the clarify-future plumbing for the human-review
        # gate. When the in-process engine pauses at the human_feedback
        # node, the registered future is what the POST endpoint
        # resolves with the user's decision so the engine resumes.
        self._human_review_snapshots: dict[str, dict[str, Any]] = {}
        self._human_review_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def register_task(self, quest_id: str, task: asyncio.Task[Any], cfg: Config) -> None:
        self._tasks[quest_id] = task
        self._configs[quest_id] = cfg

    def record_final_state(self, quest_id: str, state: dict[str, Any]) -> None:
        self._final_states[quest_id] = state

    def final_state(self, quest_id: str) -> dict[str, Any] | None:
        return self._final_states.get(quest_id)

    def register_clarify(
        self, quest_id: str, questions: dict[str, Any],
    ) -> asyncio.Future[dict[str, Any]]:
        # Production callers always invoke this from within an async
        # handler, where `get_running_loop()` is the correct call
        # (`get_event_loop()` is deprecated in 3.12+ for non-running
        # fetches). Tests that call this synchronously fall through
        # to `new_event_loop()` so the registry still works without
        # asyncio.run() ceremony around every test fixture.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._clarify_questions[quest_id] = questions
        self._clarify_futures[quest_id] = fut
        return fut

    def pending_clarify(self, quest_id: str) -> dict[str, Any] | None:
        return self._clarify_questions.get(quest_id)

    def resolve_clarify(self, quest_id: str, answers: dict[str, Any]) -> bool:
        fut = self._clarify_futures.pop(quest_id, None)
        self._clarify_questions.pop(quest_id, None)
        if fut is None or fut.done():
            return False
        fut.set_result(answers)
        return True

    def register_human_review(
        self, quest_id: str, snapshot: dict[str, Any],
    ) -> asyncio.Future[dict[str, Any]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._human_review_snapshots[quest_id] = snapshot
        self._human_review_futures[quest_id] = fut
        return fut

    def pending_human_review(self, quest_id: str) -> dict[str, Any] | None:
        return self._human_review_snapshots.get(quest_id)

    def resolve_human_review(
        self, quest_id: str, answer: dict[str, Any],
    ) -> bool:
        fut = self._human_review_futures.pop(quest_id, None)
        self._human_review_snapshots.pop(quest_id, None)
        if fut is None or fut.done():
            return False
        fut.set_result(answer)
        return True

    def get_config(self, quest_id: str) -> Config | None:
        return self._configs.get(quest_id)

    def alive(self, quest_id: str) -> bool:
        task = self._tasks.get(quest_id)
        return task is not None and not task.done()


# ---- on-disk quest scan ---------------------------------------------------


def _scan_quests(output_root: Path) -> list[dict[str, Any]]:
    """Return one record per quest directory under ``output_root``. A
    quest dir is any subdirectory containing a ``.fi/`` folder."""
    if not output_root.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(output_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        fi_dir = d / ".fi"
        if not fi_dir.is_dir():
            continue
        summary_path = d / "frontier_insight_summary.json"
        verdict = "(running)"
        score: Any = None
        provider = ""
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                verdict = "complete"
                provider = data.get("provider", "")
            except Exception:
                pass
        paper_md = d / "paper" / "paper.md"
        out.append({
            "quest_id": d.name,
            "quest_root": str(d),
            "verdict": verdict,
            "score": score,
            "provider": provider,
            "has_paper": paper_md.exists(),
            "age_s": int(time.time() - fi_dir.stat().st_mtime),
        })
    return out


def _read_log_tail(log_path: Path, n: int = 200) -> list[str]:
    if not log_path.exists():
        return []
    # Read last ~64 KB and split. Cheap enough for run logs that rarely
    # exceed a few MB during a quest.
    size = log_path.stat().st_size
    with log_path.open("rb") as f:
        f.seek(max(0, size - 65536))
        tail = f.read().decode("utf-8", errors="replace")
    lines = tail.splitlines()
    return lines[-n:]


_QUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")


def _dir_size(path: Path) -> int:
    """Recursive byte count of a directory. Used by the trash listing
    so the user can see at a glance which trashed quests are large.
    Returns 0 on any I/O hiccup rather than raising."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _resolve_quest_root(output_root: Path, quest_id: str) -> Path:
    """Sanitize the user-supplied `quest_id` before composing a
    filesystem path with it. We:

    1. Reject anything outside the strict identifier alphabet
       (digits, letters, ``_``, ``-``, ``.``). This is enough to block
       path-separator-based traversal (``..``, ``/``, ``\\``).
    2. Resolve the composed path and verify it stays inside
       ``output_root.resolve()`` — defense in depth against unexpected
       OS-level path normalization quirks (symlinks, UNC names, etc.).

    Raises ``HTTPException(400)`` on either failure so the endpoint
    code can just call this and trust the result.
    """
    if not _QUEST_ID_RE.match(quest_id):
        raise HTTPException(400, f"bad quest_id format: {quest_id!r}")
    root = output_root.resolve()
    candidate = (root / quest_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(400, f"quest_id escapes output root: {quest_id!r}") from None
    return candidate


_NODE_TAG_RE = re.compile(r"\[([a-z_]+)\]")
# Every node in the DAG. Must match `_build_graph` in core/engine.py.
# Missing entries here cause the GUI's "current node" indicator to show
# `(unknown)` during the affected node's run, even though it's logging
# normally — `_current_node_from_log` filters on this set.
_KNOWN_NODES = frozenset({
    "clarify", "ideate", "literature", "design", "design_self_critique",
    "implement_outline", "implement", "execute", "execute_reflect",
    "analyze", "cross_check", "write", "review", "human_feedback",
    # No-simulation routing: triggered when ``simulatability == "no"``.
    "auto_collect_data", "wait_for_data", "data_load",
})


def _current_node_from_log(
    lines: list[str],
    *,
    known_nodes: frozenset[str] | None = None,
) -> str:
    """Best-effort: find the most recent ``[<node>]`` tag in the log.

    Match strict ``[lowercase_underscore]`` only, and prefer the LAST
    occurrence on a line that names a known node — otherwise random
    bracketed tokens like ``['matplotlib']`` from a pip-install log
    line would be mistaken for the current node.

    ``known_nodes`` lets callers inject a test fixture's node set; the
    default is the module-level ``_KNOWN_NODES`` so existing call
    sites keep their behaviour."""
    recognisable = known_nodes if known_nodes is not None else _KNOWN_NODES
    for line in reversed(lines):
        for m in reversed(list(_NODE_TAG_RE.finditer(line))):
            tag = m.group(1)
            if tag in recognisable:
                return tag
    return "(unknown)"


# Match the engine's ISO-with-comma-millis timestamp that ``_quest_logger``
# in ``core/engine.py`` produces. Example log line:
#     2026-05-20 11:11:52,388 [INFO] [ideate] topic=...
# Two capture groups: ISO date+time (whole seconds) and sub-second
# digits (millis or micros). The sub-second part is normalised to
# microseconds when parsing so timestamps don't lose precision.
_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})[,.](\d+)")


def _parse_log_timestamp(line: str) -> float | None:
    """Best-effort parse of the engine's log-line timestamp into a
    POSIX timestamp. Returns None for lines without a parseable
    timestamp (heartbeat-less lines, e.g. raw subprocess stdout).

    Preserves sub-second precision (millis / micros) so heartbeat
    spacing on the dashboard isn't quantised to whole seconds —
    relevant when the inactivity-watchdog ticks at 1 s cadence and
    the dashboard shows ``idle=0.6s`` distinct from ``idle=1.4s``.
    """
    m = _LOG_TS_RE.match(line)
    if not m:
        return None
    try:
        from datetime import datetime
        ts_str = m.group(1).replace("T", " ")
        # ``datetime.fromisoformat`` understands the space-separated form
        # natively on Py3.11+. Treat the parsed time as local — the engine
        # logger uses ``logging.Formatter`` with no tz, so this matches.
        dt = datetime.fromisoformat(ts_str)
        base = dt.timestamp()
        # Normalise the fractional part to a fraction of a second.
        # Engine logger emits 3-digit ms by default; some sites emit 6.
        # Pad/truncate to 6 digits so the divisor is fixed at 1_000_000.
        frac_digits = m.group(2)[:6].ljust(6, "0")
        try:
            frac = int(frac_digits) / 1_000_000.0
        except ValueError:
            frac = 0.0
        return base + frac
    except (ValueError, TypeError):
        return None


def _node_progress_from_log(
    lines: list[str], known_nodes: frozenset[str], *, now: float | None = None,
) -> dict[str, float | str | None]:
    """Derive node start/elapsed/idle from a tail of run.log lines.

    Returns ``{"node_started_at", "node_elapsed_s", "node_idle_s"}``.
    ``node_started_at`` is the POSIX timestamp of the FIRST log line
    that tagged the current node (the most-recent ``[<node>]`` open).
    ``node_elapsed_s`` is "now − node_started_at". ``node_idle_s`` is
    "now − timestamp of the most recent log line" — caps at the elapsed
    time so a quest with no recent activity reports an idle <= elapsed.

    Used by the ``/api/quests/{id}`` endpoint to power a stuck-quest
    badge ("running 4 min" green → "idle 5 min" yellow → "idle 10 min"
    red). All three values are ``None`` when there's nothing parseable
    — caller renders a fall-back without elapsed/idle info.

    ``known_nodes`` is the recogniser set for ``_current_node_from_log``;
    pass the same frozenset the caller uses elsewhere so the two
    helpers stay in sync (avoids the bug where this function silently
    fell back to the module-level ``_KNOWN_NODES`` and ignored the
    caller's choice).
    """
    current = _current_node_from_log(lines, known_nodes=known_nodes)
    if current == "(unknown)":
        return {
            "node_started_at": None,
            "node_elapsed_s": None,
            "node_idle_s": None,
        }
    if now is None:
        now = time.time()
    # Walk forward from the start; the FIRST line that tags ``current``
    # is the node's start. Subsequent ``[<node>]`` mentions of the same
    # tag don't reset the start (one node can log many lines). A LATER
    # tag for a different node would mean we exited current — but then
    # ``_current_node_from_log`` would have returned that other node,
    # not ``current``. So walking forward and stopping at first match
    # is correct.
    started_at: float | None = None
    last_activity: float | None = None
    for line in lines:
        ts = _parse_log_timestamp(line)
        if ts is None:
            continue
        last_activity = ts  # any timestamped line counts as activity
        if started_at is None:
            for m in _NODE_TAG_RE.finditer(line):
                if m.group(1) == current:
                    started_at = ts
                    break
    if started_at is None:
        return {
            "node_started_at": None,
            "node_elapsed_s": None,
            "node_idle_s": None,
        }
    elapsed = max(0.0, now - started_at)
    idle = (
        max(0.0, now - last_activity) if last_activity is not None
        else elapsed
    )
    # Cap idle at elapsed: it doesn't make sense to be "idle 5 min"
    # in a node that only started 30 s ago.
    idle = min(idle, elapsed)
    return {
        "node_started_at": started_at,
        "node_elapsed_s": elapsed,
        "node_idle_s": idle,
    }


def _read_quest_failed_md(quest_root: Path) -> dict[str, Any] | None:
    """When the engine writes ``<quest_root>/quest_failed.md`` after a
    mid-graph crash, surface its summary to the dashboard so the user
    discovers the failure without grepping the file. Returns None when
    the file is absent. Returns a dict with ``present``, ``path``,
    ``failing_node``, and ``what_broke`` (one-line exception text)
    — fields parsed lenient-best-effort from the markdown template the
    engine writes (see ``Engine._write_quest_failed_md``). The full
    traceback + resume hint stays in the markdown body; the dashboard
    intentionally surfaces just enough to direct the user to the file.
    """
    path = quest_root / "quest_failed.md"
    if not path.is_file():
        return None
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    info: dict[str, Any] = {
        "present": True,
        "path": str(path),
        "failing_node": None,
        "what_broke": None,
    }
    # Templated lines in the engine's writer:
    #   **Failing node:** `<node>`
    #   ```\n<exception line>\n```
    for line in body.splitlines():
        if line.startswith("**Failing node:**"):
            info["failing_node"] = (
                line.split("**Failing node:**", 1)[1].strip().strip("`")
            )
            break
    # First fenced line under "## What broke" carries the exception.
    in_what_broke = False
    in_fence = False
    for line in body.splitlines():
        if line.startswith("## What broke"):
            in_what_broke = True
            continue
        if in_what_broke and line.strip().startswith("```"):
            if in_fence:
                break
            in_fence = True
            continue
        if in_fence:
            info["what_broke"] = line.strip()
            break
    return info


# ---- app factory ----------------------------------------------------------


def make_app(
    output_root: Path,
    *,
    max_concurrent: int = 4,
    vscode_bridge_port: int = 0,
    vscode_bridge_socket: str = "",
) -> FastAPI:
    """Build the FastAPI app. Pass the on-disk root where quest dirs
    will live; the server scans this for status and writes new quests
    here when ``POST /api/quests/start`` is called.

    ``max_concurrent`` bounds the number of in-flight quests spawned
    via the web UI interview. ``vscode_bridge_port`` / ``vscode_bridge_socket``
    are forwarded to each child quest so LLM calls keep routing
    through the same bridge the dashboard inherits from its parent.
    The socket path is the preferred transport (per-user IPC, no port
    conflict on shared hosts); the port is kept for backward compat
    with users still passing ``--vscode-bridge-port``."""
    app = FastAPI(title="Frontier Insight", version="1.0")
    registry = _QuestRegistry()
    app.state.registry = registry
    app.state.output_root = output_root
    app.state.supervisor = ProxySupervisor()
    # Stash both bridge transports on app.state so request handlers
    # can guard against a user picking ``vscode_extension`` when
    # neither is available, and so /api/*/schema endpoints can probe
    # the right address.
    app.state.vscode_bridge_port = vscode_bridge_port
    app.state.vscode_bridge_socket = vscode_bridge_socket

    # Subprocess launcher for quests spawned from the web UI.
    # The launcher lives on the app so request handlers can share it.
    from web.quest_launcher import QuestLauncher
    repo_root = Path(__file__).resolve().parent.parent
    app.state.launcher = QuestLauncher(
        repo_root=repo_root,
        max_concurrent=max_concurrent,
        vscode_bridge_port=vscode_bridge_port,
        vscode_bridge_socket=vscode_bridge_socket,
        output_root=output_root,
    )

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Interview routes — /interview (new quest), /update/<id> (mid-quest
    # re-entry), JSON schema endpoint, and POST handlers. Backed by
    # core.interview as the single source of truth.
    from web.interview_routes import register_interview_routes
    register_interview_routes(app, output_root)

    # Tools routes — /tools/<name> (form pages for proposal, critique,
    # digest, portfolio, summarize, analyze, fleet, ingest), backed
    # by web/tools_routes.py:TOOL_SPECS as the source of truth.
    from web.tools_routes import register_tools_routes
    register_tools_routes(app, output_root)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_html = static_dir / "index.html"
        if not index_html.exists():
            return HTMLResponse(
                "<h1>Frontier Insight</h1><p>UI not installed.</p>", status_code=500,
            )
        return HTMLResponse(index_html.read_text(encoding="utf-8"))

    @app.get("/trash", response_class=HTMLResponse)
    async def trash_page() -> HTMLResponse:
        page = static_dir / "trash.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Trash UI not installed</h1>", status_code=500,
            )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page() -> HTMLResponse:
        page = static_dir / "settings.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Settings UI not installed</h1>", status_code=500,
            )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    # No /about route — the landing page lives separately under
    # marketing/index.html and is meant for external deployment
    # (GitHub Pages etc.), not inside the operational --serve UI.

    @app.get("/jobs", response_class=HTMLResponse)
    async def jobs_page() -> HTMLResponse:
        """Live job-tracking page. Shows running + recent tool / system /
        quest subprocess jobs the launcher knows about, with a live
        log-tail view. Hit by the tools form after submit so the user can
        watch their proposal/critique/etc. actually run."""
        page = static_dir / "jobs.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Jobs UI not installed</h1>", status_code=500,
            )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail_page(job_id: str) -> HTMLResponse:
        """Single-job detail page. Reuses jobs.html and lets the client
        JS pick up the job_id from window.__fi_job_id."""
        if not _QUEST_ID_RE.match(job_id):
            raise HTTPException(400, f"bad job_id format: {job_id!r}")
        page = static_dir / "jobs.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Jobs UI not installed</h1>", status_code=500,
            )
        # Same trick as /quest/{id}: inject the job_id into a global so
        # the client can target this single job.
        html = page.read_text(encoding="utf-8")
        injected = html.replace(
            "</head>",
            f'<script>window.__fi_job_id = {json.dumps(job_id)};</script></head>',
            1,
        )
        return HTMLResponse(injected)

    @app.get("/compare", response_class=HTMLResponse)
    async def compare_page() -> HTMLResponse:
        page = static_dir / "compare.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Compare UI not installed</h1>", status_code=500,
            )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/axon/status")
    async def axon_status_endpoint() -> JSONResponse:
        """Health-probe the Axon sidecar (``python -m axon.api`` on
        ``127.0.0.1:8000`` by default). The /settings page polls
        this so the user can see the sidecar is hot before kicking
        off a quest. Cheap — one HTTP HEAD-equivalent probe."""
        from core.axon_sidecar import axon_status as _probe
        return JSONResponse(dict(_probe()))

    @app.post("/api/axon/launch")
    async def axon_launch_endpoint() -> JSONResponse:
        """Idempotent: spawn an Axon API sidecar if one isn't already
        listening. Used by the Settings page's 'Start Axon' button
        for users who launched FI with ``--no-axon-sidecar`` and
        changed their mind later."""
        from core.axon_sidecar import ensure_axon_up
        return JSONResponse(dict(ensure_axon_up()))

    @app.get("/api/knowledge/info")
    async def knowledge_info() -> JSONResponse:
        """Surface the AxonStore knowledge-base location + corpus
        stats on the Settings page.

        ``core.knowledge`` exports ``_AXON_AVAILABLE`` always, but
        ``_AXON_IMPORT_ERROR`` only exists when the import FAILED
        (assigned inside the ``except`` branch). My earlier draft
        imported it unconditionally, which raised ImportError on
        the success path and masked the real availability state.
        We now use ``getattr(..., None)`` to read the optional
        symbol so success + failure paths both report cleanly."""
        try:
            from core import knowledge as _knowledge_mod
        except Exception as e:
            return JSONResponse({
                "available": False,
                "reason": f"knowledge module unavailable: {e!r}",
            })
        axon_available = getattr(_knowledge_mod, "_AXON_AVAILABLE", False)
        axon_import_error = getattr(_knowledge_mod, "_AXON_IMPORT_ERROR", None)
        AxonBrain = getattr(_knowledge_mod, "AxonBrain", None)
        AxonConfig = getattr(_knowledge_mod, "AxonConfig", None)

        if not axon_available or AxonBrain is None or AxonConfig is None:
            reason = (
                f"axon package not importable: {axon_import_error!r}. "
                if axon_import_error is not None
                else "axon package not installed. "
            )
            return JSONResponse({
                "available": False,
                "reason": reason + (
                    "Install with `pip install axon` (or pull the "
                    "axon repo into PYTHONPATH)."
                ),
            })

        info: dict[str, Any] = {"available": True}
        try:
            ac = AxonConfig()  # default
            # AxonConfig isn't a pydantic BaseModel (no model_dump);
            # enumerate plain attributes. REDACT anything that looks
            # like a secret — the real AxonConfig holds api_key,
            # brave_api_key, etc. in plaintext. Returning them to the
            # browser would leak into network logs + DevTools history.
            secret_re = re.compile(r"(_?(api_)?(key|token|secret|password))$", re.I)
            cfg_dump: dict[str, Any] = {}
            for attr in sorted(dir(ac)):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(ac, attr)
                except Exception:
                    continue
                if callable(val):
                    continue
                # Coerce paths + simple types only; skip nested objects.
                if not isinstance(val, (str, int, float, bool, list, type(None))):
                    val = str(val)
                if secret_re.search(attr) and isinstance(val, str) and val:
                    val = f"<redacted len={len(val)}>"
                cfg_dump[attr] = val
            info["axon_config"] = cfg_dump
            # The actual store base — Axon's directory layout is
            # ``<store_base>/AxonStore/<user>/<project>/`` and the
            # bm25 index lives inside that. Surface both.
            base = getattr(ac, "axon_store_base", None)
            bm25 = getattr(ac, "bm25_path", None)
            if base:
                info["store_path"] = str(base)
            if bm25:
                info["bm25_path"] = str(bm25)
        except Exception as e:
            info["config_error"] = repr(e)

        # Surface the FI project name so the Settings page can
        # explicitly show "we're operating in the FrontierInsight
        # project, not Axon's default."
        info["project"] = getattr(_knowledge_mod, "FI_AXON_PROJECT", "default")

        # Document inventory via the real `list_documents` API.
        # AxonBrain returns one entry per PARENT doc (grouped by
        # source) with a chunk count — the `kind` field FI's engine
        # writes lives inside the per-chunk metadata, not at the
        # parent level, so a "count by kind" view requires drilling
        # into individual chunks. For the dashboard we just surface
        # the source × chunks breakdown — which is what the user
        # actually wants to see ("did my re-ingest land?").
        try:
            brain = AxonBrain(AxonConfig())
            # Match the engine's project setup so the counts we
            # report are the FI project's, not Axon's default.
            # ``ensure_project`` creates the project if it doesn't
            # exist yet; then ``switch_project`` activates it.
            try:
                from axon.projects import ensure_project as _ensure_project
                _ensure_project(
                    info["project"],
                    description="Frontier Insight corpus",
                )
                brain.switch_project(info["project"])
            except Exception as e:
                info["project_error"] = (
                    f"could not switch to project {info['project']!r}: {e!r}"
                )
            try:
                docs = brain.list_documents()
            except Exception as e:
                info["counts_error"] = (
                    f"list_documents() failed: {e!r}."
                )
                docs = []
            by_source: dict[str, int] = {}
            total_chunks = 0
            total_docs = 0
            for d in docs or []:
                total_docs += 1
                if not isinstance(d, dict):
                    continue
                src = str(d.get("source") or "<unknown>")
                chunks = int(d.get("chunks") or 0)
                by_source[src] = by_source.get(src, 0) + chunks
                total_chunks += chunks
            info["doc_counts_by_source"] = by_source
            info["total_documents"] = total_docs
            info["total_chunks"] = total_chunks
            info["counts_note"] = (
                "Each parent doc carries multiple text chunks. "
                "Counts above are CHUNKS grouped by source. Engine "
                "write-back uses source='fi_quest_paper' etc.; "
                "/api/knowledge/reingest tags new docs as "
                "source='web-reingest'."
            )
        except Exception as e:
            info["counts_error"] = repr(e)
        return JSONResponse(info)

    @app.post("/api/knowledge/reingest")
    async def reingest_quests() -> JSONResponse:
        """Walk outputs/ and feed each quest's paper.md back into
        Axon as ``kind=fi_quest_paper``. Useful when the user ran
        many quests with ``knowledge.enabled: false`` (the default
        for the interview-generated YAML) and now wants the corpus
        retroactively populated. Returns per-quest success/failure
        so the UI can show what landed."""
        try:
            from core import knowledge as _knowledge_mod
            from core.config import KnowledgeConfig
        except Exception as e:
            raise HTTPException(500, f"knowledge module unavailable: {e!r}")
        if not getattr(_knowledge_mod, "_AXON_AVAILABLE", False):
            raise HTTPException(
                503,
                "Axon package not installed — can't ingest. "
                "Run `pip install axon` (or pull the axon repo into "
                "PYTHONPATH) first.",
            )
        from core.knowledge import Knowledge
        # Build a minimal Knowledge instance with enabled=True and the
        # default AxonConfig — same path the engine uses for the
        # post-quest write-back.
        kn = Knowledge(KnowledgeConfig(enabled=True))
        if not kn.enabled:
            raise HTTPException(503, "Axon brain failed to initialize")
        results: list[dict[str, Any]] = []
        for d in sorted(app.state.output_root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            paper_md = d / "paper" / "paper.md"
            if not paper_md.is_file():
                results.append({"quest_id": d.name, "skipped": "no paper.md"})
                continue
            try:
                text = paper_md.read_text(encoding="utf-8")
                ok = kn.add_text(
                    kind="fi_quest_paper",
                    text=text,
                    metadata={"quest_id": d.name, "source": "web-reingest"},
                )
                results.append({"quest_id": d.name, "ok": ok})
            except Exception as e:
                results.append({"quest_id": d.name, "error": repr(e)})
        return JSONResponse({
            "total": len(results),
            "ingested": sum(1 for r in results if r.get("ok")),
            "results": results,
        })

    @app.get("/api/providers/availability")
    async def provider_availability() -> JSONResponse:
        """Probe which providers have working auth on this host so
        the Settings page can show ✓/⚠ next to each. Reuses the same
        helper the CLI interview's smart-default uses."""
        from core.interview import (
            available_providers, PROVIDER_CHOICES,
        )
        avail = set(available_providers())
        return JSONResponse({
            "providers": [
                {
                    "name": c.value,
                    "label": c.label,
                    "description": c.description,
                    "available": c.value in avail,
                }
                for c in PROVIDER_CHOICES
            ],
        })

    @app.get("/quest/{quest_id}", response_class=HTMLResponse)
    async def quest_detail(quest_id: str) -> HTMLResponse:
        """Live quest-status page. Reuses the static quest.html and
        injects the quest_id as a window-level constant so the JS
        knows what to poll. Validates quest_id against the same
        allowlist used elsewhere to block path traversal."""
        if not _QUEST_ID_RE.match(quest_id):
            raise HTTPException(400, f"bad quest_id format: {quest_id!r}")
        page = static_dir / "quest.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Quest UI not installed</h1>", status_code=500,
            )
        html = page.read_text(encoding="utf-8")
        injected = html.replace(
            "</head>",
            f'<script>window.__fi_quest_id = {json.dumps(quest_id)};</script></head>',
            1,
        )
        return HTMLResponse(injected)

    @app.post("/api/quests/{quest_id}/resume")
    async def resume_quest(quest_id: str) -> JSONResponse:
        """Spawn ``python launch.py --config <quest_root>/config.yaml
        --resume <id>`` as a subprocess via the launcher. Different
        from ``--update``: no interview; just continue from the last
        checkpoint with the existing YAML unchanged."""
        if not _QUEST_ID_RE.match(quest_id):
            raise HTTPException(400, f"bad quest_id format: {quest_id!r}")
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        yaml_path = quest_root / "config.yaml"
        if not yaml_path.is_file():
            raise HTTPException(
                400, f"no config.yaml at {yaml_path}. "
                "Resume requires the YAML that was used to start "
                "the quest. CLI-started quests may not have one.",
            )
        sqlite_path = quest_root / ".fi" / "state.sqlite"
        if not sqlite_path.is_file():
            raise HTTPException(
                400, f"no checkpoint at {sqlite_path}. The quest "
                "directory must contain a .fi/state.sqlite from a "
                "prior run before --resume can pick it up.",
            )
        from web.quest_launcher import QuestLauncherFull
        try:
            launched = app.state.launcher.launch_command(
                argv_tail=["--config", str(yaml_path), "--resume", quest_id],
                job_id=quest_id,  # reuse the quest_id so /quest/<id> tracks it
            )
        except QuestLauncherFull as e:
            return JSONResponse(
                {"error": "launcher at capacity", "detail": str(e),
                 "retry_after_seconds": 30},
                status_code=503,
                headers={"Retry-After": "30"},
            )
        return JSONResponse({
            "quest_id": quest_id,
            "pid": launched.pid,
            "resumed": True,
        })

    @app.get("/api/system/tectonic")
    async def tectonic_status(job_id: str | None = None) -> JSONResponse:
        """Report whether tectonic is already installed PLUS surface
        the most recent install attempt's exit state + log tail when
        a ``job_id`` is passed. Without the log path, a failing
        install (network error, GitHub asset 404, etc.) shows
        "starting…" forever in the UI. With it, the Settings page
        can poll the job and surface the actual error."""
        import sys as _sys
        import shutil as _shutil
        repo_root = Path(__file__).resolve().parent.parent
        local_bin = repo_root / "tools" / (
            "tectonic.exe" if _sys.platform == "win32" else "tectonic"
        )
        path_bin = _shutil.which("tectonic")
        installed = local_bin.is_file() or bool(path_bin)
        result: dict[str, Any] = {"installed": installed}
        if installed:
            result["location"] = str(local_bin) if local_bin.is_file() else path_bin
            result["source"] = "repo-local" if local_bin.is_file() else "PATH"
        # If the caller passed a job_id, surface that job's state +
        # log tail. The launcher keeps the entry after the
        # subprocess exits until the next launch reaps the slot.
        if job_id:
            status = app.state.launcher.status_for(job_id) or {}
            result["job_status"] = status
        return JSONResponse(result)

    @app.get("/api/system/job-log/{job_id}")
    async def job_log(job_id: str, n: int = 200) -> JSONResponse:
        """Read the captured stdout+stderr log of a launcher job.
        Surfaces silent failures that previously vanished into
        DEVNULL. ``n`` caps the tail length."""
        lines = app.state.launcher.get_log_tail(job_id, n=n)
        if lines is None:
            raise HTTPException(404, f"no log for job {job_id}")
        return JSONResponse({"job_id": job_id, "lines": lines})

    def _classify_job(job_id: str) -> str:
        """Categorize a launcher job by inspecting its job_id prefix.
        Used by the /api/jobs response so the UI can render different
        icons / link targets per job kind:
          tool   → /api/tools/<name> jobs (job_id starts with `<tool>-`)
          system → install-tectonic etc.
          quest  → CLI/--new/--update launches (timestamp-prefixed)
        """
        from web.tools_routes import TOOLS_BY_NAME
        for name in TOOLS_BY_NAME:
            if job_id.startswith(f"{name}-"):
                return "tool"
        if job_id.startswith("tectonic-"):
            return "system"
        return "quest"

    @app.get("/api/jobs")
    async def list_jobs() -> JSONResponse:
        """List all subprocess jobs the launcher has spawned (tools,
        --install-tectonic, --new/--update/--proposal quest launches),
        merging:
          (1) the launcher's live registry (alive jobs with pid/age)
          (2) on-disk launch.log files (everything ever logged —
              alive *or* exited) from:
                - ``<output_root>/<quest_id>/.fi/launch.log`` (quests)
                - ``<output_root>/_jobs/<job_id>/launch.log`` (tool jobs)
                - ``<output_root>/_logs/<job_id>.log`` (legacy flat
                  layout, kept readable so jobs from older sessions
                  still show up in the Jobs tab)
        Each entry the dashboard's "Jobs" tab consumes:
          {job_id, kind, alive, exit_code?, age_seconds, log_mtime}.
        """
        out_root = app.state.output_root
        live = {q.quest_id: q for q in app.state.launcher.list_alive()}
        seen: dict[str, dict[str, Any]] = {}

        # Collect (job_id, log_path, mtime, size) from each of the
        # three layouts. ``stat()`` is called ONCE per candidate here
        # (so the subsequent sort + dict build don't re-stat — 3x
        # stat calls per file in the old version was an unnecessary
        # cost on output roots with many quests). Files that vanish
        # between ``iterdir()`` and ``stat()`` (race with a deletion
        # / a cancelled-quest cleanup) are dropped silently rather
        # than 500ing the endpoint.
        def _stat_or_none(p: Path) -> tuple[float, int] | None:
            try:
                st = p.stat()
                return st.st_mtime, st.st_size
            except (FileNotFoundError, PermissionError):
                return None

        candidates: list[tuple[str, Path, float, int]] = []

        def _add(jid: str, lp: Path) -> None:
            stat = _stat_or_none(lp)
            if stat is not None:
                candidates.append((jid, lp, stat[0], stat[1]))

        # 1a. Per-quest launch logs at <quest>/.fi/launch.log.
        if out_root.is_dir():
            for quest_dir in out_root.iterdir():
                if not quest_dir.is_dir() or quest_dir.name.startswith("_"):
                    continue
                lp = quest_dir / ".fi" / "launch.log"
                if lp.is_file():
                    _add(quest_dir.name, lp)
        # 1b. Per-tool-job logs at _jobs/<job_id>/launch.log.
        jobs_root = out_root / "_jobs"
        if jobs_root.is_dir():
            for job_dir in jobs_root.iterdir():
                if not job_dir.is_dir():
                    continue
                lp = job_dir / "launch.log"
                if lp.is_file():
                    _add(job_dir.name, lp)
        # 1c. Legacy flat _logs/<id>.log — preserved so older sessions
        # remain visible in the dashboard until the user cleans them up.
        legacy_logs_dir = out_root / "_logs"
        if legacy_logs_dir.is_dir():
            for p in legacy_logs_dir.glob("*.log"):
                _add(p.stem, p)

        # Most-recent-first, capped at 200 to keep the response small.
        candidates.sort(key=lambda row: row[2], reverse=True)
        for jid, _lp, mtime, size in candidates[:200]:
            if jid in seen:
                # If both new and legacy layouts have an entry for the
                # same id (unlikely but possible during migration),
                # the newer-first sort already picked the freshest.
                continue
            seen[jid] = {
                "job_id": jid,
                "kind": _classify_job(jid),
                "alive": jid in live,
                "log_mtime": mtime,
                "log_size": size,
            }
        # 2. Live registry entries (may include jobs that haven't
        #    written a log file yet — rare but possible during startup).
        for jid, entry in live.items():
            if jid not in seen:
                seen[jid] = {
                    "job_id": jid,
                    "kind": _classify_job(jid),
                    "alive": True,
                    "log_mtime": entry.started_at,
                    "log_size": 0,
                }
            seen[jid]["pid"] = entry.pid
            seen[jid]["age_seconds"] = entry.age_seconds()
        # 3. For exited jobs, pull exit_code if the launcher still
        #    remembers them (within the recent-reap window).
        for jid, info in seen.items():
            if not info["alive"]:
                st = app.state.launcher.status_for(jid)
                if st and st.get("exit_code") is not None:
                    info["exit_code"] = st["exit_code"]
        return JSONResponse({
            "jobs": sorted(
                seen.values(),
                key=lambda j: j.get("age_seconds", 0) if j["alive"] else -j.get("log_mtime", 0),
                reverse=False,
            ),
        })

    @app.get("/api/drafts")
    async def list_drafts() -> JSONResponse:
        """List draft quest YAMLs the proposal tool dropped in
        ``outputs/_drafts/``. Each entry includes the parsed topic
        + title so the /interview UI can show a one-click picker
        for "continue this proposal as a new quest." Most-recent
        first."""
        drafts_dir = app.state.output_root / "_drafts"
        if not drafts_dir.is_dir():
            return JSONResponse({"drafts": []})
        items: list[dict[str, Any]] = []
        for p in sorted(
            drafts_dir.glob("*.yaml"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:40]:
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Parse JUST the fields the picker needs — topic + title.
            # Doing a full YAML load here is fine; the files are tiny.
            try:
                import yaml as _yaml
                doc = _yaml.safe_load(txt) or {}
            except Exception:
                doc = {}
            items.append({
                "filename": p.name,
                "mtime": p.stat().st_mtime,
                "size": p.stat().st_size,
                "topic": (doc.get("topic") or "").strip()[:200],
                "title": (doc.get("title") or "").strip()[:200],
                "provider": (doc.get("provider") or {}).get("name") if isinstance(doc.get("provider"), dict) else None,
                "companion_md": str(p.with_name(p.stem + "-proposal.md").name)
                    if (drafts_dir / (p.stem + "-proposal.md")).is_file() else None,
            })
        return JSONResponse({"drafts": items})

    @app.get("/api/drafts/{filename}")
    async def get_draft(filename: str) -> JSONResponse:
        """Return the parsed contents of a single draft YAML so the
        /interview page can pre-fill its form fields when the user
        clicks "Load draft" on a proposal output.

        Also returns the companion proposal markdown (``<stem>-proposal.md``)
        when present, so the interview page can render the full
        background/hypothesis/plan/risks alongside the form. The YAML
        alone only holds the *runnable* config (topic, title, provider,
        execution params, …); the rich proposal text lives in the MD.
        """
        # Path-traversal guard: only allow simple filenames inside
        # outputs/_drafts/, never path components like '..'.
        if "/" in filename or "\\" in filename or filename.startswith(".") or not filename.endswith(".yaml"):
            raise HTTPException(400, f"bad filename: {filename!r}")
        draft = app.state.output_root / "_drafts" / filename
        if not draft.is_file():
            raise HTTPException(404, f"draft {filename!r} not found")
        try:
            txt = draft.read_text(encoding="utf-8", errors="replace")
            import yaml as _yaml
            doc = _yaml.safe_load(txt) or {}
        except Exception as e:
            raise HTTPException(500, f"could not parse draft: {e}")
        # Companion proposal markdown (same stem with -proposal.md suffix)
        # — bundled in the response so the picker only does one round
        # trip to load both the YAML config + the readable plan.
        proposal_md = draft.with_name(draft.stem + "-proposal.md")
        md_text: str | None = None
        if proposal_md.is_file():
            try:
                md_text = proposal_md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                md_text = None
        return JSONResponse({
            "filename": filename,
            "raw": txt,
            "parsed": doc,
            "proposal_md": md_text,
            "proposal_md_filename": proposal_md.name if md_text else None,
        })

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, n: int = 400) -> JSONResponse:
        """Status + log tail for a single subprocess job. The tools
        page polls this every 2s after submitting so the user sees
        live output instead of staring at a frozen "Job started"
        banner. Returns 404 only when neither the launcher nor any
        log file knows about this job_id."""
        if not _QUEST_ID_RE.match(job_id):
            raise HTTPException(400, f"bad job_id format: {job_id!r}")
        # Probe the three known layouts in newest-design-first order:
        # per-quest .fi/launch.log → per-tool-job _jobs/<id>/launch.log
        # → legacy _logs/<id>.log. First match wins.
        out_root = app.state.output_root
        log_path: Path | None = None
        for candidate in (
            out_root / job_id / ".fi" / "launch.log",
            out_root / "_jobs" / job_id / "launch.log",
            out_root / "_logs" / f"{job_id}.log",
        ):
            if candidate.is_file():
                log_path = candidate
                break
        st = app.state.launcher.status_for(job_id)
        log_lines: list[str] = []
        log_mtime = None
        log_size = None
        if log_path is not None:
            try:
                lines = log_path.read_text(
                    encoding="utf-8", errors="replace",
                ).splitlines()
                log_lines = lines[-n:]
                log_mtime = log_path.stat().st_mtime
                log_size = log_path.stat().st_size
            except OSError:
                pass
        if st is None and log_size is None:
            raise HTTPException(404, f"no job {job_id} tracked or logged")
        out: dict[str, Any] = {
            "job_id": job_id,
            "kind": _classify_job(job_id),
            "log_tail": log_lines,
            "log_mtime": log_mtime,
            "log_size": log_size,
        }
        if st is not None:
            out["alive"] = bool(st.get("alive"))
            out["pid"] = st.get("pid")
            out["age_seconds"] = st.get("age_seconds")
            out["started_at"] = st.get("started_at")
            if "exit_code" in st:
                out["exit_code"] = st["exit_code"]
        else:
            out["alive"] = False  # known only via log file → already finished
        return JSONResponse(out)

    @app.post("/api/system/install-tectonic")
    async def install_tectonic() -> JSONResponse:
        """Spawn ``python launch.py --install-tectonic`` as a job.
        Downloads tectonic into tools/ so paper_pdf works without
        an admin install of MiKTeX.

        Idempotent: when tectonic is already on disk (repo-local
        or PATH), returns 200 with ``already_present: True`` and
        does NOT spawn another installer. Without this guard the
        user could click "Install" repeatedly and each click
        kicked off a parallel download/CTAN-fetch — wasteful + the
        spawned subprocess had no completion-state surface so the
        UI showed it as "running" forever.
        """
        # Same probe as the GET endpoint above; in-process so we
        # don't pay the HTTP round-trip.
        import sys as _sys
        import shutil as _shutil
        repo_root = Path(__file__).resolve().parent.parent
        local_bin = repo_root / "tools" / (
            "tectonic.exe" if _sys.platform == "win32" else "tectonic"
        )
        if local_bin.is_file() or _shutil.which("tectonic"):
            return JSONResponse({
                "already_present": True,
                "location": str(local_bin) if local_bin.is_file() else _shutil.which("tectonic"),
                "spawned": False,
            })
        from web.quest_launcher import QuestLauncherFull
        try:
            launched = app.state.launcher.launch_command(
                argv_tail=["--install-tectonic"],
                # Random suffix avoids same-second collision if two
                # tabs both click "install tectonic" — see tools_routes
                # for the same fix on the 8 tool launchers.
                job_id=f"tectonic-{int(time.time())}-{secrets.token_hex(3)}",
            )
        except QuestLauncherFull as e:
            return JSONResponse(
                {"error": "launcher at capacity", "detail": str(e),
                 "retry_after_seconds": 30},
                status_code=503,
                headers={"Retry-After": "30"},
            )
        return JSONResponse({
            "job_id": launched.quest_id, "pid": launched.pid,
            "spawned": True,
            "already_present": False,
        })

    @app.delete("/api/quests/{quest_id}")
    async def trash_quest(quest_id: str) -> JSONResponse:
        """Move the quest dir to ``outputs/_trash/<id>-<timestamp>``.
        Safe-by-default delete — user can restore from /trash, or
        purge forever once they're sure.

        Refuses when the quest is still running: the engine
        subprocess holds open file handles inside quest_root, and
        moving the dir mid-flight either fails on Windows (file
        locked) or causes the engine to write into a now-stale path
        and recreate a partial quest_root. User cancels first, then
        trashes."""
        if not _QUEST_ID_RE.match(quest_id):
            raise HTTPException(400, f"bad quest_id format: {quest_id!r}")
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        if not quest_root.is_dir():
            raise HTTPException(404, f"quest {quest_id} not found")
        # Alive in either tracker = refuse.
        launcher_alive = bool(
            (app.state.launcher.status_for(quest_id) or {}).get("alive")
        )
        if registry.alive(quest_id) or launcher_alive:
            raise HTTPException(
                409,
                f"quest {quest_id} is still running. Cancel it "
                f"(POST /api/quests/{quest_id}/cancel) before moving "
                f"the directory to trash.",
            )
        trash_dir = app.state.output_root / "_trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        bin_id = f"{quest_id}-{int(time.time())}"
        target = trash_dir / bin_id
        try:
            shutil.move(str(quest_root), str(target))
        except OSError as e:
            raise HTTPException(500, f"trash failed: {e}") from e
        return JSONResponse({"trashed": True, "bin_id": bin_id})

    @app.get("/api/trash")
    async def list_trash() -> JSONResponse:
        trash_dir = app.state.output_root / "_trash"
        if not trash_dir.is_dir():
            return JSONResponse({"items": []})
        items = []
        for d in sorted(trash_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            try:
                stat = d.stat()
                items.append({
                    "bin_id": d.name,
                    "trashed_at": stat.st_mtime,
                    "size_bytes": _dir_size(d),
                })
            except OSError:
                continue
        return JSONResponse({"items": items})

    @app.post("/api/trash/{bin_id}/restore")
    async def restore_trash(bin_id: str) -> JSONResponse:
        # Validate against the allowlist + a bin-id-specific check
        # (trailing -<timestamp> is added by trash_quest).
        if not re.match(r"^[A-Za-z0-9_\-.]+$", bin_id):
            raise HTTPException(400, f"bad bin_id: {bin_id!r}")
        trash_dir = app.state.output_root / "_trash"
        src = (trash_dir / bin_id).resolve()
        try:
            src.relative_to(trash_dir.resolve())
        except ValueError:
            raise HTTPException(400, "bin_id escapes trash dir") from None
        if not src.is_dir():
            raise HTTPException(404, f"trash item not found: {bin_id}")
        # Strip the trailing "-<ts>" the trash op added to recover
        # the original quest_id. Fall through to the full bin_id when
        # the timestamp isn't present.
        original_id = re.sub(r"-\d{9,}$", "", bin_id) or bin_id
        target = app.state.output_root / original_id
        if target.exists():
            raise HTTPException(
                409, f"can't restore: {target} already exists. "
                "Rename or purge the existing quest first.",
            )
        try:
            shutil.move(str(src), str(target))
        except OSError as e:
            raise HTTPException(500, f"restore failed: {e}") from e
        return JSONResponse({"restored": True, "quest_id": original_id})

    @app.delete("/api/trash/{bin_id}")
    async def purge_trash(bin_id: str) -> JSONResponse:
        """Permanent delete. Confirm modal lives in the trash UI."""
        if not re.match(r"^[A-Za-z0-9_\-.]+$", bin_id):
            raise HTTPException(400, f"bad bin_id: {bin_id!r}")
        trash_dir = app.state.output_root / "_trash"
        target = (trash_dir / bin_id).resolve()
        try:
            target.relative_to(trash_dir.resolve())
        except ValueError:
            raise HTTPException(400, "bin_id escapes trash dir") from None
        if not target.is_dir():
            raise HTTPException(404, f"trash item not found: {bin_id}")
        try:
            shutil.rmtree(target)
        except OSError as e:
            raise HTTPException(500, f"purge failed: {e}") from e
        return JSONResponse({"purged": True, "bin_id": bin_id})

    @app.post("/api/quests/{quest_id}/cancel")
    async def cancel_quest(quest_id: str) -> JSONResponse:
        """Send SIGTERM (or CTRL_BREAK_EVENT on Windows) to a
        web-launched quest. Returns 404 if the quest_id isn't
        tracked by the launcher (e.g. started via CLI directly —
        cancel that with kill from the terminal)."""
        if not _QUEST_ID_RE.match(quest_id):
            raise HTTPException(400, f"bad quest_id format: {quest_id!r}")
        canceled = app.state.launcher.cancel(quest_id)
        if not canceled:
            raise HTTPException(404, f"quest {quest_id} not tracked by web launcher")
        return JSONResponse({"quest_id": quest_id, "canceled": True})

    @app.get("/api/quests")
    async def list_quests() -> JSONResponse:
        return JSONResponse({"quests": _scan_quests(app.state.output_root)})

    @app.get("/api/quests/{quest_id}")
    async def get_quest(quest_id: str) -> JSONResponse:
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        # Race window: a quest just submitted via
        # POST /api/interview/submit?launch=true has had its quest_id
        # minted and its subprocess spawned, but the child engine has
        # not yet created `.fi/` (LangGraph init takes a few seconds —
        # model loading, ADC, env, …). Without this fallback the user
        # gets a hard 404 the instant the form redirects to /quest/<id>.
        # When the launcher is tracking the id, surface a "starting"
        # response with the subprocess pid + age so the UI shows
        # progress instead of a dead-end error.
        launcher_status = app.state.launcher.status_for(quest_id)
        if not (quest_root / ".fi").is_dir():
            if launcher_status and (launcher_status.get("alive") or launcher_status.get("age_seconds", 0) < 60):
                return JSONResponse({
                    "quest_id": quest_id,
                    "quest_root": str(quest_root),
                    "current_node": "starting",
                    "log_tail": [
                        f"[FI] subprocess pid={launcher_status.get('pid')} spawned "
                        f"{int(launcher_status.get('age_seconds') or 0)}s ago — waiting for "
                        ".fi/run.log to appear (Engine init takes ~5–15s on first start).",
                    ],
                    "figures": [],
                    "paper_preview": None,
                    "summary": None,
                    "alive": bool(launcher_status.get("alive")),
                    "pending_clarify": False,
                    "pending_human_review": None,
                    "review": None,
                    "review_panel": None,
                    "pid": launcher_status.get("pid"),
                    "started_at": launcher_status.get("started_at"),
                    "age_seconds": launcher_status.get("age_seconds"),
                    "exit_code": launcher_status.get("exit_code"),
                })
            raise HTTPException(404, f"quest {quest_id} not found")
        log_lines = _read_log_tail(quest_root / ".fi" / "run.log", n=20)
        paper_md = quest_root / "paper" / "paper.md"
        figures_dir = quest_root / "figures"
        figures = [
            p.name for p in figures_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}
        ] if figures_dir.is_dir() else []
        summary_path = quest_root / "frontier_insight_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists() else None
        )
        final_state = registry.final_state(quest_id) or {}
        # Expose the panel reviews to the GUI when present.
        review = final_state.get("review")
        review_panel = final_state.get("review_panel")
        # Merge launcher state so quests spawned via the
        # web UI's POST /api/interview/submit?launch=true also report
        # ``alive: true`` while their child process is running.
        # Without this, the in-process registry's `alive(quest_id)`
        # returns False for ANY web-launched quest (those don't
        # touch the in-process registry — only CLI/VSCode-launched
        # quests do), so the detail page's status badge would
        # incorrectly show "idle" and the Cancel button would stay
        # disabled. registry.alive(...) OR launcher.alive covers both
        # transports.
        # Already fetched above for the spawning-race fallback; reuse it.
        launcher_status = launcher_status or {}
        # Use a LARGER window for the stage-progress derivation than
        # the 20-line dashboard tail. A chatty node (literature with
        # many docs, design with a long JSON, ...) can easily push its
        # own opening ``[<node>] ...`` line off a 20-line tail, in
        # which case ``node_started_at`` would be ``None`` and the
        # elapsed/idle chip would silently hide. 500 lines is enough
        # for any practical node — typical run.logs stay under 200
        # lines per node — while still cheap on disk for a long quest.
        node_progress_lines = _read_log_tail(
            quest_root / ".fi" / "run.log", n=500,
        )
        node_progress = _node_progress_from_log(
            node_progress_lines, _KNOWN_NODES,
        )
        quest_failed = _read_quest_failed_md(quest_root)
        # Human-review gate state for the dashboard banner:
        #   - in-process pending future, OR
        #   - on-disk snapshot with no answer-file present yet.
        pending_hr_snap = registry.pending_human_review(quest_id)
        pending_human_review: dict[str, Any] | None = None
        if pending_hr_snap is not None:
            pending_human_review = pending_hr_snap
        else:
            hr_path = quest_root / ".fi" / "human_review.json"
            hr_answer = quest_root / ".fi" / "human_review_answer.json"
            if hr_path.is_file() and not hr_answer.is_file():
                try:
                    pending_human_review = json.loads(
                        hr_path.read_text(encoding="utf-8"),
                    )
                except (OSError, json.JSONDecodeError):
                    pending_human_review = None
        return JSONResponse({
            "quest_id": quest_id,
            "quest_root": str(quest_root),
            "current_node": _current_node_from_log(log_lines),
            # Stage-stuck signals derived from run.log timestamps. The
            # JS detail page uses these to render a colored badge
            # (green/yellow/red by idle threshold). All three are null
            # when the log has nothing parseable yet.
            "node_started_at": node_progress["node_started_at"],
            "node_elapsed_s": node_progress["node_elapsed_s"],
            "node_idle_s": node_progress["node_idle_s"],
            "log_tail": log_lines,
            "figures": figures,
            "paper_preview": (
                paper_md.read_text(encoding="utf-8")[:4000]
                if paper_md.exists() else None
            ),
            "summary": summary,
            "alive": (
                registry.alive(quest_id)
                or bool(launcher_status.get("alive"))
            ),
            "pending_clarify": registry.pending_clarify(quest_id) is not None,
            "pending_human_review": pending_human_review,
            "review": review,
            "review_panel": review_panel,
            # `quest_failed.md` summary when the engine wrote one
            # (post-crash diagnostic). Null when the file is absent so
            # the UI can hide the banner. Lives in the quest root,
            # cleared by the engine on a successful resume.
            "quest_failed": quest_failed,
            # Subprocess-launcher specifics for the detail page UI.
            # `pid` lets users find the process; `started_at` powers a
            # "running for X minutes" hint. Both null when the quest
            # wasn't launched via this server.
            "launcher_pid": launcher_status.get("pid"),
            "launcher_started_at": launcher_status.get("started_at"),
        })

    @app.get("/api/quests/{quest_id}/log")
    async def get_log(quest_id: str, n: int = 200) -> JSONResponse:
        log_path = _resolve_quest_root(app.state.output_root, quest_id) / ".fi" / "run.log"
        return JSONResponse({"lines": _read_log_tail(log_path, n=n)})

    @app.get("/api/quests/{quest_id}/log/stream")
    async def stream_log(quest_id: str) -> StreamingResponse:
        log_path = _resolve_quest_root(app.state.output_root, quest_id) / ".fi" / "run.log"

        async def gen():
            offset = 0
            # Wait briefly for the file to exist on cold start.
            for _ in range(30):
                if log_path.exists():
                    break
                await asyncio.sleep(0.2)
            if log_path.exists():
                offset = log_path.stat().st_size
                # Send the existing tail first.
                for line in _read_log_tail(log_path, n=50):
                    yield f"data: {line}\n\n"
            while True:
                await asyncio.sleep(0.5)
                if not log_path.exists():
                    continue
                size = log_path.stat().st_size
                if size <= offset:
                    # Heartbeat — keeps the SSE connection from being
                    # culled by intermediate proxies.
                    yield ": keepalive\n\n"
                    continue
                with log_path.open("rb") as f:
                    f.seek(offset)
                    chunk = f.read().decode("utf-8", errors="replace")
                offset = size
                for line in chunk.splitlines():
                    if line.strip():
                        yield f"data: {line}\n\n"
                # Stop tailing once the quest is done. The in-process
                # `registry.alive(quest_id)` is False for any
                # subprocess-launched quest (those are tracked by
                # `app.state.launcher`, not the registry). Without the
                # OR-merge below, the stream would close on the FIRST
                # appended chunk for every web-launched quest.
                launcher_alive = bool(
                    (app.state.launcher.status_for(quest_id) or {}).get("alive")
                )
                if not registry.alive(quest_id) and not launcher_alive:
                    yield "data: [stream closed: quest task ended]\n\n"
                    return

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/quests/{quest_id}/clarify")
    async def get_clarify(quest_id: str) -> JSONResponse:
        questions = registry.pending_clarify(quest_id)
        if questions is None:
            return JSONResponse({"pending": False, "questions": None})
        return JSONResponse({"pending": True, "questions": questions})

    @app.post("/api/quests/{quest_id}/clarify")
    async def post_clarify(quest_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise HTTPException(400, "request body must contain {'answers': {...}}")
        if not registry.resolve_clarify(quest_id, answers):
            raise HTTPException(409, f"no pending clarify for quest {quest_id}")
        return JSONResponse({"ok": True})

    @app.get("/api/quests/{quest_id}/human-review")
    async def get_human_review(quest_id: str) -> JSONResponse:
        """Return the current human-review snapshot for a quest. Two
        sources, in order of authority:

        1. The in-process registry — populated when a web-launched
           quest's engine reaches the human_feedback interrupt and
           awaits the GUI callback.
        2. ``<quest_root>/.fi/human_review.json`` on disk — populated
           by EVERY quest that hits the gate (CLI, subprocess,
           VSCode), so the dashboard can render a review for any
           paused quest even when no in-process future is wired.
        """
        snap = registry.pending_human_review(quest_id)
        if snap is not None:
            return JSONResponse({"pending": True, "source": "in_process",
                                 "snapshot": snap})
        try:
            quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        except HTTPException:
            return JSONResponse({"pending": False, "source": None,
                                 "snapshot": None})
        on_disk = quest_root / ".fi" / "human_review.json"
        if not on_disk.is_file():
            return JSONResponse({"pending": False, "source": None,
                                 "snapshot": None})
        # Mid-quest the answer may already be staged on disk by an
        # earlier POST; treat that as "no longer pending".
        if (quest_root / ".fi" / "human_review_answer.json").is_file():
            return JSONResponse({"pending": False, "source": "disk",
                                 "snapshot": None})
        try:
            snap = json.loads(on_disk.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return JSONResponse({"pending": False, "source": None,
                                 "snapshot": None})
        return JSONResponse({"pending": True, "source": "disk",
                             "snapshot": snap})

    @app.post("/api/quests/{quest_id}/human-review")
    async def post_human_review(
        quest_id: str, request: Request,
    ) -> JSONResponse:
        """Submit the user's accept / reject / refine decision. Two
        resolution paths:

        1. An in-process future (the in-memory engine path used by
           interview submit + launch=true) is resolved when present;
           the engine resumes within the same Python process.
        2. ``<quest_root>/.fi/human_review_answer.json`` is always
           written so a subprocess- or CLI-launched engine can pick
           up the decision on its next ``--resume``.
        """
        body = await request.json()
        action_raw = str(body.get("action") or "").strip().lower()
        if action_raw not in ("accept", "reject", "refine"):
            raise HTTPException(
                400, "action must be one of accept, reject, refine",
            )
        feedback = str(body.get("feedback") or "").strip()
        if action_raw == "refine" and not feedback:
            raise HTTPException(
                400, "refine requires non-empty feedback",
            )
        answer = {"action": action_raw, "feedback": feedback}
        in_process_resolved = registry.resolve_human_review(quest_id, answer)
        # Always write the disk answer so an out-of-process
        # ``--resume`` picks it up too. Best-effort; a missing
        # quest_root means the in-process resolve was authoritative.
        try:
            quest_root = _resolve_quest_root(app.state.output_root, quest_id)
            fi = quest_root / ".fi"
            fi.mkdir(parents=True, exist_ok=True)
            (fi / "human_review_answer.json").write_text(
                json.dumps(answer, indent=2) + "\n", encoding="utf-8",
            )
        except HTTPException:
            quest_root = None  # type: ignore[assignment]
        if not in_process_resolved and quest_root is None:
            raise HTTPException(
                409, f"no pending human-review for quest {quest_id}",
            )
        return JSONResponse({"ok": True, "in_process_resolved": in_process_resolved})

    @app.get("/api/quests/{quest_id}/paper")
    async def get_paper(quest_id: str) -> FileResponse:
        paper = _resolve_quest_root(app.state.output_root, quest_id) / "paper" / "paper.md"
        if not paper.exists():
            raise HTTPException(404, "paper.md not yet written")
        return FileResponse(str(paper), media_type="text/markdown")

    @app.get("/api/quests/{quest_id}/files")
    async def list_quest_files(quest_id: str) -> JSONResponse:
        """Return a flat list of every file under the quest_root,
        with relative path + size + content-type hint. Used by the
        file-browser pane on /quest/<id>. Skips ``.fi/`` (engine
        internals) so the user doesn't see SQLite + run.log clutter."""
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        items: list[dict[str, Any]] = []
        for p in sorted(quest_root.rglob("*")):
            try:
                if not p.is_file():
                    continue
                rel = p.relative_to(quest_root)
                if rel.parts and rel.parts[0] == ".fi":
                    continue
                items.append({
                    "path": str(rel).replace("\\", "/"),
                    "size_bytes": p.stat().st_size,
                    "ext": p.suffix.lower(),
                })
            except OSError:
                continue
        return JSONResponse({"quest_id": quest_id, "files": items})

    @app.get("/api/quests/{quest_id}/file")
    async def get_quest_file(quest_id: str, path: str) -> FileResponse:
        """Serve a single file from the quest_root by relative path.
        Path-traversal guarded: resolves and confirms the final
        path stays inside quest_root.

        Some clients (VS Code Live Server, certain browser extensions)
        naively append ``?preventCache=<unix-ms>`` to URLs without
        checking whether a query string already exists, producing
        ``?path=paper.md?preventCache=...`` — FastAPI then reads the
        path value as the literal ``paper.md?preventCache=...``.
        Strip everything after the first ``?``/``#`` so the file
        actually resolves instead of 404-ing.
        """
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        cleaned = path.split("?", 1)[0].split("#", 1)[0]
        target = (quest_root / cleaned).resolve()
        try:
            target.relative_to(quest_root.resolve())
        except ValueError:
            raise HTTPException(400, "path escapes quest dir") from None
        if not target.is_file():
            raise HTTPException(404, "file not found")
        return FileResponse(str(target))

    @app.get("/api/quests/{quest_id}/download")
    async def download_quest_zip(quest_id: str) -> StreamingResponse:
        """Stream the entire quest_root as a zip. Cheaper than
        materializing a temp file on disk — the BytesIO buffer
        builds in memory and streams out. Quests are typically
        a few MB; if you have a 500 MB quest, refactor to use
        ZipFile + temp file with proper cleanup."""
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in quest_root.rglob("*"):
                try:
                    if p.is_file():
                        rel = p.relative_to(quest_root)
                        # Skip the SQLite checkpoint — it's binary
                        # state that doesn't compress well + isn't
                        # useful to a recipient who can't replay it.
                        if rel.parts and rel.parts[0] == ".fi":
                            continue
                        zf.write(p, arcname=str(rel))
                except OSError:
                    continue
        buf.seek(0)

        async def gen():
            chunk = 64 * 1024
            while True:
                data = buf.read(chunk)
                if not data:
                    break
                yield data

        return StreamingResponse(
            gen(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{quest_id}.zip"'},
        )

    @app.get("/api/quests/{quest_id}/labels")
    async def get_labels(quest_id: str) -> JSONResponse:
        """Tags / labels stored in ``<quest_root>/.fi/labels.json``.
        Returns ``{"labels": [...]}``; empty list when the file
        doesn't exist yet."""
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        labels_path = quest_root / ".fi" / "labels.json"
        if not labels_path.is_file():
            return JSONResponse({"quest_id": quest_id, "labels": []})
        try:
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            labels = data.get("labels", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, OSError):
            labels = []
        return JSONResponse({"quest_id": quest_id, "labels": labels})

    @app.put("/api/quests/{quest_id}/labels")
    async def set_labels(quest_id: str, request: Request) -> JSONResponse:
        """Replace the label set for a quest. Refuses when the quest
        doesn't already exist (no ``.fi/`` dir) so a typo in the URL
        doesn't auto-create a bogus quest_root that ``_scan_quests``
        would then surface on the dashboard."""
        body = await request.json()
        labels = body.get("labels", [])
        if not isinstance(labels, list):
            raise HTTPException(400, "labels must be a list of strings")
        labels = [str(l).strip() for l in labels if str(l).strip()]
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        fi_dir = quest_root / ".fi"
        if not fi_dir.is_dir():
            raise HTTPException(
                404, f"quest {quest_id} not found (no .fi/ dir under "
                f"{quest_root}). Labels can only be set on existing quests.",
            )
        (fi_dir / "labels.json").write_text(
            json.dumps({"labels": labels}, indent=2), encoding="utf-8",
        )
        return JSONResponse({"quest_id": quest_id, "labels": labels})

    @app.get("/api/quests/{quest_id}/iterations")
    async def get_paper_iterations(quest_id: str) -> JSONResponse:
        """List paper.md snapshots across review iterations for the
        diff viewer. Looks for ``paper/paper.md.iter-N.md`` files
        the engine could write — for now just returns the current
        paper.md as a single iteration when the rolled-snapshot
        machinery isn't in place yet."""
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        paper_dir = quest_root / "paper"
        iterations: list[dict[str, Any]] = []
        if paper_dir.is_dir():
            for p in sorted(paper_dir.glob("paper.iter-*.md")):
                m = re.search(r"iter-(\d+)", p.name)
                if not m:
                    continue
                iterations.append({
                    "iter": int(m.group(1)),
                    "content": p.read_text(encoding="utf-8"),
                })
            current = paper_dir / "paper.md"
            if current.is_file():
                iterations.append({
                    "iter": len(iterations) + 1,
                    "label": "current",
                    "content": current.read_text(encoding="utf-8"),
                })
        return JSONResponse({"quest_id": quest_id, "iterations": iterations})

    @app.post("/api/quests/{quest_id}/code/execute")
    async def execute_quest_code(quest_id: str, request: Request) -> JSONResponse:
        """Save an edited ``code/experiment.py`` and re-run the
        execute node by spawning ``python launch.py --resume <id>``.
        Gated behind a config flag because this endpoint takes
        user-supplied Python code and runs it on the server. The
        existing ``--resume`` path picks up at ``execute`` if the
        upstream nodes (clarify / ideate / design / implement) have
        already completed.

        On a non-loopback bind, this endpoint is a code-execution
        risk — the warning emitted by ``_warn_if_non_loopback``
        explicitly flags it.
        """
        # The config-flag gate isn't a substitute for auth (we have
        # none) — it's a defense-in-depth opt-in for users who DO
        # bind to non-loopback.
        import os
        if not os.environ.get("FI_WEB_ALLOW_EXEC_EDIT", "").strip():
            raise HTTPException(
                403,
                "Re-execute is disabled. Start the server with "
                "FI_WEB_ALLOW_EXEC_EDIT=1 to opt in. This endpoint "
                "runs user-supplied Python on the server; only enable "
                "it on trusted hosts.",
            )
        body = await request.json()
        code = body.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise HTTPException(400, "code field is required")
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        target = quest_root / "code" / "experiment.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        # Clear downstream LangGraph state so `--resume` actually
        # re-runs the execute node. Without this, a terminal-state
        # checkpoint (paper.md present, review.verdict set) would
        # short-circuit the graph and the user's code edit would be
        # silently ignored. Reuses the same soft-invalidation
        # primitive interview_update.py uses for paper_format /
        # study_depth changes.
        from core.interview_update import soft_invalidate_checkpoint
        await soft_invalidate_checkpoint(
            quest_root,
            ["exec_result", "result_json", "figures",
             "analysis", "cross_check", "paper_md",
             "review", "review_panel"],
        )
        # Then resume the quest — the execute node re-reads
        # experiment.py from disk.
        yaml_path = quest_root / "config.yaml"
        if not yaml_path.is_file():
            raise HTTPException(400, "quest has no config.yaml; can't resume")
        from web.quest_launcher import QuestLauncherFull
        try:
            launched = app.state.launcher.launch_command(
                argv_tail=["--config", str(yaml_path), "--resume", quest_id],
                job_id=quest_id,
            )
        except QuestLauncherFull as e:
            return JSONResponse(
                {"error": "launcher at capacity", "detail": str(e)},
                status_code=503,
            )
        return JSONResponse({
            "saved": True, "rerun_pid": launched.pid,
            "invalidated_keys": ["exec_result", "result_json", "figures",
                                  "analysis", "cross_check", "paper_md",
                                  "review", "review_panel"],
        })

    @app.get("/api/quests/{quest_id}/cost")
    async def get_quest_cost(quest_id: str) -> JSONResponse:
        """Read <quest_root>/.fi/cost.jsonl rows produced by the
        engine's cost instrumentation. Returns the per-call records;
        the chart on /quest/<id> aggregates client-side."""
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        cost_path = quest_root / ".fi" / "cost.jsonl"
        if not cost_path.is_file():
            return JSONResponse({"records": [], "available": False})
        records = []
        try:
            for line in cost_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        return JSONResponse({"records": records, "available": True})

    @app.get("/api/quests/{quest_id}/figure/{name}")
    async def get_figure(quest_id: str, name: str) -> FileResponse:
        # Defend against path traversal on both segments — quest_id
        # goes through the strict-allowlist validator; name gets the
        # legacy character check (it can have hyphens / dots that the
        # regex would reject but are fine for figure filenames).
        quest_root = _resolve_quest_root(app.state.output_root, quest_id)
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "bad figure name")
        path = quest_root / "figures" / name
        if not path.exists() or not path.is_file():
            raise HTTPException(404, "figure not found")
        return FileResponse(str(path))

    @app.post("/api/quests/start")
    async def start_quest(request: Request) -> JSONResponse:
        body = await request.json()
        yaml_blob = body.get("yaml")
        if not yaml_blob:
            raise HTTPException(400, "request body must contain {'yaml': '<config text>'}")
        import yaml as _yaml
        try:
            cfg_dict = _yaml.safe_load(yaml_blob)
            cfg = Config.model_validate(cfg_dict)
        except Exception as e:
            raise HTTPException(400, f"invalid YAML config: {e}") from e
        # Ensure the quest lands under the server's configured output_root.
        cfg.output.output_dir = app.state.output_root

        engine = Engine(cfg, supervisor=app.state.supervisor)
        quest_id = engine.quest_id

        async def gui_clarify_callback(questions: dict[str, Any]) -> dict[str, Any]:
            fut = registry.register_clarify(quest_id, questions)
            return await fut

        async def gui_human_review_callback(
            snapshot: dict[str, Any],
        ) -> dict[str, Any]:
            # The web UI POSTs the user's action+feedback to
            # /api/quests/<id>/human-review which resolves the future
            # registered here. The engine then resumes with the
            # accept / reject / refine decision.
            fut = registry.register_human_review(quest_id, snapshot)
            return await fut

        async def driver():
            try:
                art = await engine.run(
                    clarify_callback=gui_clarify_callback,
                    human_feedback_callback=gui_human_review_callback,
                )
                # Snapshot review_panel + final review so the
                # detail endpoint can render the per-persona reviews
                # without re-reading the SqliteSaver checkpoint.
                registry.record_final_state(quest_id, art.raw_state or {})
            except Exception as e:
                engine._log.error("[server] quest %s failed: %s", quest_id, e)

        task = asyncio.create_task(driver())
        registry.register_task(quest_id, task, cfg)
        return JSONResponse({
            "quest_id": quest_id,
            "quest_root": str(engine.quest_root),
        })

    return app


def _warn_if_non_loopback(host: str) -> None:
    """The web UI now spawns quests via ``POST /api/interview/submit?launch=true``.
    Quest spawn is a privileged action — anyone who can reach the
    endpoint can run arbitrary ``python launch.py --config <yaml>``
    on the server's machine. Binding to anything other than a
    loopback (``127.0.0.1`` / ``::1`` / ``localhost``) exposes that
    to the network. Log a loud WARNING with the bound host so the
    user can see exactly what they're exposing; we don't refuse to
    start because some users legitimately want LAN access on a
    trusted network."""
    loopback = {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}
    if host not in loopback:
        import logging
        log = logging.getLogger("frontier_insight.serve")
        log.warning(
            "--serve bound to %s (non-loopback). The /interview "
            "endpoint launches arbitrary quests on this machine; "
            "anyone reachable on the network can submit one. "
            "Bind to 127.0.0.1 unless you trust the network.",
            host,
        )


async def serve_async(
    *,
    output_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_concurrent: int = 4,
    vscode_bridge_port: int = 0,
    vscode_bridge_socket: str = "",
) -> None:
    """Async entry point — invoked from within an existing event loop.
    Uses ``uvicorn.Server.serve`` so we don't try to nest event loops.
    The server runs until SIGINT/SIGTERM.

    ``max_concurrent`` caps how many quests the web UI's subprocess
    launcher can spawn at once. ``vscode_bridge_port`` /
    ``vscode_bridge_socket`` are passed to each spawned child so LLM
    calls keep routing through the same bridge the dashboard
    inherits. Both forwarded to ``make_app``."""
    import uvicorn  # imported here so non-server runs don't need it
    _warn_if_non_loopback(host)
    _ensure_axon_sidecar()
    app = make_app(
        output_root.resolve(),
        max_concurrent=max_concurrent,
        vscode_bridge_port=vscode_bridge_port,
        vscode_bridge_socket=vscode_bridge_socket,
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def serve(
    *,
    output_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_concurrent: int = 4,
    vscode_bridge_port: int = 0,
    vscode_bridge_socket: str = "",
) -> None:
    """Blocking entry point invoked when run standalone (no event loop)."""
    import uvicorn  # imported here so non-server runs don't need it
    _warn_if_non_loopback(host)
    _ensure_axon_sidecar()
    app = make_app(
        output_root.resolve(),
        max_concurrent=max_concurrent,
        vscode_bridge_port=vscode_bridge_port,
        vscode_bridge_socket=vscode_bridge_socket,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def _ensure_axon_sidecar() -> None:
    """Boot the Axon API sidecar so the knowledge layer is hot when
    the first quest runs. Idempotent — does nothing if Axon is
    already listening, or if ``FI_NO_AXON_SIDECAR=1`` is set."""
    if os.environ.get("FI_NO_AXON_SIDECAR"):
        return
    try:
        from core.axon_sidecar import ensure_axon_up
        ensure_axon_up()
    except Exception as e:  # noqa: BLE001
        logging.getLogger("fi.web").warning("axon sidecar bootstrap failed: %s", e)
