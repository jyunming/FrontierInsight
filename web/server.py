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
import re
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

    Phase N: also tracks the in-memory `Engine` reference so the
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
    "clarify", "ideate", "literature", "design", "implement",
    "execute", "execute_reflect", "analyze", "cross_check",
    "write", "review",
})


def _current_node_from_log(lines: list[str]) -> str:
    """Best-effort: find the most recent ``[<node>]`` tag in the log.

    Match strict ``[lowercase_underscore]`` only, and prefer the LAST
    occurrence on a line that names a known node — otherwise random
    bracketed tokens like ``['matplotlib']`` from a pip-install log
    line would be mistaken for the current node."""
    for line in reversed(lines):
        for m in reversed(list(_NODE_TAG_RE.finditer(line))):
            tag = m.group(1)
            if tag in _KNOWN_NODES:
                return tag
    return "(unknown)"


# ---- app factory ----------------------------------------------------------


def make_app(
    output_root: Path,
    *,
    max_concurrent: int = 4,
    vscode_bridge_port: int = 0,
) -> FastAPI:
    """Build the FastAPI app. Pass the on-disk root where quest dirs
    will live; the server scans this for status and writes new quests
    here when ``POST /api/quests/start`` is called.

    ``max_concurrent`` bounds the number of in-flight quests spawned
    via the web UI interview. ``vscode_bridge_port`` is forwarded to
    each child quest so LLM calls keep routing through the same
    bridge the dashboard inherits from its parent."""
    app = FastAPI(title="Frontier Insight", version="1.0")
    registry = _QuestRegistry()
    app.state.registry = registry
    app.state.output_root = output_root
    app.state.supervisor = ProxySupervisor()

    # Subprocess launcher for quests spawned from the web UI.
    # The launcher lives on the app so request handlers can share it.
    from web.quest_launcher import QuestLauncher
    repo_root = Path(__file__).resolve().parent.parent
    app.state.launcher = QuestLauncher(
        repo_root=repo_root,
        max_concurrent=max_concurrent,
        vscode_bridge_port=vscode_bridge_port,
    )

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Interview routes — /interview (new quest), /update/<id> (mid-quest
    # re-entry), JSON schema endpoint, and POST handlers. Backed by
    # core.interview as the single source of truth.
    from web.interview_routes import register_interview_routes
    register_interview_routes(app, output_root)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_html = static_dir / "index.html"
        if not index_html.exists():
            return HTMLResponse(
                "<h1>Frontier Insight</h1><p>UI not installed.</p>", status_code=500,
            )
        return HTMLResponse(index_html.read_text(encoding="utf-8"))

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
        if not (quest_root / ".fi").is_dir():
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
        # Phase N — expose the panel reviews to the GUI when present.
        review = final_state.get("review")
        review_panel = final_state.get("review_panel")
        return JSONResponse({
            "quest_id": quest_id,
            "quest_root": str(quest_root),
            "current_node": _current_node_from_log(log_lines),
            "log_tail": log_lines,
            "figures": figures,
            "paper_preview": (
                paper_md.read_text(encoding="utf-8")[:4000]
                if paper_md.exists() else None
            ),
            "summary": summary,
            "alive": registry.alive(quest_id),
            "pending_clarify": registry.pending_clarify(quest_id) is not None,
            "review": review,
            "review_panel": review_panel,
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
                # Stop tailing once the quest task is done.
                if not registry.alive(quest_id):
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

    @app.get("/api/quests/{quest_id}/paper")
    async def get_paper(quest_id: str) -> FileResponse:
        paper = _resolve_quest_root(app.state.output_root, quest_id) / "paper" / "paper.md"
        if not paper.exists():
            raise HTTPException(404, "paper.md not yet written")
        return FileResponse(str(paper), media_type="text/markdown")

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

        async def driver():
            try:
                art = await engine.run(clarify_callback=gui_clarify_callback)
                # Phase N — snapshot review_panel + final review so the
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
    to the network. Log a loud WARNING with the user's IP so they
    can't miss it; we don't refuse to start because some users
    legitimately want LAN access on a trusted network."""
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
) -> None:
    """Async entry point — invoked from within an existing event loop.
    Uses ``uvicorn.Server.serve`` so we don't try to nest event loops.
    The server runs until SIGINT/SIGTERM."""
    import uvicorn  # imported here so non-server runs don't need it
    _warn_if_non_loopback(host)
    app = make_app(output_root.resolve())
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def serve(
    *,
    output_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Blocking entry point invoked when run standalone (no event loop)."""
    import uvicorn  # imported here so non-server runs don't need it
    _warn_if_non_loopback(host)
    app = make_app(output_root.resolve())
    uvicorn.run(app, host=host, port=port, log_level="info")
