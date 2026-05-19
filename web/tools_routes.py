"""FastAPI routes for the eight CLI tools exposed in the web UI.

Each tool maps to a CLI subcommand the user could also run from the
terminal:

* ``/tools/proposal`` → ``python launch.py --proposal "<topic>"``
* ``/tools/critique`` → ``python launch.py --critique <quest_id>``
* ``/tools/digest``   → ``python launch.py --digest --days N``
* ``/tools/portfolio``→ ``python launch.py --portfolio``
* ``/tools/summarize``→ ``python launch.py --summarize <dir>``
* ``/tools/analyze``  → ``python launch.py --analyze <dir> --analyze-topic "<topic>"``
* ``/tools/fleet``    → ``python launch.py --fleet a.yaml b.yaml ...``
* ``/tools/ingest``   → ``python launch.py --ingest <path1> <path2> ...``

The pattern is repetitive (form → validate → spawn subprocess via
:class:`web.quest_launcher.QuestLauncher` → return job_id), so we
drive both the UI and the backend from a single :data:`TOOL_SPECS`
dict. Adding a ninth tool is one entry here plus a CLI flag.

File-taking tools accept EITHER multipart upload (browser-native,
files written into ``outputs/_uploads/<job_id>/``) OR a server-side
path field for power users whose files already live on the box.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse


# ---------------------------------------------------------------------------
# Tool specs — drive the UI form + the backend handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolField:
    """One input on a tool's form."""
    name: str
    label: str
    prompt: str
    kind: str  # "text" | "longtext" | "int" | "file" | "files" | "path" | "quest_picker"
    required: bool = True
    placeholder: str = ""
    default: Any = None


@dataclass(frozen=True)
class ToolSpec:
    """Drives a /tools/<name> form + its POST handler."""
    name: str
    label: str
    blurb: str
    cli_flag: str       # e.g. "--proposal" or "--digest"
    fields: tuple[ToolField, ...]
    artifact_dir: str | None = None  # relative to output_root, e.g. "_drafts" — for "View output" link
    accepts_files: bool = False
    # When True, the form gets a provider + model picker (driven
    # by the same schema as the interview's provider/model
    # questions). The POST handler maps the picked values to the
    # tool's `--<tool>-provider` CLI flag. Defaults to True for
    # any tool that fires LLM calls; False for tools that only
    # touch files (ingest) or shell out (fleet has per-config
    # provider).
    needs_llm: bool = True


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="proposal",
        label="Proposal — plan before you run",
        blurb="Get a 1-page planning doc (background, hypothesis, plan, success criteria, risks) before committing compute. Produces an outputs/_drafts/<id>-proposal.md plus a companion YAML ready to feed into the main quest.",
        cli_flag="--proposal",
        artifact_dir="_drafts",
        fields=(
            ToolField(
                name="topic",
                label="Topic",
                prompt="The research question or topic to plan for. 1–3 paragraphs is the sweet spot.",
                kind="longtext",
                placeholder="Compare three numerical integrators on a damped harmonic oscillator and report energy drift.",
            ),
        ),
    ),
    ToolSpec(
        name="critique",
        label="Critique — adversarial second-pass review",
        blurb="Reads a finished quest's paper + code + the in-quest review, then asks a different LLM to find what the original reviewer missed. Writes outputs/<quest_id>/critique.md.",
        cli_flag="--critique",
        fields=(
            ToolField(
                name="quest_id",
                label="Quest ID",
                prompt="Which finished quest to critique. Use the dashboard to find the ID.",
                kind="quest_picker",
            ),
        ),
    ),
    ToolSpec(
        name="digest",
        label="Digest — weekly project-manager summary",
        blurb="Walks quests touched in the past N days, classifies each by terminal state, computes a structured WeekDiff vs the prior digest (✅ promoted / 🆕 new / ⚠️ in-progress / 🛑 stalled / ❓ dropped), and writes outputs/_digests/<YYYY-Www>.md.",
        cli_flag="--digest",
        artifact_dir="_digests",
        fields=(
            ToolField(
                name="days",
                label="Window (days)",
                prompt="Rolling window size. Default 7 (ISO-week canonical form).",
                kind="int",
                default=7,
            ),
        ),
    ),
    ToolSpec(
        name="portfolio",
        label="Portfolio — cross-quest synthesis",
        blurb="Walks every quest under outputs/ (no time window), surfaces topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, and prioritized next-quest suggestions. Writes outputs/_portfolio/<YYYY-MM-DD>.md.",
        cli_flag="--portfolio",
        artifact_dir="_portfolio",
        fields=(),  # no inputs — just press the button
    ),
    ToolSpec(
        name="summarize",
        label="Summarize — folder synthesizer",
        blurb="Walks a folder of mixed content (papers, source code, study notes, experiment logs), classifies each file, calls the LLM once, and writes outputs/<summary_id>/summary.md. Ingests the input set + the summary into Axon as fi_summary_input / fi_summary.",
        cli_flag="--summarize",
        accepts_files=True,
        fields=(
            ToolField(
                name="folder",
                label="Folder",
                prompt="Path to the folder on the server (or upload files). Recursive walk; symlinks skipped.",
                kind="path",
                placeholder="/home/me/papers-to-digest",
                required=False,
            ),
            ToolField(
                name="files",
                label="…or upload",
                prompt="Drop files here to upload into a fresh staging folder. Picks server-path above if both are set.",
                kind="files",
                required=False,
            ),
            ToolField(
                name="kind",
                label="Content type",
                prompt="Override auto-detection.",
                kind="text",
                default="auto",
                required=False,
            ),
        ),
    ),
    ToolSpec(
        name="analyze",
        label="Analyze — no-simulation quest with pre-staged data",
        blurb="The inverse of /proposal: you already have a dataset and want FI to write a paper analyzing it. Files are copied into the new quest's data/ dir; the engine routes wait_for_data → data_load → analyze → cross_check → write → review.",
        cli_flag="--analyze",
        accepts_files=True,
        fields=(
            ToolField(
                name="path",
                label="Data folder",
                prompt="Path on the server (or upload files). Every file is copied into the new quest's data/ dir.",
                kind="path",
                required=False,
            ),
            ToolField(
                name="files",
                label="…or upload",
                prompt="Drop CSVs / Markdown / PDFs here to stage into a fresh quest.",
                kind="files",
                required=False,
            ),
            ToolField(
                name="topic",
                label="Analysis topic",
                prompt="What the analysis should produce. One sentence.",
                kind="text",
                placeholder="Compare ridership trends across regions",
            ),
        ),
    ),
    ToolSpec(
        name="fleet",
        needs_llm=False,
        label="Fleet — N quests in parallel",
        blurb="Run multiple config.yaml quests concurrently with bounded concurrency. Each YAML produces its own quest dir.",
        cli_flag="--fleet",
        accepts_files=True,
        fields=(
            ToolField(
                name="yaml_paths",
                label="YAML paths (one per line)",
                prompt="Server-side paths to config.yaml files. Or upload below.",
                kind="longtext",
                required=False,
            ),
            ToolField(
                name="files",
                label="…or upload YAMLs",
                prompt="Drop one or more .yaml configs to run in parallel.",
                kind="files",
                required=False,
            ),
        ),
    ),
    ToolSpec(
        name="ingest",
        needs_llm=False,
        label="Ingest — load papers into Axon",
        blurb="Loads PDF / Markdown / TXT files into the Axon corpus as kind=fi_local_paper so future quests retrieve them as prior work.",
        cli_flag="--ingest",
        accepts_files=True,
        fields=(
            ToolField(
                name="paths",
                label="File paths (one per line)",
                prompt="Server-side paths to PDFs or Markdown files.",
                kind="longtext",
                required=False,
            ),
            ToolField(
                name="files",
                label="…or upload",
                prompt="Drop PDFs / Markdown / TXT here to ingest.",
                kind="files",
                required=False,
            ),
        ),
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOL_SPECS}


def _spec_to_dict(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "label": spec.label,
        "blurb": spec.blurb,
        "cli_flag": spec.cli_flag,
        "artifact_dir": spec.artifact_dir,
        "accepts_files": spec.accepts_files,
        "needs_llm": spec.needs_llm,
        "fields": [
            {
                "name": f.name, "label": f.label, "prompt": f.prompt,
                "kind": f.kind, "required": f.required,
                "placeholder": f.placeholder, "default": f.default,
            }
            for f in spec.fields
        ],
    }


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_tools_routes(app: FastAPI, output_root: Path) -> None:
    """Attach the /tools/* routes onto the FastAPI app."""
    static_dir = Path(__file__).resolve().parent / "static"

    @app.get("/tools/{tool_name}", response_class=HTMLResponse)
    async def tool_page(tool_name: str) -> HTMLResponse:
        if tool_name not in TOOLS_BY_NAME:
            raise HTTPException(404, f"unknown tool: {tool_name}")
        page = static_dir / "tools.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Tools UI not installed</h1>", status_code=500,
            )
        html = page.read_text(encoding="utf-8")
        injected = html.replace(
            "</head>",
            f'<script>window.__fi_tool_name = {json.dumps(tool_name)};</script></head>',
            1,
        )
        return HTMLResponse(injected)

    @app.get("/api/tools/schema")
    async def tools_schema() -> JSONResponse:
        bridge_port = int(getattr(app.state, "vscode_bridge_port", 0) or 0)
        bridge_socket = getattr(app.state, "vscode_bridge_socket", "") or ""
        # Probe both transports. Either one open → vscode_extension
        # lights up in the UI. The IPC socket path is now the canonical
        # default for --serve (no port to wire); the TCP port is kept
        # for backward compat with chat-spawned environments where the
        # extension passed --vscode-bridge-port. We probe in parallel
        # so the slower one doesn't gate the schema fetch.
        from web._bridge_probe import is_bridge_listening, is_socket_listening
        import asyncio as _asyncio
        tcp_ok, sock_ok = await _asyncio.gather(
            is_bridge_listening(bridge_port),
            is_socket_listening(bridge_socket),
        )
        bridge_live = bool(tcp_ok or sock_ok)
        return JSONResponse({
            "tools": [_spec_to_dict(t) for t in TOOL_SPECS],
            # When the dashboard was launched from a VSCode terminal,
            # the bridge port is inherited and every tool can route
            # LLM calls through `vscode.lm.*`. The client uses this
            # flag to surface `vscode_extension` in the provider
            # picker (it is intentionally absent from the interview's
            # PROVIDER_CHOICES list).
            "vscode_bridge_available": bridge_live,
            # Source from the canonical ensemble trio so the picker
            # never drifts from what the `full` ensemble profile
            # would actually fan out across. Both lists were
            # hardcoded separately before; that drift gets users
            # who pick a single model and then opt into the
            # ensemble silently routed across a different set.
            "vscode_extension_models": _vscode_extension_model_choices(),
        })

    @app.get("/api/provider/models")
    async def provider_models(provider: str = "") -> JSONResponse:
        """Live model-list discovery for the picker. Replies with
        either {source:"dynamic", models:[...]} when a real endpoint
        (OpenAI /v1/models, Ollama /api/tags, the VSCode persistent
        bridge) answered, or {source:"static"} which signals the UI
        to fall through to the schema's curated list.

        5-minute in-memory cache keyed on provider name so the picker
        can re-fetch on every change without hammering the upstream."""
        return await _provider_models_cached(app, (provider or "").strip())

    @app.post("/api/tools/{tool_name}")
    async def run_tool(tool_name: str, request: Request) -> JSONResponse:
        spec = TOOLS_BY_NAME.get(tool_name)
        if spec is None:
            raise HTTPException(404, f"unknown tool: {tool_name}")

        # Bypass FastAPI's form-parser when the content-type is JSON
        # so power-users + tests can POST the same fields as JSON
        # without dealing with multipart boundaries.
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("application/json"):
            payload = await request.json()
            uploaded_paths: list[Path] = []
        else:
            form = await request.form()
            payload = {
                k: (v if not hasattr(v, "filename") else None)
                for k, v in form.items()
            }
            uploaded_paths = await _stage_uploads(form, output_root, spec.name)

        # Mirror the interview's submit-time guard: refuse
        # `vscode_extension` when neither bridge transport is wired.
        # The IPC socket is the default path; the TCP port is kept
        # for the chat-spawn case where the extension passes it.
        picked_provider = (payload.get("provider") or "").strip()
        bridge_port = int(getattr(app.state, "vscode_bridge_port", 0) or 0)
        bridge_socket = getattr(app.state, "vscode_bridge_socket", "") or ""
        if picked_provider == "vscode_extension" and bridge_port <= 0 and not bridge_socket:
            raise HTTPException(
                400,
                "vscode_extension provider requires a live VSCode bridge — "
                "neither --vscode-bridge-socket (the per-user IPC default) "
                "nor --vscode-bridge-port is wired. Either launch VSCode "
                "with the FI extension active (so the PersistentBridge is "
                "available) or pick a different provider.",
            )

        try:
            argv_tail = _build_argv(spec, payload, uploaded_paths, output_root)
        except ValueError as e:
            raise HTTPException(400, str(e))

        from web.quest_launcher import QuestLauncherFull
        job_id = f"{spec.name}-{int(time.time())}"
        try:
            launched = app.state.launcher.launch_command(
                argv_tail=argv_tail, job_id=job_id,
            )
        except QuestLauncherFull as e:
            return JSONResponse(
                {"error": "launcher at capacity",
                 "detail": str(e),
                 "retry_after_seconds": 30},
                status_code=503,
                headers={"Retry-After": "30"},
            )

        return JSONResponse({
            "tool": spec.name,
            "job_id": launched.quest_id,
            "pid": launched.pid,
            "argv_tail": argv_tail,
            "artifact_dir": str(output_root / spec.artifact_dir)
            if spec.artifact_dir else None,
        })


# Per-profile fan-out scope. The interview's
# ``core.interview.expand_ensemble_profile`` plus
# ``ensemble_model_trios`` already define the canonical mapping; this
# is the same mapping spelled out for the standalone-tool case where
# the engine isn't running and we just need an explicit CSV for the
# ``--{tool}-ensemble`` flag.
_ENSEMBLE_PROFILE_TO_TRIO_KEY: dict[str, str] = {
    "off": "",  # no fan-out
    "cross_check_only": "trio",
    "ideate_and_check": "trio",
    "full": "trio",
}


def _ensemble_argv_tail(
    tool: str, provider: str, payload: dict[str, Any],
) -> list[str]:
    """Build the ``--{tool}-ensemble M1,M2,M3 --{tool}-ensemble-merge X``
    tail for the spawned subprocess based on the picked profile +
    provider. Returns ``[]`` for ``off`` / missing / unknown profile.

    Cross-references the canonical trios at
    ``core.interview._ENSEMBLE_MODEL_TRIOS`` so the standalone tool
    fan-out matches the engine's node-ensemble fan-out for the same
    provider + profile.
    """
    profile = (payload.get("ensemble_profile") or "off").strip()
    if profile == "off" or profile not in _ENSEMBLE_PROFILE_TO_TRIO_KEY:
        return []
    if not provider:
        # No provider picked — fan-out has nowhere to fall back to.
        # Caller will surface this as a UI prompt rather than silently
        # dropping the profile.
        return []
    try:
        from core.interview import ensemble_model_trio
    except Exception:
        return []
    trio = ensemble_model_trio(provider, payload.get("provider_model"))
    if not trio:
        return []
    # The CLI flag accepts the same CSV the engine YAML accepts.
    csv = ",".join(trio)
    # Per-tool default merge mirrors the launch.py argparse defaults
    # exactly so behaviour doesn't drift between CLI and serve users:
    #   proposal: tournament — one canonical plan.
    #   critique: synthesize — preserve adversarial disagreement.
    #   digest:   synthesize — preserve WeekDiff disagreement.
    #   portfolio/summarize/analyze: tournament — one canonical synthesis.
    _MERGE_DEFAULTS = {
        "proposal":  "tournament",
        "critique":  "synthesize",
        "digest":    "synthesize",
        "portfolio": "tournament",
        "summarize": "tournament",
        "analyze":   "tournament",
    }
    merge = _MERGE_DEFAULTS.get(tool, "tournament")
    return [f"--{tool}-ensemble", csv, f"--{tool}-ensemble-merge", merge]


# Per-provider in-memory cache for ``/api/provider/models``. Keyed on
# provider name; entries expire after _PROVIDER_MODELS_TTL_S so a
# freshly-installed model (``ollama pull`` etc.) doesn't take an hour
# to show up in the picker. Cache lives at module scope rather than
# on app.state so test clients share it within the test process —
# call _provider_models_cache_clear() in setup if you need a fresh
# cache.
_PROVIDER_MODELS_TTL_S = 300
_provider_models_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _provider_models_cache_clear() -> None:
    _provider_models_cache.clear()


# Per-call rather than module-constant so reloading the schema (e.g.
# a test that bumps the trio) picks up changes without reimporting
# the routes module.
def _vscode_extension_model_choices() -> list[dict[str, str]]:
    """Render the picker's vscode_extension model list from the
    canonical ensemble trio in ``core.interview``. Keeps the
    single-model dropdown values in lock-step with what the `full`
    ensemble profile would fan out across, so a user who pre-picks
    one model and then enables ensemble doesn't get silently routed
    through a different three."""
    try:
        from core.interview import ensemble_model_trio
        trio = ensemble_model_trio("vscode_extension") or []
    except Exception:
        trio = []
    return [
        {"value": m, "label": m,
         "description": "From the vscode_extension ensemble trio."}
        for m in trio
    ]


async def _provider_models_cached(app: FastAPI, provider: str) -> JSONResponse:
    """Dispatch helper shared by /api/provider/models. Holds the
    cache off the request handler so it can be cleared from tests
    without touching the endpoint's signature."""
    if not provider:
        return JSONResponse(
            {"source": "static", "provider": "", "models": []},
        )
    now = time.time()
    hit = _provider_models_cache.get(provider)
    if hit and (now - hit[0]) < _PROVIDER_MODELS_TTL_S:
        return JSONResponse(hit[1])
    from core.provider_models_discover import discover
    bridge_port = int(getattr(app.state, "vscode_bridge_port", 0) or 0)
    bridge_socket = getattr(app.state, "vscode_bridge_socket", "") or ""
    models = await discover(
        provider,
        bridge_socket=bridge_socket,
        bridge_port=bridge_port,
    )
    if models is None:
        body = {"source": "static", "provider": provider, "models": []}
    else:
        body = {"source": "dynamic", "provider": provider, "models": models}
    _provider_models_cache[provider] = (now, body)
    return JSONResponse(body)


async def _stage_uploads(form: Any, output_root: Path, tool_name: str) -> list[Path]:
    """Pull file uploads from the form. Files land in
    ``outputs/_uploads/<tool>-<ts>/<filename>``. Returns the list of
    written paths so the argv builder can reference them."""
    uploaded: list[Path] = []
    upload_dir = output_root / "_uploads" / f"{tool_name}-{int(time.time())}"
    for key, value in form.multi_items() if hasattr(form, "multi_items") else form.items():
        if hasattr(value, "filename") and value.filename:
            upload_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(value.filename).name  # strip any path components
            target = upload_dir / safe_name
            content = await value.read()
            target.write_bytes(content)
            uploaded.append(target)
    return uploaded


def _build_argv(
    spec: ToolSpec,
    payload: dict[str, Any],
    uploaded_paths: list[Path],
    output_root: Path,
) -> list[str]:
    """Convert the form payload + uploaded files into the argv tail
    that launch.py expects. Each spec is hand-coded because the
    CLI shapes differ; this is the place to add a new tool.

    Per-tool provider/model are appended at the end via the
    ``--<tool>-provider`` flag launch.py already supports. The
    model goes into a fresh ``--<tool>-model`` flag we don't
    actually take (the engine reads provider.model from the
    Config); instead, we set it in the spawned subprocess's env
    via ``FI_PROVIDER_MODEL_OVERRIDE`` if we add such a path
    later. For now, the model dropdown is a UX hint — the picked
    provider is honored; the model field is captured but doesn't
    flow into argv yet. (Engine-wide per-call model overrides
    require an env-var pathway in the provider layer that's
    deferred to a follow-up.)"""
    name = spec.name
    # Common provider tail for LLM-using tools. Each PM-command
    # accepts --<tool>-provider with the standard ProviderName
    # values (openai / codex / claude_cli / etc.).
    provider = (payload.get("provider") or "").strip()
    provider_tail: list[str] = []
    if spec.needs_llm and provider:
        flag = f"--{spec.name}-provider"
        provider_tail = [flag, provider]

    # Ensemble tail: --<tool>-ensemble takes a CSV of models, not a
    # profile name. Expand the profile picked in the UI into the
    # provider's curated 3-model trio (mirrors what the interview's
    # YAML emitter does for engine-side node_ensemble). Only the two
    # Every LLM tool accepts --<tool>-ensemble now (single-shot fan-out
    # for digest/portfolio/summarize/critique/proposal; engine-level
    # node_ensemble for analyze). Map the profile picker to the CSV
    # the CLI expects.
    _ENSEMBLE_TOOLS = {"proposal", "critique", "digest", "portfolio",
                       "summarize", "analyze"}
    ensemble_tail: list[str] = []
    if spec.needs_llm and name in _ENSEMBLE_TOOLS:
        ensemble_tail = _ensemble_argv_tail(name, provider, payload)

    if name == "proposal":
        topic = (payload.get("topic") or "").strip()
        if not topic:
            raise ValueError("topic is required")
        return ["--proposal", topic, *provider_tail, *ensemble_tail]

    if name == "critique":
        quest_id = (payload.get("quest_id") or "").strip()
        if not quest_id:
            raise ValueError("quest_id is required")
        return ["--critique", quest_id, *provider_tail, *ensemble_tail]

    if name == "digest":
        days = int(payload.get("days") or 7)
        return ["--digest", "--days", str(days), *provider_tail, *ensemble_tail]

    if name == "portfolio":
        return ["--portfolio", *provider_tail, *ensemble_tail]

    if name == "summarize":
        folder = (payload.get("folder") or "").strip()
        if uploaded_paths:
            folder = str(uploaded_paths[0].parent)
        if not folder:
            raise ValueError("folder path or uploaded files required")
        argv = ["--summarize", folder]
        kind = (payload.get("kind") or "").strip()
        if kind and kind != "auto":
            argv.extend(["--summarize-kind", kind])
        argv.extend(provider_tail)
        argv.extend(ensemble_tail)
        return argv

    if name == "analyze":
        path = (payload.get("path") or "").strip()
        if uploaded_paths:
            path = str(uploaded_paths[0].parent)
        if not path:
            raise ValueError("data folder path or uploaded files required")
        topic = (payload.get("topic") or "").strip()
        if not topic:
            raise ValueError("analysis topic is required")
        return ["--analyze", path, "--analyze-topic", topic, *provider_tail, *ensemble_tail]

    if name == "fleet":
        yaml_paths_raw = (payload.get("yaml_paths") or "").strip()
        yaml_paths = [p.strip() for p in yaml_paths_raw.splitlines() if p.strip()]
        for p in uploaded_paths:
            yaml_paths.append(str(p))
        if not yaml_paths:
            raise ValueError("at least one YAML path required (or upload)")
        return ["--fleet", *yaml_paths]

    if name == "ingest":
        paths_raw = (payload.get("paths") or "").strip()
        paths = [p.strip() for p in paths_raw.splitlines() if p.strip()]
        for p in uploaded_paths:
            paths.append(str(p))
        if not paths:
            raise ValueError("at least one file path required (or upload)")
        return ["--ingest", *paths]

    raise ValueError(f"no argv builder for tool {name!r}")
