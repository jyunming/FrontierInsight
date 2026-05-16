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
        return JSONResponse({"tools": [_spec_to_dict(t) for t in TOOL_SPECS]})

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
    CLI shapes differ; this is the place to add a new tool."""
    name = spec.name

    if name == "proposal":
        topic = (payload.get("topic") or "").strip()
        if not topic:
            raise ValueError("topic is required")
        return ["--proposal", topic]

    if name == "critique":
        quest_id = (payload.get("quest_id") or "").strip()
        if not quest_id:
            raise ValueError("quest_id is required")
        return ["--critique", quest_id]

    if name == "digest":
        days = int(payload.get("days") or 7)
        return ["--digest", "--days", str(days)]

    if name == "portfolio":
        return ["--portfolio"]

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
        return ["--analyze", path, "--analyze-topic", topic]

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
