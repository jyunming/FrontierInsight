"""FastAPI routes for the ``--serve`` web UI's interview surface.

Implements the same question set as the CLI ``--new`` and the
VSCode ``@fi /new`` flows, backed by ``core.interview`` as the
single source of truth. Two surfaces:

* ``GET  /interview``                — HTML form for new-quest setup.
* ``GET  /update/{quest_id}``        — HTML form pre-filled with the
                                        quest's current editable answers.
* ``GET  /api/interview/schema``     — JSON schema (forwards
                                        ``core/interview_schema.json``).
* ``POST /api/interview/submit``     — Accepts answers, writes YAML,
                                        returns the YAML path.
* ``POST /api/interview/update/{id}``— Same shape; runs the
                                        mid-quest update flow.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from core.interview import (
    InterviewAnswers, answers_to_yaml, export_schema_json, slugify,
)


_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"


def register_interview_routes(app: FastAPI, output_root: Path) -> None:
    """Attach interview routes onto an existing FastAPI app.

    ``output_root`` is the same path the rest of the web UI uses;
    new-quest YAMLs land at ``output_root / "_drafts" / "<stamp>-<slug>.yaml"``,
    update flows mutate ``output_root / <quest_id> / config.yaml``.
    """

    @app.get("/interview", response_class=HTMLResponse)
    async def interview_page() -> HTMLResponse:
        page = _STATIC / "interview.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Interview UI not installed</h1>", status_code=500,
            )
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/update/{quest_id}", response_class=HTMLResponse)
    async def update_page(quest_id: str) -> HTMLResponse:
        """Same form template; the JS reads ``?quest_id=`` and pre-fills
        editable fields by fetching the quest's current YAML."""
        page = _STATIC / "interview.html"
        if not page.exists():
            return HTMLResponse(
                "<h1>Interview UI not installed</h1>", status_code=500,
            )
        html = page.read_text(encoding="utf-8")
        # The static page reads ?quest_id=… from window.location.search.
        # We just redirect-ish by injecting a script tag — keeps the
        # static file simple.
        injected = html.replace(
            "</head>",
            f"<script>window.__fi_update_quest_id = {json.dumps(quest_id)};</script></head>",
            1,
        )
        return HTMLResponse(injected)

    @app.get("/api/interview/schema")
    async def get_schema() -> JSONResponse:
        return JSONResponse(export_schema_json())

    @app.post("/api/interview/submit")
    async def submit_new(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            answers = _parse_answers(body)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, f"invalid answers payload: {e}")

        yaml_text = answers_to_yaml(answers, frontend="serve")
        drafts = output_root / "_drafts"
        drafts.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d-%H%M")
        yaml_path = drafts / f"{stamp}-{slugify(answers.title) or 'quest'}.yaml"
        yaml_path.write_text(yaml_text, encoding="utf-8")
        return JSONResponse({
            "yaml_path": str(yaml_path),
            "draft_only": True,
            "next_step": (
                f"Run `python launch.py --config {yaml_path}` to start "
                f"the quest, or PATCH this endpoint to launch in-process."
            ),
        })

    @app.post("/api/interview/update/{quest_id}")
    async def submit_update(quest_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        from core.interview_update import (
            load_current_answers, diff_answers, compute_invalidated_stages,
            keys_to_clear, rewrite_yaml_with_new_answers,
            soft_invalidate_checkpoint,
        )

        # Reuse the same quest_id allowlist + relative_to guard the
        # rest of the web UI uses, so a payload like
        # ``{"quest_id": "../somewhere"}`` can't escape ``output_root``
        # and write a config.yaml to an arbitrary path. Imported
        # lazily to avoid a circular dependency at module load.
        from web.server import _resolve_quest_root
        quest_root = _resolve_quest_root(output_root, quest_id)
        if not quest_root.is_dir():
            raise HTTPException(404, f"no quest directory at {quest_root}")

        try:
            current, yaml_path, raw = load_current_answers(quest_root)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, str(e))

        try:
            new_answers = _parse_answers(body)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, f"invalid answers payload: {e}")

        # Non-editable fields are pinned to the current values
        # regardless of what the client sent — the web UI form
        # should mark them disabled, but be defensive.
        new = InterviewAnswers(
            topic=current.topic,
            title=current.title,
            output_kinds=new_answers.output_kinds,
            paper_format=new_answers.paper_format,
            no_simulation=current.no_simulation,
            study_depth=new_answers.study_depth,
            comparative_baseline=new_answers.comparative_baseline,
            success_metric=new_answers.success_metric,
            budget=new_answers.budget,
            clarify_mode=new_answers.clarify_mode,
            review_panel=new_answers.review_panel,
            knowledge_enabled=new_answers.knowledge_enabled,
            provider=current.provider,
            provider_model=current.provider_model,
        )
        changes = diff_answers(current, new)
        stages = compute_invalidated_stages(changes)
        keys = keys_to_clear(stages)

        result: dict[str, Any] = {
            "quest_id": quest_id,
            "changes": {k: list(v) for k, v in changes.items()},
            "invalidated_stages": stages,
            "keys_cleared": [],
        }
        if changes:
            new_yaml = rewrite_yaml_with_new_answers(raw, new)
            backup = yaml_path.with_suffix(".yaml.before-update")
            backup.write_text(
                yaml_path.read_text(encoding="utf-8"), encoding="utf-8",
            )
            yaml_path.write_text(new_yaml, encoding="utf-8")
            result["yaml_path"] = str(yaml_path)
            result["backup_path"] = str(backup)
            if keys:
                ok = await soft_invalidate_checkpoint(quest_root, keys)
                if ok:
                    result["keys_cleared"] = keys
        return JSONResponse(result)


def _parse_answers(body: dict[str, Any]) -> InterviewAnswers:
    """Validate-and-coerce a JSON answers payload into
    :class:`InterviewAnswers`. Strict per-field type checks: the
    naïve ``bool(...)`` / ``list(...)`` coercion lets ``"false"`` get
    interpreted as ``True`` and ``"abc"`` get split into ``['a','b','c']``
    — both regressions a hostile or buggy client could exploit to
    write invalid YAML. Raises ``KeyError`` / ``TypeError`` /
    ``ValueError`` on missing or malformed fields — the caller maps
    those to a 400 response."""
    required = (
        ("topic", str),
        ("title", str),
        ("output_kinds", list),
        ("paper_format", str),
        ("no_simulation", bool),
        ("study_depth", str),
        ("comparative_baseline", str),
        ("success_metric", str),
        ("budget", str),
        ("clarify_mode", str),
        ("review_panel", list),
        ("knowledge_enabled", bool),
        ("provider", str),
    )
    for field, expected in required:
        if field not in body:
            raise KeyError(field)
        value = body[field]
        if not isinstance(value, expected):
            raise TypeError(
                f"field {field!r} must be {expected.__name__}, got {type(value).__name__}"
            )
    # output_kinds + review_panel are lists of strings — verify each
    # element so a junk payload like ``output_kinds: [1, 2, 3]``
    # doesn't get coerced silently.
    for field in ("output_kinds", "review_panel"):
        for i, item in enumerate(body[field]):
            if not isinstance(item, str):
                raise TypeError(
                    f"{field}[{i}] must be str, got {type(item).__name__}"
                )
    # provider_model can be ``""`` / ``None`` / a string; the dataclass
    # field is optional. Anything else is malformed.
    pm = body.get("provider_model")
    if pm is not None and not isinstance(pm, str):
        raise TypeError(
            f"provider_model must be str or null, got {type(pm).__name__}"
        )
    return InterviewAnswers(
        topic=body["topic"],
        title=body["title"],
        output_kinds=list(body["output_kinds"]),
        paper_format=body["paper_format"],
        no_simulation=body["no_simulation"],
        study_depth=body["study_depth"],
        comparative_baseline=body["comparative_baseline"],
        success_metric=body["success_metric"],
        budget=body["budget"],
        clarify_mode=body["clarify_mode"],
        review_panel=list(body["review_panel"]),
        knowledge_enabled=body["knowledge_enabled"],
        provider=body["provider"],
        provider_model=pm if pm else None,
    )
