"""Mid-quest re-entry of the Frontier Insight interview.

``python launch.py --update <quest_id>`` (and the VSCode
`@fi /update <id>` shortcut) lands here. The flow:

  1. Load the quest's ``config.yaml`` from ``<quest_root>/config.yaml``.
  2. Build the current-answer dict from the YAML.
  3. Walk the interview question list, filtered to fields where
     ``Question.mid_quest_editable=True``. Each prompt is pre-filled
     with the current value.
  4. Compute the diff between old and new answers. Print a clear
     summary of what changed.
  5. Consult ``STAGE_INVALIDATION`` to decide which engine stages
     need to re-fire. Clear the corresponding ``QuestState`` keys
     from the latest LangGraph checkpoint via ``aupdate_state``
     (best-effort — see "Soft invalidation" caveat below).
  6. Write the updated YAML back to ``<quest_root>/config.yaml``.
  7. Hand off to the engine's existing ``--resume`` path
     (``run_one(cfg, resume_quest_id=quest_id, ...)``).

Soft invalidation caveat
------------------------

The implementation uses ``graph.aupdate_state`` to clear stage-output
fields from the latest checkpoint. This works when:

* The user is mid-quest at a known pause (e.g. ``wait_for_data``) —
  cleared fields haven't been written yet, so the resume picks up at
  the next node with the new YAML in effect.
* The user wants downstream-only effects (e.g. swapping
  ``paper_format`` mid-quest takes effect when ``write`` next runs;
  if ``write`` already ran, the cleared ``paper_md`` field causes
  ``write`` to re-fire on resume).

For a guaranteed re-run from the BEGINNING of a stage, delete
``<quest_root>/.fi/state.sqlite`` so LangGraph re-enters from
scratch with the new YAML. The ``--update`` output prints this
hint when a destructive invalidation is requested.
"""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml


# Quest IDs are slugified identifiers (digits, letters, ``_-.``).
# Rejecting anything outside this alphabet blocks path-separator-based
# traversal — a user-supplied id like ``../somewhere`` would otherwise
# resolve outside ``output_root`` and let --update read/write
# arbitrary paths on disk. Mirrors web/server.py:_QUEST_ID_RE.
_QUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")

from core.config import Config
from core.interview import (
    EDITABLE_FIELDS,
    InterviewAnswers,
    PAPER_FORMATS,
    QUESTIONS,
    STAGE_INVALIDATION,
    answers_to_yaml,
    build_smart_defaults,
)


# Which QuestState keys does each stage SET (so clearing them
# triggers a re-run when the graph next reaches that node)?
# Keep in sync with core/engine.py:QuestState — the relevant fields
# are documented inline at QuestState's definition.
STAGE_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "analyze": ("analysis",),
    "cross_check": ("cross_check",),
    "write": ("paper_md",),
    "review": ("review", "review_panel"),
}


# Fields the user can't flip mid-quest no matter what. ``no_simulation``
# is special: switching it would re-route the engine's graph (skip
# implement/execute vs run them), and the already-completed nodes
# can't be cleanly undone. The flow refuses with a clear message
# pointing the user at "re-run from scratch with --config" instead.
HARD_REFUSE_FIELDS: frozenset[str] = frozenset({"no_simulation"})


def load_current_answers(quest_root: Path) -> tuple[InterviewAnswers, Path, dict[str, Any]]:
    """Read ``<quest_root>/config.yaml`` and project it into the
    interview-answer shape. Returns ``(answers, yaml_path, raw_dict)``;
    the raw dict is kept so non-interview-managed keys round-trip back
    into the rewritten YAML untouched."""
    yaml_path = quest_root / "config.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"No config.yaml under {quest_root}. The quest may have been "
            f"started before FI started copying config.yaml into the "
            f"quest directory. Use `python launch.py --new` to set up "
            f"a fresh quest instead."
        )
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path} did not parse as a YAML mapping")

    engine = raw.get("engine") or {}
    overrides = engine.get("clarify_overrides") or {}
    output = raw.get("output") or {}
    provider = raw.get("provider") or {}
    knowledge = raw.get("knowledge") or {}

    topic = raw.get("topic", "") or ""
    if isinstance(topic, list):
        # Defensive — YAML literal blocks may parse to list-of-lines.
        topic = "\n".join(str(line) for line in topic)

    answers = InterviewAnswers(
        topic=str(topic).strip(),
        title=str(raw.get("title", "")),
        output_kinds=list(output.get("kinds") or []),
        paper_format=str(output.get("paper_format") or "generic"),
        no_simulation=bool(engine.get("no_simulation", False)),
        study_depth=str(overrides.get("study_depth") or "journal-length"),
        comparative_baseline=str(overrides.get("comparative_baseline") or ""),
        success_metric=str(overrides.get("success_metric") or ""),
        budget=str(overrides.get("budget") or ""),
        clarify_mode=str(engine.get("clarify_mode") or "auto"),
        review_panel=list(engine.get("review_panel") or []),
        knowledge_enabled=bool(knowledge.get("enabled", False)),
        provider=str(provider.get("name") or "vscode_extension"),
        provider_model=str(provider.get("model")) if provider.get("model") else None,
    )
    return answers, yaml_path, raw


def diff_answers(old: InterviewAnswers, new: InterviewAnswers) -> dict[str, tuple[Any, Any]]:
    """Return ``{field: (old, new)}`` for every editable field that
    changed. Non-editable fields are ignored (the interview shouldn't
    have asked for them, but be defensive)."""
    old_d = asdict(old)
    new_d = asdict(new)
    out: dict[str, tuple[Any, Any]] = {}
    for field in EDITABLE_FIELDS:
        if old_d.get(field) != new_d.get(field):
            out[field] = (old_d.get(field), new_d.get(field))
    return out


def compute_invalidated_stages(changes: dict[str, tuple[Any, Any]]) -> list[str]:
    """Look up each changed field in ``STAGE_INVALIDATION`` and union
    the affected stages. Stages are returned in pipeline order
    (analyze → cross_check → write → review) so the user sees a
    natural "earliest first" listing."""
    stage_order = ("analyze", "cross_check", "write", "review")
    stages: set[str] = set()
    for field in changes:
        for stage in STAGE_INVALIDATION.get(field, ()):
            stages.add(stage)
    return [s for s in stage_order if s in stages]


def keys_to_clear(stages: list[str]) -> list[str]:
    """For a list of invalidated stages, return the QuestState keys
    that should be cleared from the latest checkpoint."""
    out: list[str] = []
    for stage in stages:
        out.extend(STAGE_STATE_KEYS.get(stage, ()))
    return out


def rewrite_yaml_with_new_answers(
    raw: dict[str, Any], new: InterviewAnswers,
) -> str:
    """Produce the updated YAML text by re-emitting via
    :func:`answers_to_yaml`, but preserve any non-interview keys
    from the original (knowledge.local_papers, engine.node_models,
    etc.) that the interview doesn't manage. Detected non-interview
    keys are merged back as a trailing block.

    Keep the emission deterministic so the diff against the original
    YAML stays minimal — that's why we re-emit rather than splice
    edits into the raw dict in place.
    """
    # The interview already covers everything the YAML emitter
    # writes. Any keys present in `raw` but NOT written by
    # answers_to_yaml are preserved by deep-merging on top.
    base_yaml = answers_to_yaml(new, frontend="cli")
    base_data = yaml.safe_load(base_yaml) or {}

    # Deep-merge raw on top of base. Skip the keys we ARE managing
    # (provider, engine.clarify_mode, engine.no_simulation,
    # engine.review_panel, engine.clarify_overrides, output.kinds,
    # output.paper_format, knowledge.enabled, topic, title) because
    # we've already overwritten them.
    managed_paths = {
        ("topic",), ("title",),
        ("provider", "name"), ("provider", "model"),
        ("engine", "clarify_mode"), ("engine", "no_simulation"),
        ("engine", "review_panel"), ("engine", "clarify_overrides"),
        ("output", "kinds"), ("output", "paper_format"),
        ("knowledge", "enabled"),
    }

    def deep_merge(dst: dict[str, Any], src: dict[str, Any], prefix: tuple[str, ...] = ()) -> None:
        """For unmanaged paths, raw (the user's customized YAML) WINS
        over the interview's default emission. This preserves
        execution.timeout_s overrides, engine.node_models, custom
        knowledge.local_papers, and anything else the user hand-edited.
        Managed paths (interview-owned) are skipped — the new
        interview answers already populated them in ``dst``."""
        for k, v in (src or {}).items():
            path = prefix + (k,)
            if path in managed_paths:
                continue
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_merge(dst[k], v, path)
            else:
                dst[k] = v

    deep_merge(base_data, raw)
    return yaml.safe_dump(base_data, sort_keys=False, indent=2, allow_unicode=True)


async def soft_invalidate_checkpoint(
    quest_root: Path, keys_to_drop: list[str],
) -> bool:
    """Clear ``keys_to_drop`` from the latest LangGraph checkpoint
    in ``<quest_root>/.fi/state.sqlite``. Returns True on success,
    False if no checkpoint exists or the update failed.

    Best-effort: any exception is caught and logged via stderr; the
    caller treats failure as "no invalidation happened" and the user
    sees a clear hint."""
    sqlite_path = quest_root / ".fi" / "state.sqlite"
    if not sqlite_path.is_file():
        return False
    try:
        # Lazy-import LangGraph so importing this module is cheap.
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except Exception as e:
        print(
            f"[FI] --update: LangGraph checkpointer unavailable ({e!r}); "
            f"soft invalidation skipped. The new YAML still applies on resume.",
            file=sys.stderr,
        )
        return False

    config = {"configurable": {"thread_id": quest_root.name}}
    try:
        async with AsyncSqliteSaver.from_conn_string(str(sqlite_path)) as saver:
            current = await saver.aget_tuple(config)
            if current is None or current.checkpoint is None:
                return False
            channel_values = dict(current.checkpoint.get("channel_values") or {})
            cleared: list[str] = []
            for key in keys_to_drop:
                if key in channel_values:
                    channel_values.pop(key, None)
                    cleared.append(key)
            if not cleared:
                return False
            new_checkpoint = dict(current.checkpoint)
            new_checkpoint["channel_values"] = channel_values
            # Bump the id so the saver writes a new row (rather than
            # silently no-op'ing on duplicate primary key).
            from uuid import uuid4
            new_checkpoint["id"] = str(uuid4())
            await saver.aput(
                config,
                new_checkpoint,
                current.metadata or {},
                {},
            )
            print(
                f"[FI] --update: cleared {len(cleared)} state key(s) "
                f"from latest checkpoint: {','.join(cleared)}"
            )
            return True
    except Exception as e:  # pragma: no cover — defensive only.
        print(
            f"[FI] --update: soft-invalidation failed ({e!r}); the new "
            f"YAML still applies on resume. For a guaranteed re-run, "
            f"delete {sqlite_path} before re-resuming.",
            file=sys.stderr,
        )
        return False


async def run_update_flow(
    *,
    quest_id: str,
    output_root: Path,
    vscode_bridge_port: int,
    interactive: bool,
    supervisor: Any,
    run_one: Callable[..., Awaitable[Any]],
    apply_vscode_bridge_override: Callable[..., None],
) -> int:
    """Top-level entry from ``launch.py``. Validates the quest_id,
    runs the interview filtered to editable fields, performs soft
    invalidation, writes the updated YAML, then resumes the quest.
    """
    # Reject quest_id values that could escape the output root (path
    # traversal). Two-layer guard: regex allowlist + post-resolve
    # ``relative_to`` check. Mirrors web/server.py:_resolve_quest_root.
    if not _QUEST_ID_RE.match(quest_id):
        print(
            f"[FI] --update: bad quest_id format: {quest_id!r}. "
            f"Allowed characters: digits, letters, '_', '-', '.'.",
            file=sys.stderr,
        )
        return 1
    root = output_root.resolve()
    quest_root = (root / quest_id).resolve()
    try:
        quest_root.relative_to(root)
    except ValueError:
        print(
            f"[FI] --update: quest_id {quest_id!r} escapes output root "
            f"{root}.",
            file=sys.stderr,
        )
        return 1
    if not quest_root.is_dir():
        print(
            f"[FI] --update: no quest directory at {quest_root}. "
            f"List candidates with `ls {root}`.",
            file=sys.stderr,
        )
        return 1

    try:
        current, yaml_path, raw = load_current_answers(quest_root)
    except (FileNotFoundError, ValueError) as e:
        print(f"[FI] --update: {e}", file=sys.stderr)
        return 1

    print("=" * 72)
    print(f"Frontier Insight — update quest {quest_id}")
    print(f"Current YAML: {yaml_path}")
    print("Editable fields only. Press Ctrl-C to cancel.")
    print("=" * 72)

    partial: dict[str, Any] = asdict(current)
    new_partial = dict(partial)

    try:
        for q in QUESTIONS:
            if not q.mid_quest_editable:
                continue
            if q.id in HARD_REFUSE_FIELDS:
                continue
            # Pre-fill with current value.
            new_partial.setdefault(q.id, partial.get(q.id))
            from launch import _cli_prompt_for  # local import to avoid cycles
            # Inject current value as the "smart default" so the
            # prompt's "(default)" parenthesis shows it. _cli_prompt_for
            # consults SMART_DEFAULTS first, then falls back to
            # q.default; we override by reading from new_partial.
            # Cheapest path: temporarily swap q.default for this call.
            current_value = new_partial.get(q.id, q.default)
            tmp_q = type(q)(
                id=q.id, label=q.label, prompt=q.prompt, kind=q.kind,
                choices=q.choices, placeholder=q.placeholder,
                default=current_value, mid_quest_editable=q.mid_quest_editable,
                frontends=q.frontends, allow_other=q.allow_other,
            )
            ans = _cli_prompt_for(tmp_q, new_partial, {})
            if ans is None:
                print("\n— update cancelled.")
                return 1
            new_partial[q.id] = ans
    except (KeyboardInterrupt, EOFError):
        print("\n— update cancelled (Ctrl-C / EOF).")
        return 1

    # Build the new InterviewAnswers, copying through non-editable
    # fields verbatim (topic / title / provider / model / no_simulation).
    new = InterviewAnswers(
        topic=current.topic,
        title=current.title,
        output_kinds=list(new_partial["output_kinds"]),
        paper_format=str(new_partial["paper_format"]),
        no_simulation=current.no_simulation,  # frozen
        study_depth=str(new_partial["study_depth"]),
        comparative_baseline=str(new_partial["comparative_baseline"]),
        success_metric=str(new_partial["success_metric"]),
        budget=str(new_partial["budget"]),
        clarify_mode=str(new_partial["clarify_mode"]),
        review_panel=list(new_partial["review_panel"]),
        knowledge_enabled=bool(new_partial["knowledge_enabled"]),
        provider=current.provider,  # frozen
        provider_model=current.provider_model,  # frozen
    )

    changes = diff_answers(current, new)
    if not changes:
        print()
        print("No changes. Quest will resume with the existing YAML.")
    else:
        print()
        print("Changes detected:")
        for field, (old_val, new_val) in changes.items():
            print(f"  - {field}: {old_val!r} → {new_val!r}")

    stages = compute_invalidated_stages(changes)
    keys = keys_to_clear(stages)
    if stages:
        print()
        print(f"Affected stages: {', '.join(stages)}")
        print(f"State keys to clear: {', '.join(keys) if keys else '(none)'}")

    # Write the updated YAML back (only when changes actually happened).
    if changes:
        new_yaml_text = rewrite_yaml_with_new_answers(raw, new)
        # Backup the old YAML so a bad --update can be undone.
        backup = yaml_path.with_suffix(".yaml.before-update")
        backup.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
        yaml_path.write_text(new_yaml_text, encoding="utf-8")
        print(f"YAML updated: {yaml_path}  (backup: {backup.name})")

        if keys:
            ok = await soft_invalidate_checkpoint(quest_root, keys)
            if not ok:
                print(
                    "  Note: soft invalidation did not change the "
                    "checkpoint. For a guaranteed re-run of the affected "
                    f"stages, delete {quest_root / '.fi' / 'state.sqlite'} "
                    "before resuming."
                )

    # Resume the quest with the (possibly updated) YAML.
    print()
    print(f"Resuming quest {quest_id}...")
    cfg = Config.from_yaml(yaml_path)
    apply_vscode_bridge_override(cfg, vscode_bridge_port)
    try:
        await run_one(
            cfg,
            supervisor=supervisor,
            profile=False,
            interactive=interactive,
            resume_quest_id=quest_id,
            source_yaml_path=yaml_path.resolve(),
        )
        return 0
    except Exception as e:
        print(f"[FI] --update: resume failed: {e!r}", file=sys.stderr)
        return 1
