"""Vendor-parity probe for the clarify interview slot contract.

Runs ``Engine._node_clarify`` (clarify_mode=auto) against the same
research topic on each non-VSCode chat-capable provider, then asserts
the returned ``clarify_answers`` satisfies the documented contract:

* Required slots: ``comparative_baseline``, ``empirical_vs_theoretical``,
  ``simulatability``, ``success_metric``, ``budget``, ``output_kinds``,
  ``study_depth``, ``paper_venue``.
* ``simulatability.default`` (or the bare-string form in auto-mode
  reduction) MUST be one of ``yes`` / ``no`` / ``uncertain``.
* ``paper_venue`` MUST be one of the nine declared values
  (generic / neurips / iclr / ieee_access / nature_mi / essay /
  report / policy_brief / whitepaper).

Out of scope: ``copilot_cli`` (agentic, broken as a chat backend) and
``github_copilot_cli`` (third-party proxy with ToS risk). VSCode
extension covered by its own typescript round-trip test.

Run via ``python -m dev.scripts.test_clarify_parity`` from the repo
root. Writes a Markdown matrix to ``dev/scripts/clarify_parity_report.md``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import Engine
from core.provider import LLMClient, resolve_endpoint_async

REQUIRED_SLOTS = {
    "comparative_baseline",
    "empirical_vs_theoretical",
    "simulatability",
    "success_metric",
    "budget",
    "output_kinds",
    "study_depth",
    "paper_venue",
}
SIMULATABILITY_VALUES = {"yes", "no", "uncertain"}
PAPER_VENUE_VALUES = {
    "generic", "neurips", "iclr", "ieee_access", "nature_mi",
    "essay", "report", "policy_brief", "whitepaper",
}
STUDY_DEPTH_VALUES = {"brief preprint", "journal-length", "comprehensive review"}


def _flatten(answer: Any) -> str:
    """``_node_clarify`` auto-mode collapses each slot to its bare
    default. But callers may still pass the full ``{default, reason}``
    dict shape (e.g. in unit tests). Accept both shapes here."""
    if isinstance(answer, dict):
        return str(answer.get("default", "")).strip().lower()
    if isinstance(answer, str):
        return answer.strip().lower()
    if isinstance(answer, list):
        return ",".join(str(x).strip().lower() for x in answer)
    return str(answer).strip().lower()


def _validate_answers(
    answers: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)``. Errors fail the run; warnings are
    surfaced in the report but don't change the exit code. Keeping the
    two channels separate so a vendor synthesizing an adjacent
    ``study_depth`` label ("short report") doesn't fail a parity run
    designed to catch hard contract breaks like a missing slot."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = REQUIRED_SLOTS - set(answers.keys())
    if missing:
        errors.append(f"missing required slots: {sorted(missing)}")
    sim = _flatten(answers.get("simulatability"))
    if sim and sim not in SIMULATABILITY_VALUES:
        errors.append(
            f"simulatability.default={sim!r} not in {sorted(SIMULATABILITY_VALUES)}"
        )
    venue = _flatten(answers.get("paper_venue"))
    if venue and venue not in PAPER_VENUE_VALUES:
        errors.append(
            f"paper_venue.default={venue!r} not in {sorted(PAPER_VENUE_VALUES)}"
        )
    depth = _flatten(answers.get("study_depth"))
    # ``study_depth`` is documented as three canonical values but is
    # plain free-text in the schema, so synthesized adjacent labels
    # are flagged as warnings rather than errors.
    if depth and depth not in STUDY_DEPTH_VALUES:
        warnings.append(
            f"study_depth.default={depth!r} outside the documented set "
            f"{sorted(STUDY_DEPTH_VALUES)}"
        )
    return errors, warnings


async def run_one_vendor(
    provider_name: str, topic: str, out_dir: Path,
) -> dict[str, Any]:
    """Fire just the clarify node for one provider. Returns a dict
    describing the result (provider, ok, answers, validation_issues,
    raw_questions, elapsed_s, error)."""
    cfg = Config(
        topic=topic,
        title="parity-test",
        provider=ProviderConfig(name=provider_name),
        engine=EngineConfig(
            clarify_mode="auto",
            ideate_reflect=False,
            max_iterations=1,
            review_loop=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=out_dir),
    )
    eng = Engine(cfg)
    # ``_node_clarify`` calls ``self._chat`` which asserts
    # ``self._client is not None``. Engine normally builds the client
    # inside ``run()``; here we mirror that minimal setup so we can
    # invoke a single node without spinning up the full graph + venv +
    # sqlite checkpoint dance.
    endpoint = await resolve_endpoint_async(cfg.provider, eng.supervisor)
    eng._client = LLMClient(endpoint)
    state = {"topic": topic, "title": "parity-test"}
    started = time.monotonic()
    try:
        try:
            patch = await eng._node_clarify(state)
            elapsed = time.monotonic() - started
            answers = patch.get("clarify_answers") or {}
            questions = patch.get("clarify_questions") or {}
            errors, warnings = _validate_answers(answers)
            return {
                "provider": provider_name,
                "ok": True,
                "elapsed_s": round(elapsed, 1),
                "answers": answers,
                "questions": questions,
                "errors": errors,
                "warnings": warnings,
            }
        except Exception as exc:
            elapsed = time.monotonic() - started
            return {
                "provider": provider_name,
                "ok": False,
                "elapsed_s": round(elapsed, 1),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    finally:
        # Resource cleanup — Engine ``__init__`` registers a per-quest
        # FileHandler on the logger and ``LLMClient`` owns an
        # ``httpx.AsyncClient``. Without explicit close, file
        # descriptors leak (Windows then locks ``.fi/run.log`` and the
        # tmp_dir rmtree on the next run fails) and any proxy-backed
        # provider leaves its subprocess running.
        try:
            if eng._client is not None:
                await eng._client.aclose()
        except Exception:
            pass
        try:
            await eng.supervisor.shutdown()
        except Exception:
            pass
        try:
            from core.engine import _close_quest_logger
            _close_quest_logger(eng.quest_id)
        except Exception:
            pass


def _md_escape(text: str) -> str:
    """Escape Markdown control characters in user-supplied strings
    (notably ``**`` which would otherwise turn the rest of the line
    bold). Wraps in backticks instead of trying to inline-escape
    everything — readable AND fences all the syntactically-active
    characters in one go."""
    if not text:
        return ""
    # Pipes need escaping inside table cells; everything else is safe
    # inside a single backtick-fenced span.
    safe = text.replace("`", "'")  # collapse stray backticks
    safe = safe.replace("|", "\\|")  # pipes inside table cells
    return f"`{safe}`"


def render_report(results: list[dict[str, Any]], topic: str) -> str:
    """Build the Markdown matrix + per-vendor sections."""
    lines: list[str] = []
    lines.append("# Vendor parity — `clarify` slot contract")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_"
    )
    lines.append("")
    # Topic is fenced in backticks so embedded ``**`` (from
    # ``10**4``-style exponents) doesn't turn the rest of the line
    # bold — a real failure mode we hit on the default topic.
    lines.append(f"**Topic:** {_md_escape(topic)}")
    lines.append("")
    lines.append(
        "Excluded by policy: `copilot_cli` (agentic, broken as a chat "
        "backend) and `github_copilot_cli` (third-party proxy). VSCode "
        "extension parity is covered by `tests/test_vscode_extension_typescript.py`."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Provider | OK | Elapsed | Sim | Venue | Errors | Warnings |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        if not r.get("ok"):
            err_msg = r.get("error", "").splitlines()[0][:60] if r.get("error") else ""
            lines.append(
                f"| `{r['provider']}` | ❌ {err_msg} | "
                f"{r['elapsed_s']}s | — | — | — | — |"
            )
            continue
        ans = r.get("answers") or {}
        sim = _flatten(ans.get("simulatability"))
        venue = _flatten(ans.get("paper_venue"))
        errors = r.get("errors") or []
        warnings = r.get("warnings") or []
        if errors:
            status = "❌"
        elif warnings:
            status = "⚠️"
        else:
            status = "✅"
        err_cell = "; ".join(errors) if errors else "—"
        warn_cell = "; ".join(warnings) if warnings else "—"
        lines.append(
            f"| `{r['provider']}` | {status} | {r['elapsed_s']}s | "
            f"{sim or '—'} | {venue or '—'} | {err_cell} | {warn_cell} |"
        )
    lines.append("")
    lines.append("## Per-vendor clarify_answers")
    lines.append("")
    for r in results:
        lines.append(f"### `{r['provider']}`")
        lines.append("")
        if not r.get("ok"):
            lines.append("```")
            lines.append(r.get("traceback") or r.get("error", "(no error captured)"))
            lines.append("```")
            lines.append("")
            continue
        lines.append("```json")
        lines.append(json.dumps(r.get("answers") or {}, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        # ``10^4`` avoids Markdown's ``**``-as-bold collision in the
        # rendered report. ``10**4`` (the Python form) would otherwise
        # bold the remainder of the table line.
        "--topic", default=(
            "Compare three numerical integrators (RK4, Velocity-Verlet, "
            "forward Euler) on a damped harmonic oscillator. Report "
            "energy drift over 10^4 periods."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["claude_cli", "codex_cli", "gemini_cli"],
        help="Provider names to exercise. Defaults to the three CLI vendors.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "dev" / "scripts" / "clarify_parity_report.md"),
    )
    args = parser.parse_args()

    tmp_dir = REPO_ROOT / "dev" / "scripts" / "_parity_tmp"
    if tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
        except OSError as e:
            # Surface (not silence) — a leftover stale file from a prior
            # run with leaked handles would have masked a real
            # regression. ignore_errors=True was the original sin here.
            print(
                f"[parity] WARNING: could not fully clean {tmp_dir}: {e}",
                file=sys.stderr,
                flush=True,
            )
            # Best-effort partial cleanup so the run can still proceed.
            shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for provider in args.providers:
        print(f"[parity] running {provider} ...", flush=True)
        result = await run_one_vendor(provider, args.topic, tmp_dir / provider)
        if result["ok"]:
            n_err = len(result.get("errors") or [])
            n_warn = len(result.get("warnings") or [])
            print(
                f"[parity] {provider}: ok in {result['elapsed_s']}s; "
                f"errors={n_err} warnings={n_warn}",
                flush=True,
            )
        else:
            print(
                f"[parity] {provider}: FAIL in {result['elapsed_s']}s "
                f"({result.get('error')})",
                flush=True,
            )
        results.append(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(results, args.topic), encoding="utf-8")
    print(f"\n[parity] report -> {out_path}", flush=True)

    # Only hard errors flip the exit code; warnings (e.g. an off-spec
    # study_depth label) surface in the report but don't fail the run.
    n_failures = sum(1 for r in results if not r["ok"])
    n_with_errors = sum(1 for r in results if r.get("ok") and (r.get("errors") or []))
    return 0 if n_failures == 0 and n_with_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
