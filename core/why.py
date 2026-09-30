"""Why did the quest do that? One answer, read from what the quest already recorded.

``python launch.py --why <quest_id>`` (web: the quest page's Why?, VS Code: ``@fi /why <quest_id>``) answers four
questions, without asking a model:

* **why it stopped** — the pause it is waiting at (``.fi/pause.json``, ``NEXT_STEP.md``);
* **why a revision was asked for** — the review's verdict and what forced it, with the reviewer's own reasons;
* **why the evidence is at this level** — what the result shows and what stands in the way of more (``EVIDENCE.json``);
* **why any step decided what it did** (``--node <step>``) — the route it chose and the facts it read, the checks it ran,
  and the model's own reasons, from the audit trace (``.fi/audit.jsonl``), with whether ``.fi/thinking.jsonl`` holds the
  model's reasoning for that step and how long it is (the text stays in the file);
* **every step's reasons** (``reasons``) — the reasons the model gave at each step, step by step.

With nothing named, the answer is the pause (when the quest is paused), the latest review (when one ran) and the evidence.
What the engine measured and what a model said about itself are kept apart: a model's reason is marked as one, and a
stated reason or a reasoning summary is the model's own account, not its hidden reasoning.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import attempt_records as _attempts
from . import audit_log
from .thinking_capture import THINKING_FILE

# Aliases a person may type for the questions; any other word is taken as a step (node) name.
TOPICS = {
    "stop": "pause", "stopped": "pause", "pause": "pause", "paused": "pause",
    "revise": "review", "revision": "review", "review": "review",
    "evidence": "evidence", "level": "evidence",
    "reasons": "reasons", "reason": "reasons", "decisions": "reasons", "choices": "reasons",
}
_MAX_REASONS = 12
_PER_STEP = 5
#: Said wherever a model's reasons or its reasoning record are shown.
OWN_ACCOUNT = ("  (A reason the model wrote, and a reasoning summary it returned, are its own account of itself: not its "
               "hidden reasoning, and not something FI checked.)")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pause(root: Path) -> list[str]:
    descriptor = _json(root / ".fi" / "pause.json")
    card = root / "NEXT_STEP.md"
    if not isinstance(descriptor, dict) and not card.is_file():
        return []
    lines = ["Why it stopped:"]
    if isinstance(descriptor, dict):
        lines.append(f"  It is waiting for you: {descriptor.get('headline') or descriptor.get('kind')}.")
    if card.is_file():
        # The card's own sections, without its title and its commands.
        body = card.read_text(encoding="utf-8").split("\n## Then", 1)[0]
        for line in body.splitlines()[1:]:
            if line.strip() and not line.startswith("Quest **"):
                lines.append("  " + line.replace("## ", "").rstrip())
    return lines


def _passes(events: list[dict[str, Any]], node: str) -> list[list[dict[str, Any]]]:
    """The events of each run of ``node``: from its ``node_started`` to the next one."""
    runs: list[list[dict[str, Any]]] = []
    for e in events:
        if e.get("node") != node:
            continue
        if e.get("kind") == "node_started" or not runs:
            runs.append([])
        runs[-1].append(e)
    return runs


def _claim(e: dict[str, Any]) -> str:
    """A model's stated reason, from the event's own fields (its text may hold a colon of its own)."""
    text = str(e.get("claim") or "")
    if e.get("decision"):
        text += f" -> {e['decision']}"
    if e.get("reason"):
        text += f" ({e['reason']})"
    return text


def _epoch(ts: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except (TypeError, ValueError):
        return None


def _belongs(name: str, node: str, steps: set[str]) -> bool:
    """A call filed under ``name`` was made by step ``node``: its own name, or one that starts with it
    (``review_moderator`` under ``review``) and is not another step's (``execute_reflect`` is its own step, not
    ``execute``'s)."""
    def under(n: str, s: str) -> bool:
        return n == s or n.startswith(s + "_") or n.startswith(s + ".")

    if not under(name, node):
        return False
    return not any(len(s) > len(node) and under(name, s) for s in steps if s != node)


def _thinking_line(root: Path, node: str, run: list[dict[str, Any]], steps: set[str] | None = None) -> str:
    """Whether ``.fi/thinking.jsonl`` holds the model's reasoning for this run of the step, and how long it is. The text
    itself stays in the file."""
    path = root / ".fi" / THINKING_FILE
    if not path.is_file():
        return (f"  The model's reasoning record: none (.fi/{THINKING_FILE} does not exist: the connection returned no "
                "reasoning, or keeping it is off).")
    times = [t for t in (_epoch(e.get("ts")) for e in run) if t is not None]
    ended = any(e.get("kind") in ("node_completed", "node_failed", "node_paused") for e in run)
    start, end = (min(times), max(times) if ended else None) if times else (None, None)
    rows = []
    for r in _attempts.read(root / ".fi", THINKING_FILE):
        if not _belongs(str(r.get("node") or ""), node, steps or set()):
            continue
        at = r.get("at")
        if start is not None and isinstance(at, (int, float)) and (at < start - 1 or (end is not None and at > end + 1)):
            continue
        rows.append(r)
    which = "this run of the step" if start is not None else "this step (the times of its runs could not be read)"
    if not rows:
        return f"  The model's reasoning record: none for {which} in .fi/{THINKING_FILE}."
    chars = sum(len(str(r.get("thinking") or "")) for r in rows)
    return (f"  The model's reasoning record: {len(rows)} call(s) of {which}, {chars:,} characters in all, in "
            f".fi/{THINKING_FILE} (open the file to read it).")


def _steps(events: list[dict[str, Any]]) -> list[str]:
    """The steps the trace has, in the order they first ran."""
    order: list[str] = []
    for e in events:
        n = e.get("node")
        if e.get("kind") == "node_started" and n and n not in order:
            order.append(str(n))
    return order


def _step(events: list[dict[str, Any]], node: str, heading: str, root: Path | None = None) -> list[str]:
    runs = _passes(events, node)
    if not runs:
        return [f"{heading}: the trace has nothing from the {node} step."]
    last = runs[-1]
    lines = [f"{heading} (the latest of {len(runs)} run(s) of the {node} step):"]
    for e in last:
        if e.get("kind") == "route_decision":
            lines.append("  Decided: " + audit_log.describe(e, tagged=False).split(": ", 1)[-1] + ".")
    for e in last:
        if e.get("kind") == "check_result":
            lines.append(f"  Check {e.get('check')}: {e.get('status')}"
                         + (f" - {e.get('summary')}" if e.get("summary") else ""))
            lines += [f"    - {p}" for p in (e.get("problems") or [])[:5]]
    for e in last:
        if e.get("kind") == "model_changed":
            lines.append("  " + audit_log.describe(e, tagged=False).split(": ", 1)[-1] + ".")
    claims = [e for e in last if e.get("kind") == "model_claim"]
    if claims:
        lines.append("  The model's own reasons (what it said, not something FI checked):")
        lines += [f"    - {_claim(e)}" for e in claims[:_MAX_REASONS]]
        if len(claims) > _MAX_REASONS:
            lines.append(f"    (and {len(claims) - _MAX_REASONS} more: `--trace <quest_id> --trace-node {node}`)")
    if root is not None:
        lines.append(_thinking_line(root, node, last, set(_steps(events))))
    failed = [e for e in last if e.get("kind") == "node_failed"]
    lines += [f"  It failed: {e.get('error', '')}" for e in failed]
    if root is not None:
        lines.append(OWN_ACCOUNT)
    return lines


def _reasons(root: Path, events: list[dict[str, Any]]) -> list[str]:
    """Every step's stated reasons from its latest run, in the order the steps first ran, with the reasoning note."""
    order = _steps(events)
    lines = ["The model's reasons, step by step (the latest run of each step):"]
    shown = 0
    for node in order:
        last = _passes(events, node)[-1]
        claims = [e for e in last if e.get("kind") == "model_claim"]
        thinking = _thinking_line(root, node, last, set(order))
        if not claims and ": none" in thinking:
            continue  # no reason and no reasoning record: nothing the model said about this step
        shown += 1
        lines.append(f"  {node}:")
        lines += [f"    - {_claim(e)}" for e in claims[:_PER_STEP]]
        if len(claims) > _PER_STEP:
            lines.append(f"    (and {len(claims) - _PER_STEP} more: `--why <quest_id> {node}`)")
        lines.append("  " + thinking)  # under the step, beside its reasons
    if not shown:
        lines.append("  No step has recorded a reason the model gave yet.")
    lines.append(OWN_ACCOUNT)
    return lines


def _evidence(root: Path) -> list[str]:
    from .evidence import read as _read_evidence  # with its seal checked, as every surface shows it

    record = _read_evidence(root)
    if not isinstance(record, dict):
        return ["Why the evidence is at this level: the quest has not assessed its evidence yet."]
    from .evidence import summary_line  # the level in words, never its identifier

    lines = ["Why the evidence is at this level:", "  " + summary_line(record)]
    gaps = record.get("gaps") or []
    if len(gaps) > 1:
        lines.append("  Everything in the way of the next level:")
        lines += [f"    - {g}" for g in gaps]
    return lines


def explain(quest_root: Path, about: str = "") -> str:
    """The answer, as plain text. ``about`` is one of :data:`TOPICS`' words or a step (node) name; empty gives the
    pause (when paused), the latest review (when one ran) and the evidence."""
    root = Path(quest_root)
    events = audit_log.read(root / ".fi" / "audit.jsonl")
    topic = TOPICS.get(about.strip().lower(), about.strip()) if about else ""
    parts: list[list[str]] = []
    if topic == "pause":
        parts.append(_pause(root) or ["Why it stopped: it is not waiting for you."])
    elif topic == "review":
        parts.append(_step(events, "review", "Why the review decided what it did", root))
    elif topic == "evidence":
        parts.append(_evidence(root))
    elif topic == "reasons":
        parts.append(_reasons(root, events))
    elif topic:
        parts.append(_step(events, topic, f"Why the {topic} step decided what it did", root))
    else:
        parts.append(_pause(root))
        if _passes(events, "review"):
            parts.append(_step(events, "review", "Why the review decided what it did", root))
        parts.append(_evidence(root))
        parts.append(["Every step's reasons: `--why <quest_id> reasons`."])
    return "\n\n".join("\n".join(p) for p in parts if p)
