"""Why did the quest do that? One answer, read from what the quest already recorded.

``python launch.py --why <quest_id>`` (web: the quest page's Why?, VS Code: ``@fi /why <quest_id>``) answers four
questions, without asking a model:

* **why it stopped** — the pause it is waiting at (``.fi/pause.json``, ``NEXT_STEP.md``);
* **why a revision was asked for** — the review's verdict and what forced it, with the reviewer's own reasons;
* **why the evidence is at this level** — what the result shows and what stands in the way of more (``EVIDENCE.json``);
* **why any step decided what it did** (``--node <step>``) — the route it chose and the facts it read, the checks it ran,
  and the model's own reasons, from the audit trace (``.fi/audit.jsonl``).

With nothing named, the answer is the pause (when the quest is paused), the latest review (when one ran) and the evidence.
What the engine measured and what a model said about itself are kept apart: a model's reason is marked as one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import audit_log

# Aliases a person may type for the four questions; any other word is taken as a step (node) name.
TOPICS = {
    "stop": "pause", "stopped": "pause", "pause": "pause", "paused": "pause",
    "revise": "review", "revision": "review", "review": "review",
    "evidence": "evidence", "level": "evidence",
}
_MAX_REASONS = 8


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


def _step(events: list[dict[str, Any]], node: str, heading: str) -> list[str]:
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
    claims = [e for e in last if e.get("kind") == "model_claim"]
    if claims:
        lines.append("  The model's own reasons (what it said, not something FI checked):")
        lines += [f"    - {audit_log.describe(e, tagged=False).split(': ', 1)[-1]}" for e in claims[:_MAX_REASONS]]
        if len(claims) > _MAX_REASONS:
            lines.append(f"    (and {len(claims) - _MAX_REASONS} more: `--trace <quest_id> --trace-node {node}`)")
    failed = [e for e in last if e.get("kind") == "node_failed"]
    lines += [f"  It failed: {e.get('error', '')}" for e in failed]
    return lines


def _evidence(root: Path) -> list[str]:
    record = _json(root / "needs" / "EVIDENCE.json")
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
        parts.append(_step(events, "review", "Why the review decided what it did"))
    elif topic == "evidence":
        parts.append(_evidence(root))
    elif topic:
        parts.append(_step(events, topic, f"Why the {topic} step decided what it did"))
    else:
        parts.append(_pause(root))
        if _passes(events, "review"):
            parts.append(_step(events, "review", "Why the review decided what it did"))
        parts.append(_evidence(root))
    return "\n\n".join("\n".join(p) for p in parts if p)
