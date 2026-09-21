"""The protocol a quest is held to, frozen before its first full run, and the amendments that may change it afterwards.

Until the first full run the protocol in ``plan.md`` is a draft: a person may edit it and the engine may complete it (add an
oracle). Right before the first full run it is frozen into ``needs/FROZEN_PROTOCOL.json`` with its SHA-256, who approved it
and when. From then on every gate (the protocol check, the oracle gate, the evidence level) reads that record, and never the
mutable design: a redesign that leaves the protocol out cannot make a gate see nothing, and one that changes it is an
*amendment request*, not a quiet edit.

An amendment is proposed (``needs/PROTOCOL_AMENDMENT_PENDING.json``: the new protocol, what changed, what asked for it), stops
the quest, and takes effect only when a person approves it as an act of its own (``needs/AMENDMENT_APPROVAL.json``, written by
``launch.py --approve-amendment``, the web page's button or ``@fi /approve-amendment``). Resuming without approving keeps the
frozen protocol. An approved amendment writes ``needs/PROTOCOL_AMENDMENT_<n>.json``, freezes a new version and starts a new
run; when results had already been seen, the old run's raw data, code, paper and evidence are archived under
``archive/run_<n>/`` first, and the amendment is recorded as *not pre-specified* so the paper says the change was made after
the results were known.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_SCHEMA = "fi.frozen-protocol/v1"
AMENDMENT_SCHEMA = "fi.protocol-amendment/v1"


def _needs(quest_root: Path) -> Path:
    return quest_root / "needs"


def frozen_path(quest_root: Path) -> Path:
    return _needs(quest_root) / "FROZEN_PROTOCOL.json"


def pending_path(quest_root: Path) -> Path:
    return _needs(quest_root) / "PROTOCOL_AMENDMENT_PENDING.json"


def approval_path(quest_root: Path) -> Path:
    return _needs(quest_root) / "AMENDMENT_APPROVAL.json"


def amendment_path(quest_root: Path, n: int) -> Path:
    return _needs(quest_root) / f"PROTOCOL_AMENDMENT_{n}.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(protocol: Any) -> str:
    return json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256(protocol: Any) -> str:
    return hashlib.sha256(canonical(protocol).encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# --- the frozen record ----------------------------------------------------------------------------------------------


def load(quest_root: Path) -> dict[str, Any] | None:
    """The frozen record, or ``None`` when the protocol has not been frozen yet.

    ``problem`` is set when the record does not match its own hash (someone edited the file): the protocol in it is still
    what is returned, and the problem is a gap the evidence level names.
    """
    record = _read(frozen_path(quest_root))
    if not isinstance(record, dict) or record.get("schema") != FROZEN_SCHEMA:
        return None
    record = dict(record)
    if record.get("sha256") != sha256(record.get("protocol")):
        record["problem"] = "needs/FROZEN_PROTOCOL.json does not match its own SHA-256 (it was edited after the freeze)"
    return record


def protocol_of(quest_root: Path) -> dict[str, Any] | None:
    """The frozen protocol itself (``None`` when nothing is frozen or the study froze without one)."""
    record = load(quest_root)
    protocol = record.get("protocol") if record else None
    return protocol if isinstance(protocol, dict) and protocol else None


def freeze(
    quest_root: Path, protocol: dict[str, Any] | None, *, approved_by: str, source: str,
) -> dict[str, Any]:
    """Freeze ``protocol`` (idempotent: an existing record is returned untouched). ``protocol`` may be ``None``: the study
    is then recorded as having run without one, which the evidence level says."""
    existing = load(quest_root)
    if existing is not None:
        return existing
    body = protocol if isinstance(protocol, dict) and protocol else None
    record = {
        "schema": FROZEN_SCHEMA,
        "version": 1,
        "run_id": "run_1",
        "protocol": body,
        "sha256": sha256(body),
        "approved_by": approved_by,
        "approved_at": now(),
        "source": source,
        "amendments": 0,
    }
    _write(frozen_path(quest_root), record)
    _write(_needs(quest_root) / "protocol_versions" / "v1.json", record)
    return record


def run_id(quest_root: Path) -> str:
    record = load(quest_root)
    return str(record.get("run_id")) if record else "run_1"


# --- what changed ---------------------------------------------------------------------------------------------------


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            out.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return out or ({prefix: {}} if prefix else {})
    return {prefix: value}


def diff(before: Any, after: Any) -> list[str]:
    """One line per key that differs between two protocols: ``grid.R0: [0.9, 1.5] -> [1, 2, 3]``."""
    a, b = _flatten(before or {}), _flatten(after or {})
    lines: list[str] = []
    for key in sorted({*a, *b}):
        if a.get(key, "<absent>") != b.get(key, "<absent>"):
            was = json.dumps(a[key], ensure_ascii=False) if key in a else "<absent>"
            now_ = json.dumps(b[key], ensure_ascii=False) if key in b else "<absent>"
            lines.append(f"{key}: {was} -> {now_}")
    return lines


# --- amendments -----------------------------------------------------------------------------------------------------


def load_pending(quest_root: Path) -> dict[str, Any] | None:
    pending = _read(pending_path(quest_root))
    return pending if isinstance(pending, dict) and pending.get("proposed_sha256") else None


def propose(
    quest_root: Path, proposed_protocol: dict[str, Any], design: dict[str, Any] | None, *,
    source: str, reason: str, results_seen: bool,
) -> dict[str, Any]:
    """Write the amendment request. ``design`` is the whole redesign that carried the new protocol, kept so that approving the
    request adopts exactly it, without asking the model for another."""
    frozen = load(quest_root) or {}
    pending = {
        "schema": AMENDMENT_SCHEMA,
        "n": int(frozen.get("amendments", 0) or 0) + 1,
        "from_sha256": frozen.get("sha256"),
        "proposed_sha256": sha256(proposed_protocol),
        "proposed_protocol": proposed_protocol,
        "changes": diff(frozen.get("protocol"), proposed_protocol),
        "source": source,
        "reason": reason,
        "results_seen": bool(results_seen),
        "design": design,
        "proposed_at": now(),
    }
    _write(pending_path(quest_root), pending)
    return pending


def approve(quest_root: Path, who: str, *, via: str) -> tuple[bool, str]:
    """Record a person's approval of the pending amendment. ``(ok, message)``; the message names what was approved."""
    who = (who or "").strip()
    if not who:
        return False, "an approval is attributed: say who approves it"
    pending = load_pending(quest_root)
    if pending is None:
        return False, "there is no protocol amendment waiting for approval"
    _write(approval_path(quest_root), {
        "proposed_sha256": pending["proposed_sha256"], "approved_by": who, "approved_at": now(), "via": via,
    })
    return True, f"approved amendment {pending['n']}: " + "; ".join(pending.get("changes") or ["(no changes listed)"])


def approval_for(quest_root: Path, pending: dict[str, Any]) -> dict[str, Any] | None:
    """The approval that goes with this exact request, or ``None`` (an approval of another request does not count)."""
    approval = _read(approval_path(quest_root))
    if isinstance(approval, dict) and approval.get("proposed_sha256") == pending.get("proposed_sha256") and approval.get("approved_by"):
        return approval
    return None


def archive_run(quest_root: Path, run: str, *, raw_root: Path | None) -> str | None:
    """Put the finished run's data where the next run cannot mix with it: the raw outcomes are moved (so the new run cannot
    reuse them), the code, paper, summary and evidence record are copied. Returns the archive folder, relative to the quest."""
    dest = quest_root / "archive" / run
    dest.mkdir(parents=True, exist_ok=True)
    moved = False
    if raw_root is not None and raw_root.is_dir():
        target = dest / "raw"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(raw_root), str(target))
        moved = True
    for name in ("code", "paper", "paper.md", "frontier_insight_summary.json", "needs/EVIDENCE.json"):
        src = quest_root / name
        if not src.exists():
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
        moved = True
    return str(dest.relative_to(quest_root)).replace("\\", "/") if moved else None


def apply(quest_root: Path, pending: dict[str, Any], approval: dict[str, Any], *, raw_root: Path | None) -> dict[str, Any]:
    """Make an approved amendment the frozen protocol: a numbered record, a new version, a new run id, the old run archived
    when results had been seen. Returns the amendment record."""
    frozen = load(quest_root) or {}
    n = int(pending["n"])
    old_run = str(frozen.get("run_id") or "run_1")
    archived = archive_run(quest_root, old_run, raw_root=raw_root) if pending.get("results_seen") else None
    record = {
        "schema": AMENDMENT_SCHEMA,
        "n": n,
        "from_sha256": pending.get("from_sha256"),
        "to_sha256": pending["proposed_sha256"],
        "changes": pending.get("changes") or [],
        "source": pending.get("source"),
        "reason": pending.get("reason"),
        "results_seen_before_change": bool(pending.get("results_seen")),
        "prespecified": not pending.get("results_seen"),
        "approved_by": approval["approved_by"],
        "approved_at": approval.get("approved_at"),
        "approved_via": approval.get("via"),
        "previous_run": old_run,
        "new_run": f"run_{n + 1}",
        "archived_to": archived,
    }
    _write(amendment_path(quest_root, n), record)
    version = int(frozen.get("version", 1) or 1) + 1
    new_frozen = {
        "schema": FROZEN_SCHEMA,
        "version": version,
        "run_id": record["new_run"],
        "protocol": pending["proposed_protocol"],
        "sha256": pending["proposed_sha256"],
        "approved_by": approval["approved_by"],
        "approved_at": approval.get("approved_at"),
        "source": f"amendment {n}",
        "amendments": n,
    }
    _write(frozen_path(quest_root), new_frozen)
    _write(_needs(quest_root) / "protocol_versions" / f"v{version}.json", new_frozen)
    pending_path(quest_root).unlink(missing_ok=True)
    approval_path(quest_root).unlink(missing_ok=True)
    return record


def decline(quest_root: Path, pending: dict[str, Any]) -> dict[str, Any]:
    """The request was resumed past without an approval: the frozen protocol stays, and the request is kept as a record."""
    record = {
        "schema": AMENDMENT_SCHEMA,
        "n": pending["n"],
        "declined": True,
        "changes": pending.get("changes") or [],
        "source": pending.get("source"),
        "reason": pending.get("reason"),
        "declined_at": now(),
    }
    _write(_needs(quest_root) / f"PROTOCOL_AMENDMENT_{pending['n']}_declined.json", record)
    pending_path(quest_root).unlink(missing_ok=True)
    approval_path(quest_root).unlink(missing_ok=True)
    return record


def amendments(quest_root: Path) -> list[dict[str, Any]]:
    """The approved amendments, oldest first."""
    out: list[dict[str, Any]] = []
    n = 1
    while True:
        record = _read(amendment_path(quest_root, n))
        if not isinstance(record, dict):
            return out
        out.append(record)
        n += 1


def post_hoc(quest_root: Path) -> list[dict[str, Any]]:
    """The approved amendments that were made after results had been seen."""
    return [a for a in amendments(quest_root) if not a.get("prespecified", True)]


def disclosure(quest_root: Path) -> str:
    """What the paper must say about the amendments, or an empty string when there are none."""
    records = amendments(quest_root)
    if not records:
        return ""
    lines = [
        "The protocol this study reports was AMENDED after it was frozen. State this in the methods, in these terms, and "
        "report only the current run's results (the earlier run is archived and is not evidence for this paper):",
    ]
    for a in records:
        when = "after the results of the earlier run had been seen (post-hoc)" if not a.get("prespecified", True) else "before any result was seen"
        lines.append(f"- amendment {a.get('n')}, {when}, approved by {a.get('approved_by')}: " + "; ".join(a.get("changes") or ["(no listed changes)"]))
        if a.get("reason"):
            lines.append(f"  reason: {str(a['reason'])[:400]}")
    return "\n".join(lines)
