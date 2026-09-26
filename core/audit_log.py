"""The quest's audit trace: one append-only file that says what happened, in order, and can be checked for tampering.

Every quest writes ``<quest_root>/.fi/audit.jsonl``, one JSON object per line:

* ``schema`` (``fi.audit/v1``), ``seq`` (1, 2, 3 ... without gaps), ``ts`` (UTC);
* ``prev`` and ``hash``: ``hash`` is the sha256 of ``prev`` and the record's own canonical JSON, so removing, reordering or
  editing a line breaks every hash after it (:func:`verify`);
* ``kind``: what happened (see :data:`KINDS`);
* ``provenance``: **who says so**. ``deterministic`` is something the engine measured or decided by code (a node started, a
  check passed, a route was taken). ``model_claim`` is the model's own account of why (assumptions, options it weighed);
  it is recorded because it can be argued with, and it is never a check result.

The trace is a record, not a control: a failure to write it never stops a quest (:meth:`AuditLog.append` swallows ``OSError``
and says so once in the quest log).

Two writers on one quest are not supported (one engine owns a quest); a thread lock covers the threads of that engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "fi.audit/v1"
GENESIS = "0" * 64

DETERMINISTIC = "deterministic"
MODEL_CLAIM = "model_claim"
PROVENANCES = (DETERMINISTIC, MODEL_CLAIM)

KINDS = (
    "quest_started",      # the engine began (or resumed) a run
    "node_started",       # a node of the graph began; after a pause the interrupted node starts again, truthfully
    "node_completed",     # ... and finished; carries its duration and which state keys it wrote
    "node_paused",        # ... and stopped for a person (the pause kind is in a ``pause_requested`` just before)
    "node_failed",        # ... and raised
    "pause_requested",    # the quest stops for a person: which pause, what is asked
    "route_decision",     # a conditional edge chose the next node, with the facts it read
    "artifact_created",   # a file the quest wrote, with its sha256
    "check_result",       # one check's verdict (protocol, oracle, run manifest, numeric warnings, evidence, design audit)
    "model_claim",        # the model's own rationale (provenance model_claim)
    "audit_repair",       # a torn last line was dropped when the file was reopened
)

# Keys the chain owns: an event's own fields never replace them.
_RESERVED = frozenset({"schema", "seq", "ts", "quest_id", "kind", "provenance", "prev", "hash", "node"})

_MAX_TEXT = 2000            # one string in an event; longer is cut and says so
_MAX_LIST = 60              # one list in an event
_REDACTED = "[redacted]"

# What looks like a credential in free text. Env values are handled separately, by value.
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b(\s*[:=]\s*)(['\"]?)[^\s'\",;]{6,}"),
)
_SECRET_ENV_NAME = re.compile(r"(?i)(api[_-]?key|secret|token|passw(or)?d|credential)")


def _secret_values() -> list[str]:
    """Values of environment variables that look like credentials, longest first (so one containing another goes whole)."""
    values = {v for k, v in os.environ.items() if _SECRET_ENV_NAME.search(k) and len(v) >= 8}
    return sorted(values, key=len, reverse=True)


def _home_variants() -> list[str]:
    home = str(Path.home())
    return sorted({home, home.replace("\\", "/")}, key=len, reverse=True) if home else []


def redact_text(text: str, *, secrets: list[str] | None = None, whole: bool = False) -> str:
    """``text`` without credentials (env values, key-shaped tokens) and with the home directory as ``~``; cut at
    :data:`_MAX_TEXT` characters, saying how many were dropped, unless ``whole`` (a kept model call is kept whole)."""
    for value in secrets if secrets is not None else _secret_values():
        text = text.replace(value, _REDACTED)
    for pattern in _SECRET_PATTERNS[:-1]:
        text = pattern.sub(_REDACTED, text)
    text = _SECRET_PATTERNS[-1].sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{_REDACTED}", text)
    for home in _home_variants():
        text = text.replace(home, "~")
    if not whole and len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT] + f"... [{len(text) - _MAX_TEXT} more characters not kept]"
    return text


def redact(value: Any, *, secrets: list[str] | None = None, whole: bool = False) -> Any:
    """``value`` (JSON-shaped) with every string passed through :func:`redact_text`; lists are cut at :data:`_MAX_LIST`.
    ``whole`` removes credentials only, and cuts nothing."""
    secrets = _secret_values() if secrets is None else secrets
    if isinstance(value, str):
        return redact_text(value, secrets=secrets, whole=whole)
    if isinstance(value, dict):
        return {str(k): redact(v, secrets=secrets, whole=whole) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        limit = len(items) if whole else _MAX_LIST
        kept = [redact(v, secrets=secrets, whole=whole) for v in items[:limit]]
        if len(items) > limit:
            kept.append(f"... [{len(items) - limit} more not kept]")
        return kept
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value), secrets=secrets, whole=whole)


def canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_of(prev: str, record_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256((prev + "\n" + canonical(record_without_hash)).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    """sha256 of a file's bytes, ``None`` when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditLog:
    """Appends events to one quest's ``audit.jsonl``. Cheap to construct: the file is only touched by :meth:`append`."""

    def __init__(self, path: Path, quest_id: str, *, enabled: bool = True) -> None:
        self.path = path
        self.quest_id = quest_id
        self.enabled = enabled
        self._lock = threading.Lock()
        self._seq = 0
        self._prev = GENESIS
        self._opened = False
        self._warned = False
        self.last_kind: str | None = None
        self.last_node: str | None = None
        self.paused_node: str | None = None     # the node that stopped for a person and has not started again
        self.write_errors = 0

    # ---- writing --------------------------------------------------------

    def _open(self) -> None:
        """Continue the chain of an existing file (a resumed quest), dropping a torn last line if a crash left one."""
        self._opened = True
        if not self.path.is_file():
            return
        data = self.path.read_bytes()
        torn = b""
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            torn, data = data[cut:], data[:cut]
            self.path.write_bytes(data)
        last = None
        for line in data.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                last = obj
                self._track(str(obj.get("kind")), obj.get("node"))
        if isinstance(last, dict) and isinstance(last.get("seq"), int) and isinstance(last.get("hash"), str):
            self._seq, self._prev = last["seq"], last["hash"]
            self.last_kind, self.last_node = last.get("kind"), last.get("node")
        if torn:
            self._write("audit_repair", None, DETERMINISTIC, {
                "dropped_bytes": len(torn), "dropped_sha256": hashlib.sha256(torn).hexdigest(),
            })

    def append(self, kind: str, *, node: str | None = None, provenance: str = DETERMINISTIC, **fields: Any) -> dict[str, Any] | None:
        """One event. Returns the record written, ``None`` when the trace is off or could not be written."""
        if not self.enabled:
            return None
        with self._lock:
            try:
                if not self._opened:
                    self._open()
                return self._write(kind, node, provenance, fields)
            except OSError as e:
                self.write_errors += 1
                if not self._warned:
                    self._warned = True
                    import logging
                    logging.getLogger(f"fi.{self.quest_id}").warning("[audit] the audit trace could not be written (%r); the quest goes on", e)
                return None

    def _write(self, kind: str, node: str | None, provenance: str, fields: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": SCHEMA, "seq": self._seq + 1, "ts": _utc_now(), "quest_id": self.quest_id, "kind": kind,
            "provenance": provenance, "prev": self._prev,
        }
        if node:
            record["node"] = node
        for key, value in redact(fields).items():
            if key not in _RESERVED:
                record[key] = value
        record["hash"] = hash_of(self._prev, record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as fh:
            fh.write((canonical(record) + "\n").encode("utf-8"))
            fh.flush()
        self._seq, self._prev = record["seq"], record["hash"]
        self.last_kind, self.last_node = kind, node
        self._track(kind, node)
        return record

    def _track(self, kind: str, node: Any) -> None:
        if kind == "node_paused":
            self.paused_node = str(node)
        elif kind in ("node_started", "node_completed", "node_failed"):
            self.paused_node = None

    def event_count(self) -> int:
        """How many events the file holds already (0 for a new quest); opens the chain if it is not open yet."""
        if not self.enabled:
            return 0
        with self._lock:
            try:
                if not self._opened:
                    self._open()
            except OSError:
                return 0
            return self._seq


# ---- reading --------------------------------------------------------------


def read(path: Path) -> list[dict[str, Any]]:
    """Every parseable event of a trace, in file order (lines that do not parse are skipped; :func:`verify` reports them)."""
    events: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return events
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


@dataclass
class Verification:
    ok: bool
    events: int
    reason: str = ""
    bad_seq: int | None = None

    def line(self) -> str:
        if self.ok:
            return f"chain intact ({self.events} events)"
        return f"CHAIN BROKEN at event {self.bad_seq}: {self.reason}"


def verify(path: Path) -> Verification:
    """Check the hash chain of a trace: every line parses, ``seq`` runs 1, 2, 3 ..., and every ``prev`` and ``hash`` follows from
    the line before. An empty or missing trace is intact (nothing was recorded)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Verification(True, 0)
    except OSError as e:
        return Verification(False, 0, f"cannot read the trace: {e!r}")
    prev, count = GENESIS, 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            return Verification(False, count, f"line {number} is not valid JSON", count + 1)
        if not isinstance(record, dict):
            return Verification(False, count, f"line {number} is not an event", count + 1)
        count += 1
        if record.get("seq") != count:
            return Verification(False, count, f"the sequence number is {record.get('seq')!r}, expected {count} (an event was removed, added or reordered)", count)
        if record.get("prev") != prev:
            return Verification(False, count, "it does not follow the event before it", count)
        claimed = record.pop("hash", None)
        if claimed != hash_of(prev, record):
            return Verification(False, count, "its content does not match its hash (the event was edited)", count)
        prev = claimed
    return Verification(True, count)


# ---- presenting -------------------------------------------------------------

DETAILS = ("summary", "checks", "debug")

# What each detail level shows. ``summary``: the shape of the run and every decision. ``checks``: plus each check's verdict,
# the artifacts and the model's stated reasons. ``debug``: everything, including each node's start.
_SUMMARY_KINDS = {"quest_started", "node_completed", "node_paused", "node_failed", "pause_requested", "route_decision", "audit_repair"}
_CHECK_KINDS = _SUMMARY_KINDS | {"check_result", "artifact_created", "model_claim"}


def select(events: Iterable[dict[str, Any]], *, node: str | None = None, detail: str = "checks") -> list[dict[str, Any]]:
    """The events to show for a node filter and a detail level."""
    if detail not in DETAILS:
        raise ValueError(f"detail must be one of {DETAILS}; got {detail!r}")
    keep = _SUMMARY_KINDS if detail == "summary" else _CHECK_KINDS if detail == "checks" else None
    out = []
    for e in events:
        if node and e.get("node") != node:
            continue
        if keep is not None and e.get("kind") not in keep:
            continue
        out.append(e)
    return out


def _tag(e: dict[str, Any]) -> str:
    return {"model_claim": " [model claim]"}.get(str(e.get("provenance")), "")


def describe(e: dict[str, Any], *, tagged: bool = True) -> str:
    """One line for an event, the same words on every surface. ``tagged`` ends a model's claim with what it is;
    a page that shows that as a badge asks for it without."""
    kind, node = str(e.get("kind")), e.get("node")
    tag = _tag if tagged else (lambda _e: "")
    where = f"{node}: " if node else ""
    if kind == "node_started":
        return f"{where}started" + (" (again after a pause)" if e.get("resumed") else "")
    if kind == "node_completed":
        wrote = e.get("wrote")
        return f"{where}done in {e.get('duration_s', '?')}s" + (f", wrote {', '.join(map(str, wrote))}" if wrote else "")
    if kind == "node_paused":
        return f"{where}stopped for you" + (f" ({e['pause']})" if e.get("pause") else "")
    if kind == "node_failed":
        return f"{where}FAILED: {e.get('error', '')}"
    if kind == "pause_requested":
        return f"{where}waiting for you ({e.get('pause')}): {e.get('headline', '')}"
    if kind == "route_decision":
        facts = e.get("facts") or {}
        shown = ", ".join(f"{k}={v}" for k, v in facts.items())
        return f"{where}next is {e.get('chosen')}" + (f" because {shown}" if shown else "")
    if kind == "artifact_created":
        return f"{where}wrote {e.get('path')} (sha256 {str(e.get('sha256'))[:12]})"
    if kind == "check_result":
        return f"{where}check {e.get('check')}: {e.get('status')}" + (f" - {e.get('summary')}" if e.get("summary") else "")
    if kind == "model_claim":
        decision, reason = e.get("decision"), e.get("reason")
        tail = f" -> {decision}" if decision else ""
        tail += f" ({reason})" if reason else ""
        return f"{where}{e.get('topic', 'rationale')}: {e.get('claim', '')}{tail}{tag(e)}"
    if kind == "audit_repair":
        return f"a torn last line ({e.get('dropped_bytes')} bytes) was dropped"
    if kind == "quest_started":
        return f"quest {'resumed' if e.get('resumed') else 'started'}"
    return f"{where}{kind}"


def render(events: Iterable[dict[str, Any]]) -> list[str]:
    """Timeline lines: ``#seq time  description``."""
    lines = []
    for e in events:
        ts = str(e.get("ts", ""))[11:19]
        lines.append(f"#{e.get('seq'):<4} {ts}  {describe(e)}")
    return lines
