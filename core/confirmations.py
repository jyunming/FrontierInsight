"""The record of every confirmation a quest made: one line per verdict, never rewritten (``.fi/confirmations.jsonl``).

Explore, then confirm (``core/phased.py``) runs a frozen version of the study once on data or seeds exploration never
saw. Each version that reaches that confirm run is a *candidate*: its code, protocol and environment are hashed when it
is frozen, and the confirm run's verdict is one line here. The lines are only ever appended:

- a confirm run made again for the same candidate (a re-run after its result was seen, a retry after it failed) adds a
  second line beside the first; the evidence reads the worst of the candidate's lines, so a failure still counts;
- a version changed after its confirmation (an improvement, a repair, a re-run the review asked for, a redesign, a run
  from an earlier step with changed code) is a NEW candidate, with its own confirm run, while the earlier lines stay;
- the paper's methods paragraph (``core/disclosure.py``) counts the candidates and how many of them did not hold.

Each line carries the hash of the line before it (``prev_sha256``), and the engine writes one ``confirmation_recorded``
event into the hash-chained decision trace for each line (``line_sha256``), so a line changed or removed after it was
written is found (:func:`problems`). This record is apart from ``.fi/phased.json``, which holds only the current
candidate and is rewritten as it moves on.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "fi.confirmation/v1"
NAME = "confirmations.jsonl"  # under .fi/
#: The decision-trace event written for each line (``core/audit_log.py``).
EVENT = "confirmation_recorded"
#: The verdict of a candidate whose confirmation held (``core/phased.status``).
CONFIRMED = "confirmed"


def path(quest_root: Path) -> Path:
    return Path(quest_root) / ".fi" / NAME


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_lines(quest_root: Path) -> list[str]:
    try:
        text = path(quest_root).read_text(encoding="utf-8")
    except OSError:
        return []
    return [line for line in text.split("\n") if line.strip()]


def read(quest_root: Path) -> list[dict[str, Any]]:
    """Every line as written (a line that cannot be read is skipped here, and is a problem in :func:`problems`)."""
    out: list[dict[str, Any]] = []
    for line in _raw_lines(quest_root):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


def line_hashes(quest_root: Path) -> list[str]:
    """The hash of each line as it stands in the file (what the trace's events name)."""
    return [_sha(line) for line in _raw_lines(quest_root)]


def append(quest_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Add one verdict line. Appended to the file, never written over it; the line names the hash of the one before."""
    lines = _raw_lines(quest_root)
    body = {"schema": SCHEMA, "n": len(lines) + 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry,
            "prev_sha256": _sha(lines[-1]) if lines else None}
    text = json.dumps(body, sort_keys=True, default=str)
    target = path(quest_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return body


def _candidate_of(entry: dict[str, Any]) -> int | None:
    """The version a line is about, or None when the line does not say it as a number (a line edited by hand: it is
    left out here, and its chain or trace problem is found by :func:`problems`)."""
    try:
        return int(entry.get("candidate") or 1)
    except (TypeError, ValueError):
        return None


def for_candidate(entries: list[dict[str, Any]], candidate: int) -> list[dict[str, Any]]:
    return [e for e in entries if _candidate_of(e) == int(candidate)]


def worst(entries: list[dict[str, Any]]) -> str:
    """A candidate's verdict from all of its lines: confirmed only when every line says so (a failure counts)."""
    verdicts = [str(e.get("verdict") or "") for e in entries]
    return next((v for v in verdicts if v != CONFIRMED), CONFIRMED) if verdicts else ""


def candidates(quest_root: Path) -> list[dict[str, Any]]:
    """One entry per candidate that reached a confirm run, oldest first: ``{"candidate", "verdict", "lines"}``."""
    entries = read(quest_root)
    seen = sorted({c for e in entries if (c := _candidate_of(e)) is not None})
    return [{"candidate": c, "verdict": worst(for_candidate(entries, c)), "lines": len(for_candidate(entries, c))}
            for c in seen]


def unsealed(quest_root: Path, events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """``(line hash, entry)`` of each line the trace does not name yet, in order: what the engine seals next."""
    named = {str(e.get("line_sha256")) for e in events if e.get("kind") == EVENT}
    out = []
    for line in _raw_lines(quest_root):
        digest = _sha(line)
        if digest in named:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            entry = {}
        out.append((digest, entry if isinstance(entry, dict) else {}))
    return out


def problems(quest_root: Path, events: list[dict[str, Any]] | None) -> list[str]:
    """Why the record cannot be trusted as written, in plain words (empty when it can): a line that cannot be read, a
    line whose ``prev_sha256`` is not the line before it, or a line the decision trace named that is no longer there as
    it was. ``events``: the trace's events (``None`` when there is no trace; then only the chain is checked)."""
    lines = _raw_lines(quest_root)
    out: list[str] = []
    prev = None
    for i, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except ValueError:
            out.append(f"line {i} of the record of confirmations (.fi/{NAME}) cannot be read")
            prev = _sha(line)
            continue
        if not isinstance(entry, dict) or entry.get("prev_sha256") != prev:
            out.append(f"line {i} of the record of confirmations (.fi/{NAME}) does not follow the line before it "
                       "(a line was changed or removed after it was written)")
        prev = _sha(line)
    if events is not None:
        named = [str(e.get("line_sha256")) for e in events if e.get("kind") == EVENT]
        have = set(_sha(line) for line in lines)
        missing = [h for h in named if h not in have]
        if missing:
            out.append(f"{len(missing)} confirmation(s) the decision trace recorded are no longer in .fi/{NAME} as "
                       "they were written (the record was changed after it was written)")
    return out
