"""The record each required check leaves when it runs: whether it passed, and on what.

The evidence ladder (core/evidence.py) used to reach ``publication_ready`` unless one of three checks had *reported* a
failure: an evidence gate, methodology audit or claim check that never ran at all left nothing to report, and the quest
read as ready. Each of them now writes a receipt under ``needs/receipts/<check>.json`` every time it runs, and the ladder
reads the receipts: a missing, unreadable or malformed receipt is ``unknown``, never a pass.

A receipt holds ``schema_version``, ``check``, ``status`` (``pass`` | ``fail`` | ``unknown`` | ``not_applicable``),
``started_at`` / ``completed_at`` (UTC, ISO 8601), ``producer`` (the step that ran it), ``input_hashes`` (SHA-256 of
what it judged, so a receipt for an earlier draft is told from one for the final draft), ``output_hash`` (SHA-256 of
its verdict), ``error`` (why a check could not judge) and ``detail`` (a short plain summary).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATUSES = ("pass", "fail", "unknown", "not_applicable")
#: The checks the top level of the evidence ladder needs a receipt from, and what each is called in a sentence.
REQUIRED = {
    "evidence_gate": "the evidence gate",
    "design_audit": "the design methodology audit",
    "claim_check": "the claim check",
}
_FIELDS = ("schema_version", "check", "status", "started_at", "completed_at", "producer", "input_hashes",
           "output_hash", "error", "detail")
#: What each check's pass must name as judged (a pass that names nothing it judged is not believed).
REQUIRED_INPUTS = {
    "evidence_gate": ("analysis", "cross_check"),
    "design_audit": ("design",),
    "claim_check": ("paper",),
}
_HEX = set("0123456789abcdef")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256(value: Any) -> str:
    """SHA-256 of text, bytes, or anything JSON can write (keys sorted, so the same verdict hashes the same)."""
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8", errors="replace")
    else:
        data = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def design_core(design: Any) -> Any:
    """What of a design the methodology audit vouches for: all of it but its ``rationale`` (the model's own account)
    and its ``protocol``, which the frozen protocol's own checks hold the run to (the engine may add an oracle there
    after the audit)."""
    if not isinstance(design, dict):
        return design
    return {k: v for k, v in design.items() if k not in ("rationale", "protocol")}


def path(quest_root: Path, check: str) -> Path:
    return Path(quest_root) / "needs" / "receipts" / f"{check}.json"


def write(
    quest_root: Path, check: str, *, status: str, producer: str, started_at: str,
    inputs: dict[str, Any] | None = None, output: Any = None, error: str = "", detail: str = "",
) -> dict[str, Any]:
    """Write (replace) the receipt of ``check``. The last run of a check is the one that counts: a check that ran on
    each draft leaves the receipt of the final one."""
    if status not in STATUSES:
        raise ValueError(f"receipt status {status!r} is not one of {STATUSES}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "check": check,
        "status": status,
        "started_at": started_at,
        "completed_at": now(),
        "producer": producer,
        "input_hashes": {k: sha256(v) for k, v in (inputs or {}).items()},
        "output_hash": sha256(output) if output is not None else "",
        "error": str(error or ""),
        "detail": str(detail or ""),
    }
    target = path(quest_root, check)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return record


def carry_output(quest_root: Path, check: str, before: Any, after: Any, why: str) -> bool:
    """Move a receipt from its verdict ``before`` to ``after``, when the engine turned the one into the other without
    changing what was judged (the plan step writing the audited design in its own canonical form). Only a receipt whose
    output was exactly ``before`` is moved; returns whether it was."""
    status, record, problem = read(quest_root, check)
    if problem or record is None or record.get("output_hash") != sha256(before):
        return False
    record["output_hash"] = sha256(after)
    record["detail"] = (str(record.get("detail") or "") + f" (carried over: {why})").strip()
    target = path(quest_root, check)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    os.replace(tmp, target)
    return True


def carry_over(quest_root: Path, check: str, key: str, before: Any, after: Any, why: str) -> bool:
    """Move a receipt from ``before`` to ``after`` of one input, when ``after`` was made from ``before`` by a step that
    cannot add what the check judged (the page-limit trim only takes sentences out of the checked draft). Only a receipt
    whose input was exactly ``before`` is moved; returns whether it was."""
    status, record, problem = read(quest_root, check)
    if problem or record is None or (record.get("input_hashes") or {}).get(key) != sha256(before):
        return False
    record["input_hashes"] = {**record["input_hashes"], key: sha256(after)}
    record["detail"] = (str(record.get("detail") or "") + f" (carried over: {why})").strip()
    target = path(quest_root, check)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return True


def read(quest_root: Path, check: str) -> tuple[str, dict[str, Any] | None, str]:
    """``(status, receipt, problem)``. A receipt that is missing, unreadable, from another schema, or missing a field
    is ``unknown`` with the problem named; only a well-formed receipt's own status is returned as it is."""
    target = path(quest_root, check)
    if not target.is_file():
        return "unknown", None, "it did not run (no record of it)"
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return "unknown", None, f"its record is unreadable ({type(e).__name__})"
    if not isinstance(record, dict) or any(f not in record for f in _FIELDS):
        return "unknown", None, "its record is incomplete"
    if record.get("schema_version") != SCHEMA_VERSION or record.get("check") != check:
        return "unknown", None, "its record is not a receipt of this check"
    status = record.get("status")
    if status not in STATUSES:
        return "unknown", None, f"its record has no valid status ({status!r})"
    hashes = record.get("input_hashes")
    if (not isinstance(record.get("started_at"), str) or not record["started_at"]
            or not isinstance(record.get("completed_at"), str) or not record["completed_at"]
            or record.get("producer") != check or not isinstance(hashes, dict)
            or not all(_is_hash(v) for v in hashes.values())
            or not (record.get("output_hash") == "" or _is_hash(record.get("output_hash")))):
        return "unknown", None, "its record is malformed"
    if status == "pass" and (any(k not in hashes for k in REQUIRED_INPUTS.get(check, ()))
                             or not _is_hash(record.get("output_hash"))):
        return "unknown", None, "its record names nothing it judged"
    return str(status), record, ""
