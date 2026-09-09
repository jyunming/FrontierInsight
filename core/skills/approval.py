"""The human gate: which exact skill contents a person has approved.

Approval is recorded against a **content hash**, never a name. FI is
meant to learn — a skill distilled again from newer work is a different
skill wearing the same name, and letting it inherit the old approval
would turn the human gate into a one-time rubber stamp on a moving
target. Changing ``SKILL.md`` lapses approval; recording that the skill
was used in one more quest does not.

The ledger lives outside the skill directories, for two reasons: a skill
installed from a package is read-only, and an approval that travelled
inside the skill would be self-certifying — anything that ships a skill
could ship its own approval with it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ENV_LEDGER = "FI_SKILLS_APPROVALS"


def ledger_path() -> Path:
    """Where approvals are recorded.

    ``FI_SKILLS_APPROVALS`` overrides, which is what tests use — no test
    should ever be able to write into a developer's real ledger.
    """
    override = os.environ.get(_ENV_LEDGER, "").strip()
    if override:
        return Path(override)
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~")
    return Path(base) / ".frontier-insight" / "skill_approvals.json"


@dataclass(frozen=True)
class Approval:
    name: str
    content_hash: str
    approved_at: str
    approved_by: str
    note: str = ""


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def approved_hash(name: str, path: Path | None = None) -> str | None:
    """The content hash a person approved for this skill, if any."""
    entry = _load(path or ledger_path()).get(name)
    if not isinstance(entry, dict):
        return None
    h = entry.get("content_hash")
    return h if isinstance(h, str) and h else None


def is_approved(name: str, content_hash: str, path: Path | None = None) -> bool:
    return approved_hash(name, path) == content_hash


def approve(
    name: str,
    content_hash: str,
    *,
    approved_by: str,
    note: str = "",
    path: Path | None = None,
) -> Approval:
    """Record a person's approval of this exact content.

    ``approved_by`` is required and must be non-empty. There is no
    default and no "system" caller: the whole point of this gate is that
    a person made the decision, so a call that cannot name one is a bug.
    """
    who = (approved_by or "").strip()
    if not who:
        raise ValueError("approve() requires a non-empty approved_by")
    if not content_hash.strip():
        raise ValueError("approve() requires a content hash")

    p = path or ledger_path()
    data = _load(p)
    record = {
        "content_hash": content_hash,
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "approved_by": who,
        "note": note.strip(),
    }
    data[name] = record
    _save(p, data)
    return Approval(name=name, **record)


def revoke(name: str, path: Path | None = None) -> bool:
    """Withdraw approval. Returns True if something was removed."""
    p = path or ledger_path()
    data = _load(p)
    if name not in data:
        return False
    del data[name]
    _save(p, data)
    return True


def all_approvals(path: Path | None = None) -> dict[str, Approval]:
    out: dict[str, Approval] = {}
    for name, rec in _load(path or ledger_path()).items():
        if not isinstance(rec, dict) or not rec.get("content_hash"):
            continue
        out[name] = Approval(
            name=name,
            content_hash=str(rec["content_hash"]),
            approved_at=str(rec.get("approved_at", "")),
            approved_by=str(rec.get("approved_by", "")),
            note=str(rec.get("note", "")),
        )
    return out
