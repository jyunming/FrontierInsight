"""Going on with checks whose expected value has no stated source: the person's choice, recorded.

Under ``rigor_profile: research`` a quest stops at the plan when a check against a known answer (an oracle) does not say
where the value it expects comes from (``core/oracle_check.py::source_gaps``). Filling that in is the way on FI recommends,
and FI asks the model once itself before it stops. Some expected values need no source a person has to hunt for, and some
people know the value is right and have no time to write why. So the stop offers a third way: go on as it is.

That choice is an act of its own, like approving a change to a frozen protocol (``core/frozen_protocol.py``): it needs a
name (``--accept-checks <quest> --approve-as <you>``, the web page's *Go on as it is* button, ``@fi /accept-checks``),
and it covers only the checks the stop named. Nothing about the check itself is relaxed: each still runs and is still
judged; the evidence record keeps the gap below *independently validated* (marked "source not confirmed", with who chose
to go on), the audit trace records the choice, and the paper is told to say so plainly.

Two records under ``needs/``: ``UNSOURCED_CHECKS.json`` (written by the stop: which checks, why, the plan version) and
``UNSOURCED_CHECKS_ACCEPTED.json`` (written by the choice).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PENDING_NAME = "UNSOURCED_CHECKS.json"
ACCEPTED_NAME = "UNSOURCED_CHECKS_ACCEPTED.json"
#: The words a person reads for a check they went on with.
NOT_CONFIRMED = "source not confirmed"


def _needs(quest_root: Path) -> Path:
    return Path(quest_root) / "needs"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, record: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(name: Any) -> str:
    return " ".join(str(name or "").split()).lower()


def fingerprint(oracle: dict[str, Any]) -> str:
    """What a choice to go on binds to: the check's name and the numbers it is judged by (``expected``, ``tolerance``,
    ``tolerance_mode``, ``case``, ``measure``). A check changed after the choice is a check nobody chose to go on with."""
    import hashlib

    fields = {"name": _key(oracle.get("name")),
              **{k: oracle.get(k) for k in ("expected", "tolerance", "tolerance_mode", "case", "measure")}}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def pending(quest_root: Path) -> dict[str, Any] | None:
    """What the stop named (``checks``: ``[{name, why, expected, fingerprint}]``, ``plan_version``), or ``None`` when the
    quest is not stopped for it."""
    record = _read(_needs(quest_root) / PENDING_NAME)
    return record if record and isinstance(record.get("checks"), list) and record["checks"] else None


def write_pending(quest_root: Path, checks: list[dict[str, Any]], *, plan_version: int | None,
                  model_missing: list[str] | None = None) -> dict[str, Any]:
    """Record what the stop names, and return the record with ``previous``: what the stop before this one named and at
    which plan version (``None`` the first time), so the stop can say what changed since."""
    earlier = pending(quest_root)
    record = {
        "checks": [{"name": str(c.get("name") or ""), "why": str(c.get("why") or ""), "expected": c.get("expected"),
                    "fingerprint": str(c.get("fingerprint") or "")}
                   for c in checks],
        "model_missing": list(model_missing or []),
        "plan_version": plan_version,
        "at": _now(),
        "previous": ({"checks": earlier.get("checks"), "plan_version": earlier.get("plan_version")}
                     if earlier else None),
    }
    _write(_needs(quest_root) / PENDING_NAME, record)
    return record


def clear_pending(quest_root: Path) -> None:
    try:
        (_needs(quest_root) / PENDING_NAME).unlink(missing_ok=True)
    except OSError:
        pass


def accepted(quest_root: Path) -> dict[str, Any] | None:
    """The person's choice to go on (``by``, ``via``, ``at``, ``checks``: the names), or ``None``."""
    record = _read(_needs(quest_root) / ACCEPTED_NAME)
    return record if record and record.get("by") and isinstance(record.get("checks"), list) else None


def accept(quest_root: Path, who: str, *, via: str) -> tuple[bool, str]:
    """Record that ``who`` goes on with the checks the stop named, as they are. ``(ok, what to tell the person)``."""
    who = " ".join(str(who or "").split())
    if not who:
        return False, ("say who is choosing to go on (--approve-as <you>): going on with checks whose expected value "
                       "has no stated source is a choice a person makes, and it is recorded with their name")
    record = pending(quest_root)
    if record is None:
        return False, ("this quest is not stopped for checks without a stated source (needs/UNSOURCED_CHECKS.json is "
                       "not there): nothing to go on with")
    named = [c for c in record["checks"] if str(c.get("name") or "").strip() and c.get("fingerprint")]
    if not named:
        return False, ("the stop did not name its checks, so there is nothing to go on with yet: resume the quest, and "
                       "the stop names them")
    earlier = accepted(quest_root)
    # Each choice binds to the check as the stop showed it (its name and numbers), and is kept per check.
    chosen = dict((earlier or {}).get("chosen") or {})
    for c in named:
        chosen[_key(c["name"])] = {"name": c["name"], "fingerprint": c["fingerprint"], "by": who, "via": via,
                                   "at": _now(), "why": c.get("why"), "expected": c.get("expected")}
    names = [str(c["name"]) for c in named]
    entry = {
        "by": who, "via": via, "at": _now(), "checks": [v["name"] for v in chosen.values()], "chosen": chosen,
        "plan_version": record.get("plan_version"),
        "history": [*((earlier or {}).get("history") or []),
                    {"by": who, "via": via, "at": _now(), "checks": names}],
    }
    if not _write(_needs(quest_root) / ACCEPTED_NAME, entry):
        return False, "the choice could not be written to needs/ (check the folder can be written to)"
    listed = ", ".join(repr(n) for n in names)
    return True, (f"recorded: {who} goes on with {len(names)} check(s) whose expected value has no stated source "
                  f"({listed}). They still run and are still judged; each is marked “{NOT_CONFIRMED}”, and the result "
                  "and the paper say so. A check whose numbers change after this stops the quest again. Resume the "
                  "quest to go on.")


def chose(quest_root: Path, oracle: dict[str, Any]) -> str | None:
    """Who chose to go on with ``oracle`` exactly as it is now (its name and numbers), or ``None``."""
    record = accepted(quest_root)
    entry = ((record or {}).get("chosen") or {}).get(_key(oracle.get("name")))
    return str(entry.get("by")) if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint(oracle) else None


def changed_since(quest_root: Path, oracle: dict[str, Any]) -> bool:
    """Whether someone chose to go on with a check of this name whose numbers have changed since."""
    record = accepted(quest_root)
    entry = ((record or {}).get("chosen") or {}).get(_key(oracle.get("name")))
    return isinstance(entry, dict) and entry.get("fingerprint") != fingerprint(oracle)


def covers(quest_root: Path, oracles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The choice to go on when it covers every check in ``oracles`` as they are now, else ``None``."""
    record = accepted(quest_root)
    if record is None or not oracles:
        return None
    return record if all(chose(quest_root, o) for o in oracles) else None


def disclosure(quest_root: Path, names: list[str]) -> str:
    """What the paper must say about ``names`` (the checks gone on with that still have no stated source), or ``""``."""
    record = accepted(quest_root)
    if record is None or not names:
        return ""
    chosen = record.get("chosen") or {}
    who = sorted({str((chosen.get(_key(n)) or {}).get("by") or record["by"]) for n in names})
    listed = ", ".join(repr(str(n)) for n in names)
    return (
        f"The expected values of these checks against known answers have no stated source: {listed}. "
        f"{' and '.join(who)} chose to go on without one. Say so plainly where the checks are described (and in the "
        "limitations): each check shows the code agrees with the value the plan expected, not that the value itself is "
        "right. Do not describe these checks as validated against an independent source."
    )
