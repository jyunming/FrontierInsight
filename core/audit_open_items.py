"""What the methodology audit proposed that the design does not hold, kept open instead of dropped.

The audit proposes changes as ``{path, value}`` (``objection["changes"]``). FI applies the ones the protocol has a home for
(``core/design_parts.py``, the checks the results are judged by stay as they were). A proposal that did not end up in the
design that runs is not lost: FI compares every proposal with the design it actually holds, from its own records, and keeps
each one that is missing as an *open item* with the audit's full text, in ``.fi/audit_open_items.json``. The code-writing
step and the code reader are shown the open items, QUALITY.md lists them, and the paper's limitations say that the audit
proposed them and they were not applied.

``applied`` holds the proposals that are in the design, so that a later design (the frozen protocol) is compared with them
too: a change reported as made that is missing there becomes an open item.

Nothing here calls a model or reads the audit's free text to decide a fact: a proposal is open when the value at its path in
the design differs from the proposed value.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .design_parts import get_part

FILE = "audit_open_items.json"

#: Said once beside the items wherever a step is shown them.
NOTE = ("The methodology audit proposed these changes to the plan and FI did not apply them, so the design above does not "
        "have them. They are advice to weigh, not part of the plan: where the code needs what one of them describes, "
        "say in a comment that the plan does not hold it.")


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def proposals(objections: Any) -> list[dict[str, Any]]:
    """Each change the audit's objections carry, with the objection's own text: ``{path, value, objection, fix, check}``.
    The last proposal for a path wins."""
    out: dict[str, dict[str, Any]] = {}
    for o in objections if isinstance(objections, list) else []:
        if not isinstance(o, dict):
            continue
        for c in o.get("changes") or []:
            if isinstance(c, dict) and isinstance(c.get("path"), str) and "value" in c:
                path = c["path"].strip()
                out[path] = {"path": path, "value": c["value"], "objection": str(o.get("objection") or ""),
                             "fix": str(o.get("fix") or ""), "check": str(o.get("check") or "")}
    return list(out.values())


def holds(design: Any, path: str, value: Any) -> bool:
    """Whether ``design`` has ``value`` at ``path``."""
    found, now = get_part(design if isinstance(design, dict) else {}, path)
    return found and _plain(now) == _plain(value)


def split(props: list[dict[str, Any]], design: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(held, missing)``: the proposals the design has, and the ones it does not."""
    held = [p for p in props if holds(design, p["path"], p["value"])]
    return held, [p for p in props if p not in held]


def read(quest_root: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        data = json.loads((quest_root / ".fi" / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data = data if isinstance(data, dict) else {}
    return {k: [x for x in data.get(k) or [] if isinstance(x, dict) and isinstance(x.get("path"), str)]
            for k in ("open", "applied")}


def _write(quest_root: Path, record: dict[str, list[dict[str, Any]]]) -> None:
    path = quest_root / ".fi" / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def reason_for(path: str, reasons: dict[str, str]) -> str:
    """The reason FI recorded for ``path`` or for the part that holds it (``protocol.grid`` for ``protocol.grid.x``)."""
    segs = path.split(".")
    for n in range(len(segs), 0, -1):
        if ".".join(segs[:n]) in reasons:
            return reasons[".".join(segs[:n])]
    return ""


def record(quest_root: Path, held: list[dict[str, Any]], missing: list[dict[str, Any]], *, reasons: dict[str, str],
           default_reason: str, iteration: int = 0) -> list[dict[str, Any]]:
    """Update the record with one audit pass: a proposal the design holds leaves the open items (and is kept as applied);
    a missing one becomes an open item (its reason: FI's recorded reason for it, else the one this pass gives, else the one
    it already had). Returns the open items. A record that cannot be written is not an error: the pass goes on."""
    now = read(quest_root)
    open_ = {x["path"]: x for x in now["open"]}
    applied = {x["path"]: x for x in now["applied"]}
    for p in held:
        open_.pop(p["path"], None)
        applied[p["path"]] = {"path": p["path"], "value": p["value"], "objection": p["objection"]}
    for p in missing:
        applied.pop(p["path"], None)
        why = (reason_for(p["path"], reasons) or default_reason or str(open_.get(p["path"], {}).get("reason") or "")
               or "FI did not apply it")
        open_[p["path"]] = {**p, "reason": why, "iteration": iteration}
    try:
        _write(quest_root, {"open": list(open_.values()), "applied": list(applied.values())})
    except OSError:
        pass
    return list(open_.values())


def check_against(quest_root: Path, protocol: Any, *, why: str) -> list[dict[str, Any]]:
    """After the protocol is frozen: each applied change to the protocol that the frozen protocol does not hold becomes an
    open item with reason ``why``. Returns those items."""
    now = read(quest_root)
    gone = [x for x in now["applied"] if x["path"].startswith("protocol.")
            and not holds({"protocol": protocol if isinstance(protocol, dict) else {}}, x["path"], x.get("value"))]
    if not gone:
        return []
    props = [{"path": x["path"], "value": x.get("value"), "objection": str(x.get("objection") or ""), "fix": "",
              "check": ""} for x in gone]
    record(quest_root, [], props, reasons={}, default_reason=why)
    return props


def _short(value: Any, limit: int = 4000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def for_code(quest_root: Path) -> dict[str, Any] | None:
    """The open items as a request that writes or reads code is shown them, or ``None`` when there are none."""
    items = read(quest_root)["open"]
    if not items:
        return None
    return {"note": NOTE, "items": [{"part": x["path"], "proposed": x.get("value"), "audit_said": x.get("objection", ""),
                                     "audit_fix": x.get("fix", ""), "not_applied_because": x.get("reason", "")}
                                    for x in items]}


def writer_note(quest_root: Path) -> str:
    """For the writer: what the audit proposed and the plan does not hold, so the limitations say it."""
    items = read(quest_root)["open"]
    if not items:
        return ""
    return ("The methodology audit proposed changes that FI did not apply, so the study was run without them. Say plainly in "
            "the limitations that they were proposed by the audit and not applied, and do not describe the study as if it "
            "had them:\n" + "\n".join(f"- `{x['path']}`: {_short(x.get('objection'), 400)} Proposed: "
                                      f"{_short(x.get('value'), 1500)} (not applied: {x.get('reason')})" for x in items))
