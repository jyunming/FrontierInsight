"""What the methodology audit proposed that the design does not hold, kept open instead of dropped.

The audit proposes changes as ``{path, value}`` (``objection["changes"]``). FI applies the ones the protocol has a home for
(``core/design_parts.py``, the checks the results are judged by stay as they were). A proposal that did not end up in the
design that runs is not lost: FI compares every proposal with the design it actually holds, from its own records, and keeps
each one that is missing as an *open item* with the audit's full text, in ``.fi/audit_open_items.json``. The code-writing
step and the code reader are shown the open items, QUALITY.md lists them, and the paper's limitations say that the audit
proposed them and they were not applied.

An item is one ``(path, value)`` as the plan reads it (:func:`normal`): the same proposal made in several passes is one item
with one ``origins`` entry per pass and objection, and two different values for one path are two items. ``applied`` holds the
proposals that are in the design, so that a later design (the frozen protocol) is compared with them too: a change reported as
made that is missing there becomes an open item. Every persisted open item is compared with every design that will run
(:func:`close_held`), whether or not the audit repeats it.

Nothing here calls a model or reads the audit's free text to decide a fact: a proposal is open when the value at its path in
the design, read as the plan reads it, differs from the proposed value.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from . import plan as _plan
from .design_parts import get_part

FILE = "audit_open_items.json"
_log = logging.getLogger(__name__)

#: Said once beside the items wherever a step is shown them.
NOTE = ("The methodology audit proposed these changes to the plan and FI did not apply them, so the design above does not "
        "have them. They are advice to weigh, not part of the plan: where the code needs what one of them describes, "
        "say in a comment that the plan does not hold it.")


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def normal(path: str, value: Any) -> Any:
    """``value`` at ``path`` as the plan reads a design (a scalar grid value becomes a list of one, and so on): the value
    is put at its path in a minimal design and read by :func:`core.plan.normalize_design`, the reading the written plan and
    the frozen protocol come from. A value the plan cannot read alone stays as it is."""
    segs = path.split(".")
    if segs == ["hypothesis"]:
        return _plain(value)
    node: Any = value
    for s in reversed(segs):
        node = {s: node}
    design, _why = _plan.normalize_design({"hypothesis": "x", **node})
    found, now = get_part(design or {}, path)
    return _plain(now if design is not None and found else value)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def item_id(path: str, value: Any) -> str:
    return hashlib.sha256(f"{path}\n{_canon(normal(path, value))}".encode("utf-8")).hexdigest()[:12]


def proposals(objections: Any) -> list[dict[str, Any]]:
    """Each change the audit's objections carry, with the objection's own text: ``{path, value, objection, fix, check}``.
    Two changes to one path with different values are both kept; an identical change from the same objection once."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for o in objections if isinstance(objections, list) else []:
        if not isinstance(o, dict):
            continue
        for c in o.get("changes") or []:
            if isinstance(c, dict) and isinstance(c.get("path"), str) and "value" in c:
                path, text = c["path"].strip(), str(o.get("objection") or "")
                out[(path, _canon(c["value"]), text)] = {
                    "path": path, "value": c["value"], "objection": text, "fix": str(o.get("fix") or ""),
                    "check": str(o.get("check") or "")}
    return list(out.values())


def holds(design: Any, path: str, value: Any) -> bool:
    """Whether ``design`` has ``value`` at ``path``, both read as the plan reads them."""
    found, now = get_part(design if isinstance(design, dict) else {}, path)
    return found and _canon(normal(path, now)) == _canon(normal(path, value))


def split(props: list[dict[str, Any]], design: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(held, missing)``: the proposals the design has, and the ones it does not."""
    held = [p for p in props if holds(design, p["path"], p["value"])]
    return held, [p for p in props if p not in held]


class Unreadable(Exception):
    """The record exists and cannot be read or kept: nothing may be written over it."""


def _path(quest_root: Path) -> Path:
    return quest_root / ".fi" / FILE


def _load(quest_root: Path) -> dict[str, list[dict[str, Any]]]:
    """The record. A missing file is an empty record. A file that cannot be read as JSON is kept aside (never overwritten),
    said plainly, and read as empty; one that cannot be opened raises :class:`Unreadable`."""
    path = _path(quest_root)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"open": [], "applied": []}
    except OSError as e:
        raise Unreadable(f"{path} cannot be read ({e!r})") from e
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not a mapping")
    except ValueError:
        aside = path.with_name(f"{path.stem}.corrupt-{int(time.time())}{path.suffix}")
        try:
            os.replace(path, aside)
        except OSError as e:
            raise Unreadable(f"{path} is damaged and could not be moved aside ({e!r})") from e
        _log.error("[design_audit] %s could not be read; it was kept as %s and the list of changes the audit proposed that "
                   "the plan does not hold starts again from the next audit", path.name, aside.name)
        print(f"[FI] The list of audit changes the plan does not hold was damaged; the damaged copy is kept as {aside.name}")
        return {"open": [], "applied": []}
    return {k: [x for x in data.get(k) or [] if isinstance(x, dict) and isinstance(x.get("path"), str)]
            for k in ("open", "applied")}


def read(quest_root: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        return _load(quest_root)
    except Unreadable as e:
        _log.error("[design_audit] %s", e)
        return {"open": [], "applied": []}


def _write(quest_root: Path, record: dict[str, list[dict[str, Any]]]) -> None:
    """Write the record whole or not at all (a temporary file, then a replace)."""
    path = _path(quest_root)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _save(quest_root: Path, record: dict[str, list[dict[str, Any]]]) -> None:
    try:
        _write(quest_root, record)
    except OSError as e:
        _log.error("[design_audit] the list of audit changes the plan does not hold could not be written (%r); the changes "
                   "are in needs/DESIGN_CRITIQUE.json and .fi/run.log only", e)
        print("[FI] The list of audit changes the plan does not hold could not be saved; see .fi/run.log")


def reason_for(path: str, reasons: dict[str, str]) -> str:
    """The reason FI recorded for ``path`` or for the part that holds it (``protocol.grid`` for ``protocol.grid.x``)."""
    segs = path.split(".")
    for n in range(len(segs), 0, -1):
        if ".".join(segs[:n]) in reasons:
            return reasons[".".join(segs[:n])]
    return ""


def _origin(p: dict[str, Any], iteration: int) -> dict[str, Any]:
    return {"iteration": iteration, "objection": p.get("objection", ""), "fix": p.get("fix", ""), "check": p.get("check", "")}


def texts(item: dict[str, Any]) -> str:
    """What the audit said for an item: each distinct objection, in order."""
    seen: list[str] = []
    for o in item.get("origins") or []:
        t = str(o.get("objection") or o.get("fix") or "").strip()
        if t and t not in seen:
            seen.append(t)
    return " / ".join(seen)


def record(quest_root: Path, held: list[dict[str, Any]], missing: list[dict[str, Any]], *, reasons: dict[str, str],
           default_reason: str, iteration: int = 0) -> list[dict[str, Any]]:
    """Update the record with one audit pass: a proposal the design holds leaves the open items and is kept as applied; a
    missing one becomes an open item, or, when the same ``(path, value)`` is already open, one more origin of it (its reason:
    FI's recorded reason for it, else the one this pass gives, else the one it already had). Returns the open items. A record
    that cannot be read is never written over."""
    try:
        now = _load(quest_root)
    except Unreadable as e:
        _log.error("[design_audit] %s; nothing is written over it", e)
        return []
    open_ = {x.get("id") or item_id(x["path"], x.get("value")): x for x in now["open"]}
    applied = {x.get("id") or item_id(x["path"], x.get("value")): x for x in now["applied"]}
    for p in held:
        i = item_id(p["path"], p["value"])
        origin = _origin(p, iteration)
        if i in open_:
            origin_list = open_[i].get("origins") or []
            applied[i] = {"id": i, "path": p["path"], "value": normal(p["path"], p["value"]),
                          "origins": [*origin_list, *([origin] if origin not in origin_list else [])]}
            del open_[i]
        else:
            old = applied.get(i, {}).get("origins") or []
            applied[i] = {"id": i, "path": p["path"], "value": normal(p["path"], p["value"]),
                          "origins": [*old, *([origin] if origin not in old else [])]}
    for p in missing:
        i = item_id(p["path"], p["value"])
        applied.pop(i, None)
        item = open_.get(i) or {"id": i, "path": p["path"], "value": normal(p["path"], p["value"]), "origins": [], "reason": ""}
        origin = _origin(p, iteration)
        if origin not in item["origins"]:
            item["origins"].append(origin)
        item["reason"] = (reason_for(p["path"], reasons) or default_reason or item.get("reason") or "FI did not apply it")
        open_[i] = item
    _save(quest_root, {"open": list(open_.values()), "applied": list(applied.values())})
    return list(open_.values())


def close_held(quest_root: Path, design: Any) -> list[dict[str, Any]]:
    """Compare every persisted open item with ``design``, a design that will run, and move each one it holds to applied
    (a person's edit of the plan that makes the change closes it). Returns the items closed."""
    try:
        now = _load(quest_root)
    except Unreadable:
        return []
    closed = [x for x in now["open"] if holds(design, x["path"], x.get("value"))]
    if not closed:
        return []
    ids = {id(x) for x in closed}
    _save(quest_root, {"open": [x for x in now["open"] if id(x) not in ids],
                       "applied": [*now["applied"], *[{k: v for k, v in x.items() if k != "reason"} for x in closed]]})
    return closed


def check_against(quest_root: Path, protocol: Any, *, why: str) -> list[dict[str, Any]]:
    """After the protocol is frozen: each applied change to the protocol that the frozen protocol does not hold becomes an
    open item with reason ``why``. Returns those items."""
    now = read(quest_root)
    frozen = {"protocol": protocol if isinstance(protocol, dict) else {}}
    gone = [x for x in now["applied"] if x["path"].startswith("protocol.") and not holds(frozen, x["path"], x.get("value"))]
    if not gone:
        return []
    for x in gone:
        for o in x.get("origins") or [{}]:
            record(quest_root, [], [{"path": x["path"], "value": x.get("value"), "objection": o.get("objection", ""),
                                     "fix": o.get("fix", ""), "check": o.get("check", "")}],
                   reasons={}, default_reason=why, iteration=int(o.get("iteration") or 0))
    return gone


def _short(value: Any, limit: int = 4000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def for_code(quest_root: Path) -> dict[str, Any] | None:
    """The open items as a request that writes or reads code is shown them, or ``None`` when there are none. The text is
    the audit's own, whole."""
    items = read(quest_root)["open"]
    if not items:
        return None
    return {"note": NOTE, "items": [{"part": x["path"], "proposed": x.get("value"), "audit_said": texts(x),
                                     "audit_fix": " / ".join(dict.fromkeys(
                                         str(o.get("fix")) for o in x.get("origins") or [] if o.get("fix"))),
                                     "not_applied_because": x.get("reason", "")} for x in items]}


def writer_note(quest_root: Path) -> str:
    """For the writer: what the audit proposed and the plan does not hold, so the limitations say it."""
    items = read(quest_root)["open"]
    if not items:
        return ""
    return ("The methodology audit proposed changes that FI did not apply, so the study was run without them. Say plainly in "
            "the limitations that they were proposed by the audit and not applied, and do not describe the study as if it "
            "had them:\n" + "\n".join(f"- `{x['path']}`: {_short(texts(x), 400)} Proposed: "
                                      f"{_short(x.get('value'), 1500)} (not applied: {x.get('reason')})" for x in items))
