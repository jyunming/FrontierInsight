"""A design is changed one part at a time: only the parts a reason concerns, never the whole object.

Two writers used to hand back a WHOLE design object (the methodology audit's ``amended_design``, and the design step on a
redesign), so a part no objection and no note named could change silently. This module is the pure side of the part
pattern the plan already uses (``core/plan.py``, ``agents/plan_parts.md``):

* a *part* is a path into the design, ``"method"``, ``"protocol.grid"``, ``"protocol.grid.x"``;
* :func:`apply_changes` applies a list of ``(path, value)`` and refuses a path that is not a part of a design;
* :func:`changed_parts` lists the parts one design changed against another (to hold a whole-object reply to the parts a
  reason named);
* :func:`allowed_for_precision` / :func:`cited_parts` say which parts a redesign may change: the route (FI's own record), or the
  paths a reviewer cited, checked against the real design (no word of a note is matched);
* :func:`merge_parts` merges a reply of parts into the previous design and returns what it refused.

Nothing here calls a model.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from .plan import PROTOCOL_PARTS

#: The top-level keys of a design a change may name. ``rationale`` is the model's own account and is never a part.
DESIGN_KEYS = (
    "hypothesis", "study_type", "variables", "method", "expected_outcome", "figures_planned", "dependencies", "protocol",
)

#: The protocol parts that size the study (how many trials per setting, how dense the grid, how it is divided).
SIZING_PARTS = ("protocol.runs_per_setting", "protocol.min_runs_per_setting", "protocol.grid", "protocol.sub_studies")

#: Protocol parts a design may gain that the draft does not have yet (the audit adds an entry for check 16, for example).
#: Any other sub-key of `protocol` must already be in the draft: an unknown one (a misspelt part) is refused.
_NEW_PROTOCOL_PARTS = ("definitions", "labels", "simplifications", "numerical_settings", "result_assertions")


def invalid_reason(path: Any, design: dict[str, Any]) -> str:
    """``""`` when ``path`` names a part of ``design`` (every step but the last must exist in it; under ``protocol`` the
    part itself must be one the protocol has or can gain), else the reason it does not. The WHOLE path is checked."""
    if not isinstance(path, str) or not path.strip():
        return "it is empty"
    segs = path.strip().split(".")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", x) for x in segs):
        return "it is not a plain path of names"
    if segs[0] not in DESIGN_KEYS and segs[0] not in design:
        return f"`{segs[0]}` is not a part of this design"
    proto = design.get("protocol")
    if segs[0] == "protocol" and len(segs) > 1:
        known = {*PROTOCOL_PARTS, *_NEW_PROTOCOL_PARTS, *(proto if isinstance(proto, dict) else {})}
        if segs[1] not in known:
            return f"`protocol.{segs[1]}` is not a part of the protocol (a misspelt name?)"
    node: Any = design
    for i, x in enumerate(segs[:-1]):
        if i == 1 and segs[0] == "protocol" and not (isinstance(node, dict) and x in node):
            node = {}  # a protocol part the design may gain: nothing below it exists yet
            continue
        nxt = node.get(x) if isinstance(node, dict) else None
        if nxt is None:
            if i == 0 and segs[0] == "protocol":
                node = {}
                continue
            return f"`{'.'.join(segs[:i + 1])}` is not in this design"
        if not isinstance(nxt, dict):
            return f"`{'.'.join(segs[:i + 1])}` is not a mapping, so nothing can be set under it"
        node = nxt
    return ""


def valid_part(path: Any, design: Any = None) -> bool:
    return not invalid_reason(path, design if isinstance(design, dict) else {})


def _set(root: dict[str, Any], segs: list[str], value: Any) -> str:
    """Set ``value`` at ``segs`` under ``root``; ``""`` or the reason it cannot be set."""
    node: Any = root
    for i, s in enumerate(segs[:-1]):
        nxt = node.get(s)
        if nxt is None and i >= 1:
            nxt = node[s] = {}
        if not isinstance(nxt, dict):
            return f"`{'.'.join(segs[:i + 1])}` is not a mapping, so a part under it cannot be set on its own"
        node = nxt
    node[segs[-1]] = copy.deepcopy(value)
    return ""


def apply_changes(design: dict[str, Any], changes: list[tuple[str, Any]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """``(design, refused)``: a copy of ``design`` with each ``(path, value)`` applied, in order, and one ``{"key",
    "reason"}`` per change that was not (a path that is not a part of a design, or one that runs through a value that is
    not a mapping). ``design`` itself is not changed."""
    out = copy.deepcopy(design)
    refused: list[dict[str, str]] = []
    for path, value in changes:
        why = invalid_reason(path, out)
        if why:
            refused.append({"key": str(path), "reason": why})
            continue
        if path.strip() == "protocol" and not isinstance(value, dict):
            refused.append({"key": "protocol", "reason": "`protocol` must be a mapping"})
            continue
        why = _set(out, path.strip().split("."), value)
        if why:
            refused.append({"key": path.strip(), "reason": why})
    return out, refused


def changed_parts(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """The parts ``after`` changes against ``before``: each top-level key that differs, and, for ``protocol``, each
    protocol part that differs (added, changed or left out). ``rationale`` is not a part."""
    out: list[str] = []
    for k in sorted({*before, *after} - {"rationale"}):
        if before.get(k) == after.get(k):
            continue
        b, a = before.get(k), after.get(k)
        if k == "protocol" and isinstance(b, dict) and isinstance(a, dict):
            out.extend(f"protocol.{p}" for p in sorted({*b, *a}) if b.get(p) != a.get(p))
        else:
            out.append(k)
    return out


def names(where: str, part: str) -> bool:
    """Whether ``where`` (an objection's pointer, free text a model wrote) names ``part``: the part's path, or a leading
    piece of it (``protocol.grid`` names ``protocol.grid.x``), or the other way round (``protocol.grid.x`` names
    ``protocol.grid``). ``protocol`` alone names no protocol part: the pointer must say which."""
    text = (where or "").lower().replace("`", " ")
    segs = part.split(".")
    for n in range(2 if segs[0] == "protocol" and len(segs) > 1 else 1, len(segs) + 1):
        prefix = ".".join(segs[:n])
        if re.search(rf"(?<![\w.]){re.escape(prefix)}(?![\w])", text):
            return True
    # the pointer may go deeper than the part (`protocol.grid.x` for the part `protocol.grid`)
    return bool(re.search(rf"(?<![\w.]){re.escape(part)}\.", text))


def covered(part: str, wheres: list[str], paths: list[str]) -> bool:
    """Whether a changed ``part`` is named: by an objection's ``where``, or by the path of a change an objection carries."""
    if any(names(w, part) for w in wheres):
        return True
    for p in paths:
        if p == part or p.startswith(part + ".") or part.startswith(p + "."):
            return True
    return False


def allowed_for_precision() -> list[str]:
    """The parts a redesign after a missed precision target may change: the ones that size the study."""
    return list(SIZING_PARTS)


def part_paths(design: dict[str, Any]) -> list[str]:
    """The paths a note may cite, listed from the design object itself: each top-level key and each part of the protocol."""
    out: list[str] = []
    for k, v in design.items():
        if k == "rationale":
            continue
        if k == "protocol" and isinstance(v, dict):
            out.extend(f"protocol.{p}" for p in v)
        else:
            out.append(k)
    return out


def cited_parts(cited: Any, design: dict[str, Any]) -> list[str]:
    """The paths in ``cited`` (what a reviewer or a mapping wrote) that are parts of ``design``, the whole path checked,
    each given as the listed path of :func:`part_paths` it lies in. The others are dropped."""
    out: list[str] = []
    listed = part_paths(design)
    for c in cited if isinstance(cited, list) else []:
        if not isinstance(c, str):
            continue
        c = c.strip().strip("`")
        if invalid_reason(c, design) or not _get_part(design, c)[0]:
            continue
        top = next((p for p in listed if c == p or c.startswith(p + ".")), None)
        if top and top not in out:
            out.append(top)
    return out


def flatten_reply(reply: dict[str, Any]) -> list[tuple[str, Any]]:
    """The ``(path, value)`` pairs of a reply of parts: ``{"method": ..., "protocol": {"grid": ...}}`` gives
    ``("method", ...)`` and ``("protocol.grid", ...)``. A key written with a dot (``"protocol.grid"``) is a path too."""
    out: list[tuple[str, Any]] = []
    for k, v in reply.items():
        if k == "protocol" and isinstance(v, dict):
            out.extend((f"protocol.{p}", pv) for p, pv in v.items())
        else:
            out.append((str(k), v))
    return out


def accepted_needs(
    previous: dict[str, Any], reply: dict[str, Any], allowed: list[str] | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """``(accepted, refused)`` of the extra parts a reply asks to change (``needs: [{path, why}]``). A need is accepted when
    its path is a valid, specific part of the design and it has a reason. It is refused when it has no reason, is not a part
    of the design, is the whole of ``protocol``, or is an ancestor of a part already allowed (it would widen the allowed
    parts to everything under it)."""
    ok: list[dict[str, str]] = []
    bad: list[dict[str, str]] = []
    raw = reply.get("needs") if isinstance(reply, dict) else None
    for n in raw if isinstance(raw, list) else []:
        path = str(n.get("path") or "").strip() if isinstance(n, dict) else ""
        why = str(n.get("why") or "").strip() if isinstance(n, dict) else ""
        reason = (invalid_reason(path, previous) if path else "it has no path"
                  or "")
        if not reason and not why:
            reason = "it gives no reason"
        elif not reason and (path == "protocol" or any(a.startswith(path + ".") for a in allowed or [])):
            reason = "it is a whole group of parts, not one specific part"
        elif not reason and allowed is not None and any(path == a or path.startswith(a + ".") for a in allowed):
            continue  # already allowed: not an extra part
        if reason:
            bad.append({"key": path or "(no path)", "reason": reason})
        else:
            ok.append({"path": path, "why": why})
    return ok, bad


def merge_parts(
    previous: dict[str, Any], reply: dict[str, Any], allowed: list[str] | None,
) -> tuple[dict[str, Any], list[str], list[dict[str, str]]]:
    """``(design, applied, refused)``: ``previous`` with the parts of ``reply`` that are in ``allowed`` replaced and every
    other part exactly as it was. ``allowed`` ``None`` allows every part of a design. A part is allowed when it is listed, or
    lies under a listed one (``protocol.grid.x`` under ``protocol.grid``), or a listed one lies under it only for a whole
    ``protocol`` reply that FI has already flattened. ``refused``: one ``{"key", "reason"}`` per part not taken."""
    changes: list[tuple[str, Any]] = []
    refused: list[dict[str, str]] = []
    extra = [n["path"] for n in accepted_needs(previous, reply, allowed)[0]]
    for path, value in flatten_reply({k: v for k, v in reply.items() if k != "needs"}):
        if path == "rationale":
            continue
        why = invalid_reason(path, previous)
        if why:
            refused.append({"key": path, "reason": why})
        elif allowed is not None and not any(path == a or path.startswith(a + ".") for a in [*allowed, *extra]):
            refused.append({"key": path, "reason": "it was not one of the parts asked for"})
        else:
            changes.append((path, value))
    design, bad = apply_changes(previous, changes)
    refused.extend(bad)
    return design, changed_parts(previous, design), refused


def _get_part(design: dict[str, Any], path: str) -> tuple[bool, Any]:
    node: Any = design
    for s in path.split("."):
        if not isinstance(node, dict) or s not in node:
            return False, None
        node = node[s]
    return True, node


def amended_from_audit(
    draft: dict[str, Any], objections: list[Any], whole: Any = None,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    """``(amended, refused, applied)``: the draft design with the changes the audit's objections carry
    (``objection["changes"]``: ``{"path", "value"}`` each) and nothing else. ``whole`` is a whole amended design an older
    reply may still carry: only the parts of it that differ from the draft AND are named (by an objection's ``where`` or
    the path of one of its changes) are taken. ``refused``: one ``{"key", "reason"}`` per change or part not taken; a part
    changed with no objection naming it is refused with that reason. ``draft`` is not changed."""
    wheres: list[str] = []
    changes: list[tuple[str, Any]] = []
    for o in objections or []:
        if not isinstance(o, dict):
            continue
        wheres.append(str(o.get("where") or ""))
        for c in o.get("changes") or []:
            if isinstance(c, dict) and isinstance(c.get("path"), str) and "value" in c:
                changes.append((c["path"].strip(), c["value"]))
    amended, refused = apply_changes(draft, changes)
    paths = [p for p, _ in changes]
    if isinstance(whole, dict) and whole:
        for part in changed_parts(draft, whole):
            if part in paths or any(part.startswith(p + ".") for p in paths):
                continue  # the objection's own change to it is the one made
            if not covered(part, wheres, paths):
                refused.append({"key": part, "reason": "no objection named it, so it was left as the draft has it"})
                continue
            found, value = _get_part(whole, part)
            if found:
                amended, bad = apply_changes(amended, [(part, value)])
                refused.extend(bad)
            elif part.startswith("protocol.") and isinstance(amended.get("protocol"), dict):
                amended["protocol"].pop(part.split(".", 1)[1], None)  # the audit left the part out
            else:
                refused.append({"key": part, "reason": "the audit left it out; a design part is not removed this way"})
    return amended, refused, changed_parts(draft, amended)
