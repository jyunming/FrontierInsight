"""A design is changed one part at a time: only the parts a reason concerns, never the whole object.

Two writers used to hand back a WHOLE design object (the methodology audit's ``amended_design``, and the design step on a
redesign), so a part no objection and no note named could change silently. This module is the pure side of the part
pattern the plan already uses (``core/plan.py``, ``agents/plan_parts.md``):

* a *part* is a path into the design, ``"method"``, ``"protocol.grid"``, ``"protocol.grid.x"``;
* :func:`apply_changes` applies a list of ``(path, value)`` and refuses a path that is not a part of a design;
* :func:`changed_parts` lists the parts one design changed against another (to hold a whole-object reply to the parts a
  reason named);
* :func:`allowed_for_precision` / :func:`allowed_for_notes` say which parts a redesign may change, from FI's own record
  (the route) or from the design's own part names in the notes;
* :func:`merge_parts` merges a reply of parts into the previous design and returns what it refused.

Nothing here calls a model.
"""
from __future__ import annotations

import copy
import re
from typing import Any

#: The top-level keys of a design a change may name. ``rationale`` is the model's own account and is never a part.
DESIGN_KEYS = (
    "hypothesis", "study_type", "variables", "method", "expected_outcome", "figures_planned", "dependencies", "protocol",
)

#: The protocol parts that size the study (how many trials per setting, how dense the grid, how it is divided).
SIZING_PARTS = ("protocol.runs_per_setting", "protocol.min_runs_per_setting", "protocol.grid", "protocol.sub_studies")

#: Words a note uses for a part, besides the part's own name. The design's own vocabulary and plain statistics words,
#: never a topic's.
_WORDS: dict[str, str] = {
    "hypothesis": r"hypothes",
    "variables": r"variables?\b|independent variable|dependent variable",
    "method": r"\bmethod\b|baseline|comparator|evaluator|circular",
    "expected_outcome": r"expected[ _]outcome",
    "figures_planned": r"\bfigures?\b|\bplots?\b",
    "dependencies": r"dependenc",
    "protocol.grid": r"\bgrid\b|\bsweep|single[- ]point",
    "protocol.runs_per_setting": r"runs[ _]per[ _]setting|\btrials\b|replicat|\brepeats\b|sample size",
    "protocol.thresholds": r"threshold|cut-?off",
    "protocol.oracles": r"oracle|closed form|invariant",
    "protocol.seed_policy": r"\bseeds?\b|random stream",
    "protocol.ci_method": r"confidence interval|\bci\b|interval",
    "protocol.failure_policy": r"failed run|diverg|\bnan\b",
    "protocol.acceptance": r"acceptance",
    "protocol.metrics": r"\bmetrics?\b",
    "protocol.sub_studies": r"sub[- ]?stud",
    "protocol.numerical_axes": r"convergence|numerical axes",
    "protocol.constants": r"\bconstants?\b",
}

#: The identifiers the review gives for a non-negotiable failure (``must_flag_hits``) and the part each concerns.
_FLAGS: dict[str, tuple[str, ...]] = {
    "circular_evaluation": ("method",),
    "single_point_eval": ("protocol.grid",),
    "weak_baseline_no_rerun": ("method",),
    "pseudo_units": ("variables",),
}


def valid_part(path: Any, known: Any = ()) -> bool:
    """A path that names a part of a design: its first segment is a design key (or a key the design already has, such as
    ``result_assertions``), and under ``protocol`` the second is a plain name (the protocol's readers refuse a part they
    do not know). No empty segment, no index."""
    if not isinstance(path, str) or not path.strip():
        return False
    segs = path.strip().split(".")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s.strip()) for s in segs):
        return False
    return segs[0] in DESIGN_KEYS or segs[0] in known


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
        if not valid_part(path, design):
            refused.append({"key": str(path), "reason": "it is not a part of a design (a design key, or `protocol.<part>`)"})
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
    piece of it (``protocol`` names every protocol part, ``protocol.grid`` names ``protocol.grid.x``), or the other way
    round (``protocol.grid.x`` names ``protocol.grid``)."""
    text = (where or "").lower().replace("`", " ")
    segs = part.split(".")
    for n in range(1, len(segs) + 1):
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


def _text_of(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return " ".join(_text_of(v) for v in item.values())
    if isinstance(item, (list, tuple)):
        return " ".join(_text_of(v) for v in item)
    return ""


def allowed_for_notes(notes: list[Any], flags: list[str] | None = None) -> list[str]:
    """The parts the notes (a review's weaknesses and suggestions, a person's words) name, by the design's own part names
    and the review's failure identifiers. Empty when no note names a part: the caller then allows the whole design and says so."""
    text = " ".join(_text_of(n) for n in notes).lower().replace("_", " ")
    out: list[str] = []
    for part, rx in _WORDS.items():
        if re.search(rx, text) and part not in out:
            out.append(part)
    for f in flags or []:
        for part in _FLAGS.get(str(f), ()):
            if part not in out:
                out.append(part)
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


def merge_parts(
    previous: dict[str, Any], reply: dict[str, Any], allowed: list[str] | None,
) -> tuple[dict[str, Any], list[str], list[dict[str, str]]]:
    """``(design, applied, refused)``: ``previous`` with the parts of ``reply`` that are in ``allowed`` replaced and every
    other part exactly as it was. ``allowed`` ``None`` allows every part of a design. A part is allowed when it is listed, or
    lies under a listed one (``protocol.grid.x`` under ``protocol.grid``), or a listed one lies under it only for a whole
    ``protocol`` reply that FI has already flattened. ``refused``: one ``{"key", "reason"}`` per part not taken."""
    changes: list[tuple[str, Any]] = []
    refused: list[dict[str, str]] = []
    for path, value in flatten_reply(reply):
        if path == "rationale":
            continue
        if not valid_part(path, previous):
            refused.append({"key": path, "reason": "it is not a part of a design"})
        elif allowed is not None and not any(path == a or path.startswith(a + ".") for a in allowed):
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
