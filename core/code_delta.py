"""What a redesign changed in the plan, for the code that was built from the plan before it.

When a plan is revised after code was written, only some of what the code reads can have changed. FI keeps, in
``.fi/code_built_from.json``, the parts of the plan the scripts were written from (the plan's settings, metrics,
thresholds, the search block and so on, each as its own text, and the signatures of the model's functions). On the next
build :func:`compare` says which of those parts differ, from FI's own record and not from a reading of the new plan's
prose. The model's functions have a record of their own (:func:`core.function_steps.function_hash` per function); this
one is for the two scripts that run the study (``simulate.py``, ``experiment.py``):

* no part changed: the scripts are kept as they are (no request, every repair a person or FI accepted stays);
* some parts changed: the model is shown those parts, before and after, and asked only for the functions of the scripts
  that must change (:func:`core.script_reply.resolve` / :func:`core.finding_repair.splice` put them in);
* nothing recorded (a quest from before this record, or a first build): nothing can be said, and the code is written whole
  as it always was, said in ``run.log``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECORD = Path(".fi") / "code_built_from.json"
#: The version of what the record holds; a record of another version says nothing (the scripts are written whole).
SCHEMA = 2
#: Parts of the protocol the scripts do not read: the model (its functions have a record of their own), and what only the
#: writer reads (the simplifications, the topic map, the definitions and the criteria). The checks (``oracles``) ARE read:
#: the simulation answers each check with an ``oracle()`` or a case FI runs, and a check that needs a new output needs
#: new code, so a change to them is a change to what the scripts are built from.
NOT_READ_BY_SCRIPTS = frozenset({"model", "simplifications", "topic_map", "definitions", "criteria"})
#: The most characters of one part's text shown, before and after, to the request that updates the scripts.
SHOWN_CHARS = 6000


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=1, default=str, ensure_ascii=False)


def parts(design: dict[str, Any] | None, functions: dict[str, str] | None = None, package: str = "") -> dict[str, str]:
    """``{part name: its text}`` for what the scripts are written from: each key of the design's protocol that the scripts
    read, the rest of the design, the model's functions (``functions``: name to the hash it was built from, so a function
    whose equation changed under the same signature changes this part, :func:`core.function_steps.function_hash`) and the
    name of the package they are in."""
    design = design if isinstance(design, dict) else {}
    protocol = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    out = {f"protocol.{k}": _canon(v) for k, v in protocol.items() if k not in NOT_READ_BY_SCRIPTS}
    rest = {k: v for k, v in design.items() if k != "protocol"}
    if rest:
        out["design"] = _canon(rest)
    if functions:
        out["functions"] = _canon(sorted(functions.items()))
    if package:
        out["package"] = package
    return out


def load(quest_root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(quest_root: Path, now: dict[str, str], *, package: str = "") -> None:
    path = Path(quest_root) / RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": SCHEMA, "package": package, "parts": now}, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    except OSError:
        pass


@dataclass
class Delta:
    """``known``: FI has a record of what the scripts were built from. ``changed``: the names of the parts that differ
    (a part that appeared or disappeared counts). ``old``/``new``: their texts. ``reason``: why nothing is known."""
    known: bool
    changed: list[str] = field(default_factory=list)
    old: dict[str, str] = field(default_factory=dict)
    new: dict[str, str] = field(default_factory=dict)
    reason: str = ""


def compare(quest_root: Path, now: dict[str, str]) -> Delta:
    record = load(quest_root)
    if record and record.get("schema") != SCHEMA:
        return Delta(False, reason="the record of what the scripts were built from was written by another version of FI")
    before = record.get("parts") if isinstance(record.get("parts"), dict) else None
    if not before:
        return Delta(False, reason="there is no record of what the scripts were built from")
    changed = sorted(k for k in {*before, *now} if before.get(k) != now.get(k))
    return Delta(True, changed, {k: str(before[k]) for k in changed if k in before},
                 {k: now[k] for k in changed if k in now})


def describe(delta: Delta, limit: int = SHOWN_CHARS) -> str:
    """What changed, for the request: each changed part, as it was and as it is now."""
    def cut(text: str) -> str:
        return text if len(text) <= limit else text[:limit] + "\n... (cut)"

    blocks = []
    for name in delta.changed:
        blocks.append(f"## {name}\nbefore:\n{cut(delta.old.get(name, '(not there)'))}\n\nafter:\n{cut(delta.new.get(name, '(removed)'))}")
    return "\n\n".join(blocks)


def splice_functions(old: str, reply: str, *, scope: set[str] | None = None) -> tuple[str | None, list[str], str]:
    """The one helper any repair of a code file uses to put a model's reply in: ``(new text, names put in, why not)``.

    ``reply`` holds the functions to replace or add (a fenced block or bare source, each ``def`` whole). Each function the
    file already has (and, when ``scope`` is given, that is in it) is replaced by name; a function the file lacks is added
    at the end; imports the reply brings are added; everything else in ``old`` stays byte for byte. ``(None, [], why)``,
    with a reason that can be sent back to the model, when the reply holds no function, does not parse, changes a
    function's parameters (only trailing parameters with defaults may be added), names none of ``scope``, or would leave
    a file that does not parse. Classes and module-level statements are not spliced (a reply that needs them is a
    whole-file reply)."""
    from . import finding_repair as _finding

    return _finding.splice(old, reply, scope, None, limit=None)
