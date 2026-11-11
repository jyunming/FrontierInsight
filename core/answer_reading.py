"""Reading a model's answer part by part: what FI kept, what it could not read, and what it moved.

A model asked for a plan, a search block or a correction writes a nested answer (a protocol of parts, each a mapping or a
list of entries). Every reader of such an answer used to report what it could not read as one sentence of text, and the
repair that followed found the part to leave out again by matching words in FI's own sentence. A sentence that did not
match cost the whole key, the whole block or the whole protocol. Here the readers report each fault where it is:

* :class:`Fault` names the one part or entry (``path``) that could not be taken, the kind of fault, FI's reason in the
  words the model is asked with, the values allowed (for a closed choice) and the entry as the model wrote it.
* :class:`Moved` records a part moved, unchanged, into a place that was empty (a search's scan written beside the search
  block, moved into it). Nothing is lost by a move, so it is not a fault. A move that would replace something already
  there is not a move: it is a ``conflict``, asked about with both versions.
* :class:`Reading` is the answer as read: ``kept`` (what FI uses), the faults, the moves, and ``nothing_read`` for an
  answer of which nothing could be read (the only case in which an answer is refused whole).

Each entry is read alone by the reader it already had, so one entry that cannot be read costs only that entry; the
checks that look at a part as a whole (every value of a grid is run by some sub-study) then run on the entries kept.
:func:`ask_text` writes the one request that names only the faulted paths, and :func:`write_record` keeps a reading in
the caller's durable record, raising when it cannot be written (a lost record would let a step repeat or a part go
missing without a word)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

#: The kinds of fault. ``unreadable``: the entry is not in a form FI can read; ``not_allowed``: a closed choice given a
#: value outside it (``allowed`` lists the values); ``missing``: a part the reading needs is absent; ``conflict``: two
#: versions of one thing, so taking either discards the other; ``duplicate``: one name given twice; ``together``: the
#: entries read one by one, but the ones kept do not stand together; ``outside_scope``: a change outside what was asked;
#: ``form``: a reply that cannot be placed at all.
KINDS = ("unreadable", "not_allowed", "missing", "conflict", "duplicate", "together", "outside_scope", "form")


@dataclass(frozen=True)
class Fault:
    """One part or entry of an answer FI could not take, where it is (``path``, such as
    ``protocol.optimisation.constraints[2]``), why (``reason``, FI's words), the values a closed choice allows, and the
    entry as the model wrote it. ``required``: a part the study cannot do without (a search's objective, a design
    variable's baseline value); it is never dropped without being said, and an unreadable one stops what needs it."""

    path: str
    kind: str
    reason: str
    allowed: tuple[str, ...] | None = None
    as_written: Any = None
    required: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}")


@dataclass(frozen=True)
class Moved:
    """A part moved, unchanged, from ``path`` to ``to``, a place that was empty (``into_empty``); or, with
    ``into_empty`` false, a part written twice alike, kept once at ``to`` (nothing of it is lost either way)."""

    path: str
    to: str
    into_empty: bool = True


@dataclass
class Reading:
    """An answer as read: ``kept`` (what FI uses, ``None`` only when nothing could be read), the faults, the moves, and
    whether nothing of it could be read."""

    kept: Any
    faults: list[Fault] = field(default_factory=list)
    moved: list[Moved] = field(default_factory=list)
    nothing_read: bool = False

    @property
    def clean(self) -> bool:
        """Read whole: nothing faulted, nothing refused."""
        return not self.faults and not self.nothing_read

    def paths(self) -> list[str]:
        """The faulted paths, each once, in the order found."""
        return list(dict.fromkeys(f.path for f in self.faults))

    def required(self) -> list[Fault]:
        """The faults of parts the study cannot do without."""
        return [f for f in self.faults if f.required]


# --- paths -------------------------------------------------------------------------------------------------------------

_SIMPLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SEGMENT = re.compile(r'\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]|\["((?:[^"\\]|\\.)*)"\]')


def join(path: str, key: Any) -> str:
    """``path`` extended by one step: a mapping key (``.name``, or ``["a name"]`` when it is not a plain word) or a list
    entry by its number counted from 1 (``[2]``, as FI's sentences say "entry 2")."""
    if isinstance(key, int) and not isinstance(key, bool):
        return f"{path}[{key}]"
    key = str(key)
    if _SIMPLE.match(key):
        return f"{path}.{key}" if path else key
    return f'{path}[{json.dumps(key, ensure_ascii=False)}]'


def split(path: str) -> list[Any]:
    """The steps of ``path``: text for a mapping key, a number (from 1) for a list entry."""
    head = re.match(r"^[A-Za-z_][A-Za-z0-9_]*", path or "")
    if head is None:
        return []
    steps: list[Any] = [head.group(0)]
    rest = path[head.end():]
    while rest:
        m = _SEGMENT.match(rest)
        if m is None:
            break
        if m.group(1) is not None:
            steps.append(m.group(1))
        elif m.group(2) is not None:
            steps.append(int(m.group(2)))
        else:
            steps.append(json.loads(f'"{m.group(3)}"'))
        rest = rest[m.end():]
    return steps


_ABSENT = object()


def value_at(value: Any, path: str, root: str = "") -> Any:
    """What ``value`` holds at ``path`` (``root``: the name of ``value`` itself, the first step of ``path``, such as
    ``protocol``); ``None`` when there is nothing there."""
    found = _at(value, path, root)
    return None if found is _ABSENT else found


def _at(value: Any, path: str, root: str) -> Any:
    steps = split(path)
    if root:
        if not steps or steps[0] != root:
            return _ABSENT
        steps = steps[1:]
    for step in steps:
        if isinstance(step, int) and isinstance(value, list) and 0 < step <= len(value):
            value = value[step - 1]
        elif isinstance(step, str) and isinstance(value, dict):
            match = next((k for k in value if str(k) == step), _ABSENT)
            if match is _ABSENT:
                return _ABSENT
            value = value[match]
        else:
            return _ABSENT
    return value


# --- reading -----------------------------------------------------------------------------------------------------------

Schema = Callable[[dict[str, Any], Any], Reading]


def read_answer(answer: Any, base: Any, schema: Schema, *, root: str = "answer") -> Reading:
    """``answer`` read by ``schema`` (a reader of one kind of answer: :func:`core.plan.read_protocol`,
    :func:`core.optimisation_plan.read_block`), entry by entry, against ``base`` (what the plan has now, or ``None``).
    An answer that is not a mapping is not read at all (``nothing_read``); everything else is the schema's reading, in
    which each fault names its own path. ``answer`` is not changed."""
    if not isinstance(answer, dict):
        return Reading(kept=None, nothing_read=True, faults=[Fault(
            root, "form", f"`{root}` must be a mapping of names to values", as_written=answer)])
    return schema(answer, base)


# --- what is said ------------------------------------------------------------------------------------------------------


def shown(value: Any, limit: int = 500) -> str:
    """``value`` as written, in one short line."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


def ask_line(fault: Fault) -> str:
    """One line of a request about one fault: where, why, as written, and the values allowed."""
    line = f"- `{fault.path}`: {fault.reason}"
    if fault.as_written is not None:
        line += (f" Both versions: {shown(fault.as_written)}." if fault.kind == "conflict"
                 else f" As written: {shown(fault.as_written)}.")
    if fault.allowed:
        line += " Allowed: " + ", ".join(f"`{a}`" for a in fault.allowed) + "."
    return line


def ask_text(faults: Iterable[Fault], lead: str = "") -> str:
    """The one request about an answer's faults: it names only the faulted paths, each with FI's reason, the entry as
    written and the values allowed, and says that everything else of the answer is kept as it was. Empty for no fault."""
    seen: set[tuple[str, str]] = set()
    lines = []
    for fault in faults:
        if (fault.path, fault.reason) in seen:
            continue
        seen.add((fault.path, fault.reason))
        lines.append(ask_line(fault))
    if not lines:
        return ""
    return ((lead.strip() + "\n") if lead.strip() else "") + (
        "Only these parts of your answer could not be read; FI keeps everything else as you wrote it. Write each of "
        "them again, at the same place, and nothing else:\n" + "\n".join(lines))


def entries_as_written(faults: Iterable[Fault], *, you: bool = False) -> str:
    """The faulted entries as written, in one sentence (``""`` when none carries its entry): the form FI's run.log and
    requests use (``the entry as written: ...``)."""
    found: list[str] = []
    for fault in faults:
        if fault.as_written is None:
            continue
        entry = shown(fault.as_written)
        if entry not in found:
            found.append(entry)
    if not found:
        return ""
    lead = (("the entry as you wrote it" if you else "the entry as written") if len(found) == 1
            else ("the entries as you wrote them" if you else "the entries as written"))
    return f"{lead}: " + " | ".join(found)


# --- the durable record ------------------------------------------------------------------------------------------------


def record_entry(reading: Reading, *, step: str, asked: Iterable[str] = (), dropped: Iterable[str] = ()) -> dict[str, Any]:
    """The record of one reading: the step that read it, the paths faulted (each with its kind and reason), the paths
    asked about and the paths left out, and the moves."""
    return {
        "step": step,
        "nothing_read": reading.nothing_read,
        "faults": [{"path": f.path, "kind": f.kind, "reason": f.reason, **({"required": True} if f.required else {})}
                   for f in reading.faults],
        "moved": [{"path": m.path, "to": m.to, **({} if m.into_empty else {"same_as_there": True})}
                  for m in reading.moved],
        "asked": list(dict.fromkeys(asked)),
        "dropped": list(dict.fromkeys(dropped)),
    }


def write_record(path: Path, entry: dict[str, Any]) -> None:
    """Add ``entry`` to the JSON list kept at ``path`` (written whole, then moved into place). A record that cannot be
    written raises: the caller's step stops as on any disk fault, and a resume repeats it. A file that holds something
    other than a list is not overwritten (that raises too)."""
    path = Path(path)
    entries: list[Any] = []
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"{path} does not hold a list of readings")
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
