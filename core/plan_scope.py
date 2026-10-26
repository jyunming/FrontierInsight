"""Which parts of the plan a request to its model concerns.

When FI asks the plan's model to fix something (a study too long for the time allowed, a check with no source, a missing
oracle ...), the model is shown the plan read-only and asked for ONE small JSON object of the parts that request concerns;
FI is the only writer of the YAML (``Engine._revise_plan_parts``). A :class:`PartScope` says which parts those are, so a
reply that names anything else is refused with a reason (``translate``) and the rest of the plan stays byte for byte as it
was.

This module is pure (no model calls, no engine state). A scope names parts of ``protocol`` in three ways:

* ``parts``: a part the reply may write whole (``grid``, ``criteria`` ...).
* ``inside``: a mapping part (``optimisation``, ``model``) of which the reply may change only some keys; the other keys of
  the mapping are the plan's own and are kept.
* ``entries``: a list part of entries with a key (``oracles`` by ``name``, ``simplifications`` by ``id``). The reply names
  only the entries it concerns: an entry the plan has changes only in ``fields`` (any field when ``fields`` is ``None``),
  an entry it does not have is added when ``may_add``; every entry the reply does not name is kept as it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntryScope:
    key: str  # the field that names an entry: "name", "id"
    fields: frozenset[str] | None = None  # the fields of an existing entry the reply may change; None: any
    may_add: bool = False  # whether the reply may add an entry the plan does not have


@dataclass(frozen=True)
class PartScope:
    parts: tuple[str, ...] = ()
    inside: dict[str, tuple[str, ...]] = field(default_factory=dict)
    entries: dict[str, EntryScope] = field(default_factory=dict)

    def names(self) -> list[str]:
        """Every protocol part the reply may name."""
        return list(dict.fromkeys([*self.parts, *self.inside, *self.entries]))

    def describe(self) -> str:
        """The scope in the request's words, appended to what is asked."""
        lines = ["Change only these parts of the plan, and answer with them alone (one JSON object, the parts under "
                 "`protocol`): " + ", ".join(f"`{n}`" for n in self.names()) + ". A reply that names any other part is "
                 "not used."]
        for part, keys in self.inside.items():
            lines.append(f"For `{part}`, unlike the other parts, write only the keys you change, among {', '.join(f'`{k}`' for k in keys)}; "
                         f"every other key of `{part}` stays as the plan has it.")
        for part, entry in self.entries.items():
            what = (f"change only {', '.join(f'`{f}`' for f in sorted(entry.fields))} of an entry the plan has"
                    if entry.fields is not None else "give an entry the plan has in full")
            lines.append(f"For `{part}`, unlike the other parts, write only the entries this request concerns, each with its `{entry.key}`: "
                         f"{what}" + ("; a new entry has a new `" + entry.key + "`" if entry.may_add else
                                      "; no entry is added") + ". Entries you do not write stay as the plan has them.")
        return "\n\n".join(lines)


def _key_of(item: Any, key: str) -> str:
    return str(item.get(key) if isinstance(item, dict) else "").strip()


def translate(scope: PartScope, old_protocol: dict[str, Any], answer: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``(answer, problems)``: the model's answer (a design block's shape, the parts under ``protocol``) made into the whole
    parts to write, each ``inside`` and ``entries`` part merged into the plan's own; ``problems`` (plain sentences, the
    reason the reply goes back with) when it names a part or a key outside ``scope`` or a part in a form that cannot be merged."""
    problems: list[str] = []
    asked = scope.names()
    outside_top = sorted(k for k in answer if k != "protocol")
    if outside_top:
        problems.append("it names " + ", ".join(f"`{k}`" for k in outside_top) + ", which this request does not concern")
    new = answer.get("protocol") if isinstance(answer.get("protocol"), dict) else {}
    outside = sorted(k for k in new if k not in asked)
    if outside:
        problems.append("it names " + ", ".join(f"`protocol.{k}`" for k in outside) + ", which this request does not concern "
                        "(the parts asked are " + ", ".join(f"`{n}`" for n in asked) + ")")
    out: dict[str, Any] = {k: new[k] for k in scope.parts if k in new}
    for part, keys in scope.inside.items():
        if part not in new:
            continue
        value, old = new[part], old_protocol.get(part)
        if not isinstance(value, dict):
            problems.append(f"`protocol.{part}` must be a mapping holding only the keys you change")
            continue
        extra = sorted(k for k in value if k not in keys)
        if extra:
            problems.append(f"`protocol.{part}` names " + ", ".join(f"`{k}`" for k in extra) + ", which this request does not "
                            "concern (the keys asked are " + ", ".join(f"`{k}`" for k in keys) + ")")
            continue
        out[part] = {**(old if isinstance(old, dict) else {}), **value}
    for part, entry in scope.entries.items():
        if part not in new:
            continue
        value, old = new[part], old_protocol.get(part)
        if not isinstance(value, list) or not all(isinstance(e, dict) and _key_of(e, entry.key) for e in value):
            problems.append(f"`protocol.{part}` must be a list of entries, each with its `{entry.key}`")
            continue
        merged = [dict(e) if isinstance(e, dict) else e for e in (old if isinstance(old, list) else [])]
        index = {_key_of(e, entry.key).lower(): i for i, e in enumerate(merged) if isinstance(e, dict)}
        for item in value:
            name = _key_of(item, entry.key)
            at = index.get(name.lower())
            if at is None:
                if not entry.may_add:
                    problems.append(f"`protocol.{part}` names `{name}`, which the plan does not have, and no entry is added here")
                    continue
                merged.append(dict(item))
                index[name.lower()] = len(merged) - 1
                continue
            changed = sorted(k for k, v in item.items() if k != entry.key and merged[at].get(k) != v)
            if entry.fields is not None:
                barred = [k for k in changed if k not in entry.fields]
                if barred:
                    problems.append(f"`protocol.{part}` entry `{name}` changes " + ", ".join(f"`{k}`" for k in barred)
                                    + ", which this request does not concern (it may change "
                                    + ", ".join(f"`{f}`" for f in sorted(entry.fields)) + ")")
                    continue
            merged[at] = {**merged[at], **{k: item[k] for k in changed}}
        out[part] = merged
    return {"protocol": out}, problems
