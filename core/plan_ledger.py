"""The plan's ledger of what it left out or approximated, and what each metric means.

A plan that drops or approximates something the topic asks for is only honest if it says so. Two keys of the protocol
carry that:

* ``protocol.simplifications``: ``[{id, replaces, because, effect, probe}]``: each thing the plan approximates or leaves
  out (``replaces``), why (``because``), what that could change in the result (``effect``) and one plain sentence of how a
  run could show whether it matters (``probe``).
* ``protocol.definitions``: ``[{metric, defined_as}]``: what each metric is, in one line.

This module is pure (no model calls, no engine state): it reads those keys, lists the plain problems, finds the topic items
that are neither covered nor named, and reads and writes the record ``.fi/ledger.json``
(``{plan_sha, items: [{id, replaces, because, effect, probe, relation_ids, status}]}``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from . import topic_requirements as _topic

#: The file in ``.fi/`` that holds the ledger.
FILE = "ledger.json"
#: The fields every simplification gives.
FIELDS = ("id", "replaces", "because", "effect", "probe")
#: Where a ledger item stands: no relation tests it yet, or one does.
STATUS_UNTESTED = "not_tested"

#: Plain-language text for the plan directive (spliced into the plan request by the engine).
DIRECTIVE_TEXT = (
    "Also write `protocol.simplifications` and `protocol.definitions`. Every item the topic asks for must appear in the "
    "plan as a setting, a metric or an equation; an item you leave out or approximate goes in `simplifications` as "
    '`{"id": "X1", "replaces": "<what the topic asks for>", "because": "<why you do not do it exactly>", '
    '"effect": "<how large a change it makes to the headline result: a number with its unit, from a closed form '
    'or a limiting case worked out at the studied settings, when one gives it; in words only when none does>", '
    '"probe": "<one sentence: how a run could show whether it matters>"}`. A simplification whose size a closed form '
    'gives is also a check in `oracles` or a result the simulation returns, so the run measures it. Every metric and key quantity has an entry in `definitions`. Take each definition from the '
    "literature you found, quote it, and cite it: "
    '`{"metric": "<metric id>", "defined_as": "<one line>", "source": "[n]", "quote": "<the passage from that source that '
    'states the definition, word for word>", "differs": ""}`. If you define it differently from the quote, say so in '
    "simplifications and put that entry's id in `differs`. Do not leave out or approximate silently.")


def _text(value: Any, limit: int = 400) -> str:
    return " ".join(str(value if value is not None else "").replace("`", "'").split())[:limit]


def parse_simplifications(protocol: Any) -> tuple[list[dict[str, str]], list[str]]:
    """``(items, problems)`` for ``protocol.simplifications``. An absent key is no items and no problem. An entry with a
    missing field is kept with what it has and named in the problems; a repeated id is dropped (the first stays)."""
    protocol = protocol if isinstance(protocol, dict) else {}
    raw = protocol.get("simplifications")
    if raw is None or raw == "":
        return [], []
    if not isinstance(raw, list):
        return [], ["`simplifications` must be a list of entries with an id, what it replaces, why, its effect and a probe"]
    items: list[dict[str, str]] = []
    problems: list[str] = []
    seen: set[str] = set()
    for n, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            problems.append(f"simplification {n} must be a mapping, not {type(entry).__name__}")
            continue
        item = {k: _text(entry.get(k)) for k in FIELDS}
        label = item["id"] or f"number {n}"
        for field in FIELDS:
            if not item[field]:
                problems.append(f"simplification {label} has no `{field}`")
        if item["id"]:
            key = item["id"].lower()
            if key in seen:
                problems.append(f"simplification id {item['id']} is used twice")
                continue
            seen.add(key)
        items.append(item)
    return items, problems


def parse_definitions(protocol: Any) -> tuple[list[dict[str, str]], list[str]]:
    """``(items, problems)`` for ``protocol.definitions``; a metric of ``protocol.metrics`` with no definition is a problem."""
    protocol = protocol if isinstance(protocol, dict) else {}
    raw = protocol.get("definitions")
    items: list[dict[str, str]] = []
    problems: list[str] = []
    sim_ids = {i["id"].lower() for i in parse_simplifications(protocol)[0] if i["id"]}
    if raw is not None and raw != "":
        if not isinstance(raw, list):
            problems.append("`definitions` must be a list of entries with a metric and what it is defined as")
        else:
            for n, entry in enumerate(raw, 1):
                if not isinstance(entry, dict):
                    problems.append(f"definition {n} must be a mapping, not {type(entry).__name__}")
                    continue
                item = {"metric": _text(entry.get("metric") or entry.get("quantity"), 100),
                        "defined_as": _text(entry.get("defined_as")), "source": _text(entry.get("source"), 100),
                        "quote": _text(entry.get("quote"), 600), "differs": _text(entry.get("differs"), 40)}
                if not item["metric"]:
                    problems.append(f"definition {n} has no `metric`")
                elif not item["defined_as"]:
                    problems.append(f"the definition of {item['metric']} has no `defined_as`")
                if item["metric"] and not (item["source"] and item["quote"]):
                    problems.append(f"definition not taken from a source: {item['metric']} has no "
                                    f"{'`source`' if not item['source'] else '`quote`'}"
                                    + (" and no `quote`" if not item["source"] and not item["quote"] else ""))
                if item["differs"] and item["differs"].lower() not in sim_ids:
                    problems.append(f"the definition of {item['metric'] or n} says it differs by {item['differs']}, "
                                    "which is not an entry of `simplifications`")
                items.append(item)
    defined = {_topic._name_key(i["metric"]) for i in items if i["defined_as"]}
    for metric in protocol.get("metrics") if isinstance(protocol.get("metrics"), list) else []:
        mid = _text(metric.get("id"), 100) if isinstance(metric, dict) else ""
        if mid and _topic._name_key(mid) not in defined:
            problems.append(f"the metric {mid} has no definition")
    return items, problems


def _squash(text: Any) -> str:
    return re.sub(r"[^0-9a-z]+", " ", str(text or "").lower()).strip()


def quote_found(quote: Any, passages: Any) -> bool:
    """Whether ``quote`` occurs in the retrieved source passages ``passages`` (a text, or a list of texts), compared with case,
    whitespace and punctuation normalised. An empty quote is not found."""
    if isinstance(passages, (list, tuple)):
        passages = " ".join(str(p) for p in passages)
    needle = _squash(quote)
    return bool(needle) and needle in _squash(passages)


def unfound_quotes(definitions: list[dict[str, str]], passages: Any) -> list[str]:
    """The metrics whose quoted definition is not in the passages."""
    return [d["metric"] for d in definitions if d.get("quote") and not quote_found(d["quote"], passages)]


def parse(protocol: Any) -> dict[str, Any]:
    """Both keys at once: ``{"simplifications", "definitions", "problems"}``."""
    simplifications, p1 = parse_simplifications(protocol)
    definitions, p2 = parse_definitions(protocol)
    return {"simplifications": simplifications, "definitions": definitions, "problems": p1 + p2}


# ---------------------------------------------------------------------------------------------------------------------
# what the topic asks for that the plan neither covers nor names
# ---------------------------------------------------------------------------------------------------------------------

def _named_in(item: dict[str, Any], ident: str, simplifications: list[dict[str, str]]) -> bool:
    """Whether a simplification names the topic item: by its id (``F1``) or by its words in ``replaces``."""
    name = item.get("name") or item.get("text") or ""
    wanted_key = _topic._name_key(name)
    wanted = _topic._tokens(name)
    for s in simplifications:
        replaces = s.get("replaces", "")
        if re.search(rf"(?<![0-9A-Za-z]){re.escape(ident)}(?![0-9A-Za-z])", replaces):
            return True
        if wanted_key and wanted_key in _topic._name_key(replaces):
            return True
        have = _topic._tokens(replaces)
        if wanted and wanted <= have:
            return True
    return False


def uncovered(req: Any, protocol: Any, *, design: Any = None) -> list[dict[str, str]]:
    """The topic items (``id`` ``F1`` / ``Q2`` / ``M1``, ``kind``, ``name``, ``sentence``) that the plan neither covers (as
    :func:`core.topic_requirements.match` judges: a setting, a metric, an equation, or the plan's own ``topic_map``) nor names
    in ``protocol.simplifications``. Items the name match cannot settle either way are listed too: they are not shown to be
    covered. ``[]`` for an empty reading."""
    if not isinstance(req, dict) or _topic.is_empty(req):
        return []
    plan = _topic.view(design if isinstance(design, dict) else {}, protocol if isinstance(protocol, dict) else {})
    gaps, unresolved = _topic.match(req, plan)
    simplifications, _problems = parse_simplifications(protocol)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for kind, key, singular in (("factors", "F", "factor"), ("quantities", "Q", "quantity"), ("formulas", "M", "formula"),
                                  ("searches", "S", "search")):
        for i, item in enumerate(req.get(kind) or []):
            ident = _topic._item_id(kind, i)
            name = str(item.get("name") or item.get("text") or "")
            # A level the setting's values leave out, or a search the plan does not run, is settled by the name match.
            level_gap = next((g for g in gaps if g["kind"] in ("level", "search") and g["name"] == name), None)
            if ident not in unresolved and level_gap is None:
                continue
            if _named_in(item, ident, simplifications):
                continue
            seen.add(ident)
            sentence = level_gap["sentence"] if level_gap else _topic.gap(singular, item)["sentence"]
            out.append({"id": ident, "kind": singular, "name": name, "sentence": sentence})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------------------------------------------------

def ledger_record(plan_sha: str, simplifications: list[dict[str, str]],
                  relations: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """``{plan_sha, items}``: each simplification with the ids of the relations that test it (``relations``: ``{id: [ids]}``)
    and its status (``not_tested`` until one does)."""
    relations = relations or {}
    items = []
    for s in simplifications:
        ids = [str(r) for r in relations.get(s.get("id", ""), [])]
        items.append({**{k: s.get(k, "") for k in FIELDS}, "relation_ids": ids,
                      "status": "tested" if ids else STATUS_UNTESTED})
    return {"plan_sha": str(plan_sha), "items": items}


def write_ledger(path: Any, record: dict[str, Any]) -> None:
    """Write the record to ``path`` (atomically: a killed run leaves the old file or the new one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_ledger(path: Any) -> dict[str, Any] | None:
    """The record at ``path``, or ``None`` when it is missing or not a ledger."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("items"), list) or not isinstance(data.get("plan_sha"), str):
        return None
    return data


def untested(record: Any) -> list[dict[str, Any]]:
    """The ledger's items no relation tests (the paper says: approximation not tested)."""
    items = record.get("items") if isinstance(record, dict) else None
    return [i for i in items or [] if isinstance(i, dict) and not i.get("relation_ids")]
