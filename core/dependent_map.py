"""What the plan's model says each of the plan's dependent variables is, when FI placed it on no declared result.

The plan lists its dependent variables as free text; the frozen protocol declares results by id. An entry that is not exactly
a declared id (a metric or a derived result) is asked once, with the others of its plan (:data:`MAX_ENTRIES`), keyed ``D1``,
``D2`` ... so a model's shortened wording cannot lose its row. Three answers only: ``result`` (one or more declared ids),
``setting`` (a setting the study varies) or ``not_reported``. The answers are held to the ids and settings FI already has.

An answer only adds: the entry's wording is added to the item the code reader is asked for each id it names, the ids of the
first entry are added to the main quantity's names, and the quality report says what the plan's model answered (and which
model). It never removes a requirement, never unlocks a finding, and never changes which findings stop the quest. A small
model ties an entry to the wrong result about one time in twenty (2 of 40 in one measurement with a small model, 0 of 40 with
a larger one), which is why nothing rests on it. Stored in the frozen protocol as ``dependent_map``, so its hash covers it.
"""

from __future__ import annotations

from typing import Any

from . import code_review as _code_review
from . import contract as _contract

#: Entries asked in one call (the rest stay unplaced).
MAX_ENTRIES = 15
#: The most times the part is asked: the first ask and one more, for the rows that were missing or unusable.
MAX_ASKS = 2
ANSWERS = ("result", "setting", "not_reported")
KEY = "dependent_map"


def declared_ids(protocol: Any) -> list[str]:
    """The ids of the declared results a dependent variable may be placed on: the metrics and the derived results."""
    p = protocol if isinstance(protocol, dict) else {}
    return [str(m["id"]).strip() for k in ("metrics", "derived") for m in (p.get(k) if isinstance(p.get(k), list) else [])
            if isinstance(m, dict) and str(m.get("id") or "").strip()]


def asked(design: Any) -> dict[str, str]:
    """``{"D1": entry text, ...}``: the plan's dependent variables that are not exactly a declared id, at most
    :data:`MAX_ENTRIES`. Empty when the plan has no metrics."""
    protocol = design.get("protocol") if isinstance(design, dict) and isinstance(design.get("protocol"), dict) else {}
    names = _code_review.unplaced_dependents(design, protocol, contract_only=False)
    return {f"D{i}": n for i, n in enumerate(names[:MAX_ENTRIES], 1)}


def results_lines(protocol: dict[str, Any], plain: str) -> str:
    """The reported numbers, one per line (``plain``, as the other parts show them), then the derived results."""
    lines = [plain] if plain else []
    for d in protocol.get("derived") or []:
        if isinstance(d, dict) and str(d.get("id") or "").strip():
            what = " ".join(str(d.get("label") or d.get("formula") or "").split())[:160]
            lines.append(f"- {d['id']}: {what} (worked out afterwards)")
    return "\n".join(lines)


def names_lines(entries: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {' '.join(t.split())[:200]} (used in the design's list of dependent variables)"
                     for k, t in entries.items())


def check(value: Any, entries: dict[str, str], design: Any) -> list[str]:
    """Why ``value`` cannot be used; empty when it can. Every entry has its row (by its D id), each with one of the three
    answers, held to the declared ids and the settings the study varies."""
    if not isinstance(value, dict):
        return ["the answer for `names` must be a mapping keyed by the D ids"]
    p = design.get("protocol") if isinstance(design, dict) and isinstance(design.get("protocol"), dict) else {}
    ids = declared_ids(p)
    varied = _contract.varied_settings(p)
    out: list[str] = []
    for key in entries:
        row = value.get(key)
        if not isinstance(row, dict):
            out.append(f"`{key}` is not answered (use the D id exactly as written)")
            continue
        answer = row.get("answer")
        if answer not in ANSWERS:
            out.append(f"`{key}`: answer must be one of {', '.join(ANSWERS)}")
        elif answer == "result":
            inputs = row.get("inputs")
            if not isinstance(inputs, list) or not inputs or not all(i in ids for i in inputs):
                out.append(f"`{key}`: a result needs `inputs`, the reported numbers it is ({', '.join(ids)})")
        elif answer == "setting" and row.get("setting") not in varied:
            out.append(f"`{key}`: `setting` `{row.get('setting')}` is not one the study changes ({', '.join(varied) or 'none'})")
    return out + [f"`{k}` was not asked about" for k in value if k not in entries]


def apply(protocol: dict[str, Any], value: dict[str, Any], entries: dict[str, str]) -> dict[str, Any]:
    """``protocol`` with ``dependent_map`` added: ``{entry text: {"answer", "by" | "setting"}}`` (nothing else changes)."""
    rows: dict[str, Any] = {}
    for key, text in entries.items():
        row = value[key]
        kept: dict[str, Any] = {"answer": row["answer"]}
        if row["answer"] == "result":
            kept["by"] = list(dict.fromkeys(str(i) for i in row["inputs"]))
        elif row["answer"] == "setting":
            kept["setting"] = str(row["setting"])
        rows[text] = kept
    return {**protocol, KEY: rows}


def rows_of(protocol: Any) -> dict[str, dict[str, Any]]:
    """The stored ``dependent_map`` rows of a frozen protocol (``{}`` when none or unreadable)."""
    raw = protocol.get(KEY) if isinstance(protocol, dict) else None
    return ({str(k): v for k, v in raw.items() if isinstance(v, dict) and v.get("answer") in ANSWERS}
            if isinstance(raw, dict) else {})


def sentence(entry: str, row: dict[str, Any]) -> str:
    """What the plan's model said about one entry, in its own terms (never what the study does)."""
    if row.get("answer") == "result":
        return f"\"{entry}\" is the reported result " + ", ".join(f"`{i}`" for i in row.get("by") or [])
    if row.get("answer") == "setting":
        return f"\"{entry}\" is the setting `{row.get('setting')}`"
    return f"\"{entry}\" is not reported by the study"
