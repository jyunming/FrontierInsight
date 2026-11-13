"""What the plan's model says each of the plan's dependent variables is, when FI placed it on no declared result.

The plan lists its dependent variables as free text; the frozen protocol declares results by id. An entry that is not exactly
a declared id (a metric or a derived result) is asked once, before the plan is frozen, with the others of its plan
(:data:`MAX_ENTRIES`), keyed ``D1``, ``D2`` ... so a model's shortened wording cannot lose its row. Four answers only:
``result`` (one or more declared ids), ``setting`` (a setting the study varies), ``derived`` (a new derived result: its code
name, the declared results it is worked out from and what it is) or ``not_reported``. The answers are held to the ids and
settings FI already has; a new derived result's name must be one the plan does not use yet (a declared result under another
spelling is answered ``result``, never added twice).

The same question asks for the uncertainties the plan names for a declared result that are not themselves a declared result
(a slope's standard error, a spread across runs): the plan's model lists each under ``uncertainties``, keyed by the declared
result it belongs to (held to FI's ids), and answers ``derived`` or ``not_reported``. The question is asked only when there is a
dependent variable to ask about; it adds no call of its own.

A ``derived`` answer declares the result in the plan (a ``derived`` entry marked ``asked``, as an answer of the ``names`` part
is), so FI's frame lists it, the code writes its function as for any derived result, and a frame entry a repair may not add is
never needed. The other answers only add: the entry's wording is added to the item the code reader is asked for each id a
``result`` answer names, and the quality report says what the plan's model answered (and which model). An answer never
removes a requirement, never unlocks a finding, and never makes a result the main quantity, so it never changes which findings
stop the quest: a fault in a result the plan's model tied the first entry to is left out and said, as without the answer.
A small model ties an entry to the wrong result about one time in twenty (2 of 40 in one measurement with a small model, 0 of
40 with a larger one), which is why nothing rests on a tie. Stored in the frozen protocol as ``dependent_map``, so its hash
covers it.
"""

from __future__ import annotations

import re
from typing import Any

from . import code_review as _code_review
from . import contract as _contract

#: Entries asked in one call (the rest stay unplaced); the uncertainties listed are held to the same number.
MAX_ENTRIES = 15
#: The most times the part is asked: the first ask and one more, for the rows that were missing or unusable.
MAX_ASKS = 2
ANSWERS = ("result", "setting", "derived", "not_reported")
#: The answers for an uncertainty the plan names.
UNCERTAINTY_ANSWERS = ("derived", "not_reported")
KEY = "dependent_map"
_CODE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _new_result_problems(key: str, row: dict[str, Any], design: Any, taken: dict[str, str]) -> list[str]:
    """Why a ``derived`` answer cannot declare its new result; ``taken`` maps each new name already given (normalized) to
    the row that gave it, and gains this one."""
    p = design.get("protocol") if isinstance(design, dict) and isinstance(design.get("protocol"), dict) else {}
    ids = declared_ids(p)
    settings = _contract.varied_settings(p)
    out: list[str] = []
    new = row.get("id")
    if not isinstance(new, str) or not _CODE_NAME.fullmatch(new.strip()):
        out.append(f"`{key}`: a new derived result needs `id`, a name for it in code (letters, digits and underscores)")
    else:
        same = next((i for i in ids if _norm(i) == _norm(new)), None)
        if same is not None:
            out.append(f"`{key}`: `{new}` is already the reported number `{same}`; answer \"result\" with it")
        elif any(_norm(s) == _norm(new) for s in settings):
            out.append(f"`{key}`: `{new}` is a setting the study changes; give the new result a name of its own")
        elif _norm(new) in taken:
            out.append(f"`{key}`: `{new}` is given twice (also for `{taken[_norm(new)]}`); each new result has its own name")
        else:
            taken[_norm(new)] = key
    inputs = row.get("inputs")
    if not isinstance(inputs, list) or not inputs or not all(i in ids for i in inputs):
        out.append(f"`{key}`: a new derived result needs `inputs`, the reported numbers it is worked out from ({', '.join(ids)})")
    if not str(row.get("what") or "").strip():
        out.append(f"`{key}`: a new derived result needs `what`, what the number is, in a few words")
    return out


def check(value: Any, entries: dict[str, str], design: Any, taken: dict[str, str] | None = None) -> list[str]:
    """Why ``value`` cannot be used; empty when it can. Every entry has its row (by its D id), each with one of the four
    answers, held to the declared ids and the settings the study varies. ``taken``: the new derived names given so far
    (normalized -> row), filled in here so the uncertainties' new names are held to them too."""
    if not isinstance(value, dict):
        return ["the answer for `names` must be a mapping keyed by the D ids"]
    p = design.get("protocol") if isinstance(design, dict) and isinstance(design.get("protocol"), dict) else {}
    ids = declared_ids(p)
    varied = _contract.varied_settings(p)
    taken = {} if taken is None else taken
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
        elif answer == "derived":
            out += _new_result_problems(key, row, design, taken)
    return out + [f"`{k}` was not asked about" for k in value if k not in entries]


def check_uncertainties(value: Any, design: Any, taken: dict[str, str] | None = None) -> list[str]:
    """Why the ``uncertainties`` answer cannot be used; empty when it can (no answer, or an empty one, is usable: nothing
    is listed). Each key is a declared result, each row says what the uncertainty is and is ``derived`` (a new result, held
    as a dependent variable's is) or ``not_reported``."""
    if value is None or value == {}:
        return []
    if not isinstance(value, dict):
        return ["the answer for `uncertainties` must be a mapping keyed by the reported numbers"]
    p = design.get("protocol") if isinstance(design, dict) and isinstance(design.get("protocol"), dict) else {}
    ids = declared_ids(p)
    taken = {} if taken is None else taken
    out: list[str] = []
    for rid, row in list(value.items())[:MAX_ENTRIES]:
        if rid not in ids:
            out.append(f"uncertainties: `{rid}` is not one of the reported numbers ({', '.join(ids)})")
            continue
        if not isinstance(row, dict) or row.get("answer") not in UNCERTAINTY_ANSWERS:
            out.append(f"uncertainties `{rid}`: answer must be one of {', '.join(UNCERTAINTY_ANSWERS)}")
            continue
        if not str(row.get("what") or "").strip():
            out.append(f"uncertainties `{rid}`: give `what`, the uncertainty the plan names, in a few words")
        elif row["answer"] == "derived":
            out += _new_result_problems(f"uncertainties {rid}", row, design, taken)
    return out


def uncertainty_entry(rid: str, row: dict[str, Any]) -> str:
    """The text an uncertainty is stored and disclosed under."""
    return f"the uncertainty of `{rid}` ({' '.join(str(row.get('what') or '').split())[:160]})"


def _new_derived(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(row["id"]).strip(), "inputs": list(dict.fromkeys(str(i) for i in row["inputs"])), "role": "result",
            "label": " ".join(str(row.get("what") or "").split())[:300], "asked": True}


def apply(protocol: dict[str, Any], value: dict[str, Any], entries: dict[str, str],
          uncertainties: dict[str, Any] | None = None) -> dict[str, Any]:
    """``protocol`` with ``dependent_map`` added: ``{entry text: {"answer", "by" | "setting"}}``, and with each new derived
    result answered (for an entry or an uncertainty) declared in ``derived``. Nothing else changes."""
    rows: dict[str, Any] = {}
    derived = [d for d in protocol.get("derived") or []]
    for key, text in entries.items():
        row = value[key]
        kept: dict[str, Any] = {"answer": row["answer"]}
        if row["answer"] == "result":
            kept["by"] = list(dict.fromkeys(str(i) for i in row["inputs"]))
        elif row["answer"] == "setting":
            kept["setting"] = str(row["setting"])
        elif row["answer"] == "derived":
            derived.append(_new_derived(row))
            kept["by"] = [derived[-1]["id"]]
        rows[text] = kept
    for rid, row in list((uncertainties or {}).items())[:MAX_ENTRIES]:
        kept = {"answer": row["answer"]}
        if row["answer"] == "derived":
            derived.append(_new_derived(row))
            kept["by"] = [derived[-1]["id"]]
        rows[uncertainty_entry(rid, row)] = kept
    return {**protocol, **({"derived": derived} if derived else {}), KEY: rows}


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
    if row.get("answer") == "derived":
        return f"\"{entry}\" is the new derived result " + ", ".join(f"`{i}`" for i in row.get("by") or []) + ", which the plan now declares"
    return f"\"{entry}\" is not reported by the study"
