"""What the literature check looks up: FI's own typed record of the study, never the analysis model's prose.

The check used to search and classify each string of ``analysis.key_findings``, so every restated FI note ("the engine's
checks passed", "3 of 6 figures drawn") was sent to scholarly sources and stored as a finding. Here the items come from
what FI already holds as typed records: the topic's main quantities (``.fi/topic_requirements.json``), the plan's own map
from each to a result (:func:`core.topic_requirements.mapping_of`, with the reader's verdicts in
``.fi/topic_covered.json``) and each result's value (:func:`core.paper_values.records_of`). Nothing is read from the
analysis's text, so no wording of an FI note can become a query, a sentence to classify or a stored claim.

An :class:`Item` is one result. ``to_check`` means FI holds a value for it and wrote the ``sentence`` itself; ``not_held``
means it does not (a result left out, a map that does not stand, no value, a degenerate run), with ``why_not`` in FI's words.
A main quantity whose map the reader never confirmed is searched only as the plan's own result (its label, ``quantity``
empty): the person's quantity is never named for it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contract as _contract
from . import paper_values as _pv
from . import result_checks as _checks
from . import results_by_setting as _rbs
from . import topic_requirements as _req

#: The most results one pass looks up (the cap the check always had).
CAP = 10

TO_CHECK = "to_check"
NOT_HELD = "not_held"

_QUERY_LIMIT = 200
#: What a ``protocol.topic_map`` entry may name that is not a result with a value (a setting, an equation, a search).
_NOT_A_RESULT = {"setting", "grid", "design_variable", "variable", "equation", "optimisation", "optimization", "search"}
_MARKUP = re.compile(r"[$\\{}^]")


@dataclass(frozen=True)
class Item:
    """One result the literature check is about. ``result`` is the plan's result id (a quantity id such as ``Q3`` when
    no result is mapped); ``label`` the name a reader is shown; ``query`` what is searched; ``sentence`` what FI wrote of the
    value; ``quantity`` the topic quantities the confirmed map says it answers (``None`` for a result checked as the plan's
    own: nothing then reads it as a check of the person's quantity)."""

    result: str
    label: str
    query: str
    sentence: str
    status: str
    why_not: str = ""
    quantity: str | None = None


def plain(text: Any) -> str:
    """A label without the markup the plan's labels carry (``$E_a$``), whitespace folded."""
    return " ".join(_MARKUP.sub("", str(text or "")).split())


def _query(name: str, context: str) -> str:
    return " ".join(f"{plain(name)} {plain(context)}".split())[:_QUERY_LIMIT].strip()


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _cell_means(rec: Any, ident: str) -> list[float]:
    """The mean of each cell's trials for a result, over the cells that hold numbers for it."""
    out: list[float] = []
    for cell in getattr(rec, "cells", None) or []:
        entry = (cell.get("metrics") or {}).get(ident)
        values = entry.get("values") if isinstance(entry, dict) else None
        numbers = [float(v) for v in values or [] if _finite(v)]
        if numbers:
            out.append(sum(numbers) / len(numbers))
    return out


def _value_sentence(rec: Any, answers: dict[str, float], ident: str, label: str) -> tuple[str, str]:
    """``(sentence, "")`` from FI's record for result ``ident``, or ``("", why)`` when FI holds no value of it. The study-level
    value is FI's own key (:func:`core.paper_values.resolve`), which refuses one when the result changes with a setting; then
    the sentence gives the range of the settings' values and says that FI offers no single value. No direction is written:
    FI computes none here, and says so."""
    if ident in answers:
        return f"{label}: {_pv.sig(answers[ident], 4)} (one value for the whole study).", ""
    if rec is None or ident not in rec.metrics():
        return "", "the record holds no value of it"
    axes = rec.varied() or list(rec.axes())
    key = ident + "@" + ",".join(f"{a}=all" for a in axes) if axes else ident
    try:
        value = _pv.resolve(key, rec)
    except Exception:  # noqa: BLE001 -- a value FI cannot work out is a value it does not hold
        value = None
    if value is not None and value.filled and value.text and value.text != _pv.NOT_RECORDED:
        return f"{label}: {value.text}. FI computed no comparison across settings.", ""
    means = _cell_means(rec, ident)
    if not means:
        return "", "the record holds no number for it"
    low, high = min(means), max(means)
    if low == high:
        return f"{label}: {_pv.sig(low, 4)} at each of {len(means)} settings. FI computed no comparison across settings.", ""
    return (f"{label} changes with the study's settings: from {_pv.sig(low, 4)} to {_pv.sig(high, 4)} over {len(means)} "
            "settings (FI offers no single value for the study). FI computed no comparison across settings."), ""


_Held = tuple[Any, dict[str, float], dict[str, str], set[str]]


def _readers_record(quest_root: Path | str, protocol: dict[str, Any] | None) -> _Held:
    """``(the paper's records or None, the plan-named answers FI kept, the results FI left out with the check's note, the
    results the runs disagree on)``. The answers are the record of a result no trial returned, matched on the exact id."""
    root = Path(quest_root)
    left_out = _checks.left_out_by_record(root)
    try:
        rec = _pv.records_of(root, protocol)
    except Exception:  # noqa: BLE001 -- no readable record: no value is held, said per item
        rec = None
    if rec is not None and rec.cells:
        return rec, dict(rec.answers), left_out, set(rec.by_run)
    # a study with no trial record: the answers FI kept at execute are the record (the plan's own ids only)
    planned = {str(m["id"]) for key in ("metrics", "derived")
               for m in (protocol.get(key) if isinstance(protocol, dict) else None) or []
               if isinstance(m, dict) and m.get("id")}
    single, by_run = _pv._read_answers(root)
    return (rec, {k: v for k, v in single.items() if k in planned and k not in left_out}, left_out,
            {k for k in by_run if k in planned and k not in left_out})


def _held(held: _Held, ident: str, label: str, degenerate: bool) -> tuple[str, str]:
    """``(sentence, "")`` or ``("", why not)`` for one result id."""
    rec, answers, left_out, by_run = held
    if degenerate:
        return "", "every value of the run is about 0: a broken run, not a result"
    if ident in left_out:
        return "", left_out[ident] or "FI left it out of the results"
    if ident in by_run:
        return "", "the runs gave different values; FI offers no single value for the study"
    return _value_sentence(rec, answers, ident, label)


def _from_quantities(req: dict[str, Any], plan: dict[str, Any], protocol: dict[str, Any] | None, covered: Any,
                     held: _Held, context: str, degenerate: bool) -> list[Item]:
    history = _req.verdict_history(covered)
    order: list[str] = []
    groups: dict[str, dict[str, Any]] = {}
    for index, quantity in enumerate(req.get("quantities") or []):
        if not isinstance(quantity, dict) or quantity.get("main") is not True:
            continue
        ident = _req._item_id("quantities", index)
        name = str(quantity.get("name") or ident)
        mapped = _req.mapping_of(req, plan, "quantities", quantity)
        if mapped is None or mapped[0] == "bad":
            if mapped is None:
                why = "the plan maps no result to it"
            elif {str(m.get("by")).partition(":")[0].strip().lower() for m in _req.map_entries(req, plan, "quantities", quantity)} \
                    & _NOT_A_RESULT:
                why = "the plan maps it to a setting, an equation or a search, not to a result"
            else:
                why = str(mapped[1])
            key = ident
            groups[key] = {"names": [name], "confirmed": False, "result": "", "why": why}
            order.append(key)
            continue
        element = mapped[1]
        confirmed = mapped[0] == "ok" or _req.settled_covered(history, ident, quantity=True)
        group = groups.get(element)
        if group is None:
            group = groups[element] = {"names": [], "confirmed": False, "result": element, "why": ""}
            order.append(element)
        if confirmed:
            group["names"].append(name)
            group["confirmed"] = True
    items: list[Item] = []
    for key in order[:CAP]:
        group = groups[key]
        result = group["result"]
        if not result:
            items.append(Item(key, group["names"][0], "", "", NOT_HELD, group["why"]))
            continue
        label = plain(_rbs.plain_label(protocol, result, result))
        if group["confirmed"]:
            names = group["names"]
            query_name = names[0]
            quantity: str | None = "; ".join(names)
        else:
            query_name, quantity = label, None  # the plan's own result: the person's quantity is never named for it
        sentence, why = _held(held, result, label, degenerate)
        if why:
            items.append(Item(result, label, "", "", NOT_HELD, why, quantity))
        else:
            items.append(Item(result, label, _query(query_name, context), sentence, TO_CHECK, "", quantity))
    return items


def _declared_results(protocol: dict[str, Any] | None) -> list[str]:
    """The plan's results of role ``result``: the declared metrics and the quantities it works out from them."""
    roles = _contract.raw_roles(protocol)
    out = [i for i, role in roles.items() if role == "result"]
    for d in (protocol.get("derived") if isinstance(protocol, dict) else None) or []:
        if isinstance(d, dict) and d.get("id") and d.get("role", "result") == "result" and str(d["id"]) not in out:
            out.append(str(d["id"]))
    return out


def _from_plan(protocol: dict[str, Any] | None, held: _Held, context: str, degenerate: bool) -> list[Item]:
    items: list[Item] = []
    for result in _declared_results(protocol)[:CAP]:
        label = plain(_rbs.plain_label(protocol, result, result))
        sentence, why = _held(held, result, label, degenerate)
        items.append(Item(result, label, "", "", NOT_HELD, why) if why
                     else Item(result, label, _query(label, context), sentence, TO_CHECK))
    return items


def _measured(entry: Any) -> tuple[str, str]:
    """``(the value as text, why not)`` of one entry of ``result_json.measurements``: a number, a string label, or the
    ``value`` field of ``{value, unit, n_observations, source_file_ids}`` (never ``n_observations`` or a file id)."""
    unit = ""
    if isinstance(entry, dict):
        unit = str(entry.get("unit") or "").strip()
        entry = entry.get("value")
    if _finite(entry):
        return _pv.sig(float(entry), 4) + (f" {unit}" if unit else ""), ""
    if isinstance(entry, str) and entry.strip():
        return entry.strip(), ""
    return "", "the data holds no value of it"


def _from_data(result_json: Any, context: str) -> list[Item]:
    """No-simulation path: the design's variables the data gave a value for (``measurements``), and each one it did not
    (``missing_measurements``). Top-level counts such as ``n_files`` are never items."""
    data = result_json if isinstance(result_json, dict) else {}
    measurements = data.get("measurements") if isinstance(data.get("measurements"), dict) else {}
    items: list[Item] = []
    for name, entry in measurements.items():
        label = plain(str(name).replace("_", " "))
        text, why = _measured(entry)
        items.append(Item(str(name), label, "", "", NOT_HELD, why) if why
                     else Item(str(name), label, _query(label, context), f"{label}: {text}.", TO_CHECK))
    missing = data.get("missing_measurements") if isinstance(data.get("missing_measurements"), list) else []
    for name in missing:
        if isinstance(name, str) and name.strip() and name not in measurements:
            items.append(Item(name, plain(name.replace("_", " ")), "", "", NOT_HELD, "the data holds no value of it"))
    return items[:CAP]


def items(*, quest_root: Path | str, protocol: dict[str, Any] | None, requirements: dict[str, Any] | None,
          covered: Any, design: Any, context: str, no_simulation: bool = False, result_json: Any = None,
          degenerate: bool = False) -> list[Item]:
    """What the literature check looks up, in the topic's order, one per result, at most :data:`CAP`.

    ``context``: the first query of the literature step this pass (else the topic's sentence): the only text besides a
    result's name that goes into a query. With no protocol and not the no-simulation path there is nothing to look up."""
    if no_simulation:
        return _from_data(result_json, context)
    if not isinstance(protocol, dict) or not protocol:
        return []
    held = _readers_record(quest_root, protocol)
    req = requirements if isinstance(requirements, dict) else None
    if req is not None and not _req.is_empty(req) and req.get("quantities"):
        plan = _req.view(design, protocol)
        return _from_quantities(req, plan, protocol, covered, held, context, degenerate)
    return _from_plan(protocol, held, context, degenerate)
