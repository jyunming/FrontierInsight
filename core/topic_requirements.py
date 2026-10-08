"""What the topic asks for, read once from the topic text by a model other than the plan's, and held against the plan.

A plan can drift from the topic it was written for: a factor the topic says to compare is not among the settings the
experiment sweeps, a quantity it says to report is not measured, a formula it states in full is replaced by another the
code then implements. Each of these is invisible to every later check, which judges the plan and the code against the
plan. So at plan time (:meth:`core.engine.Engine._cover_the_topic_in_the_plan`):

1. ONE call (``agents/topic_requirements.md``, node ``oracle_review.topic``, so the model named for ``oracle_review``
   answers it) lists from the topic text alone: the factors to vary or compare (with the levels or range the topic states),
   the quantities to report, the formulas it states, the comparisons and the constraints, and which of them the question
   cannot be answered without. It is kept in ``.fi/topic_requirements.json``.
2. Each is looked for in the plan: a factor as a setting to sweep (``protocol.grid``) or a design variable of a search, with
   the stated levels or range covered; a quantity as a metric; a formula as an equation of the model. What a name match can
   settle is settled here; the rest (a different name, a formula written another way) goes to ONE more call
   (``agents/topic_cover.md``) that judges only those items.
3. What is missing goes back to the plan in one bounded request (the same one as the other missing parts of a plan).
   What is still missing after that stops the quest before anything runs when it is what the question is about, and is said
   in the paper's limitations when it is not.

This module holds the parts that need no model: reading the reply, matching, the plain sentences, merging an answer,
and the text the code-writing requests are shown (:func:`code_block`).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

#: The file in ``.fi/`` that holds the requirements, and the one that holds how far they are met.
FILE = "topic_requirements.json"
COVERED_FILE = "topic_covered.json"
#: The call's node: ``oracle_review`` answers it (a dot name falls back to its first part).
NODE = "oracle_review.topic"
#: plan.md's section.
HEADING = "What the topic asks for"
#: The most chat calls the judgement of unresolved items may make in a quest.
MAX_JUDGE_CALLS = 4

_SECTIONS = ("factors", "quantities", "formulas")
_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "by", "per", "and", "or", "in", "on", "for", "to", "at", "as", "with", "vs", "versus",
    "type", "types", "value", "values", "level", "levels", "parameter", "parameters", "setting", "settings", "factor",
    "factors", "number", "amount", "size", "nm", "mm", "um", "s", "ms", "hz", "unit", "units", "used", "use", "each",
    "every", "different", "which", "what", "how", "its", "their", "is", "are", "be", "from", "than", "across",
})
#: A sentence of the topic that says what the study should find or what a number should come out as is a result, not a
#: specification: it is not shown to a request that writes code.
_RESULT_MARKERS = re.compile(
    r"\b(expected (value|result|outcome|answer)s?|should (give|yield|equal|come out|reach|be (about|close|equal|near))|"
    r"known (answer|value|result)s?|reference (value|answer|result)s?|correct (answer|value|result)s?|ground[- ]truth|"
    r"is expected to (give|yield|be|reach|equal)|results? (should|must) (be|equal|match|agree))\b", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------------------------------
# reading the extraction
# ---------------------------------------------------------------------------------------------------------------------

def _text(value: Any, limit: int = 300) -> str:
    return " ".join(str(value if value is not None else "").replace("`", "'").split())[:limit]


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    try:
        out = float(str(value).strip().replace(",", ""))
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _levels(value: Any) -> list[Any]:
    """The levels a topic states for a factor: numbers or names (not a mix), at most 40."""
    items = value if isinstance(value, list) else ([] if value in (None, "") else [value])
    numbers = [n for n in (_number(v) for v in items) if n is not None]
    if numbers and len(numbers) == len(items):
        return numbers[:40]
    names = [_text(v, 80) for v in items if _text(v, 80)]
    return names[:40]


def parse(obj: Any) -> dict[str, Any] | None:
    """The requirements from the reader's reply, strictly: ``None`` when the reply names no factor, quantity or formula
    and no comparison (nothing was read). Only these keys are kept; every text is one bounded line."""
    if not isinstance(obj, dict):
        return None
    factors: list[dict[str, Any]] = []
    for item in obj.get("factors") if isinstance(obj.get("factors"), list) else []:
        item = item if isinstance(item, dict) else {"name": item}
        name = _text(item.get("name"), 100)
        if not name:
            continue
        rng = [n for n in (_number(v) for v in (item.get("range") if isinstance(item.get("range"), list) else [])) if n is not None]
        factors.append({"name": name, "levels": _levels(item.get("levels")),
                        "range": sorted(rng[:2]) if len(rng) == 2 else [],
                        "stated": _text(item.get("stated")), "main": item.get("main") is True})
    quantities: list[dict[str, Any]] = []
    for item in obj.get("quantities") if isinstance(obj.get("quantities"), list) else []:
        item = item if isinstance(item, dict) else {"name": item}
        name = _text(item.get("name"), 100)
        if name:
            quantities.append({"name": name, "stated": _text(item.get("stated")), "main": item.get("main") is True})
    formulas: list[dict[str, Any]] = []
    for item in obj.get("formulas") if isinstance(obj.get("formulas"), list) else []:
        item = item if isinstance(item, dict) else {"text": item}
        text = _text(item.get("text"), 400)
        if text:
            formulas.append({"name": _text(item.get("name"), 100) or text[:40], "text": text, "main": item.get("main") is True})
    comparisons = []
    for item in obj.get("comparisons") if isinstance(obj.get("comparisons"), list) else []:
        item = item if isinstance(item, dict) else {"what": item}
        what = _text(item.get("what"))
        if what:
            between = item.get("between") if isinstance(item.get("between"), list) else []
            comparisons.append({"what": what, "between": [_text(b, 80) for b in between if _text(b, 80)][:8]})
    constraints = [_text(c) for c in (obj.get("constraints") if isinstance(obj.get("constraints"), list) else []) if _text(c)]
    results = [_text(r, 200) for r in (obj.get("stated_results") if isinstance(obj.get("stated_results"), list) else [])
               if _text(r, 200)]
    if not (factors or quantities or formulas or comparisons):
        return None
    return {"factors": factors[:20], "quantities": quantities[:30], "formulas": formulas[:12],
            "comparisons": comparisons[:12], "constraints": constraints[:12], "stated_results": results[:20]}


def is_empty(req: Any) -> bool:
    return not (isinstance(req, dict) and any(req.get(k) for k in _SECTIONS))


# ---------------------------------------------------------------------------------------------------------------------
# the plan as it can be searched
# ---------------------------------------------------------------------------------------------------------------------

def _tokens(text: Any) -> set[str]:
    """Words of a name, lower case, without filler and with a plural's ``s`` taken off."""
    out: set[str] = set()
    for word in re.split(r"[^0-9a-zA-Z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))):
        word = word.lower()
        if not word or word in _STOP_WORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def view(design: Any, protocol: Any = None) -> dict[str, Any]:
    """What the plan can be searched for: ``axes`` ({name: values}, from ``protocol.grid`` and a search's design variables),
    ``metrics`` ([(id, text)]), ``equations`` ([{id, formula, source}]) and ``assumptions`` (text). ``protocol`` is the one
    that runs when it differs from the design's own (the frozen one)."""
    design = design if isinstance(design, dict) else {}
    if not isinstance(protocol, dict):
        protocol = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    axes: dict[str, list[Any]] = {}
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    for axis, values in grid.items():
        axes[str(axis)] = list(values) if isinstance(values, list) else [values]
    block = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else {}
    for variable in block.get("design_variables") if isinstance(block.get("design_variables"), list) else []:
        if not isinstance(variable, dict) or not str(variable.get("name") or "").strip():
            continue
        values = variable.get("values") if isinstance(variable.get("values"), list) else [
            v for v in (variable.get("low"), variable.get("high")) if v is not None]
        axes[str(variable["name"]).strip()] = list(values)
    metrics: list[tuple[str, str]] = []
    for metric in protocol.get("metrics") if isinstance(protocol.get("metrics"), list) else []:
        if isinstance(metric, dict) and metric.get("id"):
            metrics.append((str(metric["id"]), f"{metric.get('id')} {metric.get('estimand') or ''} {metric.get('unit') or ''}"))
    objective = block.get("objective") if isinstance(block.get("objective"), dict) else {}
    if objective.get("quantity"):
        metrics.append((str(objective["quantity"]), str(objective["quantity"])))
    variables = design.get("variables") if isinstance(design.get("variables"), dict) else {}
    for name in variables.get("dependent") if isinstance(variables.get("dependent"), list) else []:
        metrics.append((str(name), str(name)))
    model = protocol.get("model") if isinstance(protocol.get("model"), dict) else {}
    equations = [{"id": str(e.get("id") or ""), "formula": str(e.get("formula") or ""), "source": str(e.get("source") or "")}
                 for e in (model.get("equations") if isinstance(model.get("equations"), list) else []) if isinstance(e, dict)]
    assumptions = " ".join(str(a) for a in (model.get("assumptions") if isinstance(model.get("assumptions"), list) else []))
    topic_map = [m for m in (protocol.get("topic_map") if isinstance(protocol.get("topic_map"), list) else [])
                 if isinstance(m, dict) and str(m.get("item") or "").strip() and str(m.get("by") or "").strip()]
    return {"axes": axes, "metrics": metrics, "equations": equations, "assumptions": assumptions, "topic_map": topic_map}


def _norm_formula(text: Any) -> str:
    return re.sub(r"[\s*`$\\{}]+", "", str(text or "").lower())


def _covers(values: list[Any], factor: dict[str, Any]) -> str | None:
    """Why the values of a setting do not cover the levels or range the topic states for it, ``""`` when they do, and
    ``None`` when a name match cannot tell (named levels held as numbers, numbers held as names): a reader judges those."""
    wanted = factor.get("levels") or []
    numbers = [n for n in (_number(v) for v in values) if n is not None]
    if wanted and all(isinstance(w, (int, float)) for w in wanted):
        if not numbers:
            return None
        missing = [w for w in wanted if not any(math.isclose(w, n, rel_tol=1e-6, abs_tol=1e-12) for n in numbers)]
        return f"the values {', '.join(f'{m:g}' for m in missing[:6])} are not among its settings" if missing else ""
    if wanted:  # named levels
        have = {" ".join(str(v).lower().split()) for v in values if isinstance(v, str)}
        if not have:
            return None  # numbers standing for named levels: a reader judges them
        have_tokens = [_tokens(h) for h in have]
        missing = [w for w in wanted if not any(_tokens(w) <= t or t <= _tokens(w) for t in have_tokens if t)]
        return f"the levels {', '.join(str(m) for m in missing[:6])} are not among its settings" if missing else ""
    rng = factor.get("range") or []
    if len(rng) == 2 and not numbers:
        return None
    if len(rng) == 2:
        lo, hi = min(numbers), max(numbers)
        slack = 0.02 * max(abs(rng[0]), abs(rng[1]), 1e-12)
        if lo > rng[0] + slack or hi < rng[1] - slack:
            return f"its settings run from {lo:g} to {hi:g}, not {rng[0]:g} to {rng[1]:g}"
    return ""


def _item_id(kind: str, index: int) -> str:
    return {"factors": "F", "quantities": "Q", "formulas": "M"}[kind] + str(index + 1)


def _name_key(text: Any) -> str:
    return " ".join(re.sub(r"[^0-9a-zA-Z]+", " ", str(text or "")).lower().split())


def mapping_of(req: dict[str, Any], plan: dict[str, Any], kind: str, item: dict[str, Any]) -> tuple[str, str] | None:
    """What the plan itself says covers a topic item (``protocol.topic_map``: ``item`` = the item's name or formula, ``by`` =
    ``setting:<name>`` / ``metric:<id>`` / ``equation:<id>``), checked here and not taken on trust: ``("ok", element)`` when
    the named element exists in the plan (for a factor with stated levels or a range, its values cover them), ``("bad",
    why)`` when it does not, ``None`` when the plan maps nothing for the item."""
    wanted = {_name_key(item.get("name")), _name_key(item.get("text"))} - {""}
    entries = [m for m in plan.get("topic_map") or [] if _name_key(m.get("item")) in wanted]
    if not entries:
        return None
    why = "nothing it names exists"
    for entry in entries:
        head, _sep, name = str(entry["by"]).partition(":")
        head, name = head.strip().lower(), name.strip()
        if kind == "factors" and head in ("setting", "grid", "design_variable", "variable"):
            axis = next((a for a in plan["axes"] if _name_key(a) == _name_key(name)), None)
            if axis is None:
                why = f"the setting {name} is not among the plan's settings"
                continue
            reach = _covers(plan["axes"][axis], item)
            coded = reach is None and len(plan["axes"][axis]) >= len(item.get("levels") or [0])
            if reach and not coded:
                why = f"{reach}"
                continue
            return "ok", axis
        if kind == "quantities" and head == "metric":
            found = next((i for i, _t in plan["metrics"] if _name_key(i) == _name_key(name)), None)
            if found is not None:
                return "ok", found
            why = f"the metric {name} is not among the plan's metrics"
        if kind == "formulas" and head == "equation":
            found = next((e["id"] for e in plan["equations"] if _name_key(e["id"]) == _name_key(name) and e["formula"].strip()), None)
            if found is not None:
                return "ok", found
            why = f"the equation {name} is not among the plan's equations"
    return "bad", why


def match(req: dict[str, Any], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """``(gaps, unresolved)``: the gaps a name match settles (a factor named by a setting whose values do not cover what the
    topic states for it), and the ids (``F1``, ``Q2``, ``M1``) of the items it cannot settle either way. An item a name
    match finds is covered and appears in neither."""
    gaps: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for i, factor in enumerate(req.get("factors") or []):
        wanted = _tokens(factor["name"])
        axis = next((a for a in plan["axes"] if wanted and (_tokens(a) <= wanted or wanted <= _tokens(a)) and _tokens(a)), None)
        mapped = mapping_of(req, plan, "factors", factor) if axis is None else None
        if mapped is not None and mapped[0] == "ok":
            continue
        if axis is None:
            unresolved.append(_item_id("factors", i))
            continue
        why = _covers(plan["axes"][axis], factor)
        if why is None:
            unresolved.append(_item_id("factors", i))
        elif why:
            gaps.append(gap("level", factor, why=why, axis=axis))
    for i, quantity in enumerate(req.get("quantities") or []):
        wanted = _tokens(quantity["name"])
        if not (wanted and any(wanted <= _tokens(text) for _id, text in plan["metrics"])):
            mapped = mapping_of(req, plan, "quantities", quantity)
            if mapped is not None and mapped[0] == "ok":
                continue
            unresolved.append(_item_id("quantities", i))
    for i, formula in enumerate(req.get("formulas") or []):
        mine = _norm_formula(formula["text"])
        if not (mine and any(mine in _norm_formula(e["formula"]) or (_norm_formula(e["formula"]) and
                                                                    _norm_formula(e["formula"]) in mine)
                             for e in plan["equations"])):
            mapped = mapping_of(req, plan, "formulas", formula)
            if mapped is not None and mapped[0] == "ok":
                continue
            unresolved.append(_item_id("formulas", i))
    return gaps, unresolved


def item_of(req: dict[str, Any], ident: str) -> tuple[str, dict[str, Any]] | None:
    kind = {"F": "factors", "Q": "quantities", "M": "formulas"}.get(ident[:1])
    try:
        return (kind, req[kind][int(ident[1:]) - 1]) if kind else None
    except (KeyError, ValueError, IndexError):
        return None


def gap(kind: str, item: dict[str, Any], *, why: str = "", axis: str = "") -> dict[str, Any]:
    """One thing the topic asks for that the plan does not have, with the sentence a person reads."""
    name = item.get("name") or item.get("text") or ""
    if kind == "factor":
        levels = item.get("levels") or ([f"{item['range'][0]:g} to {item['range'][1]:g}"] if item.get("range") else [])
        sentence = (f"The topic says to vary or compare {name}"
                    + (f" ({', '.join(str(x) for x in levels[:8])})" if levels else "")
                    + ", but the plan's settings do not include it.")
    elif kind == "level":
        sentence = f"The topic states which values of {name} to use, but {why}."
    elif kind == "quantity":
        sentence = f"The topic asks to report {name}, but the plan does not measure it."
    else:
        sentence = (f"The topic states the formula `{item.get('text')}`, but the plan's model does not use it and does not "
                    "say why it uses another.")
    return {"kind": kind, "name": str(name), "main": bool(item.get("main")), "axis": axis, "sentence": sentence}


# ---------------------------------------------------------------------------------------------------------------------
# the judgement of what a name match could not settle
# ---------------------------------------------------------------------------------------------------------------------

def judge_parts(req: dict[str, Any], plan: dict[str, Any], unresolved: list[str]) -> tuple[str, str]:
    """``(items, plan)`` as the judging prompt shows them (JSON text)."""
    items = []
    for ident in unresolved:
        found = item_of(req, ident)
        if found is None:
            continue
        kind, item = found
        items.append({"id": ident, "kind": kind[:-1] if kind != "quantities" else "quantity",
                      **{k: v for k, v in item.items() if k in ("name", "levels", "range", "text", "stated") and v}})
    shown = {
        "settings_swept": {a: v[:40] for a, v in plan["axes"].items()},
        "metrics": [{"id": i, "text": " ".join(t.split())[:200]} for i, t in plan["metrics"]][:60],
        "equations": [{"id": e["id"], "formula": e["formula"][:300], "source": e["source"][:200]} for e in plan["equations"]][:40],
        "assumptions": plan["assumptions"][:800],
    }
    return json.dumps(items, indent=2, ensure_ascii=False), json.dumps(shown, indent=2, ensure_ascii=False, default=str)


def parse_judgement(obj: Any, unresolved: list[str]) -> dict[str, str]:
    """``{id: "covered" | "replaced" | "missing"}`` for the ids asked about; an id the reply leaves out or answers with
    anything else is absent (not judged: never counted as missing)."""
    out: dict[str, str] = {}
    items = obj.get("items") if isinstance(obj, dict) else None
    for entry in items if isinstance(items, list) else []:
        if not isinstance(entry, dict):
            continue
        ident = str(entry.get("id") or "").strip().upper()
        verdict = str(entry.get("verdict") or "").strip().lower()
        if ident in unresolved and verdict in ("covered", "replaced", "missing"):
            out[ident] = verdict
    return out


def gaps_from(req: dict[str, Any], plan: dict[str, Any], verdicts: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    """``(gaps, unjudged)``: the gaps a name match settled plus the unresolved items the judgement called missing, and the
    names of the items nobody could judge (no answer, a failed call, the calls spent). An item nobody judged is not
    called a gap, and is never passed in silence: plan.md and the paper's limitations say it was not checked."""
    gaps, unresolved = match(req, plan)
    unjudged: list[str] = []
    for ident in unresolved:
        found = item_of(req, ident)
        if found is None:
            continue
        kind, item = found
        if ident not in verdicts:
            unjudged.append(str(item.get("name") or item.get("text") or ident))
        elif verdicts[ident] == "missing":
            gaps.append(gap({"factors": "factor", "quantities": "quantity", "formulas": "formula"}[kind], item))
    return gaps, unjudged


def main_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The gaps the question cannot be answered without: a factor or a quantity the topic's reader marked as main (a formula
    is a specification the code is shown, so a missing one is said, not a stop)."""
    return [g for g in gaps if g["main"] and g["kind"] in ("factor", "level", "quantity")]


# ---------------------------------------------------------------------------------------------------------------------
# the request to the plan, and what is kept of its answer
# ---------------------------------------------------------------------------------------------------------------------

def mapped_items(req: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
    """``{item name: the plan's element}`` for every topic item the plan's own mapping covers and a name match did not find
    (kept in the record: where the plan and a reader disagree, this is what overruled it)."""
    out: dict[str, str] = {}
    for kind in _SECTIONS:
        for item in req.get(kind) or []:
            by_name = {"factors": "factors", "quantities": "quantities", "formulas": "formulas"}[kind]
            found = mapping_of(req, plan, by_name, item)
            if found is not None and found[0] == "ok":
                out[str(item.get("name") or item.get("text"))] = found[1]
    return out


def request(topic: str, gaps: list[dict[str, Any]]) -> str:
    """The request to the plan's model for what the topic asks and the plan lacks. Nothing else may change."""
    lines = [f"- {g['sentence']}" for g in gaps]
    return (
        "The topic asks for things this plan does not yet include:\n" + "\n".join(lines)
        + f"\n\nThe topic: “{' '.join(str(topic or '').split())[:1500]}”\n\n"
        "Write them into the design block. A setting to vary or compare is a key of `protocol.grid` with every value the topic "
        "states (or, for a search for the best design, a design variable of `protocol.optimisation` with its range). A quantity "
        "to report is an entry of `protocol.metrics` (an `id`, a `kind`, an `estimand` and a `unit`). A formula the topic "
        "states is an equation of `protocol.model.equations`, written as the topic writes it; if the plan needs another "
        "formula in its place, keep that one and say in its `source` that it replaces the topic's, and why. Also say, in "
        "`protocol.topic_map`, which element of the plan covers each thing the topic asks for (the ones above and the ones the "
        "plan already has): a list of entries `item` (the name of the factor or quantity, or the formula, as the topic "
        "words it) and `by` (`setting:<the key in protocol.grid or the design variable>`, `metric:<the metric id>` or "
        "`equation:<the equation id>`). FI checks that each named element exists and covers what the topic states. Change "
        "nothing else: not a check, a threshold, a tolerance, a criterion, or a setting the plan already has right.")


def merge(old: dict[str, Any], new: dict[str, Any], gaps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """``old`` (a design block as written) with only what was asked taken from ``new``: settings of ``protocol.grid`` that
    are new, or that a gap names; metrics and equations whose id is new; design variables of a search that are new. Every
    check, threshold, tolerance and the rest are ``old``'s. ``None`` when nothing was taken."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return None
    o = old.get("protocol") if isinstance(old.get("protocol"), dict) else {}
    n = new.get("protocol") if isinstance(new.get("protocol"), dict) else {}
    protocol = dict(o)
    taken = False
    named = {g.get("axis") for g in gaps if g.get("axis")}
    new_grid = n.get("grid") if isinstance(n.get("grid"), dict) else {}
    if new_grid:
        grid = dict(o.get("grid") if isinstance(o.get("grid"), dict) else {})
        for axis, values in new_grid.items():
            if axis not in grid or axis in named:
                if grid.get(axis) != values:
                    grid[axis] = values
                    taken = True
        protocol["grid"] = grid
    for key, ident in (("metrics", "id"),):
        added = [m for m in (n.get(key) if isinstance(n.get(key), list) else []) if isinstance(m, dict) and m.get(ident)]
        have = {str(m.get(ident)) for m in (o.get(key) if isinstance(o.get(key), list) else []) if isinstance(m, dict)}
        fresh = [m for m in added if str(m[ident]) not in have]
        if fresh:
            protocol[key] = [*(o.get(key) if isinstance(o.get(key), list) else []), *fresh]
            taken = True
    new_model = n.get("model") if isinstance(n.get("model"), dict) else {}
    old_model = o.get("model") if isinstance(o.get("model"), dict) else {}
    old_eq = old_model.get("equations") if isinstance(old_model.get("equations"), list) else []
    have_ids = {str(e.get("id")) for e in old_eq if isinstance(e, dict)}
    fresh_eq = [e for e in (new_model.get("equations") if isinstance(new_model.get("equations"), list) else [])
                if isinstance(e, dict) and e.get("id") and str(e["id"]) not in have_ids]
    # An old equation the answer says now replaces the topic's (its source says so) is taken as rewritten.
    new_by_id = {str(e.get("id")): e for e in (new_model.get("equations") or []) if isinstance(e, dict)}
    rewritten = [dict(new_by_id[str(e.get("id"))]) if (isinstance(e, dict) and str(e.get("id")) in new_by_id
                                                       and new_by_id[str(e.get("id"))].get("source") != e.get("source")
                                                       and "topic" in str(new_by_id[str(e.get("id"))].get("source") or "").lower())
                 else e for e in old_eq]
    if fresh_eq or rewritten != old_eq:
        protocol["model"] = {**old_model, "equations": [*rewritten, *fresh_eq]}
        taken = True
    block_old = o.get("optimisation") if isinstance(o.get("optimisation"), dict) else None
    block_new = n.get("optimisation") if isinstance(n.get("optimisation"), dict) else None
    if block_old is not None and block_new is not None:
        have_vars = {str(v.get("name")) for v in block_old.get("design_variables") or [] if isinstance(v, dict)}
        extra = [v for v in block_new.get("design_variables") or [] if isinstance(v, dict) and str(v.get("name")) not in have_vars]
        if extra:
            protocol["optimisation"] = {**block_old, "design_variables": [*(block_old.get("design_variables") or []), *extra]}
            taken = True
    new_map = [m for m in (n.get("topic_map") if isinstance(n.get("topic_map"), list) else []) if isinstance(m, dict)]
    if new_map:
        old_map = [m for m in (o.get("topic_map") if isinstance(o.get("topic_map"), list) else []) if isinstance(m, dict)]
        keep = [m for m in old_map if _name_key(m.get("item")) not in {_name_key(x.get("item")) for x in new_map}]
        if [*keep, *new_map] != old_map:
            protocol["topic_map"] = [*keep, *new_map]
            taken = True
    out = {**old, "protocol": protocol}
    new_vars = new.get("variables") if isinstance(new.get("variables"), dict) else {}
    old_vars = old.get("variables") if isinstance(old.get("variables"), dict) else {}
    if new_vars and old_vars:
        merged = dict(old_vars)
        for key in ("independent", "dependent"):
            have = list(old_vars.get(key) or [])
            more = [v for v in (new_vars.get(key) or []) if v not in have]
            if more:
                merged[key] = [*have, *more]
        out["variables"] = merged
    return out if taken else None


# ---------------------------------------------------------------------------------------------------------------------
# plan.md, the paper's limitations, the stop
# ---------------------------------------------------------------------------------------------------------------------

def plan_lines(req: dict[str, Any], gaps: list[dict[str, Any]], *, reviewer: str, same_model: bool | None,
               unjudged: list[str] | None = None) -> list[str]:
    """What plan.md says in *What the topic asks for*, in plain words."""
    who = ("another model than the plan's" if same_model is False else
           "the plan's own model (no other model is named for `oracle_review`)" if same_model else "a model")
    lines = [f"> Read from the topic by {who}{f' ({reviewer})' if reviewer else ''}, then looked for in the plan.", ""]
    open_names = {g["name"] for g in gaps}
    unchecked = set(unjudged or [])
    for factor in req.get("factors") or []:
        levels = factor.get("levels") or ([f"{factor['range'][0]:g} to {factor['range'][1]:g}"] if factor.get("range") else [])
        lines.append(f"- To vary or compare: {factor['name']}" + (f" ({', '.join(str(x) for x in levels[:8])})" if levels else "")
                     + (" - NOT in the plan's settings" if factor["name"] in open_names else
                        " - could not be checked against the plan" if factor["name"] in unchecked else "") + ".")
    for quantity in req.get("quantities") or []:
        lines.append(f"- To report: {quantity['name']}" + (" - NOT measured in the plan" if quantity["name"] in open_names else
                                                                " - could not be checked against the plan" if quantity["name"] in unchecked else "") + ".")
    for formula in req.get("formulas") or []:
        lines.append(f"- Formula the topic states: `{formula['text']}`"
                     + (" - NOT in the plan's model" if formula["name"] in open_names else
                      " - could not be checked against the plan" if formula["name"] in unchecked else "") + ".")
    for comparison in req.get("comparisons") or []:
        lines.append(f"- To compare: {comparison['what']}")
    for constraint in req.get("constraints") or []:
        lines.append(f"- Constraint: {constraint}")
    return lines


def write_note(quest_root: Any) -> str:
    """For the writer: what the topic asked for that the plan could not include, so the limitations say it was not studied.
    Empty when nothing is open."""
    from pathlib import Path
    try:
        record = json.loads((Path(quest_root) / ".fi" / COVERED_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    left = record.get("left") if isinstance(record, dict) else None
    left = left if isinstance(left, list) else []
    unjudged = [str(n) for n in (record.get("unjudged") if isinstance(record, dict) else None) or [] if str(n).strip()]
    parts: list[str] = []
    if isinstance(record, dict) and record.get("unreadable"):
        parts.append("FI could not read the topic for what it explicitly asks for (the factors to vary, the quantities and the "
                     "formulas), so the plan was not held against it. Say plainly in the limitations that this was not checked.")
    if left:
        sentences = "; ".join(str(g.get("sentence") or "").rstrip(".") for g in left if isinstance(g, dict))
        parts.append(f"The topic asked for more than this study did: {sentences}. Say plainly in the limitations that these "
                     "were not studied, and do not write a result for them.")
    if unjudged:
        parts.append(f"FI could not check whether the plan includes what the topic asks for about {', '.join(unjudged)}. Say "
                     "plainly in the limitations that this was not checked.")
    return " ".join(parts)


def stop_steps(quest_id: str, gaps: list[dict[str, Any]], asked: int) -> list[str]:
    """The card of the stop, in plain words: what the topic asks for that the plan could not include."""
    return [
        "Nothing was run. The topic asks for something the plan could not include, so an experiment run from it would not "
        "answer the topic's question: " + " ".join(g["sentence"] for g in gaps),
        (f"FI asked the plan's model {asked} time(s) to add it, and the plan still lacks it." if asked else
         "FI did not ask the plan's model to add it (no model was available to ask)."),
        "You do not need to write it yourself. What to do:",
        "1. Try another model for the plan step: give `provider.node_models.plan_revise` (or `provider.model`) another model in "
        f"the quest's config.yaml, then `python launch.py --update {quest_id}`.",
        f"2. Or say it in words: `--resume {quest_id} --revise-plan \"<what to add, with its values>\"`.",
    ]


# ---------------------------------------------------------------------------------------------------------------------
# what a request that writes code is shown
# ---------------------------------------------------------------------------------------------------------------------

def topic_for_code(topic: Any, req: Any = None) -> str:
    """The topic as a request that writes or repairs code may see it: whole sentences that state a result or an expected
    value are left out (the code is judged against such numbers by FI, never written toward them), and so is each result
    the topic's reader listed as one. What is stated as a formula, a setting or a constraint stays."""
    text = str(topic or "").strip()
    for result in (req.get("stated_results") if isinstance(req, dict) else None) or []:
        if len(str(result)) >= 3:
            text = text.replace(str(result), "(left out: a result)")
    kept: list[str] = []
    for line in text.splitlines():
        pieces = re.split(r"(?<=[.;!?])\s+", line)
        kept.append(" ".join(p for p in pieces if not _RESULT_MARKERS.search(p)))
    return "\n".join(kept).strip()


def code_block(topic: Any, req: Any = None, *, limit: int = 6000, scrub: Any = None) -> str:
    """The text every request that writes or repairs code is shown about the topic: the topic itself (minus any result it
    states) and the formulas it states, which are the specification the plan's equations are to agree with. ``scrub(text)``
    (when given) takes out the numbers the plan's checks hold. The size cap is for the whole block, formulas included: the
    formulas are kept first (at most a third of it), the topic fills the rest."""
    body = topic_for_code(topic, req)
    if not body:
        return "(the topic is not available)"
    formulas = [f for f in (req.get("formulas") if isinstance(req, dict) else None) or [] if isinstance(f, dict)]
    tail = ""
    if formulas:
        tail = ("\n\nFormulas the topic states (use exactly these, as the plan's equations do; a different formula is a "
                "different study):\n" + "\n".join(f"- {f.get('name')}: {f.get('text')}" for f in formulas))
        if len(tail) > limit // 3:
            tail = tail[:limit // 3].rstrip() + "\n(more formulas are not shown)"
    if callable(scrub):
        body, tail = scrub(body), scrub(tail)
    room = max(0, limit - len(tail))
    if len(body) > room:
        body = body[:max(0, room - 40)].rstrip() + "\n(the rest of the topic is not shown)"
    return body + tail
