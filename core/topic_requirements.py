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


def _extreme(bound: Any, pick: Any) -> Any:
    """A design variable's ``low`` or ``high`` as one number: an array variable gives one per element, and the smallest low
    (the largest high) stands for the range it covers."""
    if not isinstance(bound, list):
        return bound
    leaves = [x for item in bound for x in ([_extreme(item, pick)] if isinstance(item, list) else [item])]
    leaves = [x for x in leaves if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return pick(leaves) if leaves else None

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
                        "stated": _text(item.get("stated")), "main": item.get("main") is True,
                        **({"compared": item["compared"]} if isinstance(item.get("compared"), bool) else {})})
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
            "comparisons": comparisons[:12], "constraints": constraints[:30], "stated_results": results[:20]}


def is_empty(req: Any) -> bool:
    return not (isinstance(req, dict) and any(req.get(k) for k in _SECTIONS))


# ---------------------------------------------------------------------------------------------------------------------
# the topic's own words, and what it compares
# ---------------------------------------------------------------------------------------------------------------------

def _sentences(topic: Any) -> list[str]:
    """The topic's sentences, each on one line as written (a line break or ``.``/``!``/``?`` and a space ends one)."""
    out: list[str] = []
    for line in str(topic or "").splitlines():
        for piece in re.split(r"(?<=[.!?])\s+", line.strip()):
            piece = " ".join(piece.split())
            if piece:
                out.append(piece)
    return out


def _squash(text: Any) -> str:
    return " ".join(str(text or "").lower().replace("`", "'").split())


def topic_sentence(topic: Any, stated: Any = "", text: Any = "", limit: int = 320) -> str:
    """The topic's own sentence for an item, word for word: the sentence that holds ``stated`` (the phrase the topic's
    reader copied), else, for an item the reader wrote in its own words (``text``: a constraint, a comparison), the one
    sentence that holds most of its words (at least two, and at least three in five of them). ``""`` when the topic has
    no such sentence: the item is then shown only as the reader wrote it."""
    sentences = _sentences(topic)
    if not sentences:
        return ""
    want = _squash(stated)
    if len(want) >= 3:
        for sentence in sentences:
            if want in _squash(sentence):
                return sentence[:limit]
        # A phrase that runs over two sentences of the topic: the first one it starts in.
        joined = _squash(" ".join(sentences))
        if want in joined:
            head = want[:40]
            for sentence in sentences:
                if head and head in _squash(sentence):
                    return sentence[:limit]
    words = _tokens(text)
    if len(words) >= 2:
        best, best_share = "", 0.0
        for sentence in sentences:
            share = len(words & _tokens(sentence)) / len(words)
            if share > best_share:
                best, best_share = sentence, share
        if best_share >= 0.6:
            return best[:limit]
    return ""


def carries_result(text: Any, req: Any = None) -> bool:
    """Whether ``text`` states a result: a sentence such as "the expected value is ..." or one that holds a result the
    topic's reader listed (``stated_results``). Such a sentence is never quoted to the plan or in plan.md."""
    text = str(text or "")
    if not text.strip():
        return False
    if _RESULT_MARKERS.search(text):
        return True
    squashed = _squash(text)
    for result in (req.get("stated_results") if isinstance(req, dict) else None) or []:
        said = result.get("stated") or result.get("text") if isinstance(result, dict) else result
        said = _squash(said)
        if len(said) >= 3 and said in squashed:
            return True
    return False


def quotable(topic: Any, req: Any, stated: Any = "", text: Any = "") -> str:
    """What may be quoted for an item: the topic's own sentence (:func:`topic_sentence`), else the reader's copied phrase,
    and ``""`` when that sentence or phrase states a result (:func:`carries_result`)."""
    sentence = topic_sentence(topic, stated, text)
    if sentence:
        return "" if carries_result(sentence, req) else sentence
    phrase = " ".join(str(stated or "").split())[:320]
    return "" if not phrase or carries_result(phrase, req) else phrase


def compared_axes(req: Any, design: Any, protocol: Any = None) -> dict[str, str]:
    """``{setting of the plan: the topic's factor}`` for each factor the topic compares (:func:`compared`), placed as the
    plan check places it: a setting whose name matches the factor's words, or the setting the plan's own topic map
    (``protocol.topic_map``) gives for it."""
    if not isinstance(req, dict):
        return {}
    plan = view(design, protocol)
    entries = [m for m in plan.get("topic_map") or [] if isinstance(m, dict)]
    out: dict[str, str] = {}
    for factor in req.get("factors") or []:
        if not isinstance(factor, dict) or not factor.get("name") or not compared(req, factor):
            continue
        wanted = _tokens(factor["name"])
        for axis in plan["axes"]:
            if wanted and _tokens(axis) and (_tokens(axis) <= wanted or wanted <= _tokens(axis)):
                out.setdefault(str(axis), str(factor["name"]))
        keys = {_name_key(factor.get("name")), _name_key(factor.get("text"))} - {""}
        for entry in entries:
            by = str(entry.get("by") or "")
            if _name_key(entry.get("item")) in keys and by.startswith("setting:"):
                out.setdefault(by.split(":", 1)[1].strip(), str(factor["name"]))
    return out


def compared(req: Any, factor: Any) -> bool:
    """Whether the topic compares results across the values of ``factor`` (so each value is a setting the study runs on
    its own, or every value is compared inside each run), and not only uses them: five values fitted together in each run,
    a range every run samples. The reader's own answer (``compared``) when it gave one. A reading made before the reader was
    asked that: ``True`` when the topic states the factor's levels or range (values to cover, as before), and otherwise,
    when the reader listed the topic's comparisons, whether one of them names the factor; ``True`` when it listed none."""
    if not isinstance(factor, dict):
        return True
    if isinstance(factor.get("compared"), bool):
        return factor["compared"]
    if factor.get("levels") or factor.get("range"):
        return True
    comparisons = [c for c in (req.get("comparisons") if isinstance(req, dict) else None) or []
                   if isinstance(c, dict) and c.get("between")]
    if not comparisons:
        return True
    name = _tokens(factor.get("name"))
    for comparison in comparisons:
        for entry in comparison.get("between") or []:
            words = _tokens(entry)
            if words and name and (words <= name or name <= words):
                return True
    return False


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
    """What the plan can be searched for: ``axes`` ({name: values}, from ``protocol.grid``, what each run compares inside
    it: ``protocol.in_each_run``, a search's design variables and the settings its own scan compares:
    ``protocol.optimisation.grid``),
    ``metrics`` ([(id, text)]), ``equations`` ([{id, formula, source}]) and ``assumptions`` (text). ``protocol`` is the one
    that runs when it differs from the design's own (the frozen one)."""
    design = design if isinstance(design, dict) else {}
    if not isinstance(protocol, dict):
        protocol = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    axes: dict[str, list[Any]] = {}
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    for axis, values in grid.items():
        axes[str(axis)] = list(values) if isinstance(values, list) else [values]
    # What each run compares inside it (all five temperatures of one fit, every procedure on the same data) is varied by
    # the study as much as a setting is: each run goes through all its values.
    inside = protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else {}
    for axis, values in inside.items():
        axes[str(axis)] = list(values) if isinstance(values, list) else [values]
    block = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else {}
    for variable in block.get("design_variables") if isinstance(block.get("design_variables"), list) else []:
        if not isinstance(variable, dict) or not str(variable.get("name") or "").strip():
            continue
        values = variable.get("values") if isinstance(variable.get("values"), list) else [
            v for v in (_extreme(variable.get("low"), min), _extreme(variable.get("high"), max)) if v is not None]
        axes[str(variable["name"]).strip()] = list(values)
    # The block's own scan compares the study across settings (of a design variable, or of a condition held fixed): those
    # are the settings the plan varies, whatever the plan's kind of study.
    for axis, values in (block.get("grid") if isinstance(block.get("grid"), dict) else {}).items():
        axes[str(axis)] = list(values) if isinstance(values, list) else [values]
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
    return {"axes": axes, "metrics": metrics, "equations": equations, "assumptions": assumptions, "topic_map": topic_map,
            "search": bool(block)}


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
    the named element exists, its name relates to the item's own words (as the name matcher judges) and, for a factor with
    stated levels or a range, its values cover them; ``("judge", element)`` when it exists and covers them but its name is
    unrelated (an axis whose values merely happen to fit): only a reader's confirmation makes that a cover; ``("bad",
    why)`` when it does not exist or does not cover; ``None`` when the plan maps nothing for the item."""
    wanted = {_name_key(item.get("name")), _name_key(item.get("text"))} - {""}
    entries = [m for m in plan.get("topic_map") or [] if _name_key(m.get("item")) in wanted]
    if not entries:
        return None
    why = "nothing it names exists"
    for entry in entries:
        head, _sep, name = str(entry["by"]).partition(":")
        head, name = head.strip().lower(), name.strip()
        if kind == "factors" and head in ("optimisation", "optimization", "search") and plan.get("search"):
            # What the search for the best design itself finds (a free-form design, the best member of a family) is not a
            # setting with values to cover: the plan says its block covers it, and a reader confirms that it does.
            return "judge", "optimisation"
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
            wanted_tokens, axis_tokens = _tokens(item.get("name")), _tokens(axis)
            related = bool(axis_tokens and wanted_tokens) and (axis_tokens <= wanted_tokens or wanted_tokens <= axis_tokens)
            return ("ok" if related else "judge"), axis
        if kind == "quantities" and head == "metric":
            found = next((i for i, _t in plan["metrics"] if _name_key(i) == _name_key(name)), None)
            if found is not None:
                wanted_tokens = _tokens(item.get("name"))
                text = next((t for i, t in plan["metrics"] if i == found), found)
                return ("ok" if wanted_tokens and wanted_tokens <= _tokens(text) else "judge"), found
            why = f"the metric {name} is not among the plan's metrics"
        if kind == "formulas" and head == "equation":
            found = next((e["id"] for e in plan["equations"] if _name_key(e["id"]) == _name_key(name) and e["formula"].strip()), None)
            if found is not None:
                return "judge", found  # an equation id says nothing about the formula: a reader confirms it implements it
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
        if axis is None and not compared(req, factor):
            # The topic uses its values (all of them in each run, a range every run covers) and compares no result across
            # them: it is not a setting the study must sweep. Asking for one put five values fitted together into the grid
            # as five settings, each run five times over.
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
    stated = " ".join(str(item.get("stated") or "").split())
    if kind == "factor":
        levels = item.get("levels") or ([f"{item['range'][0]:g} to {item['range'][1]:g}"] if item.get("range") else [])
        sentence = (f"The topic says to vary or compare {name}"
                    + (f" ({', '.join(str(x) for x in levels[:8])})" if levels else "")
                    + (f" (its words: “{stated}”)" if stated else "")
                    + ", but the plan's settings do not include it.")
    elif kind == "level":
        sentence = f"The topic states which values of {name} to use, but {why}."
    elif kind == "quantity":
        sentence = (f"The topic asks to report {name}" + (f" (its words: “{stated}”)" if stated else "")
                    + ", but the plan does not measure it.")
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
        row = {"id": ident, "kind": kind[:-1] if kind != "quantities" else "quantity",
               **{k: v for k, v in item.items() if k in ("name", "levels", "range", "text", "stated") and v}}
        mapped = mapping_of(req, plan, kind, item)
        if mapped is not None and mapped[0] == "judge":
            row["plan_says_it_is_covered_by"] = mapped[1]
        items.append(row)
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


def request(topic: str, gaps: list[dict[str, Any]], *, search: bool = False) -> str:
    """The request to the plan's model for what the topic asks and the plan lacks. Nothing else may change. ``search``:
    the plan is a search for the best design, which has no ``protocol.grid`` beside its block (the settings to compare
    belong to the block's own scan, ``protocol.optimisation.grid``)."""
    lines = [f"- {g['sentence']}" for g in gaps]
    where = (
        "A setting to vary or compare is a key of `protocol.optimisation.grid` with every value the topic states (this "
        "plan is a search for the best design, so do NOT write a `protocol.grid`): either one of the design variables, or "
        "a condition held fixed during the search, which you also list under `protocol.optimisation.fixed` with the value "
        "the search holds it at (a condition whose values are names, such as a family or a mode, needs no `fixed` entry: "
        "FI holds it at the first name listed). What the search itself finds (the best design) needs no setting: say in "
        "`protocol.topic_map` that `optimisation` covers it. "
        if search else
        "A setting to vary or compare is a key of `protocol.grid` with every value the topic states (or, for a search for "
        "the best design, a design variable of `protocol.optimisation` with its range, and its value in `protocol.optimisation.baseline.values`). "
        "When the topic's words put all its values into each run (one fit across all of them, every method applied to the "
        "same simulated data), it is a key of `protocol.in_each_run` with all its values instead, not a setting of "
        "`protocol.grid`: a grid setting is run on its own, once per value. ")
    return (
        "The topic asks for things this plan does not yet include:\n" + "\n".join(lines)
        + f"\n\nThe topic: “{' '.join(str(topic or '').split())[:1500]}”\n\n"
        "Write them into the design block. " + where + "A quantity "
        "to report is an entry of `protocol.metrics` (an `id`, a `kind`, an `estimand` and a `unit`). A formula the topic "
        "states is an equation of `protocol.model.equations`, written as the topic writes it; if the plan needs another "
        "formula in its place, keep that one and say in its `source` that it replaces the topic's, and why. Also say, in "
        "`protocol.topic_map`, which element of the plan covers each thing the topic asks for (the ones above and the ones the "
        "plan already has): a list of entries `item` (the name of the factor or quantity, or the formula, as the topic "
        "words it) and `by` (`setting:<the key in protocol.grid or protocol.optimisation.grid, or the design variable>`,`metric:<the metric id>` or "
        "`equation:<the equation id>`). FI checks that each named element exists and covers what the topic states. Change "
        "nothing else: not a check, a threshold, a tolerance, a criterion, or a setting the plan already has right.")


def merge(old: dict[str, Any], new: dict[str, Any], gaps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """``old`` (a design block as written) with only what was asked taken from ``new``: settings of ``protocol.grid`` that
    are new, or that a gap names; what each run compares inside it (``protocol.in_each_run``) the same way, and a factor
    ``old`` compares inside each run stays there, whichever of the two places the answer wrote it in; metrics and equations
    whose id is new; design variables of a search that are new. Every check, threshold, tolerance and the rest are
    ``old``'s. ``None`` when nothing was taken."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return None
    o = old.get("protocol") if isinstance(old.get("protocol"), dict) else {}
    n = new.get("protocol") if isinstance(new.get("protocol"), dict) else {}
    protocol = dict(o)
    taken = False
    named = {g.get("axis") for g in gaps if g.get("axis")}
    new_grid = n.get("grid") if isinstance(n.get("grid"), dict) else {}
    search = isinstance(o.get("optimisation"), dict)
    if search:
        # A plan that is a search for the best design has no `protocol.grid` beside its block: the settings to compare
        # belong to the block's own scan (`optimisation.grid`), so the model's answer is read that way, whichever of the
        # two places it wrote them in.
        nb = n.get("optimisation") if isinstance(n.get("optimisation"), dict) else {}
        new_grid = {**(nb.get("grid") if isinstance(nb.get("grid"), dict) else {}), **new_grid}
        scan = dict(o["optimisation"].get("grid") if isinstance(o["optimisation"].get("grid"), dict) else {})
        for axis, values in new_grid.items():
            if axis not in scan or axis in named:
                if scan.get(axis) != values:
                    scan[axis] = values
                    taken = True
        held = dict(o["optimisation"].get("fixed") if isinstance(o["optimisation"].get("fixed"), dict) else {})
        for key, value in (nb.get("fixed") if isinstance(nb.get("fixed"), dict) else {}).items():
            if key not in held and key in scan:  # a condition the answer holds fixed so that the scan can compare it
                held[key] = value
                taken = True
        if scan:
            protocol["optimisation"] = {**o["optimisation"], "grid": scan, **({"fixed": held} if held else {})}
        new_grid = {}
    old_inside = o.get("in_each_run") if isinstance(o.get("in_each_run"), dict) else {}
    new_inside = n.get("in_each_run") if isinstance(n.get("in_each_run"), dict) else {}
    have_grid = o.get("grid") if isinstance(o.get("grid"), dict) else {}
    # A factor each run compares inside it is never a setting of the grid as well: the answer's values for it are read
    # as its list, and one the plan already sweeps as a setting stays a setting.
    asked_inside = {**{a: v for a, v in new_grid.items() if a in old_inside},
                    **{a: v for a, v in new_inside.items() if a not in have_grid}}
    new_grid = {a: v for a, v in new_grid.items() if a not in old_inside}
    if asked_inside:
        inside = dict(old_inside)
        for axis, values in asked_inside.items():
            if (axis not in inside or axis in named) and isinstance(values, list) and inside.get(axis) != values:
                inside[axis] = values
                taken = True
        protocol["in_each_run"] = inside
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
            merged_block = {**protocol.get("optimisation", block_old),
                            "design_variables": [*(block_old.get("design_variables") or []), *extra]}
            # A variable the baseline has no value for is a block the plan reader refuses: the answer's baseline value of
            # each added variable comes with it (the design to beat keeps every value it already had).
            old_base = block_old.get("baseline") if isinstance(block_old.get("baseline"), dict) else {}
            new_base = block_new.get("baseline") if isinstance(block_new.get("baseline"), dict) else {}
            old_values = old_base.get("values") if isinstance(old_base.get("values"), dict) else {}
            new_values = new_base.get("values") if isinstance(new_base.get("values"), dict) else {}
            names = {" ".join(str(x.get("name")).split()) for x in extra}
            added = {k: v for k, v in new_values.items() if " ".join(str(k).split()) in names and k not in old_values}
            if added and old_base:
                merged_block["baseline"] = {**old_base, "values": {**old_values, **added}}
            protocol["optimisation"] = merged_block
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
               unjudged: list[str] | None = None, topic: str = "") -> list[str]:
    """What plan.md says in *What the topic asks for*, in plain words, each factor and quantity with the topic's own
    sentence for it (``topic``; the reader's copied phrase when the topic is not given)."""
    who = ("another model than the plan's" if same_model is False else
           "the plan's own model (no other model is named for `oracle_review`)" if same_model else "a model")
    lines = [f"> Read from the topic by {who}{f' ({reviewer})' if reviewer else ''}, then looked for in the plan.", ""]
    open_names = {g["name"] for g in gaps}
    unchecked = set(unjudged or [])

    def words(item: dict[str, Any]) -> str:
        said = quotable(topic, req, item.get("stated"))
        return f" The topic: “{said}”" if said else ""

    for factor in req.get("factors") or []:
        levels = factor.get("levels") or ([f"{factor['range'][0]:g} to {factor['range'][1]:g}"] if factor.get("range") else [])
        head = ("- To vary or compare: " if compared(req, factor) else
                "- To use as the topic states, in every run (it compares no result across it, so it is not a setting to "
                "sweep): ")
        lines.append(f"{head}{factor['name']}" + (f" ({', '.join(str(x) for x in levels[:8])})" if levels else "")
                     + (" - NOT in the plan's settings" if factor["name"] in open_names else
                        " - could not be checked against the plan" if factor["name"] in unchecked else "")
                     + "." + words(factor))
    for quantity in req.get("quantities") or []:
        lines.append(f"- To report: {quantity['name']}" + (" - NOT measured in the plan" if quantity["name"] in open_names else
                                                                " - could not be checked against the plan" if quantity["name"] in unchecked else "")
                     + "." + words(quantity))
    for formula in req.get("formulas") or []:
        lines.append(f"- Formula the topic states: `{formula['text']}`"
                     + (" - NOT in the plan's model" if formula["name"] in open_names else
                      " - could not be checked against the plan" if formula["name"] in unchecked else "") + ".")
    for comparison in req.get("comparisons") or []:
        said = quotable(topic, req, text=comparison["what"])
        lines.append(f"- To compare: {comparison['what']}" + (f" (the topic: “{said}”)" if said else ""))
    for constraint in req.get("constraints") or []:
        said = quotable(topic, req, text=constraint)
        lines.append(f"- Constraint: {constraint}" + (f" (the topic: “{said}”)" if said and _squash(said) != _squash(constraint) else ""))
    return lines


def write_note(quest_root: Any) -> str:
    """For the writer: what the topic asked for that the plan could not include, so the limitations say it was not studied.
    Empty when nothing is open."""
    from pathlib import Path
    try:
        record = json.loads((Path(quest_root) / ".fi" / COVERED_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    try:
        requirements = json.loads((Path(quest_root) / ".fi" / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        requirements = {}
    left = record.get("left") if isinstance(record, dict) else None
    left = left if isinstance(left, list) else []
    unjudged = [str(n) for n in (record.get("unjudged") if isinstance(record, dict) else None) or [] if str(n).strip()]
    parts: list[str] = []
    if isinstance(record, dict) and record.get("unreadable"):
        parts.append("FI could not read the topic for what it explicitly asks for (the factors to vary, the quantities and the "
                     "formulas), so the plan was not held against it. Say plainly in the limitations that this was not checked.")
    if left:
        sentences = "; ".join(str(g.get("sentence") or "").rstrip(".") for g in left if isinstance(g, dict))
        parts.append(f"The topic asked for more than this study did: {sentences}. Say in the limitations that the study did not "
                     "do each of them, by the short name FI's record of this run (below) gives it; do not write a result "
                     "for them.")
    if unjudged:
        parts.append(f"FI could not check whether the plan includes what the topic asks for about {', '.join(unjudged)}. Say "
                     "plainly in the limitations that this was not checked.")
    ledger_left = [str(g).strip() for g in (record.get("ledger_left") if isinstance(record, dict) else None) or []
                   if str(g).strip()]
    if ledger_left:
        steps = any(g.startswith("`protocol.steps`") for g in ledger_left)
        parts.append("The plan did not give all of what it was asked for about what it simplifies and how its metrics are "
                     f"defined ({'; '.join(g.rstrip('.') for g in ledger_left)}). Say plainly in the limitations that the "
                     "simplifications or definitions of this study are not fully stated"
                     + (", and that the steps of the method are not all described in full" if steps else "")
                     + ", and do not present a metric's definition as taken from a source.")
    own = [str(m).strip() for m in (record.get("unsourced_definitions") if isinstance(record, dict) else None) or [] if str(m).strip()]
    if own:
        parts.append(f"These metrics are the study's own definitions, not taken from a source: {', '.join(own)}. State plainly in "
                     "the limitations which metrics are the study's own definitions, and do not cite a source for them.")
    raw = requirements.get("constraints") if isinstance(requirements, dict) else None
    constraints = [str(c).strip().rstrip(".").strip() for c in (raw if isinstance(raw, list) else [])]
    constraints = [c for c in constraints if c]
    if constraints:
        parts.append(f"The topic set these requirements on the study and the paper: {'; '.join(constraints)}. State the seed "
                     "the code used and the measured run time against the topic's limit in the Methods, from FI's record of "
                     "this run (below). For each other requirement the study did not meet (the code files, the "
                     "uncertainty of every number), say plainly in the limitations that it was not met and why; never leave "
                     "one out silently, and never claim one was met unless the record shows it.")
    return " ".join(parts)


def stop_tried(gaps: list[dict[str, Any]], asked: int, problem: str = "") -> list[str]:
    """What FI did before the plain stop (the card's "tried" lines): it asked the plan's model, and the plan still lacks
    what the topic asks for. ``problem``: why the plan's model's last answer was not used."""
    return [
        (f"asked the plan's model {asked} time(s) to add it, each time quoting why the last answer was not used"
         if asked else "had no model to ask for it (or the plan was already settled)"),
        *([f"the last answer was not used: {' '.join(str(problem).split())[:400]}"] if problem else []),
        *[g["sentence"] for g in gaps[:4]],
        "did not run a study that leaves it out: it would not answer the topic's question",
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


#: Where the topic's words win over FI's own defaults, and where they never do.
PRECEDENCE = ("For what to vary, which values, how many runs, how long a run may take, how many code files there are and "
              "the page limit, the topic's words hold over FI's other instructions; FI's checks, its tests and its rules "
              "about results always apply, whatever the topic says")


#: The most characters the checklist shown to the plan may take (each item carries the topic's own sentence for it).
CHECKLIST_LIMIT = 4200


def checklist_block(req: Any, *, limit: int = CHECKLIST_LIMIT, topic: Any = "") -> str:
    """The topic's requirements as plain lines the plan is shown before it is written (``""`` for an empty or failed
    reading): factors with their stated levels or range, quantities to report, stated formulas, the comparisons and
    constraints, each with the topic's own sentence for it (:func:`topic_sentence`; the reader's copied phrase when
    ``topic`` is not given), so a reader's short name never stands in for what the topic says. A factor the topic uses
    without comparing a result across it is said to be that, not a setting to vary. Never ``stated_results`` (a plan is
    not shown the answer). Pure; at most ``limit`` characters, what does not fit is counted in a last line."""
    if not isinstance(req, dict) or is_empty(req):
        return ""

    shown: dict[str, str] = {}

    def said(stated: Any = "", text: Any = "") -> str:
        words = quotable(topic, req, stated, text)
        if not words:
            return ""
        if words in shown:  # a sentence quoted in full above is named by its mark
            return f" — the topic {shown[words]}"
        shown[words] = f"[T{len(shown) + 1}]"
        return f" — the topic {shown[words]}: “{words}”"

    lines: list[str] = []
    for item in req.get("factors") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        levels = item.get("levels") or []
        rng = item.get("range") or []
        if levels:
            listed = ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in levels[:10])
            detail = f": {listed}" + (f" (and {len(levels) - 10} more)" if len(levels) > 10 else "")
        else:
            detail = f": {rng[0]:g} to {rng[1]:g}" if len(rng) == 2 else ""
        if compared(req, item):
            lines.append(f"- vary or compare {item['name']}{detail}{said(item.get('stated'))}")
        else:
            lines.append(f"- use {item['name']}{detail} as the topic states it, in every run; the topic compares no result "
                         "across it, so it is not a setting of `protocol.grid` (all its values together in each run go in "
                         f"`protocol.in_each_run`){said(item.get('stated'))}")
    for item in req.get("quantities") or []:
        if isinstance(item, dict) and item.get("name"):
            lines.append(f"- report {item['name']}{said(item.get('stated'))}")
    for item in req.get("formulas") or []:
        if isinstance(item, dict) and item.get("text"):
            name = str(item.get("name") or "").strip()
            lines.append(f"- formula{' ' + name if name else ''}: {str(item['text'])[:160]}")
    for item in req.get("comparisons") or []:
        if isinstance(item, dict) and item.get("what"):
            lines.append(f"- compare: {str(item['what'])[:200]}{said(text=item['what'])}")
    for text in req.get("constraints") or []:
        if not str(text).strip():
            continue
        words = quotable(topic, req, text=text)
        if not words:
            lines.append(f"- constraint: {str(text)[:160]}")
        elif words not in shown:  # the topic's own sentence in place of the reader's words
            lines.append(f"- constraint{said(text=text).replace(' — the topic', ', the topic', 1)}")
        else:  # its sentence is quoted above already
            lines.append(f"- constraint: {str(text)[:160]} (the topic {shown[words]})")
    head = ("What the topic asks for (each item must appear as a setting, a metric or an equation of the plan, "
            "or be named in `simplifications`). " + PRECEDENCE + ":")
    out = [head]
    used = len(head)
    for i, line in enumerate(lines):
        if used + len(line) + 1 > limit - 40:
            out.append(f"(and {len(lines) - i} more)")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)
