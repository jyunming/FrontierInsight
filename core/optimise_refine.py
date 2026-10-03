"""A person's refine of a search for the best design: "push it further", "I want at most 46 K", "6 K better".

The decisions behind it (2026-09-30): there is no improvement a search must reach; a person who is not satisfied with the
best design FI found gets closer to what they want by refining. So a refine of a search for the best design (a
``protocol.optimisation`` block, :mod:`core.optimisation_plan`) is read here, without a model, before the writer sees it:

* **search further** (:data:`SEARCH`): the note asks to push further, names a target value of the objective ("at most
  46 K", "reach 0.8"), an improvement over the baseline ("6 K better"), or more evaluations ("100 more evaluations").
  FI adds a round to the search (``.fi/optimisation/continue.json``, :func:`add_round`): the search goes on from the
  best design so far with the added evaluations (one start's share, ``per_start``, unless the note gives a number),
  stops when a design that meets every limit reaches the target, and the check at finer settings runs again on the
  result (:mod:`core.optimise`, :mod:`core.optimum_check`); the paper's *Best design found* section says whether the
  target was reached (:mod:`core.best_design_report`).
* **a new study** (:data:`NEW_STUDY`): the note asks to change what is optimised, its limits, or the ranges the design
  may take. That is not a continuation: it is a new study. Nothing is changed; the review pause says so and how to
  start one, and the note is not kept (so no later step can act on it).
* **unclear** (:data:`UNCLEAR`): a number whose meaning for the objective cannot be told (``at most 5`` for a quantity
  made as high as possible), or a target the best design already reaches. Nothing is changed; the pause asks again.
* anything else is an ordinary refine and goes to the writer as before.

Every reading is said back in plain words (``read_as``) at the pause, in run.log and in the paper, so a person sees
how FI understood them.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from . import optimise as _optimise

SEARCH, NEW_STUDY, UNCLEAR = "search", "new_study", "unclear"

# A number: a thousands separator is a comma followed by three digits ("1,500"); any other comma is a decimal point.
_NUM = r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)"
# An explicit request to search further (a note that only mentions a "better design", or more "runs" of something,
# is not one: it may be about the text).
_PUSH = re.compile(
    r"\b(?:push(?:\s+(?:it|this|the\s+(?:search|design|result|optimi[sz]ation)))?\s+further|keep\s+(?:searching|"
    r"optimi[sz]ing)|search\s+(?:further|longer)|continue\s+(?:the\s+)?(?:search|optimi[sz]ation)|"
    r"(?:more|bigger|larger)\s+search\s+budget|find\s+a\s+better\s+design|improve\s+the\s+design\s+further)\b"
    r"|再推|繼續搜尋|再優化|繼續優化", re.I)
_BUDGET = re.compile(r"\b(\d{1,7})\s+(?:more|extra|additional)\s+(?:evaluations?|runs?|designs?)\b"
                     r"|\b(?:run|use|spend|add|allow|with|another)\s+(\d{1,7})\s+evaluations?\b"
                     r"|\bsearch\s+budget\s+of\s+(\d{1,7})\b|多\s*(\d{1,7})\s*次", re.I)
_LOW = r"at\s+most|no\s+more\s+than|below|under|less\s+than|lower\s+than|down\s+to|≤|<=|<|至多|最多|低於"
_HIGH = r"at\s+least|no\s+less\s+than|above|over|more\s+than|higher\s+than|up\s+to|≥|>=|>|至少|高於"
_NEUTRAL = r"target(?:\s+(?:of|value\s+of|value))?|reach|get\s+(?:it\s+)?to|目標"
_COMPARE = re.compile(rf"(?:(?<![\w])(?P<op>{_LOW}|{_HIGH}|{_NEUTRAL})|(?P<sym>≤|<=|<|≥|>=|>))\s*:?\s*{_NUM}"
                      r"(?P<rest>[^.;\n]{0,40})", re.I)
_BETTER = re.compile(r"^\s*(?:%|[^\s\d.;,]{1,12})?\s*(?:better|lower|higher|improvement|less|more|reduction|gain|"
                     r"smaller|larger|cooler|faster)\b", re.I)
_IMPROVE_BY = re.compile(rf"\b(?:improve(?:ment)?|better|reduce|reduction|lower|raise|increase)\s+(?:it\s+)?"
                         rf"(?:by|of)\s+(?:at\s+least\s+)?{_NUM}(?P<rest>[^.;\n]{{0,40}})", re.I)
_BY_X_BETTER = re.compile(rf"{_NUM}(?P<rest>\s*(?:%|[^\s\d.;,]{{1,12}})?\s+(?:better|lower|higher|cooler|less|more)"
                          r"\s+than\s+the\s+baseline)", re.I)
_OBJECTIVE_CHANGE = re.compile(
    r"\b(?:change|switch|replace)\b[^.;\n]{0,30}\b(?:the\s+objective|the\s+goal\s+of\s+the\s+(?:search|study)|"
    r"what\s+(?:is|we|you)\s+optimi[sz]e)", re.I)
_DIRECTION = re.compile(r"\b(minimi[sz]e|maximi[sz]e|optimi[sz]e\s+for)\s+(?:the\s+)?([A-Za-z_][\w\-]*(?:\s+[A-Za-z_]"
                        r"[\w\-]*){0,2})", re.I)
_LIMIT_CHANGE = re.compile(
    r"\b(?:drop|remove|relax|loosen|tighten|ignore|lift|change|raise|increase|decrease|add|new|another)\b"
    r"[^.;\n]{0,40}\b(?:limits?|constraints?|caps?)\b", re.I)
_RANGE_CHANGE = re.compile(
    r"\b(?:widen|extend|expand|enlarge|narrow|change|increase|decrease)\b[^.;\n]{0,30}\b(?:range|ranges|bounds?)\b"
    r"|\boutside\s+(?:the|its)\s+range", re.I)
_NEGATED = re.compile(r"\b(?:don'?t|do\s+not|does\s+not|not|never|without|no)\s+(?:\w+\s+){0,2}$", re.I)
_STOPWORDS = {"the", "a", "an", "of", "it", "this", "design", "result", "value", "instead", "and", "or", "to", "in"}


def _matches(pattern: re.Pattern[str], note: str) -> list[re.Match[str]]:
    """The matches of ``pattern`` in ``note`` that are not negated ("do not change the limits" does not ask it)."""
    return [m for m in pattern.finditer(note) if not _NEGATED.search(note[max(0, m.start() - 30):m.start()])]


def _names(text: str, names: list[str]) -> list[str]:
    """Which of ``names`` (a quantity's or a design variable's) ``text`` mentions, as a name of its own (``y`` in
    "y-axis" is not the variable ``y``)."""
    return [n for n in names if n and re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text, re.I)]


def _float(text: str) -> float | None:
    text = text.strip()
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _words(text: Any) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", str(text or "").lower()) if w and w not in _STOPWORDS}


def _sign(block: dict[str, Any]) -> float:
    return 1.0 if str(block["objective"].get("direction")) == "minimise" else -1.0


# Units FI converts between by an SI prefix alone (9500 mK is 9.5 K). Anything else with a different unit is refused, not
# guessed (a Celsius figure for a kelvin quantity, a W for a K).
_BASES = {"K", "Pa", "W", "V", "A", "Hz", "J", "s", "m", "g", "N", "Ω", "ohm", "T", "F", "H", "C", "L", "mol", "eV",
          "Wh", "B", "bar", "S"}
_PREFIXES = {"p": 1e-12, "n": 1e-9, "µ": 1e-6, "μ": 1e-6, "u": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "M": 1e6,
             "G": 1e9, "T": 1e12}
_PERCENT = re.compile(r"\s*(?:%|percent\b|per\s*cent\b)", re.I)
_TOKEN = re.compile(r"\s*([A-Za-zµμΩ°][A-Za-z0-9µμΩ°/^·-]*)")
# A unit written out in words, as the symbol it names.
_NAMED = {"kelvin": "K", "kelvins": "K", "watt": "W", "watts": "W", "volt": "V", "volts": "V", "pascal": "Pa",
          "pascals": "Pa", "hertz": "Hz", "joule": "J", "joules": "J", "seconds": "s", "meters": "m", "metres": "m",
          "grams": "g", "celsius": "°C", "fahrenheit": "°F"}


def _split_unit(text: str) -> tuple[float, str] | None:
    """``(prefix factor, base unit)`` of a unit written as an SI prefix and a known base unit ("mK" is 1e-3 of K)."""
    if text in _BASES:
        return 1.0, text
    if len(text) > 1 and text[0] in _PREFIXES and text[1:] in _BASES:
        return _PREFIXES[text[0]], text[1:]
    return None


def _unit_after(rest: str, unit: str) -> str:
    """The unit written right after a number (``rest`` is the text after it): ``"%"``, the objective's own unit, or a
    known unit; ``""`` when the number is bare (or followed by a word that is not a unit)."""
    unit = _sym(unit)
    if _PERCENT.match(rest):
        return "%"
    m = _TOKEN.match(rest)
    if not m:
        return ""
    token = m.group(1).rstrip("-/")
    token = _NAMED.get(token.lower(), token)
    # A unit of the objective's own, a known one, or a compound / degree one (K/s, °C), which FI cannot convert.
    if (unit and token == unit) or _split_unit(token) or token.startswith("°") or re.search(r"[/^·]", token):
        return token
    return ""


def _sym(unit: str) -> str:
    """A unit written out in words ("kelvin") as its symbol ("K")."""
    return _NAMED.get(unit.strip().lower(), unit.strip())


# A word between a number and "better" that names a unit ("2 min better than the baseline").
_SLOT = re.compile(r"\s*([^\s\d.;,%]{1,12})\s+(?:better|lower|higher|cooler|less|more|smaller|larger|faster)\b", re.I)
_BETTER_WORDS = {"better", "lower", "higher", "cooler", "less", "more", "smaller", "larger", "faster", "improvement",
                 "reduction", "gain"}


def _is_percent_unit(unit: str) -> bool:
    return unit.strip().lower() in {"%", "percent", "per cent", "pct"}


def _with_unit(value: float, rest: str, unit: str, quantity: str, sign: float, relative_ok: bool = False) -> tuple[float, bool, str, str]:
    """``(value in the objective's unit, whether it is a percent, the unit as written, a refusal)`` for a number
    followed by ``rest``. A percent of a quantity that is not itself in percent is relative (the caller says to what);
    a unit of its own is converted when it differs only by an SI prefix, else refused; a bare number is in the
    objective's unit."""
    written = _unit_after(rest, unit)
    if not written and relative_ok:
        slot = _SLOT.match(rest)
        if slot and slot.group(1).lower() not in _BETTER_WORDS:
            written = slot.group(1)  # a unit FI does not know: refused below, not read in the objective's unit
    if not written or (_is_percent_unit(unit) and written == "%"):
        return value, False, written, ""
    if written == "%":
        if not unit and not relative_ok:
            return value, True, written, (
                f"it gives a percent, but {quantity} has no unit, so FI cannot tell what it is a percent of. Give the "
                f"value as a plain number (for example \"at {'most' if sign > 0 else 'least'} 0.85\")")
        return value, True, written, ""
    converted = value if written == _sym(unit) else _convert(value, written, unit)
    if converted is None:
        return value, False, written, (
            f"it gives the target in {written}, but this study's {quantity} is in {unit or 'no unit'} and FI does not "
            f"convert between them. Give the value in {unit or 'the same terms as the study'} (for example \"at "
            f"{'most' if sign > 0 else 'least'} X{' ' + unit if unit else ''}\")")
    return converted, False, written, ""


def _convert(value: float, written: str, unit: str) -> float | None:
    """``value`` written in ``written`` as a value in the objective's ``unit``, or ``None`` when FI cannot convert."""
    unit = _sym(unit)
    if written == unit:
        return value
    a, b = _split_unit(written), _split_unit(unit)
    if a and b and a[1] == b[1]:
        return float(f"{value * a[0] / b[0]:.12g}")
    return None


def _tied(note: str, m: re.Match[str], block: dict[str, Any]) -> bool:
    """Whether the number ``m`` matched is about the objective: the objective's unit right after it, or the objective
    named (its name, or a word of its meaning) next to it, or "than the baseline"."""
    objective = block["objective"]
    unit = str(objective.get("unit") or "").strip()
    rest = (m.groupdict().get("rest") or "").strip()
    if unit and unit != "%" and re.match(rf"{re.escape(unit)}(?![\w])", rest):
        return True
    if re.search(r"\bthan\s+the\s+baseline\b", note[m.start():m.end() + 40], re.I):
        return True
    written = _unit_after(rest, unit)
    if written and written != "%" and not _is_percent_unit(unit) and written != _sym(unit) \
            and _convert(1.0, written, unit) is None:
        # A unit that does not fit the objective ("at most 100 s", "10 MB") is only read as a target, to be refused, where
        # the comparison follows the objective's own name directly ("f at most 9.5 W", "f should be at most 9.5 W")
        # or says "target" / "reach"; a note about the paper that merely names the quantity is an ordinary refine.
        before = note[max(0, m.start() - 60):m.start()].replace("_", " ")
        name = re.escape(str(objective.get("quantity") or "").replace("_", " "))
        return bool(re.search(rf"(?<![\w-]){name}\s*(?:target\s+)?(?:(?:is|should\s+be|must\s+be|to\s+be)\s+)?:?\s*$", before, re.I)
                    or re.fullmatch(_NEUTRAL, (m.groupdict().get("op") or "").strip(), re.I))
    # A number followed by a word of its own ("200 words", "3 significant figures") is about something else.
    word = re.match(r"([A-Za-z]+)", rest)
    if word and not written and not _BETTER.match(rest) and word.group(1).lower() not in _words(objective.get("quantity")):
        return False
    # The objective named next to the number, outside the comparison's own words ("below" is not a word of a meaning).
    near = note[max(0, m.start() - 50):m.start()] + " " + note[m.end():m.end() + 50]
    if _names(near.replace("_", " "), [str(objective.get("quantity") or "").replace("_", " ")]):
        return True
    ops = _words(re.sub(r"\\s\+|[\\()|?:]", " ", _LOW + " " + _HIGH + " " + _NEUTRAL))
    meaning = {w for w in _words(objective.get("meaning")) if len(w) > 3} - ops
    return bool(meaning & _words(near))


def _new_study(note: str, block: dict[str, Any]) -> str:
    """Why the note asks for a new study (a change to what is optimised, its limits or its ranges), or ``""``. Only a
    request that names the study's own quantities or design variables counts: "minimise jargon" or "change the y-axis
    range of Figure 2" is about the paper."""
    objective = block["objective"]
    quantity = str(objective.get("quantity") or "")
    limited = [str(c.get("quantity") or "") for c in block.get("constraints") or []]
    variables = [str(v.get("name") or "") for v in block.get("design_variables") or []]
    mine = _words(quantity) | _words(objective.get("meaning"))
    if _matches(_OBJECTIVE_CHANGE, note):
        return "it asks to change what is optimised"
    for m in _matches(_DIRECTION, note):
        verb, what = m.group(1).lower(), _words(m.group(2))
        wants = "minimise" if verb.startswith("minimi") else "maximise" if verb.startswith("maximi") else ""
        other = _names(m.group(2), [*limited, *variables])
        if other and not (what & mine):
            return f"it asks to {m.group(1).lower()} {other[0]}, not {quantity}"
        if what & mine and wants and wants != objective.get("direction"):
            return f"it asks to {wants} {quantity}, which the study makes as {'low' if wants == 'maximise' else 'high'} " \
                   "as possible"
    for m in _matches(_LIMIT_CHANGE, note):
        if _names(note[max(0, m.start() - 40):m.end() + 40], limited):
            return "it asks to change the study's limits"
    for name in limited:
        hit = _names(note, [name]) and re.search(
            rf"(?<![\w-]){re.escape(name)}(?![\w-])\s*(?<![\w])(?:{_LOW}|{_HIGH})\s*{_NUM}", note, re.I)
        if hit:
            return f"it sets a new limit on {name} (the study's is " + next(
                str(c.get("limit")) for c in block.get("constraints") or [] if str(c.get("quantity")) == name) + ")"
    for m in _matches(_RANGE_CHANGE, note):
        if _names(note[max(0, m.start() - 40):m.end() + 40], variables):
            return "it asks to change the range a design variable may take"
    return ""


def _baseline_value(best: dict[str, Any] | None) -> float | None:
    value = ((best or {}).get("baseline") or {}).get("objective")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _best_value(best: dict[str, Any] | None) -> float | None:
    value = ((best or {}).get("best") or {}).get("objective")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _checked_value(best: dict[str, Any] | None) -> float | None:
    """The best design's objective at the finest settings of FI's check, when it was checked."""
    value = ((best or {}).get("check") or {}).get("objective")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def read_request(note: str, block: dict[str, Any] | None, best: dict[str, Any] | None) -> dict[str, Any]:
    """What a refine ``note`` asks of a search for the best design (``block``: the plan's optimisation block,
    normalised; ``best``: ``results/best_design.json``): ``{}`` for an ordinary refine, else ``{"kind": SEARCH |
    NEW_STUDY | UNCLEAR, "says": one plain sentence, "round": {...} (SEARCH only)}``. Only a number tied to the
    objective (its unit after it, its name or meaning next to it, or "than the baseline") is read as a target."""
    note = str(note or "").strip()
    if not note or not isinstance(block, dict) or not isinstance(block.get("objective"), dict):
        return {}
    objective = block["objective"]
    quantity = str(objective.get("quantity") or "the objective")
    unit = str(objective.get("unit") or "")
    u = f" {unit}" if unit else ""
    sign = _sign(block)
    why_new = _new_study(note, block)
    if why_new:
        return {"kind": NEW_STUDY, "says": (
            f"Your refine was not carried out: {why_new}. That is a new study, not a continuation of this one (this "
            f"study makes {quantity} as {'low' if sign > 0 else 'high'} as possible, within its limits and ranges). "
            "Nothing was changed. To run it, start a new quest that asks for it; to continue this one, refine with "
            "\"push further\" or a target for " + quantity + ".")}
    budget = _BUDGET.search(note)
    added = next((int(g) for g in (budget.groups() if budget else ()) if g), None)
    target: float | None = None
    read_as = ""
    # An improvement is measured from the baseline where it is reported: at the finest settings of FI's check when
    # there is one, else at the search's settings.
    checked_base = ((best or {}).get("check") or {}).get("baseline_objective")
    base_where = "at the finest settings" if isinstance(checked_base, (int, float)) and not isinstance(
        checked_base, bool) else "at the search's settings"
    baseline = float(checked_base) if base_where == "at the finest settings" else _baseline_value(best)
    improve = next((m for p in (_IMPROVE_BY, _BY_X_BETTER) for m in p.finditer(note) if _tied(note, m, block)), None)
    compare = next((m for m in _COMPARE.finditer(note) if _tied(note, m, block)), None)
    if improve is not None or (compare is not None and _BETTER.match(compare.group("rest") or "")):
        m = improve or compare
        amount = _float(m.group(3 if "op" in m.re.groupindex else 1))
        if amount is None or baseline is None:
            return {"kind": UNCLEAR, "says": (
                "Your refine was not carried out: it asks for an improvement over the baseline, but the baseline has no "
                "value in FI's record of the search, so the target cannot be worked out. Give the target as a value of "
                f"{quantity} instead (for example \"at most ...{u}\"). Nothing was changed.")}
        amount, percent, _written, refusal = _with_unit(amount, m.group("rest") or "", unit, quantity, sign, True)
        if refusal:
            return {"kind": UNCLEAR, "says": f"Your refine was not carried out: {refusal}. Nothing was changed."}
        step = abs(amount) / 100.0 * abs(baseline) if percent else abs(amount)
        target = baseline - sign * step
        said = f"{_fmt(abs(amount))}%" if percent else f"{_fmt(abs(amount))}{u}"
        read_as = (f"{quantity} {'at most' if sign > 0 else 'at least'} {_fmt(target)}{u} ({said} better than the "
                   f"baseline's {_fmt(baseline)}{u} {base_where})")
    elif compare is not None:
        value = _float(compare.group(3))
        op = (compare.group("op") or compare.group("sym")).lower()
        low = re.fullmatch(_LOW, op, re.I) is not None
        high = re.fullmatch(_HIGH, op, re.I) is not None
        # The unit written with the number decides what it means: a percent is relative to the best design so far, a
        # unit of its own is converted to the objective's (or refused), a bare number is in the objective's unit.
        if value is None:
            compare = None
            written, percent, refusal = "", False, ""
        else:
            value, percent, written, refusal = _with_unit(value, compare.group("rest") or "", unit, quantity, sign)
        if refusal:
            return {"kind": UNCLEAR, "says": f"Your refine was not carried out: {refusal}. Nothing was changed."}
        if value is not None:
            checked_now, current_now = _checked_value(best), _best_value(best)
            ref_now = checked_now if checked_now is not None else current_now
            ref_where = "at the finest settings" if checked_now is not None else "at the search's settings"
            converted_from = f" (written as {compare.group(3)} {written})" if written and written != _sym(unit) \
                and not percent and not _is_percent_unit(unit) else ""
            if percent and ((low and sign > 0) or (high and sign < 0) or not (low or high)):
                if ref_now is None:
                    return {"kind": UNCLEAR, "says": (
                        f"Your refine was not carried out: \"{compare.group(0).strip()}\" is a percent of the best "
                        f"design, but the search has no recorded value of {quantity} to take it from. Give the target "
                        f"as a value of {quantity}{' in ' + unit if unit else ''} instead. Nothing was changed.")}
                target = ref_now - sign * abs(value) / 100.0 * abs(ref_now)
                read_as = (f"{quantity} {'at most' if sign > 0 else 'at least'} {_fmt(target)}{u} ({_fmt(abs(value))}% "
                           f"{'below' if sign > 0 else 'above'} the current best {_fmt(ref_now)}{u} {ref_where})")
            elif (low and sign > 0) or (high and sign < 0) or not (low or high):
                target = value
                read_as = f"{quantity} {'at most' if sign > 0 else 'at least'} {_fmt(target)}{u}{converted_from}"
            elif high and sign > 0 and baseline is not None:
                # "at least 6 K" of a quantity made as low as possible: an improvement of at least that much.
                step = abs(value) / 100.0 * abs(baseline) if percent else abs(value)
                target = baseline - step
                said = f"{_fmt(abs(value))}%" if percent else f"{_fmt(abs(value))}{u}"
                read_as = (f"{quantity} at most {_fmt(target)}{u} (at least {said} better than the "
                           f"baseline's {_fmt(baseline)}{u} {base_where})")
            else:
                return {"kind": UNCLEAR, "says": (
                    f"Your refine was not carried out: FI could not tell what \"{compare.group(0).strip()}\" asks of "
                    f"{quantity}, which this study makes as {'low' if sign > 0 else 'high'} as possible. Say \"at "
                    f"{'most' if sign > 0 else 'least'} X{u}\" for a value to reach, or \"X{u} better than the baseline\" "
                    "for an improvement. Nothing was changed.")}
    if target is None and added is None and not _PUSH.search(note):
        return {}
    # Whether the target is already reached is judged where it is reported: at the finest settings of FI's check when
    # the best design was checked (a coarse search can look better than it is), else at the search's settings.
    checked, current = _checked_value(best), _best_value(best)
    reference = checked if checked is not None else current
    if target is not None and reference is not None and sign * (reference - target) <= 0:
        where = "at the finest settings of FI's check" if checked is not None else "at the search's settings"
        return {"kind": UNCLEAR, "says": (
            f"Your refine was not carried out: the best design already reaches {read_as} ({quantity} = "
            f"{_fmt(reference)}{u} {where}). Ask for a stricter target, or say \"push further\" to search on without "
            "one. Nothing was changed.")}
    per_start = int(((block.get("evaluation_budget") or {}).get("per_start")) or 1)
    added = added or per_start
    round_: dict[str, Any] = {"added": int(added), "target": target, "asked": note[:300], "read_as": read_as}
    same_design = ((best or {}).get("check") or {}).get("design") == ((best or {}).get("best") or {}).get("design")
    if target is not None and checked is not None and current is not None and same_design:
        # The search compares designs at its own settings: its target is moved by how far the best design's value there
        # is from its value at the finest settings, so a design that reaches it there is likely to reach the person's
        # target at the finest settings (which the check then says).
        round_["search_target"] = target - (checked - current)
    plain = (f"FI searches further from the best design so far, with {added} more evaluation(s)"
             + (f", toward {read_as}" if read_as else "") + ", then checks the result at finer settings again. This "
             "refine is answered by the search alone: send any change to the paper's text as a refine of its own.")
    return {"kind": SEARCH, "says": plain, "round": round_}


def add_round(quest_root: Path, block: dict[str, Any], round_: dict[str, Any]) -> list[dict[str, Any]]:
    """Add ``round_`` to the rounds of the search kept for this plan (a file kept for another plan is started over).
    ``round_["refine"]`` (which of the person's refines asked for it) makes this idempotent: a review step run again
    for the same refine (a resume) does not add the round twice."""
    path = Path(quest_root) / _optimise.ROUNDS_PATH
    rounds = _optimise.read_rounds(quest_root, block)
    if round_.get("refine") is not None and any(
            r.get("refine") == round_["refine"] and r.get("asked") == round_.get("asked") for r in rounds):
        return rounds
    rounds.append({k: round_.get(k) for k in ("added", "target", "search_target", "asked", "read_as", "refine")
                   if k in ("added", "target", "asked", "read_as") or round_.get(k) is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"block_sha256": _optimise.block_sha(block), "rounds": rounds}, indent=1) + "\n",
                    encoding="utf-8")
    return rounds


def hint(block: dict[str, Any] | None) -> str:
    """The line the review pause shows for a search for the best design: how to ask for more, in the person's words."""
    if not isinstance(block, dict) or not isinstance(block.get("objective"), dict):
        return ""
    objective = block["objective"]
    quantity = str(objective.get("quantity") or "the objective")
    u = f" {objective['unit']}" if objective.get("unit") else ""
    sign = _sign(block)
    per_start = int(((block.get("evaluation_budget") or {}).get("per_start")) or 1)
    return (f"Not satisfied with the best design? Refine with \"push further\", a target (\"{quantity} "
            f"{'at most' if sign > 0 else 'at least'} X{u}\"), or an improvement (\"X{u} better than the baseline\"): FI "
            f"searches further from the best design so far ({per_start} more evaluations unless you say how many, for "
            "example \"100 more evaluations\"), checks it again at finer settings, and says whether the target was "
            "reached. Changing what is optimised, its limits or its ranges is a new study.")
