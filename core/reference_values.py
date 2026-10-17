"""The value a measured quantity should take at each setting, computed by FI itself from the frozen plan.

A paper that compares each result with what it should be is checkable at a glance; one that prints a column of
"reference" values the writer supplied is not, because nothing says where they came from. A real quest's paper printed
``pi0 x 0.05`` beside its BH false discovery rates; the numbers were right, but they came from the script's own
``pi0_ref`` field through the analysis, so no check ever looked at them. Here the plan states the reference once, as a
formula, and FI computes it per setting:

* **Where it is written.** An optional ``reference`` on a ``protocol.metrics`` entry (core/metric_spec.py)::

      "reference": {"value": "Ea_hat", "formula": "Ea_true_J_per_mol", "kind": "value",
                    "when": {"pi1": 0.0}, "checked_by": "noise_free_Ea", "note": "the true activation energy"}

  ``formula`` is in the calculator language of :mod:`core.oracle_forms` (``expected_formula``), with the special
  functions; its names are the setting's grid values, then the plan's fixed settings, then the numbers of
  ``protocol.thresholds`` (:func:`inputs`). Nothing else: a number the simulation computes (a theory SE written by the
  script, a ``pi0_ref`` field) cannot be named, so a reference can never be the script's own word presented as theory.
  ``value`` names the value ``run_trial`` returns that the reference is for (default: the metric's id, when the trials
  return a value of that name). ``kind`` is ``value`` (the number it should equal), ``bound`` (an upper limit) or
  ``lower_bound``. ``when`` is one filter or a list of them (a grid setting and its value): the settings the formula
  holds for; any other setting has no reference. ``checked_by`` names the known-answer check whose case pins the formula.
* **When it is checked.** At plan time (:func:`problems`, called by core/plan.py): the formula must compute on every
  grid setting it covers from those names alone, and agree with the expected value of each known-answer check of the
  same quantity at that check's case (:func:`cross_check`); a reference that fails either is left out of the plan, with
  the reason. It is part of the frozen protocol, so it cannot change once the run has started.
* **What it gives.** :func:`compute`: per setting, the reference or "no reference" with the reason, and per quantity
  whether a check FI ran confirms it: a check that passed and agrees at its case ("agrees with the known-answer check"),
  none ("not cross-checked"), or one that did not pass (no confirmation). :func:`copies_measured` drops a reference that
  evaluates to the measured means more closely than their uncertainty allows, to the precision its own numbers are
  written to (a bare number, an expression of one, a plan's number holding one): a value copied from the run.

The comparison itself (``agrees`` / ``differs`` / ``within bound``), and the tables, are core/results_by_setting.py.
No model is called here; FI stays domain-general: the formulas are the plan's, the arithmetic is FI's.
"""

from __future__ import annotations

import itertools
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from . import oracle_check as _oracle
from . import oracle_forms as _forms

#: The kinds of reference: the value it should equal, an upper limit, a lower limit.
KINDS = ("value", "bound", "lower_bound")
_KIND_WORDS = {"value": "value", "exact": "value", "equal": "value", "bound": "bound", "upper_bound": "bound",
               "upper": "bound", "upper limit": "bound", "lower_bound": "lower_bound", "lower": "lower_bound",
               "lower limit": "lower_bound"}
#: The most grid settings a plan-time check of a formula computes it on.
MAX_CHECKED_SETTINGS = 5000

#: How the plan writes a reference (one field of a metric entry), for the plan's prompt.
PLAN_FIELD = (
    '"reference": {"value": "<the name run_trial returns this number under, when it is not the id>", '
    '"formula": "<what the number should be, as one formula FI computes itself from the grid settings, the fixed '
    'settings and the thresholds by name>", "kind": "<value | bound (an upper limit it must not exceed) | lower_bound>", '
    '"when": {"<grid setting>": <the value it holds for; leave out when it holds for every setting>}, '
    '"checked_by": "<the name of the oracle whose case gives this formula a known answer>", '
    '"note": "<where the formula comes from, in plain words>"}'
)
#: The rule that goes with it.
PLAN_RULE = (
    "Give a metric a `reference` only when what it should be is a formula of the grid settings, the fixed settings or "
    "the thresholds (a nominal coverage, a true parameter, an exact error rate or power); leave it out otherwise, and "
    "never write a number the simulation computes into it. FI computes the formula for every setting and prints it "
    "beside each result, so a quantity with no reference reads \"no reference\"."
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _filters(when: Any) -> list[dict[str, Any]] | None:
    """``when`` as a list of filters, ``[]`` for none, ``None`` when it is not written as filters."""
    if when is None:
        return []
    items = when if isinstance(when, list) else [when]
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not item:
            return None
        if not all(str(k).strip() and (_num(v) is not None or (isinstance(v, str) and v.strip()))
                   for k, v in item.items()):
            return None
        out.append({str(k).strip(): v for k, v in item.items()})
    return out


def shape_problem(ref: Any) -> str | None:
    """Why ``ref`` is not written as a reference (the words after "its `reference`"), or ``None``."""
    if not isinstance(ref, dict):
        return "must be a mapping with a `formula` (and optionally `value`, `kind`, `when`, `checked_by`, `note`)"
    formula = ref.get("formula")
    if isinstance(formula, bool) or not isinstance(formula, (str, int, float)) or not str(formula).strip():
        return "needs a `formula`: what the number should be, as one formula of the settings, fixed settings and thresholds"
    kind = ref.get("kind")
    if kind is not None and str(kind).strip().lower() not in _KIND_WORDS:
        return "`kind` must be `value`, `bound` (an upper limit) or `lower_bound`"
    for key in ("value", "checked_by", "note"):
        if ref.get(key) is not None and not isinstance(ref[key], str):
            return f"`{key}` must be text"
    if _filters(ref.get("when")) is None:
        return "`when` must be a setting and its value (or a list of them), such as {\"pi1\": 0}"
    return None


def normalized(ref: dict[str, Any]) -> dict[str, Any]:
    """``ref`` with its ``kind`` in the canonical word and ``when`` as a list of filters (it must pass
    :func:`shape_problem`)."""
    out = {k: v for k, v in ref.items() if v is not None}
    out["formula"] = str(ref["formula"]).strip()
    out["kind"] = _KIND_WORDS[str(ref.get("kind") or "value").strip().lower()]
    filters = _filters(ref.get("when")) or []
    if filters:
        out["when"] = filters
    else:
        out.pop("when", None)
    for key in ("value", "checked_by", "note"):
        if isinstance(out.get(key), str):
            out[key] = out[key].strip()
            if not out[key]:
                del out[key]
    return out


def inputs(protocol: dict[str, Any] | None) -> dict[str, float]:
    """The names a reference formula may use besides the setting's own grid values: the plan's fixed settings
    (:func:`core.oracle_forms.fixed_settings`), over the numbers of ``protocol.thresholds``. (A known-answer check's
    formula does not read the thresholds, which are pass or fail limits; a reference does, because a plan states its
    nominal level and its true parameters there.)"""
    found: dict[str, float] = {}
    thresholds = protocol.get("thresholds") if isinstance(protocol, dict) else None
    for name, value in (thresholds.items() if isinstance(thresholds, dict) else []):
        number = _num(value)
        if number is not None and str(name).strip():
            found[str(name).strip()] = number
    found.update(_forms.fixed_settings(protocol))
    return found


def grid_axes(protocol: dict[str, Any] | None) -> dict[str, list[Any]]:
    grid = protocol.get("grid") if isinstance(protocol, dict) else None
    return {str(k): list(v) for k, v in grid.items() if isinstance(v, list) and v} if isinstance(grid, dict) else {}


def grid_settings(protocol: dict[str, Any] | None, limit: int = MAX_CHECKED_SETTINGS) -> list[dict[str, Any]]:
    """The grid's settings (at most ``limit``), as the trial runner lists them (the first axis slowest)."""
    axes = grid_axes(protocol)
    if not axes:
        return [{}]
    return [dict(zip(axes, combo)) for combo in itertools.islice(itertools.product(*axes.values()), limit)]


def _same(a: Any, b: Any) -> bool:
    x, y = _num(a), _num(b)
    if x is not None and y is not None:
        return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)
    return isinstance(a, str) and isinstance(b, str) and a.strip().lower() == b.strip().lower()


def applies(ref: dict[str, Any], setting: dict[str, Any]) -> bool:
    """Whether the reference holds at ``setting`` (its ``when``, if any, matches)."""
    filters = _filters(ref.get("when")) or []
    return not filters or any(all(k in setting and _same(setting[k], v) for k, v in f.items()) for f in filters)


def value_at(ref: dict[str, Any], setting: dict[str, Any], given: dict[str, float]) -> tuple[float | None, str]:
    """``(the reference at setting, "")`` or ``(None, why not)``. ``given``: :func:`inputs` of the plan."""
    value, why, missing, _used = _forms._formula_result({"expected_formula": ref.get("formula"), "case": setting}, given)
    if value is not None:
        return value, ""
    if missing:
        return None, (f"its formula uses {', '.join(missing)}, which is not a setting of the grid, a fixed setting or "
                      "a threshold of the plan")
    return None, f"its formula cannot be computed here ({why})"


def _oracles(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    items = protocol.get("oracles") if isinstance(protocol, dict) else None
    return [o for o in items if isinstance(o, dict) and str(o.get("name") or "").strip()] if isinstance(items, list) else []


def value_name(metric: dict[str, Any]) -> str:
    """The name of the value the trials return that the metric's reference is for."""
    ref = metric.get("reference")
    named = ref.get("value") if isinstance(ref, dict) else None
    return str(named or "").strip() or str(metric.get("id") or "").strip()


def cross_check(protocol: dict[str, Any] | None, metric: dict[str, Any],
                judged: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """What the plan's known-answer checks say about a metric's reference: ``{"state", "check", "why"}``.

    The checks it is compared with: the one ``checked_by`` names, else every check whose ``measure`` is the value's name
    or the metric's id. A check is comparable when its measure is that one name (a formula such as ``max(0, x - 0.04)``
    measures something else) and its case is one the reference holds for. Then the reference computed at the check's case
    must equal the check's expected value within its tolerance: ``disagrees`` otherwise (the reference is not used). With
    ``judged`` (the engine's verdicts, :func:`core.oracle_check.last_judged`): ``agrees`` only through a check the engine
    measured and passed; one that did not pass confirms nothing (``not_confirmed``). Without it (plan time), agreeing with
    the expected value is ``agrees``. ``unchecked`` when no check can be compared."""
    ref = normalized(metric["reference"])
    ident = str(metric.get("id") or "").strip()
    name = value_name(metric)
    oracles = _oracles(protocol)
    named = str(ref.get("checked_by") or "").strip().lower()
    if named:
        chosen = [o for o in oracles if str(o["name"]).strip().lower() == named]
        if not chosen:
            return {"state": "unchecked", "check": ref.get("checked_by"),
                    "why": f"it names the check {ref.get('checked_by')!r}, which the plan does not have"}
    else:
        chosen = [o for o in oracles if str(o.get("measure") or "").strip() in {name, ident}]
    given = inputs(protocol)
    passed = {str(n).strip().lower() for n in _oracle.engine_passed(judged)} if judged is not None else None
    judged_names = {str(j.get("name") or "").strip().lower() for j in judged or []}
    agreed: list[str] = []
    not_passed: list[str] = []
    why_not: list[str] = []
    for o in chosen:
        oname = str(o["name"]).strip()
        if passed is not None and oname.lower() in judged_names and oname.lower() not in passed:
            not_passed.append(oname)
            continue
        if not _forms.is_name(o.get("measure")):
            why_not.append(f"the check {oname!r} measures `{o.get('measure')}`, not the value itself")
            continue
        raw_case = o.get("case")
        case: dict[str, Any] = dict(raw_case) if isinstance(raw_case, dict) else {}
        if not applies(ref, case):
            why_not.append(f"the check {oname!r} runs at a setting the reference does not cover")
            continue
        value, why = value_at(ref, case, given)
        expected, limit, _mode = _oracle.limit_of(o)
        if value is None or expected is None or limit is None:
            why_not.append(f"the check {oname!r} cannot be compared ({why or 'it has no numeric expected value and tolerance'})")
            continue
        if abs(value - expected) > limit:
            return {"state": "disagrees", "check": oname,
                    "why": (f"at the case of the check {oname!r} ({_case_words(case)}) the formula gives {_fmt(value)}, "
                            f"where the check expects {_fmt(expected)} within {_fmt(limit)}")}
        agreed.append(oname)
    if agreed:
        confirmed = [n for n in agreed if passed is None or n.lower() in passed]
        if confirmed:
            done = "which FI ran and which passed" if passed is not None else "at the check's case"
            return {"state": "agrees", "check": confirmed[0],
                    "why": f"it agrees with the known-answer check {confirmed[0]!r} ({done})"}
        return {"state": "not_confirmed", "check": agreed[0],
                "why": f"it agrees with the expected value of the check {agreed[0]!r}, which FI has no passed run of"}
    if not_passed:
        return {"state": "not_confirmed", "check": not_passed[0],
                "why": f"its known-answer check {not_passed[0]!r} did not pass, so it confirms nothing"}
    return {"state": "unchecked", "check": None,
            "why": "; ".join(why_not) if why_not else "no known-answer check of the plan measures this value"}


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _case_words(case: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {v}" for k, v in case.items()) or "no settings"


def problems(protocol: dict[str, Any] | None) -> list[str]:
    """Why each reference of the plan cannot be used, one sentence each, in the words :mod:`core.plan` refuses a
    protocol entry with (``"`protocol.metrics` entry `id`: its `reference` ..."``); empty when every reference is
    usable. Checked: the shape, the ``when`` filters against the grid, the formula computed on every grid setting it
    covers (at most :data:`MAX_CHECKED_SETTINGS`) from the setting, the fixed settings and the thresholds alone, and its
    agreement with the known-answer checks of the same quantity (:func:`cross_check`)."""
    metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
    out: list[str] = []
    axes = grid_axes(protocol)
    settings = grid_settings(protocol)
    given = inputs(protocol)
    for m in metrics if isinstance(metrics, list) else []:
        if not isinstance(m, dict) or m.get("reference") is None:
            continue
        head = f"`protocol.metrics` entry `{str(m.get('id') or '').strip()}`: its `reference`"
        shape = shape_problem(m["reference"])
        if shape:
            out.append(f"{head} {shape}")
            continue
        ref = normalized(m["reference"])
        bad_filter = next((k for f in ref.get("when") or [] for k in f if k not in axes), None)
        if bad_filter is not None:
            out.append(f"{head} holds `when` {bad_filter!r}, which is not a setting of `protocol.grid`")
            continue
        covered = [s for s in settings if applies(ref, s)]
        if not covered:
            out.append(f"{head} holds for no setting of the grid (its `when` matches none)")
            continue
        for s in covered:
            value, why = value_at(ref, s, given)
            if value is None:
                out.append(f"{head} cannot be computed at {_case_words(s)}: {why}")
                break
        else:
            agreement = cross_check(protocol, {**m, "reference": ref})
            if agreement["state"] == "disagrees":
                out.append(f"{head} disagrees with a known-answer check: {agreement['why']}")
    return out


_ENTRY = re.compile(r"`protocol\.metrics` entry `([^`]*)`: its `reference`")


def without_bad_reference(protocol: dict[str, Any], why: str | None) -> tuple[dict[str, Any] | None, str]:
    """``(protocol with that metric's reference left out, a note)`` when ``why`` refuses a metric's reference; else
    ``(None, "")``. One reference that cannot be used is dropped, not the metric (and not the metrics)."""
    match = _ENTRY.search(why or "")
    metrics = protocol.get("metrics")
    if not match or not isinstance(metrics, list):
        return None, ""
    ident = match.group(1)
    fixed: list[Any] = []
    changed = False
    for m in metrics:
        if isinstance(m, dict) and str(m.get("id") or "").strip() == ident and "reference" in m:
            fixed.append({k: v for k, v in m.items() if k != "reference"})
            changed = True
        else:
            fixed.append(m)
    if not changed:
        return None, ""
    reason = (why or "").split(": its `reference` ", 1)[-1]
    return ({**protocol, "metrics": fixed},
            f"the reference value of `{ident}` was left out of the plan ({reason}); that quantity reads \"no reference\"")


_LITERAL = re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _finest(formula: str, given: dict[str, float]) -> float | None:
    """The finest relative precision a reference's numbers are written to: its decimal literals and the decimal values
    of the plan's names it uses (``0.17748`` is written to 5e-6 / 0.17748). Whole numbers count as exact. ``None``
    when it writes none."""
    used = _forms.names(formula, {**_forms.FUNCTIONS, **_forms.SPECIAL_FUNCTIONS})
    texts = _LITERAL.findall(formula) + [repr(float(v)) for k, v in given.items()
                                         if k in used and isinstance(v, (int, float)) and math.isfinite(v)
                                         and not float(v).is_integer()]
    shares: list[float] = []
    for text in texts:
        try:
            number = Decimal(text)
        except (InvalidOperation, ValueError):
            continue
        exponent = number.as_tuple().exponent
        if number.is_finite() and number != 0 and isinstance(exponent, int):
            shares.append(0.5 * 10.0 ** exponent / abs(float(number)))
    return min(shares) if shares else None


def copies_measured(ref: dict[str, Any], measured: list[tuple[float, float | None, float]],
                    given: dict[str, float] | None = None) -> bool:
    """Whether a reference reproduces the measured means more closely than they are known, so it was read off the run,
    not worked out: in more than half of the settings it covers (``measured``: ``(mean, 95% half-width, the reference
    FI computed there)`` of each), the mean is the reference to the precision the reference's numbers are written to
    (:func:`_finest`), and that precision is finer than a tenth of the half-width. A bare number (``0.1775``), an
    expression of one (``0.17748 + 0*m``) and a plan's number holding one (``p_seen`` = 0.17748) are all caught; a
    formula of the settings whose numbers are short (``1 - (1 - 0.05/m)**m``), or a nominal 0.95, is not."""
    text = str(ref.get("formula") or "").strip()
    share = _finest(text, given or {}) if text else None
    if share is None or not measured:
        return False
    hits = 0
    for mean, half, value in measured:
        tol = share * abs(value)
        if half is not None and half > 0 and tol < half / 10 and abs(mean - value) <= tol:
            hits += 1
    return hits * 2 > len(measured)


def compute(protocol: dict[str, Any] | None, settings: list[dict[str, Any]], trial_names: set[str] | None = None,
            judged: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The references of the plan's metrics at each of ``settings`` (each a mapping of setting names to values):
    ``{"quantities": [...], "by_setting": [...]}``.

    Each quantity: ``metric``, ``value`` (the trial value it is for), ``kind``, ``formula``, ``note``, ``source``
    ("formula in the frozen plan"), ``checked`` (:func:`cross_check`), and ``status`` ("computed" or "no reference")
    with ``why``. Each entry of ``by_setting``: ``metric``, ``value``, ``setting``, ``reference`` (a number or ``None``)
    and ``why`` (empty when there is one). A metric without a reference is listed with status "no reference"; an old
    plan without any gives every quantity "no reference"."""
    metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
    given = inputs(protocol)
    quantities: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for m in metrics if isinstance(metrics, list) else []:
        if not isinstance(m, dict) or not str(m.get("id") or "").strip():
            continue
        ident = str(m["id"]).strip()
        raw = m.get("reference")
        name = value_name(m)
        if trial_names is not None and name not in trial_names and raw is not None and ident in trial_names:
            # The plan names the value its reference is for by a name the trials never return (a "<id>_ref" it made up),
            # but the trials do return the metric itself: the reference is for that, as `value` defaults to the id. A
            # reference for a value no trial returns stays "no reference".
            name = ident
            if isinstance(raw, dict):
                raw = {**raw, "value": ident}
        if trial_names is not None and name not in trial_names:
            missing: dict[str, Any] = {"metric": ident, "value": None, "status": "no reference",
                                       "why": f"the trials return no value named {name!r}"}
            if raw is None:
                missing["why"] = "the trials return no value of this name (it is computed from several values)"
            quantities.append(missing)
            continue
        if raw is None:
            quantities.append({"metric": ident, "value": name, "status": "no reference",
                               "why": ""})
            continue
        shape = shape_problem(raw)
        if shape:
            quantities.append({"metric": ident, "value": name, "status": "no reference", "why": f"its reference {shape}"})
            continue
        ref = normalized(raw)
        checked = cross_check(protocol, {**m, "reference": ref}, judged)
        entry: dict[str, Any] = {"metric": ident, "value": name, "kind": ref["kind"], "formula": ref["formula"],
                 "note": ref.get("note", ""), "source": "formula in the frozen plan", "checked": checked,
                 "status": "computed", "why": ""}
        if checked["state"] == "disagrees":
            entry.update(status="no reference", why=f"it disagrees with a known-answer check: {checked['why']}")
            quantities.append(entry)
            continue
        quantities.append(entry)
        for s in settings:
            if not applies(ref, s):
                rows.append({"metric": ident, "value": name, "setting": dict(s), "reference": None,
                             "why": "the reference holds for other settings"})
                continue
            value, why = value_at(ref, s, given)
            rows.append({"metric": ident, "value": name, "setting": dict(s), "reference": value, "why": why})
    return {"quantities": quantities, "by_setting": rows}
