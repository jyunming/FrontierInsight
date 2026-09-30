"""How a quest judges whether its code got better: two to five checks of correctness, fixed before the first full run.

A later step will change a quest's code and keep a change only when it made nothing worse. That needs a yardstick fixed
before any change is made, computed by FI and never read from the code's own report, and it must measure whether the code
is *right*, never whether the study's finding came out as hoped: a yardstick built on the headline number would reward
bending the code towards the answer. So the plan's protocol declares ``criteria``, each one of:

* ``oracle``: a check against a known answer the protocol already declares (``protocol.oracles``), by its name. ``use:
  error`` (the default) is how far its measured value is from the value it expects, ``|value - expected|`` (for a
  convergence-rate check that is the gap between the observed and the claimed order; for an invariant, the worst
  violation); ``use: value`` is the measured value itself. No second way of measuring: the number is the one the oracle
  gate recorded in ``needs/ORACLE_CHECK.json``.
* ``case`` + ``measure``: one run of the simulation on ``case`` that FI makes itself (the trial contract's
  ``run_trial``/``run_cell``, as for an oracle with a case), and the number ``measure`` that run returns.
* ``trials``: a number every trial returns (a key of ``run_trial``'s dict), read from FI's own record of the trials; the
  value is the rate at which its standard error shrinks with the number of trials (batch means: 0.5 for independent
  trials, less when trials repeat or depend on each other).

Each has a ``direction`` (``lower`` or ``higher`` is better, or ``target``: closest to ``target`` is best), an optional
``target`` (for ``lower`` the most it may be, for ``higher`` the least, for ``target`` the value aimed at) and its own
``tolerance``: for ``target``, how close counts as met; for every criterion, the change a later version may show before
it counts as worse. A number the script measured itself (an oracle with no case, answered by the script's own
``oracle()``) is shown and never counted.

The criteria live inside the protocol, so they are frozen with it, covered by its hash and changed only by an amendment
(:mod:`core.frozen_protocol`). After every run the engine computes each one and appends a row to
``.fi/criteria_history.jsonl``: the run, the commit of ``code/`` that ran, and each criterion's value and whether it was
met. Nothing is decided from the rows yet.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIN, MAX = 2, 5
HISTORY = ".fi/criteria_history.jsonl"

#: The three directions, and how the plan says each.
DIRECTIONS = {"lower": "lower is better", "higher": "higher is better", "target": "closest to a target is best"}
_DIRECTION_WORDS = {
    "lower": "lower", "lower is better": "lower", "lower_is_better": "lower", "min": "lower", "minimise": "lower",
    "minimize": "lower", "smaller": "lower", "smaller is better": "lower", "decrease": "lower", "less": "lower",
    "higher": "higher", "higher is better": "higher", "higher_is_better": "higher", "max": "higher", "maximise": "higher",
    "maximize": "higher", "larger": "higher", "larger is better": "higher", "increase": "higher", "more": "higher",
    "target": "target", "within tolerance of a target": "target", "within_tolerance": "target", "closest": "target",
    "close to target": "target", "equal": "target", "match": "target",
}
_USES = ("error", "value")
# Keys that would take the number from the script's own report: refused, never read.
_SCRIPT_KEYS = ("result", "result_json", "results", "from_results", "path")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def headline_names(protocol: dict[str, Any] | None) -> set[str]:
    """The study's headline numbers (lower case): the ids of ``protocol.metrics`` and the metric the precision target names."""
    if not isinstance(protocol, dict):
        return set()
    names = set()
    metrics = protocol.get("metrics")
    for item in metrics if isinstance(metrics, list) else [metrics] if isinstance(metrics, dict) else []:
        if isinstance(item, dict) and str(item.get("id") or "").strip():
            names.add(str(item["id"]).strip().lower())
    precision = protocol.get("precision")
    if isinstance(precision, dict) and str(precision.get("metric") or "").strip():
        names.add(str(precision["metric"]).strip().lower())
    return names


def _oracle_names(protocol: dict[str, Any] | None) -> dict[str, str]:
    from . import oracle_check

    return {str(o["name"]).strip().lower(): str(o["name"]).strip() for o in oracle_check.declared(protocol)}


def _one(item: Any, index: int, protocol: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """One criterion in its clean form, or ``(None, why)``."""
    if not isinstance(item, dict):
        return None, f"`protocol.criteria` entry {index} is not a mapping (`name`, what FI measures, `direction`, `tolerance`)"
    name = str(item.get("name") or "").strip()
    if not name:
        return None, f"`protocol.criteria` entry {index} has no `name`"
    label = f"`protocol.criteria` entry {name!r}"
    if any(item.get(k) not in (None, "") for k in _SCRIPT_KEYS):
        return None, (f"{label} takes its number from the script's own results; a criterion must be one FI computes "
                      "itself: a declared check (`oracle`), a run FI makes (`case` and `measure`) or FI's record of the "
                      "trials (`trials`)")
    sources = [k for k in ("oracle", "case", "trials") if item.get(k) not in (None, "", {})]
    if not sources:
        return None, (f"{label} does not say what FI measures: name a declared check (`oracle`), a run FI makes "
                      "(`case` and `measure`) or a number every trial returns (`trials`)")
    if len(sources) > 1:
        return None, f"{label} names {' and '.join(f'`{s}`' for s in sources)}; give only one of them"
    out: dict[str, Any] = {"name": name}
    if isinstance(item.get("what"), str) and item["what"].strip():
        out["what"] = " ".join(item["what"].split())
    headline = headline_names(protocol)
    source = sources[0]
    measured = ""
    if source == "oracle":
        known = _oracle_names(protocol)
        ref = str(item.get("oracle") or "").strip()
        if ref.lower() not in known:
            return None, (f"{label} names no declared check ({ref!r}); the protocol's checks are: "
                          f"{', '.join(known.values()) or 'none'}")
        use = str(item.get("use") or "error").strip().lower()
        if use not in _USES:
            return None, f"{label}: `use` must be `error` (how far from the expected value) or `value` (the value itself)"
        out.update(oracle=known[ref.lower()], use=use)
    elif source == "case":
        if not isinstance(item.get("case"), dict):
            return None, f"{label}: `case` must be the settings of one run, such as {{dt: 0.05}}"
        measure = item.get("measure")
        if not isinstance(measure, str) or not measure.strip():
            return None, f"{label} names a `case` but no `measure` (the number that run returns)"
        measured = measure.strip()
        out.update(case=dict(item["case"]), measure=measured)
    else:
        trials = item.get("trials")
        if not isinstance(trials, str) or not trials.strip():
            return None, f"{label}: `trials` must be the name of a number every trial returns"
        measured = trials.strip()
        out.update(trials=measured)
    for word in (measured, name):
        if word and word.lower() in headline:
            return None, (f"{label} is about {word!r}, a headline number of the study (`protocol.metrics`): judging the "
                          "code by the study's own finding would reward bending the code towards it. Use a check of "
                          "correctness instead (a check against a known answer, a convergence order, an invariant)")
    direction = _DIRECTION_WORDS.get(re.sub(r"\s+", " ", str(item.get("direction") or "").strip().lower()))
    if direction is None:
        return None, f"{label}: `direction` must be `lower`, `higher` or `target` (closest to `target` is best)"
    out["direction"] = direction
    target = item.get("target")
    if target is not None and _num(target) is None:
        return None, f"{label}: `target` must be a number"
    if direction == "target" and target is None:
        return None, f"{label}: a `target` direction needs the number aimed at in `target`"
    if target is not None:
        out["target"] = target
    tolerance = _num(item.get("tolerance"))
    if tolerance is None or tolerance < 0:
        return None, (f"{label} needs its own `tolerance`, a number of 0 or more: how close to the target counts as met, "
                      "and how much a later version may change before it counts as worse")
    out["tolerance"] = item["tolerance"]
    return out, None


def normalize(items: Any, protocol: dict[str, Any] | None) -> tuple[list[dict[str, Any]] | None, str | None]:
    """``(criteria, None)`` when every criterion can be computed by FI, else ``(None, why)``. Strict, for a plan a person
    wrote. ``protocol`` is the protocol they belong to (its checks and headline numbers)."""
    if items is None or items == "" or items == []:
        return [], None
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return None, "`protocol.criteria` must be a list (each with a `name`, what FI measures, a `direction` and a `tolerance`)"
    if len(items) > MAX:
        return None, f"`protocol.criteria` lists {len(items)}; give at most {MAX}, the checks of correctness that matter most"
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        fixed, why = _one(item, index, protocol)
        if fixed is None:
            return None, why
        if any(c["name"].lower() == fixed["name"].lower() for c in out):
            return None, f"`protocol.criteria` names {fixed['name']!r} twice"
        out.append(fixed)
    return out, None


def repair(items: Any, protocol: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[str]]:
    """``(the criteria that can be used, a sentence per one left out)``: for a plan the model drafted, so one criterion that
    cannot be computed costs itself, not the others."""
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return [], [(normalize(items, protocol)[1] or "`protocol.criteria` is not a list") + "; it was left out of the plan"]
    kept: list[dict[str, Any]] = []
    notes: list[str] = []
    for index, item in enumerate(items, start=1):
        fixed, why = _one(item, index, protocol)
        if fixed is None:
            notes.append(f"{why}; it was left out of the plan, put it right there if it matters")
        elif any(c["name"].lower() == fixed["name"].lower() for c in kept):
            notes.append(f"`protocol.criteria` names {fixed['name']!r} twice; the second was left out of the plan")
        elif len(kept) >= MAX:
            notes.append(f"`protocol.criteria` may list at most {MAX}; {fixed['name']!r} was left out of the plan")
        else:
            kept.append(fixed)
    return kept, notes


def declared(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The protocol's criteria that can be computed (an unusable one is left out here; the plan has said so)."""
    items = protocol.get("criteria") if isinstance(protocol, dict) else None
    return repair(items, protocol)[0] if items not in (None, "", []) else []


def _oracle_of(protocol: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    from . import oracle_check

    return next((o for o in oracle_check.declared(protocol) if str(o["name"]).strip() == name), None)


def _script_measured(protocol: dict[str, Any] | None, criterion: dict[str, Any]) -> bool:
    """Whether the criterion's oracle is answered by the script's own ``oracle()`` (it names no case FI can run)."""
    from . import oracle_check

    oracle = _oracle_of(protocol, criterion.get("oracle", "")) if criterion.get("oracle") else None
    return oracle is not None and oracle_check.case_of(oracle) is None


def describe(criterion: dict[str, Any], protocol: dict[str, Any] | None = None) -> str:
    """One criterion in a sentence a domain scientist reads without the source."""
    name = criterion["name"]
    if criterion.get("oracle"):
        what = (f"how far the check “{criterion['oracle']}” lands from the value it expects"
                if criterion.get("use") == "error" else f"the value the check “{criterion['oracle']}” measures")
        if _script_measured(protocol, criterion):
            what += " (measured by the script's own code, so shown but not counted: give that check a case FI can run)"
    elif criterion.get("case") is not None:
        settings = ", ".join(f"{k}={_fmt(v)}" for k, v in criterion["case"].items())
        what = f"{criterion['measure']} from one run FI makes itself at {settings or 'the default settings'}"
    else:
        what = (f"how fast the standard error of {criterion['trials']} shrinks as trials are added "
                "(0.5 when the trials are independent)")
    direction, target, tol = criterion["direction"], criterion.get("target"), criterion["tolerance"]
    if direction == "target":
        bar = f"closest to {_fmt(target)} is best; met within {_fmt(tol)} of {_fmt(target)}"
    else:
        bar = DIRECTIONS[direction]
        if target is not None:
            bar += f"; met at {_fmt(target)} or {'below' if direction == 'lower' else 'above'}"
        else:
            bar += "; no bar is set, only whether it gets better or worse"
    lead = f"**{name}**: " + (f"{criterion['what']}; " if criterion.get("what") else "")
    return f"{lead}{what}; {bar}; a change smaller than {_fmt(tol)} counts as no change."


def plan_notes(protocol: dict[str, Any] | None) -> list[str]:
    """What *Checks already made* says about the criteria: none, only one, or one the script measures itself."""
    if not isinstance(protocol, dict):
        return []
    items = declared(protocol)
    if not items:
        return ["The plan has no criterion for judging whether the code got better (`criteria` in the protocol): every "
                "run is recorded as having no criterion, and a later change to the code cannot be shown to be better."]
    notes = []
    if len(items) < MIN:
        notes.append(f"The plan has only one criterion for judging whether the code got better; {MIN} to {MAX} "
                     "independent checks of correctness make a change harder to misjudge.")
    for c in items:
        if _script_measured(protocol, c):
            notes.append(f"The criterion {c['name']!r} rests on the check {c['oracle']!r}, which is measured by the script's "
                         "own code (it names no case FI can run), so its number is shown but not counted.")
    return notes


# --- computing them after a run ------------------------------------------------------------------------------------


def se_rate(values: list[float]) -> float | None:
    """How fast the standard error of the mean of ``values`` (in trial order) shrinks as trials are added: the slope of
    log(spread of batch means) against log(batch size), sign flipped. 0.5 for independent trials; lower when trials repeat
    or depend on each other. ``None`` with fewer than 16 values or no spread."""
    xs = [v for v in (_num(x) for x in values) if v is not None]
    if len(xs) < 16:
        return None
    points: list[tuple[float, float]] = []
    size = 1
    while len(xs) // size >= 4:
        count = len(xs) // size
        means = [statistics.fmean(xs[i * size:(i + 1) * size]) for i in range(count)]
        spread = statistics.stdev(means)
        if spread > 0:
            points.append((math.log(size), math.log(spread)))
        size *= 2
    if len(points) < 3:
        return None
    mx = statistics.fmean(p[0] for p in points)
    my = statistics.fmean(p[1] for p in points)
    den = sum((p[0] - mx) ** 2 for p in points)
    return -sum((p[0] - mx) * (p[1] - my) for p in points) / den if den else None


def meets(criterion: dict[str, Any], value: float | None) -> bool | None:
    """Whether ``value`` meets the criterion (``None``: nothing measured, or no bar set)."""
    if value is None:
        return None
    target = _num(criterion.get("target"))
    tol = _num(criterion.get("tolerance")) or 0.0
    if criterion["direction"] == "target":
        return abs(value - target) <= tol if target is not None else None
    if target is None:
        return None
    return value <= target if criterion["direction"] == "lower" else value >= target


def evaluate(items: list[dict[str, Any]], *, judged: list[dict[str, Any]], case_values: dict[str, float],
             series: dict[str, dict[str, list[float]]], why_missing: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Each criterion's value after a run, from what FI measured: ``judged`` (the oracle gate's verdicts, as in
    ``needs/ORACLE_CHECK.json``), ``case_values`` (criterion name -> the number FI's own run of its case returned) and
    ``series`` (a trial number's values per setting, in trial order, from FI's record). ``why_missing`` says, per source
    (``oracle``, ``case``, ``trials``), why nothing could be measured."""
    why_missing = why_missing or {}
    by_name = {str(j.get("name")).strip(): j for j in judged if isinstance(j, dict)}
    out = []
    for c in items:
        value: float | None = None
        measured_by: str | None = None
        why = ""
        if c.get("oracle"):
            j = by_name.get(c["oracle"])
            v, e = (_num(j.get("value")), _num(j.get("expected"))) if j else (None, None)
            if v is None:
                why = why_missing.get("oracle") or f"the check {c['oracle']!r} gave no number this run"
            else:
                value = v if c.get("use") == "value" else (abs(v - e) if e is not None else None)
                why = "" if value is not None else f"the check {c['oracle']!r} has no expected value"
                measured_by = "FI" if j.get("measured_by") == "engine" else "the script"
        elif c.get("case") is not None:
            value = _num(case_values.get(c["name"]))
            why = "" if value is not None else (why_missing.get("case") or f"FI's run of the case did not return {c['measure']!r}")
            measured_by = "FI" if value is not None else None
        else:
            rates = [r for r in (se_rate(vs) for vs in (series.get(c["trials"]) or {}).values()) if r is not None]
            value = statistics.median(rates) if rates else None
            why = "" if value is not None else (why_missing.get("trials") or
                                               f"FI's record has too few trials of {c['trials']!r} (16 per setting are needed)")
            measured_by = "FI" if value is not None else None
        row = {"name": c["name"], "value": value, "met": meets(c, value), "counts": measured_by == "FI",
               "measured_by": measured_by, "direction": c["direction"], "target": c.get("target"),
               "tolerance": c["tolerance"]}
        if why:
            row["why"] = why
        out.append(row)
    return out


def history_path(quest_root: Path) -> Path:
    return Path(quest_root) / HISTORY


def history(quest_root: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = history_path(quest_root).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def record(quest_root: Path, *, run: str, code_commit: str | None, results: list[dict[str, Any]],
           code_changed: bool | None = None) -> dict[str, Any]:
    """Append one row to ``.fi/criteria_history.jsonl`` and return it. ``results`` empty is recorded as "no criterion"."""
    row: dict[str, Any] = {
        "n": len(history(quest_root)) + 1, "run": run,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": code_commit, "criteria": results,
    }
    if code_changed is not None:
        row["code_changed_since_commit"] = code_changed
    if not results:
        row["note"] = "no criterion"
    try:
        path = history_path(quest_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str, allow_nan=False) + "\n")
    except (OSError, ValueError):
        pass  # a record that cannot be written must never stop a quest
    return row


def summary_line(results: list[dict[str, Any]]) -> str:
    """One plain line for run.log."""
    if not results:
        return ("no criterion: the plan names no check of correctness to judge the code by, so this run cannot be "
                "compared with a later one")
    counted = [r for r in results if r.get("counts")]
    met = [r for r in counted if r.get("met") is True]
    parts = []
    for r in results:
        if r.get("value") is None:
            state = f"not measured ({r.get('why') or 'no number'})"
        else:
            state = {True: "met", False: "not met", None: "no bar set"}[r.get("met")]
            state = f"{_fmt(r['value'])}, {state}"
            if not r.get("counts"):
                state += ", measured by the script itself so not counted"
        parts.append(f"{r['name']}: {state}")
    return (f"{len(met)} of {len(counted)} checks of correctness that FI measured itself are met; "
            + "; ".join(parts))
