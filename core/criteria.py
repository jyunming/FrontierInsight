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
  A run of the simulation that FI makes itself is an oracle with a ``case`` (:func:`core.oracle_check.case_of`): its
  expected value and where that value comes from are the protocol's, so a number the simulation computes about itself
  (an ``error`` against a reference it holds) is never the yardstick. A criterion has no case of its own.
* ``trials``: a number every trial returns (a key of ``run_trial``'s dict), read from FI's own record of the trials; the
  value is the rate at which the standard error of its mean shrinks with the number of trials (batch means pooled over
  the settings: 0.5 for independent trials, less when trials repeat or depend on each other). It needs at least
  :data:`MIN_TRIALS` trials in settings of :data:`MIN_PER_SETTING` or more.

Each has a ``direction`` (``lower`` or ``higher`` is better, or ``target``: closest to ``target`` is best), an optional
``target`` (for ``lower`` the most it may be, for ``higher`` the least, for ``target`` the value aimed at) and its own
``tolerance``: for ``target``, how close counts as met; for every criterion, the change a later version may show before
it counts as worse. A number the script measured itself (an oracle with no case, answered by the script's own
``oracle()``) is shown and never counted.

The criteria live inside the protocol, so they are frozen with it, covered by its hash and changed only by an amendment
(:mod:`core.frozen_protocol`). After every run the engine computes each one and appends a row to
``.fi/criteria_history.jsonl``: the protocol's run and version, the commit of ``code/`` that ran, and each criterion's
value, whether it was met and whether it counts. The improve loop (:mod:`core.improve`) decides from them: it changes
the simulation one step at a time and keeps a version only when no criterion got worse beyond its own tolerance.
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
#: The trials (over all settings) the standard-error rate needs before it is steady enough to judge by: at 256, 1 run in
#: 250 of independent trials lands more than 0.15 from 0.5; at 64 it is 1 in 6.
MIN_TRIALS = 256
#: Trials one setting needs to count towards it: three batch sizes of at least 8 batches each.
MIN_PER_SETTING = 32
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


def _number_or_text(value: Any) -> float | None:
    """A number, or text that is one: YAML reads ``1e-8`` (no point before the ``e``) as text."""
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    return _num(value)


def _key(name: Any) -> str:
    """A name as the headline check compares it: lower case, without spaces, dashes or underscores."""
    return re.sub(r"[\s_\-]+", "", str(name or "").lower())


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def headline_names(protocol: dict[str, Any] | None) -> set[str]:
    """The study's headline numbers (as :func:`_key` writes them): the ids of ``protocol.metrics`` and the metric the
    precision target names."""
    if not isinstance(protocol, dict):
        return set()
    names = set()
    metrics = protocol.get("metrics")
    for item in metrics if isinstance(metrics, list) else [metrics] if isinstance(metrics, dict) else []:
        if isinstance(item, dict) and str(item.get("id") or "").strip():
            names.add(_key(item["id"]))
    precision = protocol.get("precision")
    if isinstance(precision, dict) and str(precision.get("metric") or "").strip():
        names.add(_key(precision["metric"]))
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
                      "itself: a declared check against a known answer (`oracle`) or FI's record of the trials (`trials`)")
    if item.get("case") not in (None, "", {}) or item.get("measure") not in (None, ""):
        return None, (f"{label} names a run of its own (`case`, `measure`); a run FI makes is declared as a check against "
                      "a known answer (an oracle with that `case`, `measure`, `expected` value and `reference`) and the "
                      "criterion names that check in `oracle`, so the number it is judged against is the protocol's, "
                      "never one the simulation computes about itself")
    sources = [k for k in ("oracle", "trials") if item.get(k) not in (None, "", {})]
    if not sources:
        return None, (f"{label} does not say what FI measures: name a declared check against a known answer (`oracle`) "
                      "or a number every trial returns (`trials`)")
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
    else:
        trials = item.get("trials")
        if not isinstance(trials, str) or not trials.strip():
            return None, f"{label}: `trials` must be the name of a number every trial returns"
        measured = trials.strip()
        out.update(trials=measured)
    # A criterion that names a headline number, written in any case or with spaces, dashes or underscores. (A per-trial
    # number that a headline is only built on, `final_size` under `mean_final_size`, is not refused: a trials criterion
    # measures how its standard error shrinks, not the finding.)
    for word in (measured, name):
        if word and _key(word) in headline:
            return None, (f"{label} is about {word!r}, a headline number of the study (`protocol.metrics`): judging the "
                          "code by the study's own finding would reward bending the code towards it. Use a check of "
                          "correctness instead (a check against a known answer, a convergence order, an invariant)")
    direction = _DIRECTION_WORDS.get(re.sub(r"\s+", " ", str(item.get("direction") or "").strip().lower()))
    if direction is None:
        return None, f"{label}: `direction` must be `lower`, `higher` or `target` (closest to `target` is best)"
    out["direction"] = direction
    target = item.get("target")
    if target is not None and _number_or_text(target) is None:
        return None, f"{label}: `target` must be a number (write 1.0e-6, not 1e-6)"
    if direction == "target" and target is None:
        return None, f"{label}: a `target` direction needs the number aimed at in `target`"
    if target is not None:
        out["target"] = _number_or_text(target) if isinstance(target, str) else target
    tolerance = _number_or_text(item.get("tolerance"))
    if tolerance is None or tolerance < 0:
        return None, (f"{label} needs its own `tolerance`, a number of 0 or more (write 1.0e-8, not 1e-8): how close to "
                      "the target counts as met, and how much a later version may change before it counts as worse")
    out["tolerance"] = tolerance if isinstance(item.get("tolerance"), str) else item["tolerance"]
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
            notes.append(f"{why}; it was left out of the plan and is not used")
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
    else:
        what = (f"how fast the standard error of {criterion['trials']} shrinks as trials are added "
                f"(0.5 when the trials are independent; needs {MIN_TRIALS} trials or more, in settings of "
                f"{MIN_PER_SETTING} or more)")
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


def countable(protocol: dict[str, Any] | None, *, fi_runs: bool = True) -> list[dict[str, Any]]:
    """The protocol's criteria whose number FI can measure itself (not one resting on a check the script answers).
    ``fi_runs``: whether FI runs the simulation itself (two scripts, the trial contract); when it does not, it measures
    none of them."""
    if not fi_runs:
        return []
    return [c for c in declared(protocol) if not _script_measured(protocol, c)]


def plan_notes(protocol: dict[str, Any] | None, *, fi_runs: bool = True) -> list[str]:
    """What *Checks already made* says about the criteria: none, only one, or one the script measures itself."""
    if not isinstance(protocol, dict):
        return []
    items = declared(protocol)
    if not fi_runs:
        return ["This quest runs its experiment as one script, so FI runs none of its checks itself: whatever criteria "
                "the plan names are shown after each run but none counts, and no run can be shown to be better than "
                "another. Two scripts (the default, `execution.split_analysis: auto`; not `false`) let FI "
                "run the simulation and measure them."]
    if not countable(protocol):
        return ["The plan has no criterion FI can measure itself for judging whether the code got better (`criteria` in "
                "the protocol, each naming a check against a known answer that FI runs, or a number every trial "
                "returns): no run can be shown to be better than another."]
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


def se_rate(cells: list[list[float]] | list[float]) -> float | None:
    """How fast the standard error of a mean shrinks as trials are added, from ``cells`` (one list of values per setting,
    each in trial order; a setting with fewer than :data:`MIN_PER_SETTING` is left out): the slope of log(spread of batch means) against log(batch size), sign flipped, with the spread
    pooled over the settings (each around its own mean), every point made of at least 8 batches, corrected for the bias
    of a log of a spread, and weighted by its degrees of freedom. 0.5 for independent trials; lower when trials repeat or
    depend on each other. ``None`` with fewer than :data:`MIN_TRIALS` values in all (in those settings), or no spread."""
    if cells and not isinstance(cells[0], list):
        cells = [cells]  # type: ignore[list-item]
    series = [[v for v in (_num(x) for x in xs) if v is not None] for xs in cells]  # type: ignore[union-attr]
    series = [xs for xs in series if len(xs) >= MIN_PER_SETTING]
    if sum(len(xs) for xs in series) < MIN_TRIALS:
        return None
    points: list[tuple[float, float, int]] = []
    size = 1
    while True:
        squares, dof = 0.0, 0
        for xs in series:
            count = len(xs) // size
            if count < 8:
                continue
            means = [statistics.fmean(xs[i * size:(i + 1) * size]) for i in range(count)]
            centre = statistics.fmean(means)
            squares += sum((m - centre) ** 2 for m in means)
            dof += count - 1
        if dof < 7:
            break
        if squares > 0:
            points.append((math.log(size), 0.5 * math.log(squares / dof) + 1 / (2 * dof), dof))
        size *= 2
    if len(points) < 3:
        return None
    weight = sum(p[2] for p in points)
    mx = sum(p[0] * p[2] for p in points) / weight
    my = sum(p[1] * p[2] for p in points) / weight
    den = sum(p[2] * (p[0] - mx) ** 2 for p in points)
    return -sum(p[2] * (p[0] - mx) * (p[1] - my) for p in points) / den if den else None


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


def evaluate(items: list[dict[str, Any]], *, judged: list[dict[str, Any]], series: dict[str, dict[str, list[float]]],
             why_missing: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Each criterion's value after a run, from what FI measured: ``judged`` (the oracle gate's verdicts, as in
    ``needs/ORACLE_CHECK.json``) and ``series`` (a trial number's values per setting, in trial order, from FI's record).
    ``why_missing`` says, per source (``oracle``, ``trials``), why nothing could be measured."""
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
                if value is not None:
                    measured_by = "FI" if j.get("measured_by") == "engine" else "the script"
        else:
            cells = list((series.get(c["trials"]) or {}).values())
            value = se_rate(cells)
            if value is None and not why_missing.get("trials"):
                usable = sum(len(xs) for xs in cells if len(xs) >= MIN_PER_SETTING)
                why = (f"FI's record has {sum(len(xs) for xs in cells)} trial(s) of {c['trials']!r}, {usable} of them in "
                       f"settings of {MIN_PER_SETTING} or more; {MIN_TRIALS} in such settings are needed, and they must vary")
            else:
                why = "" if value is not None else why_missing["trials"]
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
           code_changed: bool | None = None, protocol_version: int | None = None, protocol_sha256: str | None = None,
           attempt: str | None = None, protocol_problem: str | None = None,
           improve: dict[str, Any] | None = None, result_digest: dict[str, str] | None = None) -> dict[str, Any]:
    """Append one row to ``.fi/criteria_history.jsonl`` and return it. ``n`` counts the rows (one per run of the code);
    ``run`` is the frozen protocol's run (``run_1`` until an amendment), so rows are compared only within one
    ``protocol_sha256``. ``attempt`` is the id of this run's record in ``.fi/attempts.jsonl``. ``results`` empty is
    recorded as "no criterion". ``improve``: for a round of the improve loop (:mod:`core.improve`), the round, whether
    its version was kept as the best, and the criteria that got better or worse against the version kept before it.
    ``result_digest``: for a full run that produced results, one hash per result (``improve.headline_digest``), so a
    later run can say whether the study's results changed (``core/changelog.py``) without keeping their values here."""
    row: dict[str, Any] = {
        "n": len(history(quest_root)) + 1, "run": run, "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256, "attempt": attempt,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": code_commit, "criteria": results,
    }
    if code_changed is not None:
        row["code_changed_since_commit"] = code_changed
    if protocol_problem:
        row["protocol_problem"] = protocol_problem
    if improve:
        row["improve"] = improve
    if result_digest:
        row["result_digest"] = dict(result_digest)
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
    lead = (f"{len(met)} of {len(counted)} checks of correctness that FI measured itself are met; " if counted else
            "none of the checks of correctness was measured by FI itself this run, so none counts; ")
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
    return lead + "; ".join(parts)
