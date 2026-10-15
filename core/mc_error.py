"""A known-answer check on a random quantity is judged on enough trials for its tolerance.

A check against a known answer compares one number with an expected value within a tolerance. When the number is a
random quantity (a rate, a mean, a probability over repeats of a random simulation), what is measured carries the Monte
Carlo error of the trials it came from, and a tolerance smaller than that error fails a correct simulation by chance.
A real quest (false discovery rates, 2026-10-15) judged ``max(0, fdr_bh - 0.04)`` on ONE trial: that trial's false
discovery proportion was 1/17, so the check measured 0.0188 against a tolerance of 0.01, while an independent 400,000-repeat
simulation gives 0.0401 +- 0.00007, inside the bound. The simulation was right; three runs were thrown away and the paper
said the method might be broken.

This module holds the rules; :func:`core.trial_runner.measure_oracles` runs the trials and :func:`core.oracle_check.judged`
judges with them:

* **Which checks are read as a mean over trials** (:func:`reads_a_mean`). A check of a quantity (a special case, a
  published value, a convergence rate) is a number of the setting, which for a random study is its mean over trials. A
  rule every single trial keeps (an invariant, a symmetry, a second implementation: its worst violation, expecting 0) is
  read as a mean only when every name its ``measure`` uses is one the protocol declares among its ``metrics`` as a mean
  or proportion over runs: the plan itself says that number is an expected value, so a bound on it is a bound on an
  expected value. Otherwise it stays per trial, so a violation is never averaged away. ``per_trial: true`` always means
  one trial.
* **The number** is the check's formula applied to the means of the names over the trials (what the study reports for a
  setting), never the mean of the formula per trial: ``max(0, mean(fdr) - 0.04)``, not the mean of ``max(0, fdr - 0.04)``.
* **Its Monte Carlo error** (:func:`summarise`) is the standard error of the formula's inner quantity (inside an outer
  ``max(0, .)`` or ``abs(.)``, which only says which side counts), from the trials' own spread: the delta method with
  the trials' covariance, exact for a formula that is linear in the names. A bound check ``max(0, X - b)`` expecting 0 is
  therefore judged one-sided: the mean of ``X - b`` must not exceed the tolerance, and its band is that of ``X``.
* **How many trials** (:func:`trials_for`): FI runs a first batch, reads the spread, and runs enough more (within the time
  the check is given) for the standard error to be a quarter of the tolerance or less.
* **The tolerance it is judged at** (:func:`limit_used`): the plan's own when three standard errors fit inside it. When
  they do not (the time ran out first), the check is judged at three standard errors and the record says plainly that it
  was judged at that precision, never at the plan's. With fewer than :data:`MIN_TRIALS` trials the spread itself is not
  known well enough to widen anything: the plan's tolerance stands.
* **A failure within the error** (:func:`within_error`): a check outside its tolerance by less than three standard errors
  of the trials it was judged on still fails, but it is not decided by them: it is not taken as a fault of the
  simulation to repair, and the paper does not say the method may be broken.

A check on trials that return the same numbers every time is deterministic and judged as before, on trial 0's number.
A random simulation whose check quantity did not move in the first batch (a rare outcome that has not happened yet) is
sized from the spread a proportion at the expected value has (:func:`trials_for_constant`), never judged on trial 0.
Nothing here makes a check pass that is outside both its tolerance and its Monte Carlo error.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Collection
from typing import Any

from . import metric_spec as _metric_spec
from . import oracle_check as _oracle
from . import oracle_forms as _forms

#: How many standard errors the band around a mean is (about 99.7% two-sided for a normal mean). Two would fail a correct
#: simulation on one check in twenty, and a plan has several.
BAND_SE = 3.0
#: The standard error FI aims for, as a share of the tolerance: a quarter keeps the band (three of them) inside it.
TARGET_SHARE = 0.25
#: The first batch of trials, from which the spread is read.
PILOT_TRIALS = 20
#: Fewer trials than this give a spread too rough to widen a tolerance by.
MIN_TRIALS = 10
#: The most trials FI runs for one check.
MAX_TRIALS = 20000
#: The share of a check's time the first batch may use (the rest is for the trials the spread says are needed).
PILOT_SHARE = 1 / 3


def mean_names(protocol: dict[str, Any] | None) -> frozenset[str]:
    """The names the protocol declares as metrics that are a mean or a proportion over every run (``metrics``; a mean over
    a subset of the runs, ``given``, is not a plain mean and is left out)."""
    metrics = (protocol or {}).get("metrics") if isinstance(protocol, dict) else None
    if not metrics:
        return frozenset()
    kept, _dropped = _metric_spec.repair(metrics)
    return frozenset(str(m["id"]) for m in kept if m.get("kind") in _metric_spec.KINDS and not m.get("given"))


def _per_trial_flag(oracle: dict[str, Any]) -> bool:
    flag = oracle.get("per_trial")
    return flag is True or (isinstance(flag, str) and flag.strip().lower() in ("true", "yes", "1"))


def measure_names(measure: str, returned: dict[str, Any] | None = None) -> list[str]:
    """The names a check's ``measure`` reads from what the simulation returns: the measure itself when the simulation
    returns it under that exact name, else the formula's names."""
    if returned is not None and measure in returned:
        return [measure]
    return _forms.names(measure)


def reads_a_mean(oracle: dict[str, Any], means: Collection[str] = ()) -> bool:
    """Whether ``oracle``'s number is a mean over trials (see the module docstring): a check of a quantity (a special
    case, a published value, a convergence rate); a rule each trial keeps (an invariant, a symmetry, a second
    implementation: its worst violation, expecting 0) only when its measure uses nothing but names the protocol declares
    as means or proportions over runs. ``per_trial: true`` is never. (Two implementations whose counts differ by +1 in
    one trial and -1 in another agree on the mean: a rule's violation is never averaged away.)"""
    if _per_trial_flag(oracle):
        return False
    if _oracle.kind_of(oracle) not in _forms.VIOLATION_KINDS:
        return True
    measure = str(oracle.get("measure") or "").strip()
    used = _forms.names(measure) if measure else []
    return bool(used) and set(used) <= set(means)


def inner(measure: str) -> tuple[str, str]:
    """``(the quantity inside an outer max(0, .) or abs(.), "bound" | "abs" | "")``: the part of the formula whose
    error counts. ``max(0, X - b)`` (or ``max(X - b, 0)``) is a one-sided bound and ``abs(E)`` a distance; anything else
    is the formula itself."""
    try:
        tree = ast.parse(measure.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return measure, ""
    body = tree.body
    if not (isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and not body.keywords):
        return measure, ""

    def zero(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value == 0

    if body.func.id == "max" and len(body.args) == 2:
        a, b = body.args
        if zero(a) and not zero(b):
            return ast.unparse(b), "bound"
        if zero(b) and not zero(a):
            return ast.unparse(a), "bound"
    if body.func.id == "abs" and len(body.args) == 1:
        return ast.unparse(body.args[0]), "abs"
    return measure, ""


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _value_of(text: str, values: dict[str, float]) -> float | None:
    if text in values:
        return float(values[text])
    return _forms.evaluate(text, values).value


def summarise(rows: list[dict[str, Any]], measure: str) -> dict[str, Any] | None:
    """What the trials ``rows`` (each the values one trial returned) say about ``measure``: ``trials`` (those with a
    finite number for every name it uses), ``dropped`` (the others), ``means``, ``value`` (the formula on the means),
    ``inner`` and ``inner_mean`` (the quantity whose error counts, :func:`inner`), ``se`` (its standard error; ``None``
    with fewer than two trials or when it cannot be computed), ``varies`` (whether any name differs between trials;
    ``None`` with one trial) and ``problem`` when the formula gives no number on the means. ``None`` when no trial has
    the names."""
    measure = str(measure or "").strip()
    first = next((r for r in rows if isinstance(r, dict)), None)
    if first is None:
        return None
    used = measure_names(measure, first)
    if not used:
        return None
    kept = [r for r in rows if isinstance(r, dict) and all(_number(r.get(n)) for n in used)]
    n = len(kept)
    if n == 0:
        return None
    columns = {name: [float(r[name]) for r in kept] for name in used}
    means = {name: math.fsum(col) / n for name, col in columns.items()}
    inner_text, shape = inner(measure) if measure not in first else (measure, "")
    # Whether the quantity itself differs between trials (names that move together, as two implementations' counts
    # do, can leave it the same in every trial: then it is judged as a deterministic check is).
    varies: bool | None = None
    if n >= 2:
        varies = False
        reference = _value_of(inner_text, {name: col[0] for name, col in columns.items()})
        for i in range(1, n):
            other = _value_of(inner_text, {name: col[i] for name, col in columns.items()})
            if reference is None or other is None or not math.isclose(reference, other, rel_tol=1e-12, abs_tol=1e-15):
                varies = True
                break
    out: dict[str, Any] = {"trials": n, "dropped": len(rows) - n, "means": means, "inner": inner_text, "shape": shape,
                           "varies": varies, "se": None}
    got = _forms.evaluate(measure, means) if measure not in first else _forms.Evaluated(value=means[measure])
    out["value"] = got.value
    if got.value is None:
        out["problem"] = got.problem or "the formula gives no number on the trials' means"
    centre = _value_of(inner_text, means)
    out["inner_mean"] = centre
    if n < 2 or centre is None:
        return out
    if not varies:
        out["se"] = 0.0
        return out
    # The delta method: the gradient of the inner quantity at the means (central differences), and the trials'
    # covariance of the names. Exact for a formula linear in the names (one name: its standard deviation over sqrt(n)).
    grad: list[float] = []
    for name in used:
        sd = math.sqrt(math.fsum((x - means[name]) ** 2 for x in columns[name]) / (n - 1))
        h = max(1e-6 * abs(means[name]), 1e-3 * sd, 1e-12)
        up = _value_of(inner_text, {**means, name: means[name] + h})
        down = _value_of(inner_text, {**means, name: means[name] - h})
        if up is None or down is None:
            return out
        grad.append((up - down) / (2 * h))
    var = 0.0
    for a, name_a in enumerate(used):
        for b, name_b in enumerate(used):
            if grad[a] == 0 or grad[b] == 0:
                continue
            cov = math.fsum((x - means[name_a]) * (y - means[name_b])
                            for x, y in zip(columns[name_a], columns[name_b])) / (n - 1)
            var += grad[a] * grad[b] * cov
    out["se"] = math.sqrt(max(var, 0.0) / n)
    return out


def trials_for(summary: dict[str, Any], limit: float | None, *, most: int = MAX_TRIALS) -> int:
    """How many trials in all bring the standard error of ``summary`` to :data:`TARGET_SHARE` of ``limit`` (never
    fewer than it already has, never more than ``most``)."""
    n = int(summary.get("trials") or 0)
    se = summary.get("se")
    if not summary.get("varies") or not _number(se) or not se or limit is None or limit <= 0 or n < 2:
        return n
    wanted = n * (se / (TARGET_SHARE * limit)) ** 2
    return max(n, min(int(most), int(math.ceil(wanted))))


def returned_varies(rows: list[dict[str, Any]]) -> bool:
    """Whether any number the trials returned differs between them: the simulation is random. (A check's own quantity
    can still be the same in every trial of a first batch: a rare outcome that did not happen yet.)"""
    first = next((r for r in rows if isinstance(r, dict)), None)
    if first is None:
        return False
    for key, v in first.items():
        if not _number(v):
            continue
        for r in rows[1:]:
            other = r.get(key) if isinstance(r, dict) else None
            if _number(other) and not math.isclose(float(v), float(other), rel_tol=1e-12, abs_tol=1e-15):
                return True
    return False


def trials_for_constant(rows: list[dict[str, Any]], measure: str, expected: float | None, limit: float | None, *,
                        most: int = MAX_TRIALS) -> int:
    """How many trials in all a check needs whose quantity was the same in every trial of a random simulation (a rare
    outcome that has not happened yet: 20 trials of an event with probability 0.05 are all 0 more than a third of the
    time). For a quantity that stayed at 0 or 1 (a 0/1 outcome, or a count of a rare one), the spread a proportion has at
    the expected value (the rule of three's 3/n when the expected value is not a probability), sized as
    :func:`trials_for` sizes any other; for anything else, as many as the check's time allows (``most``)."""
    n = len(rows)
    summary = summarise(rows, measure)
    if summary is None or limit is None or limit <= 0:
        return n
    centre = summary.get("inner_mean")
    if centre not in (0.0, 1.0):
        return max(n, int(most))
    rule_of_three = min(0.5, 3.0 / max(n, 1))
    p = expected if expected is not None and 0.0 < expected < 1.0 else rule_of_three
    sd = math.sqrt(p * (1.0 - p))
    return max(n, min(int(most), int(math.ceil((sd / (TARGET_SHARE * limit)) ** 2))))


def limit_used(limit: float | None, mc: dict[str, Any] | None) -> tuple[float | None, str]:
    """``(the limit a check is judged at, how)``: ``"planned"`` (the plan's own; three standard errors fit inside it),
    ``"widened"`` (three standard errors, wider than the plan's, because the trials FI could run in the check's time do
    not resolve the plan's tolerance), ``"few_trials"`` (too few trials to know the spread: the plan's own), or ``""``
    for a check that is not a mean over trials whose trials differ."""
    if limit is None or not isinstance(mc, dict) or not mc.get("varies"):
        return limit, ""
    se, n = mc.get("se"), int(mc.get("trials") or 0)
    if not _number(se) or n < MIN_TRIALS:
        return limit, "few_trials"
    band = BAND_SE * float(se)
    if band <= limit:
        return limit, "planned"
    return band, "widened"


def within_error(value: float | None, expected: float | None, limit: float | None, mc: dict[str, Any] | None) -> bool:
    """Whether a check that fails at :func:`limit_used` fails by less than three standard errors of the trials it was
    judged on: the trials do not decide it (the true value may be inside the tolerance)."""
    if value is None or expected is None or limit is None or not isinstance(mc, dict) or not mc.get("varies"):
        return False
    used, _how = limit_used(limit, mc)
    se = mc.get("se")
    if used is None or not _number(se):
        return False
    gap = abs(value - expected)
    return used < gap <= limit + BAND_SE * float(se)


def words(mc: dict[str, Any] | None, limit: float | None) -> str:
    """How the check's number was had, in a few plain words for a record line: ``the mean over 310 trials, Monte Carlo
    error +-0.0024``; ``""`` for a check that is not a mean over trials whose trials differ."""
    if not isinstance(mc, dict) or not mc.get("varies"):
        return ""
    n, se = int(mc.get("trials") or 0), mc.get("se")
    text = f"the mean over {n} trial{'s' if n != 1 else ''}"
    if _number(se):
        text += f", Monte Carlo error ±{_oracle.fmt_digits(float(se), 2)} (one standard error)"
    used, how = limit_used(limit, mc)
    if how == "widened" and used is not None:
        text += (f"; judged at ±{_oracle.fmt_digits(used, 2)}, three standard errors, because the trials FI could run "
                 f"in the check's time do not resolve the plan's ±{_oracle.fmt_digits(limit, 2)}")
    elif how == "few_trials":
        text += "; too few trials to know their spread, so the plan's tolerance stands"
    return text


def record(summary: dict[str, Any], limit: float | None, *, read_as: str, planned_trials: int) -> dict[str, Any]:
    """The ``mc`` entry a check keeps (``needs/ORACLE_CHECK.json``): the trials, their error and the tolerance it was
    judged at."""
    used, how = limit_used(limit, summary)
    se = summary.get("se")
    return {
        "read_as": read_as, "trials": int(summary.get("trials") or 0), "dropped": int(summary.get("dropped") or 0),
        "varies": summary.get("varies"), "se": se, "band": BAND_SE * se if _number(se) else None,
        "inner": summary.get("inner"), "inner_mean": summary.get("inner_mean"), "shape": summary.get("shape") or "",
        "wanted_trials": int(planned_trials), "limit_used": used, "precision": how,
    }
