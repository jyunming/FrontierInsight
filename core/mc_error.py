"""A known-answer check on a random quantity is decided by enough trials, or it is not decided: never passed by chance.

A check against a known answer compares one number with an expected value within a tolerance. When the number is a
random quantity (a rate, a mean, a probability over repeats of a random simulation), what is measured carries the Monte
Carlo error of the trials it came from, and a tolerance smaller than that error fails a correct simulation by chance.
A real quest (false discovery rates, 2026-10-15) judged ``max(0, fdr_bh - 0.04)`` on ONE trial: that trial's false
discovery proportion was 1/17, so the check measured 0.0188 against a tolerance of 0.01, while an independent 400,000-repeat
simulation gives 0.0401 +- 0.00007, inside the bound. The simulation was right; three runs were thrown away and the paper
said the method might be broken.

This module holds the rules; :func:`core.trial_runner.measure_oracles` runs the trials and :func:`core.oracle_check.judged`
judges with them. The first rule is that a check the trials cannot decide is never a pass.

* **Which checks are read as a mean over trials** (:func:`reads_a_mean`). A check of a quantity (a special case, a
  published value, a convergence rate) is a number of the setting, which for a random study is its mean over trials;
  ``per_trial: true`` makes it a rule of every trial instead. A rule (an invariant, a symmetry, a second implementation:
  its violation, expecting 0) is read the same way: on the mean over trials unless the check says ``per_trial: true``,
  and then it is judged on every trial FI runs, the worst one deciding (:func:`worst`). The protocol's metrics never decide this: a number the plan declares as a
  mean is still a number each trial has.
* **The number** of a mean check is the check's formula applied to the means of the names over the trials
  (``max(0, mean(fdr) - 0.04)``) only when every name it uses is one the protocol declares among its ``metrics`` as a
  mean or a proportion over runs (the plan then says each is an expected value). Otherwise it is the mean of the formula
  taken trial by trial (:func:`per_trial_summary`): ``abs(x_num - x_exact)`` is a distance in each trial, and the two
  means can agree while every trial misses.
* **Its Monte Carlo error** (:func:`summarise`, :func:`per_trial_summary`) is the standard error of that number (for
  the formula on the means, of the quantity inside an outer ``max(0, .)`` or ``abs(.)``, from the trials' covariance).
* **The decision** (:func:`decide`): three standard errors around the measured value inside the tolerance is a pass;
  three standard errors outside it is a failure; anything else is not decided yet, and FI adds trials (sized from the
  spread, at most four times as many each step) until the trials decide it, the time the checks are given runs out, or
  :data:`MAX_TRIALS` is reached. Then it is NOT DECIDED: not a pass, not counted as validation, said in the paper. A
  tolerance is never widened, and a tolerance of 0 stays 0.
* **Failed trials count** (:func:`failed_share_verdict`): a trial that raised, or gave no number for the check, is not
  left out quietly. More than :data:`FAULT_SHARE` of the check's trials failing is a failure of the check (a fault, sent
  to repair); any failed trial at all makes a check that would pass not decided, since the trials that finished cannot
  stand for the case (a wrong mean whose largest draws crash would pass on the survivors).
* **A number that does not move** in a first batch of at least :data:`MIN_TRIALS` trials is judged at once on trial
  0's number, as a deterministic check is, so a constant quantity never runs to the time limit (fewer trials that
  agree show no constant, and leave it not decided).
* **A 0/1 outcome of a random simulation** (every trial gives 0 or 1; the plan runs more than one trial per setting,
  :func:`repeats_trials`, or, when the plan does not say, some number the simulation returns varies; and
  both values occurred, the check expects a rate strictly between 0 and 1, or the protocol declares the number a
  proportion and the tolerance is not exactly 0) is judged on its binomial interval, never on its sample spread (:func:`wilson`, :func:`binomial_verdict`), at every
  number of trials: ten zeros are not a rate of 0 (their interval at three standard errors reaches 0.47), so they do
  not decide a check that expects 0.05 within 0.01; 2000 zeros fail it.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

from . import metric_spec as _metric_spec
from . import oracle_check as _oracle
from . import oracle_forms as _forms
from . import trial_checks as _trial_checks

#: How many standard errors the band around a measured mean is (about 99.7% two-sided for a normal mean). Two would
#: decide a correct simulation wrongly on one check in twenty, and a plan has several.
BAND_SE = 3.0
#: The first batch of trials, from which the spread is read (and on which a rule of every trial is judged).
PILOT_TRIALS = 20
#: How many trials a rule of every trial is judged on when no study size is given: the first batch, then more up to this
#: many (a disagreement that shows only in a later trial must not pass on the first batch alone).
EVERY_TRIALS = 10 * PILOT_TRIALS
#: Fewer trials than this give a spread too rough to decide anything by.
MIN_TRIALS = 10
#: The most trials FI runs for one check.
MAX_TRIALS = 20000
#: The share of a check's time the first batch may use (the rest is for the trials the spread says are needed).
PILOT_SHARE = 1 / 3
#: How many times the trials may grow in one step (the spread of a first batch is rough).
GROWTH = 4
#: More than this share of a check's trials failing is a failure of the check: a fault in the simulation, repaired.
#: The same share as a setting of the run (core/trial_checks.py), so a check and the run judge failed trials alike.
FAULT_SHARE = _trial_checks.FAULT_FRACTION


def mean_names(protocol: dict[str, Any] | None) -> frozenset[str]:
    """The names the protocol declares as metrics that are a mean or a proportion over every run (``metrics``; a mean over
    a subset of the runs, ``given``, is not a plain mean and is left out)."""
    metrics = (protocol or {}).get("metrics") if isinstance(protocol, dict) else None
    if not metrics:
        return frozenset()
    kept, _dropped = _metric_spec.repair(metrics)
    return frozenset(str(m["id"]) for m in kept if m.get("kind") in _metric_spec.KINDS and not m.get("given"))


def proportion_names(protocol: dict[str, Any] | None) -> frozenset[str]:
    """The names the protocol declares as metrics of kind ``proportion`` over every run: a 0/1 outcome, by the record,
    judged on its binomial interval even when every trial gave its expected 0 or 1 (:func:`statistic`)."""
    metrics = (protocol or {}).get("metrics") if isinstance(protocol, dict) else None
    if not metrics:
        return frozenset()
    kept, _dropped = _metric_spec.repair(metrics)
    return frozenset(str(m["id"]) for m in kept if m.get("kind") == "proportion" and not m.get("given"))


def repeats_trials(protocol: dict[str, Any] | None) -> bool | None:
    """Whether the plan runs more than one trial per setting (``runs_per_setting`` or ``min_runs_per_setting`` above
    1): a random study. ``False`` for one run per setting (a deterministic study), ``None`` when the plan says neither."""
    if not isinstance(protocol, dict):
        return None
    runs = [protocol.get(k) for k in ("runs_per_setting", "min_runs_per_setting")]
    runs = [int(r) for r in runs if _number(r)]
    if not runs:
        return None
    return max(runs) > 1


def paired(protocol: dict[str, Any] | None, quest_root: Path | str | None = None) -> bool:
    """Whether trial ``t`` of every setting runs on one seed (common random numbers, :func:`core.trial_runner.trial_seed`
    with ``key=""``; a check's trials then get those seeds too): :func:`core.metric_spec.shares_draws`, the one decision
    the trial seeds and the analysis's paired contrast both follow. Text in the ``seed_policy`` never decides it."""
    return _metric_spec.shares_draws(protocol, quest_root)


def per_trial_flag(oracle: dict[str, Any]) -> bool | None:
    """What a check's ``per_trial`` says: ``True`` (a rule of every trial), ``False`` (a number on the average over
    trials), or ``None`` when it says nothing. The whole numbers 1 and 0 mean the same as true and false."""
    flag = oracle.get("per_trial")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, int) and flag in (0, 1):
        return bool(flag)
    if isinstance(flag, str):
        word = flag.strip().lower()
        if word in ("true", "yes", "1"):
            return True
        if word in ("false", "no", "0"):
            return False
    return None


def unsaid_reading(protocol: dict[str, Any] | None) -> list[str]:
    """The checks whose reading the plan has to say and has not: each rule (an invariant, a symmetry, a second
    implementation) whose ``measure`` uses a number the protocol declares as a mean or proportion over runs
    (:func:`mean_names`), with ``per_trial`` unset. Whether such a rule holds in every single trial or only on average is
    a fact about the science, so FI asks the plan once (the plan step's per-part request) and never guesses it; a check
    still unsaid is read on the mean over trials (:func:`reads_a_mean`). A check of a quantity is read on the mean already and
    is not asked about (``per_trial: true`` would only make a random quantity a rule of every trial)."""
    if not isinstance(protocol, dict):
        return []
    means = mean_names(protocol)
    out: list[str] = []
    for oracle in protocol.get("oracles") or []:
        if not isinstance(oracle, dict) or per_trial_flag(oracle) is not None or not str(oracle.get("name") or "").strip():
            continue
        if _oracle.kind_of(oracle) not in _forms.VIOLATION_KINDS:
            continue
        if set(_forms.names(str(oracle.get("measure") or ""))) & means:
            out.append(str(oracle["name"]).strip())
    return out


def reads_a_mean(oracle: dict[str, Any]) -> bool:
    """Whether ``oracle``'s number is a mean over trials (see the module docstring): every check unless it says
    ``per_trial: true`` (a rule of every single trial, the worst trial deciding). A check that says nothing is about the
    mean: a rule such as a false discovery rate at most q is a mean over trials, and a single trial's proportion can be
    anything, so reading it per trial fails correct code (the graded s1 plan, measured on the offline bench: reading the
    unset case as the mean caught 4 of 4 planted faults with 0 of 2 false stops, the per-trial reading 4 of 4 with 1 of 2).
    Nothing else decides it, not the check's kind."""
    return per_trial_flag(oracle) is not True


def measure_names(measure: str, returned: dict[str, Any] | None = None) -> list[str]:
    """The names a check's ``measure`` reads from what the simulation returns: the measure itself when the simulation
    returns it under that exact name, else the formula's names."""
    if returned is not None and measure in returned:
        return [measure]
    return _forms.names(measure)


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


def per_trial_values(rows: list[dict[str, Any]], measure: str) -> list[float | None]:
    """The check's formula on each trial's own numbers (``None`` for a trial that gives no finite number)."""
    out: list[float | None] = []
    for r in rows:
        if not isinstance(r, dict):
            out.append(None)
            continue
        got = float(r[measure]) if measure in r and _number(r[measure]) else (
            None if measure in r else _forms.evaluate(measure, {k: v for k, v in r.items() if _number(v)}).value)
        out.append(got if got is not None and math.isfinite(got) else None)
    return out


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def per_trial_summary(rows: list[dict[str, Any]], measure: str) -> dict[str, Any] | None:
    """What the trials ``rows`` say about the mean of ``measure`` taken trial by trial: ``trials`` (those with a finite
    number), ``dropped`` (the others), ``value`` (the mean), ``se`` (its standard error, ``None`` with fewer than two
    trials), ``varies`` (whether the number differs between trials; ``None`` with one trial), ``on: "trials"``. ``None``
    when no trial gives a number."""
    measure = str(measure or "").strip()
    values = per_trial_values(rows, measure)
    kept = [v for v in values if v is not None]
    n = len(kept)
    if n == 0:
        return None
    mean = math.fsum(kept) / n
    varies = None if n < 2 else any(not _same(kept[0], v) for v in kept[1:])
    se = None
    if n >= 2:
        se = 0.0 if not varies else math.sqrt(math.fsum((v - mean) ** 2 for v in kept) / (n - 1) / n)
    return {"on": "trials", "trials": n, "dropped": len(values) - n, "value": mean, "inner": measure, "shape": "",
            "inner_mean": mean, "varies": varies, "se": se, "first": kept[0]}


def summarise(rows: list[dict[str, Any]], measure: str) -> dict[str, Any] | None:
    """What the trials ``rows`` (each the values one trial returned) say about ``measure`` read as a formula of the
    names' means: ``trials`` (those with a finite number for every name it uses), ``dropped`` (the others), ``means``,
    ``value`` (the formula on the means), ``inner`` and ``inner_mean`` (the quantity whose error counts, :func:`inner`),
    ``se`` (its standard error; ``None`` with fewer than two trials or when it cannot be computed), ``varies`` (whether
    the quantity differs between trials; ``None`` with one trial), ``on: "means"`` and ``problem`` when the formula gives
    no number on the means. ``None`` when no trial has the names."""
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
    # do, can leave it the same in every trial).
    varies: bool | None = None
    if n >= 2:
        varies = False
        reference = _value_of(inner_text, {name: col[0] for name, col in columns.items()})
        for i in range(1, n):
            other = _value_of(inner_text, {name: col[i] for name, col in columns.items()})
            if reference is None or other is None or not _same(reference, other):
                varies = True
                break
    out: dict[str, Any] = {"on": "means", "trials": n, "dropped": len(rows) - n, "means": means, "inner": inner_text,
                           "shape": shape, "varies": varies, "se": None}
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


def statistic(rows: list[dict[str, Any]], measure: str, means: Any = (), expected: Any = None, limit: Any = None,
              proportions: Any = (), *, repeated: bool | None = None) -> dict[str, Any] | None:
    """The mean check's number and its error (see the module docstring): the formula on the names' means when every name
    it uses is a declared mean or proportion (``means``), else the mean of the formula trial by trial. A 0/1 outcome
    (``binary``; ``expected`` and ``limit`` are the check's) carries its Wilson ``interval`` and the standard error
    that interval stands for. ``proportions``: the names the protocol declares proportions (:func:`proportion_names`)."""
    measure = str(measure or "").strip()
    first = next((r for r in rows if isinstance(r, dict)), None)
    if first is None:
        return None
    used = measure_names(measure, first)
    stat = summarise(rows, measure) if used and set(used) <= set(means or ()) else per_trial_summary(rows, measure)
    if stat is None:
        return None
    # A 0/1 outcome of a random simulation: its uncertainty is its binomial interval, never its sample spread (ten
    # zeros have a spread of 0, which says nothing about a rate of 0.05). One rule at every number of trials. A 0/1
    # number is an outcome when both values occurred, when the check expects a rate strictly between 0 and 1, or when
    # the protocol declares it a proportion (then all zeros expected 0 within 0.01 are not a rate of 0, unless the
    # tolerance is exactly 0: an exact claim, read as a constant). One that is the same in every trial, expected to be
    # exactly that and not declared a proportion (a distance that is exactly 0) is a constant.
    # Whether the trials are random is the plan's to say (``repeated``: it runs more than one trial per setting,
    # :func:`repeats_trials`), not the returned numbers': a trial that returns only its 0/1 outcome, zero every time,
    # varies in nothing. Only when the plan does not say are the returned numbers read.
    values = _outcome_values(rows, stat)
    rate = expected is not None and _number(expected) and 0.0 < float(expected) < 1.0
    exact = _number(limit) and float(limit) == 0.0
    declared = str(stat.get("inner") or "") in {str(p) for p in proportions or ()} and not exact
    both = len(set(values)) > 1
    random = both or (repeated if repeated is not None else returned_varies(rows))
    if values and all(v in (0.0, 1.0) for v in values) and random and (rate or declared or both):
        lo, hi = wilson(int(sum(values)), len(values))
        stat.update(binary=True, successes=int(sum(values)), interval=[lo, hi], se=(hi - lo) / (2 * BAND_SE))
    return stat


def _outcome_values(rows: list[dict[str, Any]], stat: dict[str, Any]) -> list[float]:
    """The per-trial values of the quantity a mean check's ``stat`` is about: the formula trial by trial, or, on the
    names' means, the one name inside it (``[]`` for a formula of several names: its mean is not a proportion)."""
    inner_text = str(stat.get("inner") or "")
    if stat.get("on") == "trials":
        return [v for v in per_trial_values(rows, inner_text) if v is not None]
    if not _forms.is_name(inner_text):
        return []
    return [float(r[inner_text]) for r in rows if isinstance(r, dict) and _number(r.get(inner_text))]


def wilson(successes: int, trials: int, z: float = BAND_SE) -> tuple[float, float]:
    """The Wilson interval of a proportion, ``successes`` of ``trials``, at ``z`` standard errors (the checks' three):
    ten zeros give (0, 0.47)."""
    n = max(int(trials), 1)
    k = min(max(int(successes), 0), n)
    p = k / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (0.0 if k == 0 else max(0.0, centre - half)), (1.0 if k == n else min(1.0, centre + half))


def returned_varies(rows: list[dict[str, Any]]) -> bool:
    """Whether any number the trials returned differs between them: the simulation is random."""
    first = next((r for r in rows if isinstance(r, dict)), None)
    if first is None:
        return False
    for key, v in first.items():
        if not _number(v):
            continue
        for r in rows[1:]:
            other = r.get(key) if isinstance(r, dict) else None
            if _number(other) and not _same(float(v), float(other)):
                return True
    return False


def binomial_verdict(stat: dict[str, Any], expected: float, limit: float) -> str | None:
    """:func:`decide` for a 0/1 outcome (``stat["interval"]``, its Wilson interval): ``"pass"`` when the interval lies
    inside what the check accepts (``expected`` within ``limit``; for a bound or a distance expecting 0, at most
    ``limit``), ``"fail"`` when it lies wholly outside, else ``None``. The tolerance is never widened."""
    lo, hi = (float(x) for x in stat["interval"])
    if stat.get("on") == "means" and stat.get("shape") in ("bound", "abs") and expected == 0:
        low, high = -math.inf, float(limit)
    else:
        low, high = float(expected) - float(limit), float(expected) + float(limit)
    if lo >= low and hi <= high:
        return "pass"
    if hi < low or lo > high:
        return "fail"
    return None


def decide(centre: float, expected: float, limit: float, se: float, *, shape: str = "") -> str | None:
    """``"pass"`` when three standard errors around the measured ``centre`` lie inside the tolerance, ``"fail"`` when
    they lie outside it, ``None`` when the trials do not decide it. A bound ``max(0, X)`` expecting 0 is judged on its
    one side (``centre`` is X); ``abs(X)`` expecting 0 is a distance (``centre`` is X). The tolerance is never widened."""
    if shape == "bound" and expected == 0:
        gap = centre
    elif shape == "abs" and expected == 0:
        gap = abs(centre)
    else:
        gap = abs(centre - expected)
    band = BAND_SE * se
    if gap + band <= limit:
        return "pass"
    if gap - band > limit:
        return "fail"
    return None


def _centre(stat: dict[str, Any], expected: float) -> tuple[float | None, str]:
    """The number :func:`decide` is given for ``stat``, and its shape."""
    if stat.get("on") == "means" and stat.get("shape") in ("bound", "abs") and expected == 0:
        return stat.get("inner_mean"), str(stat["shape"])
    return stat.get("value"), ""


def mean_verdict(stat: dict[str, Any], expected: float | None, limit: float | None) -> str | None:
    """:func:`decide` for a mean check's ``stat`` (a 0/1 outcome of a random simulation: :func:`binomial_verdict`);
    ``None`` while it is not decided, and always with fewer than :data:`MIN_TRIALS` trials (moving or not)."""
    if expected is None or limit is None:
        return None
    centre, shape = _centre(stat, float(expected))
    se, n = stat.get("se"), int(stat.get("trials") or 0)
    if centre is None or not _number(se):
        return None
    if n < MIN_TRIALS:  # fewer trials than this show neither a spread nor a constant
        return None
    if stat.get("binary"):
        return binomial_verdict(stat, float(expected), float(limit))
    return decide(float(centre), float(expected), float(limit), float(se), shape=shape)


def trials_wanted(stat: dict[str, Any], expected: float, limit: float, *, most: int = MAX_TRIALS) -> int:
    """How many trials in all the next step runs: as many as the spread says would decide the check (with a margin),
    at least half as many again, at most :data:`GROWTH` times as many as there are, never more than ``most``. A 0/1
    outcome's spread is its binomial interval's (:func:`statistic`), so ten zeros still ask for more trials."""
    n = max(int(stat.get("trials") or 0), 1)
    se = float(stat.get("se") or 0.0)
    centre, shape = _centre(stat, expected)
    target = GROWTH * n
    if centre is not None and se > 0:
        gap = (centre if shape == "bound" else abs(centre) if shape == "abs" else abs(centre - expected))
        room = abs(limit - gap) / BAND_SE
        if room > 0:
            target = int(math.ceil(1.2 * n * (se / room) ** 2))
    return max(1, min(int(most), max(n + max(n // 2, MIN_TRIALS), min(target, GROWTH * n))))


def worst(values: list[float | None], expected: float) -> int | None:
    """The index of the trial whose number is furthest from ``expected`` (``None`` when no trial gave one)."""
    best: int | None = None
    for i, v in enumerate(values):
        if v is not None and (best is None or abs(v - expected) > abs(values[best] - expected)):  # type: ignore[operator]
            best = i
    return best


def failed_share_verdict(failed: int, started: int, verdict: str | None) -> tuple[str | None, str]:
    """``(the verdict once failed trials count, why)``: more than :data:`FAULT_SHARE` of ``started`` failing is a
    ``"fail"``; any failed or non-finite trial makes a ``"pass"`` (or an open one) ``"undecided"``, since the mean of
    the trials that survived is not the case's mean; else as it was."""
    if started <= 0 or failed <= 0:
        return verdict, ""
    if failed / started > FAULT_SHARE:
        return "fail", "failed_trials"
    if verdict != "fail":
        return "undecided", "failed_trials"
    return verdict, ""


def is_partial(mc: dict[str, Any] | None) -> bool:
    """True for a rule of every trial that held in every trial judged, when time ran out before all were (verdict
    ``"partial"``). Stop and repair logic reads it as a pass; what was checked is reported as not fully checked."""
    return isinstance(mc, dict) and mc.get("verdict") == "partial"


def partial_words(mc: dict[str, Any] | None) -> str:
    """The plain sentence for a partial check, ``""`` for any other."""
    if not is_partial(mc):
        return ""
    return (f"passed on {int(mc.get('judged') or 0)} of {int(mc.get('planned') or 0)} trials; the rest were not judged "
            "in the time allowed")


def words(mc: dict[str, Any] | None) -> str:
    """How the check's number was had, in a few plain words for a record line: ``the mean over 310 trials, Monte Carlo
    error ±0.0024 (one standard error)``, ``the worst of 20 trials``; ``""`` for a check judged on one number."""
    if not isinstance(mc, dict):
        return ""
    n, se = int(mc.get("trials") or 0), mc.get("se")
    read_as = mc.get("read_as")
    if read_as == "mean" and mc.get("varies") is not False:
        text = f"the mean over {n} trial{'s' if n != 1 else ''}"
        if _number(se):
            text += f", Monte Carlo error ±{_oracle.fmt_digits(float(se), 2)} (one standard error)"
    elif read_as == "every_trial":
        text = f"the worst of {n} trial{'s' if n != 1 else ''}"
        asked = int(mc.get("trials_asked") or 0)
        if is_partial(mc):
            text += f" ({partial_words(mc)})"
        elif asked > n and mc.get("why") != "failed_trials":
            text += f" (judged on {n} of {asked} trials)"
    else:
        text = "one trial's number, the same in every trial"
    failed, started = int(mc.get("failed_trials") or 0), int(mc.get("started") or 0)
    if failed:
        text += (f"; {failed} of {started} trials on its case failed"
                 + (f" (first: {mc['failed_example']})" if mc.get("failed_example") else ""))
    return text


def undecided_words(mc: dict[str, Any] | None, limit: float | None) -> str:
    """Why a check was not decided, in one plain clause."""
    if not isinstance(mc, dict):
        return "the trials did not decide it"
    if mc.get("why") == "failed_trials":
        return (f"{mc.get('failed_trials')} of {mc.get('started')} trials on its case failed, so the trials that "
                "finished cannot stand for the case and the check was not decided (it is not a pass)")
    if mc.get("read_as") == "every_trial":
        return (f"a rule of every trial held in the {int(mc.get('trials') or 0)} trials FI could run in the time it had, "
                f"fewer than the {int(mc.get('trials_asked') or 0)} it needs, so it is not shown to hold in all of them")
    n, se = int(mc.get("trials") or 0), mc.get("se")
    when = ("the most FI runs for one check" if mc.get("why") == "most_trials" else "all FI could run in the time it had")
    tolerance = f"±{_oracle.fmt_digits(float(limit), 2)}" if _number(limit) else "its tolerance"
    interval = mc.get("binomial_interval")
    if isinstance(interval, list) and len(interval) == 2 and all(_number(x) for x in interval):
        # A 0/1 outcome is decided on its binomial interval (:func:`binomial_verdict`), so that interval is what is shown.
        lo, hi = (_oracle.fmt_digits(float(x), 2) for x in interval)
        return (f"the range of the share of {n} trials ({when}), from {lo} to {hi}, lies neither inside nor outside "
                f"the tolerance {tolerance}")
    band = f"±{_oracle.fmt_digits(BAND_SE * float(se), 2)}" if _number(se) else "their band"
    return (f"three standard errors ({band}) around the mean of {n} trials ({when}) lie neither inside nor outside "
            f"the tolerance {tolerance}")


def record(stat: dict[str, Any] | None, *, read_as: str, verdict: str | None, why: str, failed: int, started: int,
           failed_example: str = "", **extra: Any) -> dict[str, Any]:
    """The ``mc`` entry a check keeps (``needs/ORACLE_CHECK.json``): how it was read, its trials and their error, and
    the verdict they gave (``pass``, ``fail``, ``undecided``, or ``partial``: passed on every trial judged but time ran
    out before all ``planned`` were; it carries ``judged`` and ``planned``) with why it was not decided (``time``, ``most_trials``,
    ``failed_trials``)."""
    stat = stat or {}
    se = stat.get("se")
    out = {
        "read_as": read_as, "on": stat.get("on") or "", "trials": int(stat.get("trials") or 0),
        "dropped": int(stat.get("dropped") or 0), "started": int(started), "failed_trials": int(failed),
        "varies": stat.get("varies"), "se": se, "band": BAND_SE * se if _number(se) else None,
        "inner": stat.get("inner"), "inner_mean": stat.get("inner_mean"), "shape": stat.get("shape") or "",
        "verdict": verdict or "undecided", "why": why if (verdict or "undecided") == "undecided" or why == "failed_trials"
        else "",
        **({"binomial_interval": stat["interval"]} if stat.get("binary") else {}), **extra,
    }
    if failed_example:
        out["failed_example"] = failed_example
    return out
