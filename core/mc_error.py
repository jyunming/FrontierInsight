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
  :func:`repeats_trials`, or, when the plan does not say, some number the simulation returns varies; and the
  protocol declares the number a proportion, never guessed from its values, and the tolerance is not exactly 0) is judged on its binomial interval, never on its sample spread (:func:`wilson`, :func:`binomial_verdict`), at every
  number of trials: ten zeros are not a rate of 0 (their interval at three standard errors reaches 0.47), so they do
  not decide a check that expects 0.05 within 0.01; 2000 zeros fail it.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from . import metric_spec as _metric_spec
from . import oracle_check as _oracle
from . import oracle_forms as _forms
from . import trial_checks as _trial_checks
from .split_run import _STOCHASTIC_RE

#: How many standard errors the band around a measured mean is (about 99.7% two-sided for a normal mean). Two would
#: decide a correct simulation wrongly on one check in twenty, and a plan has several.
BAND_SE = 3.0
#: The band, in standard errors, for a mean of n trials when n is under :data:`MIN_TRIALS`, by n - 1 (Student's t
#: with n - 1 degrees of freedom at the two-sided coverage of :data:`BAND_SE` normal ones, about 99.7%): with so few
#: trials the spread itself is uncertain, so the band is wider (4 trials: about 9 standard errors).
SMALL_N_BAND_SE = {1: 235.801, 2: 19.207, 3: 9.219, 4: 6.62, 5: 5.507, 6: 4.904, 7: 4.53, 8: 4.277}


def band_se(trials: int) -> float:
    """How many standard errors the band around a mean of ``trials`` trials is: :data:`BAND_SE`, wider under
    :data:`MIN_TRIALS` trials (:data:`SMALL_N_BAND_SE`)."""
    return SMALL_N_BAND_SE.get(int(trials) - 1, BAND_SE) if int(trials) < MIN_TRIALS else BAND_SE


#: The first batch of trials, from which the spread is read (and on which a rule of every trial is judged).
PILOT_TRIALS = 20
#: How many trials a rule of every trial is judged on when no study size is given: the first batch, then more up to this
#: many (a disagreement that shows only in a later trial must not pass on the first batch alone).
EVERY_TRIALS = 10 * PILOT_TRIALS
#: Fewer trials than this give a spread too rough to decide anything by.
MIN_TRIALS = 10
#: The fixed seed of the bootstrap behind the error of a number taken over the trials together (:func:`pooled_summary`):
#: the same trials always give the same error, so a verdict does not move between two runs of the checks.
BOOTSTRAP_SEED = 20251101
#: How many resamples of the trials that error is read from (fewer for very many trials).
BOOTSTRAP_MAX, BOOTSTRAP_MIN = 300, 100
#: A check with an sd() is not decided on fewer trials than this: the bootstrap error of a standard deviation is biased low.
SD_MIN_TRIALS = 30
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
    return _metric_spec.proportion_ids(protocol)


# The words that say a design draws random numbers are kept where the split-run keeps them (one list).


def repeats_trials(protocol: dict[str, Any] | None, design: dict[str, Any] | None = None) -> bool | None:
    """Whether the plan runs more than one trial per setting (``runs_per_setting`` or ``min_runs_per_setting`` above
    1): a random study. ``False`` for one run per setting (a deterministic study), ``None`` when the plan says neither.

    This is the one reading of the plan's run counts: the frame the simulation is written in (``run_trial`` or
    ``run_cell``), the raw-data check, the checks of a random quantity and the planned runs all follow it, so a plan that
    says one run per setting beside a minimum of 200 is never a deterministic study in one place and a random one in
    another. With ``design`` given, a design whose own words (hypothesis, method, expected outcome, variables) name a random
    process (:data:`_STOCHASTIC_RE`) also repeats trials, whatever the counts say."""
    if isinstance(design, dict) and _STOCHASTIC_RE.search(json.dumps(
            {key: design.get(key) for key in ("hypothesis", "method", "expected_outcome", "variables")},
            ensure_ascii=False, default=str)):
        return True
    if not isinstance(protocol, dict):
        return None
    runs = [protocol.get(k) for k in ("runs_per_setting", "min_runs_per_setting")]
    runs = [int(r) for r in runs if _number(r)]
    if not runs:
        return None
    return max(runs) > 1


def design_repeats_trials(design: Any) -> bool:
    """:func:`repeats_trials` read from a whole design (its protocol and its words), ``False`` when it says nothing."""
    if not isinstance(design, dict):
        return False
    protocol = design.get("protocol")
    return bool(repeats_trials(protocol if isinstance(protocol, dict) else None, design))


def planned_repeats_trials(quest_root: Path | str, protocol: dict[str, Any] | None,
                           design: dict[str, Any] | None = None) -> bool | None:
    """:func:`repeats_trials` of ``protocol`` with the words of ``design`` (given, else the quest's ``plan.md``): the reading
    made before a protocol is frozen, and the one that is frozen with it. Without a design (a quest FI did not plan) it is
    the protocol's counts alone."""
    if not isinstance(design, dict):
        from . import plan as _plan  # a late import: core.plan imports this module

        design = _plan.load_design(Path(quest_root))[0]
    return repeats_trials(protocol, design if isinstance(design, dict) else None)


def quest_repeats_trials(quest_root: Path | str, protocol: dict[str, Any] | None) -> bool | None:
    """Whether the study repeats its trials: what every check that runs or judges the trials asks, so the frame, the
    known-answer cases, the raw-data check and the judging of a random quantity read one answer. After the freeze it is the
    reading frozen with the protocol (``needs/FROZEN_PROTOCOL.json``, covered by its own checksum), so a plan.md edited
    afterwards cannot change it for the same run. Before the freeze, and for a frozen record written before it kept the
    reading, it is :func:`planned_repeats_trials` (the plan.md's words with the protocol's counts)."""
    from . import frozen_protocol as _frozen

    kept = _frozen.repeats_trials_of(Path(quest_root))
    return kept if kept is not None else planned_repeats_trials(quest_root, protocol)


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


#: Set on a check by FI when the plan's model has answered the question of how it holds over the trials.
ASKED_KEY = "reading_asked"


def unsaid_reading(protocol: dict[str, Any] | None, design: dict[str, Any] | None = None) -> list[str]:
    """The checks whose reading the plan's model has not been asked: every check FI can run on a case of its own, when the
    study repeats its trials (or the plan does not say it does not), until the focused question (``PART_RULES["per_trial"]``
    in core/plan.py) has been answered for it (:data:`ASKED_KEY`); a check whose measure takes mean() or sd() over the trials is
    about the trials together and is not asked. A ``per_trial`` the plan volunteered while writing the
    plan is not an answer: asked one check at a time, the model reads the check's own definition (6 plans, 53 checks: all
    answered right), while volunteered beside everything else it got 3 of 53 wrong. Whether a check holds in every single
    trial or only on average is a fact about the science, so FI asks once and never guesses it; a check still unsaid is
    read on the mean over trials (:func:`reads_a_mean`) unless its kind is a rule of every trial."""
    if not isinstance(protocol, dict) or repeats_trials(protocol, design) is False:
        return []
    return [str(o["name"]).strip() for o in protocol.get("oracles") or []
            if isinstance(o, dict) and str(o.get("name") or "").strip() and _oracle.case_of(o) is not None
            and not o.get(ASKED_KEY) and not _forms.across_calls(_oracle.case_of(o)[1])]


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
    return _forms.names(measure, across=True)  # (a measure with mean() / sd() over trials: the names inside them too)


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


def _spread(xs: list[float]) -> float | None:
    """The sample standard deviation (n - 1) of ``xs``; ``None`` with fewer than two."""
    n = len(xs)
    if n < 2:
        return None
    m = math.fsum(xs) / n
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - 1))


def pooled_summary(rows: list[dict[str, Any]], measure: str, expected: Any = None, limit: Any = None,
                   proportions: Any = (), repeated: bool | None = None) -> dict[str, Any] | None:
    """What the trials ``rows`` say about a measure that takes ``mean()`` or ``sd()`` over the trials
    (:func:`core.oracle_forms.across_form`), such as ``abs(mean(hit) - 0.95)`` (a coverage) or ``abs(sd(est) / sd_ref - 1)``
    (a spread). The measure is worked out once on all the trials together: each ``mean(x)`` and ``sd(x)`` is taken over the
    trials of the value ``x`` gives in each (``sd`` with n - 1) and any other name is its mean over the trials. Fields as
    :func:`summarise`, plus ``across: True``. The error is, for the mean of a 0/1 outcome of a random simulation that the
    protocol declares a proportion, its Wilson interval mapped through the formula (:func:`statistic`'s rule for
    a 0/1 outcome); else a bootstrap over the trials with a fixed seed, never 0 for a random study: when every trial gives the
    same value the check is a constant (an exact invariant such as ``abs(mean(d))`` with ``d`` always 0, decided on that
    value), except that an ``sd()`` of one is not decided. ``sd_small`` marks fewer than :data:`SD_MIN_TRIALS` trials with an
    sd(). ``None`` when the measure takes nothing over trials or no trial has its names."""
    measure = str(measure or "").strip()
    form = _forms.across_form(measure)
    if form is None:
        return None
    bare = [n for n in _forms.names(form.text) if n not in form.aliases]
    columns: dict[str, list[float]] = {name: [] for name in [*form.aliases, *bare]}
    for r in rows:
        if not isinstance(r, dict):
            continue
        numeric = {k: float(v) for k, v in r.items() if _number(v)}
        got: dict[str, float] = {}
        for alias, (_op, arg) in form.aliases.items():
            v = numeric.get(arg) if _forms.is_name(arg) else _forms.evaluate(arg, numeric).value
            if v is None or not math.isfinite(v):
                break
            got[alias] = v
        else:
            if all(name in numeric for name in bare):
                for name, v in {**got, **{name: numeric[name] for name in bare}}.items():
                    columns[name].append(v)
    n = len(columns[next(iter(columns))])
    if n == 0:
        return None
    inner_text, shape = inner(measure)
    inner_form = form.rewrite(inner_text)

    def taken(indexes: list[int] | range) -> dict[str, float] | None:
        env: dict[str, float] = {}
        for name, col in columns.items():
            xs = [col[i] for i in indexes]
            if name in form.aliases and form.aliases[name][0] == "sd":
                got = _spread(xs)
                if got is None:
                    return None
                env[name] = got
            else:
                env[name] = math.fsum(xs) / len(xs)
        return env

    flat = all(_same(min(col), max(col)) for col in columns.values())
    sd_used = any(op == "sd" for op, _arg in form.aliases.values())
    out: dict[str, Any] = {"on": "means", "trials": n, "dropped": len(rows) - n, "inner": inner_text, "shape": shape,
                           "varies": None if n < 2 else True, "se": None, "across": True}
    if sd_used and n < SD_MIN_TRIALS:
        out["sd_small"] = True
    everyone = taken(range(n))
    if everyone is None:  # an sd() of fewer than two trials: not a fault, only too few trials
        out["value"] = out["inner_mean"] = None
        out["problem"] = "an sd() needs at least two trials that gave a number"
        return out
    out["means"] = {k: v for k, v in everyone.items() if k not in form.aliases}
    got = _forms.evaluate(form.text, everyone)
    out["value"] = got.value
    if got.value is None:
        out["problem"] = got.problem or "the formula gives no number on the trials together"
    centre = _forms.evaluate(inner_form, everyone).value
    out["inner_mean"] = centre
    if n < 2 or centre is None:
        return out
    # A rate of 0/1 outcomes of a random simulation is judged on its binomial interval, not on a spread of the outcomes,
    # but only for a result the plan declares a proportion: a count or a mean that happens to be 0/1 is not one.
    random_study = not flat or (repeated if repeated is not None else returned_varies(rows))
    only = next(iter(form.aliases)) if len(form.aliases) == 1 else ""
    if (only and form.aliases[only][0] == "mean" and _forms.is_name(form.aliases[only][1])
            and form.aliases[only][1] in {str(x) for x in proportions or ()}
            and all(v in (0.0, 1.0) for v in columns[only]) and random_study
            and not (_number(limit) and float(limit) == 0.0)):
        lo, hi = wilson(int(sum(columns[only])), n)

        def reach(text: str) -> tuple[float, float] | None:
            found = [v for v in (_forms.evaluate(text, {**everyone, only: lo + (hi - lo) * i / 100}).value
                                 for i in range(101)) if v is not None]
            return (min(found), max(found)) if found else None

        whole, inside = reach(form.text), reach(inner_form)
        if whole is not None and inside is not None:
            out.update(binary=True, successes=int(sum(columns[only])), interval=list(whole),
                       se=(inside[1] - inside[0]) / (2 * BAND_SE))
            return out
    if flat:
        # Every trial gives the same value: a constant, decided on that value, as a check that does not move is. Only a
        # spread of it says nothing in a random study.
        out["se"] = None if sd_used and random_study else 0.0
        return out
    rng = random.Random(BOOTSTRAP_SEED)
    seen: list[float] = []
    for _ in range(max(BOOTSTRAP_MIN, min(BOOTSTRAP_MAX, 2_000_000 // n))):
        env = taken(rng.choices(range(n), k=n))
        v = _forms.evaluate(inner_form, env).value if env is not None else None
        if v is not None:
            seen.append(v)
    se = _spread(seen) if len(seen) >= 2 else None
    out["se"] = se if se else None  # an error of exactly 0 from a random study decides nothing
    return out


def statistic(rows: list[dict[str, Any]], measure: str, means: Any = (), expected: Any = None, limit: Any = None,
              proportions: Any = (), *, repeated: bool | None = None) -> dict[str, Any] | None:
    """The mean check's number and its error (see the module docstring): the formula on the names' means when every name
    it uses is a declared mean or proportion (``means``), else the mean of the formula trial by trial. A 0/1 outcome
    (``binary``, only for a name the protocol declares a proportion; ``expected`` and ``limit`` are the check's) carries
    its Wilson ``interval`` and the standard error
    that interval stands for. ``proportions``: the names the protocol declares proportions (:func:`proportion_names`)."""
    measure = str(measure or "").strip()
    first = next((r for r in rows if isinstance(r, dict)), None)
    if first is None:
        return None
    if measure not in first and _forms.across_form(measure) is not None:
        # mean() / sd() over the trials: the measure is worked out on all the trials together, never trial by trial.
        return pooled_summary(rows, measure, expected, limit, proportions, repeated)
    used = measure_names(measure, first)
    # How far one quantity falls beyond a limit, expecting 0 (the check FI writes for a commission's limit) is a
    # statement about the quantity's mean: the shortfall of the mean, never the mean of each trial's shortfall (a rate's
    # trials fall 0 or the whole limit short).
    limited = (shortfall_of(measure) if measure not in first and _number(expected) and float(expected) == 0.0 else None)
    stat = (summarise(rows, measure) if used and (set(used) <= set(means or ()) or limited) else
            per_trial_summary(rows, measure))
    if stat is None:
        return None
    # A 0/1 outcome the plan declares a proportion, in a random simulation: its uncertainty is its binomial interval,
    # never its sample spread (ten zeros have a spread of 0, which says nothing about a rate of 0.05). One rule at
    # every number of trials. The kind is the plan's to say, not the values': a count or a mean that happens to be 0 or 1
    # in every trial is judged on its sample spread like any other number. A declared proportion with a tolerance of
    # exactly 0 is an exact claim, read as a constant.
    # Whether the trials are random is the plan's to say (``repeated``: it runs more than one trial per setting,
    # :func:`repeats_trials`), not the returned numbers': a trial that returns only its 0/1 outcome, zero every time,
    # varies in nothing. Only when the plan does not say are the returned numbers read.
    values = _outcome_values(rows, stat)
    exact = _number(limit) and float(limit) == 0.0
    declared = str(stat.get("inner") or "") in {str(p) for p in proportions or ()} and not exact
    both = len(set(values)) > 1
    random = both or (repeated if repeated is not None else returned_varies(rows))
    if values and all(v in (0.0, 1.0) for v in values) and random and declared:
        lo, hi = wilson(int(sum(values)), len(values))
        stat.update(binary=True, successes=int(sum(values)), interval=[lo, hi], se=(hi - lo) / (2 * BAND_SE))
    elif limited is not None and not exact and limited[0] in {str(p) for p in proportions or ()}:
        # A declared proportion against a limit: its uncertainty is its binomial interval, carried to the signed
        # distance from the limit the check is judged on (ten ones in ten trials do not show a rate of 0.9 or more).
        name, at, side = limited
        outcomes = [float(r[name]) for r in rows if isinstance(r, dict) and _number(r.get(name))]
        if outcomes and all(v in (0.0, 1.0) for v in outcomes) and (len(set(outcomes)) > 1 or random):
            lo, hi = wilson(int(sum(outcomes)), len(outcomes))
            signed = sorted(at - p if side == "at_least" else p - at for p in (lo, hi))
            stat.update(binary=True, successes=int(sum(outcomes)), interval=signed, se=(signed[1] - signed[0]) / (2 * BAND_SE))
    return stat


def shortfall_of(measure: str) -> tuple[str, float, str] | None:
    """``(name, limit, "at_least" | "at_most")`` for a measure that is how far one returned name falls beyond a limit,
    ``max(0, limit - (name))`` or ``max(0, (name) - limit)`` (core/known_values.py writes it); else ``None``."""
    try:
        body = ast.parse(str(measure or "").strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not (isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id == "max" and not body.keywords
            and len(body.args) == 2 and isinstance(body.args[0], ast.Constant) and body.args[0].value == 0
            and isinstance(body.args[1], ast.BinOp) and isinstance(body.args[1].op, ast.Sub)):
        return None

    def number(node: ast.AST) -> float | None:
        sign = -1.0 if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) else 1.0
        node = node.operand if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) else node
        ok = isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        return sign * float(node.value) if ok else None  # type: ignore[union-attr]

    left, right = body.args[1].left, body.args[1].right
    if number(left) is not None and isinstance(right, ast.Name):
        return right.id, number(left), "at_least"  # type: ignore[return-value]
    if isinstance(left, ast.Name) and number(right) is not None:
        return left.id, number(right), "at_most"  # type: ignore[return-value]
    return None


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


def decide(centre: float, expected: float, limit: float, se: float, *, shape: str = "",
           band_se: float = BAND_SE) -> str | None:
    """``"pass"`` when three standard errors around the measured ``centre`` lie inside the tolerance, ``"fail"`` when
    they lie outside it, ``None`` when the trials do not decide it. A bound ``max(0, X)`` expecting 0 is judged on its
    one side (``centre`` is X); ``abs(X)`` expecting 0 is a distance (``centre`` is X). The tolerance is never widened."""
    if shape == "bound" and expected == 0:
        gap = centre
    elif shape == "abs" and expected == 0:
        gap = abs(centre)
    else:
        gap = abs(centre - expected)
    band = band_se * se
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
    ``None`` while it is not decided, and always with fewer than :data:`MIN_TRIALS` trials (moving or not; for a check
    the time left with fewer, see :func:`missed_on_few`)."""
    if expected is None or limit is None:
        return None
    centre, shape = _centre(stat, float(expected))
    se, n = stat.get("se"), int(stat.get("trials") or 0)
    if centre is None or not _number(se):
        return None
    if n < MIN_TRIALS or stat.get("sd_small"):  # too few trials to show a spread or a constant (an sd(): see SD_MIN_TRIALS)
        return None
    if stat.get("binary"):
        return binomial_verdict(stat, float(expected), float(limit))
    return decide(float(centre), float(expected), float(limit), float(se), shape=shape)


def missed_on_few(stat: dict[str, Any], expected: float | None, limit: float | None) -> bool:
    """A check the time left with 2 to ``MIN_TRIALS - 1`` trials is still a failure when it is missed by far: the mean
    of trials whose number moves lies outside the tolerance by more than the band for so few (:func:`band_se`, Student's
    t at the coverage of :data:`BAND_SE`). Only a failure is decided this way, never a pass (so few trials cannot show
    a rare draw is absent), and only for a mean over single trials with its plain standard error: a number that does
    not move, a 0/1 outcome, a number worked out over all the trials together and an sd() are not."""
    if expected is None or limit is None or stat.get("binary") or stat.get("across") or stat.get("sd_small"):
        return False
    se, n = stat.get("se"), int(stat.get("trials") or 0)
    if not 2 <= n < MIN_TRIALS or not _number(se) or not float(se) > 0 or not stat.get("varies"):
        return False
    if stat.get("on") not in (None, "", "trials"):
        return False
    centre, shape = _centre(stat, float(expected))
    if centre is None:
        return False
    return decide(float(centre), float(expected), float(limit), float(se), shape=shape, band_se=band_se(n)) == "fail"


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
    if read_as == "mean" and mc.get("across"):
        text = f"worked out over all {n} trial{'s' if n != 1 else ''} together"
        if _number(se):
            text += f", Monte Carlo error ±{_oracle.fmt_digits(float(se), 2)} (one standard error)"
    elif read_as == "mean" and mc.get("varies") is not False:
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
    if mc.get("judged_on"):  # decided again on the main run's own trials (oracle_check.rejudge_on_delivered)
        text += f"; judged on {mc['judged_on']}"
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
    when = ("the most FI runs for one check" if mc.get("why") == "most_trials"
            else "all the trials of the delivered run" if mc.get("why") == "delivered"
            else "all FI could run in the time it had")
    tolerance = f"±{_oracle.fmt_digits(float(limit), 2)}" if _number(limit) else "its tolerance"
    interval = mc.get("binomial_interval")
    if isinstance(interval, list) and len(interval) == 2 and all(_number(x) for x in interval):
        # A 0/1 outcome is decided on its binomial interval (:func:`binomial_verdict`), so that interval is what is shown.
        lo, hi = (_oracle.fmt_digits(float(x), 2) for x in interval)
        return (f"the range of the share of {n} trials ({when}), from {lo} to {hi}, lies neither inside nor outside "
                f"the tolerance {tolerance}")
    if mc.get("across"):  # a number worked out over all the trials together (mean() / sd() in the measure)
        if mc.get("sd_small"):
            return (f"an sd() is not decided on fewer than {SD_MIN_TRIALS} trials, and FI had {n} ({when}), so the "
                    f"check was not decided")
        if not _number(se):
            return f"the {n} trials ({when}) give no error to judge the number by, so the check was not decided"
        return (f"three standard errors (±{_oracle.fmt_digits(BAND_SE * float(se), 2)}) around the number worked out over "
                f"all {n} trials ({when}) lie neither inside nor outside the tolerance {tolerance}")
    if n < MIN_TRIALS:
        if n >= 2 and _number(se) and float(se) == 0 and mc.get("varies") is False:
            return (f"the {n} trials FI could run ({when}) all gave the same number, and a number that does not move is "
                    f"judged only on at least {MIN_TRIALS} trials, so the check was not decided")
        if n >= 2 and _number(se) and float(se) > 0:
            return (f"{n} trials ({when}) are fewer than the {MIN_TRIALS} a check needs to pass; on so few only a miss "
                    f"of the tolerance {tolerance} by more than {_band_words(n, float(se))} is decided (a failure), so "
                    "the check was not decided")
        return f"{n} trial(s) ({when}) are fewer than the {MIN_TRIALS} a check needs, so the check was not decided"
    band = _band_words(n, float(se)) if _number(se) else "their band"
    return f"{band} around the mean of {n} trials ({when}) lie neither inside nor outside the tolerance {tolerance}"


def _band_words(n: int, se: float) -> str:
    """The band the verdict used around a mean of ``n`` trials, in words."""
    if n >= MIN_TRIALS:
        return f"three standard errors (±{_oracle.fmt_digits(BAND_SE * se, 2)})"
    return (f"the band for only {n} trials ({_oracle.fmt_digits(band_se(n), 2)} standard errors, "
            f"±{_oracle.fmt_digits(band_se(n) * se, 2)})")


def record(stat: dict[str, Any] | None, *, read_as: str, verdict: str | None, why: str, failed: int, started: int,
           failed_example: str = "", **extra: Any) -> dict[str, Any]:
    """The ``mc`` entry a check keeps (``needs/ORACLE_CHECK.json``): how it was read, its trials and their error, and
    the verdict they gave (``pass``, ``fail``, ``undecided``, or ``partial``: passed on every trial judged but time ran
    out before all ``planned`` were; it carries ``judged`` and ``planned``) with why it was not decided (``time``, ``most_trials``,
    ``failed_trials``, or ``delivered``: still not decided on the delivered run's own trials)."""
    stat = stat or {}
    se = stat.get("se")
    out = {
        "read_as": read_as, "on": stat.get("on") or "", "trials": int(stat.get("trials") or 0),
        "dropped": int(stat.get("dropped") or 0), "started": int(started), "failed_trials": int(failed),
        "varies": stat.get("varies"), "se": se,
        "band": (BAND_SE if stat.get("across") else band_se(int(stat.get("trials") or 0))) * se if _number(se)
        else None,
        "inner": stat.get("inner"), "inner_mean": stat.get("inner_mean"), "shape": stat.get("shape") or "",
        "verdict": verdict or "undecided", "why": why if (verdict or "undecided") == "undecided" or why == "failed_trials"
        else "",
        **({"binomial_interval": stat["interval"]} if stat.get("binary") else {}),
        **({"across": True} if stat.get("across") else {}), **({"sd_small": True} if stat.get("sd_small") else {}), **extra,
    }
    if failed_example:
        out["failed_example"] = failed_example
    return out
