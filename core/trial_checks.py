"""Two checks FI makes itself on what the trials gave, with no model: the failed trials of each setting, and whether the
intervals and standard errors a simulation reports are right against the true values it was given.

A simulation knows its truth: the plan fixes the true values it generates the data with, so FI can measure how often a
reported 95% interval covers the true value, and whether a reported standard error is about as large as the spread of
the estimates themselves. A wrong code (a noise level typed into a fitting function, a weight on the wrong scale) shows
up here with numbers, where a reader of the code and a test written by the model that wrote it may both miss it.

**Failed trials of a setting (:func:`failures`).** FI already stopped when every trial failed. A setting where most trials
fail is as bad: its numbers rest on the few that happened to survive, and a quest once reported coverages computed from 2
of 100 trials. A setting where more than :data:`FAULT_FRACTION` of the trials failed is a fault: the simulation is sent back
to be repaired with the reasons the trials gave; if the repairs end with it, the setting's numbers are left out of what the
analysis is given and the paper says so; if more than half of the settings are so, the quest stops (too little would be
left to report). Every failed trial is recorded and the paper states how many failed (core/paper_facts.py).

**Calibration against the truth (:func:`calibration`).** Read only from what the plan declares, never from the
names the code happens to use. A metric of the protocol with a ``known_value``
(:func:`core.metric_spec.normalize`) says the true value the simulation is given (a number, or the threshold or grid
parameter that holds it, on the estimate's own scale), the name run_trial returns for the estimate, the name for its
standard error (or variance) and/or the names of its interval's two ends, and the interval's level. FI computes the rest
itself from the trials: how far the mean estimate lies from the true value in units of the estimates' own spread
(``bias``), the ratio of the estimates' standard deviation over the trials to the typical reported standard error (the
root of the mean squared one), and, only when the trial reports its interval's two ends, whether each trial's interval
covers the true value. FI never builds a z interval of its own from a standard error: a fit with few degrees of freedom
reports a wider interval, and a z interval would under-cover by design (more visibly the more trials). A trial that ran
but gave no number for the estimate is a miss (``no_estimate``). It never reads a coverage flag the code computed. A metric whose declared names the trials do not return is a fault (the code does not keep the plan's
contract). Without any declaration the calibration is "not checked", and the paper and ``QUALITY.md`` say so.

For each setting with at least :data:`MIN_TRIALS` trials: the coverage with its Wilson interval, and the ratio with its
sampling interval, both at the family level 1 - :data:`FAMILY_ALPHA` / k over the k settings judged (Bonferroni: with 25
settings each interval is at 99.8%, so a clean run does not stumble on one unlucky setting); and the same two numbers pooled
over every setting, at 95%. A setting, or the pool, is a **gross miss** when the whole coverage interval lies more than
:data:`COVERAGE_GAP` below the level, or the whole interval of the ratio lies outside [:data:`RATIO_LOW`,
:data:`RATIO_HIGH`]. **One gross miss is a fault FI owns** (a wrong code at one corner of the grid is as wrong as one
everywhere; a standard error too small by 1.6 everywhere is caught by the pool): it is sent to repair with the numbers, and
if it stays the quest stops (a wrong code must never reach a paper). A smaller miss is not a fault (an interval of the wrong
form can truly under-cover: a normal interval with three degrees of freedom covers about 0.86): it is recorded and
reported, and so is a coverage above the level beyond its sampling error (intervals wider than they need be).

**The multiplier of the standard error the interval uses (:func:`_judge_multiplier`).** The calibration above judges an
interval by how often it covers, which a moderate miss in its form does not move beyond sampling error (a normal quantile
where the plan fixed t with 3 degrees of freedom covers about 0.90, not below 0.80). When the plan also declares the
interval's ``degrees_of_freedom`` (a whole number, or ``"normal"``: it says the interval is the estimate plus and minus
one multiplier of its standard error), FI recovers the multiplier the code used from the trials' own records,
``(upper - lower) / (2 * standard error)`` in each trial, and compares it with the plan's: the Student-t quantile of the
level with those degrees of freedom, or the normal one. They must agree in every trial of a setting to within
:data:`MULTIPLIER_TOLERANCE` (relative), plus what the rounding of the three values allows (a code that rounds what it returns
is read from the values themselves: the unit they were rounded to, :func:`_grain`, never assumed). A setting whose trials do not share one multiplier is a miss too (the interval is not
the estimate plus and minus a multiplier of the standard error the trial reports, as the declaration says). One missing
setting is a fault, sent to repair with both numbers, and a stop if it stays; it needs no true value. Nothing is guessed: a
plan that declares no ``degrees_of_freedom`` (the multiplier changes between settings, or the plan does not say) is not judged
here, and a function-level check of the quantile function cannot see which argument its caller passes.

Pure functions over FI's per-setting trial summary (``raw/trials.json`` cells, :func:`core.trial_runner._summary`); nothing
here knows a field of science, and no name of a quantity is written into it.
"""

from __future__ import annotations

import json
import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

from . import metric_spec as _metric_spec
from .frame_table import hits as _table_hits
from . import stats as _stats

#: A setting where more than this share of the trials failed is a fault (its numbers rest on the trials that survived).
FAULT_FRACTION = 0.5
#: A setting where more than this share failed is named in run.log (the paper states every failed trial: core/paper_facts.py).
STATE_FRACTION = 0.05
#: A coverage is grossly low when its whole interval is more than this below the level of the intervals. Chosen from the
#: replay of real quests (tests/test_trial_checks.py): clean quests' lowest coverages were 0.81 to 0.88 (normal intervals
#: with few degrees of freedom, a wrong form that is reported, not a fault), so 0.80 for a 95% level is below every one.
COVERAGE_GAP = 0.15
#: A standard error is grossly wrong when the whole interval of (spread of the estimates / reported error) is outside these.
#: Clean quests' ratios were 0.86 to 1.26 per setting; an error too small by 1.6 or more is outside.
RATIO_LOW = 0.67
RATIO_HIGH = 1.5
#: The chance, over all the settings of one quantity together, that a right code shows a gross miss by sampling alone
#: (each setting's interval is at 1 - FAMILY_ALPHA / k, Bonferroni over its k settings).
FAMILY_ALPHA = 0.05
#: The fewest trials of a setting that are judged.
MIN_TRIALS = 10
#: The interval's multiplier of the standard error matches the plan's when every trial of a setting lies within this share of
#: it, plus what the record's own rounding allows (:func:`_grain`). Chosen for a quantile typed with four significant digits
#: (3.182 for 3.1824, off by 1.4e-4, is the plan's), and finer than any difference of multiplier that moves a 95% interval's
#: coverage by a tenth of a point (t with 40 degrees of freedom against 41 differ by 4e-4, and both lie within it). The
#: recovered multiplier of an unrounded record agrees to about 1e-12.
MULTIPLIER_TOLERANCE = 1e-3
#: A series of values is read as rounded to ``k`` decimals when every value is a multiple of ``10**-k`` to this share of it
#: (float arithmetic leaves about 1e-16 of the value); with none of 0..:data:`GRAIN_MAX_DECIMALS` it is not rounded.
GRAIN_MAX_DECIMALS = 10
_GRAIN_EPS = 1e-12
#: An estimate is grossly biased when the whole sampling band of (mean estimate - true value) / spread of the estimates lies
#: beyond this many spreads (a bias of 1.1 spreads takes a right-width 95% interval down to about 0.80, the coverage line).
BIAS_GROSS = 1.0
#: ... and reported (not a fault) beyond this many.
BIAS_REPORTED = 0.5
#: An estimate lies inside the interval its own trial reported (an interval is built around its estimate). Measured on 128 estimates
#: of 515 recorded runs' settings: every one 1.000; one trial's estimates moved to the next trial: 0.825. Below this share of a
#: setting's trials the rows of the record may not belong together, and a statistic that pairs this estimate with another across
#: trials may not be right.
INSIDE_MIN = 0.9
#: The reason FI's harness gives a trial its time limit cut off (core/trial_runner.OUT_OF_TIME).
_OUT_OF_TIME = "the study's time (execution.timeout_s) ran out"
#: Where FI keeps what the checks found on the last run of the trials (QUALITY.md and the paper's required items read it).
RECORD = Path(".fi") / "trial_checks.json"

_Z95 = 1.959963984540054

def _number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


# ---- the record FI reads -----------------------------------------------------------------------------------------------


def read_cells(quest_root: Path | str) -> list[dict[str, Any]]:
    """The per-setting summary of the last run of the trials (``raw/trials.json``), or ``[]``."""
    try:
        data = json.loads((Path(quest_root) / "raw" / "trials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    cells = data.get("cells") if isinstance(data, dict) else None
    return [c for c in cells or [] if isinstance(c, dict)]


def read_thresholds(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / "raw" / "trials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    got = data.get("thresholds") if isinstance(data, dict) else None
    return dict(got) if isinstance(got, dict) else {}


def ledger_reasons(quest_root: Path | str) -> dict[str, Counter]:
    """``{setting key: Counter(reason)}`` of the failed trials in FI's ledger (``raw/ledger.jsonl``)."""
    out: dict[str, Counter] = {}
    try:
        with (Path(quest_root) / "raw" / "ledger.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if '"failed"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") == "trial" and row.get("status") == "failed":
                    out.setdefault(str(row.get("cell")), Counter())[str(row.get("reason") or "no reason given")] += 1
    except OSError:
        pass
    return out


def reasons_of_run(run_cells: Iterable[Any], quest_root: Path | str | None = None) -> dict[str, Counter]:
    """``{setting key: Counter(reason, with the place in the quest's code where it was raised)}`` of a run's failed trials
    (:class:`core.trial_runner.CellRun` objects)."""
    from . import trial_runner as _tr

    out: dict[str, Counter] = {}
    for c in run_cells:
        for r in getattr(c, "rows", []) or []:
            if r.get("status") == "ok":
                continue
            reason = str(r.get("reason") or "no reason given")
            where = _tr.last_frame(str(r.get("traceback") or ""), quest_root) if r.get("traceback") else ""
            out.setdefault(str(c.key), Counter())[reason + (f" (at {where})" if where else "")] += 1
    return out


# ---- failed trials per setting ---------------------------------------------------------------------------------------


@dataclass
class SettingFailures:
    key: str
    planned: int
    failed: int
    reasons: list[tuple[str, int]] = field(default_factory=list)
    #: Trials the study's time limit cut off (not failures of the code, not in ``failed``).
    out_of_time: int = 0

    @property
    def ran(self) -> int:
        """The planned trials the time limit did not cut off."""
        return max(0, self.planned - self.out_of_time)

    @property
    def fraction(self) -> float:
        return self.failed / self.ran if self.ran else 0.0

    @property
    def fault(self) -> bool:
        return self.ran > 0 and self.fraction > FAULT_FRACTION

    @property
    def stated(self) -> bool:
        return self.ran > 0 and self.fraction > STATE_FRACTION

    def says(self) -> str:
        why = "; ".join(f"{n}x {r if len(r) <= 200 else r[:197] + '...'}" for r, n in self.reasons[:3])
        return (f"{self.key or 'the one setting'}: {self.failed} of {self.ran} trials failed ({self.fraction:.0%})"
                + (f" - {why}" if why else ""))


def truth_names(protocol: dict[str, Any] | None) -> list[str]:
    """The names a ``known_value`` declares (the estimate, its standard error or variance, its interval's ends), in the plan's
    order: where the truth is declared, a trial with no number for one of them is a miss of the calibration
    (:func:`calibration`), and the frame of simulate.py lets a trial return them (:func:`core.skeleton.returnable_names`)."""
    out: list[str] = []
    for spec in _metric_spec.declared(protocol):
        kv = spec.get("known_value") or {}
        out += [str(kv[k]) for k in ("estimate", "standard_error", "variance") if kv.get(k)]
        out += [str(n) for n in kv.get("interval") or []]
    return list(dict.fromkeys(out))


def _truth_names(protocol: dict[str, Any] | None) -> set[str]:
    return set(truth_names(protocol))


def _metric_names(protocol: dict[str, Any] | None) -> list[tuple[str, dict[str, list[Any]]]]:
    """``[(id, undefined_when)]``: the plan's metrics a trial returns under their own id (a mean over a subset of the trials,
    ``given``, has no value outside the subset, so it is not one), with the settings where the plan says it has no value.
    A name a ``known_value`` declares is the calibration's (:func:`_truth_names`)."""
    truth = _truth_names(protocol)
    out = []
    for spec in _metric_spec.declared(protocol):
        if not spec.get("given") and spec["id"] not in truth:
            out.append((spec["id"], spec.get("undefined_when") or {}))
    return out


def _undefined_here(when: dict[str, list[Any]], cell: dict[str, Any]) -> bool:
    """Whether the plan says a number has no value at this setting (``undefined_when``)."""
    here = cell.get("cell") or {}
    for axis, values in when.items():
        if axis not in here:
            continue
        v = here[axis]
        for u in values:
            a, b = _number(v), _number(u)
            if (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)) or str(v) == str(u):
                return True
    return False


def _ran(rec: dict[str, Any]) -> set[int]:
    """The trials of a summary record that ran to the end (the trial numbers any of its values were returned for)."""
    return {t for m in (rec.get("metrics") or {}).values() if isinstance(m, dict)
            for t in (m.get("trials") or []) if isinstance(t, int)}


@dataclass
class Undefined:
    """A value of the plan that has no number (NaN, an infinity, nothing) in ``trials`` of the ``ran`` trials of a setting
    (``key``) that ran to the end. Not a failed trial: a power where there is no true effect has no value, by its definition.
    ``all`` when it has none in any trial there."""

    key: str
    metric: str
    trials: int
    ran: int

    @property
    def all(self) -> bool:
        return self.trials >= self.ran

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "metric": self.metric, "trials": self.trials, "ran": self.ran, "all": self.all}

    def says(self) -> str:
        """One plain line. With no value in any trial of the setting it is the value's definition there (a power where
        there is no true effect): said in QUALITY.md and run.log, not required in the paper's Limitations."""
        return said_undefined(self.as_dict())


def said_undefined(u: dict[str, Any]) -> str:
    """The plain line for one record of :func:`undefined` (``Undefined.as_dict``)."""
    where = str(u.get("key") or "the one setting")
    if u.get("all"):
        return f"`{u.get('metric')}` has no value at {where} by its definition: every trial there"
    return f"`{u.get('metric')}` has no value in {u.get('trials')} of {u.get('ran')} trials at {where}"


def undefined(cells: list[dict[str, Any]], protocol: dict[str, Any] | None = None) -> list[Undefined]:
    """Where a value the plan declares has no number in trials that ran to the end, per setting (and per value compared
    inside a run) and metric. A setting the plan names in the metric's ``undefined_when`` is not listed (the plan said so).
    The names a ``known_value`` declares are not here: a missing estimate of a known truth is a miss of the calibration."""
    names = _metric_names(protocol)
    out: list[Undefined] = []
    if not names:
        return out
    for rec in cells:
        metrics = rec.get("metrics") or {}
        ran = _ran(rec)
        for name, when in names:
            m = metrics.get(name)
            if not isinstance(m, dict) or not ran or _undefined_here(when, rec):
                continue
            empty = len(ran - set(_series(rec, name)))
            if empty:
                out.append(Undefined(str(rec.get("key") or ""), name, empty, len(ran)))
    return out


def failures(cells: list[dict[str, Any]], reasons: dict[str, Counter] | None = None,
             protocol: dict[str, Any] | None = None) -> list[SettingFailures]:
    """The failed trials of each setting the trials were run at, as FI's own record has them (a trial that raised, returned
    nothing, or ran out of time; a value compared inside a run is not a setting of its own: a run that fails fails for every
    value). Settings with no failed trial are included, with ``failed == 0``. A trial that ran to the end and returned NaN
    for a value is not a failed trial: see :func:`undefined` and, where the truth is declared, :func:`calibration`.
    ``protocol`` is not read (kept for the callers)."""
    out = []
    for c in cells:
        if c.get("level"):
            continue
        planned = int(c.get("planned") or 0)
        failed = int(c.get("failed") or 0)
        key = str(c.get("key") or "")
        got = Counter((reasons or {}).get(key) or Counter())
        # A trial FI's own time limit cut off is not a failure of the code: counted apart, never sent to a repair.
        late = sum(n for r, n in got.items() if r.startswith(_OUT_OF_TIME))
        if late:
            got = Counter({r: n for r, n in got.items() if not r.startswith(_OUT_OF_TIME)})
            failed = max(0, failed - late)
        out.append(SettingFailures(key, planned, failed, got.most_common(), out_of_time=late))
    return out


def is_level_fault(reason):
    """True when ``reason`` is FI's own message about a run's result per compared value (``read_returned`` writes it as
    ``"{entry}() returned `{name}` ..."``): a value given for some of the compared values and not for all."""
    return "() returned `" in str(reason)


def failures_directive(bad: list[SettingFailures], level_rule: str = "") -> str:
    """The plain message the simulation's repair is given for settings where most trials failed. ``level_rule``: FI's one
    rule for a result compared inside each run, said only when a failure is of that kind."""
    rule = (" " + level_rule) if level_rule and any(is_level_fault(r) for s in bad for r, _ in s.reasons) else ""
    shown = "\n".join(f"- {s.says()}" for s in bad[:8]) + (f"\n- and {len(bad) - 8} more setting(s)" if len(bad) > 8 else "")
    return ("FI ran the study's trials, and in these settings more than half of the trials failed, so their numbers would "
            "rest on the few trials that happened to survive:\n" + shown + "\n\nRepair the simulation so that a trial "
            "at these settings runs to the end and gives its numbers (the reasons and places above say where it stops). "
            "Never return NaN, an infinity or a made-up value in place of a trial that failed: let the trial raise, so FI "
            "counts it as failed. A number the plan says has no value at some settings is the plan's to declare, not the code's." + rule
            + " Keep everything else as it is.")


# ---- calibration against the truth -----------------------------------------------------------------------------------


@dataclass
class SettingCheck:
    key: str
    n: int
    value: float
    low: float
    high: float
    gross: bool
    below: bool  # a coverage below the level, or a ratio outside [1/1.25, 1.25], beyond its sampling error
    above: bool = False  # a coverage above the level beyond its sampling error: wider intervals than need be, not a fault


@dataclass
class Quantity:
    """One reported uncertainty FI judged: ``kind`` is ``coverage`` or ``se_ratio`` (or ``missing``: the trials do not
    return the names the plan declares, ``names``, for the metric ``label``, which is a fault of its own); ``names`` the
    trial values it is read from; ``level`` the intervals' level (coverage); ``truth`` the declared true value."""

    kind: str
    label: str
    names: list[str]
    level: float | None = None
    truth: str = ""
    settings: list[SettingCheck] = field(default_factory=list)
    #: ``multiplier``: the plan's multiplier of the standard error (its t or normal quantile) and the degrees of freedom the
    #: plan declared (a whole number or ``"normal"``).
    expected: float | None = None
    df: int | str | dict[str, dict[str, Any]] | None = None
    #: ``multiplier`` where the plan gives its multiplier per value of a factor: the plan's multiplier at each setting (by key).
    expected_by: dict[str, float] = field(default_factory=dict)
    #: ``multiplier`` with a plan that gives its multiplier per value: the values the settings ran that the plan does not name
    #: (not judged; the caller says so in ``not_checked``).
    unlisted: list[str] = field(default_factory=list)
    #: ``multiplier``: the settings whose intervals are symmetric on the log scale, so judged there (:func:`_interval_form`).
    on_log: list[str] = field(default_factory=list)
    #: The same number over every setting together (at 95%): what catches a moderate miss present everywhere.
    pooled: SettingCheck | None = None
    #: ``se_ratio`` only: the smallest (spread / reported error) ratio this run's trials could have flagged as a gross miss
    #: (:func:`ratio_detection_floor`). A reported standard error too small by less than this passes the check, whatever its
    #: size: the paper says so (core/paper_facts.py).
    detects_from: float | None = None
    #: The columns of the trials a disclosure keeps from what FI hands on, at the settings it marks (:func:`withheld_values`):
    #: the reported standard error (or variance) and the declared interval's two ends, or for a coverage the two ends.
    withheld: list[str] = field(default_factory=list)

    @property
    def judged(self) -> int:
        return len(self.settings)

    @property
    def gross(self) -> int:
        return sum(1 for s in self.settings if s.gross)

    @property
    def fault(self) -> bool:
        return self.kind == "missing" or self.gross > 0 or bool(self.pooled and self.pooled.gross)

    @property
    def below(self) -> int:
        return sum(1 for s in self.settings if s.below)

    @property
    def above(self) -> int:
        return sum(1 for s in self.settings if s.above)

    def says(self) -> str:
        if self.kind == "bias":
            vals = [s.value for s in self.settings]
            rng = f"{min(vals):+.2f} to {max(vals):+.2f}" if vals else "no value"
            return (f"(mean of `{self.label}` - the true value) / the spread of the estimates: {rng} over {self.judged} "
                    f"setting(s){self._pool('+.2f')} (0 when the estimate is right); beyond {BIAS_GROSS:g} beyond sampling "
                    f"error in {self.gross}")
        if self.kind == "inside":
            vals = [s.value for s in self.settings]
            return (f"the share of trials whose estimate `{self.label}` lies inside the interval the same trial reported: "
                    f"{min(vals):.3f} to {max(vals):.3f} over {self.judged} setting(s) (1 when the rows of the record belong "
                    f"together); below {INSIDE_MIN:g} in {self.below}")
        if self.kind == "no_estimate":
            vals = [s.value for s in self.settings]
            return (f"trials that ran but gave no number for `{self.label}`, the estimate of a declared true value: up to "
                    f"{max(vals):.0%} of a setting's trials, in {sum(1 for v in vals if v > 0)} setting(s); more than "
                    f"{FAULT_FRACTION:.0%} in {self.gross}")
        if self.kind == "multiplier":
            lows = [s.low for s in self.settings]
            highs = [s.high for s in self.settings]
            vals = [s.value for s in self.settings]
            exp = self.expected or float("nan")
            fixes = ("the normal quantile" if self.df == "normal" else f"t with {self.df} degrees of freedom")
            if isinstance(self.df, dict):
                (axis, table), = self.df.items()
                fixes = f"per `{axis}`: " + ", ".join(f"{v} {'the normal quantile' if d == 'normal' else f't with {d} degrees of freedom'}"
                                                      for v, d in list(table.items())[:6])
            fixes += f" at the {self.level:.0%} level" if self.level else ""
            split = any(s.gross and s.high - s.low > 0.05 * self.expected_by.get(s.key, exp) for s in self.settings)
            used = f"{min(vals):.4g} to {max(vals):.4g}" if vals else "no value"
            if split:
                used = f"no single multiplier in a setting (trials from {min(lows):.4g} to {max(highs):.4g})"
            fixed = (" or ".join(f"{e:.4g}" for e in sorted(set(self.expected_by.values()), reverse=True))
                     if self.expected_by else f"{exp:.4g}")
            scale = (f"; the intervals are symmetric on the log scale in {len(self.on_log)} setting(s), as for a normal "
                     "interval of ln(estimate), and FI judged their multiplier there (the standard error of ln(estimate) "
                     "is the reported standard error divided by the estimate)" if self.on_log else "")
            return (f"the multiplier of the standard error in the interval `{self.label}`: {used} over {self.judged} "
                    f"setting(s); the plan fixes {fixed} ({fixes}); not the plan's in {self.gross}{scale}")
        if self.kind == "missing":
            return (f"the trials do not return {', '.join(f'`{n}`' for n in self.names)}, which the plan declares for "
                    f"`{self.label}` (its estimate and its uncertainty), so FI could not measure them against the true "
                    "value")
        if self.kind == "coverage":
            vals = [s.value for s in self.settings]
            what = (f"coverage of the true value by `{self.label}` at the {self.level:.0%} level" if self.level is not None
                    else f"coverage `{self.label}`")
            rng = f"{min(vals):.3f} to {max(vals):.3f}" if vals else "no value"
            return (f"{what}: {rng} over {self.judged} setting(s)" + self._pool(".3f")
                    + f"; grossly low (more than {COVERAGE_GAP:g} below the level, beyond sampling error) in {self.gross}"
                    + (f"; above the level beyond sampling error in {self.above} (wider than need be, not a fault)"
                       if self.above else ""))
        vals = [s.value for s in self.settings]
        rng = f"{min(vals):.3g} to {max(vals):.3g}" if vals else "no value"
        return (f"spread of `{self.names[0]}` over the trials / its reported standard error `{self.names[1]}`: {rng} over "
                f"{self.judged} setting(s){self._pool('.3g')} (1 when the error is right); outside [{RATIO_LOW:g}, "
                f"{RATIO_HIGH:g}] beyond sampling error in {self.gross}")

    def _pool(self, fmt: str) -> str:
        p = self.pooled
        if p is None:
            return ""
        return (f", {format(p.value, fmt)} over all of them together"
                + (" (grossly off beyond sampling error)" if p.gross else ""))

    def worst(self, n: int = 4) -> list[SettingCheck]:
        if self.kind in ("coverage", "inside"):
            return sorted(self.settings, key=lambda s: s.value)[:n]
        if self.kind in ("bias", "no_estimate"):
            return sorted(self.settings, key=lambda s: -abs(s.value))[:n]
        if self.kind == "multiplier":
            return sorted(self.settings, key=lambda s: -abs(s.value / (self.expected_by.get(s.key) or self.expected or 1.0) - 1.0))[:n]
        return sorted(self.settings, key=lambda s: -abs(math.log(max(s.value, 1e-300))))[:n]


@dataclass
class Calibration:
    quantities: list[Quantity] = field(default_factory=list)
    #: Why something that looked like an interval or a standard error was not checked (one line each).
    not_checked: list[str] = field(default_factory=list)
    #: The ``inside`` consistency findings (:func:`_judge_inside`): said in the paper and the quality report only. Kept apart from
    #: ``quantities`` so no rule that reads which settings were judged, or whether anything was checked, counts them.
    inside: list[Quantity] = field(default_factory=list)

    @property
    def faults(self) -> list[Quantity]:
        return [q for q in self.quantities if q.fault]

    @property
    def checked(self) -> bool:
        return any(q.judged for q in self.quantities)


def _series(cell: dict[str, Any], name: str) -> dict[int, float]:
    """``{trial: value}`` of one value of a setting (finite numbers only)."""
    m = (cell.get("metrics") or {}).get(name)
    if not isinstance(m, dict):
        return {}
    vals, trials = m.get("values") or [], m.get("trials") or list(range(len(m.get("values") or [])))
    out = {}
    for t, v in zip(trials, vals):
        f = _number(v)
        if f is not None and isinstance(t, int):
            out[t] = f
    return out


def _z(level: float) -> float:
    """The two-sided normal critical value of a level (1.96 at 0.95)."""
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def _truth_of(truth: Any, cell: dict[str, Any], thresholds: dict[str, Any]) -> float | None:
    """The declared true value in one setting: the number itself, a threshold's value, or the setting's own value of a grid
    parameter (``None`` when it is not a number there)."""
    if not isinstance(truth, str):
        return _number(truth)
    if truth in thresholds:
        return _number(thresholds[truth])
    here = _number((cell.get("cell") or {}).get(truth))
    return here if here is not None else _metric_spec.fixed_value(truth, thresholds)


#: What the record says when the plan declares nothing FI can measure the reported uncertainty against.
NO_DECLARATION = ("the plan declares no metric with a known true value (`known_value`: the true value, the names run_trial "
                  "returns for the estimate and its standard error or interval, and the interval's level), so FI could "
                  "not measure the reported intervals and standard errors against the truth")


def _z_family(k: int) -> float:
    """The critical value of each of ``k`` settings' intervals when, together, a right code shows a gross miss by sampling
    alone with chance :data:`FAMILY_ALPHA` at most."""
    return NormalDist().inv_cdf(1.0 - FAMILY_ALPHA / (2.0 * max(1, k)))


def _judge_multiplier(q: "Quantity", per: list[tuple[str, list[tuple[float, float]]]]) -> None:
    """Each setting's recovered multiplier of the standard error against the plan's (``q.expected``): ``value`` is the median
    multiplier of the setting's trials, ``low`` / ``high`` the smallest and largest. A setting misses when any trial's multiplier
    lies farther from the plan's than :data:`MULTIPLIER_TOLERANCE` plus that trial's own rounding allowance (so a half-width
    that is not a multiplier of the trial's own standard error misses too). A setting with fewer than :data:`MIN_TRIALS` trials
    is not judged."""
    for key, pairs in per:
        want = q.expected_by.get(key) or q.expected or 0.0
        mults = [m for m, _a in pairs]
        if len(mults) < MIN_TRIALS or want <= 0:
            continue
        off = max(abs(m / want - 1.0) - a for m, a in pairs)
        ordered = sorted(mults)
        mid = len(ordered) // 2
        med = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        lo, hi = ordered[0], ordered[-1]
        q.settings.append(SettingCheck(key, len(ordered), med, lo, hi, gross=off > MULTIPLIER_TOLERANCE, below=False))


def _quantile_of(df: Any, level: float) -> float:
    return _stats.t_quantile(0.5 + level / 2.0, math.inf if df == "normal" else df)


def _multiplier_quantity(kv: dict[str, Any], df: Any, cells: list[dict[str, Any]], spec: dict[str, Any]) -> "Quantity":
    """The multiplier of one declared interval, judged at every setting against the plan's: the one multiplier the plan fixes,
    or, where it gives ``degrees_of_freedom`` per value of a factor (``{factor: {value: df}}``), the multiplier of the value the
    setting ran (a value the plan does not list is not judged, and is named in ``unlisted`` so the record says so)."""
    level = kv["level"]
    by_value = df if isinstance(df, dict) else None
    mult = Quantity("multiplier", " .. ".join(kv["interval"]), [*kv["interval"], kv["standard_error"]], level=level,
                    expected=None if by_value else _quantile_of(df, level), df=df)
    per = []
    for c in cells:
        if _undefined_here(spec.get("undefined_when") or {}, c):
            continue
        key, want = str(c.get("key")), mult.expected
        if by_value:
            (axis, table), = by_value.items()
            here = table.get(str((c.get("cell") or {}).get(axis)))
            if here is None:
                shown = (c.get("cell") or {}).get(axis)
                if shown is not None and str(shown) not in mult.unlisted and _multipliers(c, kv, 1.0):
                    mult.unlisted.append(str(shown))
                continue
            want = mult.expected_by[key] = _quantile_of(here, level)
        per.append((key, _multipliers(c, kv, want)))
        if _interval_form(c, kv) == LOG and per[-1][1]:
            mult.on_log.append(key)
    if by_value:
        mult.expected = max(mult.expected_by.values(), default=None)
    _judge_multiplier(mult, per)
    return mult


def _compared_values_off_the_plan(q: "Quantity", cells: list[dict[str, Any]], ident: str) -> str:
    """For a plan that fixes ONE multiplier while its own study compares values inside each run (``in_each_run``): when the
    record shows that some values of the comparison kept the plan's multiplier in every setting and the others used one other
    multiplier, the same in every trial of every setting (a method of the comparison with its own multiplier), the plan's single
    declaration does not cover its own comparison. FI does not send the code back or stop for it: those values are taken out of
    the multiplier check (their ``SettingCheck`` is dropped) and the returned sentence says so, with the multiplier measured.
    Nothing is taken out unless a value of the comparison matches the plan everywhere, every setting of the others misses it
    alike, and no setting has trials of different multipliers: a miss in some settings of a value only, or a multiplier that
    changes from trial to trial, still faults. A plan that says the multiplier per value (``degrees_of_freedom`` as a mapping)
    is held to it exactly instead. Returns ``""`` when nothing was taken out."""
    level_of = {str(c.get("key")): json.dumps(c["level"], sort_keys=True, default=str) for c in cells
                if isinstance(c.get("level"), dict) and c["level"]}
    exp = q.expected or 0.0
    z_or_t = [_quantile_of(d, q.level) for d in (*range(1, 201), "normal")] if q.level else []
    groups: dict[str, list[SettingCheck]] = {}
    for st in q.settings:
        g = level_of.get(st.key)
        if g is None:
            if st.gross:
                return ""
            continue
        groups.setdefault(g, []).append(st)
    off = {g: ss for g, ss in groups.items() if any(x.gross for x in ss)}
    if not off or len(off) == len(groups):
        return ""
    used = []
    for ss in off.values():
        meds = [x.value for x in ss]
        if (not all(x.gross for x in ss) or any(x.high - x.low > 0.05 * exp for x in ss)
                or max(meds) / min(meds) - 1.0 > MULTIPLIER_TOLERANCE):
            return ""
        used.append(sum(meds) / len(meds))
        if not any(abs(used[-1] / c - 1.0) <= MULTIPLIER_TOLERANCE for c in z_or_t):
            return ""  # a constant that is no t or normal quantile of the plan's level is a wrong multiplier, not a method's
    names = ", ".join("`" + ", ".join(f"{a}={v}" for a, v in json.loads(g).items()) + "`" for g in list(off)[:4])
    q.settings = [st for st in q.settings if level_of.get(st.key) not in off]
    return (f"`{q.label}` ({ident}): the plan fixes one multiplier, {exp:.4g}, for the interval, and the study compares "
            f"{names} inside each run; there the interval's multiplier was {min(used):.4g}"
            + (f" to {max(used):.4g}" if max(used) - min(used) > 0.01 else "")
            + " in every setting, another from the plan's, while the other compared values kept the plan's. FI did not hold "
              "those values to the plan's multiplier (to hold each compared value to its own, the plan gives "
              "`degrees_of_freedom` per value)")


def _grain(values: Iterable[float]) -> float:
    """The unit a series of values was rounded to, read from the values themselves: ``10**-k`` for the smallest ``k`` from 0 to
    :data:`GRAIN_MAX_DECIMALS` at which every value is a multiple of it, or 0.0 when none is (the values are not rounded)."""
    vals = [v for v in values if math.isfinite(v)]
    for k in range(GRAIN_MAX_DECIMALS + 1):
        scale = 10.0 ** k
        if all(abs(v * scale - round(v * scale)) <= 1e-9 + _GRAIN_EPS * abs(v * scale) for v in vals):
            return 10.0 ** -k
    return 0.0


#: How exactly an interval must be symmetric (relative to its width) to be read as symmetric on a scale.
SYMMETRY_TOLERANCE = 1e-6
OWN, LOG = "own", "log"


def _slack(grain: float, value: float, exact32: bool) -> float:
    """Half the unit a value was rounded to: its decimal :func:`_grain`, or, for values that are all exactly single-precision
    numbers (none is a multiple of a decimal unit), the relative spacing of that precision."""
    return grain / 2.0 if grain > 0 else (abs(value) * 2.0 ** -24 if exact32 else 0.0)


def _is_float32(values: Iterable[float]) -> bool:
    try:
        return all(struct.unpack("f", struct.pack("f", v))[0] == v for v in values)
    except (OverflowError, struct.error):
        return False


def _interval_form(cell: dict[str, Any], kv: dict[str, Any]) -> str:
    """The scale a setting's intervals are symmetric on, measured from its trials' own records: ``"log"`` when every trial
    that reports an estimate, both ends and a standard error (all positive) has an interval symmetric on the log scale,
    ``ln(upper / estimate) = ln(estimate / lower)``, within :data:`SYMMETRY_TOLERANCE`, and not every one is also symmetric
    on the estimate's own scale; ``"own"`` otherwise (every trial symmetric there, or trials that fit neither form, or trials
    of mixed forms: today's reading, the fail-closed one)."""
    lo, hi = kv["interval"]
    est, se = _series(cell, kv["estimate"]), _series(cell, kv["standard_error"])
    low, high = _series(cell, lo), _series(cell, hi)
    own_all, log_all, n = True, True, 0
    grains = [_grain(x.values()) for x in (est, low, high)]
    f32 = [g == 0.0 and _is_float32(x.values()) for g, x in zip(grains, (est, low, high))]
    for i in se:
        if not (i in low and i in high and i in est and se[i] > 0):
            continue
        e, l, u = est[i], low[i], high[i]
        n += 1
        width = u - l
        # What the rounding of the three values (their unit, or single precision) can move each test by is allowed for.
        se_e, se_l, se_u = (_slack(g, v, f) for g, v, f in zip(grains, (e, l, u), f32))
        own = width > 0 and abs((u - e) - (e - l)) <= SYMMETRY_TOLERANCE * width + se_l + se_u + 2.0 * se_e
        log = (e > 0 and l > 0 and u > 0 and u > l
               and abs(math.log(u / e) - math.log(e / l))
               <= SYMMETRY_TOLERANCE * math.log(u / l) + se_l / l + se_u / u + 2.0 * se_e / e)
        own_all, log_all = own_all and own, log_all and log
    return LOG if n and log_all and not own_all else OWN


def _multipliers(cell: dict[str, Any], kv: dict[str, Any], want: float) -> list[tuple[float, float]]:
    """``(multiplier, allowance)`` for each trial of one setting that reports the interval and a positive standard error: the
    multiplier of the standard error in the interval, and the share of ``want`` (the plan's multiplier) that the rounding of
    the values it is read from can move it by: each series' own unit (:func:`_grain`) is half an unit of error per value.
    On the estimate's own scale it is ``(upper - lower) / (2 * standard error)``. For intervals symmetric on the log scale
    (:func:`_interval_form`) the standard error the trial holds is the delta-method one of the estimate, so the standard error
    of its logarithm is ``se / estimate`` and the multiplier is ``ln(upper / lower) / (2 * se / estimate)``."""
    lo, hi = kv["interval"]
    se = _series(cell, kv["standard_error"])
    low, high = _series(cell, lo), _series(cell, hi)
    est = _series(cell, kv["estimate"])
    on_log = _interval_form(cell, kv) == LOG
    g_lo, g_hi, g_se, g_est = (_grain(low.values()), _grain(high.values()), _grain(se.values()),
                               _grain(est.values()))
    out = []
    for i in sorted(se):
        # The standard error the code had may be as small as the value it returned less half its unit: that bounds the error
        # of the multiplier. A value rounded down to the unit itself says too little to judge that trial.
        smallest = se[i] - g_se / 2.0
        if i in low and i in high and se[i] > 0 and smallest > 0:
            if on_log:
                if i not in est:
                    continue  # no estimate in this trial: the setting is judged on the trials that have one
                e, l, u = est[i], low[i], high[i]
                s = se[i] / e
                q = math.log(u / l) / (2.0 * s)
                rounding = ((g_hi / u + g_lo / l) / (4.0 * s) + abs(q) * (g_se / 2.0 / smallest + g_est / 2.0 / e))
            else:
                q = (high[i] - low[i]) / (2.0 * se[i])
                rounding = ((g_lo + g_hi) / 4.0 + abs(q) * g_se / 2.0) / smallest
            out.append((q, rounding / want))
    return out


def _coverage_check(key: str, hits: list[float], level: float, z: float = _Z95,
                    weights: list[float] | None = None) -> SettingCheck | None:
    """``weights``: how much each trial counts (1 over the draw's records, :func:`_draw_weights`; ``None``: each counts 1),
    so a draw that several settings share is counted once."""
    n = len(hits)
    if n < MIN_TRIALS:
        return None
    w = weights if weights is not None and len(weights) == n else [1.0] * n
    k, total = sum(h * x for h, x in zip(hits, w)), sum(w)
    ci = _stats.wilson_interval(k, total, z=z)
    if ci is None or total <= 0:
        return None
    lo, hi = ci
    return SettingCheck(key, max(1, round(total)), k / total, lo, hi, gross=hi < level - COVERAGE_GAP, below=hi < level,
                        above=lo > level)


def _ratio_of(est: list[float], se: list[float], constant_ok: bool = False) -> tuple[float, int] | None:
    """``(spread of the estimates / root mean square of their errors, degrees of freedom)``, or ``None`` when there is no
    spread to judge (a setting without noise; with ``constant_ok``, estimates that do not vary against an error the trials
    report read 0) or no error reported."""
    n = len(est)
    if n < MIN_TRIALS:
        return None
    mean = sum(est) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in est) / (n - 1))
    rms = math.sqrt(sum(s * s for s in se) / n)
    floor = 1e-12 * max(abs(mean), 1e-300)
    if rms <= floor:
        return None
    if sd <= floor:
        return (0.0, n - 1) if constant_ok else None
    return sd / rms, n - 1


def _ratio_band(key: str, n: int, ratio: float, free: float, z: float = _Z95) -> SettingCheck:
    half = z / math.sqrt(2.0 * free)
    lo, hi = ratio * math.exp(-half), ratio * math.exp(half)
    return SettingCheck(key, n, ratio, lo, hi, gross=lo > RATIO_HIGH or hi < RATIO_LOW, below=lo > 1.25 or hi < 0.8)


#: The key of the check over every setting together.
POOLED = "all settings together"


def _bias_band(key: str, n: float, b: float, z: float = _Z95) -> SettingCheck:
    half = z / math.sqrt(n)
    lo, hi = b - half, b + half
    return SettingCheck(key, max(1, round(n)), b, lo, hi, gross=lo > BIAS_GROSS or hi < -BIAS_GROSS,
                        below=lo > BIAS_REPORTED or hi < -BIAS_REPORTED)


def _judge_bias(q: "Quantity", per: list[tuple[Any, ...]]) -> None:
    """Each setting's (mean estimate - truth) / spread of the estimates at the family level, and pooled at 95%. A setting
    with no spread (no noise) is not judged. ``per``: ``(key, estimates, truth[, share])``; ``share`` is the part of the
    setting's trials that are draws no other judged setting ran (1 when none shares its draws)."""
    found = []
    for key, est, truth, *rest in per:
        n = len(est)
        if n < MIN_TRIALS:
            continue
        mean = sum(est) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in est) / (n - 1))
        if sd <= 1e-12 * max(abs(mean), 1e-300):
            continue
        found.append((key, n, (mean - truth) / sd, rest[0] if rest else 1.0))
    z = _z_family(len(found))
    for key, n, b, _share in found:
        q.settings.append(_bias_band(key, n, b, z))
    if len(found) > 1:
        total = sum(n * sh for _k, n, _b, sh in found)
        q.pooled = _bias_band(POOLED, total, sum(n * sh * b for _k, n, b, sh in found) / total)


def _inside_share(cell: dict[str, Any], est: str, ends: list[str]) -> tuple[int, int]:
    """``(trials, trials whose estimate lies inside their own interval)`` of one setting, over the trials that gave the estimate and both ends."""
    e, lo, hi = _series(cell, est), _series(cell, ends[0]), _series(cell, ends[1])
    used = [t for t in e if t in lo and t in hi]
    return len(used), sum(1 for t in used if lo[t] <= e[t] <= hi[t])


def _judge_inside(q: "Quantity", per: list[tuple[str, int, int]]) -> None:
    """Each setting's share of trials whose estimate lies inside their own interval; one below :data:`INSIDE_MIN` is ``below``
    (never ``gross``: it is a sign that the rows of the record may be misplaced, which the paper says, not a fault FI can repair)."""
    for key, n, inside in per:
        if n >= MIN_TRIALS:
            share = inside / n
            q.settings.append(SettingCheck(key, n, share, share, share, gross=False, below=share < INSIDE_MIN))


def _judge_no_estimate(q: "Quantity", per: list[tuple[str, int, int]]) -> None:
    """Each setting's share of the trials that ran to the end and gave no number for the estimate of a declared truth: more
    than :data:`FAULT_FRACTION` is a gross miss (they are misses of the truth, not values that have no meaning there)."""
    for key, ran, none in per:
        if ran and none:
            share = none / ran
            q.settings.append(SettingCheck(key, ran, share, share, share, gross=share > FAULT_FRACTION, below=True))


def _judge_coverage(q: "Quantity", per: list[tuple[Any, ...]]) -> None:
    """Each setting's coverage at the family level, and the pooled coverage at 95% (``per``: ``(key, hits[, weights])``;
    with weights, the pool counts each random draw once)."""
    judged = [(key, hits, *rest) for key, hits, *rest in per if len(hits) >= MIN_TRIALS]
    z = _z_family(len(judged))
    for key, hits, *_w in judged:
        got = _coverage_check(key, hits, q.level or 0.95, z)
        if got is not None:
            q.settings.append(got)
    if len(judged) > 1:
        weights = [x for _k, hits, *w in judged for x in (w[0] if w and w[0] is not None else [1.0] * len(hits))]
        q.pooled = _coverage_check(POOLED, [h for _k, hits, *_w in judged for h in hits], q.level or 0.95, weights=weights)


def ratio_detection_floor(free: float, z: float = _Z95) -> float:
    """The smallest ratio of (spread of the estimates / reported standard error) that :func:`_ratio_band` would call a gross
    miss on ``free`` degrees of freedom at critical value ``z``: the whole sampling interval of the ratio must lie above
    :data:`RATIO_HIGH`, so the ratio itself must exceed ``RATIO_HIGH * exp(z / sqrt(2 * free))``. A reported standard error too
    small by less than this passes the check: the power limit of what the trials can show."""
    return RATIO_HIGH * math.exp(z / math.sqrt(2.0 * max(1, free)))


def _judge_ratio(q: "Quantity", per: list[tuple[Any, ...]]) -> None:
    """Each setting's spread / reported error at the family level, and the pooled ratio at 95% (the root of the mean squared
    ratio weighted by each setting's degrees of freedom; ``per``: ``(key, estimates, errors[, share])``, ``share`` as in
    :func:`_judge_bias`: the pool's degrees of freedom count each random draw once)."""
    found = [(key, len(e), _ratio_of(e, s), rest[0] if rest else 1.0) for key, e, s, *rest in per]
    found = [(key, n, r, sh) for key, n, r, sh in found if r is not None]
    if not found:
        # No setting shows any spread while the trials report an error: the estimates never vary, anywhere (a simulation
        # that ignores its seed). One setting without noise among settings that have some is only a setting without noise
        # (a correct estimator reports a small error there): it is skipped as before.
        found = [(key, len(e), r, rest[0] if rest else 1.0) for key, e, s, *rest in per
                 if (r := _ratio_of(e, s, constant_ok=True)) is not None]
    z = _z_family(len(found))
    for key, n, (ratio, free), _sh in found:
        q.settings.append(_ratio_band(key, n, ratio, free, z))
    # The power of this run's check: the best any one of its judgements could do (the pool has the most degrees of freedom).
    floors = [ratio_detection_floor(r[1], z) for _k, _n, r, _sh in found]
    if len(found) > 1:
        free = sum(r[1] * sh for _k, _n, r, sh in found)
        pooled = math.sqrt(sum(r[1] * sh * r[0] ** 2 for _k, _n, r, sh in found) / free)
        q.pooled = _ratio_band(POOLED, round(sum(n * sh for _k, n, _r, sh in found)), pooled, free)
        floors.append(ratio_detection_floor(max(1.0, free)))
    if floors:
        q.detects_from = min(floors)


_OWN = "_own_metrics"  # in a record :func:`_with_main` completed: the names it held itself


def _own_here(cell: dict[str, Any], names: Iterable[Any]) -> bool:
    """Whether ``cell`` holds any of ``names`` itself (a record :func:`_with_main` did not complete holds all it has)."""
    own = cell.get(_OWN)
    return own is None or any(n in own for n in names)


def _with_main(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``cells`` where each record kept for a value a run compares inside it (``level``) also holds what its setting's own record
    holds under the names it lacks: the record keeps the estimate with the setting and the standard error and interval ends
    with each value compared, and the two are one setting's numbers for each trial. Nothing else changes."""
    mains = {json.dumps(c.get("cell") or {}, sort_keys=True, default=str): c for c in cells if not c.get("level")}
    out = []
    for c in cells:
        level = c.get("level")
        main = mains.get(json.dumps({a: v for a, v in (c.get("cell") or {}).items() if a not in (level or {})}, sort_keys=True,
                                    default=str)) if isinstance(level, dict) else None
        out.append({**c, "metrics": {**(main.get("metrics") or {}), **(c.get("metrics") or {})}, _OWN: set(c.get("metrics") or {})}
                   if main is not None else c)
    return out


def _cell_hits(cell: dict[str, Any], kv: dict[str, Any], truth: float) -> tuple[dict[int, float], dict[int, float], list[float]]:
    """``(estimates, standard errors, hits)`` of one setting for a declared known value: each trial's estimate and standard
    error by trial, and 1/0 per trial for whether the interval the trial reports (the declared two ends) covers ``truth``.
    A trial that reports only a standard error makes no coverage claim FI could judge: FI does not build a z interval of
    its own (a fit with few degrees of freedom would under-cover it by design), so ``hits`` is empty."""
    spread = kv.get("standard_error") or kv.get("variance")
    e = _series(cell, kv["estimate"])
    s = _series(cell, spread) if spread else {}
    if spread and "standard_error" not in kv:
        s = {i: math.sqrt(v) for i, v in s.items() if v >= 0}
    ends = kv.get("interval")
    # Every trial that ran to the end is judged: one with no number for the estimate or its interval misses the truth (a
    # repair that returns NaN in place of a failing fit must not drop those trials from the count).
    # ... in a record that returns these values at all (a value compared inside a run may be kept in its own records only).
    held = cell.get("metrics") or {}
    if not ends or not all(n in held for n in [kv["estimate"], *ends]):
        return e, s, []
    return e, s, _table_hits(held, ends, truth) or []


def _draw_weights(cells: list[dict[str, Any]], protocol: dict[str, Any] | None, quest_root: Path | str | None,
                  held: dict[str, Any]) -> dict[str, dict[int, float]] | None:
    """``{setting key: {trial: weight}}`` where each random draw counts once over ``cells``: a trial's weight is 1 over the
    number of the settings' trials that ran on its draw. ``None`` when no two settings ran a trial on the same seed (FI's
    ledger). The decision and the draw key are the paper's own statistics' (core/paper_values.py ``Records.share_draws`` and
    ``_draw_key``): one rule, so a measure pooled over settings that share draws is judged on the different draws."""
    if quest_root is None or len(cells) < 2:
        return None
    from . import paper_values as _pv

    if "rec" not in held:  # the ledger is read once per check
        held["rec"] = _pv.Records([], protocol)
        held["rec"].seeds = _pv._ledger_seeds(Path(quest_root))
    rec = held["rec"]
    if not rec.seeds or not rec.share_draws(cells):
        return None
    drawn: dict[Any, list[tuple[str, int]]] = {}
    for c in cells:
        ids = sorted(_ran(c))
        for i, t in enumerate(ids):
            drawn.setdefault(_pv._draw_key(rec, c, ids, i), []).append((str(c.get("key")), t))
    out: dict[str, dict[int, float]] = {}
    for users in drawn.values():
        for key, t in users:
            out.setdefault(key, {})[t] = 1.0 / len(users)
    return out


def _share(weights: dict[int, float] | None, trials: Any) -> float:
    """The part of ``trials`` that count as their own draw: the mean weight of those trials (1 without shared draws)."""
    ids = list(trials)
    if weights is None or not ids:
        return 1.0
    return sum(weights.get(t, 1.0) for t in ids) / len(ids)


def calibration(cells: list[dict[str, Any]], protocol: dict[str, Any] | None = None,
                thresholds: dict[str, Any] | None = None, quest_root: Path | str | None = None) -> Calibration:
    """What the trials' intervals and standard errors say against the true values the plan declares (see the module's
    notes). Only the protocol's ``known_value`` declarations are read."""
    out = Calibration()
    cells = _with_main(cells)
    protocol = protocol if isinstance(protocol, dict) else {}
    # What the plan fixes by name: its constants and thresholds (the true value of a study is among them).
    thresholds = _metric_spec.fixed_numbers(protocol, thresholds)
    axes = {str(k) for part in ("grid", "in_each_run") if isinstance(protocol.get(part), dict) for k in protocol[part]}
    known = [s for s in _metric_spec.declared(protocol) if s.get("known_value")]
    if not known:
        out.not_checked.append(NO_DECLARATION)
        return out
    ledger: dict[str, Any] = {}
    returned = {n for c in cells for n in (c.get("metrics") or {})}
    if not returned:
        out.not_checked.append("no setting has trials to judge, so the calibration was not checked")
        return out
    for spec in known:
        kv, ident = spec["known_value"], spec["id"]
        truth, level, est = kv["true_value"], kv["level"], kv["estimate"]
        df = kv.get("degrees_of_freedom")
        if df is not None and all(n in returned for n in [kv["standard_error"], *kv["interval"]]):
            # Needs the standard error and the interval only, not the true value: judged even when that names nothing.
            mult = _multiplier_quantity(kv, df, cells, spec)
            if not isinstance(df, dict) and (note := _compared_values_off_the_plan(mult, cells, ident)):
                out.not_checked.append(note)
            if mult.unlisted:
                out.not_checked.append(f"`{mult.label}` ({ident}): the plan's `degrees_of_freedom` does not name "
                                       + ", ".join(f"`{u}`" for u in mult.unlisted[:4])
                                       + ", so the multiplier of the interval there was not checked")
            if mult.judged:
                out.quantities.append(mult)
            elif not mult.unlisted:
                out.not_checked.append(f"`{mult.label}` ({ident}): no setting had {MIN_TRIALS} or more trials with an "
                                       "interval and a standard error, too few to judge the multiplier of the standard error "
                                       "in the interval")
        if isinstance(truth, str) and truth not in axes and _metric_spec.fixed_value(truth, thresholds) is None:
            out.not_checked.append(f"`{ident}`: its true value `{truth}` is neither a threshold, a constant nor a grid "
                                   "parameter of the plan")
            continue
        fixed_truth = _metric_spec.fixed_value(truth, thresholds)
        truth_label = (f"the plan's {truth} = {float(truth):g}" if not isinstance(truth, str) else
                       f"the plan's {truth}" + (f" = {fixed_truth:g}" if fixed_truth is not None else " (per setting)"))
        spread = kv.get("standard_error") or kv.get("variance")
        ends = kv.get("interval")
        missing = [n for n in [est, *([spread] if spread else []), *(ends or [])] if n not in returned]
        if missing:
            out.quantities.append(Quantity("missing", ident, missing, level=level, truth=truth_label))
            continue
        if ends:  # an estimate inside its own trial's interval: the rows of the record belong together (needs no true value)
            inside = Quantity("inside", est, [est, *ends], truth=truth_label)
            _judge_inside(inside, [(str(c.get("key")), *_inside_share(c, est, list(ends))) for c in cells
                                   if _own_here(c, [est, *ends]) and not c.get("level")])
            if inside.below:  # reported only where it is off: a record whose rows agree adds nothing
                out.inside.append(inside)
        cover = (Quantity("coverage", " .. ".join(ends), list(ends), level=level, truth=truth_label,
                          withheld=list(ends)) if ends else None)
        ratio = (Quantity("se_ratio", est, [est, spread], truth=truth_label, withheld=[spread, *(ends or [])])
                 if spread else None)
        bias = Quantity("bias", est, [est], truth=truth_label)
        empty = Quantity("no_estimate", est, [est], truth=truth_label)
        cover_per: list[tuple[Any, ...]] = []
        ratio_per: list[tuple[Any, ...]] = []
        bias_per: list[tuple[Any, ...]] = []
        empty_per: list[tuple[str, int, int]] = []
        mine = [c for c in cells if _own_here(c, [est, spread, *(ends or [])])]  # a completed record counts for what it holds itself
        judged_cells = [c for c in mine if (t0 := _truth_of(truth, c, thresholds)) is not None
                        and not _undefined_here(spec.get("undefined_when") or {}, c) and est in (c.get("metrics") or {})]
        draw_weights = _draw_weights(judged_cells, protocol, quest_root, ledger)
        for c in mine:
            key = str(c.get("key"))
            t = _truth_of(truth, c, thresholds)
            if t is None or _undefined_here(spec.get("undefined_when") or {}, c) or est not in (c.get("metrics") or {}):
                continue
            e, s, hits = _cell_hits(c, kv, t)
            w = (draw_weights or {}).get(key)
            ran = _ran(c)
            cover_per.append((key, hits, [w.get(i, 1.0) for i in sorted(ran)] if w and hits else None))
            if _own_here(c, [est]):  # the estimate is judged where the record keeps it, not again with each value compared
                bias_per.append((key, list(e.values()), t, _share(w, e)))
                empty_per.append((key, len(ran), len(ran - set(e))))
            if ratio is not None:
                used = [i for i in e if i in s and s[i] >= 0]
                ratio_per.append((key, [e[i] for i in used], [s[i] for i in used], _share(w, used)))
        if cover is not None:
            _judge_coverage(cover, cover_per)
        if ratio is not None:
            _judge_ratio(ratio, ratio_per)
        _judge_bias(bias, bias_per)
        _judge_no_estimate(empty, empty_per)
        if empty.settings:
            out.quantities.append(empty)
        for q in (cover, ratio, bias):
            if q is None:
                continue
            if q.judged:
                out.quantities.append(q)
            else:
                out.not_checked.append(f"`{q.label}` ({ident}): no setting had {MIN_TRIALS} or more trials with these "
                                       "values (and some spread of the estimate), too few to judge")
    if not out.quantities and not out.not_checked:
        out.not_checked.append(NO_DECLARATION)
    return out


#: How far a coverage the analysis reports may lie from FI's own count of the same trials (an analysis that averages the
#: settings' coverages instead of pooling their trials, or drops a trial FI kept, differs by a trial or so).
RECOUNT_TOLERANCE = 0.02


def _setting_of(key: str, cell: dict[str, Any]) -> bool:
    """Whether a stratum key of the analysis's report (``"noise=0.01"``, ``"noise=0.01,n=12"``) names this setting: every
    axis it names has this setting's value (the same number, or the same text)."""
    from .run_manifest import _parse_cell

    parsed = _parse_cell(key)
    here = cell.get("cell") or {}
    if not parsed or not set(parsed) <= set(here):
        return False
    for axis, raw in parsed.items():
        a, b = _number(here[axis]), None
        try:
            b = float(raw)
        except ValueError:
            pass
        if not ((a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
                or str(here[axis]).strip() == raw.strip()):
            return False
    return True


def reported_coverage_mismatches(cells: list[dict[str, Any]], protocol: dict[str, Any] | None, result_json: Any,
                                 thresholds: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Where the analysis reports the coverage of a declared known value and FI, counting the same trials, gets another
    number. A ``proportion`` metric with a ``known_value`` IS that coverage: its value in the report (the number under its
    id, or ``<id>_count`` / ``<id>_total``), at the top level (every setting together) or in a stratum keyed like
    ``noise=0.01`` (the settings it names), is compared with FI's share of those settings' trials whose interval covers the
    plan's true value. Each mismatch: ``{"id", "where", "reported", "fi", "n"}``. A value that is not a share (outside 0
    to 1) or a stratum FI cannot place is not judged."""
    protocol = protocol if isinstance(protocol, dict) else {}
    # What the plan fixes by name: its constants and thresholds (the true value of a study is among them).
    thresholds = _metric_spec.fixed_numbers(protocol, thresholds)
    if not isinstance(result_json, dict):
        return []
    walked = _metric_spec._walk(result_json)
    out: list[dict[str, Any]] = []
    for spec in _metric_spec.declared(protocol):
        kv = spec.get("known_value")
        if not kv or spec["kind"] != "proportion":
            continue
        ident = spec["id"]
        places: dict[tuple[str, ...], float] = {}
        for path, value in walked.items():
            leaf, parent = path[-1], path[:-1]
            if leaf == ident and _number(value) is not None:
                places[parent] = float(value)
            elif leaf == f"{ident}_count" and _number(value) is not None:
                total = _number(walked.get(parent + (f"{ident}_total",)))
                if total:
                    places.setdefault(parent, float(value) / total)
        for parent, reported in sorted(places.items()):
            if not 0.0 <= reported <= 1.0:
                continue
            stratum = parent[-1] if parent else ""
            chosen = [c for c in cells if not c.get("level") and (not parent or _setting_of(stratum, c))]
            if parent and not chosen:
                continue
            hits: list[float] = []
            for c in chosen:
                truth = _truth_of(kv["true_value"], c, thresholds)
                if truth is not None:
                    hits.extend(_cell_hits(c, kv, truth)[2])
            if len(hits) < MIN_TRIALS:
                continue
            fi = sum(hits) / len(hits)
            if abs(fi - reported) > max(RECOUNT_TOLERANCE, 1.5 / len(hits)):
                out.append({"id": ident, "where": "/".join(parent) or "the whole study", "reported": reported,
                            "fi": fi, "n": len(hits)})
    return out


def calibration_directive(faults: list[Quantity]) -> str:
    """The plain message the simulation's repair is given for a reported uncertainty that is grossly wrong. The numbers are
    FI's measurements on the run; no expected value of the study is in it."""
    lines = []
    for q in faults[:6]:
        worst = "; ".join(f"{s.key}: {s.value:.3g} (n={s.n})" for s in q.worst()
                          + ([q.pooled] if q.pooled is not None and q.pooled.gross else []))
        lines.append(f"- {q.says()}." + (f" Worst settings: {worst}" if worst else
                                         " Return them from run_trial under exactly these names."))
    plan = any(q.kind == "multiplier" for q in faults)
    head = ("FI measured the intervals and standard errors the simulation reports against the true values it was given, "
            "over the trials of each setting, and found them " + ("wrong" if plan else "grossly wrong") + ":\n")
    tail = ("A reported standard error must be about the spread of the estimate over repeated trials, and a 95% "
            "interval must cover the true value in about 95% of them. Find what makes them disagree in the code (a noise "
            "level or a weight that is fixed instead of computed from the trial's own data, a variance on the wrong scale, "
            "a wrong critical value) and fix it. Do not change the true values, the settings or the number of trials.")
    if plan:
        tail = ("The plan fixes the multiplier of the standard error that each `the multiplier of the standard error` line above "
                "names; FI recovered the multiplier the code used from the interval and the standard error the trials return. "
                "Make every place that forms that interval use the plan's multiplier (check the arguments each call passes, "
                "such as the degrees of freedom, not only the function that computes the quantile), and return the interval "
                "and the standard error under exactly these names. Where a line above says the intervals are symmetric on the "
                "log scale, FI judged the multiplier on that scale: keep the interval symmetric there (do not make it "
                "symmetric on the estimate's own scale). For any other quantity above: " + tail)
    return head + "\n".join(lines) + "\n\n" + tail


# ---- what FI does with them --------------------------------------------------------------------------------------------

OK, REPAIR, STOP, LEAVE_OUT, DISCLOSE = "ok", "repair", "stop", "leave_out", "disclose"
#: The kinds of reported uncertainty whose disagreement with the trials, once the repairs are spent, is a measured property
#: of the estimator rather than a code fault FI can name, and only while the error is right over all settings together (the
#: pooled measure is inside its range): the quest goes on and the paper says, setting by setting, where it was measured. (``multiplier``: the code does not use the multiplier the plan fixes, a fault; ``missing``, ``bias`` and
#: ``no_estimate`` concern the estimate itself: those still stop.)
DISCLOSABLE = ("coverage", "se_ratio")


def of_settings(cell: dict[str, Any], keys: set[str]) -> bool:
    """Whether a summary record belongs to one of the settings ``keys``: the setting itself, or a record of a value its runs
    compared inside them (keyed ``<setting>,<value>``)."""
    key = str(cell.get("key") or "")
    return key in keys or bool(cell.get("level")) and any(key.startswith(k + ",") for k in keys if k)


@dataclass
class Verdict:
    """What FI does after the two checks of a run's trials: ``action`` is :data:`OK` (go on), :data:`REPAIR` (send the
    simulation back with ``directive``), :data:`STOP` (the repairs are spent and a fault stands: the quest stops, ``say`` /
    ``tried`` are its plain record) or :data:`LEAVE_OUT` (the repairs are spent and only some settings are faulty: their
    numbers are left out of what the analysis is given, ``left_out``) or :data:`DISCLOSE` (the repairs are spent and the only
    fault standing is a reported standard error or interval measured not to match the trials at some settings only (right over all
    of them together), at the settings the record's
    quantities mark ``gross``: the quest goes on, and the paper and the quality report say where)."""

    fails: list[SettingFailures]
    calibration: Calibration
    action: str = OK
    left_out: list[str] = field(default_factory=list)
    directive: str = ""
    say: str = ""
    tried: list[str] = field(default_factory=list)
    #: The fault is failed trials in some settings only (no calibration fault): once the repairs are spent those settings
    #: are left out, never a stop.
    partial: bool = False
    #: Values of the plan with no number in some trials of a setting (not failures: recorded, and said in the paper).
    undefined: list[Undefined] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {**summary(self.fails, self.calibration), "action": self.action, "left_out_settings": list(self.left_out),
                "undefined": [u.as_dict() for u in self.undefined], **({"say": self.say} if self.say else {})}


def judge(cells: list[dict[str, Any]], protocol: dict[str, Any] | None = None, thresholds: dict[str, Any] | None = None,
          reasons: dict[str, Counter] | None = None, *, repairs_left: bool, no_calibration: str = "",
          quest_root: Path | str | None = None, level_rule: str = "") -> Verdict:
    """Both checks on a run's per-setting summary, and what follows from them. A setting where most trials failed is judged
    by the failure check alone (its numbers rest on the trials that survived), so the calibration reads the other settings.
    While a repair is left, any fault is sent back; once they are spent, a calibration fault or a failure in more than half
    of the settings stops the quest, and failures in at most half of them leave those settings out; where the plan declares
    a true value, only when every remaining setting's calibration was judged against it (and passed): one that was not
    checked makes it a stop. With
    ``no_calibration`` (the reason), only the failed trials are checked and the record says the calibration was not."""
    fails = failures(cells, reasons, protocol)
    bad = [f for f in fails if f.fault]
    run = [f for f in fails if f.planned]
    every = bool(bad) and len(bad) == len(run)
    # More than half of the settings left out would leave too little to report: that is a stop, never a paper.
    most = bool(bad) and len(bad) * 2 > len(run)
    keys = {f.key for f in bad}
    cal = (Calibration(not_checked=[no_calibration]) if no_calibration else
           calibration([c for c in cells if not of_settings(c, keys)], protocol, thresholds, quest_root))
    out = Verdict(fails, cal, undefined=undefined(cells, protocol))
    if not bad and not cal.faults:
        return out
    if cal.faults or most or repairs_left:
        parts = ([failures_directive(bad, level_rule)] if bad else []) + ([calibration_directive(cal.faults)] if cal.faults else [])
        out.directive = "\n\n".join(parts)
        out.action = REPAIR if repairs_left else STOP
        # Only some settings (at most half) fail and nothing else is wrong: once the repairs are spent, those settings
        # are left out.
        out.partial = not cal.faults and not most
        if cal.faults:
            q = cal.faults[0]
            out.say = ("FI measured the intervals and standard errors the simulation reports against the true values it "
                       f"was given, and they are grossly wrong ({q.says()})") if q.kind != "multiplier" else (
                "FI recovered the multiplier of the standard error that the simulation's intervals use from the trials' own "
                f"records, and it is not the one the plan fixes ({q.says()})")
            out.tried = ["ran every trial of the study and measured the reported intervals and standard errors against "
                         "the true values the simulation was given, per setting",
                         *[q.says() for q in cal.faults[:2]]]
        else:
            many = "every setting" if every else f"{len(bad)} of {len(run)} settings"
            out.say = f"in {many} more than half of the trials failed ({bad[0].says()})"
            if most and not every:
                out.say += "; more than half of the settings would be left out, which leaves too little to report"
            out.tried = ["ran every trial of the study and counted the failed trials of each setting",
                         *[f.says() for f in bad[:2]]]
        # What stands is a reported uncertainty measured not to match the trials (and no failed trials): that is a result
        # about the estimator, kept for the settings where it was measured, not a reason to write no paper.
        # Only when the error is right over all the settings together: the check's own pooled measure, at the same range and
        # sampling-error rule as per setting, is inside its accepted range for every such quantity. A pooled measure that is
        # itself outside it (a fault everywhere, or a fault wide enough to move the whole) still stops.
        if (out.action == STOP and cal.faults and not bad
                and all(q.kind in DISCLOSABLE and q.pooled is not None and not q.pooled.gross for q in cal.faults)):
            out.action = DISCLOSE
        return out
    # Where the plan declares a true value, settings are left out only when what remains is shown sound: every remaining
    # setting's intervals and standard errors were judged against the truth (and passed, else there was a fault above).
    # "Not checked" is not a pass there: the failures may be the visible part of a wrong code whose other settings merely
    # did not crash. Without a declared truth there is nothing to judge the rest against, and the settings are left out
    # (and said in the paper); only more than half of them left out stops the quest (above).
    remaining = {f.key for f in run if f.key not in keys}
    judged = {s.key for q in cal.quantities for s in q.settings}
    unjudged = sorted(remaining - judged)
    declared = any(spec.get("known_value") for spec in _metric_spec.declared(protocol))
    if unjudged and declared and not no_calibration:  # a search never uses a failed design: nothing it leaves out is analysed
        out.action = STOP
        why = cal.not_checked[0] if not cal.checked and cal.not_checked else (
            f"{len(unjudged)} of the {len(remaining)} remaining settings had no interval or standard error FI could judge "
            f"({', '.join(unjudged[:4])}{', ...' if len(unjudged) > 4 else ''})")
        out.say = (f"in {len(bad)} of {len(run)} settings more than half of the trials failed ({bad[0].says()}); they could "
                   "be left out only if FI could show that the remaining settings' numbers are sound, and it could not: "
                   + why)
        out.tried = ["ran every trial of the study and counted the failed trials of each setting",
                     *[f.says() for f in bad[:2]],
                     "looked for a check of the remaining settings' intervals and standard errors against the true values: "
                     + why]
        return out
    out.action = LEAVE_OUT
    out.left_out = [f.key for f in bad]
    return out


# ---- the record --------------------------------------------------------------------------------------------------------


def summary(fails: list[SettingFailures], cal: Calibration | None) -> dict[str, Any]:
    """What the record keeps: the settings with failed trials, and each quantity the calibration check judged."""
    return {
        "failures": [{"key": s.key, "planned": s.planned, "failed": s.failed, "fraction": round(s.fraction, 4),
                      "fault": s.fault, "reasons": [[r, n] for r, n in s.reasons[:5]]} for s in fails if s.failed],
        "out_of_time": [{"key": s.key, "planned": s.planned, "trials": s.out_of_time} for s in fails if s.out_of_time],
        "settings": len(fails),
        "calibration": None if cal is None else {
            "quantities": [{"kind": q.kind, "label": q.label, "names": q.names, "level": q.level, "truth": q.truth,
                            "judged": q.judged, "gross": q.gross, "below": q.below, "above": q.above, "fault": q.fault,
                            **({"expected": round(q.expected, 6), "degrees_of_freedom": q.df}
                               if q.kind == "multiplier" and q.expected is not None else {}),
                            **({"judged_on_log_scale": len(q.on_log)} if q.on_log else {}),
                            **({"detects_from": round(q.detects_from, 4)}
                               if q.kind == "se_ratio" and q.detects_from is not None else {}),
                            "says": q.says(), "withheld": list(q.withheld),
                            "settings": [{"key": s.key, "n": s.n, "value": round(s.value, 6), "low": round(s.low, 6),
                                          "high": round(s.high, 6), "gross": s.gross, "below": s.below} for s in q.settings],
                            "pooled": None if q.pooled is None else {
                                "n": q.pooled.n, "value": round(q.pooled.value, 6), "low": round(q.pooled.low, 6),
                                "high": round(q.pooled.high, 6), "gross": q.pooled.gross}}
                           for q in cal.quantities],
            "not_checked": list(cal.not_checked),
            "checked": cal.checked,
            "inside": [{"kind": q.kind, "label": q.label, "names": q.names, "truth": q.truth, "judged": q.judged, "gross": 0,
                        "below": q.below, "above": 0, "fault": False, "says": q.says(), "withheld": [],
                        "settings": [{"key": s.key, "n": s.n, "value": round(s.value, 6), "low": round(s.low, 6),
                                      "high": round(s.high, 6), "gross": False, "below": s.below} for s in q.settings]}
                       for q in cal.inside],
        },
    }


def withheld_values(quest_root: Path | str) -> dict[str, set[str]]:
    """``{setting key: the trial columns not handed on there}`` once the last run's record is a disclosure (``DISCLOSE``): at
    each setting a reported standard error or interval was measured not to match, its error and interval columns (the
    estimate stays). The one source of what FI keeps from the analysis, the results table and the paper's numbers there; empty
    for any other action."""
    rec = read_record(quest_root)
    if rec.get("action") != DISCLOSE:
        return {}
    out: dict[str, set[str]] = {}
    for q in ((rec.get("calibration") or {}).get("quantities") or []):
        if not isinstance(q, dict) or not q.get("fault") or q.get("kind") not in DISCLOSABLE:
            continue
        for s in q.get("settings") or []:
            if isinstance(s, dict) and s.get("gross"):
                out.setdefault(str(s.get("key")), set()).update(str(n) for n in q.get("withheld") or [])
    return out


def without_withheld(cells: list[dict[str, Any]], withheld: dict[str, set[str]]) -> list[dict[str, Any]]:
    """``cells`` with, at each setting of ``withheld`` (and the records of values compared inside it), those columns taken
    out. The cells themselves are not changed."""
    if not withheld:
        return cells
    out = []
    for c in cells:
        gone: set[str] = set()
        for key, names in withheld.items():
            if of_settings(c, {key}):
                gone |= names
        out.append({**c, "metrics": {k: v for k, v in (c.get("metrics") or {}).items() if k not in gone}}
                   if gone and isinstance(c.get("metrics"), dict) else c)
    return out


def left_out_settings(quest_root: Path | str) -> list[dict[str, Any]]:
    """The settings whose numbers the last run left out (most of their trials failed and the repairs were spent), each with
    its failed and planned counts, from the record."""
    rec = read_record(quest_root)
    keys = set(rec.get("left_out_settings") or [])
    return [f for f in rec.get("failures") or [] if isinstance(f, dict) and f.get("key") in keys]


def write_record(quest_root: Path | str, record: dict[str, Any]) -> bool:
    """Write the record; whether it was written."""
    try:
        path = Path(quest_root) / RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, allow_nan=True), encoding="utf-8")
        return True
    except OSError:
        return False


#: Why FI's checks of the trials did not run on the last run, by how the simulation ran its trials (the record's
#: ``not_run``): FI checks only the trials it runs itself (``run_trial`` / ``run_cell``, or a search's designs).
NOT_RUN = {
    "one_script": "the code is one script that runs its own trials, so FI did not run them",
    "own_loop": "the simulation loops over its own trials (it defines no run_trial or run_cell for FI to call), so FI did "
                "not run them",
}


def record_not_run(quest_root: Path | str, mode: str) -> None:
    """Record that the last run's trials were not FI's to check (``mode`` is a key of :data:`NOT_RUN`): the paper's
    required items and the quality report say so, instead of saying nothing."""
    write_record(quest_root, {"not_run": mode})


def read_record(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
