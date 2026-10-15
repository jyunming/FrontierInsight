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
analysis is given and the paper says so; if every setting is so, the quest stops. A setting where more than
:data:`STATE_FRACTION` failed is recorded and the paper states it.

**Calibration against the truth (:func:`calibration`).** Read from the names the trials return, and only when they pair up
without doubt (anything else is "not checked", with the reason):

* an *estimate* and its *standard error*: two names that differ by one word, ``se`` (or ``var``, a variance, whose square
  root is used); e.g. ``theta_hat`` and ``theta_se``, ``rate_hat_0`` and ``rate_var_0``;
* an *interval*: two names that differ by ``lower``/``upper`` (or ``lo``/``hi``); e.g. ``theta_t_lower`` and ``theta_t_upper``;
* a *coverage indicator*: a name with ``covered`` or ``coverage`` whose every value is 0 or 1 (the trial's interval
  covered the truth, or not);
* the *true value*: a protocol threshold, or a value every trial of a setting returns alike, whose name says ``true``
  and shares a word with the estimate's (``theta_true_units`` for ``weighted_theta``), and that lies within the range of the
  estimates (or of the interval's ends) in at least half the settings;
* the *level* of the intervals: a number in the name (``coverage_95``), else the one level the plan states (a threshold
  named ``level``, ``confidence`` or ``nominal``, or a percentage in a metric's description of an interval). None, or
  several that differ: the coverage is not checked.

For each setting with at least :data:`MIN_TRIALS` trials: the coverage (from the indicator, or of the true value by the
interval), with its Wilson 95% interval; and the ratio of the estimates' standard deviation over the trials to the typical
reported standard error (the root of the mean squared one), with its sampling interval. A setting is a **gross miss** when
the whole Wilson interval lies more than :data:`COVERAGE_GAP` below the level, or the whole interval of the ratio lies
outside [:data:`RATIO_LOW`, :data:`RATIO_HIGH`]. A quantity grossly missed in at least :data:`GROSS_SHARE` of the
settings it was judged in is a **fault FI owns**: it is sent to repair with the numbers, and if it stays the quest stops (a wrong code must never
reach a paper). A smaller miss is not a fault (an interval of the wrong form can truly under-cover: a normal interval with
three degrees of freedom covers about 0.86): it is recorded and reported.

Pure functions over FI's per-setting trial summary (``raw/trials.json`` cells, :func:`core.trial_runner._summary`); nothing
here knows a field of science, and no name of a quantity is written into it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import stats as _stats

#: A setting where more than this share of the trials failed is a fault (its numbers rest on the trials that survived).
FAULT_FRACTION = 0.5
#: A setting where more than this share failed is stated in the paper.
STATE_FRACTION = 0.05
#: A coverage is grossly low when its whole Wilson interval is more than this below the level of the intervals.
COVERAGE_GAP = 0.25
#: A standard error is grossly wrong when the whole interval of (spread of the estimates / reported error) is outside these.
RATIO_LOW = 0.5
RATIO_HIGH = 2.0
#: A quantity is a fault when it is grossly missed in at least this share of the settings it was judged in. Chosen from the
#: replay of real quests (tests/test_trial_checks.py): clean quests had no grossly missed setting at all (several thousand
#: judged settings; the lowest coverage 0.81, standard-error ratios 0.86 to 1.11), while a noise level typed into a fitting
#: function missed in 12 of 25 settings (coverage down to 0.15) and weights on the wrong scale in 9 of 9 (ratios 10 to 18).
#: "More than half the settings" would have let the first through at 12 of 25.
GROSS_SHARE = 0.25
#: The fewest trials of a setting that are judged.
MIN_TRIALS = 10
#: Where FI keeps what the checks found on the last run of the trials (QUALITY.md and the paper's required items read it).
RECORD = Path(".fi") / "trial_checks.json"

_Z95 = 1.959963984540054

# The words that give a name its part (statistics words, not a field's): what is left of the name is the quantity.
_ESTIMATE = {"hat", "est", "estimate", "estimated", "fitted"}
_SE = {"se", "stderr", "sem"}
_SE_TWO = {("std", "err"), ("std", "error"), ("standard", "error")}
_VAR = {"var", "variance"}
_TRUE = {"true", "truth"}
_COVER = {"covered", "coverage", "cover", "covers"}
_LOWER = {"lower", "lo", "lb"}
_UPPER = {"upper", "hi", "ub"}
_LEVEL_WORDS = {"level", "confidence", "nominal"}
_NOT_A_LEVEL = {"limit", "lower", "upper", "min", "max", "minimum", "maximum", "cutoff", "tolerance", "threshold", "bootstrap"}


def tokens(name: Any) -> list[str]:
    """The words of a name, lower case: ``weighted_Theta_hat`` -> ``weighted, theta, hat``; ``lnB`` -> ``ln, b``;
    ``x0`` -> ``x, 0``."""
    return [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+(?:\.\d+)?", str(name or ""))]


def _role(name: str) -> tuple[str, tuple[str, ...]]:
    """``(part, the quantity's words)``: the part a name plays (``estimate``, ``se``, ``var``, ``true``, ``cover``,
    ``lower``, ``upper``, or ``""`` for a plain name) and its words without the part's word."""
    words = tokens(name)
    for a, b in _SE_TWO:
        for i in range(len(words) - 1):
            if words[i] == a and words[i + 1] == b:
                return "se", tuple(words[:i] + words[i + 2:])
    found = [(role, w) for w in words for role, group in (("se", _SE), ("var", _VAR), ("true", _TRUE), ("cover", _COVER),
                                                          ("lower", _LOWER), ("upper", _UPPER), ("estimate", _ESTIMATE))
             if w in group]
    roles = {r for r, _w in found}
    if len(roles) > 1:
        # one name in two parts (``k_hat_var``): read as the part that is not ``estimate`` when there is exactly one
        roles.discard("estimate")
    if len(roles) != 1:
        return "", tuple(words)
    role = roles.pop()
    drop = {w for r, w in found if r == role or r == "estimate"}
    return role, tuple(w for w in words if w not in drop)


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

    @property
    def fraction(self) -> float:
        return self.failed / self.planned if self.planned else 0.0

    @property
    def fault(self) -> bool:
        return self.planned > 0 and self.fraction > FAULT_FRACTION

    @property
    def stated(self) -> bool:
        return self.planned > 0 and self.fraction > STATE_FRACTION

    def says(self) -> str:
        why = "; ".join(f"{n}x {r}" for r, n in self.reasons[:3])
        return (f"{self.key or 'the one setting'}: {self.failed} of {self.planned} trials failed ({self.fraction:.0%})"
                + (f" - {why}" if why else ""))


def failures(cells: list[dict[str, Any]], reasons: dict[str, Counter] | None = None) -> list[SettingFailures]:
    """The failed trials of each setting the trials were run at (a value compared inside a run is not a setting of its
    own: a run that fails fails for every value). Settings with no failed trial are included, with ``failed == 0``."""
    out = []
    for c in cells:
        if c.get("level"):
            continue
        planned = int(c.get("planned") or 0)
        failed = int(c.get("failed") or 0)
        key = str(c.get("key") or "")
        got = (reasons or {}).get(key) or Counter()
        out.append(SettingFailures(key, planned, failed, got.most_common()))
    return out


def failures_directive(bad: list[SettingFailures]) -> str:
    """The plain message the simulation's repair is given for settings where most trials failed."""
    shown = "\n".join(f"- {s.says()}" for s in bad[:8]) + (f"\n- and {len(bad) - 8} more setting(s)" if len(bad) > 8 else "")
    return ("FI ran the study's trials, and in these settings more than half of the trials failed, so their numbers would "
            "rest on the few trials that happened to survive:\n" + shown + "\n\nRepair the simulation so that a trial "
            "at these settings runs to the end (the reasons and places above say where it stops). A setting where a trial "
            "truly has no result may return a value that says so, but it must not fail. Keep everything else as it is.")


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


@dataclass
class Quantity:
    """One reported uncertainty FI judged: ``kind`` is ``coverage`` or ``se_ratio``; ``names`` the trial values it is read
    from; ``level`` the intervals' level (coverage); ``truth`` how the true value was found."""

    kind: str
    label: str
    names: list[str]
    level: float | None = None
    truth: str = ""
    settings: list[SettingCheck] = field(default_factory=list)

    @property
    def judged(self) -> int:
        return len(self.settings)

    @property
    def gross(self) -> int:
        return sum(1 for s in self.settings if s.gross)

    @property
    def fault(self) -> bool:
        return self.gross > 0 and self.gross >= GROSS_SHARE * self.judged

    @property
    def below(self) -> int:
        return sum(1 for s in self.settings if s.below)

    def says(self) -> str:
        if self.kind == "coverage":
            vals = [s.value for s in self.settings]
            what = (f"coverage of the true value by `{self.label}` at the {self.level:.0%} level" if self.level is not None
                    else f"coverage `{self.label}`")
            rng = f"{min(vals):.3f} to {max(vals):.3f}" if vals else "no value"
            return (f"{what}: {rng} over {self.judged} setting(s); grossly low (more than {COVERAGE_GAP:g} below the level, "
                    f"beyond sampling error) in {self.gross}")
        vals = [s.value for s in self.settings]
        rng = f"{min(vals):.3g} to {max(vals):.3g}" if vals else "no value"
        return (f"spread of `{self.names[0]}` over the trials / its reported standard error `{self.names[1]}`: {rng} over "
                f"{self.judged} setting(s) (1 when the error is right); outside [{RATIO_LOW:g}, {RATIO_HIGH:g}] beyond "
                f"sampling error in {self.gross}")

    def worst(self, n: int = 4) -> list[SettingCheck]:
        if self.kind == "coverage":
            return sorted(self.settings, key=lambda s: s.value)[:n]
        return sorted(self.settings, key=lambda s: -abs(math.log(max(s.value, 1e-300))))[:n]


@dataclass
class Calibration:
    quantities: list[Quantity] = field(default_factory=list)
    #: Why something that looked like an interval or a standard error was not checked (one line each).
    not_checked: list[str] = field(default_factory=list)

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


def _levels(protocol: dict[str, Any] | None, thresholds: dict[str, Any]) -> set[float]:
    """The levels of the intervals the plan states: a threshold named for a level, or a percentage in the description of a
    metric that is about an interval."""
    found: set[float] = set()
    for name, value in (thresholds or {}).items():
        words = set(tokens(name))
        v = _number(value)
        if v is not None and words & _LEVEL_WORDS and not words & _NOT_A_LEVEL and 0.5 < v < 1.0:
            found.add(round(v, 6))
    metrics = (protocol or {}).get("metrics") if isinstance(protocol, dict) else None
    for m in metrics if isinstance(metrics, list) else []:
        text = str((m or {}).get("estimand") or "") if isinstance(m, dict) else ""
        if not re.search(r"\binterval|\bcover|\bcontain", text, re.I):
            continue
        for pct in re.findall(r"(\d{2}(?:\.\d+)?)\s*%", text):
            p = float(pct) / 100.0
            if 0.5 < p < 1.0:
                found.add(round(p, 6))
    return found


def _name_level(name: str) -> float | None:
    for w in tokens(name):
        try:
            p = float(w)
        except ValueError:
            continue
        if 50.0 <= p < 100.0:
            return round(p / 100.0, 6)
    return None


def _truths(cells: list[dict[str, Any]], names: list[str], thresholds: dict[str, Any]) -> list[tuple[str, tuple[str, ...], Any]]:
    """Every true value on offer: ``(its name, its words, value)``; ``value`` is a number (a threshold) or ``{setting key:
    number}`` (a value that every trial of a setting returns alike)."""
    out: list[tuple[str, tuple[str, ...], Any]] = []
    for name, value in (thresholds or {}).items():
        role, stem = _role(name)
        v = _number(value)
        if role == "true" and v is not None and stem:
            out.append((f"the plan's {name} = {v:g}", stem, v))
    for name in names:
        role, stem = _role(name)
        if role != "true" or not stem:
            continue
        per: dict[str, float] = {}
        alike = True
        for c in cells:
            vals = list(_series(c, name).values())
            if not vals:
                continue
            if max(vals) - min(vals) > 1e-9 * max(1.0, abs(vals[0])):
                alike = False
                break
            per[str(c.get("key"))] = vals[0]
        if alike and per:
            out.append((f"the trials' `{name}`", stem, per))
    return out


def _truth_for(stem: tuple[str, ...], truths: list[tuple[str, tuple[str, ...], Any]]) -> tuple[str, Any] | None | str:
    """The one true value for a quantity: the truth that shares the most words with it (at least one that is not a number,
    and every number the truth's name has); ``None`` when there is none, ``"ambiguous"`` when two differ and tie."""
    best: list[tuple[int, str, Any]] = []
    have = set(stem)
    for label, tstem, value in truths:
        words = set(tstem)
        digits = {w for w in words if w[0].isdigit()}
        if not digits <= have:
            continue
        shared = words & have
        if not any(not w[0].isdigit() for w in shared):
            continue
        best.append((len(shared), label, value))
    if not best:
        return None
    top = max(s for s, _l, _v in best)
    tied = [(label, v) for s, label, v in best if s == top]
    values = {json.dumps(v, sort_keys=True) for _l, v in tied}
    if len(values) > 1:
        return "ambiguous"
    return tied[0]


def _truth_in(value: Any, key: str) -> float | None:
    if isinstance(value, dict):
        return _number(value.get(key))
    return _number(value)


def _plausible(truth: Any, cells: list[dict[str, Any]], spans: dict[str, tuple[float, float]]) -> bool:
    """Whether the true value lies within the range of what was reported for it (the estimates, or the ends of the
    intervals) in at least half the settings that reported it."""
    seen = inside = 0
    for c in cells:
        key = str(c.get("key"))
        if key not in spans:
            continue
        t = _truth_in(truth, key)
        if t is None:
            continue
        lo, hi = spans[key]
        seen += 1
        inside += lo <= t <= hi
    return seen > 0 and inside * 2 >= seen


def _coverage_check(key: str, hits: list[float], level: float) -> SettingCheck | None:
    n = len(hits)
    if n < MIN_TRIALS:
        return None
    k = sum(hits)
    ci = _stats.wilson_interval(k, n)
    if ci is None:
        return None
    lo, hi = ci
    return SettingCheck(key, n, k / n, lo, hi, gross=hi < level - COVERAGE_GAP, below=hi < level)


def _ratio_check(key: str, est: list[float], se: list[float]) -> SettingCheck | None:
    n = len(est)
    if n < MIN_TRIALS:
        return None
    mean = sum(est) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in est) / (n - 1))
    rms = math.sqrt(sum(s * s for s in se) / n)
    scale = max(abs(mean), 1e-300)
    if rms <= 0 or sd <= 1e-12 * scale:
        return None  # no spread to judge (a setting without noise), or no error reported
    ratio = sd / rms
    half = _Z95 / math.sqrt(2.0 * (n - 1))
    lo, hi = ratio * math.exp(-half), ratio * math.exp(half)
    return SettingCheck(key, n, ratio, lo, hi, gross=lo > RATIO_HIGH or hi < RATIO_LOW, below=lo > 1.25 or hi < 0.8)


def calibration(cells: list[dict[str, Any]], protocol: dict[str, Any] | None = None,
                thresholds: dict[str, Any] | None = None) -> Calibration:
    """What the trials' intervals and standard errors say against the true values (see the module's notes)."""
    out = Calibration()
    if thresholds is None:
        thresholds = (protocol or {}).get("thresholds") if isinstance((protocol or {}).get("thresholds"), dict) else {}
    thresholds = thresholds or {}
    names = sorted({n for c in cells for n in (c.get("metrics") or {})})
    roles = {n: _role(n) for n in names}
    truths = _truths(cells, names, thresholds)
    plan_levels = _levels(protocol, thresholds)

    def level_of(name: str) -> tuple[float | None, str]:
        own = _name_level(name)
        if own is not None:
            return own, ""
        if len(plan_levels) == 1:
            return next(iter(plan_levels)), ""
        if not plan_levels:
            return None, "the plan states no level for its intervals"
        return None, "the plan states several levels (" + ", ".join(f"{p:g}" for p in sorted(plan_levels)) + ")"

    # coverage indicators: 0/1 in every trial
    for name in names:
        if roles[name][0] != "cover":
            continue
        values = [v for c in cells for v in _series(c, name).values()]
        if not values or any(v not in (0.0, 1.0) for v in values):
            continue
        level, why = level_of(name)
        if level is None:
            out.not_checked.append(f"`{name}`: {why}")
            continue
        q = Quantity("coverage", name, [name], level=level, truth="the run's own record of whether each interval covered it")
        for c in cells:
            got = _coverage_check(str(c.get("key")), list(_series(c, name).values()), level)
            if got is not None:
                q.settings.append(got)
        out.quantities.append(q)

    # intervals: lower and upper of one quantity
    stems: dict[tuple[str, ...], dict[str, str]] = {}
    for name in names:
        role, stem = roles[name]
        if role in ("lower", "upper") and stem:
            stems.setdefault(stem, {}).setdefault(role, name)
    for stem, ends in sorted(stems.items()):
        if set(ends) != {"lower", "upper"}:
            continue
        lo_name, hi_name = ends["lower"], ends["upper"]
        label = f"{lo_name} .. {hi_name}"
        level, why = level_of(lo_name)
        if level is None:
            level, why2 = level_of(hi_name)
            why = why2 if level is None else ""
        if level is None:
            out.not_checked.append(f"the interval `{label}`: {why}")
            continue
        found = _truth_for(stem, truths)
        if found is None:
            out.not_checked.append(f"the interval `{label}`: no true value the plan declares shares its name")
            continue
        if found == "ambiguous":
            out.not_checked.append(f"the interval `{label}`: two true values fit its name equally")
            continue
        truth_label, truth = found
        spans: dict[str, tuple[float, float]] = {}
        for c in cells:
            lo_s, hi_s = _series(c, lo_name), _series(c, hi_name)
            if lo_s and hi_s:
                spans[str(c.get("key"))] = (min(lo_s.values()), max(hi_s.values()))
        if not _plausible(truth, cells, spans):
            out.not_checked.append(f"the interval `{label}`: {truth_label} lies outside the intervals' range in most settings, "
                                   "so it is not this quantity's true value")
            continue
        q = Quantity("coverage", label, [lo_name, hi_name], level=level, truth=truth_label)
        for c in cells:
            key = str(c.get("key"))
            t = _truth_in(truth, key)
            lo_s, hi_s = _series(c, lo_name), _series(c, hi_name)
            if t is None:
                continue
            hits = [1.0 if lo_s[i] <= t <= hi_s[i] else 0.0 for i in lo_s if i in hi_s]
            got = _coverage_check(key, hits, level)
            if got is not None:
                q.settings.append(got)
        out.quantities.append(q)

    # an estimate and its standard error (or variance)
    plain: dict[tuple[str, ...], list[str]] = {}
    for name in names:
        role, stem = roles[name]
        if role in ("", "estimate") and stem:
            plain.setdefault(stem, []).append(name)
    for name in names:
        role, stem = roles[name]
        if role not in ("se", "var") or not stem:
            continue
        estimates = plain.get(stem) or []
        if not estimates:
            continue
        if len(estimates) > 1:
            out.not_checked.append(f"`{name}`: more than one estimate has its name ({', '.join(estimates)})")
            continue
        est = estimates[0]
        found = _truth_for(stem, truths)
        if found is None:
            out.not_checked.append(f"`{est}` with `{name}`: the plan declares no true value that shares its name, so FI cannot "
                                   "tell whether the trials of a setting all estimate the same value")
            continue
        if found == "ambiguous":
            out.not_checked.append(f"`{est}` with `{name}`: two true values fit its name equally")
            continue
        truth_label, truth = found
        spans = {}
        for c in cells:
            e = _series(c, est)
            if e:
                spans[str(c.get("key"))] = (min(e.values()), max(e.values()))
        if not _plausible(truth, cells, spans):
            out.not_checked.append(f"`{est}` with `{name}`: {truth_label} lies outside the estimates' range in most settings, "
                                   "so it is not this quantity's true value")
            continue
        q = Quantity("se_ratio", est, [est, name], truth=truth_label)
        for c in cells:
            e, s = _series(c, est), _series(c, name)
            pairs = [(e[i], s[i]) for i in e if i in s and (s[i] >= 0)]
            if role == "var":
                pairs = [(x, math.sqrt(v)) for x, v in pairs]
            got = _ratio_check(str(c.get("key")), [x for x, _ in pairs], [v for _, v in pairs])
            if got is not None:
                q.settings.append(got)
        out.quantities.append(q)
    out.quantities = [q for q in out.quantities if q.judged]
    if not out.quantities:
        out.not_checked.append("the run reports no interval or standard error that FI could pair with a true value the "
                               "plan declares (an estimate with `_se` or `_var`, an interval with `_lower`/`_upper`, or a "
                               "0/1 `covered` value), so the calibration was not checked")
    return out


def calibration_directive(faults: list[Quantity]) -> str:
    """The plain message the simulation's repair is given for a reported uncertainty that is grossly wrong. The numbers are
    FI's measurements on the run; no expected value of the study is in it."""
    lines = []
    for q in faults[:6]:
        worst = "; ".join(f"{s.key}: {s.value:.3g} (n={s.n})" for s in q.worst())
        lines.append(f"- {q.says()}. Worst settings: {worst}")
    return ("FI measured the intervals and standard errors the simulation reports against the true values it was given, "
            "over the trials of each setting, and found them grossly wrong:\n" + "\n".join(lines)
            + "\n\nA reported standard error must be about the spread of the estimate over repeated trials, and a 95% "
            "interval must cover the true value in about 95% of them. Find what makes them disagree in the code (a noise "
            "level or a weight that is fixed instead of computed from the trial's own data, a variance on the wrong scale, "
            "a wrong critical value) and fix it. Do not change the true values, the settings or the number of trials.")


# ---- the record --------------------------------------------------------------------------------------------------------


def summary(fails: list[SettingFailures], cal: Calibration | None) -> dict[str, Any]:
    """What the record keeps: the settings with failed trials, and each quantity the calibration check judged."""
    return {
        "failures": [{"key": s.key, "planned": s.planned, "failed": s.failed, "fraction": round(s.fraction, 4),
                      "fault": s.fault, "reasons": [[r, n] for r, n in s.reasons[:5]]} for s in fails if s.failed],
        "settings": len(fails),
        "calibration": None if cal is None else {
            "quantities": [{"kind": q.kind, "label": q.label, "names": q.names, "level": q.level, "truth": q.truth,
                            "judged": q.judged, "gross": q.gross, "below": q.below, "fault": q.fault, "says": q.says(),
                            "settings": [{"key": s.key, "n": s.n, "value": round(s.value, 6), "low": round(s.low, 6),
                                          "high": round(s.high, 6), "gross": s.gross} for s in q.settings]}
                           for q in cal.quantities],
            "not_checked": list(cal.not_checked),
            "checked": cal.checked,
        },
    }


def write_record(quest_root: Path | str, record: dict[str, Any]) -> None:
    try:
        path = Path(quest_root) / RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, allow_nan=True), encoding="utf-8")
    except OSError:
        pass


def read_record(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
