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
itself from the trials: whether each trial's interval covers the true value (from its two ends, or the estimate plus and
minus z times its standard error at the declared level), and the ratio of the estimates' standard deviation over the
trials to the typical reported standard error (the root of the mean squared one). It never reads a coverage flag the
code computed. A metric whose declared names the trials do not return is a fault (the code does not keep the plan's
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

Pure functions over FI's per-setting trial summary (``raw/trials.json`` cells, :func:`core.trial_runner._summary`); nothing
here knows a field of science, and no name of a quantity is written into it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

from . import metric_spec as _metric_spec
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


def _declared_names(protocol: dict[str, Any] | None) -> list[tuple[str, dict[str, list[Any]]]]:
    """``[(name, undefined_when)]``: every value a trial must return as a number, by the plan: each protocol metric's own
    id and the names of its ``known_value``, with the settings where the plan says that number has no value. A mean over a
    subset of the trials (``given``) has no value outside the subset, so its own id is not one."""
    out: list[tuple[str, dict[str, list[Any]]]] = []
    for spec in _metric_spec.declared(protocol):
        when = spec.get("undefined_when") or {}
        kv = spec.get("known_value") or {}
        names = ([] if spec.get("given") else [spec["id"]]) + [
            kv[k] for k in ("estimate", "standard_error", "variance") if kv.get(k)] + list(kv.get("interval") or [])
        out.extend((n, when) for n in dict.fromkeys(names))
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


def _no_value_trials(records: list[dict[str, Any]], names: list[tuple[str, dict[str, list[Any]]]]) -> dict[int, str]:
    """``{trial: the first declared name it gave no number for}``: a trial that ran but returned NaN, an infinity, or
    nothing at all for a value the plan declares, at a setting where the plan does not say it has none."""
    out: dict[int, str] = {}
    for rec in records:
        metrics = rec.get("metrics") or {}
        ran = {t for m in metrics.values() if isinstance(m, dict) for t in (m.get("trials") or []) if isinstance(t, int)}
        for name, when in names:
            m = metrics.get(name)
            if not isinstance(m, dict) or _undefined_here(when, rec):
                continue
            finite = set(_series(rec, name))
            for trial in sorted(ran - finite):
                out.setdefault(trial, name)
    return out


def failures(cells: list[dict[str, Any]], reasons: dict[str, Counter] | None = None,
             protocol: dict[str, Any] | None = None) -> list[SettingFailures]:
    """The failed trials of each setting the trials were run at (a value compared inside a run is not a setting of its
    own: a run that fails fails for every value). Settings with no failed trial are included, with ``failed == 0``.

    A trial that ran but returned NaN, an infinity or nothing for a value the plan declares (``protocol.metrics``) counts
    as failed: a repair that hides a failing trial behind a NaN would otherwise turn a setting where most trials fail into
    one where none do. Only the settings the plan names in a metric's ``undefined_when`` are spared."""
    names = _declared_names(protocol)
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
        if names:
            records = [c] + [r for r in cells if r.get("level") and of_settings(r, {key})]
            empty = _no_value_trials(records, names)
            for name in empty.values():
                got[f"the trial returned no number (NaN, an infinity or nothing) for `{name}`, which the plan declares"] += 1
            failed = min(planned, failed + len(empty)) if planned else failed
        out.append(SettingFailures(key, planned, failed, got.most_common(), out_of_time=late))
    return out


def failures_directive(bad: list[SettingFailures]) -> str:
    """The plain message the simulation's repair is given for settings where most trials failed."""
    shown = "\n".join(f"- {s.says()}" for s in bad[:8]) + (f"\n- and {len(bad) - 8} more setting(s)" if len(bad) > 8 else "")
    return ("FI ran the study's trials, and in these settings more than half of the trials failed, so their numbers would "
            "rest on the few trials that happened to survive:\n" + shown + "\n\nRepair the simulation so that a trial "
            "at these settings runs to the end and gives its numbers (the reasons and places above say where it stops). "
            "Never return NaN, an infinity or a made-up value in place of a trial that failed: FI counts such a trial as "
            "failed. A number the plan says has no value at some settings is the plan's to declare, not the code's. Keep "
            "everything else as it is.")


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
    #: The same number over every setting together (at 95%): what catches a moderate miss present everywhere.
    pooled: SettingCheck | None = None

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
    return _number((cell.get("cell") or {}).get(truth))


#: What the record says when the plan declares nothing FI can measure the reported uncertainty against.
NO_DECLARATION = ("the plan declares no metric with a known true value (`known_value`: the true value, the names run_trial "
                  "returns for the estimate and its standard error or interval, and the interval's level), so FI could "
                  "not measure the reported intervals and standard errors against the truth")


def _z_family(k: int) -> float:
    """The critical value of each of ``k`` settings' intervals when, together, a right code shows a gross miss by sampling
    alone with chance :data:`FAMILY_ALPHA` at most."""
    return NormalDist().inv_cdf(1.0 - FAMILY_ALPHA / (2.0 * max(1, k)))


def _coverage_check(key: str, hits: list[float], level: float, z: float = _Z95) -> SettingCheck | None:
    n = len(hits)
    if n < MIN_TRIALS:
        return None
    k = sum(hits)
    ci = _stats.wilson_interval(k, n, z=z)
    if ci is None:
        return None
    lo, hi = ci
    return SettingCheck(key, n, k / n, lo, hi, gross=hi < level - COVERAGE_GAP, below=hi < level, above=lo > level)


def _ratio_of(est: list[float], se: list[float]) -> tuple[float, int] | None:
    """``(spread of the estimates / root mean square of their errors, degrees of freedom)``, or ``None`` when there is no
    spread to judge (a setting without noise) or no error reported."""
    n = len(est)
    if n < MIN_TRIALS:
        return None
    mean = sum(est) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in est) / (n - 1))
    rms = math.sqrt(sum(s * s for s in se) / n)
    if rms <= 0 or sd <= 1e-12 * max(abs(mean), 1e-300):
        return None
    return sd / rms, n - 1


def _ratio_band(key: str, n: int, ratio: float, dof: int, z: float = _Z95) -> SettingCheck:
    half = z / math.sqrt(2.0 * dof)
    lo, hi = ratio * math.exp(-half), ratio * math.exp(half)
    return SettingCheck(key, n, ratio, lo, hi, gross=lo > RATIO_HIGH or hi < RATIO_LOW, below=lo > 1.25 or hi < 0.8)


#: The key of the check over every setting together.
POOLED = "all settings together"


def _judge_coverage(q: "Quantity", per: list[tuple[str, list[float]]]) -> None:
    """Each setting's coverage at the family level, and the pooled coverage at 95%."""
    judged = [(key, hits) for key, hits in per if len(hits) >= MIN_TRIALS]
    z = _z_family(len(judged))
    for key, hits in judged:
        got = _coverage_check(key, hits, q.level or 0.95, z)
        if got is not None:
            q.settings.append(got)
    if len(judged) > 1:
        q.pooled = _coverage_check(POOLED, [h for _k, hits in judged for h in hits], q.level or 0.95)


def _judge_ratio(q: "Quantity", per: list[tuple[str, list[float], list[float]]]) -> None:
    """Each setting's spread / reported error at the family level, and the pooled ratio at 95% (the root of the mean squared
    ratio weighted by each setting's degrees of freedom)."""
    found = [(key, len(e), _ratio_of(e, s)) for key, e, s in per]
    found = [(key, n, r) for key, n, r in found if r is not None]
    z = _z_family(len(found))
    for key, n, (ratio, dof) in found:
        q.settings.append(_ratio_band(key, n, ratio, dof, z))
    if len(found) > 1:
        dof = sum(r[1] for _k, _n, r in found)
        pooled = math.sqrt(sum(r[1] * r[0] ** 2 for _k, _n, r in found) / dof)
        q.pooled = _ratio_band(POOLED, sum(n for _k, n, _r in found), pooled, dof)


def _cell_hits(cell: dict[str, Any], kv: dict[str, Any], truth: float) -> tuple[dict[int, float], dict[int, float], list[float]]:
    """``(estimates, standard errors, hits)`` of one setting for a declared known value: each trial's estimate and standard
    error by trial, and 1/0 per trial for whether its interval (the declared two ends, else the estimate plus and minus z
    times the standard error at the declared level) covers ``truth``."""
    spread = kv.get("standard_error") or kv.get("variance")
    e = _series(cell, kv["estimate"])
    s = _series(cell, spread) if spread else {}
    if spread and "standard_error" not in kv:
        s = {i: math.sqrt(v) for i, v in s.items() if v >= 0}
    ends = kv.get("interval")
    if ends:
        lo, hi = _series(cell, ends[0]), _series(cell, ends[1])
        hits = [1.0 if lo[i] <= truth <= hi[i] else 0.0 for i in lo if i in hi]
    else:
        z = _z(kv["level"])
        hits = [1.0 if abs(e[i] - truth) <= z * s[i] else 0.0 for i in e if i in s and s[i] >= 0]
    return e, s, hits


def calibration(cells: list[dict[str, Any]], protocol: dict[str, Any] | None = None,
                thresholds: dict[str, Any] | None = None) -> Calibration:
    """What the trials' intervals and standard errors say against the true values the plan declares (see the module's
    notes). Only the protocol's ``known_value`` declarations are read."""
    out = Calibration()
    protocol = protocol if isinstance(protocol, dict) else {}
    if thresholds is None:
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
    thresholds = thresholds or {}
    axes = {str(k) for part in ("grid", "in_each_run") if isinstance(protocol.get(part), dict) for k in protocol[part]}
    known = [s for s in _metric_spec.declared(protocol) if s.get("known_value")]
    if not known:
        out.not_checked.append(NO_DECLARATION)
        return out
    returned = {n for c in cells for n in (c.get("metrics") or {})}
    if not returned:
        out.not_checked.append("no setting has trials to judge, so the calibration was not checked")
        return out
    for spec in known:
        kv, ident = spec["known_value"], spec["id"]
        truth, level, est = kv["true_value"], kv["level"], kv["estimate"]
        if isinstance(truth, str) and truth not in thresholds and truth not in axes:
            out.not_checked.append(f"`{ident}`: its true value `{truth}` is neither a threshold nor a grid parameter of the "
                                   "plan")
            continue
        truth_label = (f"the plan's {truth} = {float(truth):g}" if not isinstance(truth, str) else
                       f"the plan's {truth}" + (f" = {_number(thresholds[truth]):g}" if _number(thresholds.get(truth))
                                                is not None else " (per setting)"))
        spread = kv.get("standard_error") or kv.get("variance")
        ends = kv.get("interval")
        missing = [n for n in [est, *([spread] if spread else []), *(ends or [])] if n not in returned]
        if missing:
            out.quantities.append(Quantity("missing", ident, missing, level=level, truth=truth_label))
            continue
        cover = Quantity("coverage", " .. ".join(ends) if ends else f"{est} +/- z*{spread}",
                         list(ends) if ends else [est, spread], level=level, truth=truth_label)
        ratio = Quantity("se_ratio", est, [est, spread], truth=truth_label) if spread else None
        cover_per: list[tuple[str, list[float]]] = []
        ratio_per: list[tuple[str, list[float], list[float]]] = []
        for c in cells:
            key = str(c.get("key"))
            t = _truth_of(truth, c, thresholds)
            if t is None:
                continue
            e, s, hits = _cell_hits(c, kv, t)
            cover_per.append((key, hits))
            if ratio is not None:
                pairs = [(e[i], s[i]) for i in e if i in s and s[i] >= 0]
                ratio_per.append((key, [x for x, _ in pairs], [v for _, v in pairs]))
        _judge_coverage(cover, cover_per)
        if ratio is not None:
            _judge_ratio(ratio, ratio_per)
        for q in (cover, ratio):
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
    if thresholds is None:
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
    thresholds = thresholds or {}
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
    return ("FI measured the intervals and standard errors the simulation reports against the true values it was given, "
            "over the trials of each setting, and found them grossly wrong:\n" + "\n".join(lines)
            + "\n\nA reported standard error must be about the spread of the estimate over repeated trials, and a 95% "
            "interval must cover the true value in about 95% of them. Find what makes them disagree in the code (a noise "
            "level or a weight that is fixed instead of computed from the trial's own data, a variance on the wrong scale, "
            "a wrong critical value) and fix it. Do not change the true values, the settings or the number of trials.")


# ---- what FI does with them --------------------------------------------------------------------------------------------

OK, REPAIR, STOP, LEAVE_OUT = "ok", "repair", "stop", "leave_out"


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
    numbers are left out of what the analysis is given, ``left_out``)."""

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

    def record(self) -> dict[str, Any]:
        return {**summary(self.fails, self.calibration), "action": self.action, "left_out_settings": list(self.left_out),
                **({"say": self.say} if self.say else {})}


def judge(cells: list[dict[str, Any]], protocol: dict[str, Any] | None = None, thresholds: dict[str, Any] | None = None,
          reasons: dict[str, Counter] | None = None, *, repairs_left: bool, no_calibration: str = "") -> Verdict:
    """Both checks on a run's per-setting summary, and what follows from them. A setting where most trials failed is judged
    by the failure check alone (its numbers rest on the trials that survived), so the calibration reads the other settings.
    While a repair is left, any fault is sent back; once they are spent, a calibration fault or a failure in more than half
    of the settings stops the quest, and failures in at most half of them leave those settings out, but only when every
    remaining setting's calibration was judged against the truth (and passed): one that was not checked makes it a stop. With
    ``no_calibration`` (the reason), only the failed trials are checked and the record says the calibration was not."""
    fails = failures(cells, reasons, protocol)
    bad = [f for f in fails if f.fault]
    run = [f for f in fails if f.planned]
    every = bool(bad) and len(bad) == len(run)
    # More than half of the settings left out would leave too little to report: that is a stop, never a paper.
    most = bool(bad) and len(bad) * 2 > len(run)
    keys = {f.key for f in bad}
    cal = (Calibration(not_checked=[no_calibration]) if no_calibration else
           calibration([c for c in cells if not of_settings(c, keys)], protocol, thresholds))
    out = Verdict(fails, cal)
    if not bad and not cal.faults:
        return out
    if cal.faults or most or repairs_left:
        parts = ([failures_directive(bad)] if bad else []) + ([calibration_directive(cal.faults)] if cal.faults else [])
        out.directive = "\n\n".join(parts)
        out.action = REPAIR if repairs_left else STOP
        # Only some settings (at most half) fail and nothing else is wrong: once the repairs are spent, those settings
        # are left out.
        out.partial = not cal.faults and not most
        if cal.faults:
            q = cal.faults[0]
            out.say = ("FI measured the intervals and standard errors the simulation reports against the true values it "
                       f"was given, and they are grossly wrong ({q.says()})")
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
        return out
    # Settings are left out only when what remains is shown sound: every remaining setting's intervals and standard errors
    # were judged against the truth (and passed, else there was a fault above). "Not checked" is not a pass: the failures
    # may be the visible part of a wrong code whose other settings merely did not crash.
    remaining = {f.key for f in run if f.key not in keys}
    judged = {s.key for q in cal.quantities for s in q.settings}
    unjudged = sorted(remaining - judged)
    if unjudged and not no_calibration:  # a search never uses a failed design, so nothing it leaves out is analysed
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
                            "says": q.says(),
                            "settings": [{"key": s.key, "n": s.n, "value": round(s.value, 6), "low": round(s.low, 6),
                                          "high": round(s.high, 6), "gross": s.gross} for s in q.settings],
                            "pooled": None if q.pooled is None else {
                                "n": q.pooled.n, "value": round(q.pooled.value, 6), "low": round(q.pooled.low, 6),
                                "high": round(q.pooled.high, 6), "gross": q.pooled.gross}}
                           for q in cal.quantities],
            "not_checked": list(cal.not_checked),
            "checked": cal.checked,
        },
    }


def left_out_settings(quest_root: Path | str) -> list[dict[str, Any]]:
    """The settings whose numbers the last run left out (most of their trials failed and the repairs were spent), each with
    its failed and planned counts, from the record."""
    rec = read_record(quest_root)
    keys = set(rec.get("left_out_settings") or [])
    return [f for f in rec.get("failures") or [] if isinstance(f, dict) and f.get("key") in keys]


def write_record(quest_root: Path | str, record: dict[str, Any]) -> None:
    try:
        path = Path(quest_root) / RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, allow_nan=True), encoding="utf-8")
    except OSError:
        pass


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
