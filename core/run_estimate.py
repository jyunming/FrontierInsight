"""How long the whole run will take, measured on a few trials before it starts.

A study is every setting of the plan's grid times its runs per setting, each setting in its own process
(``core/trial_runner.py``). Before the full run FI times a handful of real trials, adds them up and compares the total
with the time the study is allowed (the topic's own limit when it states one, else ``execution.timeout_s``):

* **What is timed.** One *base* setting (the middle value of every setting that varies) and, for each varying setting, the
  base with that one setting at its first and at its last value: at most ``1 + 2 * MAX_AXES`` settings. A numerical
  resolution that dominates the cost (a finer mesh, a smaller step) is a setting like any other, so it shows up as a
  large factor between the base and one end. Each timed setting runs its first trial (the process start, the loading of
  the simulation and the first call, as the real run pays them) and, for a stochastic simulation, more trials until the
  later ones have taken :data:`TIMING_LATER_S` (at most :data:`TIMING_TRIALS`), timed on a fine clock: their mean is what
  each further run costs (``Probe.per_trial_s``), and the rest is what the setting costs once whatever its runs
  (``Probe.fixed_s``), so a smaller run is costed with its process start paid once (:meth:`Probe.at`).
* **What is added up.** The cost of the setting a value belongs to is the base's cost times, for each varying setting,
  the ratio measured for that value (values in between are read between their timed neighbours, in proportion), so the
  total is the number of settings times the base's cost times the mean ratio of each varying setting. A trial that does
  not finish within its allowance is a lower bound, and the estimate says it is "at least".
* **What it is compared with.** The estimate with a margin of :data:`SAFETY` (one half more) against the time allowed.
  A run that would not fit is given fewer runs per setting (:func:`fit_runs`), never fewer settings, and the 95% interval
  those runs give is stated (:func:`expected_precision`), or said to be unknown before the run. For FI's own limit the
  runs go down only to the fewest the plan states; for the topic's own limit they may go down to
  :data:`MIN_RUNS_FOR_LIMIT` whether or not the plan states a fewest. Every cut is an item the paper must name
  (:data:`core.paper_facts.FEWER_RUNS`).

None of the timed trials is a result: they are run apart from the study, write nothing into FI's ledger and are not
kept. Everything here is arithmetic on the measurements; nothing is a judgement about the science.

The same trials are also a **correctness pilot** (:func:`pilot_problems`): what they returned is read before the full run,
so a simulation that measures nothing is found in minutes, not after the study:

* a metric of the protocol that the trials return (``protocol.metrics``) and that is not a number in ANY pilot trial at ANY
  pilot setting is undefined everywhere. Before it counts, every value of every setting is tried (the others at the base) and
  the settings with few values (at most three) in every combination (:func:`confirm_cells`), so a metric that applies only to
  some of the settings is not mistaken for a broken one; if those cannot all be tried, nothing is claimed;
* a ``mean`` metric that is exactly one value in every pilot trial although the settings change and other results move, and a
  simulation whose every returned number is the same at every setting (:func:`core.flat_output.flat_sweep`). Values that did
  not change are only an observation (a conserved quantity or a true absence of effect look the same), so each of these, and
  the stuck proportions below, is a finding only with evidence from the simulation's own code that the returned value is
  built from no setting (:func:`core.flat_output.unused_outputs`, :func:`core.flat_output.unread_settings`); without the
  code's text, or when the code cannot be followed with certainty, nothing is claimed;
* proportions stuck at a bound: **at least two different** ``proportion`` metrics that are each exactly 0 (or each exactly 1)
  in every pilot trial that returned them, over at least :data:`MIN_STUCK_TRIALS` trials at :data:`MIN_STUCK_SETTINGS`
  settings. The protocol has no structured field that tells a rare-event probability (where 0 is the expected outcome of a
  small pilot) from any other, so one metric at a bound is never a finding; two or more at once are the sign of a simulated
  object that is itself degenerate. Names the plan expects to stay put (``expected_constants``) are exempt. Several proportions
  at opposite bounds that flip only between settings are deliberately **not** a finding: a sharp but honest transition sampled
  by a few settings looks the same, and a pilot of a few trials cannot tell the two apart.

A metric that is undefined at some settings and has a value at others is the plan's own business and is never a finding. The
findings go to the existing repair of the simulation (``RunChecksMixin._pilot_gate``); a metric the trials do not return at all
(a quantity the analysis derives) is left to the checks that follow the run.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import in_run as _in_run
from . import optimise_search as _search
from . import outside_run as _outside
from . import trial_checks as _checks
from . import trial_runner as _tr

#: The margin added to the measured total before it is compared with the time allowed.
SAFETY = 1.5
#: The most varying settings that are timed one by one (the first ones, largest range first); the rest are taken as 1.
MAX_AXES = 6
#: A timed stochastic setting runs its first trial and then more, until the later ones have taken this many seconds, at
#: most :data:`TIMING_TRIALS` in all: one short trial on a coarse clock measured nothing (r12: 0 or 16 ms for a 4 ms trial).
TIMING_LATER_S = 1.0
TIMING_TRIALS = 2000
#: The fewest runs per setting FI goes down to for the topic's own time limit, whatever the plan states (and never below what
#: the plan's checks need, :func:`runs_checks_need`). The paper names every such cut (``core.paper_facts.FEWER_RUNS``).
MIN_RUNS_FOR_LIMIT = 10
RECORD = Path(".fi") / "run_estimate.json"


def plain_duration(seconds: float) -> str:
    """``"45 seconds"``, ``"12 minutes"``, ``"3 h 20 min"``, ``"2 days 4 h"``: a length of time as a person says it."""
    s = max(0.0, float(seconds))
    if s < 90:
        n = max(1, round(s))
        return f"{n} second{'s' if n != 1 else ''}"
    minutes = s / 60.0
    if minutes < 60:
        return f"{round(minutes)} minutes"
    hours = minutes / 60.0
    if hours < 48:
        whole = int(hours)
        rest = round((hours - whole) * 60)
        if rest == 60:
            whole, rest = whole + 1, 0
        return f"{whole} h" + (f" {rest} min" if rest and whole < 10 else "")
    days = int(hours // 24)
    return f"{days} days" + (f" {round(hours - 24 * days)} h" if round(hours - 24 * days) else "")


def axes_of(grid: dict[str, list[Any]]) -> list[tuple[str, list[Any]]]:
    """The settings of ``grid`` as ``(name, values)``, in the grid's order."""
    return [(str(k), list(v)) for k, v in grid.items() if isinstance(v, list) and v]


def cell_count(grid: dict[str, list[Any]]) -> int:
    """How many settings the study has: every combination of the values, or, for a grid with sub-studies, the settings in
    at least one of them."""
    if getattr(grid, "sub_studies", None) and len(axes_of(grid)) == len(grid):
        return _search.grid_size(grid)
    n = 1
    for _name, values in axes_of(grid):
        n *= len(values)
    return n


def trial_count(grid: dict[str, list[Any]], runs: int, deterministic: bool) -> int:
    return cell_count(grid) * (1 if deterministic else max(1, int(runs)))


@dataclass
class Probe:
    """One timed setting: the whole setting's cost as the real run would pay it (``seconds``), and whether it is only a
    lower bound (the timed trial did not finish within its allowance)."""

    axis: Any  # the setting's name; None for the base; CORNER or PART for the probes that are not one setting's end
    index: int | None
    cell: dict[str, Any]
    seconds: float
    at_least: bool = False
    #: ``{metric: sample standard deviation}`` of the timed trials of a stochastic simulation (empty otherwise): the spread
    #: a precision target on a mean is judged by.
    spread: dict[str, float] = field(default_factory=dict)
    #: What the first two timed trials returned (``{metric: number}``, not-a-number kept): the pilot's evidence of correctness.
    trials: list[dict[str, float]] = field(default_factory=list)
    #: What the setting costs that does not depend on its runs (the process, the loading, the first call's extra), and what
    #: each run costs (the mean of the timed trials after the first): ``seconds`` is ``fixed_s + runs * per_trial_s``.
    #: ``None`` when they were not measured apart (a probe made by hand, or a setting run once): the cost is then taken to
    #: follow the runs in proportion.
    fixed_s: float | None = None
    per_trial_s: float | None = None
    #: How many trials of the setting were timed.
    timed_trials: int = 0
    #: The time the timed trials spent inside the simulation itself (without the process start and the loading), when known.
    call_s: float | None = None

    def at(self, runs: int, runs_now: int) -> "Probe":
        """This setting's cost with ``runs`` runs instead of ``runs_now``."""
        if self.fixed_s is not None and self.per_trial_s is not None:
            seconds = max(0.0, self.fixed_s) + max(1, int(runs)) * self.per_trial_s
        else:
            seconds = self.seconds * max(1, int(runs)) / max(1, int(runs_now))
        return Probe(self.axis, self.index, dict(self.cell), max(seconds, 1e-6), self.at_least, dict(self.spread),
                     [dict(t) for t in self.trials], self.fixed_s, self.per_trial_s, self.timed_trials)


@dataclass
class Estimate:
    seconds: float
    at_least: bool
    complete: bool
    cells: int
    trials: int
    #: ``{setting: {value text: how many times the base's cost}}`` for the values that were timed.
    ratios: dict[str, dict[str, float]] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    base_seconds: float = 0.0
    timed: int = 0
    #: When two or more settings are each costly, the settings at their costly ends together and how many times the base's
    #: cost that was measured: ``{"cell": {...}, "ratio": 3.2}``.
    together: dict[str, Any] | None = None
    #: ``{metric: the largest standard deviation any timed setting's trials showed}``.
    spreads: dict[str, float] = field(default_factory=dict)
    #: A study of separate parts: how many parts it has, and how many of them had a setting timed (a part not timed makes the
    #: estimate a lower bound). ``0, 0`` for a study of one grid.
    parts: int = 0
    parts_timed: int = 0
    #: ``[(part, its settings, the seconds one of its settings took)]``: the cost summed per part.
    part_costs: list[tuple[str, int, float]] = field(default_factory=list)


def probe_key(cell: dict[str, Any], runs: int) -> str:
    """What a timing of one setting depends on: the setting and the runs per setting (the cost of the whole setting)."""
    return f"{int(runs)}|{_tr.cell_key(cell)}"


def probe_cells(grid: dict[str, list[Any]]) -> list[tuple[str | None, int | None, dict[str, Any]]]:
    """The settings to time: ``(setting, index of its value, the cell)``; the first is the base (``None, None``)."""
    axes = axes_of(grid)
    mids = {name: len(values) // 2 for name, values in axes}
    base = {name: values[mids[name]] for name, values in axes}
    out: list[tuple[str | None, int | None, dict[str, Any]]] = [(None, None, dict(base))]
    varying = sorted([a for a in axes if len(a[1]) >= 2], key=lambda a: -len(a[1]))[:MAX_AXES]
    order = {name: i for i, (name, _v) in enumerate(axes)}
    for name, values in sorted(varying, key=lambda a: order[a[0]]):
        for index in sorted({0, len(values) - 1} - {mids[name]}):
            out.append((name, index, {**base, name: values[index]}))
    return out


#: A setting whose end costs at least this many times the base is "costly".
COSTLY = 2.0
#: Up to this many settings are added one by one when the settings interact; beyond it the mean ratios are used, capped.
ENUM_LIMIT = 50000


class _Label:
    """A probe's label that is not a setting's name (:attr:`Probe.axis` holds either). It equals only itself, so a study
    whose setting is called ``part`` or ``together`` is timed and summed like any other setting."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


#: The label of the probe that times the most expensive setting (every costly end together).
CORNER = _Label("together")
#: The label of a probe that times one setting of a sub-study (``Probe.index``: the number of the sub-study, from 0).
PART = _Label("part")


def _cost_ratio(p: Probe, base: Probe) -> float:
    """How many times the base a setting costs, for deciding which ends are costly. The time inside the simulation is compared
    when both were measured: a slow process start (a loaded machine) is the same for both and would hide a costly end."""
    if p.call_s is not None and base.call_s is not None and base.call_s > 0:
        return max(p.call_s / base.call_s, p.seconds / base.seconds if p.seconds > 0 else 0.0)
    return p.seconds / base.seconds


def costly_ends(grid: dict[str, list[Any]], probes: list[Probe]) -> dict[str, int]:
    """For each setting whose timed end costs at least :data:`COSTLY` times the base, the index of its costlier end."""
    base = next((p for p in probes if p.axis is None), None)
    if base is None or base.seconds <= 0:
        return {}
    out: dict[str, int] = {}
    for name, _values in axes_of(grid):
        ends = [(_cost_ratio(p, base), p.index) for p in probes if p.axis == name and p.index is not None]
        if ends:
            ratio, index = max(ends)
            if ratio >= COSTLY and index is not None:
                out[name] = index
    return out


def corner_cell(grid: dict[str, list[Any]], costly: dict[str, int]) -> dict[str, Any]:
    """The base with every costly setting at its costly end: the most expensive setting the study has."""
    cell = dict(probe_cells(grid)[0][2])
    for name, values in axes_of(grid):
        if name in costly:
            cell[name] = values[costly[name]]
    return cell


def _interpolated(known: dict[int, float], n: int) -> list[float]:
    """The ratio at each of ``n`` positions: the timed ones as measured, the others in proportion between their timed
    neighbours (a straight line on a logarithmic scale), never below the smaller or above the larger of the two."""
    out: list[float] = []
    points = sorted(known)
    for i in range(n):
        if i in known:
            out.append(known[i])
            continue
        lower = max((p for p in points if p < i), default=None)
        upper = min((p for p in points if p > i), default=None)
        if lower is None or upper is None:
            out.append(known[points[0]] if lower is None else known[lower])
            continue
        a, b = math.log(max(known[lower], 1e-9)), math.log(max(known[upper], 1e-9))
        out.append(math.exp(a + (b - a) * (i - lower) / (upper - lower)))
    return out


def estimate(grid: dict[str, list[Any]], runs: int, deterministic: bool, probes: list[Probe]) -> Estimate | None:
    """The total time of the study from the timed settings, or ``None`` when the base was not timed.

    Each setting alone changes the cost by the ratio measured for its values (read in proportion between timed ones). Settings
    are not assumed to multiply: when two or more are each costly, the base with all of them at their costly ends was timed
    too (``together``), and no setting costs more than that one (the study's most expensive setting), nor is any cost taken
    beyond what the product of the ratios measured one by one says; a setting measured dearer than that product raises every
    setting's cost in the same proportion."""
    base = next((p for p in probes if p.axis is None), None)
    if base is None or base.seconds <= 0:
        return None
    axes = axes_of(grid)
    ratios: dict[str, dict[str, float]] = {}
    per_axis: list[list[float]] = []
    complete = sum(1 for p in probes if p.axis not in (CORNER, PART)) == len(probe_cells(grid))
    for name, values in axes:
        mid = len(values) // 2
        known = {mid: 1.0}
        for p in probes:
            if p.axis == name and p.index is not None:
                known[p.index] = p.seconds / base.seconds
                ratios.setdefault(name, {})[str(values[p.index])] = round(p.seconds / base.seconds, 3)
        per_axis.append(_interpolated(known, len(values)))
    cells = cell_count(grid)
    mean_ratio = 1.0
    for per_value in per_axis:
        mean_ratio *= sum(per_value) / len(per_value)
    total_ratio = cells * mean_ratio
    # Sub-studies: the settings of the study are those of each sub-study (not every combination), so the cost is summed over them.
    staged = bool(getattr(grid, "sub_studies", None)) and len(axes) == len(grid)
    if staged:
        total_ratio = _search.grid_weight(grid, {name: per_axis[k] for k, (name, _v) in enumerate(axes)})
    costly = costly_ends(grid, probes)
    corner = next((p for p in probes if p.axis == CORNER), None)
    together = None
    if corner is not None and len(costly) >= 2:
        c = corner.seconds / base.seconds
        product = 1.0
        for (name, _values), per_value in zip(axes, per_axis):
            if name in costly:
                product *= per_value[costly[name]]
        scale = max(1.0, c / product) if product > 0 else 1.0
        together = {"cell": dict(corner.cell), "ratio": round(c, 3)}
        if cells <= ENUM_LIMIT:
            total_ratio = 0.0
            for combo in (itertools.product(*per_axis) if not staged else
                          ([per_axis[k][i] for k, i in enumerate(picked)] for picked in _search.iter_grid_indexes(grid))):
                total_ratio += min(math.prod(combo) * scale, max(c, 1.0))
        else:
            total_ratio = min(total_ratio * scale, cells * max(c, 1.0))
    seconds, at_least = base.seconds * total_ratio, any(p.at_least for p in probes)
    sizes = _search.sub_study_sizes(grid) if staged else []
    timed_parts = {p.index: p for p in probes if p.axis == PART}
    part_costs: list[tuple[str, int, float]] = []
    if sizes and timed_parts:
        # Settings of different parts can cost very different amounts (what a run computes differs), so the cost is summed
        # per part: its settings times the time of a setting of it that was timed (the base's time for a part not timed, and
        # then the total is only a lower bound). The larger of this and the per-setting reading above is taken.
        for k, (name, size) in enumerate(sizes):
            probe = timed_parts.get(k)
            part_costs.append((name, size, probe.seconds if probe is not None else base.seconds))
        seconds = max(seconds, sum(size * cost for _n, size, cost in part_costs))
        at_least = at_least or len(timed_parts) < len(sizes)
    return Estimate(seconds=seconds, at_least=at_least, complete=complete,
                    cells=cells, trials=trial_count(grid, runs, deterministic), ratios=ratios, base=dict(base.cell),
                    base_seconds=base.seconds, timed=len(probes), together=together, spreads=spreads_of(probes),
                    parts=len(sizes) if timed_parts else 0, parts_timed=len(timed_parts), part_costs=part_costs)


def spreads_of(probes: list[Probe]) -> dict[str, float]:
    """The largest spread of each metric over the timed settings (the most cautious reading of the pilot trials)."""
    out: dict[str, float] = {}
    for p in probes:
        for name, sd in p.spread.items():
            out[name] = max(out.get(name, 0.0), sd)
    return out


def fits(est: Estimate, limit_s: float) -> bool:
    return est.seconds * SAFETY <= limit_s


def known_too_long(est: Estimate, limit_s: float, *, timed_again: bool = False) -> bool:
    """Whether the study cannot fit: the padded total is over the limit, and when it is only a lower bound it is over
    already. A lower bound that is under the limit says nothing. ``timed_again``: the smaller run FI chose (with the
    margin) timed a second time; it is over only when it is over the limit without the margin, since the margin was spent
    choosing it and a second timing of the same run differs by about as much."""
    return est.seconds * (1.0 if timed_again else SAFETY) > limit_s


async def measure(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *, runs: int,
    deterministic: bool, limit_s: float, budget_s: float, env: dict[str, str] | None = None,
    thresholds: dict[str, Any] | None = None, cached: dict[str, Probe] | None = None,
    in_each_run: dict[str, list[Any]] | None = None,
) -> tuple[list[Probe] | None, str]:
    """Time the settings of :func:`probe_cells`. ``(probes, "")``, or ``(None, why)`` when the simulation could not be timed
    (it failed, or could not be loaded: the real run's repair is what handles that). ``budget_s`` is the most time all the
    timing together may take; the settings not reached are left out (the estimate then says it is not complete).
    ``cached``: settings already timed on this grid (by their key), not timed again. ``in_each_run``: the values each run
    goes through together; a timed run is given all of them, as the study gives them, so one run's time is one trial's."""
    quest_root = Path(quest_root)
    total_trials = trial_count(grid, runs, deterministic)
    allowance = max(60.0, 3.0 * limit_s / max(1, total_trials))
    started = time.monotonic()
    probes: list[Probe] = []

    async def time_cell(axis: str | None, index: int | None, cell: dict[str, Any], most_s: float = 0.0) -> tuple[bool, str]:
        """Time one setting into ``probes``; ``(False, why)`` when the simulation could not be timed."""
        key = probe_key(cell, runs)
        if cached and key in cached:
            hit = cached[key]
            probes.append(Probe(axis, index, cell, hit.seconds, hit.at_least, dict(hit.spread), [dict(t) for t in hit.trials],
                                hit.fixed_s, hit.per_trial_s, hit.timed_trials, hit.call_s))
            return True, ""
        left = budget_s - (time.monotonic() - started)
        if probes and left < 5:
            return True, "budget"
        timeout = int(max(5, min(max(allowance, most_s), max(left, 60.0) if not probes else left)))
        probe, why = await _time_one(executor, python, quest_root, module, cell, runs=runs, deterministic=deterministic,
                                     timeout_s=timeout, env=env, thresholds=thresholds, in_each_run=in_each_run)
        if probe is None:
            return False, f"at {_tr.cell_key(cell) or 'its only setting'}: {why}"
        probe.axis, probe.index = axis, index
        probes.append(probe)
        return True, ""

    cells = probe_cells(grid)
    # A study of separate parts: one setting of every part is timed right after the base, before the time for timing runs out.
    # A setting of a part that does not finish in (the time allowed / the part's settings) shows the part alone is over the limit,
    # so that is the most a part's setting is given (a lower bound it leaves is then already over).
    parts = [(PART, k, cell) for k, (_name, cell) in enumerate(_search.part_cells(grid))]
    sizes = [size for _name, size in _search.sub_study_sizes(grid)]
    for axis, index, cell in cells[:1] + parts + cells[1:]:
        ok, why = await time_cell(axis, index, cell, math.ceil(limit_s / max(1, sizes[index])) + 1 if axis == PART else 0.0)
        if not ok:
            return None, why
        if why == "budget":
            return probes, ""
        if probes[-1].at_least and axis is None:
            return probes, ""  # the base alone is already past what the study allows: nothing more to learn by timing the rest
    # Settings that are each costly may not multiply: the most expensive setting (all costly ends together) is timed itself.
    costly = costly_ends(grid, probes)
    if len(costly) >= 2:
        ok, why = await time_cell(CORNER, None, corner_cell(grid, costly))
        if not ok:
            return None, why
    return probes, ""


async def _time_one(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, cell: dict[str, Any], *, runs: int,
    deterministic: bool, timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
    in_each_run: dict[str, list[Any]] | None = None,
) -> tuple[Probe | None, str]:
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    (quest_root / _tr.HARNESS_PATH).write_text(_tr.HARNESS_SOURCE, encoding="utf-8")
    spec, out = work / "pilot.json", work / "pilot.out.jsonl"
    out.unlink(missing_ok=True)
    key = _tr.cell_key(cell)
    # A stochastic setting: its first trial, then more until the later ones have taken TIMING_LATER_S (at most
    # TIMING_TRIALS, never more than the setting's runs), so what one run costs is measured, not read off one short trial.
    n = 1 if deterministic else min(max(1, int(runs)), TIMING_TRIALS)
    trials = [{"trial": t, "seed": None if deterministic else _tr.trial_seed(0, key, t)} for t in range(n)]
    nonce = f"pilot{time.time_ns()}"
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    spec.write_text(json.dumps({"module": str(module).replace("\\", "/"), "entry": "run_cell" if deterministic else "run_trial",
                                "cell": {**cell, **inside}, "levels": _in_run.level_cells(inside), "trials": trials,
                                "nonce": nonce, "thresholds": dict(thresholds or {}),
                                **({"stop_after_s": TIMING_LATER_S} if n > 1 else {})},
                               default=str), encoding="utf-8")
    t0 = time.monotonic()
    result = await _outside.execute_keeping_only_finished(
        executor, [str(python), _tr.HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(),
                   out.relative_to(quest_root).as_posix()], cwd=quest_root, timeout_s=timeout_s, env=env)
    wall = time.monotonic() - t0
    try:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        rows = []
    for name in (spec, out):
        name.unlink(missing_ok=True)
    rows = [r for r in rows if isinstance(r, dict) and r.get("nonce") == nonce]
    loaded = next((r for r in rows if r.get("load_error")), None)
    if loaded is not None:
        return None, f"the simulation could not be loaded ({loaded['load_error']})"
    done = sorted((r for r in rows if r.get("status") == "ok"), key=lambda r: int(r.get("trial") or 0))
    failures = [r for r in rows if r.get("status") == "failed"]
    # One trial, or for a stochastic setting at least two (the harness stops once the later ones took TIMING_LATER_S).
    wanted = min(n, 2)
    # A trial that fails now and then is judged as the full run judges it (core/trial_checks.py: more than half failing at
    # a setting is a fault): the timing does not hold the simulation to a stricter rule than the run does. A setting where
    # too few trials finish to time it, or more than that share failed, is a setting the simulation fails at.
    if failures and (len(done) < wanted or len(failures) > _checks.FAULT_FRACTION * (len(done) + len(failures))):
        failed = failures[0]
        where = _tr.last_frame(str(failed.get("traceback") or ""), quest_root)
        return None, f"a trial failed: {failed.get('reason') or 'no reason given'}" + (f" (at {where})" if where else "")
    if getattr(result, "timed_out", False) and len(done) < wanted:
        # Past its allowance with `len(done)` trials in: the next one costs at least what is left of the allowance.
        per_trial = timeout_s / (len(done) + 1)
        seconds = per_trial * (1 if deterministic else max(1, int(runs)))
        return Probe(None, None, cell, seconds, at_least=True, timed_trials=len(done)), ""
    if len(done) < wanted:
        return None, f"the trial did not report (exit code {getattr(result, 'returncode', '?')})"
    durations = [float(r.get("duration_s") or 0.0) for r in done]
    overhead = max(0.0, wall - sum(durations))  # the process, the loading and the harness
    if len(done) == 1:
        seconds, fixed, per_trial = wall, (wall if deterministic else None), (0.0 if deterministic else None)
    else:
        # The process, the loading and the first call once; every later run costs what the later timed ones did on average,
        # from one trial's start to the next (the call and the harness writing its result), when the harness says when.
        starts = [r.get("at_s") for r in done]
        if len(done) >= 3 and all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in starts):
            per_trial = max(0.0, (float(starts[-1]) - float(starts[1])) / (len(done) - 2))
        else:
            per_trial = sum(durations[1:]) / (len(durations) - 1)
        # never negative: later trials slower than the start and the first call would under-cost a small run
        fixed = max(0.0, overhead + durations[0] - per_trial)
        seconds = fixed + int(runs) * per_trial
    spread: dict[str, float] = {}
    if len(done) >= 2:
        # a number per value spreads as the widest of its values does: the precision is wanted at every one of them
        values: dict[str, dict[Any, list[float]]] = {}
        for r in done:
            for k, name, v in _in_run.flat("", dict(r.get("values") or {})):
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    values.setdefault(str(name), {}).setdefault(k, []).append(float(v))
        for name, by_value in values.items():
            for xs in by_value.values():
                if len(xs) >= 2:
                    spread[name] = max(spread.get(name, 0.0), statistics.stdev(xs))  # the sample standard deviation
    return Probe(None, None, cell, max(seconds, 1e-6), spread=spread,
                 trials=[dict(r.get("values") or {}) for r in done[:2]], fixed_s=fixed, per_trial_s=per_trial,
                 timed_trials=len(done), call_s=sum(durations)), ""


# ---- the correctness pilot -----------------------------------------------------------------------------------------------

#: Settings that take at most this many values are tried in every combination before a metric is called undefined.
SMALL_AXIS = 3
#: The most settings tried for that (one trial each); beyond it nothing is claimed about a metric that looks undefined.
CONFIRM_MAX = 40
#: A metric is called constant only when it was returned in at least this many pilot trials, at this many different settings.
MIN_CONSTANT_TRIALS = 6
MIN_CONSTANT_SETTINGS = 3
#: A proportion is called stuck at a bound only when it was returned in at least this many pilot trials at this many settings.
MIN_STUCK_TRIALS = 4
MIN_STUCK_SETTINGS = 2
PILOT_RECORD = Path(".fi") / "pilot_check.json"


@dataclass
class PilotProblem:
    """What the pilot found wrong with one metric (or with the whole output), in plain words."""

    metric: str
    kind: str  # "undefined" | "constant" | "flat"
    says: str


@dataclass
class Pilot:
    """What the pilot trials showed: how many trials at how many settings, the metrics read, and what was wrong."""

    trials: int = 0
    settings: int = 0
    checked: list[str] = field(default_factory=list)
    problems: list[PilotProblem] = field(default_factory=list)
    #: Metrics that looked undefined but whose subsets could not all be tried: nothing is claimed about them.
    unsure: list[str] = field(default_factory=list)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def observations(probes: list[Probe]) -> list[tuple[dict[str, Any], dict[str, float]]]:
    """Every pilot trial as ``(its setting, what it returned)``. A run that compared values inside it is its setting's
    numbers per run, and one observation per value with that value's numbers (its setting names the value too)."""
    out: list[tuple[dict[str, Any], dict[str, float]]] = []
    for p in probes:
        for t in p.trials:
            per_run, per_level = _in_run.split(t)
            if per_run or not per_level:  # a run that gives numbers per value only has nothing per run
                out.append((dict(p.cell), dict(per_run)))
            for lk, named in per_level.items():
                out.append(({**p.cell, **_in_run.parse_level_key(lk)}, dict(named)))
    return out


def _settings_text(obs: list[tuple[dict[str, Any], dict[str, float]]], limit: int = 6) -> str:
    seen: list[str] = []
    for cell, _v in obs:
        key = ", ".join(f"{k} = {v}" for k, v in cell.items())
        if key not in seen:
            seen.append(key)
    shown = "; ".join(seen[:limit])
    return shown + (f"; and {len(seen) - limit} more" if len(seen) > limit else "")


def _setting_count(obs: list[tuple[dict[str, Any], dict[str, float]]], inside: Iterable[str] = ()) -> int:
    """How many settings the observations were made at; ``inside``: the names of what each run compares inside it, which
    are values of a run and not settings (left out of the count)."""
    skip = set(inside)
    return len({_tr.cell_key({k: v for k, v in c.items() if k not in skip}) for c, _v in obs})


def undefined_metrics(obs: list[tuple[dict[str, Any], dict[str, float]]], ids: list[str]) -> list[str]:
    """The metrics (of ``ids``) that the pilot trials returned and that were not a number in any of them."""
    out = []
    for name in ids:
        values = [v[name] for _c, v in obs if name in v]
        if values and not any(_finite(x) for x in values):
            out.append(name)
    return out


def confirm_cells(grid: dict[str, list[Any]], ran: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """``(settings still to try, whether that covers everything)``: every combination of the settings that take at most
    :data:`SMALL_AXIS` values and every value of every setting on its own (the others at the base value), without those
    already run. ``False`` when there are more than :data:`CONFIRM_MAX` of them: only the first are returned and the set is
    not whole."""
    axes = axes_of(grid)
    base = dict(probe_cells(grid)[0][2])
    small = [(n, v) for n, v in axes if 2 <= len(v) <= SMALL_AXIS]
    wanted: list[dict[str, Any]] = [
        {**base, **dict(zip((n for n, _v in small), values))} for values in itertools.product(*(v for _n, v in small))]
    for name, values in axes:
        wanted += [{**base, name: value} for value in values]
    done = {_tr.cell_key(c) for c in ran}
    todo: list[dict[str, Any]] = []
    for cell in wanted:
        key = _tr.cell_key(cell)
        if key not in done:
            done.add(key)
            todo.append(cell)
    return todo[:CONFIRM_MAX], len(todo) <= CONFIRM_MAX


def _say_undefined(name: str, obs: list[tuple[dict[str, Any], dict[str, float]]], tried_all: str) -> str:
    own = [(c, v) for c, v in obs if name in v]
    sample = sorted({repr(v[name]) for _c, v in own})[:3]
    return (f"the metric `{name}` was not a number in any of the {len(own)} pilot trials, at any of the {_setting_count(own)} "
            f"settings tried ({_settings_text(own)}){tried_all}. The values that came back: {', '.join(sample)}. A metric may be "
            "undefined at the settings it does not apply to, but it must have a value at the settings it does apply to")


def pilot_problems(obs: list[tuple[dict[str, Any], dict[str, float]]], kinds: dict[str, str], *,
                   exempt: set[str] | frozenset[str] = frozenset(), tried_all: str = "",
                   unsure: set[str] | frozenset[str] = frozenset(), axes: set[str] | frozenset[str] = frozenset(),
                   source: str | None = None, inside: Iterable[str] = ()) -> Pilot:
    """What the pilot trials show. ``inside``: the names of what each run compares inside it (``protocol.in_each_run``), whose
    values are not settings. ``kinds``: the protocol's metrics (``{id: kind}``); only those the trials returned are read.
    ``tried_all``: a clause saying that every combination of the small settings was tried (when it was); ``unsure``: metrics
    that looked undefined but whose subsets were not all tried (nothing is claimed about them). ``exempt``: names the plan
    expects to stay put. ``source``: the text of the simulation, for the evidence a value that never changed needs (see the
    module's notes); without it only an undefined metric is a finding. Nothing here is a judgement about the science: a metric
    undefined everywhere, and, when the code shows it is built from no setting, one that never moves while others do and an
    output that is the same everywhere."""
    from . import flat_output as _flat

    pilot = Pilot(trials=len(obs), settings=_setting_count(obs, inside))
    returned = sorted({n for _c, v in obs for n in v})
    pilot.checked = [n for n in kinds if n in returned]
    if not obs:
        return pilot
    for name in undefined_metrics(obs, pilot.checked):
        if name in unsure:
            pilot.unsure.append(name)
            continue
        pilot.problems.append(PilotProblem(name, "undefined", _say_undefined(name, obs, tried_all)))
    ignoring = set(_flat.unused_outputs(source, pilot.checked) or []) if source is not None else set()
    moved = {n for n in returned
             if len({v[n] for _c, v in obs if n in v and _finite(v[n])}) > 1}
    varied_cells = {_tr.cell_key(c) for c, _v in obs}
    if len(obs) >= MIN_CONSTANT_TRIALS and len(varied_cells) >= MIN_CONSTANT_SETTINGS and moved:
        for name in pilot.checked:
            if kinds.get(name) != "mean" or name in exempt or name in axes or name in moved or name not in ignoring:
                continue
            own = [v[name] for _c, v in obs if name in v]
            if len(own) >= MIN_CONSTANT_TRIALS and all(_finite(x) for x in own) and _flat._all_same([float(x) for x in own]):
                pilot.problems.append(PilotProblem(
                    name, "constant",
                    f"the metric `{name}` was exactly {float(own[0]):g} in all {len(own)} pilot trials, at "
                    f"{_setting_count([(c, v) for c, v in obs if name in v])} settings ({_settings_text(obs)}), although "
                    f"other results changed with the settings: a result that never responds to them is not computed from them"))
    stuck: list[tuple[str, float, list[tuple[dict[str, Any], dict[str, float]]]]] = []
    if len(varied_cells) >= 2:
        for name in pilot.checked:
            if kinds.get(name) != "proportion" or name in exempt or name in axes or name not in ignoring:
                continue
            own_obs = [(c, v) for c, v in obs if name in v]
            own = [v[name] for _c, v in own_obs]
            if (len(own) >= MIN_STUCK_TRIALS and _setting_count(own_obs) >= MIN_STUCK_SETTINGS
                    and all(_finite(x) for x in own) and (all(float(x) == 0.0 for x in own) or all(float(x) == 1.0 for x in own))):
                stuck.append((name, float(own[0]), own_obs))
    # Only proportions stuck at the SAME bound count together: one at 1 beside its complement at 0 (a success and a
    # failure fraction) is what a deterministic study gives, not a sign that the simulated object is degenerate.
    by_bound = {b: [t for t in stuck if t[1] == b] for b in (0.0, 1.0)}
    stuck = max(by_bound.values(), key=len)
    if len(stuck) >= 2:
        names = ", ".join(f"`{n}`" for n, _b, _o in stuck)
        for name, bound, own_obs in stuck:
            pilot.problems.append(PilotProblem(
                name, "stuck",
                f"the proportion `{name}` was exactly {bound:g} in all {len(own_obs)} pilot trials, at {_setting_count(own_obs)} "
                f"settings ({_settings_text(own_obs)}), and so were {', '.join(f'`{n}`' for n, _b, _o in stuck if n != name)}: "
                f"{len(stuck)} proportions ({names}) each gave the same outcome in every trial, which a simulated object that "
                "is sampled does not do"))
    by_cell: dict[str, Any] = {}
    for cell, values in obs:
        run = by_cell.setdefault(_tr.cell_key(cell), type("PilotCell", (), {
            "cell": dict(cell), "rows": [], "level": any(k in cell for k in inside)})())
        run.rows.append({"status": "ok", "values": values})
    flat = _flat.flat_sweep(list(by_cell.values()), source=source, expected=set(exempt))
    if flat:
        pilot.problems.append(PilotProblem("", "flat", f"in the {len(obs)} pilot trials, {flat}"))
    return pilot


def directive(pilot: Pilot) -> str:
    """The plain message the simulation's repair is given: what the pilot trials showed, exactly, and what to make true.
    No expected value is in it."""
    return (
        f"FI ran {pilot.trials} trials of the simulation at {pilot.settings} settings before the full study, to check that "
        "it measures what the protocol names, and found:\n"
        + "\n".join(f"- {p.says}" for p in pilot.problems)
        + "\n\nRepair the simulation so that each of these results is computed from the trial's inputs and has a value "
        "wherever it applies. Keep everything else as it is."
    )


async def run_values(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, cell: dict[str, Any], *, deterministic: bool,
    timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
    in_each_run: dict[str, list[Any]] | None = None,
) -> dict[str, Any] | None:
    """What one trial of ``cell`` returns (nothing is written to FI's ledger), or ``None`` when it failed or did not finish.
    With values compared inside each run, the run is given all of them and may return a number per value."""
    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    (quest_root / _tr.HARNESS_PATH).write_text(_tr.HARNESS_SOURCE, encoding="utf-8")
    spec, out = work / "pilot.json", work / "pilot.out.jsonl"
    out.unlink(missing_ok=True)
    key = _tr.cell_key(cell)
    nonce = f"pilot{time.time_ns()}"
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    spec.write_text(json.dumps({"module": str(module).replace("\\", "/"), "entry": "run_cell" if deterministic else "run_trial",
                                "cell": {**cell, **inside}, "levels": _in_run.level_cells(inside),
                                "trials": [{"trial": 0, "seed": None if deterministic else _tr.trial_seed(0, key, 0)}],
                                "nonce": nonce, "thresholds": dict(thresholds or {})}, default=str), encoding="utf-8")
    await _outside.execute_keeping_only_finished(
        executor, [str(python), _tr.HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(),
                   out.relative_to(quest_root).as_posix()], cwd=quest_root, timeout_s=timeout_s, env=env)
    try:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        rows = []
    for name in (spec, out):
        name.unlink(missing_ok=True)
    row = next((r for r in rows if isinstance(r, dict) and r.get("nonce") == nonce and r.get("status") == "ok"), None)
    return dict(row.get("values") or {}) if row else None


async def pilot(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, grid: dict[str, list[Any]],
    probes: list[Probe], kinds: dict[str, str], *, deterministic: bool, budget_s: float, allowance_s: float,
    env: dict[str, str] | None = None, thresholds: dict[str, Any] | None = None,
    exempt: set[str] | frozenset[str] = frozenset(), in_each_run: dict[str, list[Any]] | None = None,
) -> Pilot:
    """Read what the timed trials returned (:func:`pilot_problems`). When a metric looks undefined everywhere, the settings with
    few values are first tried in every combination, one trial each, within ``budget_s``: a metric that applies to only some
    of the settings has a value there. If they cannot all be tried, the metric is left alone (nothing is claimed)."""
    obs = observations(probes)
    axes = {name for name, _v in axes_of(grid)}
    try:
        source = (Path(quest_root) / str(module)).read_text(encoding="utf-8", errors="replace")
    except OSError:
        source = None
    suspects = undefined_metrics(obs, [n for n in kinds if any(n in v for _c, v in obs)])
    tried_all, unsure = "", set()
    if suspects:
        inside = set(in_each_run or ())
        todo, whole = confirm_cells(grid, [{k: v for k, v in c.items() if k not in inside} for c, _v in obs])
        started = time.monotonic()
        reached = True
        for cell in todo:
            left = budget_s - (time.monotonic() - started)
            if left < 5:
                reached = False
                break
            values = await run_values(executor, python, quest_root, module, cell, deterministic=deterministic,
                                      timeout_s=int(max(5, min(allowance_s, left))), env=env, thresholds=thresholds,
                                      in_each_run=in_each_run)
            if values is None:
                reached = False
                continue
            obs.extend(observations([Probe(None, None, dict(cell), 0.0, trials=[values])]))
        if whole and reached:
            tried_all = (f"; FI also tried every value of every setting and every combination of the settings that take few "
                         f"values ({_setting_count(obs, tuple(in_each_run or ()))} settings in all)")
        else:
            unsure = set(suspects)
    return pilot_problems(obs, kinds, exempt=exempt, tried_all=tried_all, unsure=unsure, axes=axes, source=source,
                          inside=tuple(in_each_run or ()))


# ---- the record: so a resume neither times nor asks again -------------------------------------------------------------


def read_pilot_record(quest_root: Path) -> dict[str, Any]:
    try:
        record = json.loads((Path(quest_root) / PILOT_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def write_pilot_record(quest_root: Path, record: dict[str, Any]) -> bool:
    try:
        path = Path(quest_root) / PILOT_RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


# ---- the record: so a resume neither times nor asks again -------------------------------------------------------------


def read_record(quest_root: Path) -> dict[str, Any]:
    try:
        record = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def write_record(quest_root: Path, record: dict[str, Any]) -> bool:
    try:
        path = Path(quest_root) / RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


# ---- words ---------------------------------------------------------------------------------------------------------------


def says(est: Estimate, limit_s: float, *, topic: bool = False) -> str:
    """The one plain line for run.log and the console. ``topic``: the limit is the one the topic states."""
    whose = "the topic's" if topic else "the"
    line = (f"FI estimates the experiment takes {'at least ' if est.at_least else 'about '}"
            f"{plain_duration(est.seconds * SAFETY)}; {whose} limit is {plain_duration(limit_s)}")
    if est.parts and est.parts_timed < est.parts:
        line += (f" (the time FI allows for timing ran out after {est.parts_timed} of the study's {est.parts} parts, so the "
                 "rest are counted at the cost of the base: this is a lower bound)")
    return line


def progress_line(done_cells: int, cells: int, done_trials: int, trials: int, elapsed_s: float) -> str:
    """``"3 of 96 settings done, about 6 h left"``: where the study is, for the "still running" line. The time left is what the
    trials done so far took, for the trials still to do; with none done yet it is not guessed."""
    if cells <= 0:
        return ""
    head = f"{done_cells} of {cells} setting{'s' if cells != 1 else ''} done"
    if done_trials <= 0 or trials <= done_trials or elapsed_s <= 0:
        return head
    return f"{head}, about {plain_duration(elapsed_s / done_trials * (trials - done_trials))} left"


def _stated_min(protocol: dict[str, Any] | None) -> int | None:
    """The plan's own `protocol.min_runs_per_setting`, or ``None``."""
    least = protocol.get("min_runs_per_setting") if isinstance(protocol, dict) else None
    return int(least) if isinstance(least, (int, float)) and not isinstance(least, bool) and least >= 1 else None


def precision_known(protocol: dict[str, Any] | None, spreads: dict[str, float] | None = None) -> bool:
    """Whether the runs a precision target needs can be worked out before the run: always, but for a target on a mean
    whose spread the timed trials did not show (fewer than two trials, or all equal)."""
    if not isinstance(protocol, dict):
        return True
    precision = protocol.get("precision")
    target = precision.get("target_half_width") if isinstance(precision, dict) else None
    if not (isinstance(target, (int, float)) and not isinstance(target, bool) and float(target) > 0):
        return True
    metric, kind = precision_metric(protocol)
    if kind != "mean":
        return True
    sd = (spreads or {}).get(metric or "")
    return isinstance(sd, (int, float)) and sd > 0


def precision_metric(protocol: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """``(the metric the precision target is on, its kind)``: `protocol.precision.metric` looked up among `protocol.metrics`; with
    no metric named, the only metric the plan has. ``(name, None)`` when its kind is not known."""
    if not isinstance(protocol, dict):
        return None, None
    precision = protocol.get("precision")
    named = str(precision.get("metric")).strip() if isinstance(precision, dict) and precision.get("metric") else None
    metrics = [m for m in protocol.get("metrics") or [] if isinstance(m, dict)]
    pick = next((m for m in metrics if named and str(m.get("id")) == named), None)
    if pick is None and not named and len(metrics) == 1:
        pick = metrics[0]
    if pick is not None:
        return str(pick.get("id")), (str(pick.get("kind")) if pick.get("kind") else None)
    return named, None


def runs_floor(protocol: dict[str, Any] | None, before_runs: int, spreads: dict[str, float] | None = None) -> int:
    """The fewest runs per setting a smaller run may have: the plan's own minimum (`protocol.min_runs_per_setting`) and what its
    stated precision needs, never more than the run already had; 1 when the plan states neither.

    The precision target is read by the kind of the metric it is on: a proportion needs `trials_for_half_width` (the worst case,
    a probability near 0.5); a mean needs n >= (z * s / h) ** 2 with `s` the spread the timed trials of the same setting
    showed (`spreads`, two trials or more), and when that spread is not known or is zero no run may be cut at all."""
    from .stats import _Z95, trials_for_half_width

    floor = 1
    if isinstance(protocol, dict):
        least = protocol.get("min_runs_per_setting")
        if isinstance(least, (int, float)) and not isinstance(least, bool) and least >= 1:
            floor = max(floor, int(least))
        precision = protocol.get("precision")
        target = precision.get("target_half_width") if isinstance(precision, dict) else None
        if isinstance(target, (int, float)) and not isinstance(target, bool) and 0 < float(target):
            metric, kind = precision_metric(protocol)
            if kind == "mean":
                sd = (spreads or {}).get(metric or "")
                if sd is None or not sd > 0:
                    return int(before_runs)  # the spread cannot be estimated: no cut below the plan's own runs
                floor = max(floor, int(math.ceil(round((_Z95 * sd / float(target)) ** 2, 9))))
            else:
                need = trials_for_half_width(float(target))
                if need:
                    floor = max(floor, need)
    return min(int(before_runs), floor)


def runs_checks_need(protocol: dict[str, Any] | None, grid: dict[str, list[Any]]) -> int:
    """The most runs per setting any check of the plan needs to be judged (``core.criteria.runs_per_setting_needed``: a
    standard-error criterion needs 32 per setting); 0 when none needs any."""
    from .criteria import runs_per_setting_needed

    return runs_per_setting_needed(protocol, cell_count(grid) if grid else 1)


def fewest_for_topic(protocol: dict[str, Any] | None, grid: dict[str, list[Any]], runs: int) -> int:
    """The fewest runs per setting the topic's own limit may take a run down to: :data:`MIN_RUNS_FOR_LIMIT`, or what the
    plan's checks need when that is more (:func:`runs_checks_need`), never more than the run already has."""
    return min(int(runs), max(MIN_RUNS_FOR_LIMIT, runs_checks_need(protocol, grid)))


def floor_is_stated(protocol: dict[str, Any] | None) -> bool:
    """Whether the plan itself states how few runs per setting it may have (`protocol.min_runs_per_setting`, or a precision
    target): only then may FI lower the runs to fit its own limit (``execution.timeout_s``). The topic's own time limit is
    the one exception: it may take the runs down to :data:`MIN_RUNS_FOR_LIMIT` whatever the plan states, and the paper then
    names the cut with the numbers (:func:`fit_runs`)."""
    if not isinstance(protocol, dict):
        return False
    least = protocol.get("min_runs_per_setting")
    if isinstance(least, (int, float)) and not isinstance(least, bool) and least >= 1:
        return True
    precision = protocol.get("precision")
    target = precision.get("target_half_width") if isinstance(precision, dict) else None
    return isinstance(target, (int, float)) and not isinstance(target, bool) and float(target) > 0


def seconds_at(grid: dict[str, list[Any]], runs: int, runs_now: int, deterministic: bool, probes: list[Probe]) -> float | None:
    """The estimate (without the margin) of the same study with ``runs`` runs per setting, from the settings timed for
    ``runs_now``: each setting's process and first call are paid once whatever the runs, the rest scales with them."""
    est = estimate(grid, runs, deterministic, [p.at(runs, runs_now) for p in probes])
    return None if est is None else est.seconds


@dataclass
class Fit:
    """FI's smaller run: ``runs`` per setting, every setting kept. ``below_plan`` when that is fewer than the plan's own
    minimum (``plan_floor``: its `min_runs_per_setting` and what its precision target needs, ``None`` when the plan states
    neither or the target's needs are not known), which only the topic's own time limit may ask for. ``precision_known``:
    False for a precision target on a mean whose spread the timed trials did not show, so the precision those runs give is
    not known before the run."""

    runs: int
    seconds: float
    plan_floor: int | None
    below_plan: bool
    precision_known: bool = True


def fit_runs(grid: dict[str, list[Any]], runs: int, deterministic: bool, probes: list[Probe], protocol: dict[str, Any] | None,
             *, limit_s: float, topic_limit: bool, spreads: dict[str, float] | None = None) -> Fit | None:
    """The most runs per setting (fewer than ``runs``) whose estimate, with its margin, fits ``limit_s``, every setting kept;
    ``None`` when none does. For FI's own limit the fewest allowed is the plan's own (:func:`runs_floor`, when the plan
    states one: FI invents no number of runs for its own limit, and with a target on a mean whose spread is not known it
    does not cut at all). For the limit the topic itself states (``topic_limit``) the runs may go down to
    :data:`MIN_RUNS_FOR_LIMIT`, below the plan's minimum or with none stated: the topic's limit comes before the plan's
    runs, and the paper names the cut with its numbers. Neither goes below the runs the plan's checks need to be judged
    (:func:`runs_checks_need`). The result is timed again before it is used."""
    if deterministic or int(runs) <= 1 or not probes:
        return None
    known = precision_known(protocol, spreads)
    if not floor_is_stated(protocol):
        floor = None
    elif known:
        floor = runs_floor(protocol, runs, spreads)
    else:
        # The precision target's needs are not known, so the plan's fewest is only its stated minimum, if any: never the
        # plan's own runs read as a "minimum".
        stated = _stated_min(protocol)
        floor = min(int(runs), stated) if stated is not None else None
        if not topic_limit:
            return None  # FI's own limit never cuts below a precision it cannot work out
    need = runs_checks_need(protocol, grid)
    if topic_limit:
        lowest: int | None = max(MIN_RUNS_FOR_LIMIT, need)
    else:
        lowest = None if floor is None else max(floor, need)
    if lowest is None or lowest >= int(runs):
        return None

    def fits_at(n: int) -> float | None:
        s = seconds_at(grid, n, runs, deterministic, probes)
        return s if s is not None and s * SAFETY <= limit_s else None

    if fits_at(lowest) is None:
        return None
    lo, hi = lowest, int(runs) - 1  # lo fits; find the largest that does (the cost grows with the runs)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits_at(mid) is not None:
            lo = mid
        else:
            hi = mid - 1
    return Fit(runs=lo, seconds=float(fits_at(lo) or 0.0), plan_floor=floor, below_plan=floor is not None and lo < floor,
               precision_known=known)


def expected_precision(protocol: dict[str, Any] | None, runs: int, spreads: dict[str, float] | None = None) -> dict[str, Any] | None:
    """The 95% interval half-width ``runs`` runs per setting give, for the metric the plan's precision target is on (or, with
    no target, its first proportion, else its first mean whose spread the timed trials showed): a proportion at its worst
    case (near 0.5), a mean from the spread the timed trials showed. ``None`` when there is nothing to say it of (no such
    metric, or a mean whose spread is not known)."""
    from .stats import _Z95

    if not isinstance(protocol, dict) or runs < 1:
        return None
    metric, kind = precision_metric(protocol)
    precision = protocol.get("precision") if isinstance(protocol.get("precision"), dict) else {}
    target = precision.get("target_half_width")
    target = float(target) if isinstance(target, (int, float)) and not isinstance(target, bool) and target > 0 else None
    if metric is None:
        metrics = [m for m in protocol.get("metrics") or [] if isinstance(m, dict) and m.get("id")]
        first = next((m for m in metrics if m.get("kind") == "proportion"), None)
        if first is not None:
            metric, kind = str(first.get("id")), "proportion"
        else:
            spread_known = [m for m in metrics if m.get("kind") == "mean" and isinstance((spreads or {}).get(str(m["id"])), (int, float))
                            and (spreads or {})[str(m["id"])] > 0]
            if not spread_known:
                return None
            metric, kind = str(spread_known[0]["id"]), "mean"
    if kind == "mean":
        sd = (spreads or {}).get(metric)
        if not isinstance(sd, (int, float)) or not sd > 0:
            return None
        half, how = _Z95 * float(sd) / math.sqrt(runs), "from the spread the timed trials showed"
    else:
        half, how = _Z95 * math.sqrt(0.25 / runs), "at most: a proportion's interval is widest near 0.5"
    return {"metric": metric, "runs": int(runs), "half_width": round(half, 4), "how": how, "target": target}
