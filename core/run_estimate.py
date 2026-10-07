"""How long the whole run will take, measured on a few trials before it starts.

A study is every setting of the plan's grid times its runs per setting, each setting in its own process
(``core/trial_runner.py``). Before the full run FI times a handful of real trials, adds them up and compares the total
with the time the study is allowed (``execution.timeout_s``):

* **What is timed.** One *base* setting (the middle value of every setting that varies) and, for each varying setting, the
  base with that one setting at its first and at its last value: at most ``1 + 2 * MAX_AXES`` settings. A numerical
  resolution that dominates the cost (a finer mesh, a smaller step) is a setting like any other, so it shows up as a
  large factor between the base and one end. Each timed setting runs its first trial (the process start, the loading of
  the simulation and the first call, as the real run pays them) and, for a stochastic simulation, a second one, which
  is what the remaining trials of the setting cost.
* **What is added up.** The cost of the setting a value belongs to is the base's cost times, for each varying setting,
  the ratio measured for that value (values in between are read between their timed neighbours, in proportion), so the
  total is the number of settings times the base's cost times the mean ratio of each varying setting. A trial that does
  not finish within its allowance is a lower bound, and the estimate says it is "at least".
* **What it is compared with.** The estimate with a margin of :data:`SAFETY` (one half more) against the time allowed.

None of the timed trials is a result: they are run apart from the study, write nothing into FI's ledger and are not
kept. Everything here is arithmetic on the measurements; nothing is a judgement about the science.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import trial_runner as _tr

#: The margin added to the measured total before it is compared with the time allowed.
SAFETY = 1.5
#: The most varying settings that are timed one by one (the first ones, largest range first); the rest are taken as 1.
MAX_AXES = 6
#: How many times FI asks the plan to make the run smaller before it stops.
ASKS = 2
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

    axis: str | None
    index: int | None
    cell: dict[str, Any]
    seconds: float
    at_least: bool = False


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
    """The total time of the study from the timed settings, or ``None`` when the base was not timed."""
    base = next((p for p in probes if p.axis is None), None)
    if base is None or base.seconds <= 0:
        return None
    axes = axes_of(grid)
    factor = 1.0
    ratios: dict[str, dict[str, float]] = {}
    complete = len(probes) == len(probe_cells(grid))
    for name, values in axes:
        mid = len(values) // 2
        known = {mid: 1.0}
        for p in probes:
            if p.axis == name and p.index is not None:
                known[p.index] = p.seconds / base.seconds
                ratios.setdefault(name, {})[str(values[p.index])] = round(p.seconds / base.seconds, 3)
        per_value = _interpolated(known, len(values))
        factor *= sum(per_value) / len(per_value)
    cells = cell_count(grid)
    return Estimate(seconds=cells * base.seconds * factor, at_least=any(p.at_least for p in probes), complete=complete,
                    cells=cells, trials=trial_count(grid, runs, deterministic), ratios=ratios, base=dict(base.cell),
                    base_seconds=base.seconds, timed=len(probes))


def fits(est: Estimate, limit_s: float) -> bool:
    return est.seconds * SAFETY <= limit_s


def known_too_long(est: Estimate, limit_s: float) -> bool:
    """Whether the study cannot fit: the padded total is over the limit, and when it is only a lower bound it is over
    already. A lower bound that is under the limit says nothing."""
    return est.seconds * SAFETY > limit_s


async def measure(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *, runs: int,
    deterministic: bool, limit_s: float, budget_s: float, env: dict[str, str] | None = None,
    thresholds: dict[str, Any] | None = None, cached: dict[str, Probe] | None = None,
) -> tuple[list[Probe] | None, str]:
    """Time the settings of :func:`probe_cells`. ``(probes, "")``, or ``(None, why)`` when the simulation could not be timed
    (it failed, or could not be loaded: the real run's repair is what handles that). ``budget_s`` is the most time all the
    timing together may take; the settings not reached are left out (the estimate then says it is not complete).
    ``cached``: settings already timed on this grid (by their key), not timed again."""
    quest_root = Path(quest_root)
    total_trials = trial_count(grid, runs, deterministic)
    allowance = max(60.0, 3.0 * limit_s / max(1, total_trials))
    started = time.monotonic()
    probes: list[Probe] = []
    for axis, index, cell in probe_cells(grid):
        key = probe_key(cell, runs)
        if cached and key in cached:
            probes.append(Probe(axis, index, cell, cached[key].seconds, cached[key].at_least))
            continue
        left = budget_s - (time.monotonic() - started)
        if probes and left < 5:
            break
        timeout = int(max(5, min(allowance, max(left, 60.0) if not probes else left)))
        probe, why = await _time_one(executor, python, quest_root, module, cell, runs=runs, deterministic=deterministic,
                                     timeout_s=timeout, env=env, thresholds=thresholds)
        if probe is None:
            return None, why
        probe.axis, probe.index = axis, index
        probes.append(probe)
        if probe.at_least and axis is None:
            break  # the base alone is already past what the study allows: nothing more to learn by timing the rest
    return probes, ""


async def _time_one(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, cell: dict[str, Any], *, runs: int,
    deterministic: bool, timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
) -> tuple[Probe | None, str]:
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    (quest_root / _tr.HARNESS_PATH).write_text(_tr.HARNESS_SOURCE, encoding="utf-8")
    spec, out = work / "pilot.json", work / "pilot.out.jsonl"
    out.unlink(missing_ok=True)
    key = _tr.cell_key(cell)
    n = 1 if deterministic else (2 if int(runs) >= 2 else 1)
    trials = [{"trial": t, "seed": None if deterministic else _tr.trial_seed(0, key, t)} for t in range(n)]
    nonce = f"pilot{time.time_ns()}"
    spec.write_text(json.dumps({"module": str(module).replace("\\", "/"), "entry": "run_cell" if deterministic else "run_trial",
                                "cell": cell, "trials": trials, "nonce": nonce, "thresholds": dict(thresholds or {})},
                               default=str), encoding="utf-8")
    t0 = time.monotonic()
    result = await executor.execute(
        [str(python), _tr.HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(),
         out.relative_to(quest_root).as_posix()], cwd=quest_root, timeout_s=timeout_s, env=env)
    wall = time.monotonic() - t0
    try:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        rows = []
    for name in (spec, out):
        name.unlink(missing_ok=True)
    rows = [r for r in rows if isinstance(r, dict) and r.get("nonce") == nonce]
    if any(r.get("load_error") for r in rows):
        return None, "the simulation could not be loaded"
    done = sorted((r for r in rows if r.get("status") == "ok"), key=lambda r: int(r.get("trial") or 0))
    if any(r.get("status") == "failed" for r in rows):
        return None, "a trial failed"
    if getattr(result, "timed_out", False) and len(done) < n:
        # Past its allowance with `len(done)` trials in: the next one costs at least what is left of the allowance.
        per_trial = timeout_s / (len(done) + 1)
        seconds = per_trial * (1 if deterministic else max(1, int(runs)))
        return Probe(None, None, cell, seconds, at_least=True), ""
    if len(done) < n:
        return None, "the trial did not report"
    durations = [float(r.get("duration_s") or 0.0) for r in done]
    if n == 1:
        seconds = wall
    else:
        # The process, the loading and the first call once; the later trials cost what the second one did.
        seconds = max(0.0, wall - sum(durations)) + durations[0] + (int(runs) - 1) * durations[1]
    return Probe(None, None, cell, max(seconds, 1e-6)), ""


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


def says(est: Estimate, limit_s: float) -> str:
    """The one plain line for run.log and the console."""
    return (f"FI estimates the experiment takes {'at least ' if est.at_least else 'about '}"
            f"{plain_duration(est.seconds * SAFETY)}; the limit is {plain_duration(limit_s)}")


def progress_line(done_cells: int, cells: int, done_trials: int, trials: int, elapsed_s: float) -> str:
    """``"3 of 96 settings done, about 6 h left"``: where the study is, for the "still running" line. The time left is what the
    trials done so far took, for the trials still to do; with none done yet it is not guessed."""
    if cells <= 0:
        return ""
    head = f"{done_cells} of {cells} setting{'s' if cells != 1 else ''} done"
    if done_trials <= 0 or trials <= done_trials or elapsed_s <= 0:
        return head
    return f"{head}, about {plain_duration(elapsed_s / done_trials * (trials - done_trials))} left"


def request(est: Estimate, limit_s: float, runs: int, deterministic: bool) -> str:
    """What the plan's model is asked, with the measured numbers: shrink the run, change nothing that decides whether a
    result is right."""
    lines = []
    for name, per in est.ratios.items():
        shown = "; ".join(f"`{name}` = {v} costs {r:g} times the time of `{name}` = {est.base.get(name)}" for v, r in per.items())
        lines.append(f"- {shown}")
    per_trial = "" if deterministic else f", each setting {int(runs)} runs"
    return (
        "The experiment this plan describes would take longer than the time it is allowed on this machine, so it cannot "
        f"finish. FI timed a few real trials: the whole experiment ({est.cells} settings{per_trial}) would take "
        f"{'at least ' if est.at_least else 'about '}{plain_duration(est.seconds * SAFETY)}, and it is allowed "
        f"{plain_duration(limit_s)}.\n"
        + ("How the time changes with each setting, as measured:\n" + "\n".join(lines) + "\n" if lines else "")
        + f"Make the experiment smaller so that it takes about {plain_duration(limit_s / 2)} or less: use fewer runs per "
        "setting (`protocol.runs_per_setting`) and/or fewer or coarser settings in `protocol.grid` (keep the same setting "
        "names; a numerical resolution may be made coarser, a domain smaller, a list of values shorter). Write only "
        "`protocol.grid` and `protocol.runs_per_setting` again.\n"
        "Change nothing else in the plan: not a check, a threshold, a tolerance or a criterion, and not any other part of "
        "the protocol."
    )
