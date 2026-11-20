"""Make the simulation of a study faster before the study runs, without changing a result.

Every study that does not fit the time allowed, and every study that fits but that FI expects to take
:data:`WORTH_SPEEDING_UP_S` or more (its padded whole-study estimate), gets this step after the correctness checks and
before the study is sized (a study that does not fit is sped up before it is cut). FI runs a fixed sample of
the study itself, before and after each edit of the simulation: the same settings, the same trials, the same seeds. An
edit is kept only when every number the sample returns is the same (to nine digits, or different only at rounding level:
by no more than :data:`ROUNDING` of how much that number varies over the sample and of the number itself,
:func:`compare`; a speed-up must not
change what the study measures) and the sample is faster by at least :data:`MIN_SAVING` and by more than the timing's own noise
(:func:`fast_enough`), and then again in repeated timings of both versions, taken alternately, that do not overlap
(:func:`clearly_faster`). Nothing the code prints and no time the code reports is read: the harness times the calls.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import in_run as _in_run
from . import outside_run as _outside
from . import run_estimate as _estimate
from . import trial_runner as _tr

#: Trials of each sampled setting (the first is a warm-up and is not timed).
SAMPLE_TRIALS = 12
#: The sample is run this many times; the faster of the runs is the time (noise is slower, never faster).
REPEATS = 2
#: An edit is kept only when it saves at least this share of the sample's time (and more than the timing's noise).
MIN_SAVING = 0.10
#: A change that passes that first look is timed again, this many times each, old and new alternating (old, new, old,
#: new, ...) so a machine that gets busier or quieter hits both. At most ``2 * CONFIRM_PAIRS`` more runs of the sample,
#: and only for a change that already looks faster; a round's own time is set against its saving by the rounds' stop.
CONFIRM_PAIRS = 4
RTOL, ATOL = 1e-9, 1e-12
#: Two numbers are also the same when they differ only at rounding level: by no more than this share of how much that
#: number varies over the unchanged code's sample (its largest value less its smallest, over every setting and trial),
#: and of the number itself. An edit that only adds the same terms in another order (a vectorised sum, a fused loop)
#: moves a result by about 1e-12 to 1e-8 of itself, which nine digits rejected; a result an edit gets wrong moves by far
#: more than a millionth of its range.
ROUNDING = 1e-6
#: A study that fits the time allowed and that FI expects to take less than this (padded, the whole study) is not sped
#: up: the step would cost about as much time as it could save. A study that does not fit is always sped up first.
WORTH_SPEEDING_UP_S = 600.0


def worth_speeding_up(padded_s: float | None, fits: bool = True) -> bool:
    """Whether a study is sped up before it runs: always when it does not fit the time allowed (before it is cut,
    whatever its length); when it fits, only when its padded whole-study estimate (the search and FI's check at finer
    settings included) is at least :data:`WORTH_SPEEDING_UP_S`."""
    if not fits:
        return True
    return padded_s is not None and float(padded_s) >= WORTH_SPEEDING_UP_S


def not_worth_it_line(padded_s: float) -> str:
    """The one line said when a study that fits is too short to be sped up."""
    return (f"The study fits and is expected to take about {_estimate.plain_duration(padded_s)}, under "
            f"{_estimate.plain_duration(WORTH_SPEEDING_UP_S)}, so FI does not try to make the simulation faster.")


@dataclass
class Sample:
    seconds: float = 0.0
    values: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    failed: str = ""
    #: The wall time the sample took in all (its processes included), for what a round of the step costs.
    wall_s: float = 0.0
    #: The time of each run of the sample (``seconds`` is the least of them), for judging a speed-up on repeats.
    times: list[float] = field(default_factory=list)


def sample_cells(grid: dict[str, list[Any]], oracles: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The settings of the sample: the ones FI times anyway (the base and each setting's two ends), and the setting of
    each known-answer case of the plan (the other settings taken from the base), so a speed-up is held to the cases the
    plan's checks run, too."""
    from . import oracle_check as _oracle

    cells = [dict(cell) for _axis, _index, cell in _estimate.probe_cells(grid)]
    base = dict(cells[0]) if cells else {}
    for o in oracles or []:
        case = _oracle.case_of(o)
        if case and isinstance(case[0], dict):
            cell = {**base, **{k: v for k, v in case[0].items() if k in base}}
            if cell not in cells:
                cells.append(cell)
    return cells


#: A sampled setting may take this many times what the timing measured for it before FI stops it.
SAMPLE_ALLOWANCE = 3.0


def bounded_sample(cells: list[dict[str, Any]], probes: list[Any] | None, *, runs: int, deterministic: bool,
                   budget_s: float, samples: int) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    """The settings of the sample FI can afford, and the time each may take, from what the timing measured of that very
    setting: a setting the timing did not see finish (``at_least``) or never ran is left out (its cost is not known; the
    step times each setting of the sample the study's timing never ran before it calls this, :func:`untimed_cells`), and
    each is given :data:`SAMPLE_ALLOWANCE` times its own measured cost. The
    ``samples`` samples the step runs (before, and after each edit) together fit in ``budget_s``: the costliest settings
    are left out first. ``([], {}, why)`` when none can be sampled in that time; otherwise ``why`` says how many settings
    were left out, or is empty."""
    by_key = {_tr.cell_key(p.cell): p for p in probes or [] if getattr(p, "cell", None) is not None}
    n = 1 if deterministic else SAMPLE_TRIALS

    def cost(p: Any) -> float:
        if getattr(p, "fixed_s", None) is not None and getattr(p, "per_trial_s", None) is not None:
            return max(0.0, float(p.fixed_s)) + n * max(0.0, float(p.per_trial_s))
        return float(p.seconds) * (1 if deterministic else n / max(1, int(runs)))

    costed: list[tuple[float, dict[str, Any]]] = []
    for cell in cells:
        p = by_key.get(_tr.cell_key(cell))
        if p is None or getattr(p, "at_least", False):
            continue
        costed.append((cost(p), cell))
    unfinished = len(cells) - len(costed)
    if not costed:
        return [], {}, "FI's timing did not see any of the sample's settings finish, so how long they take is not known"
    share = budget_s / max(1, samples) / REPEATS
    kept = list(costed)
    while kept and sum(c for c, _cell in kept) > share:
        kept.remove(max(kept, key=lambda item: item[0]))
    if not kept:
        cheapest = min(c for c, _cell in costed)
        return [], {}, (f"one setting of the sample takes about {_estimate.plain_duration(cheapest)}, more than the "
                        f"{_estimate.plain_duration(share)} the time for timing leaves each sample")
    order = {id(cell): i for i, cell in enumerate(cells)}
    kept.sort(key=lambda item: order[id(item[1])])
    over = len(costed) - len(kept)
    reasons = ([f"FI's timing did not see {unfinished} finish"] if unfinished else []) + (
        [f"{over} would not fit in the time left for timing"] if over else [])
    note = (f"{unfinished + over} of the sample's {len(cells)} settings are left out ({'; '.join(reasons)})"
            if reasons else "")
    return ([cell for _c, cell in kept],
            {_tr.cell_key(cell): max(60, math.ceil(SAMPLE_ALLOWANCE * c)) for c, cell in kept}, note)


#: How much coarser a copy of a sampled setting is made, in turn, when no setting of the sample finishes in the time for
#: timing: each of the plan's numerical settings (a mesh step, a time step, a number of points) this many times coarser.
COARSER = (2, 4, 8, 16)


def own_budget_s(budget_s: float, padded_s: float, limit_s: float, frac: float) -> float:
    """The time the step may spend timing and sampling: what the study's timing left it, and never less than the least
    of the share of the time allowed the timing itself is given (``engine.pilot_timeout_frac``) and the least saving FI
    keeps on the study (:data:`MIN_SAVING` of its padded estimate). The timing of a slow study spends its whole share,
    so the step used to get about nothing of its own."""
    own = min(max(0.0, float(frac)) * max(0.0, float(limit_s)), MIN_SAVING * max(0.0, float(padded_s)))
    return max(max(0.0, float(budget_s)), own)


def numerical_settings(protocol: dict[str, Any] | None) -> dict[str, str]:
    """The plan's numerical settings and the direction that is finer (``smaller`` or ``larger``): ``protocol``'s
    ``numerical_settings``, and a search's (``protocol.optimisation.numerical_settings``). Only these may be made coarser
    for a copy of a setting: they are the plan's own word for how finely the simulation computes, not what it studies."""
    protocol = protocol if isinstance(protocol, dict) else {}
    search = protocol.get("optimisation")
    search = search if isinstance(search, dict) else {}
    out: dict[str, str] = {}
    for block in (protocol.get("numerical_settings"), search.get("numerical_settings")):
        if not isinstance(block, dict):
            continue
        for name, spec in block.items():
            finer = str(spec.get("finer") or "").strip().lower() if isinstance(spec, dict) else ""
            if finer in ("smaller", "larger"):
                out[str(name)] = finer
    return out


def coarser_copy(cell: dict[str, Any], settings: dict[str, str], factor: int) -> dict[str, Any] | None:
    """``cell`` with each of the plan's numerical settings it holds made ``factor`` times coarser (a ``smaller``-is-finer
    step multiplied, a ``larger``-is-finer count divided, a whole number kept whole and at least 1). ``None`` when no
    setting of the cell changes, so there is no coarser copy."""
    out = dict(cell)
    changed = False
    for name, finer in settings.items():
        value = cell.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            continue
        new: float = value * factor if finer == "smaller" else value / factor
        if isinstance(value, int):
            new = max(1, int(round(new)))
        if new != value:
            out[name] = new
            changed = True
    return out if changed else None


def real_values(probe: Any) -> bool:
    """Whether a timed setting finished with real numbers: not cut by its time (``at_least``), and every number its timed
    trials returned is finite (a copy too coarse for the simulation returns not-a-number, or nothing, with a reason)."""
    if probe is None or getattr(probe, "at_least", False):
        return False
    trials = [t for t in getattr(probe, "trials", None) or [] if isinstance(t, dict)]
    numbers = [v for t in trials for v in t.values()]
    return bool(numbers) and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                 for v in numbers)


def untimed_cells(cells: list[dict[str, Any]], probes: list[Any] | None) -> list[dict[str, Any]]:
    """The settings of the sample the study's timing never ran (a known-answer case's setting, a search's design): each is
    timed on its own before the sample is chosen, so its time limit comes from its own cost, not from another setting's."""
    seen = {_tr.cell_key(p.cell) for p in probes or [] if getattr(p, "cell", None) is not None}
    return [cell for cell in cells if _tr.cell_key(cell) not in seen]


def sample_share_s(budget_s: float, samples: int) -> float:
    """The most one run of the sample may take so that every sample of the step fits in ``budget_s``."""
    return budget_s / max(1, samples) / REPEATS


def why_not_run(rows: list[dict[str, Any]], result: Any, timeout_s: float, quest_root: Path | str | None = None) -> str:
    """Why a run of the sample did not give every trial: the simulation could not be loaded, a trial failed (its reason
    and where), the run reached its time limit (with the limit), or it ended without reporting (its exit code)."""
    loaded = next((r for r in rows if r.get("load_error")), None)
    if loaded is not None:
        return f"the simulation could not be loaded: {str(loaded['load_error'])[:300]}"
    failed = next((r for r in rows if r.get("status") == "failed"), None)
    if failed is not None:
        where = _tr.last_frame(str(failed.get("traceback") or ""), quest_root)
        return f"a trial failed: {str(failed.get('reason') or 'no reason given')[:300]}" + (f" (at {where})" if where else "")
    if getattr(result, "timed_out", False):
        return f"the run reached its time limit of {_estimate.plain_duration(float(timeout_s))}"
    return f"the run ended without reporting (exit code {getattr(result, 'returncode', '?')})"


async def run_sample(executor: Any, python: Path | str, quest_root: Path, module: str, cells: list[dict[str, Any]], *,
                     deterministic: bool, thresholds: dict[str, Any] | None, in_each_run: dict[str, list[Any]] | None,
                     env: dict[str, str] | None, timeout_s: int, log: Any = None,
                     timeouts: dict[str, int] | None = None, repeats: int = REPEATS) -> Sample:
    """Run ``SAMPLE_TRIALS`` fixed-seed trials of each cell (``repeats`` times, ``REPEATS`` unless said) and return the values of the first run and
    the least time any run took (the calls only, the first trial of each cell left out). A run that does not give every
    trial ends the sample, and ``failed`` says at which setting and why (:func:`why_not_run`)."""
    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    (quest_root / _tr.HARNESS_PATH).write_text(_tr.HARNESS_SOURCE, encoding="utf-8")
    ex = _outside.OutsideExecutor(executor, quest_root, log=log)
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    out_sample = Sample()
    started = time.monotonic()
    best: float | None = None
    for rep in range(max(1, int(repeats))):
        total = 0.0
        values: dict[str, list[dict[str, Any]]] = {}
        for cell in cells:
            key = _tr.cell_key(cell)
            n = 1 if deterministic else SAMPLE_TRIALS
            spec, out = work / "speed.json", work / "speed.out.jsonl"
            out.unlink(missing_ok=True)
            nonce = f"speed{time.time_ns()}"
            spec.write_text(json.dumps({
                "module": module.replace("\\", "/"), "entry": "run_cell" if deterministic else "run_trial",
                "cell": {**cell, **inside}, "kind": "scan", "levels": _in_run.levels_json(_in_run.level_cells(inside)),
                "trials": [{"trial": t, "seed": None if deterministic else _tr.trial_seed(0, key, t)} for t in range(n)],
                "nonce": nonce, "thresholds": dict(thresholds or {})}, default=str), encoding="utf-8")
            limit = (timeouts or {}).get(key, timeout_s)
            result = await ex.execute([str(python), _tr.HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(),
                                       out.relative_to(quest_root).as_posix()], cwd=quest_root, timeout_s=limit, env=env)
            try:
                rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, ValueError):
                rows = []
            for p in (spec, out):
                p.unlink(missing_ok=True)
            done = sorted((r for r in rows if isinstance(r, dict) and r.get("nonce") == nonce and r.get("status") == "ok"),
                          key=lambda r: int(r.get("trial") or 0))
            if len(done) < n:
                ours = [r for r in rows if isinstance(r, dict) and r.get("nonce") in (nonce, None)]
                out_sample.failed = (f"{len(done)} of {n} trials ran at {key or 'its only setting'}: "
                                     f"{why_not_run(ours, result, limit, quest_root)}")
                out_sample.wall_s = time.monotonic() - started
                return out_sample
            values[key] = [dict(r.get("values") or {}) for r in done]
            total += sum(float(r.get("duration_s") or 0.0) for r in done[1:] or done)
        if rep == 0:
            out_sample.values = values
        out_sample.times.append(total)
        best = total if best is None else min(best, total)
    out_sample.seconds = float(best or 0.0)
    out_sample.wall_s = time.monotonic() - started
    return out_sample


def _numbers(x: Any, path: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_numbers(v, f"{path}/{k}"))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(_numbers(v, f"{path}[{i}]"))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out[path] = float(x)
    return out


def _leaves(x: Any, path: str = "") -> dict[str, Any]:
    """Every value a row holds, by its path (``/a/b``, ``/a[2]``), with its own type: a float, a whole number, a yes/no,
    a text or anything else the row returns."""
    out: dict[str, Any] = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_leaves(v, f"{path}/{k}"))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(_leaves(v, f"{path}[{i}]"))
    else:
        out[path] = x
    return out


def spreads(*samples: Sample) -> dict[str, float]:
    """How much each number varies over ``samples``: its largest value less its smallest, by name, over every setting and
    trial (finite values only, so a not-a-number or an infinity cannot widen it)."""
    low: dict[str, float] = {}
    high: dict[str, float] = {}
    for sample in samples:
        for rows in sample.values.values():
            for row in rows:
                for name, v in _numbers(row).items():
                    if math.isfinite(v):
                        low[name] = min(v, low.get(name, v))
                        high[name] = max(v, high.get(name, v))
    return {name: high[name] - low[name] for name in low}


def _kind(v: Any) -> str:
    if isinstance(v, bool):
        return "yes/no"
    if isinstance(v, float):
        return "float"
    if isinstance(v, int):
        return "whole"
    return type(v).__name__


def _same(x: Any, y: Any, spread: float | None) -> tuple[bool, float | None]:
    """Whether ``x`` (the unchanged code's) and ``y`` are the same value, and, when they are only at rounding level, the
    difference as a share of ``spread``. Only two floats may differ at all: by nine digits, or, with ``spread``, by no
    more than :data:`ROUNDING` of both ``spread`` and the number itself. Two not-a-numbers are the same, and so is an
    infinity that stays the same infinity; a finite number that becomes infinite or not-a-number (or back), or an
    infinity that changes sign, is not. Whole numbers, yes/no values and texts are the same only when equal (a whole
    number and a float that is equal to it are the same)."""
    if _kind(x) == "float" and _kind(y) == "float":
        if (math.isnan(x) and math.isnan(y)) or x == y:
            return True, None
        if not (math.isfinite(x) and math.isfinite(y)):
            return False, None
        gap = abs(x - y)
        if gap <= ATOL + RTOL * abs(y):
            return True, None
        # and of the number itself: one huge value of it in the sample (a stand-in for "no answer", say) widens its
        # range so far that any change would fit in a millionth of it
        if spread and spread > 0 and gap <= ROUNDING * min(spread, max(abs(x), abs(y))):
            return True, gap / spread
        return False, None
    numeric = {"float", "whole"}
    if _kind(x) in numeric and _kind(y) in numeric:
        return x == y, None  # a whole number is never "the same" at rounding level: equal or changed
    return _kind(x) == _kind(y) and x == y, None


def _shown(v: Any) -> str:
    return f"{v:.12g}" if _kind(v) in ("float", "whole") else repr(v)


def _differences(before: Sample, after: Sample, scale: dict[str, float] | None = None
                 ) -> tuple[list[tuple[str, int | None, str, str]], list[tuple[str, int, str, float]]]:
    """Every place the values of two samples are not the same (:func:`_same`), as ``(setting, trial, name, text)``; a
    setting not run in both has ``trial`` and ``name`` empty. With ``scale`` (how much each number varies,
    :func:`spreads`), a float that differs only at rounding level is the same: those are returned apart, as
    ``(setting, trial, name, difference as a share of how much it varies)``. Without it, only nine digits count."""
    bad: list[tuple[str, int | None, str, str]] = []
    rounded: list[tuple[str, int, str, float]] = []
    for key in sorted(before.values.keys() | after.values.keys()):
        a, b = before.values.get(key), after.values.get(key)
        if a is None or b is None or len(a) != len(b):
            bad.append((key, None, "", f"{key}: not run in both"))
            continue
        for t, (ra, rb) in enumerate(zip(a, b)):
            na, nb = _leaves(ra), _leaves(rb)
            for name in sorted(na.keys() | nb.keys()):
                if name not in na or name not in nb:
                    bad.append((key, t, name, f"{key} trial {t} {name}: only in one"))
                    continue
                x, y = na[name], nb[name]
                same, share = _same(x, y, (scale or {}).get(name) if scale is not None else None)
                if same and share is not None:
                    rounded.append((key, t, name, share))
                elif not same:
                    bad.append((key, t, name, f"{key} trial {t} {name}: {_shown(x)} -> {_shown(y)}"))
    return bad, rounded


@dataclass
class Comparison:
    """What :func:`compare` found: the numbers that changed (``bad``, as text), and how many differ only at rounding level
    (``rounded``) with the largest of those differences as a share of how much that number varies (``largest``)."""
    bad: list[str] = field(default_factory=list)
    rounded: int = 0
    largest: float = 0.0

    def note(self) -> str:
        """The plain words for the numbers that differ only at rounding level; empty when there are none."""
        if not self.rounded:
            return ""
        return (f"{self.rounded} number(s) differ only by rounding, by at most {self.largest:.1g} of how much each "
                "varies over the sample")


def compare(before: Sample, after: Sample, *, ignore: set[tuple[str, int, str]] | frozenset[Any] = frozenset(),
            reference: tuple[Sample, ...] = ()) -> Comparison:
    """Whether ``after`` gives the same numbers as ``before``, the unchanged code's sample. A number is the same when it
    is equal to nine digits, or when it differs only at rounding level: by no more than :data:`ROUNDING` of how much that
    number varies over ``before`` (and over ``reference``, more samples of the unchanged code, for a sample of one setting
    whose own numbers do not vary), and of the number itself (one huge value in the sample cannot widen the range
    so far that a real change fits in it). A missing or a new number, a not-a-number for a number, and any larger difference
    count as changed. The ``(setting, trial, name)`` of ``ignore`` are left out (see :func:`varying`)."""
    bad, rounded = _differences(before, after, spreads(before, *reference))
    out = Comparison(bad=[text for key, t, name, text in bad if (key, t, name) not in ignore])
    shares = [share for key, t, name, share in rounded if (key, t, name) not in ignore]
    out.rounded, out.largest = len(shares), max(shares, default=0.0)
    return out


def disagreements(before: Sample, after: Sample, *, ignore: set[tuple[str, int, str]] | frozenset[Any] = frozenset(),
                  reference: tuple[Sample, ...] = ()) -> list[str]:
    """Where the sample's numbers are not the same (:func:`compare`), by setting, trial and name."""
    return compare(before, after, ignore=ignore, reference=reference).bad


def varying(first: Sample, again: Sample) -> set[tuple[str, int, str]]:
    """The numbers that are not the same, to nine digits, in two runs of the same, unchanged code (same settings, same
    seeds): the code does not fix them, so they cannot tell whether an edit changed a result. A number in only one of the
    runs counts. Nine digits here, as before (not rounding level): a number that moves at all between two runs of the
    same code is left out of every comparison."""
    return {(key, t, name) for key, t, name, _text in _differences(first, again)[0] if t is not None}


def count_numbers(sample: Sample) -> int:
    """How many values the sample holds (by setting, trial and name), every value compared (:func:`_differences`)."""
    return sum(len(_leaves(r)) for rows in sample.values.values() for r in rows)


def fast_enough(before_s: float, after_s: float, noise_s: float = 0.0) -> bool:
    """Whether an edit made the sample clearly faster: it saves at least :data:`MIN_SAVING` of the time before
    (``before_s - after_s >= MIN_SAVING * before_s``) AND more than the timing's own noise, the difference between two
    samples of the unchanged code (``before_s - after_s > noise_s``)."""
    saved = before_s - after_s
    return before_s > 0 and saved >= MIN_SAVING * before_s and saved > max(0.0, noise_s)


def clearly_faster(old_times: list[float], new_times: list[float]) -> bool:
    """Whether repeated timings show the new version faster: EVERY timing of the new version is below EVERY timing of the old
    one (the gain is larger than the spread of the repeats, so a change that does nothing, whose timings overlap the
    old ones, is not kept), AND the medians differ by at least :data:`MIN_SAVING` of the old median. A busy machine slows
    some runs, but the two versions alternate (:func:`alternate`), so it hits both. Needs at least two timings of each."""
    if len(old_times) < 2 or len(new_times) < 2:
        return False
    old_med, new_med = statistics.median(old_times), statistics.median(new_times)
    return old_med > 0 and max(new_times) < min(old_times) and old_med - new_med >= MIN_SAVING * old_med


async def alternate(time_old: Any, time_new: Any, pairs: int = CONFIRM_PAIRS) -> tuple[list[float], list[float], str]:
    """Time the old and the new version ``pairs`` times each, alternating old, new, old, new... ``time_old`` and ``time_new``
    are async callables that put that version's code in place, run the sample once and return a :class:`Sample`. Returns
    ``(old times, new times, why)``; ``why`` is not empty when a run of the sample did not run (the timings so far are
    then not enough to judge)."""
    old_times: list[float] = []
    new_times: list[float] = []
    for _ in range(max(1, int(pairs))):
        for call, into in ((time_old, old_times), (time_new, new_times)):
            sample = await call()
            if sample.failed or sample.seconds <= 0:
                return old_times, new_times, sample.failed or "no time measured"
            into.append(sample.seconds)
    return old_times, new_times, ""


def can_afford_repeats(new_run_s: float, old_run_s: float, saves_on_study_s: float) -> bool:
    """Whether the repeats of :func:`confirm` cost no more than the change claims to save on the study: ``CONFIRM_PAIRS``
    runs of the sample of each version, taking ``new_run_s`` and ``old_run_s`` of wall time. A long sample is timed once,
    as before, since the repeats would cost more than the change could give back."""
    return CONFIRM_PAIRS * (max(0.0, new_run_s) + max(0.0, old_run_s)) <= max(0.0, saves_on_study_s)


async def confirm(root: Path, old_files: Any, new_files: Any, sample: Any, entry: dict[str, Any], after: Sample,
                  saves_on_study_s: float, *, before_s: float = 0.0, ignore: Any = frozenset(),
                  guards_touched: Any = None, unchanged: Sample | None = None,
                  seen: Comparison | None = None) -> bool:
    """The second look at a change that looked faster at first (one timing of each version is not a measurement of speed):
    the old and the new code are timed alternately (:func:`alternate`; ``sample(1)`` runs the sample once) and the change
    stays only when the timings do not overlap (:func:`clearly_faster`) and every run of the new code returns the same
    numbers (:func:`compare`) as ``unchanged``, the unchanged code's sample, how much each number varies taken from it
    (its first run, ``after``, when none is given; ``ignore``: the numbers that differ between runs of the same code, see
    :func:`varying`). ``seen`` (the first comparison's :class:`Comparison`) takes the most numbers and the largest share at
    rounding level any repeat showed, so the record covers every comparison. Skipped (returns True) when the repeats would cost more than the change saves
    (:func:`can_afford_repeats`, on the wall time of a run of each version). ``guards_touched()`` lists the files FI judges the code by that
    the runs changed (``outcome`` is then ``aborted``). Sets ``outcome`` and ``reason`` of ``entry`` when the change is not
    kept, and ``repeats_old`` / ``repeats_new`` when it was timed again. Leaves the new code in place."""
    from . import improve as _improve

    per_run_s = max(after.seconds, after.wall_s / max(1, len(after.times) or REPEATS))
    # the old code is slower by as much as the first look showed (``before_s``: its sample time, ``after.seconds`` the new one's)
    old_run_s = per_run_s * max(1.0, before_s / max(after.seconds, 1e-9))
    if not can_afford_repeats(per_run_s, old_run_s, saves_on_study_s):
        return True

    async def run_old() -> Any:
        _improve.restore(root, old_files)
        return await sample(1)

    async def run_new() -> Any:
        _improve.restore(root, new_files)
        out = await sample(1)
        if not out.failed:
            found = compare(unchanged if unchanged is not None else after, out, ignore=ignore)
            if found.bad:
                out.failed = f"{len(found.bad)} value(s) of the sample changed, first: {found.bad[0]}"
            elif seen is not None:
                seen.rounded, seen.largest = max(seen.rounded, found.rounded), max(seen.largest, found.largest)
        return out

    try:
        old_t, new_t, why = await alternate(run_old, run_new)
    except BaseException:
        _improve.restore(root, old_files)  # whatever went wrong, the code is back to the version that was kept
        raise
    _improve.restore(root, new_files)
    entry["repeats_old"] = [round(t, 3) for t in old_t]
    entry["repeats_new"] = [round(t, 3) for t in new_t]
    touched = list(guards_touched() or []) if guards_touched else []
    if touched:
        entry.update(outcome="aborted", reason="its repeated runs changed " + ", ".join(touched[:4]), touched=touched)
        return False
    if why:
        entry.update(outcome="not kept", reason=f"the repeated timings did not run ({why})")
        return False
    if not clearly_faster(old_t, new_t):
        entry.update(outcome="not kept",
                     reason=f"not faster: timed again, old and new alternating, the old version took {min(old_t):.3f} to "
                            f"{max(old_t):.3f} s and the new one {min(new_t):.3f} to {max(new_t):.3f} s (the first look, "
                            f"{after.seconds:.3f} s, was one timing); they overlap or differ by less than a tenth")
        return False
    return True


#: How many functions of the measured breakdown the model is shown.
PROFILE_TOP = 10


async def profile_text(executor: Any, python: Path | str, quest_root: Path, module: str, cell: dict[str, Any], *,
                       deterministic: bool, thresholds: dict[str, Any] | None, in_each_run: dict[str, list[Any]] | None,
                       env: dict[str, str] | None, timeout_s: int, log: Any = None) -> str:
    """Where one setting of the sample spends its time, measured by FI (Python's own profiler around the same harness the
    sample runs): the :data:`PROFILE_TOP` functions that take the most time themselves, each with how often it was called
    and the time spent in it and in what it calls. The model is shown what costs the time instead of guessing it from
    the code. Empty when the setting could not be profiled."""
    import pstats

    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    (quest_root / _tr.HARNESS_PATH).write_text(_tr.HARNESS_SOURCE, encoding="utf-8")
    key = _tr.cell_key(cell)
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    spec, out, prof = work / "profile.json", work / "profile.out.jsonl", work / "profile.prof"
    for path in (out, prof):
        path.unlink(missing_ok=True)
    n = 1 if deterministic else 2
    spec.write_text(json.dumps({
        "module": module.replace("\\", "/"), "entry": "run_cell" if deterministic else "run_trial",
        "cell": {**cell, **inside}, "kind": "scan", "levels": _in_run.levels_json(_in_run.level_cells(inside)),
        "trials": [{"trial": t, "seed": None if deterministic else _tr.trial_seed(0, key, t)} for t in range(n)],
        "nonce": f"profile{time.time_ns()}", "thresholds": dict(thresholds or {}),
        "profile": prof.relative_to(quest_root).as_posix()}, default=str), encoding="utf-8")
    ex = _outside.OutsideExecutor(executor, quest_root, log=log)
    try:
        # The profile's path is named in the environment too, so the outside copy brings the file back.
        named = {**(env if env is not None else dict(os.environ)), "FI_PROFILE_OUT": str(prof)}
        await ex.execute([str(python), _tr.HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(),
                          out.relative_to(quest_root).as_posix()], cwd=quest_root, timeout_s=timeout_s, env=named)
        stats = pstats.Stats(str(prof))
    except Exception:  # noqa: BLE001 -- no breakdown is no breakdown; the step goes on without it
        return ""
    finally:
        for path in (spec, out):
            path.unlink(missing_ok=True)
    rows = []
    total = 0.0
    for (file, line, name), (_cc, calls, own, cum, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        total += own
        if "trial_harness" in file or name.startswith("<"):
            continue
        where = f"{Path(file).name}:{line}" if file and file != "~" else "built-in"
        rows.append((own, calls, cum, f"{name} ({where})"))
    prof.unlink(missing_ok=True)
    rows.sort(key=lambda r: -r[0])
    if not rows or total <= 0:
        return ""
    lines = [f"- {label}: {own:.2f} s of its own ({own / total:.0%} of the setting), called {calls:,} times, "
             f"{cum:.2f} s with what it calls" for own, calls, cum, label in rows[:PROFILE_TOP]]
    return (f"Where the time goes, measured by FI on one setting of the sample ({total:.1f} s in all; the functions that "
            "take the most time themselves):\n" + "\n".join(lines))


def speed_block(est_seconds: float, limit_s: float, before: Sample, cells: list[dict[str, Any]], runs: int,
                profile: str = "") -> str:
    per_cell = before.seconds / max(1, len(cells))
    if est_seconds > limit_s:
        need = f"It has to be about {est_seconds / max(limit_s, 1e-9):.1f} times faster to fit."
    else:
        need = "It fits, but it is long: every change that makes it faster shortens the run."
    return (f"FI estimates the whole study at about {_estimate.plain_duration(est_seconds)}; the time allowed is "
            f"{_estimate.plain_duration(limit_s)}. {need}\n"
            f"FI timed {len(cells)} settings ({SAMPLE_TRIALS} trials each, fixed seeds): about "
            f"{per_cell * 1000:.0f} ms of computing per setting on average for the trials after the first, in the "
            f"code below. The study runs {runs} trials of each setting."
            + (f"\n\n{profile}" if profile else ""))
