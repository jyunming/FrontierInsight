"""Make a study that does not fit the time allowed fit, without changing a result.

FI runs a fixed sample of the study itself, before and after one edit of the simulation: the same settings, the same
trials, the same seeds. The edit is kept only when every number the sample returns is the same (to nine digits; a
speed-up must not change what the study measures) and the time the sample takes drops by a stated margin. Nothing the
code prints and no time the code reports is read: the harness times the calls.
"""

from __future__ import annotations

import json
import math
import os
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
#: An edit is kept only when the sample takes at most this share of the time it took before.
KEEP_BELOW = 0.75
RTOL, ATOL = 1e-9, 1e-12
#: The speed-up is sought until the sample is this much faster than the study needs (a margin for the noise of a timing).
MARGIN = 1.25


@dataclass
class Sample:
    seconds: float = 0.0
    values: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    failed: str = ""


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
    """The settings of the sample FI can afford, and the time each may take, from what the timing measured: a setting
    the timing did not see finish (``at_least``) is left out (its cost is not known, only that it is large), a setting
    the timing never ran is costed as the base, and each is given :data:`SAMPLE_ALLOWANCE` times its measured cost. The
    ``samples`` samples the step runs (before, and after each edit) together fit in ``budget_s``: the costliest settings
    are left out first. ``([], {}, why)`` when none can be sampled in that time; otherwise ``why`` says how many settings
    were left out, or is empty."""
    by_key = {_tr.cell_key(p.cell): p for p in probes or [] if getattr(p, "cell", None) is not None}
    base = next((p for p in probes or [] if getattr(p, "axis", 0) is None), None)
    n = 1 if deterministic else SAMPLE_TRIALS

    def cost(p: Any) -> float:
        if getattr(p, "fixed_s", None) is not None and getattr(p, "per_trial_s", None) is not None:
            return max(0.0, float(p.fixed_s)) + n * max(0.0, float(p.per_trial_s))
        return float(p.seconds) * (1 if deterministic else n / max(1, int(runs)))

    costed: list[tuple[float, dict[str, Any]]] = []
    for cell in cells:
        p = by_key.get(_tr.cell_key(cell)) or base
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


async def run_sample(executor: Any, python: Path | str, quest_root: Path, module: str, cells: list[dict[str, Any]], *,
                     deterministic: bool, thresholds: dict[str, Any] | None, in_each_run: dict[str, list[Any]] | None,
                     env: dict[str, str] | None, timeout_s: int, log: Any = None,
                     timeouts: dict[str, int] | None = None) -> Sample:
    """Run ``SAMPLE_TRIALS`` fixed-seed trials of each cell (``REPEATS`` times) and return the values of the first run and
    the least time any run took (the calls only, the first trial of each cell left out)."""
    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    (quest_root / _tr.HARNESS_PATH).write_text(_tr.HARNESS_SOURCE, encoding="utf-8")
    ex = _outside.OutsideExecutor(executor, quest_root, log=log)
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    out_sample = Sample()
    best: float | None = None
    for rep in range(REPEATS):
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
                "cell": {**cell, **inside}, "levels": _in_run.levels_json(_in_run.level_cells(inside)),
                "trials": [{"trial": t, "seed": None if deterministic else _tr.trial_seed(0, key, t)} for t in range(n)],
                "nonce": nonce, "thresholds": dict(thresholds or {})}, default=str), encoding="utf-8")
            await ex.execute([str(python), _tr.HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(),
                              out.relative_to(quest_root).as_posix()], cwd=quest_root,
                             timeout_s=(timeouts or {}).get(key, timeout_s), env=env)
            try:
                rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, ValueError):
                rows = []
            for p in (spec, out):
                p.unlink(missing_ok=True)
            done = sorted((r for r in rows if isinstance(r, dict) and r.get("nonce") == nonce and r.get("status") == "ok"),
                          key=lambda r: int(r.get("trial") or 0))
            if len(done) < n:
                out_sample.failed = f"{len(done)} of {n} trials ran at {key or 'its only setting'}"
                return out_sample
            values[key] = [dict(r.get("values") or {}) for r in done]
            total += sum(float(r.get("duration_s") or 0.0) for r in done[1:] or done)
        if rep == 0:
            out_sample.values = values
        best = total if best is None else min(best, total)
    out_sample.seconds = float(best or 0.0)
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


def _differences(before: Sample, after: Sample) -> list[tuple[str, int | None, str, str]]:
    """Every place the numbers of two samples are not the same, as ``(setting, trial, name, text)``; a setting not run in
    both has ``trial`` and ``name`` empty."""
    bad: list[tuple[str, int | None, str, str]] = []
    for key in sorted(before.values.keys() | after.values.keys()):
        a, b = before.values.get(key), after.values.get(key)
        if a is None or b is None or len(a) != len(b):
            bad.append((key, None, "", f"{key}: not run in both"))
            continue
        for t, (ra, rb) in enumerate(zip(a, b)):
            na, nb = _numbers(ra), _numbers(rb)
            for name in sorted(na.keys() | nb.keys()):
                if name not in na or name not in nb:
                    bad.append((key, t, name, f"{key} trial {t} {name}: only in one"))
                    continue
                x, y = na[name], nb[name]
                if math.isnan(x) and math.isnan(y):
                    continue
                if not abs(x - y) <= ATOL + RTOL * abs(y):
                    bad.append((key, t, name, f"{key} trial {t} {name}: {x:.12g} -> {y:.12g}"))
    return bad


def disagreements(before: Sample, after: Sample, *, ignore: set[tuple[str, int, str]] | frozenset[Any] = frozenset()) -> list[str]:
    """Where the sample's numbers are not the same (a missing or a new number counts), by setting, trial and name. The
    ``(setting, trial, name)`` of ``ignore`` are left out (see :func:`varying`)."""
    return [text for key, t, name, text in _differences(before, after) if (key, t, name) not in ignore]


def varying(first: Sample, again: Sample) -> set[tuple[str, int, str]]:
    """The numbers that are not the same in two runs of the same, unchanged code (same settings, same seeds): the code does
    not fix them, so they cannot tell whether an edit changed a result. A number in only one of the runs counts."""
    return {(key, t, name) for key, t, name, _text in _differences(first, again) if t is not None}


def count_numbers(sample: Sample) -> int:
    """How many numbers the sample holds (by setting, trial and name)."""
    return sum(len(_numbers(r)) for rows in sample.values.values() for r in rows)


def fast_enough(before_s: float, after_s: float) -> bool:
    return before_s > 0 and after_s <= KEEP_BELOW * before_s


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
        "cell": {**cell, **inside}, "levels": _in_run.levels_json(_in_run.level_cells(inside)),
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
    return (f"FI estimates the whole study at about {_estimate.plain_duration(est_seconds)}; the time allowed is "
            f"{_estimate.plain_duration(limit_s)}. It has to be about {est_seconds / max(limit_s, 1e-9):.1f} times faster.\n"
            f"FI timed {len(cells)} settings ({SAMPLE_TRIALS} trials each, fixed seeds): about "
            f"{per_cell * 1000:.0f} ms of computing per setting on average for the trials after the first, in the "
            f"code below. The study runs {runs} trials of each setting."
            + (f"\n\n{profile}" if profile else ""))
