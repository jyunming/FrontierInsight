"""Make a study that does not fit the time allowed fit, without changing a result.

FI runs a fixed sample of the study itself, before and after one edit of the simulation: the same settings, the same
trials, the same seeds. The edit is kept only when every number the sample returns is the same (to nine digits; a
speed-up must not change what the study measures) and the time the sample takes drops by a stated margin. Nothing the
code prints and no time the code reports is read: the harness times the calls.
"""

from __future__ import annotations

import json
import math
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


async def run_sample(executor: Any, python: Path | str, quest_root: Path, module: str, cells: list[dict[str, Any]], *,
                     deterministic: bool, thresholds: dict[str, Any] | None, in_each_run: dict[str, list[Any]] | None,
                     env: dict[str, str] | None, timeout_s: int, log: Any = None) -> Sample:
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
                "cell": {**cell, **inside}, "levels": _in_run.level_cells(inside),
                "trials": [{"trial": t, "seed": None if deterministic else _tr.trial_seed(0, key, t)} for t in range(n)],
                "nonce": nonce, "thresholds": dict(thresholds or {})}, default=str), encoding="utf-8")
            await ex.execute([str(python), _tr.HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(),
                              out.relative_to(quest_root).as_posix()], cwd=quest_root, timeout_s=timeout_s, env=env)
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


def _leaves(x: Any, path: str = "") -> dict[str, Any]:
    """Every value a trial returned, by its place: numbers as floats, and every other value (a label, a flag, ``None``, an
    object the trial runner wrote as text) as it is, so a speed edit that changes one is seen too."""
    out: dict[str, Any] = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_leaves(v, f"{path}/{k}"))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(_leaves(v, f"{path}[{i}]"))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out[path] = float(x)
    else:
        out[path] = x
    return out


def disagreements(before: Sample, after: Sample) -> list[str]:
    """Where the sample's values are not the same (a missing or a new value counts), by setting, trial and name: numbers
    within the tolerance, every other value exactly."""
    bad: list[str] = []
    for key in sorted(before.values.keys() | after.values.keys()):
        a, b = before.values.get(key), after.values.get(key)
        if a is None or b is None or len(a) != len(b):
            bad.append(f"{key}: not run in both")
            continue
        for t, (ra, rb) in enumerate(zip(a, b)):
            na, nb = _leaves(ra), _leaves(rb)
            for name in sorted(na.keys() | nb.keys()):
                if name not in na or name not in nb:
                    bad.append(f"{key} trial {t} {name}: only in one")
                    continue
                x, y = na[name], nb[name]
                if not (isinstance(x, float) and isinstance(y, float)):
                    if x != y:
                        bad.append(f"{key} trial {t} {name}: {str(x)[:60]} -> {str(y)[:60]}")
                    continue
                if math.isnan(x) and math.isnan(y):
                    continue
                if not abs(x - y) <= ATOL + RTOL * abs(y):
                    bad.append(f"{key} trial {t} {name}: {x:.12g} -> {y:.12g}")
    return bad


def fast_enough(before_s: float, after_s: float) -> bool:
    return before_s > 0 and after_s <= KEEP_BELOW * before_s


def speed_block(est_seconds: float, limit_s: float, before: Sample, cells: list[dict[str, Any]], runs: int) -> str:
    per_cell = before.seconds / max(1, len(cells))
    return (f"FI estimates the whole study at about {_estimate.plain_duration(est_seconds)}; the time allowed is "
            f"{_estimate.plain_duration(limit_s)}. It has to be about {est_seconds / max(limit_s, 1e-9):.1f} times faster.\n"
            f"FI timed {len(cells)} settings ({SAMPLE_TRIALS} trials each, fixed seeds): about "
            f"{per_cell * 1000:.0f} ms of computing per setting on average for the trials after the first, in the "
            f"code below. The study runs {runs} trials of each setting.")
