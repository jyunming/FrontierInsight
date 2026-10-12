"""Run the behaviour probes (core/probes.py) on a quest's simulation and keep the record.

:mod:`core.probes` is pure: it decides which cells to run and judges what came back. This module runs the cells, through
FI's own trial harness (``trial_runner.HARNESS_SOURCE``, the process :mod:`core.hidden_check` and the study use): FI calls
``run_trial`` (or ``run_cell`` for a deterministic simulation) in a process of the quest's Python, with a seed FI gives and
the same seeds in every cell (paired), and reads the numbers itself. The harness copy and the specs live in
``.fi/probes_run/`` and are removed afterwards. No model is called, and no model ever supplies a number.

* **Budget.** ``budget_s`` is the CPU time (wall seconds of the cells' processes) the probes may spend. Before a relation's
  cells run, FI estimates their cost from the dearest cell run so far; a relation that would go over is
  ``unconfirmed("too costly to try")``, never ``broken``.
* **Confirmation.** A ``broken`` relation is run once more with fresh seeds and a different size of the change (for the
  kinds that have one); only when both runs disagree with the plan is it ``broken``. Otherwise it is ``unconfirmed``.
* **Saturation.** The generic test (:func:`core.probes.saturation`) on the numbers the cells returned; the quantities the
  plan expects to stay put, a setting echoed back, and a constant the code marks ``# constant by design`` are left out.
* **Record.** ``.fi/probes.json`` = ``{plan_sha, asked, relations, dropped, results: {code_sha: {rid: {verdict, numbers,
  says, strength}}}, repairs}`` and a plain ``needs/PROBES.json`` summary. Results are keyed by the hash of ``code/``
  (:func:`core.gate_receipt.fingerprint`): a result for the same code is reused, nothing is run again (resume).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gate_receipt as _receipt
from . import probes as _probes
from . import trial_runner as _tr

#: Where the record is kept (quest-relative), and the plain summary.
RECORD = ".fi/probes.json"
SUMMARY = "needs/PROBES.json"
_WORK = Path(".fi") / "probes_run"
_SIMULATE = Path("code") / "simulate.py"
#: CPU seconds the probes may spend, unless the caller says otherwise.
DEFAULT_BUDGET_S = 120
#: A cell's process may take this long, at most, however much budget is left.
DEFAULT_TIMEOUT_S = 300
#: Seeds used by the extra sweep that saturation needs when the relations ran fewer than this many cells.
_SWEEP_SEEDS = 4
TOO_COSTLY = "too costly to try"
#: The hash of seeds for a confirmation run: other than the first run's, so a lucky draw is not repeated.
_FRESH_BASE = 7919


@dataclass
class CellResult:
    """What one cell's process returned: per seed the metrics dict (``None`` when that trial failed)."""

    cell: dict[str, Any]
    rows: list[dict[str, float] | None]
    seconds: float
    reason: str = ""
    seeds: list[int | None] = field(default_factory=list)

    def values_of(self, metric: str) -> list[float | None]:
        return [None if r is None or metric not in r else r[metric] for r in self.rows]

    def has(self, metric: str) -> bool:
        return any(r is not None and metric in r for r in self.rows)


# ------------------------------------------------------------------------------------------------------- record on disk


def load(quest_root: Path) -> dict[str, Any] | None:
    try:
        rec = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def code_sha(quest_root: Path) -> str:
    """The hash of the simulation's code as the gate receipt makes it (drawing code left out), ``""`` with no code."""
    try:
        return str(_receipt.fingerprint(Path(quest_root)).get("computation") or "")
    except Exception:  # noqa: BLE001 -- a hash that cannot be made is "no key", never a crash
        return ""


def plan_sha(relations: list[_probes.Relation], grid: dict[str, list[Any]], seeds: int, deterministic: bool) -> str:
    payload = json.dumps({"relations": [r.to_dict() for r in relations], "grid": grid, "seeds": seeds,
                          "deterministic": deterministic}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write(quest_root: Path, record: dict[str, Any], sha: str) -> None:
    root = Path(quest_root)
    (root / RECORD).parent.mkdir(parents=True, exist_ok=True)
    (root / RECORD).write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    (root / SUMMARY).parent.mkdir(parents=True, exist_ok=True)
    (root / SUMMARY).write_text(json.dumps(summary(record, sha), indent=2, default=str) + "\n", encoding="utf-8")


def summary(record: dict[str, Any], sha: str) -> dict[str, Any]:
    """The plain version of the record for ``needs/``: what held, what broke (with the sentence), what could not be checked."""
    results = (record.get("results") or {}).get(sha) or {}
    quote = {str(r.get("id")): r.get("quote", "") for r in record.get("relations") or [] if isinstance(r, dict)}
    out: dict[str, Any] = {"code_sha256": sha, "held": [], "broken": [], "unconfirmed": []}
    for rid, res in results.items():
        item = {"id": rid, "says": res.get("says", ""), "strength": res.get("strength", "supporting")}
        if quote.get(rid):
            item["plan_says"] = quote[rid]
        if res.get("verdict") in ("held", "broken", "unconfirmed"):
            out[str(res["verdict"])].append(item)
    out["not_understood"] = list(record.get("dropped") or [])
    return out


# ------------------------------------------------------------------------------------------------------------ running


def _seed_list(n: int, base: int, deterministic: bool) -> list[int | None]:
    return [None] if deterministic else [_tr.trial_seed(base, "", i) for i in range(n)]


def _entries(quest_root: Path) -> set[str]:
    try:
        return _tr.entries(Path(quest_root) / _SIMULATE)
    except OSError:
        return set()


async def _run_cell(executor: Any, python: Path | str, quest_root: Path, entry: str, cell: dict[str, Any],
                    seeds: list[int | None], *, timeout_s: int, env: dict[str, str] | None,
                    thresholds: dict[str, Any] | None) -> CellResult:
    """One process of the quest's Python: the harness calls the entry once per seed on the cell and writes the rows."""
    root = Path(quest_root)
    work = root / _WORK
    work.mkdir(parents=True, exist_ok=True)
    harness = work / "harness.py"
    harness.write_text(_tr.HARNESS_SOURCE, encoding="utf-8")
    nonce = hashlib.sha256(f"probe|{time.time_ns()}|{json.dumps(cell, sort_keys=True, default=str)}".encode()).hexdigest()[:24]
    spec, out = work / "spec.json", work / "out.jsonl"
    out.unlink(missing_ok=True)
    spec.write_text(json.dumps({
        "module": _SIMULATE.as_posix(), "entry": entry, "cell": cell, "nonce": nonce,
        "thresholds": dict(thresholds or {}),
        "trials": [{"trial": i, "seed": s} for i, s in enumerate(seeds)],
    }, default=str), encoding="utf-8")
    t0 = time.monotonic()
    result = await executor.execute(
        [str(python), harness.relative_to(root).as_posix(), spec.relative_to(root).as_posix(), out.relative_to(root).as_posix()],
        cwd=root, timeout_s=timeout_s, env=env)
    seconds = time.monotonic() - t0
    try:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        rows = []
    mine = [r for r in rows if isinstance(r, dict) and r.get("nonce") == nonce]
    for r in mine:
        if r.get("load_error"):
            return CellResult(cell, [None] * len(seeds), seconds, f"the simulation could not be loaded: {r['load_error']}", seeds)
    by_trial = {r.get("trial"): r for r in mine}
    got: list[dict[str, float] | None] = []
    reason = ""
    for i in range(len(seeds)):
        r = by_trial.get(i)
        if r is not None and r.get("status") == "ok":
            got.append({str(k): float(v) for k, v in (r.get("values") or {}).items()})
        else:
            got.append(None)
            if not reason:
                reason = (str(r.get("reason")) if r is not None and r.get("reason") else
                          "it ran out of time" if getattr(result, "timed_out", False) else
                          f"it did not report (exit code {getattr(result, 'returncode', '?')})")
    return CellResult(cell, got, seconds, reason, seeds)


class _Runner:
    """Runs and remembers cells (one process each), and keeps the clock against the budget."""

    def __init__(self, executor: Any, python: Path | str, root: Path, entry: str, deterministic: bool, *,
                 budget_s: float, timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None) -> None:
        self.executor, self.python, self.root, self.entry = executor, python, root, entry
        self.deterministic, self.budget_s, self.timeout_s, self.env, self.thresholds = deterministic, budget_s, timeout_s, env, thresholds
        self.cache: dict[tuple[str, int, int], CellResult] = {}
        self.spent = 0.0
        self.dearest = 0.0

    def key(self, cell: dict[str, Any], base: int, n: int) -> tuple[str, int, int]:
        return (_probes._cell_key(cell), base, n)

    def would_cost(self, cells: list[dict[str, Any]], base: int, n: int) -> float:
        new = [c for c in cells if self.key(c, base, n) not in self.cache]
        return len(new) * self.dearest

    def over(self, extra: float) -> bool:
        return bool(self.dearest) and self.spent + extra > self.budget_s

    async def get(self, cell: dict[str, Any], base: int, n: int) -> CellResult:
        k = self.key(cell, base, n)
        if k in self.cache:
            return self.cache[k]
        left = max(5, int(self.budget_s - self.spent)) if self.dearest else self.timeout_s
        res = await _run_cell(self.executor, self.python, self.root, self.entry, cell,
                              _seed_list(n, base, self.deterministic), timeout_s=min(self.timeout_s, left),
                              env=self.env, thresholds=self.thresholds)
        self.spent += res.seconds
        self.dearest = max(self.dearest, res.seconds)
        self.cache[k] = res
        return res


def _alias(metric: str, metric_alias: dict[str, str] | None) -> str:
    return (metric_alias or {}).get(metric, metric)


def _verdict_for(rel: _probes.Relation, results: list[CellResult], plan: _probes.RunPlan, metric_alias: dict[str, str] | None
                 ) -> _probes.Verdict:
    name = _alias(rel.metric, metric_alias)
    if not any(r.has(name) for r in results):
        failed = next((r for r in results if r.reason), None)
        why = (f"the simulation failed at {_probes._fmt_cell(failed.cell)}: {failed.reason}" if failed is not None
               else f"the simulation does not return `{name}`")
        return _probes.Verdict(rel.id, "unconfirmed", {}, f"Could not check this: {why}.")
    obs = [_probes.Observation(r.cell, r.values_of(name)) for r in results]
    v = _probes.judge(rel, obs)
    if v.verdict == "unconfirmed":
        failed = next((r for r in results if r.reason and any(x is None for x in r.values_of(name))), None)
        if failed is not None:
            return _probes.Verdict(rel.id, "unconfirmed", v.numbers,
                                   f"Could not check this: the simulation failed at {_probes._fmt_cell(failed.cell)}: {failed.reason}.")
    return v


def _alt_plan(rel: _probes.Relation, grid: dict[str, list[Any]], seeds: int, deterministic: bool) -> _probes.RunPlan:
    """The cells for a second run of ``rel`` with a different size of the change where the kind has one."""
    g = {k: list(v) for k, v in grid.items()}
    s = rel.setting
    if s and s in g and rel.kind in ("sensitivity", "invariance", "monotone", "convergence"):
        vs = _probes._sorted_values(g[s])
        if rel.kind in ("sensitivity", "invariance") and len(vs) >= 3:
            g[s] = vs[:-1]
        elif rel.kind in ("monotone", "convergence") and len(vs) >= 4:
            g[s] = vs[1:]
    return _probes.plan_runs([rel], g, seeds, deterministic)


def _files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    code = Path(root) / "code"
    for p in sorted(code.rglob("*.py")) if code.is_dir() else []:
        if any(part in ("__pycache__", ".git", "tests") for part in p.relative_to(code).parts):
            continue
        try:
            out[p.relative_to(code).as_posix()] = p.read_text(encoding="utf-8")
        except (OSError, ValueError):
            pass
    return out


def exempt_for_saturation(quest_root: Path, protocol: dict[str, Any] | None, grid: dict[str, list[Any]],
                          names: list[str]) -> set[str]:
    """Names saturation leaves out: what the plan expects to stay put, a setting echoed back, and a quantity the code marks
    ``# constant by design: <reason>`` on a constant that it uses."""
    from . import degenerate_results as _dr
    from . import reported_settings as _rs

    out = set(_probes.exempt_names(protocol))
    for n in names:
        if any(_rs.names_setting(n, k) or _rs.names_setting(k, n) for k in grid):
            out.add(n)
    try:
        files = _files(quest_root)
        marks = _dr.design_marks(files)
        for n in names:
            if any(_dr._uses(files, m, n) for m in marks):
                out.add(n)
    except Exception:  # noqa: BLE001 -- a mark that cannot be read leaves the name in
        pass
    return out


def reported_names(quest_root: Path, protocol: dict[str, Any] | None, relations: list[_probes.Relation] | None = None,
                   extra: set[str] | None = None) -> set[str]:
    """The quantities a result can rest on, which saturation may call stuck: the keys the analysis reported (the printed
    result is ``extra``; the summaries it wrote under ``data/results/`` are read here), the plan's metrics and the metrics
    of the relations. A diagnostic the simulation returns but nobody reports (a temperature it echoes, the true rate)
    is not a result; it is allowed to be constant, as in :mod:`core.degenerate_results`."""
    from . import degenerate_results as _dr
    from . import reported_settings as _rs

    out = (set(extra or set()) | _probes._protocol_parts(protocol)[1]
           | {r.metric for r in relations or [] if r.kind != "invariance"})
    try:
        for _where, value in _rs.reported_of(Path(quest_root), None):
            out |= _dr.keys_in(value)
    except Exception:  # noqa: BLE001
        pass
    return out


def saturation_of(cells: list[dict[str, Any]], quest_root: Path, protocol: dict[str, Any] | None,
                  grid: dict[str, list[Any]], reported: set[str] | None = None,
                  hold: set[str] | None = None) -> list[_probes.Verdict]:
    """:func:`core.probes.saturation` on ``cells`` (``{"metrics": {name: [values]}}``) with the exemptions applied. With
    ``reported``, only the quantities it names (exactly, or with a unit word added) are judged; ``hold``: names a relation
    says do not depend on the setting (an invariance), which are allowed to be constant."""
    from . import reported_settings as _rs

    names = sorted({n for c in cells for n in (c.get("metrics") or {})})
    if reported is not None:
        names = [n for n in names if any(n == k or _rs.names_setting(n, k) or _rs.names_setting(k, n) for k in reported)]
        cells = [{**c, "metrics": {k: v for k, v in (c.get("metrics") or {}).items() if k in names}} for c in cells]
    return _probes.saturation(cells, exempt_for_saturation(quest_root, protocol, grid, names) | set(hold or ()))


def stored_cells(quest_root: Path) -> list[dict[str, Any]]:
    """The cells the study itself ran (``raw/trials.json``), in the shape :func:`saturation` reads; ``[]`` when none."""
    try:
        data = json.loads((Path(quest_root) / _tr.RAW_DIRNAME / _tr.SUMMARY_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for c in data.get("cells") or []:
        if isinstance(c, dict) and isinstance(c.get("metrics"), dict) and c["metrics"]:  # a setting that never finished has none
            out.append({"cell": c.get("cell"), "metrics": {str(k): list(v.get("values") or []) if isinstance(v, dict) else list(v)
                                                           for k, v in c["metrics"].items()}})
    return out


def _entry_of(have: set[str]) -> tuple[str | None, bool]:
    """``(the function FI calls, deterministic)``."""
    if "run_trial" in have:
        return "run_trial", False
    if "run_cell" in have:
        return "run_cell", True
    return None, False


async def run(executor: Any, python: Path | str, quest_root: Path, protocol: dict[str, Any] | None,
              relations: list[_probes.Relation], *, reported: set[str] | None = None, seeds: int = _probes.DEFAULT_SEEDS, budget_s: float = DEFAULT_BUDGET_S,
              timeout_s: int = DEFAULT_TIMEOUT_S, env: dict[str, str] | None = None,
              metric_alias: dict[str, str] | None = None, asked: Any = None, dropped: list[dict[str, str]] | None = None,
              base_seed: int = 0, confirm: bool = True, sweep: bool = True, write: bool = True) -> dict[str, Any]:
    """Run ``relations`` on the quest's simulation and return (and write) the record. A result already kept for the same
    plan and the same code is reused; only the relations without one are run. ``metric_alias``: protocol metric id -> the
    name ``run_trial`` returns, when they differ. ``confirm``: re-run a ``broken`` relation (the rule; off only to see the
    first run). ``sweep``: add the cells saturation needs when the relations ran too few."""
    root = Path(quest_root)
    grid, _metrics = _probes._protocol_parts(protocol)
    entry, deterministic = _entry_of(_entries(root))
    sha = code_sha(root)
    thresholds = protocol.get("thresholds") if isinstance(protocol, dict) and isinstance(protocol.get("thresholds"), dict) else None
    psha = plan_sha(relations, grid, seeds, deterministic)
    kept = load(root) or {}
    same_plan = kept.get("plan_sha") == psha
    results: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (kept.get("results") or {}).items()} if same_plan else {}
    record: dict[str, Any] = {
        "plan_sha": psha, "asked": asked if asked is not None else kept.get("asked", False),
        "relations": [r.to_dict() for r in relations], "dropped": list(dropped if dropped is not None else kept.get("dropped") or []),
        "results": results, "repairs": kept.get("repairs", 0)}
    mine = results.setdefault(sha, {})
    if entry is None or not sha:
        why = "FI cannot call the simulation on one case (it needs `run_trial` or `run_cell` in code/simulate.py)"
        for r in relations:
            mine.setdefault(r.id, {"verdict": "unconfirmed", "numbers": {}, "says": f"Could not check this: {why}.", "strength": r.strength})
        if write and sha:
            _write(root, record, sha)
        return record
    pending = [r for r in relations if r.id not in mine]
    want_sat = sha and not any(k.startswith("saturation:") for k in mine) and "saturation" not in mine
    if not pending and not want_sat:
        return record
    runner = _Runner(executor, python, root, entry, deterministic, budget_s=budget_s, timeout_s=timeout_s, env=env,
                     thresholds=thresholds)
    order = sorted(pending, key=lambda r: (r.strength != "core", relations.index(r)))
    try:
        plan = _probes.plan_runs(order, grid, seeds, deterministic)
        n = plan.runs[0].seeds if plan.runs else (1 if deterministic else seeds)
        cells_of = {rid: [run for run in plan.runs if rid in run.relations] for rid in uses_ids(plan)}
        bad = {u["id"]: u["reason"] for u in plan.unrunnable}
        for rel in order:
            if rel.id in bad:
                mine[rel.id] = _res(_probes.Verdict(rel.id, "unconfirmed", {}, f"Could not check this: {bad[rel.id]}."), rel)
                continue
            runs = cells_of.get(rel.id, [])
            cells = [r.cell for r in runs]
            if runner.over(runner.would_cost(cells, base_seed, n)):
                mine[rel.id] = _res(_probes.Verdict(rel.id, "unconfirmed", {"budget_s": budget_s, "spent_s": round(runner.spent, 2)},
                                                    f"Could not check this: {TOO_COSTLY} (the probes' time budget is {budget_s:g} s)."), rel)
                continue
            results_ = []
            stopped = False
            for cell in cells:
                if runner.over(runner.would_cost([cell], base_seed, n)):
                    stopped = True
                    break
                results_.append(await runner.get(cell, base_seed, n))
            if stopped:
                mine[rel.id] = _res(_probes.Verdict(rel.id, "unconfirmed", {"budget_s": budget_s, "spent_s": round(runner.spent, 2)},
                                                    f"Could not check this: {TOO_COSTLY} (the probes' time budget is {budget_s:g} s)."), rel)
                continue
            v = _verdict_for(rel, results_, plan, metric_alias)
            if v.verdict == "broken" and confirm:
                v = await _confirm(runner, rel, v, grid, seeds, deterministic, base_seed, metric_alias, budget_s)
            mine[rel.id] = _res(v, rel)
        if want_sat:
            await _saturation(runner, root, protocol, grid, mine, base_seed, n, sweep,
                              reported_names(root, protocol, relations, reported),
                              {_alias(r.metric, metric_alias) for r in relations if r.kind == "invariance"})
    finally:
        shutil.rmtree(root / _WORK, ignore_errors=True)
    record["seconds_spent"] = round(runner.spent, 2)
    if write:
        _write(root, record, sha)
    return record


def uses_ids(plan: _probes.RunPlan) -> list[str]:
    return list(plan.uses.keys())


def _res(v: _probes.Verdict, rel: _probes.Relation | None, strength: str | None = None) -> dict[str, Any]:
    return {"verdict": v.verdict, "numbers": v.numbers, "says": v.says,
            "strength": strength or (rel.strength if rel is not None else "core")}


async def _confirm(runner: _Runner, rel: _probes.Relation, first: _probes.Verdict, grid: dict[str, list[Any]], seeds: int,
                   deterministic: bool, base_seed: int, metric_alias: dict[str, str] | None, budget_s: float) -> _probes.Verdict:
    """The confirmation rule: a second run with fresh seeds and a different size of the change; both must disagree."""
    plan = _alt_plan(rel, grid, seeds, deterministic)
    if plan.unrunnable or not plan.runs:
        plan = _probes.plan_runs([rel], grid, seeds, deterministic)
    n = plan.runs[0].seeds if plan.runs else seeds
    fresh = base_seed + _FRESH_BASE
    cells = [r.cell for r in plan.runs]
    if runner.over(runner.would_cost(cells, fresh, n)):
        return _probes.Verdict(rel.id, "unconfirmed", {"first_run": first.numbers},
                               f"Could not check this again: {TOO_COSTLY}. The first run said: {first.says}")
    second_results = [await runner.get(c, fresh, n) for c in cells]
    second = _verdict_for(rel, second_results, plan, metric_alias)
    if second.verdict == "broken":
        return _probes.Verdict(rel.id, "broken", {**first.numbers, "second_run": second.numbers}, f"{first.says} A second run with new seeds and a different change agrees: {second.says}")
    return _probes.Verdict(rel.id, "unconfirmed", {"first_run": first.numbers, "second_run": second.numbers},
                           f"Looked wrong once but not on a second run with new seeds ({second.says}). First run: {first.says}")


async def _saturation(runner: _Runner, root: Path, protocol: dict[str, Any] | None, grid: dict[str, list[Any]],
                      mine: dict[str, Any], base_seed: int, n: int, sweep: bool, reported: set[str],
                      hold: set[str]) -> None:
    """The generic test on the cells the relations ran (and a short sweep of each setting if they were too few)."""
    def collect() -> list[dict[str, Any]]:
        out = []
        for (ckey, b, nn), res in runner.cache.items():
            if b != base_seed:
                continue
            ok = [r for r in res.rows if r is not None]
            if ok:
                out.append({"cell": res.cell, "metrics": {k: [r[k] for r in ok if k in r] for k in ok[0]}})
        return out

    cells = collect()
    if sweep and len(cells) < _probes.SATURATION_MIN_CELLS and grid:
        base = _probes._base(grid)
        extra = [base]
        for s, vals in grid.items():
            vs = _probes._sorted_values(vals)
            extra += [{**base, s: vs[0]}, {**base, s: vs[-1]}]
        # enough draws that an indicator pinned at 0 or 1 is not just luck (see probes.INDICATOR_MIN_VALUES), within the seed limit
        k = min(_probes.MAX_SEEDS, max(_SWEEP_SEEDS, -(-_probes.INDICATOR_MIN_VALUES // len(extra))))
        for cell in extra:
            if runner.over(runner.would_cost([cell], base_seed, k)):
                break
            await runner.get(cell, base_seed, k)
        cells = collect()
    verdicts = saturation_of(cells, root, protocol, grid, reported, hold) if cells else []
    for v in verdicts:
        mine[v.relation_id] = _res(v, None, "core")
    kinds = {v.verdict for v in verdicts}
    mine["saturation"] = {"verdict": "broken" if "broken" in kinds else "unconfirmed" if kinds else "held",
                          "numbers": {"cells": len(cells), "flagged": len(verdicts)},
                          "says": ("No result is NaN, pinned at a bound or constant over the settings run." if not verdicts
                                   else f"{len(verdicts)} result(s) look stuck; see the saturation entries."),
                          "strength": "core"}
