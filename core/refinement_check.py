"""FI's own check of a measurement's numerical settings: the headline results run again at finer settings.

A measurement's plan names the numerical resolutions of its simulation (a sampling density, a step, a mesh size) in
``protocol.numerical_settings``: the one value the study runs each at (``search``) and finer values (``check``; FI's fixed
rule when left out, :func:`core.optimisation_plan.check_levels`). Nothing in the plan's prose is read: after the run, with
the simulation's code as it ran, FI itself runs the settings where each headline result was lowest and highest (and a
few more spread over the study) once at the study's value and once at each finer value of one numerical setting, the
others at the study's values, through the same harness the run used, and judges each result by these rules:

* **a value that exists at one level and not at another** (it could be computed at the study's setting and not at a
  finer one, or the other way round) is not converged, whatever the numbers;
* **a refinement that changed nothing** (every result at every setting the very same number at every level) checked
  nothing: the code may not read the setting at all. It is said so, and is not a sign of convergence;
* otherwise the numerical error of the study's value is estimated from the levels (:func:`core.optimum_check.numerical_error`:
  the grid-convergence index with the observed order from three levels, a safety factor of 3 from two) and compared with
  the spread of that result over the whole study (the largest minus the smallest value the study measured), never with
  the value itself, so a result near zero is not judged by a relative error. It is converged when that error is at most
  the plan's ``tolerance`` (a fraction of the spread), else :data:`DEFAULT_TOLERANCE`.

Only a deterministic simulation (``run_cell``) is judged: with random numbers the levels differ by noise as well, and
that is not told apart here (the record says it was not judged). The record (``needs/REFINEMENT_CHECK.json``) keeps the
hash of the code it ran; a setting not converged, a refinement that changed nothing and a setting that could not be run
are gaps in the evidence (:func:`evidence_gaps`). It never stops the quest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import hidden_check as _hidden
from . import in_run as _in_run
from . import optimisation_plan as _optim
from . import optimum_check as _optimum
from . import trial_runner as _trial_runner

RECORD = "needs/REFINEMENT_CHECK.json"
#: The most settings run again (each is one run per level of each numerical setting).
MAX_CELLS = 8
#: A result is converged when its numerical error is at most this share of its spread over the study, unless the plan
#: gives its own ``tolerance``.
DEFAULT_TOLERANCE = 0.05


def settings(protocol: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The measurement's numerical settings, each with its ``levels`` (the study's value, then the finer ones) and the
    ``ratios`` from one level to the next: a measurement's ``protocol.numerical_settings``, or for a search with a first
    scan the search's own (``protocol.optimisation.numerical_settings``); ``{}`` for a search without a scan (its best
    designs have their own check: core/optimum_check.py) or none."""
    if not isinstance(protocol, dict):
        return {}
    block = protocol.get("optimisation")
    if block is not None:
        # A search's first scan is a measurement of its own (core/optimise_search.py::scan_grid): its results are run
        # again at the search's finer numerical settings too. The best designs have their own check (optimum_check).
        if not (isinstance(block, dict) and isinstance(block.get("grid"), dict) and block["grid"]):
            return {}
        raw = block.get("numerical_settings")
    else:
        raw = protocol.get("numerical_settings")
    out: dict[str, dict[str, Any]] = {}
    for name, item in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(item, dict) or not isinstance(item.get("search"), (int, float)):
            continue
        finer, where = _optim.check_levels(item)
        levels = [item["search"], *finer]
        larger = str(item.get("finer") or "smaller") == "larger"
        ratios = [(b / a if larger else a / b) if a and b else None for a, b in zip(levels, levels[1:])]
        tol = item.get("tolerance")
        out[str(name)] = {"levels": levels, "ratios": ratios, "from": where,
                          "tolerance": float(tol) if isinstance(tol, (int, float)) and not isinstance(tol, bool)
                          and 0 < tol < 1 else DEFAULT_TOLERANCE}
    return out


def _mean(metric: Any) -> float | None:
    values = metric.get("values") if isinstance(metric, dict) else None
    xs = [float(v) for v in values or [] if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    return sum(xs) / len(xs) if xs else None


def pick_cells(cells: list[dict[str, Any]], names: set[str], most: int = MAX_CELLS) -> list[dict[str, Any]]:
    """The settings to run again: for each headline result (``names``; every result when none is named), the setting
    where the study measured it lowest and the one where it was highest, then settings spread evenly over the study, at
    most ``most`` in all. ``cells``: the run settings of ``raw/trials.json`` (not the per-value records)."""
    runs = [c for c in cells if isinstance(c, dict) and c.get("level") is None and isinstance(c.get("cell"), dict)]
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(cell: dict[str, Any]) -> None:
        key = _trial_runner.cell_key(cell)
        if key not in seen and len(picked) < most:
            seen.add(key)
            picked.append(dict(cell))

    wanted = sorted(names) or sorted({str(n) for c in runs for n in (c.get("metrics") or {})})
    for name in wanted:
        scored = [(m, c["cell"]) for c in runs if (m := _mean((c.get("metrics") or {}).get(name))) is not None]
        if scored:
            add(min(scored, key=lambda s: s[0])[1])
            add(max(scored, key=lambda s: s[0])[1])
    step = max(1, len(runs) // max(1, most))
    for c in runs[::step]:
        add(c["cell"])
    return picked


def spreads(cells: list[dict[str, Any]]) -> dict[str, float]:
    """``{result (record key and name): the largest minus the smallest value the study measured}``."""
    seen: dict[str, list[float]] = {}
    for c in cells:
        if not isinstance(c, dict):
            continue
        level = c.get("level")
        for name, metric in (c.get("metrics") or {}).items():
            m = _mean(metric)
            if m is not None:
                seen.setdefault(_result_name(level, name), []).append(m)
    return {k: max(v) - min(v) for k, v in seen.items()}


def _result_name(level: Any, name: str) -> str:
    """A result's name with the value it was computed at inside the run, when it was one (``pitch=80: dof``)."""
    if isinstance(level, dict) and level:
        return f"{_in_run.level_key(level)}: {name}"
    return str(name)


def _flat(values: dict[str, Any]) -> dict[str, float]:
    """One run's numbers by result name (each value's numbers under the value's name), not-a-number kept."""
    out: dict[str, float] = {}
    for key, name, v in _in_run.flat("", values or {}):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{key}: {name}" if key else str(name)] = float(v)
    return out


def judge(setting: dict[str, Any], runs: dict[str, list[dict[str, float]]], spread: dict[str, float]) -> dict[str, Any]:
    """The verdict on one numerical setting. ``runs``: for each setting run again (by its key), what each level returned
    (the study's value first). ``spread``: each result's spread over the study."""
    results: list[dict[str, Any]] = []
    changed = False
    for cell_key, levels in runs.items():
        names = sorted({n for level in levels for n in level})
        for name in names:
            values = [level.get(name) for level in levels]
            finite = [v is not None and math.isfinite(v) for v in values]
            if not any(finite):
                continue
            if len(set(json.dumps(v) for v in values)) > 1:
                changed = True
            if not all(finite):
                results.append({"setting": cell_key, "result": name, "values": values, "status": "lost",
                                "how": "it has a value at one level and none at another"})
                continue
            error = _optimum.numerical_error(values, setting["ratios"])
            scale = spread.get(name) or 0.0
            share = (error["error"] / scale) if error.get("error") is not None and scale > 0 else None
            ok = error.get("error") == 0.0 or (share is not None and share <= setting["tolerance"])
            results.append({"setting": cell_key, "result": name, "values": values, "error": error.get("error"),
                            "share_of_spread": share, "how": error.get("how"),
                            "status": "converged" if ok else "not_converged"})
    if not results:
        return {"verdict": "not_run", "results": [], "why": "no setting could be run again at the finer levels"}
    if not changed:
        return {"verdict": "no_effect", "results": results,
                "why": "every result was the very same number at every level: the refinement checked nothing (the code "
                       "may not read this setting)"}
    bad = [r for r in results if r["status"] != "converged"]
    return {"verdict": "not_converged" if bad else "converged", "results": results,
            "worst": max(results, key=lambda r: (r["status"] != "converged", r.get("share_of_spread") or 0.0))}


async def run(executor: Any, python: Path | str, quest_root: Path, protocol: dict[str, Any], *, deterministic: bool,
              timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
              in_each_run: dict[str, list[Any]] | None, names: set[str]) -> dict[str, Any]:
    """Run the settings :func:`pick_cells` chooses again at every level of each numerical setting and judge them."""
    from . import run_estimate as _estimate  # one setting through the harness, outside FI's record (as when timed)

    quest_root = Path(quest_root)
    wanted = settings(protocol)
    record: dict[str, Any] = {"code_sha256": _hidden.code_sha(quest_root), "settings": {}}
    if not wanted:
        return {**record, "status": "none"}
    if not deterministic:
        return {**record, "status": "not_judged",
                "why": "the simulation uses random numbers: the levels would differ by noise as well, and FI does not "
                       "tell the two apart here"}
    try:
        cells = json.loads((quest_root / "raw" / "trials.json").read_text(encoding="utf-8")).get("cells") or []
    except (OSError, ValueError, AttributeError):
        cells = []
    chosen = pick_cells(cells, names)
    spread = spreads(cells)
    total = sum(len(s["levels"]) for s in wanted.values()) * max(1, len(chosen))
    each = max(5, int(timeout_s / max(1, total)))
    for name, setting in wanted.items():
        runs: dict[str, list[dict[str, float]]] = {}
        failed: list[str] = []
        for cell in chosen:
            got: list[dict[str, float]] = []
            for level in setting["levels"]:
                probe, why = await _estimate._time_one(
                    executor, python, quest_root, "code/simulate.py", {**cell, name: level}, runs=1, deterministic=True,
                    timeout_s=each, env=env, thresholds=thresholds, in_each_run=in_each_run, name="refine")
                if probe is None or probe.at_least or not probe.trials:
                    failed.append(f"{_trial_runner.cell_key(cell)} at {name}={level}: {why or 'did not finish in time'}")
                    break
                got.append(_flat(probe.trials[0]))
            if len(got) == len(setting["levels"]):
                runs[_trial_runner.cell_key(cell)] = got
        verdict = judge(setting, runs, spread)
        record["settings"][name] = {"levels": setting["levels"], "from": setting["from"],
                                    "tolerance": setting["tolerance"], "settings_run": len(runs),
                                    **({"could_not_run": failed[:5]} if failed else {}), **verdict}
    verdicts = {s["verdict"] for s in record["settings"].values()}
    record["status"] = "converged" if verdicts == {"converged"} else "gaps"
    return record


def write(quest_root: Path, record: dict[str, Any]) -> None:
    path = Path(quest_root) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")


def read(quest_root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _plain(value: float | None) -> str:
    return "none" if value is None or not math.isfinite(value) else f"{value:.4g}"


def lines(record: dict[str, Any]) -> list[str]:
    """One plain sentence per numerical setting, for run.log, the writer and the evidence."""
    out: list[str] = []
    if record.get("status") == "not_judged":
        return [f"the numerical settings were not run again at finer values: {record.get('why')}"]
    for name, s in (record.get("settings") or {}).items():
        levels = ", ".join(_plain(v) for v in s.get("levels") or [])
        verdict = s.get("verdict")
        if verdict == "converged":
            worst = s.get("worst") or {}
            out.append(f"`{name}` run again at finer values ({levels}) at {s.get('settings_run')} setting(s): every result "
                       f"is converged (the largest numerical error is {_plain((worst.get('share_of_spread') or 0) * 100)}% "
                       f"of that result's spread over the study, within {s.get('tolerance', 0) * 100:g}%)")
        elif verdict == "no_effect":
            out.append(f"`{name}` run again at finer values ({levels}): {s.get('why')}")
        elif verdict == "not_run":
            out.append(f"`{name}` could not be run again at finer values ({levels}): "
                       + "; ".join(s.get("could_not_run") or [s.get("why") or ""]))
        else:
            bad = [r for r in s.get("results") or [] if r.get("status") != "converged"]
            shown = "; ".join(
                f"{r['result']} at {r['setting']}: " + ", ".join(_plain(v) for v in r.get("values") or [])
                + (" (has a value at one level and none at another)" if r.get("status") == "lost" else
                   f" (numerical error {_plain((r.get('share_of_spread') or 0) * 100)}% of its spread)")
                for r in bad[:4])
            out.append(f"`{name}` run again at finer values ({levels}): {len(bad)} result(s) are not converged: {shown}")
    return out


def evidence_gaps(quest_root: Path, protocol: dict[str, Any] | None) -> list[str]:
    """Why the numerical settings are not shown converged: a setting not converged, a refinement that changed nothing, a
    setting that could not be run again, or a record missing or on other code. Empty with no numerical settings."""
    if not settings(protocol):
        return []
    record = read(quest_root)
    if not record:
        return ["the plan's numerical settings were not run again at finer values (no record of FI's refinement check)"]
    if record.get("code_sha256") != _hidden.code_sha(quest_root):
        return ["FI's refinement check ran on other code than the code as it is now"]
    if record.get("status") == "not_judged":
        return []
    return [f"numerical settings: {line}" for name, s in (record.get("settings") or {}).items()
            if s.get("verdict") != "converged" for line in lines({"settings": {name: s}})]
