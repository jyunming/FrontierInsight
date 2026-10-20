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
* otherwise the numerical error of the study's value is how far the value moved from the study's level to the finest
  one, plus the error left at the finest level (:func:`core.optimum_check.numerical_error`: the grid-convergence index
  with the observed order from three levels, a safety factor of 3 from two), and it is compared with the spread of that
  result over the whole study (the largest minus the smallest value the study measured), never with the value itself,
  so a result near zero is not judged by a relative error; only a result the study never moves is compared with its own
  value. It is converged when that error is at most the plan's ``tolerance`` (a fraction of the spread), else
  :data:`DEFAULT_TOLERANCE`. Only the headline results are judged (for a search, what it optimises and what limits it);
  the others count only for whether the refinement changed anything.

Only a deterministic simulation (``run_cell``) is judged: with random numbers the levels differ by noise as well, and
that is not told apart here (the record says it was not judged, and that is a gap). The record
(``needs/REFINEMENT_CHECK.json``) keeps the hash of the code it ran and a key of the settings it judged (:func:`key`); a
setting not converged, a refinement that changed nothing, a setting run again only in part or not at all are gaps in the
evidence (:func:`evidence_gaps`). It never stops the quest.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from . import criteria as _criteria
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
        if not isinstance(item, dict) or not isinstance(item.get("search"), (int, float)) \
                or isinstance(item.get("search"), bool):
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


def key(protocol: dict[str, Any] | None) -> str:
    """A fingerprint of what the check judged: the numerical settings and the study's settings (its grid, its parts and
    what each run goes through, a search's scan too). A change to one calls for the check again."""
    block = protocol.get("optimisation") if isinstance(protocol, dict) and isinstance(protocol.get("optimisation"), dict) \
        else {}
    study = {part: (protocol or {}).get(part) for part in ("grid", "sub_studies", "in_each_run", "constants",
                                                           "thresholds")} if isinstance(protocol, dict) else {}
    scan = {part: block.get(part) for part in ("grid", "sub_studies", "in_each_run", "fixed", "design_variables",
                                               "baseline")}
    return hashlib.sha256(json.dumps([settings(protocol), study, scan], sort_keys=True, default=str).encode(
        "utf-8")).hexdigest()


def _judged(name: str, names: set[str]) -> bool:
    """A headline result, or a headline result at one value compared inside each run (``speed=2: drag``), the names
    compared as the headline check compares them (``mean_width`` is ``Mean width``); every result when no headline is
    named."""
    if not names:
        return True
    keys = {_criteria._key(n) for n in names}
    return _criteria._key(name) in keys or _criteria._key(str(name).split(": ", 1)[-1]) in keys


def _mean(metric: Any) -> float | None:
    values = metric.get("values") if isinstance(metric, dict) else None
    xs = [float(v) for v in values or [] if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    return sum(xs) / len(xs) if xs else None


def pick_cells(cells: list[dict[str, Any]], names: set[str], most: int = MAX_CELLS) -> list[dict[str, Any]]:
    """The settings to run again: for each headline result (``names``; every result when none is named), the setting
    where the study measured it lowest and the one where it was highest, then settings spread evenly over the study, at
    most ``most`` in all. ``cells``: the records of ``raw/trials.json``; a result computed at each value compared inside
    a run (a per-value record) picks the run setting it came from, lowest and highest over every value."""
    records = [c for c in cells if isinstance(c, dict) and isinstance(c.get("cell"), dict)]
    runs = [c for c in records if c.get("level") is None]
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(cell: dict[str, Any]) -> None:
        key = _trial_runner.cell_key(cell)
        if key not in seen and len(picked) < most:
            seen.add(key)
            picked.append(dict(cell))

    def run_setting(c: dict[str, Any]) -> dict[str, Any]:
        level = c.get("level") if isinstance(c.get("level"), dict) else {}
        return {k: v for k, v in c["cell"].items() if k not in level}

    every = sorted({str(n) for c in records for n in (c.get("metrics") or {})})
    wanted = [n for n in every if _judged(n, names)] or every  # a headline named as no result is: every result
    for name in wanted:
        scored = [(m, run_setting(c)) for c in records
                  if (m := _mean((c.get("metrics") or {}).get(name))) is not None]
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
    """A result's name with the value it was computed at inside the run, when it was one (``speed=2: drag``)."""
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


def judge(setting: dict[str, Any], runs: dict[str, list[dict[str, float]]], spread: dict[str, float],
          names: set[str] | None = None) -> dict[str, Any]:
    """The verdict on one numerical setting. ``runs``: for each setting run again (by its key), what each level returned
    (the study's value first). ``spread``: each result's spread over the study. ``names``: the headline results, the
    only ones judged (every result when none is named); every result counts for whether anything changed."""
    results: list[dict[str, Any]] = []
    changed = False
    returned = {n for levels in runs.values() for level in levels for n in level}
    if names and not any(_judged(n, names) for n in returned):
        names = set()  # the headline is named as no result the run returns: every result is judged, never none
    for cell_key, levels in runs.items():
        for name in sorted({n for level in levels for n in level}):
            values = [level.get(name) for level in levels]
            finite = [v is not None and math.isfinite(v) for v in values]
            if not any(finite):
                continue
            if len(set(json.dumps(v) for v in values)) > 1:
                changed = True
            if not _judged(name, names or set()):
                continue
            if not all(finite):
                results.append({"setting": cell_key, "result": name, "values": values, "status": "lost",
                                "how": "it has a value at one level and none at another"})
                continue
            finest = _optimum.numerical_error(values, setting["ratios"])
            # The paper's numbers are the study's value (the first level): its error is how far it is from the finest
            # level, plus what the finest level may still be off by.
            error = None if finest.get("error") is None else abs(values[0] - values[-1]) + finest["error"]
            scale, against = spread.get(name) or 0.0, "spread"
            if scale <= 1e-6 * max(abs(v) for v in values):
                # A result the study never moves (its spread is nothing, or round-off): there is no spread to judge
                # by, so its own value is the scale.
                scale, against = abs(values[-1]), "value"
            share = (error / scale) if error is not None and scale > 0 else None
            tiny = error is not None and error <= 1e-12 * (1.0 + abs(values[-1]))
            ok = tiny or (share is not None and share <= setting["tolerance"])
            results.append({"setting": cell_key, "result": name, "values": values, "error": error,
                            "error_at_finest": finest.get("error"), "share_of_spread": share, "against": against,
                            "how": finest.get("how"), "status": "converged" if ok else "not_converged"})
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
    record: dict[str, Any] = {"code_sha256": _hidden.code_sha(quest_root), "key": key(protocol), "settings": {}}
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
    # The time is shared by what is left: a finer level costs more than the study's own, and gets what the coarser runs
    # did not use (one equal share each would run out first at the finest level).
    left = sum(len(s["levels"]) for s in wanted.values()) * max(1, len(chosen))
    deadline = time.monotonic() + timeout_s
    for name, setting in wanted.items():
        runs: dict[str, list[dict[str, float]]] = {}
        failed: list[str] = []
        for cell in chosen:
            got: list[dict[str, float]] = []
            for i, level in enumerate(setting["levels"]):
                each = max(5, int((deadline - time.monotonic()) / max(1, left)))
                left -= 1
                probe, why = await _estimate._time_one(
                    executor, python, quest_root, "code/simulate.py", {**cell, name: level}, runs=1, deterministic=True,
                    timeout_s=each, env=env, thresholds=thresholds, in_each_run=in_each_run, name="refine")
                if probe is None or probe.at_least or not probe.trials:
                    failed.append(f"{_trial_runner.cell_key(cell)} at {name}={level}: {why or 'did not finish in time'}")
                    left -= len(setting["levels"]) - i - 1
                    break
                got.append(_flat(probe.trials[0]))
            if len(got) == len(setting["levels"]):
                runs[_trial_runner.cell_key(cell)] = got
        verdict = judge(setting, runs, spread, names)
        record["settings"][name] = {"levels": setting["levels"], "from": setting["from"],
                                    "tolerance": setting["tolerance"], "settings_run": len(runs),
                                    "settings_chosen": len(chosen), "not_run": len(failed),
                                    **({"could_not_run": failed[:5]} if failed else {}), **verdict}
    verdicts = {s["verdict"] for s in record["settings"].values()}
    partial = any(s.get("not_run") for s in record["settings"].values())
    record["status"] = "converged" if verdicts == {"converged"} and not partial else "gaps"
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
        part = (f"; {s['not_run']} of the {s.get('settings_chosen')} setting(s) chosen could not be run again: "
                + "; ".join(s.get("could_not_run") or []) if s.get("not_run") and verdict != "not_run" else "")
        if verdict == "converged":
            worst = s.get("worst") or {}
            out.append(f"`{name}` run again at finer values ({levels}) at {s.get('settings_run')} setting(s): every result "
                       f"is converged (the largest numerical error of the study's value is "
                       f"{_plain((worst.get('share_of_spread') or 0) * 100)}% of that result's spread over the study, "
                       f"within {s.get('tolerance', 0) * 100:g}%){part}")
        elif verdict == "no_effect":
            out.append(f"`{name}` run again at finer values ({levels}): {s.get('why')}{part}")
        elif verdict == "not_run":
            out.append(f"`{name}` could not be run again at finer values ({levels}): "
                       + "; ".join(s.get("could_not_run") or [s.get("why") or ""]))
        else:
            bad = [r for r in s.get("results") or [] if r.get("status") != "converged"]
            shown = "; ".join(
                f"{r['result']} at {r['setting']}: " + ", ".join(_plain(v) for v in r.get("values") or [])
                + (" (has a value at one level and none at another)" if r.get("status") == "lost" else
                   (f" (numerical error of the study's value {_plain(r['share_of_spread'] * 100)}% of "
                    + ("its spread" if r.get("against") != "value" else "its value, which the study never moves") + ")"
                    if isinstance(r.get("share_of_spread"), (int, float)) else
                    (f" (numerical error of the study's value {_plain(r['error'])}, with no spread or value to "
                     "compare it with)" if isinstance(r.get("error"), (int, float)) else
                     " (its numerical error could not be estimated)")))
                for r in bad[:4])
            out.append(f"`{name}` run again at finer values ({levels}): {len(bad)} result(s) are not converged: "
                       f"{shown}{part}")
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
    if record.get("key") != key(protocol):
        return ["FI's refinement check ran on other settings than the plan's as they are now"]
    if record.get("status") == "not_judged":
        return [f"numerical settings: {line}" for line in lines(record)]
    return [f"numerical settings: {line}" for name, s in (record.get("settings") or {}).items()
            if s.get("verdict") != "converged" or s.get("not_run") for line in lines({"settings": {name: s}})]
