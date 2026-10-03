"""FI checks the best design its search found, at finer numerical settings, before anything is written about it.

A search scores every design at the search's own numerical settings (a coarse mesh, a large time step, few runs): the
cheapest settings at which the search can afford its budget. A search is also very good at finding where those settings
are wrong: a coarse mesh that underestimates the temperature at one fin spacing makes that spacing look best. So after
the search (:mod:`core.optimise`), the ENGINE (never the simulation's script, which does not know it is being checked)
evaluates again, through the same harness and in a process of its own for each design:

1. **Refinement**: the best designs of the search (the best of each starting point, up to
   :data:`core.optimisation_plan.CHECK_CANDIDATES`) and the baseline, at each finer level of the numerical settings (the
   plan's ``check`` values, or FI's fixed rule: half, then a quarter of the search value; which one is recorded). The
   best design is chosen at the finest level, not at the search's; each design's numerical error is estimated from how
   its value changes from level to level (Roache's grid-convergence index: a safety factor of 1.25 and the observed order
   with three levels or more, 3 with two).
2. **Limits**: every limit is checked at the finest level (with the plan's ``constraint_margin``). When the search's best
   breaks one there, the next candidate that meets every limit is used, and the record says so.
3. **Improvement over the baseline**: at the finest level, against the plan's threshold for "better" when it gives one,
   else against the numerical error of the two designs (the improvement must be larger than their errors added). Whether
   it is larger than that error is recorded either way. With randomness, the baseline and the best design are run again
   with fresh seeds the search never used (the same seeds for both, so they are compared run by run), and the improvement
   is the lower end of its 95% interval.
4. **Starting points**: whether the best designs from different starting points agree at the finest level.
5. **Neighbourhood**: each variable of the best design nudged both ways (2% of its range, or one whole number), at the
   finest level: a nearby design that is better means the search had not finished; a variable whose nudge changes nothing
   is probably not used by the simulation; a design at the edge of its range may be better outside it.
6. **Budget**: whether the search stopped because it converged, or because its budget or its time ran out.

The check never stops the quest: a failed check is the study's honest result ("the improvement over the baseline
disappears at finer settings"), written into ``needs/OPTIMUM_CHECK.json`` (one plain verdict and a plain sentence per
check, with the numbers) and run.log, and it limits the evidence level (:func:`evidence_gaps`). Its evaluations are its
own: they are not taken from the search's budget, and they have their own time limit (``execution.timeout_s``); a check
cut short by it is ``unverified``, never a pass.

Like :mod:`core.optimise_search`, the check itself is a generator that asks for evaluations and is sent the answers
(:func:`check`), so the same logic runs through FI's harness (:func:`run_check`) and in-process (:func:`check_sync`).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from . import optimisation_plan as _plan
from . import optimise_search as _search
from . import stats as _stats

SCHEMA = "fi.optimum-check/v1"
CHECK_PATH = Path("needs") / "OPTIMUM_CHECK.json"
RECORD = Path(".fi") / "optimisation" / "check.json"
#: The environment variable that tells the analysis script where FI's check is.
CHECK_ENV = "FI_OPTIMUM_CHECK"
#: How far each variable is nudged around the best design, as a share of its range.
NUDGE = 0.02
#: The seed family of the check's fresh runs (never one the search used).
SEED_KEY = "fi-optimum-check"
VERDICTS = {
    "verified": "the improvement over the baseline holds at finer numerical settings",
    "improvement_not_shown": "the improvement over the baseline is not shown at finer numerical settings",
    "infeasible": "no design found meets every limit at finer numerical settings",
    "not_local_optimum": "a nearby design is better at finer numerical settings, so the search had not finished",
    "unverified": "the best design could not be checked at finer numerical settings",
}
PASSED, FAILED, NOT_CHECKED = "passed", "failed", "not_checked"
BLIND_SPOTS = [
    "Refining the numerical settings rules out a numerical error only: an error of the model itself (an assumption, a "
    "material value) is the same at every setting, so the improvement is an improvement within the model.",
    "The numerical error is estimated from how the values change from one setting to the next (a grid-convergence "
    "estimate); a simulation that is far from converged at every level can look converged.",
]


class _Stop(Exception):
    """The check's time is up."""


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def _describe(design: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {_fmt(v)}" for k, v in design.items())


# --- the numerical settings of each level ----------------------------------------------------------------------------


def levels_of(block: dict[str, Any]) -> dict[str, Any]:
    """The finer levels the check evaluates at: ``levels`` (one dict of setting values per level, coarse to fine),
    ``ratios`` (the refinement ratio from the level before, the search's settings first: the smallest among the settings
    that change at that step), ``where`` (``plan`` or ``rule`` for each setting) and ``search``. With no numerical setting
    named, one level at the search's own settings (a repeat, which cannot show a numerical error)."""
    settings = block.get("numerical_settings") if isinstance(block.get("numerical_settings"), dict) else {}
    search = _search.search_settings(block)
    if not settings:
        return {"search": search, "levels": [{}], "ratios": [None], "where": {}, "named": False}
    per: dict[str, tuple[float, list[float], str, str]] = {}
    for name, setting in settings.items():
        setting = setting if isinstance(setting, dict) else {"search": setting}
        values, where = _plan.check_levels(setting)
        per[str(name)] = (setting["search"], list(values), where, str(setting.get("finer") or "smaller"))
    count = max(len(v[1]) for v in per.values())
    levels = [{name: vals[min(i, len(vals) - 1)] for name, (_s, vals, _w, _f) in per.items()} for i in range(count)]
    ratios: list[float | None] = []
    before = {name: s for name, (s, _v, _w, _f) in per.items()}
    for level in levels:
        changed = []
        for name, value in level.items():
            prev, finer = float(before[name]), per[name][3]
            ratio = (prev / float(value)) if finer == "smaller" else (float(value) / prev)
            if ratio > 1.0 + 1e-12:
                changed.append(ratio)
        ratios.append(min(changed) if changed else None)
        before = dict(level)
    return {"search": search, "levels": levels, "ratios": ratios,
            "where": {name: w for name, (_s, _v, w, _f) in per.items()}, "named": True}


# --- the numerical error of one design -------------------------------------------------------------------------------


def numerical_error(values: list[float], ratios: list[float | None]) -> dict[str, Any]:
    """The numerical error of the finest of ``values`` (coarse to fine; ``ratios[i]`` is the refinement ratio from
    ``values[i]`` to ``values[i + 1]``), after Roache's grid-convergence index: with three levels or more, the observed
    order from the finest three and a safety factor of 1.25; with two, a safety factor of 3 and an assumed first order;
    3 as well when the finest step changed the value more than the step before it (not converging) or the value swings
    back and forth. ``{"error": None}`` when there is no finer level to compare with."""
    vals = [float(v) for v in values if _number(v)]
    if len(vals) != len(values) or len(vals) < 2:
        return {"error": None, "order": None, "how": "no finer level to compare with"}
    f1, f2 = vals[-1], vals[-2]
    r21 = ratios[-1] if ratios and ratios[-1] else None
    e21 = f2 - f1
    tiny = 1e-12 * (1.0 + abs(f1))
    if r21 is None or r21 <= 1.0:
        return {"error": None, "order": None, "how": "the settings did not change between the last two levels"}
    if len(vals) == 2:
        return {"error": 3.0 * abs(e21) / (r21 - 1.0), "order": None,
                "how": "two levels only: a safety factor of 3 and an assumed first order (a third level would show "
                       "whether the values are converging)"}
    f3 = vals[-3]
    r32 = ratios[-2] if len(ratios) >= 2 and ratios[-2] else r21
    e32 = f3 - f2
    if abs(e21) <= tiny:
        return {"error": 0.0, "order": None, "how": "the two finest levels give the same value"}
    if abs(e32) <= tiny or abs(e21) > abs(e32):
        return {"error": 3.0 * abs(e21) / (r21 - 1.0), "order": None, "asymptotic": False,
                "how": "the finest step changed the value more than the step before it: not converging yet (safety "
                       "factor 3)"}
    s = 1.0 if e32 / e21 > 0 else -1.0
    if s < 0:
        return {"error": 3.0 * max(abs(e21), abs(e32)) / (r21 - 1.0), "order": None, "asymptotic": False,
                "how": "the value swings back and forth as the settings get finer (safety factor 3)"}
    p = 1.0
    for _ in range(200):
        q = math.log((r21 ** p - s) / (r32 ** p - s)) if abs(r21 - r32) > 1e-12 else 0.0
        new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if not math.isfinite(new):
            break
        if abs(new - p) < 1e-10:
            p = new
            break
        p = new
    p = min(max(p, 0.5), 8.0)
    return {"error": 1.25 * abs(e21) / (r21 ** p - 1.0), "order": round(p, 4), "asymptotic": True,
            "how": f"three levels: observed order {p:.2f}, safety factor 1.25"}


# --- the check -------------------------------------------------------------------------------------------------------


def _limits(block: dict[str, Any]) -> list[tuple[str, str, float, str]]:
    out = []
    for c in block.get("constraints") or []:
        parsed = _search._parse_limit(c.get("limit"))
        if parsed is not None:
            out.append((str(c["quantity"]), parsed[0], parsed[1], str(c.get("unit") or "")))
    return out


def _slack(op: str, bound: float, value: float) -> float:
    return bound - value if op == "<=" else value - bound


def candidates(block: dict[str, Any], rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """The designs the check recomputes besides the baseline: the best design (other than the baseline) that met every
    limit from each starting point (the global stage, a coarse scan and an exhaustive run count as one each), best first,
    at most ``count``. A design two starting points both found is recomputed once; ``also_found_by`` names the others."""
    space = _search.Space(block["design_variables"])
    sign = _search._sign(block)
    base_key = space.key(space.canonical(block["baseline"]["values"]))
    feasible = [r for r in rows if isinstance(r, dict) and r.get("event", "evaluation") == "evaluation"
                and r.get("status") == "ok" and r.get("feasible") and _number(r.get("objective"))]
    groups: dict[str, dict[str, Any]] = {}
    for row in sorted(feasible, key=lambda r: (sign * r["objective"], r.get("n") or 0)):
        key = space.key(space.canonical(row["design"]))
        if key == base_key:
            continue
        group = str(row.get("start")) if row.get("start") is not None else str(row.get("stage"))
        groups.setdefault(group, {**row, "group": group, "key": key, "also_found_by": []})
    out: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for row in sorted(groups.values(), key=lambda r: (sign * r["objective"], r.get("n") or 0)):
        if row["key"] in by_key:
            by_key[row["key"]]["also_found_by"].append(row["group"])
            continue
        if len(out) >= max(1, count):
            continue
        by_key[row["key"]] = row
        out.append(row)
    return out


def _starts(rows: list[dict[str, Any]]) -> set[Any]:
    """The starting points the search ran from (a local search's ``start``), among the evaluations that worked."""
    return {r.get("start") for r in rows if isinstance(r, dict) and r.get("status") == "ok"
            and r.get("start") is not None and r.get("stage") not in ("baseline", "scan", "global")}


def _threshold(block: dict[str, Any], base_value: float) -> tuple[float | None, dict[str, Any] | None]:
    tolerance = block.get("improvement_tolerance")
    value = tolerance.get("value") if isinstance(tolerance, dict) else tolerance
    if not _number(value):
        return None, None
    mode = str(tolerance.get("mode") or "absolute") if isinstance(tolerance, dict) else "absolute"
    in_unit = float(value) * abs(base_value) if mode == "relative" else float(value)
    return in_unit, {"value": value, "mode": mode, "in_unit": in_unit}


def check(block: dict[str, Any], search_record: dict[str, Any], rows: list[dict[str, Any]], *, noisy: bool = False,
          check_runs: int = 1):
    """The check as a generator of requests (see the module docstring); returns the record (without the file hashes
    and the seeds, which :func:`run_check` adds). A request is ``{"kind": "evaluate", "cell", "design", "level", "role"}``
    and its answer ``{"values": {...}, "trials": [{...}, ...]}``, ``{"failed": why}`` or ``{"stop": "time"}``."""
    if not isinstance(search_record.get("best"), dict):
        return _nothing_to_check(block, search_record)
    objective = block["objective"]
    quantity = str(objective["quantity"])
    space = _search.Space(block["design_variables"])
    info = levels_of(block)
    levels = info["levels"]
    finest = len(levels) - 1
    limits = _limits(block)
    margin = float(block["constraint_margin"]) if _number(block.get("constraint_margin")) else 0.0
    sign = _search._sign(block)
    counts = {"designs": 0, "failed": 0}
    cache: dict[tuple[str, int], dict[str, Any]] = {}

    def evaluate(design: dict[str, Any], level: int, role: str):
        design = space.canonical(design)
        key = (space.key(design), level)
        if key in cache:
            return cache[key]
        cell = {**_search.cell_of(block, design), **levels[level]}
        reply = yield {"kind": "evaluate", "cell": cell, "design": dict(design), "level": level, "role": role}
        reply = reply if isinstance(reply, dict) else {"failed": "no answer for this design"}
        if reply.get("stop"):
            raise _Stop()
        counts["designs"] += 1
        judged = _search.judge(block, reply.get("values") if "values" in reply else None,
                               str(reply.get("failed") or ""))
        if judged["status"] != "ok":
            counts["failed"] += 1
        runs = [float(t[quantity]) for t in reply.get("trials") or [] if isinstance(t, dict) and _number(t.get(quantity))]
        judged = {**judged, "runs": runs}
        cache[key] = judged
        return judged

    search_base = search_record.get("baseline") or {}
    k = min(_plan.CHECK_CANDIDATES, max(1, int((search_record.get("evaluations") or {}).get("starts") or 1)))
    base = {"design": space.canonical(block["baseline"]["values"]), "role": "baseline",
            "search": search_base.get("objective"), "search_constraints": dict(search_base.get("constraints") or {}),
            "levels": [None] * len(levels)}
    cands = [{"design": space.canonical(c["design"]), "role": "candidate", "search": c["objective"],
              "search_constraints": dict(c.get("constraints") or {}), "levels": [None] * len(levels),
              "group": c["group"], "search_n": c.get("n"), "also_found_by": list(c["also_found_by"])}
             for c in candidates(block, rows, k)]
    probes: dict[str, Any] = {}
    finished = True
    try:
        # The baseline and the best designs at every finer level, coarse to fine; then the neighbourhood of the design
        # chosen at the finest level, at the finest level.
        for level in range(len(levels)):
            for item in [base, *cands]:
                item["levels"][level] = yield from evaluate(item["design"], level, item["role"])
        chosen = _choose(cands, finest, limits, margin, sign)
        if chosen is not None:
            yield from _probe(space, chosen, finest, evaluate, probes)
    except _Stop:
        finished = False
    return _record(block, search_record, info, base, cands, probes, rows=rows, finished=finished, limits=limits,
                   margin=margin, noisy=noisy, check_runs=check_runs, counts=counts)


def _meets(judged: dict[str, Any] | None, limits: list[tuple[str, str, float, str]], margin: float) -> bool:
    if not judged or judged.get("status") != "ok":
        return False
    for name, op, bound, _unit in limits:
        value = (judged.get("constraints") or {}).get(name)
        if not _number(value) or _slack(op, bound, float(value)) < margin:
            return False
    return True


def _choose(entries: list[dict[str, Any]], level: int, limits: list[tuple[str, str, float, str]], margin: float,
            sign: float) -> dict[str, Any] | None:
    ok = [e for e in entries if _meets(e["levels"][level], limits, margin)]
    if not ok:
        return None
    return min(ok, key=lambda e: sign * e["levels"][level]["objective"])


def _probe(space: Any, chosen: dict[str, Any], level: int, evaluate: Callable, out: dict[str, Any]):
    """Each variable of the chosen design nudged both ways (2% of its range; one whole number), inside its range, into
    ``out`` (filled as it goes, so a check cut short keeps what it did). A design within half a nudge of an edge of its
    range is at that edge, and is not nudged past it."""
    design = chosen["design"]
    for variable, kind in zip(space.variables, space.kinds):
        name = str(variable["name"])
        if kind == "choice":
            continue
        low, high, x = float(variable["low"]), float(variable["high"]), float(design[name])
        step = 1.0 if kind == "integer" else NUDGE * (high - low)
        edge = "low" if x - low < step / 2 else "high" if high - x < step / 2 else None
        out[name] = {"at_bound": edge, "step": step, "sides": {}, "value": design[name]}
        for side, value in (("down", max(low, x - step)), ("up", min(high, x + step))):
            if (side == "down" and edge == "low") or (side == "up" and edge == "high"):
                continue
            nudged = dict(design)
            nudged[name] = int(round(value)) if kind == "integer" else _search._snap(value)
            if space.key(space.canonical(nudged)) == space.key(space.canonical(design)):
                continue
            judged = yield from evaluate(nudged, level, "probe")
            out[name]["sides"][side] = {"value": space.canonical(nudged)[name], "judged": judged}


def _same(a: dict[str, Any], b: dict[str, Any], space: Any) -> bool:
    ua, ub = space.unit(space.canonical(a)), space.unit(space.canonical(b))
    return max((abs(x - y) for x, y in zip(ua, ub)), default=0.0) < 0.01


def _series(item: dict[str, Any], name: str | None = None) -> list[Any]:
    """The value of the objective (or of one constrained quantity) at the search's settings, then at each level."""
    if name is None:
        return [item.get("search"), *[(j or {}).get("objective") for j in item["levels"]]]
    return [(item.get("search_constraints") or {}).get(name),
            *[((j or {}).get("constraints") or {}).get(name) for j in item["levels"]]]


def _o(op: str) -> str:
    return "at most" if op == "<=" else "at least"


def _record(block, search_record, info, base, cands, probes, *, rows, finished, limits, margin, noisy, check_runs,
            counts):  # noqa: ANN001,C901 -- one pass over what the check measured, check by check
    objective = block["objective"]
    unit = str(objective.get("unit") or "")
    u = f" {unit}" if unit else ""
    sign = _search._sign(block)
    space = _search.Space(block["design_variables"])
    levels, ratios, named = info["levels"], info["ratios"], info["named"]
    finest = len(levels) - 1
    floor = 1e-9
    checks: dict[str, dict[str, Any]] = {}

    def at(item: dict[str, Any] | None) -> dict[str, Any] | None:
        return item["levels"][finest] if item else None

    def estimate(item: dict[str, Any], name: str | None = None) -> dict[str, Any]:
        if not named:
            return {"error": None, "order": None, "how": "no numerical setting is named, so there is no finer level"}
        if noisy:
            # With randomness the search's value came from other seeds (and a lucky design looked better there than it
            # is): only the check's levels, all run with the same fresh seeds, show the numerical error.
            return numerical_error(_series(item, name)[1:], ratios[1:])
        return numerical_error(_series(item, name), ratios)

    def err(item: dict[str, Any], name: str | None = None) -> float | None:
        """The design's numerical error; 0 with no numerical setting named (nothing finer to compare with, said by the
        refinement check), ``None`` when it could not be estimated (a level gave no value, or too few levels)."""
        if not named:
            return 0.0
        e = estimate(item, name).get("error")
        return float(e) if _number(e) else None

    search_best = search_record.get("best") if isinstance(search_record.get("best"), dict) else None
    # Every design has a result at every level (a result may be a failure): what the time limit may have cut short.
    measured_all = all(all(j is not None for j in x["levels"]) for x in [base, *cands])
    base_final = at(base)
    base_ok = bool(base_final and base_final.get("status") == "ok")
    chosen = _choose(cands, finest, limits, margin, sign) if measured_all else None
    ran_finest = [c for c in cands if (at(c) or {}).get("status") == "ok"]
    # The search's own best, when it is one of the candidates (it is not when nothing beat the baseline).
    first = (cands[0] if cands and search_best is not None and space.key(cands[0]["design"]) ==
             space.key(space.canonical(search_best["design"])) else None)
    baseline_was_best = search_best is not None and space.key(space.canonical(search_best["design"])) == space.key(
        base["design"])

    # --- 1. refinement
    if not measured_all:
        checks["refinement"] = {"status": NOT_CHECKED, "says": "the check's time limit (execution.timeout_s) ran out "
                                                               "before every design was evaluated at the finer settings"}
    elif not base_ok:
        checks["refinement"] = {"status": FAILED, "says": "the baseline could not be evaluated at the finer settings ("
                                + str((base_final or {}).get("reason") or "no value") + ")"}
    elif not named:
        checks["refinement"] = {"status": NOT_CHECKED, "says": (
            "the plan names no numerical setting (a mesh size, a time step, a tolerance), so the designs could only be "
            "evaluated again at the same settings: whether an improvement is a numerical error was not checked")}
    else:
        failed = [(x, j) for x in [base, *cands] for j in x["levels"] if (j or {}).get("status") != "ok"]
        says = "the baseline and the best designs were evaluated again at each finer level"
        if failed:
            says = (f"{len({id(x) for x, _j in failed})} design(s) could not be evaluated at every finer level ("
                    + "; ".join(str(j.get("reason") or "no value") for _x, j in failed[:2]) + ")")
        if first is not None and chosen is not None and chosen is not first and _meets(at(first), limits, margin):
            says += (f"; at the finest settings another of the search's best designs ({_describe(chosen['design'])}) is "
                     "better than the one the search chose, so it is the one reported")
        checks["refinement"] = {"status": FAILED if failed else PASSED, "says": says}

    # --- 2. limits
    broke = []
    if first is not None and measured_all and (at(first) or {}).get("status") == "ok":
        for name, op, bound, cu in limits:
            value = (at(first).get("constraints") or {}).get(name)
            if _number(value) and _slack(op, bound, float(value)) < margin:
                broke.append((name, op, bound, cu, float(value)))
    constraints_out: dict[str, Any] = {}
    if chosen is not None:
        for name, op, bound, cu in limits:
            value = float((at(chosen).get("constraints") or {})[name])
            slack = _slack(op, bound, value)
            e = err(chosen, name)
            constraints_out[name] = {"value": value, "limit": f"{op} {_fmt(bound)}", "unit": cu, "slack": slack,
                                     "numerical_error": e if named else None,
                                     "active": bool(abs(slack) <= max(e or 0.0, floor * (1.0 + abs(bound))))}
    if not measured_all:
        checks["constraints"] = {"status": NOT_CHECKED, "says": "not every design was evaluated at the finest settings"}
    elif not limits:
        checks["constraints"] = {"status": PASSED, "says": "the plan sets no limit"}
    elif cands and not ran_finest:
        checks["constraints"] = {"status": NOT_CHECKED, "says": "none of the best designs could be evaluated at the "
                                                                "finest settings"}
    elif broke:
        name, op, bound, cu, value = broke[0]
        c_u = f" {cu}" if cu else ""
        margin_text = f" with a margin of {_fmt(margin)}" if margin else ""
        tail = (f"; the design reported is the next best that meets every limit ({_describe(chosen['design'])})"
                if chosen is not None else "; no other design checked meets every limit there")
        checks["constraints"] = {"status": FAILED, "broken": [
            {"quantity": n, "limit": f"{o} {_fmt(b)}", "value": v} for n, o, b, _c, v in broke],
            "says": f"the best design the search found breaks the limit on {name} at finer settings ({_fmt(value)}{c_u} "
                    f"against a limit of {_o(op)} {_fmt(bound)}{c_u}{margin_text}){tail}"}
    elif chosen is None and cands:
        checks["constraints"] = {"status": FAILED, "says": "none of the best designs meets every limit at the finest "
                                                           "settings"}
    else:
        active = [n for n, c in constraints_out.items() if c["active"]]
        checks["constraints"] = {"status": PASSED, "says": "every limit is met at the finest settings" + (
            f"; the design sits on the limit on {', '.join(active)} (it is met by less than its numerical error)"
            if active else "")}

    # --- 3. better than the baseline
    improvement: dict[str, Any] | None = None
    search_base_value = (search_record.get("baseline") or {}).get("objective")
    if chosen is not None and base_ok:
        base_value, best_value = base_final["objective"], at(chosen)["objective"]
        value = sign * (base_value - best_value)
        # The same design at the search's settings (not the search's best, when another one is reported).
        search_value = (sign * (search_base_value - chosen["search"])
                        if _number(search_base_value) and _number(chosen.get("search")) else None)
        e_best, e_base = err(chosen), err(base)
        e_sum = e_best + e_base if e_best is not None and e_base is not None else None
        threshold, threshold_info = _threshold(block, base_value)
        bar = threshold if threshold is not None else e_sum
        interval, measure, why_not = None, value, ""
        if noisy:
            diffs = [sign * (b - c) for b, c in zip(base_final.get("runs") or [], at(chosen).get("runs") or [])]
            interval = _stats.paired_permutation_test(diffs) if len(diffs) >= 2 else None
            if interval is None:
                measure = None
                why_not = ("a study with randomness needs at least two fresh runs of each design to tell an improvement "
                           "from chance (`check_runs` of at least 2)")
            else:
                measure = interval["ci_lower"]
        if bar is None and not why_not:
            why_not = ("the numerical error of the two designs could not be estimated (a finer level gave no value, or "
                       "there is only one finer level: give a second finer `check` value)")
        tiny = floor * (1.0 + abs(base_value))
        shown = measure is not None and bar is not None and measure > bar + tiny
        beyond_error = (measure is not None and measure > e_sum + tiny) if named and e_sum is not None else None
        improvement = {"value": value, "search": search_value, "unit": unit,
                       "relative": value / abs(base_value) if base_value else None,
                       "numerical_error": e_sum if named else None, "numerical_error_best": e_best if named else None,
                       "numerical_error_baseline": e_base if named else None, "threshold": threshold_info,
                       "bar": bar, "rule": "plan" if threshold is not None else "rule", "interval": interval,
                       "shown": bool(shown), "beyond_numerical_error": beyond_error}
        what = (f"{_fmt(abs(value))}{u} {'better' if value > 0 else 'worse'} than the baseline at the finest settings"
                + (" on average" if noisy else ""))
        if baseline_was_best:
            what += " (at the search's own settings no design beat the baseline; this is the best other design found)"
        elif search_value is not None:
            what += (f" (at the search's own settings: {_fmt(abs(search_value))}{u} "
                     f"{'better' if search_value > 0 else 'worse'})")
        ci = f" (the lower end of its 95% interval is {_fmt(measure)}{u})" if noisy and measure is not None else ""
        e_text = _fmt(e_sum) if e_sum is not None else "?"
        if not named and value > 0 and not why_not:
            # Nothing finer to compare with: the improvement is measured again, but its numerical error is not known.
            status, says = NOT_CHECKED, (f"{what}; the plan names no numerical setting, so its numerical error was not "
                                         "estimated")
        elif why_not:
            status, says = NOT_CHECKED, f"{what}; {why_not}"
        elif shown:
            status = PASSED
            says = (f"{what}, more than the plan's threshold of {_fmt(threshold)}{u}{ci}" if threshold is not None else
                    f"{what}, more than the numerical error of the two designs ({e_text}{u}){ci}")
            if beyond_error is False:
                says += (f"; but not more than the numerical error of the two designs ({e_text}{u}), so it is not "
                         "shown to be more than a numerical error")
            elif named and beyond_error is None:
                says += "; the numerical error of the two designs could not be estimated"
        else:
            status = FAILED
            if value <= 0:
                says = (f"the improvement over the baseline disappears at finer settings: the best design is {what}"
                        if search_value is not None and search_value > 0 else f"the best design is {what}")
            elif noisy and measure is not None and measure <= bar + tiny and measure <= 0:
                says = (f"the best design is {what}{ci}, so the improvement over the baseline is not shown to be more "
                        "than chance")
            elif threshold is not None:
                says = f"{what}, not more than the plan's threshold of {_fmt(threshold)}{u}{ci}"
            elif named:
                says = (f"{what}, within the numerical error of the two designs ({e_text}{u}){ci}: the "
                        "improvement over the baseline is not shown to be more than a numerical error")
            else:
                says = f"{what}{ci}, not better than the baseline"
        if limits and not _meets(base_final, limits, margin):
            says += "; the baseline itself breaks a limit at the finest settings"
        checks["improvement"] = {"status": status, "says": says}
    elif not measured_all or not base_ok:
        checks["improvement"] = {"status": NOT_CHECKED, "says": "the baseline and the best design were not both "
                                                                "evaluated at the finest settings"}
    elif cands and not ran_finest:
        checks["improvement"] = {"status": NOT_CHECKED, "says": "none of the best designs could be evaluated at the "
                                                                "finest settings"}
    elif not cands:
        checks["improvement"] = {"status": FAILED, "says": (
            "the search found no design other than the baseline that meets every limit, so nothing improves on it")}
    else:
        checks["improvement"] = {"status": NOT_CHECKED, "says": "none of the best designs meets every limit at the "
                                                                "finest settings"}

    # --- 4. the starting points
    method = str((search_record.get("method") or {}).get("used") or "")
    starts = _starts(rows)
    if method == "exhaustive":
        checks["starts"] = {"status": PASSED, "says": "every combination of the design variables was evaluated"}
    elif chosen is None:
        checks["starts"] = {"status": NOT_CHECKED, "says": "no design was chosen at the finest settings"}
    elif len(starts) < 2:
        checks["starts"] = {"status": NOT_CHECKED, "says": (
            "the search ran from one starting point, so whether another start finds a different design was not checked")}
    else:
        others = [c for c in cands if c is not chosen and _meets(at(c), limits, margin)]
        worse, flat = [], []
        chosen_value = at(chosen)["objective"]
        threshold, _t = _threshold(block, chosen_value)
        for other in others:
            diff = sign * (at(other)["objective"] - chosen_value)
            tol = max(threshold or 0.0, (err(chosen) or 0.0) + (err(other) or 0.0), floor * (1.0 + abs(chosen_value)))
            if _same(other["design"], chosen["design"], space):
                continue
            (flat if diff <= tol else worse).append((other, diff))
        agreeing = len(chosen.get("also_found_by") or [])
        if worse:
            other, diff = worse[0]
            checks["starts"] = {"status": FAILED, "says": (
                f"the starting points found different designs: another start's best ({_describe(other['design'])}) is "
                f"{_fmt(diff)}{u} worse at the finest settings, so the best design found may not be the best there is")}
        elif flat:
            checks["starts"] = {"status": PASSED, "says": (
                f"another starting point found a different design ({_describe(flat[0][0]['design'])}) that is as good "
                "within the numerical error: more than one design is about as good here")}
        elif agreeing or others:
            checks["starts"] = {"status": PASSED, "says": "the starting points found the same design"}
        else:
            checks["starts"] = {"status": NOT_CHECKED, "says": (
                "the other starting points' best designs break a limit or could not be evaluated at the finest "
                "settings (or none was found), so there is nothing to compare this design with")}

    # --- 5. the neighbourhood
    at_bound, no_effect, better, sensitivity = [], [], [], {}
    if chosen is not None:
        centre = at(chosen)
        c_err = err(chosen) or 0.0
        for name, probe in probes.items():
            if probe["at_bound"]:
                at_bound.append({"variable": name, "edge": probe["at_bound"], "value": probe["value"]})
            sides = {s: p for s, p in probe["sides"].items() if (p["judged"] or {}).get("status") == "ok"}
            if not sides:
                continue
            if all(abs(p["judged"]["objective"] - centre["objective"]) <= 1e-12 * (1.0 + abs(centre["objective"]))
                   and all(abs(v - float((centre.get("constraints") or {}).get(n, v))) <= 1e-12 * (1.0 + abs(v))
                           for n, v in (p["judged"].get("constraints") or {}).items())
                   for p in sides.values()):
                no_effect.append(name)
            xs = {s: float(p["value"]) for s, p in sides.items()}
            fs = {s: float(p["judged"]["objective"]) for s, p in sides.items()}
            if "down" in xs and "up" in xs:
                sensitivity[name] = (fs["up"] - fs["down"]) / (xs["up"] - xs["down"])
            else:
                s = next(iter(xs))
                sensitivity[name] = (fs[s] - centre["objective"]) / (xs[s] - float(probe["value"]))
            for p in sides.values():
                judged = p["judged"]
                if not _meets(judged, limits, margin):
                    continue
                gain = sign * (centre["objective"] - judged["objective"])
                if noisy:
                    diffs = [sign * (a - b) for a, b in zip(centre.get("runs") or [], judged.get("runs") or [])]
                    test = _stats.paired_permutation_test(diffs) if len(diffs) >= 2 else None
                    clear = test is not None and test["ci_lower"] > c_err
                else:
                    clear = gain > c_err + floor * (1.0 + abs(centre["objective"]))
                if clear:
                    better.append({"variable": name, "value": p["value"], "objective": judged["objective"], "gain": gain})
    if chosen is None:
        checks["neighbourhood"] = {"status": NOT_CHECKED, "says": "no design was chosen at the finest settings"}
    elif not probes and finished:
        checks["neighbourhood"] = {"status": NOT_CHECKED, "says": "the design variables are all choices: nothing to nudge"}
    else:
        parts: list[str] = []
        if better:
            b = max(better, key=lambda x: x["gain"])
            parts.append(f"moving {b['variable']} to {_fmt(b['value'])} gives a design {_fmt(b['gain'])}{u} better at the "
                         "finest settings, more than the numerical error: this is not the best design nearby (the search "
                         "had not finished, or its coarser settings misled it)")
        for edge in at_bound:
            parts.append(f"the best design sits at the {'lower' if edge['edge'] == 'low' else 'upper'} edge of the range "
                         f"allowed for {edge['variable']} ({_fmt(edge['value'])}): a design outside the range may be "
                         "better (widening the range is a change to the plan)")
        if no_effect:
            parts.append(f"nudging {', '.join(no_effect)} changes nothing the simulation returns: the simulation may not "
                         "use it")
        status = FAILED if parts else PASSED
        says = "; ".join(parts) if parts else "no nearby design is better, and every variable changes the result"
        if not finished:
            status = FAILED if better else NOT_CHECKED
            if not parts:
                says = "no nudge evaluated so far gives a better design"
            says += " (the check's time ran out before every nudge was evaluated)"
        checks["neighbourhood"] = {"status": status, "says": says, "at_bound": at_bound, "no_effect": no_effect,
                                   "better_neighbour": sorted(better, key=lambda x: -x["gain"])[:3],
                                   "sensitivity": sensitivity, "nudge": NUDGE}

    # --- 6. the search's budget
    ev = search_record.get("evaluations") or {}
    stopped = str(ev.get("stopped_because") or "")
    if stopped in ("converged", "finished"):
        checks["budget"] = {"status": PASSED, "says": ("every combination was evaluated" if stopped == "finished"
                                                       else "every starting point converged")}
    else:
        why = {"budget": f"the search used its whole budget ({ev.get('search')} of {ev.get('budget')} evaluations) "
                         "before every starting point converged",
               "share": "at least one starting point used its whole share of the budget before it converged",
               "time": "the search's time limit (execution.timeout_s) was reached before every starting point "
                       "converged"}.get(stopped, f"the search stopped because {stopped or 'of an unknown reason'}")
        checks["budget"] = {"status": FAILED, "says": (
            f"{why}: the design is the best of {ev.get('search')} evaluations, not shown to be the best the search "
            "would find with more")}

    # --- the verdict
    imp_status = (checks.get("improvement") or {}).get("status")
    if not measured_all or not base_ok:
        verdict, reason = "unverified", (checks["refinement"]["says"])
    elif cands and not ran_finest:
        # The best designs did not run at the finest settings: nothing is known about their limits.
        verdict, reason = "unverified", checks["refinement"]["says"]
    elif cands and chosen is None:
        verdict, reason = "infeasible", checks["constraints"]["says"]
    elif imp_status == FAILED:
        verdict, reason = "improvement_not_shown", checks["improvement"]["says"]
    elif not named:
        # Evaluated again at the same settings only: nothing was checked at finer ones.
        verdict, reason = "unverified", checks["refinement"]["says"]
    elif any((j or {}).get("status") != "ok" for x in (base, chosen) if x for j in x["levels"]):
        # The reported design (or the baseline) gave no value at some finer level: its numbers are not all there.
        verdict, reason = "unverified", checks["refinement"]["says"]
    elif imp_status != PASSED:
        verdict, reason = "unverified", checks["improvement"]["says"]
    elif better:
        verdict, reason = "not_local_optimum", checks["neighbourhood"]["says"]
    elif not finished:
        verdict, reason = "unverified", ("the check's time limit (execution.timeout_s) ran out before every nudge "
                                         "around the best design was evaluated")
    else:
        verdict, reason = "verified", checks["improvement"]["says"]
    # The best design WAS evaluated at the finer settings, but a part of the check could not be done.
    partly = verdict == "unverified" and chosen is not None and base_ok and named and all(
        (j or {}).get("status") == "ok" for x in (base, chosen) for j in x["levels"])
    if reason.startswith("the improvement over the baseline"):
        says = reason[0].upper() + reason[1:] + "."  # the check's own sentence already says it
    elif partly:
        says = ("The best design was checked at finer numerical settings, but not every part of the check could be "
                f"done: {reason}.")
    else:
        says = VERDICTS[verdict][0].upper() + VERDICTS[verdict][1:] + (f": {reason}." if reason else ".")
    if chosen is not None and first is not None and chosen is not first:
        # The one sentence a person reads must not let the search's own best pass for the one that was checked.
        j = at(first) or {}
        why_other = ("could not be evaluated there" if j.get("status") != "ok" else
                     "breaks a limit there" if not _meets(j, limits, margin) else "is not as good there")
        says += (f" The design reported is not the one the search chose ({_describe(first['design'])}), which "
                 f"{why_other}.")

    def view(item: dict[str, Any] | None) -> dict[str, Any] | None:
        if item is None:
            return None
        e = estimate(item)
        out = {"design": item["design"],
               "objective": {"search": item.get("search"), "check": [(j or {}).get("objective") for j in item["levels"]]},
               "status": [(j or {}).get("status") for j in item["levels"]],
               "constraints": dict((at(item) or {}).get("constraints") or {}),
               "meets_every_limit": _meets(at(item), limits, margin) if at(item) else None,
               "numerical_error": e.get("error"), "error_estimate": e.get("how"), "observed_order": e.get("order")}
        if item.get("group") is not None:
            out["start"] = item["group"]
            out["also_found_by"] = item.get("also_found_by") or []
        return out

    best_view = view(chosen)
    if best_view is not None:
        best_view["constraints"] = constraints_out
        best_view["replaces_search_best"] = first is not None and chosen is not first
    return {
        "schema": SCHEMA,
        "objective": {k: objective.get(k) for k in ("quantity", "direction", "unit", "meaning") if objective.get(k)},
        "settings": {"search": info["search"], "levels": levels, "ratios": ratios, "where": info["where"],
                     "named": named},
        # ``planned`` counts runs (each evaluation × ``runs_each``), as plan.md does.
        "evaluations": {"check": counts["designs"], "failed": counts["failed"], "runs_each": check_runs if noisy else 1,
                        "runs": counts["designs"] * (check_runs if noisy else 1),
                        "planned": _plan.budget(block)["check"], "search": ev.get("search"), "budget": ev.get("budget"),
                        "stopped_because": stopped},
        "baseline": view(base),
        "best": best_view,
        "search_best": {"design": search_best.get("design"), "objective": search_best.get("objective")},
        "candidates": [view(c) for c in cands],
        "improvement": improvement,
        "checks": checks,
        "verdict": verdict,
        "passed": verdict == "verified" and all(c.get("status") == PASSED for c in checks.values()),
        "finished": finished,
        "says": says,
        "blind_spots": list(BLIND_SPOTS),
    }


def check_sync(block: dict[str, Any], search_record: dict[str, Any], rows: list[dict[str, Any]],
               evaluate: Callable[[dict[str, Any]], Any], *, noisy: bool = False, check_runs: int = 1) -> dict[str, Any]:
    """Run the check here, calling ``evaluate(cell)`` for each design: a dict of numbers, or (with randomness) a list of
    such dicts, one per fresh run; an exception is a failed evaluation."""
    gen = check(block, search_record, rows, noisy=noisy, check_runs=check_runs)
    try:
        request = next(gen)
        while True:
            try:
                got = evaluate(dict(request["cell"]))
                if isinstance(got, list):
                    names = set.intersection(*(set(t) for t in got)) if got else set()
                    reply = {"values": {n: sum(float(t[n]) for t in got) / len(got) for n in names},
                             "trials": [dict(t) for t in got]}
                else:
                    reply = {"values": dict(got), "trials": [dict(got)]}
            except Exception as exc:  # noqa: BLE001 -- a failed evaluation is recorded, not raised
                reply = {"failed": f"{type(exc).__name__}: {exc}"[:300]}
            request = gen.send(reply)
    except StopIteration as done:
        return done.value


# --- through FI's harness ---------------------------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_seeds(base_seed: int, runs: int, search_runs: int) -> tuple[list[int], list[int], str]:
    """``(the check's seeds, the search's seeds, the family the check's come from)``: fresh ones, none of which the
    search used (FI's harness derives each run's seed from the family and the run's number)."""
    from . import trial_runner as _trials

    search = [_trials.trial_seed(base_seed, "", t) for t in range(max(1, search_runs))]
    key = SEED_KEY
    for attempt in range(100):
        seeds = [_trials.trial_seed(base_seed, key, t) for t in range(max(1, runs))]
        if not set(seeds) & set(search) and len(set(seeds)) == len(seeds):
            return seeds, search, key
        key = f"{SEED_KEY}|{attempt + 1}"
    raise RuntimeError("no fresh seeds could be found for the check")


def _key(search_key: str, entry: str) -> str:
    """What decides the check: the search (its key covers the simulation, the plan's block and the seed), the entry,
    and FI's own check, search and harness code."""
    from . import trial_runner as _trials

    from . import optimise as _optimise

    code = [Path(m.__file__).read_bytes() for m in (_search, _plan, _optimise, _stats)]
    return _sha(b"\1".join([search_key.encode(), entry.encode(), Path(__file__).read_bytes(), *code,
                            _trials.HARNESS_SOURCE.encode()]))


def restore(quest_root: Path, text: str | None = None, key: str | None = None) -> bool:
    """Put FI's check back where a script changed or removed it: ``text`` is FI's copy from memory, else the one kept in
    ``.fi/optimisation/check.json``; with ``key`` (FI's copy from memory too), that record is put back as well.
    ``True`` when something was put back."""
    quest_root = Path(quest_root)
    put_back = False
    if text is not None and key is not None:
        record_text = json.dumps({"key": key, "text": text})
        try:
            current_record = (quest_root / RECORD).read_bytes()
        except OSError:
            current_record = None
        if current_record != record_text.encode("utf-8"):
            (quest_root / RECORD).parent.mkdir(parents=True, exist_ok=True)
            (quest_root / RECORD).write_bytes(record_text.encode("utf-8"))
            put_back = True
    if text is None:
        try:
            saved = json.loads((quest_root / RECORD).read_text(encoding="utf-8"))
            text = saved.get("text") if isinstance(saved, dict) else None
        except (OSError, ValueError):
            return False
    if not isinstance(text, str):
        return put_back
    path = quest_root / CHECK_PATH
    try:
        current = path.read_bytes()
    except OSError:
        current = None
    if current == text.encode("utf-8"):
        return put_back
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return True


def read(quest_root: Path) -> dict[str, Any] | None:
    """``needs/OPTIMUM_CHECK.json`` as FI wrote it, or ``None``."""
    try:
        data = json.loads((Path(quest_root) / CHECK_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(quest_root: Path, record: dict[str, Any], key: str) -> str:
    text = json.dumps(record, indent=1, allow_nan=False, default=str) + "\n"
    path = Path(quest_root) / CHECK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    saved = Path(quest_root) / RECORD
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(json.dumps({"key": key, "text": text}).encode("utf-8"))
    return text


def _clean(value: Any) -> Any:
    """A record with every non-finite number as ``None`` (JSON has none)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


async def run_check(executor: Any, python: Any, quest_root: Path, module: str, protocol: dict[str, Any], search: Any,
                    *, base_seed: int, timeout_s: int, env: dict[str, str] | None = None,
                    log: Any = None, fresh: bool = False) -> tuple[dict[str, Any], str, str]:
    """Check the search's best design (see the module docstring) and write ``needs/OPTIMUM_CHECK.json``:
    ``(the record, its text, the key it is kept under)``. ``search`` is the :class:`core.optimise.SearchRun`; ``timeout_s`` bounds the check alone.
    Never raises for a problem of the simulation: a check that cannot finish is ``unverified``. ``fresh``: never reuse
    a kept check (the confirm run of a frozen search: its check runs once more, on the confirm run's new seeds)."""
    from . import optimise as _optimise

    quest_root = Path(quest_root)
    block, _why = _plan.normalize((protocol or {}).get("optimisation"))
    record_in = search.record or {}
    entry = str(record_in.get("entry") or "run_cell")
    noisy = entry == "run_trial"
    files = getattr(search, "files", None) or {}
    try:
        search_saved = json.loads(files.get(_optimise.RECORD.as_posix(), "{}"))
    except ValueError:
        search_saved = {}
    search_key = str(search_saved.get("key") or "") if isinstance(search_saved, dict) else ""
    # The hash of the check FI attached to the search's record (which FI puts back from memory): a check kept on disk is
    # reused only when it is that one, so a script that rewrote it cannot have its version reused.
    kept_sha = search_saved.get("check_sha256") if isinstance(search_saved, dict) else None
    key = _key(search_key, entry)
    try:
        saved = json.loads((quest_root / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = None
    if (not fresh and isinstance(saved, dict) and saved.get("key") == key and search_key
            and isinstance(saved.get("text"), str)
            and kept_sha and _sha(saved["text"].encode("utf-8")) == kept_sha):
        try:
            again = json.loads(saved["text"])
        except ValueError:
            again = None
        # A check that finished is kept whatever its verdict; one its time limit cut short runs again.
        if isinstance(again, dict) and again.get("finished") is not False:
            restore(quest_root, saved["text"], key)
            if log is not None:
                log.info("[optimise] the search is unchanged: the check at finer settings FI already ran is used")
            return again, saved["text"], key
    ledger_text = files.get(_optimise.LEDGER_PATH.as_posix()) or ""
    rows = [json.loads(line) for line in ledger_text.splitlines() if line.strip()]
    rows = [r for r in rows if isinstance(r, dict) and r.get("event") == "evaluation"]
    runs = int(block.get("check_runs") or block.get("runs_per_evaluation") or 1) if (noisy and block) else 1
    search_runs = int(record_in.get("runs_per_evaluation") or 1)
    seeds, search_seeds, seed_key = check_seeds(int(base_seed), runs, search_runs) if noisy else ([], [], "")
    started = time.monotonic()
    deadline = started + max(1, int(timeout_s))
    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
    if block is None or record_in.get("best") is None:
        record = _nothing_to_check(block, record_in)
    else:
        if log is not None:
            planned = _plan.budget(block)["check"]
            log.info("[optimise] checking the best design at finer numerical settings (at most %s runs of the "
                     "simulation, apart from the search's budget)", planned)
        gen = check(block, record_in, rows, noisy=noisy, check_runs=runs)
        load_error = ""
        try:
            request = next(gen)
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    reply: dict[str, Any] = {"stop": "time"}
                elif load_error:
                    reply = {"failed": load_error}
                else:
                    reply, _err, load_error = await _optimise._evaluate(
                        executor, python, quest_root, module, entry, request["cell"], runs=runs,
                        base_seed=int(base_seed), timeout_s=left, env=env, thresholds=thresholds, seed_key=seed_key)
                request = gen.send(reply)
        except StopIteration as done:
            record = done.value
    record = _clean(record)
    record["ledger_sha256"] = _sha(ledger_text.encode("utf-8")) if ledger_text else None
    record["entry"] = entry
    record["time"] = {"seconds": round(time.monotonic() - started, 3), "limit_seconds": int(timeout_s)}
    if noisy:
        record["seeds"] = {"check": seeds, "search": search_seeds}
    text = _write(quest_root, record, key)
    return json.loads(text), text, key


def _nothing_to_check(block: dict[str, Any] | None, search_record: dict[str, Any]) -> dict[str, Any]:
    verdict = "infeasible" if block is not None else "unverified"
    says = ("The search found no design that meets every limit, so there was nothing to check at finer settings."
            if block is not None else
            VERDICTS[verdict][0].upper() + VERDICTS[verdict][1:] + ": the plan's optimisation block cannot be read.")
    return {"schema": SCHEMA, "verdict": verdict, "passed": False, "finished": True, "says": says,
            "checks": {}, "best": None, "baseline": None, "improvement": None,
            "evaluations": {"check": 0, "failed": 0, **{k: (search_record.get("evaluations") or {}).get(k)
                                                        for k in ("search", "budget", "stopped_because")}},
            "blind_spots": list(BLIND_SPOTS), "notes": []}


# --- plain words ------------------------------------------------------------------------------------------------------

_LABELS = {"refinement": "finer settings", "constraints": "limits", "improvement": "better than the baseline",
           "starts": "starting points", "neighbourhood": "nearby designs", "budget": "search budget"}


def summary_lines(record: dict[str, Any]) -> list[str]:
    """Plain lines for run.log about the check."""
    if not isinstance(record, dict) or not record:
        return []
    ev = record.get("evaluations") or {}
    lines = [f"[optimise] check at finer numerical settings: {record.get('says')}"]
    best = record.get("best")
    if isinstance(best, dict):
        objective = record.get("objective") or {}
        unit = f" {objective['unit']}" if objective.get("unit") else ""
        values = [v for v in (best.get("objective") or {}).get("check") or [] if _number(v)]
        if values:
            lines.append(f"[optimise] checked best design: {_describe(best.get('design') or {})}: "
                         f"{objective.get('quantity', 'the objective')} = {_fmt(values[-1])}{unit} at the finest settings")
    for name, label in _LABELS.items():
        c = (record.get("checks") or {}).get(name)
        if isinstance(c, dict):
            mark = {"passed": "ok", "failed": "NOT PASSED", "not_checked": "not checked"}.get(str(c.get("status")),
                                                                                                  str(c.get("status")))
            lines.append(f"[optimise]   {label}: {mark}: {c.get('says')}")
    where = (record.get("settings") or {}).get("where") or {}
    if where:
        rule = "FI's fixed rule (the plan gives no finer values)"
        lines.append("[optimise]   the finer settings come from "
                     + "; ".join(f"{n}: {'the plan' if w == 'plan' else rule}" for n, w in where.items()))
    runs = ev.get("runs", ev.get("check", 0))
    lines.append(f"[optimise]   the check ran the simulation {runs} time(s) (at most {ev.get('planned', '?')} planned), "
                 "apart from the search's budget")
    return lines


def summary_line(record: dict[str, Any]) -> str:
    """One plain sentence: the verdict and why."""
    return str((record or {}).get("says") or "The best design has not been checked at finer numerical settings.")


def attach_summary(record: dict[str, Any], text: str | None = None) -> dict[str, Any]:
    """What ``results/best_design.json`` carries of the check (the full record is ``needs/OPTIMUM_CHECK.json``, whose
    hash is ``record_sha256`` when ``text`` is given). ``improvement_numerical_error`` is the numerical error of the two
    designs added (the bar for the improvement by default); ``best_numerical_error`` the best design's own."""
    best = record.get("best") or {}
    base = record.get("baseline") or {}
    imp = record.get("improvement") or {}

    def finest(part: dict[str, Any]) -> Any:
        values = (part.get("objective") or {}).get("check") or []
        return values[-1] if values else None

    return {"verdict": record.get("verdict"), "says": record.get("says"), "passed": record.get("passed"),
            "design": best.get("design"), "objective": finest(best), "baseline_objective": finest(base),
            "improvement": imp.get("value"), "improvement_numerical_error": imp.get("numerical_error"),
            "best_numerical_error": best.get("numerical_error"), "interval": imp.get("interval"),
            "record": CHECK_PATH.as_posix(),
            **({"record_sha256": _sha(text.encode("utf-8"))} if text is not None else {})}


# --- the evidence ladder ----------------------------------------------------------------------------------------------


def evidence_gaps(quest_root: Path, protocol: dict[str, Any] | None) -> dict[str, list[str]]:
    """What the check says against each level of the evidence ladder, for a search for the best design (empty for any
    other study): one plain sentence per gap, keyed by the level it keeps the quest below."""
    from . import optimise as _optimise

    out: dict[str, list[str]] = {"protocol_runtime_matched": [], "independently_validated": [],
                                 "statistically_adequate": [], "publication_ready": []}
    block_raw = _optimise.block_of(protocol)
    if block_raw is None:
        return {}
    quest_root = Path(quest_root)
    block, _why = _plan.normalize(block_raw)
    best_path = quest_root / _optimise.BEST_PATH
    ledger_path = quest_root / _optimise.LEDGER_PATH
    if not best_path.is_file() or not ledger_path.is_file():
        out["protocol_runtime_matched"].append(
            "FI's record of the search for the best design (results/best_design.json and raw/optimisation_ledger.jsonl) "
            "is missing: nothing shows FI ran the search")
        return out
    ledger_bytes = ledger_path.read_bytes()
    record = read(quest_root)
    if block is not None:
        rows = []
        for line in ledger_bytes.decode("utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("event") == "evaluation":
                rows.append(row)
        budget = _plan.evaluations(block)[0]
        counted = sum(1 for r in rows if r.get("counted", True))
        if budget is not None and counted > budget:
            out["protocol_runtime_matched"].append(
                f"the search's record holds {counted} evaluations, more than the {budget} the plan allows")
        space = _search.Space(block["design_variables"])
        outside = [r for r in rows if isinstance(r.get("design"), dict) and (
            set(r["design"]) != set(space.names) or not space.inside(r["design"]))]
        if outside:
            out["protocol_runtime_matched"].append(
                f"{len(outside)} evaluation(s) in the search's record are of a design outside the plan's ranges")
    if record is None:
        out["independently_validated"].append(
            "the best design was not checked at finer numerical settings (needs/OPTIMUM_CHECK.json is missing): an "
            "improvement that is only a numerical error cannot be ruled out")
        return out
    checks = record.get("checks") or {}
    verdict = str(record.get("verdict") or "")
    if record.get("ledger_sha256") and record["ledger_sha256"] != _sha(ledger_bytes):
        out["protocol_runtime_matched"].append(
            "the search's record (raw/optimisation_ledger.jsonl) changed after FI checked the best design")
    elif not record.get("ledger_sha256") and checks:
        out["protocol_runtime_matched"].append(
            "the check of the best design does not say which record of the search it checked")
    try:
        attached = json.loads(best_path.read_text(encoding="utf-8")).get("check") or {}
    except (OSError, ValueError, AttributeError):
        attached = {}
    if not attached.get("record_sha256"):
        out["independently_validated"].append(
            "the check of the best design was not recorded in results/best_design.json, so the check file cannot be "
            "shown to be the one FI wrote for this search")
    elif attached.get("record_sha256") != _sha((quest_root / CHECK_PATH).read_bytes()):
        out["independently_validated"].append(
            "the check of the best design (needs/OPTIMUM_CHECK.json) is not the one FI recorded in "
            "results/best_design.json: it changed after FI wrote it")

    def says(name: str) -> str:
        return str((checks.get(name) or {}).get("says") or "")

    if record.get("finished") is False:
        out["independently_validated"].append(
            f"the check of the best design at finer numerical settings was not finished ({record.get('says')})")
    elif verdict == "unverified" and not checks:
        out["independently_validated"].append(str(record.get("says") or "the best design was not checked"))
    elif (verdict == "unverified" and (checks.get("constraints") or {}).get("status") == NOT_CHECKED
          and (checks.get("refinement") or {}).get("status") != FAILED):
        out["independently_validated"].append(str(record.get("says")))
    if (checks.get("refinement") or {}).get("status") == NOT_CHECKED and "no numerical setting" in says("refinement"):
        out["independently_validated"].append(
            "the plan names no numerical setting (a mesh size, a time step), so the best design could not be recomputed "
            "at finer settings: an improvement that is only a numerical error cannot be ruled out")
    elif (checks.get("refinement") or {}).get("status") == FAILED:
        out["independently_validated"].append(says("refinement"))
    if (checks.get("constraints") or {}).get("status") == FAILED:
        out["independently_validated"].append(says("constraints"))
    if verdict == "infeasible" and not checks:
        out["independently_validated"].append(str(record.get("says") or "no design meets every limit"))
    imp = checks.get("improvement") or {}
    named = bool((record.get("settings") or {}).get("named"))
    if imp.get("status") in (FAILED, NOT_CHECKED) and imp.get("says"):
        out["statistically_adequate"].append(says("improvement"))
    elif imp.get("status") == PASSED and (record.get("improvement") or {}).get("beyond_numerical_error") is False:
        out["statistically_adequate"].append(
            "the improvement over the baseline is above the plan's threshold but within the numerical error of the two "
            "designs, so it is not shown to be more than a numerical error")
    elif imp.get("status") == PASSED and named and (record.get("improvement") or {}).get("beyond_numerical_error") is None:
        out["statistically_adequate"].append(
            "the numerical error of the two designs could not be estimated, so the improvement over the baseline is not "
            "shown to be more than a numerical error")
    for name in ("neighbourhood", "starts", "budget"):
        c = checks.get(name) or {}
        if c.get("status") in (FAILED, NOT_CHECKED) and c.get("says"):
            if name == "neighbourhood" and c.get("status") == NOT_CHECKED and "all choices" in str(c.get("says")):
                continue
            out["publication_ready"].append(str(c["says"]))
    return {k: list(dict.fromkeys(v)) for k, v in out.items() if v}
