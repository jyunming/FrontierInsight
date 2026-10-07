"""The search for the best design: which design to evaluate next, and which one is best. Standard library only.

This file is self-contained on purpose (no import of FI): FI runs it in its own process, and writes a copy of it into
the quest, where it runs in the quest's own Python (``code/fi_search.py``, so ``code/run.py`` repeats the same search
without FI installed; ``.fi/optimisation/optimise_search.py``, where it drives a scipy or Optuna method, which only the quest's
environment may have).

The search never calls the simulation itself. :func:`search` is a generator that yields a request and is sent the
answer:

* ``{"kind": "evaluate", "cell": ..., "design": ..., "stage": ..., "start": ...}``: evaluate this design. The answer
  is ``{"values": {...}}`` (what the simulation returned), ``{"failed": "why"}``, or ``{"stop": "why"}`` (the time is up);
  ``elapsed_s`` may ride along.
* ``{"kind": "drive", "spec": ...}``: run one step of a library method (:func:`drive`) where the library is installed;
  the answer is what :func:`drive` returned there.

So the caller owns every evaluation (FI runs each in a process of its own and writes the record; ``run.py`` does the
same without FI), and the search owns the budget: it never asks for more evaluations than ``starts × per_start``, and
a design already evaluated is not evaluated (or counted) again.

A search can be continued (``rounds``: a person asked, after seeing the result, to search further, perhaps toward a
target value of the objective). Each round adds its evaluations to the budget and runs the local search again from the
best design found so far (and, when the added budget holds more than one start's share, from new points of a Latin
hypercube sample), and stops as soon as a design that meets every limit reaches the round's target. The search before
the rounds is the same search, with the same seed: what the rounds add is recorded under ``continued``.

Methods (``search_method`` in the plan):

* ``bounded_local``: a Nelder-Mead simplex inside the ranges from each starting point, each with ``per_start``
  evaluations; the first starting point is the baseline, the others the best points of the coarse scan and then a
  Latin hypercube sample.
* ``global_then_local``: the coarse scan (or, without one, a differential-evolution search with half the budget) over
  the whole range, then the local search from the best points found.
* ``exhaustive``: every combination, when every variable is whole-numbered or a choice and they all fit the budget.
* ``scipy:<name>`` / ``optuna:<sampler>``: the library's method, when the quest's environment has it; otherwise
  ``bounded_local``, and the record says why.

A design that breaks a limit is recorded and never chosen; a failed evaluation (an exception, a missing or non-finite
number) is recorded and never chosen. The random choices (the sample, the evolution) come from the quest's seed, so the
same seed gives the same search.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import sys
import time

LEDGER_SCHEMA = "fi.optimisation-ledger/v1"
RECORD_SCHEMA = "fi.best-design/v1"
BUILT_IN = ("bounded_local", "global_then_local", "exhaustive")
_LIMIT = re.compile(r"^\s*(<=|>=|<|>|≤|≥|=<|=>)\s*(\S+)\s*$")
_OPS = {"<=": "<=", "<": "<=", "≤": "<=", "=<": "<=", ">=": ">=", ">": ">=", "≥": ">=", "=>": ">="}
# After this many evaluations in a row that all failed, with none that worked, the simulation is broken, not the design.
_FAILURES_BEFORE_GIVING_UP = 10


class _Stop(Exception):
    """The whole search stops (the budget or the time is spent)."""


class _StartDone(Exception):
    """One local search stops (its share of the budget is spent, or it converged)."""


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _kind(variable):
    kind = str(variable.get("kind") or "").strip().lower()
    if kind:
        return kind
    return "choice" if isinstance(variable.get("values"), list) else "continuous"


def _snap(value):
    """A continuous value to twelve significant digits: a design taken into the unit cube and back (2.711 comes back as
    2.7110000000000003) is the same design, and is not evaluated, or counted, again."""
    return float(f"{float(value):.12g}")


def _fmt(value):
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


# --- the design space ------------------------------------------------------------------------------------------------


class Space:
    """The design variables, and the unit cube the search moves in (one coordinate per variable, 0 to 1)."""

    def __init__(self, variables):
        self.variables = [dict(v) for v in variables]
        self.names = [str(v["name"]) for v in self.variables]
        self.kinds = [_kind(v) for v in self.variables]

    @property
    def dims(self):
        return len(self.variables)

    def value(self, index, u):
        v, kind = self.variables[index], self.kinds[index]
        u = 0.5 if not _number(u) else min(1.0, max(0.0, float(u)))
        if kind == "choice":
            values = v["values"]
            return values[min(int(u * len(values)), len(values) - 1)]
        low, high = float(v["low"]), float(v["high"])
        x = low + u * (high - low)
        if kind == "integer":
            return int(min(max(round(x), int(v["low"])), int(v["high"])))
        return min(max(_snap(x), low), high)  # rounded, then kept inside the range

    def design(self, u):
        """The design at a point of the unit cube (clipped into it; whole numbers rounded)."""
        return {name: self.value(i, u[i]) for i, name in enumerate(self.names)}

    def canonical(self, design):
        """A design with each value in its variable's type: an int for a whole-numbered one, a float otherwise."""
        out = {}
        for name, kind, v in zip(self.names, self.kinds, self.variables):
            value = design[name]
            if kind == "integer":
                out[name] = int(round(float(value)))
            elif kind == "choice":
                out[name] = next((c for c in v["values"] if c == value or str(c) == str(value)), value)
            else:
                out[name] = min(max(_snap(value), float(v["low"])), float(v["high"]))
        return out

    def unit(self, design):
        out = []
        for name, kind, v in zip(self.names, self.kinds, self.variables):
            value = design[name]
            if kind == "choice":
                values = [str(c) for c in v["values"]]
                index = values.index(str(value)) if str(value) in values else 0
                out.append((index + 0.5) / len(values))
            else:
                low, high = float(v["low"]), float(v["high"])
                out.append(min(1.0, max(0.0, (float(value) - low) / (high - low))))
        return out

    def key(self, design):
        return json.dumps([design[name] for name in self.names])

    def inside(self, design):
        for name, kind, v in zip(self.names, self.kinds, self.variables):
            value = design.get(name)
            if kind == "choice":
                if not any(value == c for c in v["values"]):
                    return False
            elif not _number(value) or not (v["low"] <= value <= v["high"]):
                return False
            elif kind == "integer" and not float(value).is_integer():
                return False
        return True

    def combos(self):
        """Every design, when every variable is whole-numbered or a choice (``None`` otherwise)."""
        axes = []
        for kind, v in zip(self.kinds, self.variables):
            if kind == "choice":
                axes.append(list(v["values"]))
            elif kind == "integer":
                axes.append(list(range(int(v["low"]), int(v["high"]) + 1)))
            else:
                return None
        return [dict(zip(self.names, combo)) for combo in itertools.product(*axes)]

    def count(self):
        total = 1
        for kind, v in zip(self.kinds, self.variables):
            if kind == "choice":
                total *= len(v["values"])
            elif kind == "integer":
                total *= int(v["high"]) - int(v["low"]) + 1
            else:
                return None
        return total

    def steps(self):
        """The first step of a local search along each coordinate: a tenth of the range, at least one whole number
        or one choice."""
        out = []
        for kind, v in zip(self.kinds, self.variables):
            if kind == "choice":
                out.append(max(0.1, 1.0 / len(v["values"])))
            elif kind == "integer":
                out.append(max(0.1, 1.0 / max(1, int(v["high"]) - int(v["low"]))))
            else:
                out.append(0.1)
        return out


def search_settings(block):
    """The value each numerical setting takes during the search."""
    settings = block.get("numerical_settings") if isinstance(block.get("numerical_settings"), dict) else {}
    return {str(name): (s["search"] if isinstance(s, dict) else s) for name, s in settings.items()}


def check_settings(block):
    """The finer values each numerical setting will be checked at: the plan's, or half then a quarter of the search
    value (twice and four times when larger is finer). Kept equal to ``core.optimisation_plan.check_levels``."""
    settings = block.get("numerical_settings") if isinstance(block.get("numerical_settings"), dict) else {}
    out = {}
    for name, s in settings.items():
        s = s if isinstance(s, dict) else {"search": s}
        if isinstance(s.get("check"), list) and s["check"]:
            out[str(name)] = list(s["check"])
            continue
        search = s["search"]
        levels = [search * 2, search * 4] if str(s.get("finer") or "smaller") == "larger" else [search / 2, search / 4]
        if float(search).is_integer() and all(float(v).is_integer() for v in levels):
            levels = [int(v) for v in levels]
        out[str(name)] = levels
    return out


def cell_of(block, design):
    """What the simulation is called with: the conditions held fixed, the search's numerical settings, the design."""
    fixed = block.get("fixed") if isinstance(block.get("fixed"), dict) else {}
    return {**{str(k): v for k, v in fixed.items()}, **search_settings(block), **design}


def scan_grid(block):
    """The coarse scan as a grid of cells: every fixed condition and search setting at its one value, each design
    variable over the scan's values (or at the baseline's value when the scan leaves it out)."""
    grid = block.get("grid") if isinstance(block.get("grid"), dict) else {}
    baseline = block["baseline"]["values"]
    out = {name: [value] for name, value in cell_of(block, {}).items()}
    for v in block["design_variables"]:
        name = str(v["name"])
        values = grid.get(name)
        out[name] = list(values) if isinstance(values, list) and values else [baseline[name]]
    return out


def scan_rows(summary, block):
    """The coarse scan's results (a ``fi.trials/v1`` summary: one entry per setting) as evaluated designs."""
    names = [str(v["name"]) for v in block["design_variables"]]
    out = []
    for cell in (summary or {}).get("cells") or []:
        design = {name: cell["cell"][name] for name in names if name in cell.get("cell", {})}
        if len(design) != len(names):
            continue
        metrics = cell.get("metrics") or {}
        if cell.get("failed") or not cell.get("ok"):
            out.append({"design": design, "values": None,
                        "reason": cell.get("reason") or f"{cell.get('failed', 0)} of {cell.get('planned', 0)} run(s) of "
                                                         "this setting failed"})
            continue
        values = {}
        for name, m in metrics.items():
            vals = [x for x in m.get("values") or [] if _number(x)]
            if vals and len(vals) == len(m.get("values") or []):
                values[name] = math.fsum(vals) / len(vals)
        out.append({"design": design, "values": values, "reason": "", "elapsed_s": cell.get("elapsed_s")})
    return out


# --- judging one evaluation --------------------------------------------------------------------------------------------


def _parse_limit(text):
    """``("<=" or ">=", number)``, read the way ``core.optimisation_plan`` reads a limit (a bound is inclusive)."""
    match = _LIMIT.match(str(text))
    if not match:
        return None
    try:
        number = float(match.group(2))
    except ValueError:
        return None
    return (_OPS[match.group(1)], number) if math.isfinite(number) else None


def judge(block, values, reason=""):
    """What one evaluation gave: ``status`` (ok / failed), the ``objective``, each constrained quantity, how far past
    its limits the design is (``violation``, 0 when it meets every limit) and whether it is ``feasible``."""
    failed = {"status": "failed", "objective": None, "constraints": {}, "violation": None, "feasible": False,
              "reason": reason or "the simulation reported nothing for this design"}
    if not isinstance(values, dict):
        return failed
    quantity = block["objective"]["quantity"]
    if quantity not in values:
        return {**failed, "reason": f"the simulation returned {', '.join(sorted(map(str, values))[:8]) or 'nothing'} "
                                    f"but not the objective `{quantity}`"}
    objective = values[quantity]
    if not _number(objective):
        return {**failed, "reason": f"the objective `{quantity}` is not a finite number ({objective!r})"}
    constraints, violation = {}, 0.0
    for c in block.get("constraints") or []:
        name = str(c["quantity"])
        limit = _parse_limit(c.get("limit"))
        if limit is None:  # never dropped: a limit that cannot be read is not a design that meets it
            return {**failed, "reason": f"the limit on `{name}` ({c.get('limit')!r}) cannot be read"}
        if name not in values:
            return {**failed, "reason": f"the simulation did not return `{name}`, which a limit is set on"}
        value = values[name]
        if not _number(value):
            return {**failed, "reason": f"`{name}` is not a finite number ({value!r})"}
        constraints[name] = float(value)
        op, bound = limit
        over = max(0.0, value - bound) if op == "<=" else max(0.0, bound - value)
        violation += over / max(1.0, abs(bound))
    return {"status": "ok", "objective": float(objective), "constraints": constraints, "violation": violation,
            "feasible": violation == 0.0, "reason": ""}


def _sign(block):
    return 1.0 if block["objective"]["direction"] == "minimise" else -1.0


def _rank(judged, sign):
    """Smaller is better: a feasible design by its objective, then an infeasible one by how far past its limits it is,
    then a failed one."""
    if judged["status"] != "ok":
        return (2, 0.0, 0.0)
    if judged["feasible"]:
        return (0, sign * judged["objective"], 0.0)
    return (1, judged["violation"], sign * judged["objective"])


def effective_method(block):
    """``(method, where it comes from)``: the plan's ``search_method``, or the rule ``core.optimisation_plan`` shows in
    plan.md (``exhaustive`` when every design fits the budget, ``global_then_local`` with a coarse scan, else
    ``bounded_local``). Kept equal to ``core.optimisation_plan.effective_method``."""
    method = block.get("search_method")
    if isinstance(method, str) and method:
        return method, "plan"
    space = Space(block["design_variables"])
    budget = block.get("evaluation_budget") if isinstance(block.get("evaluation_budget"), dict) else None
    count = space.count()
    if count is not None and budget is not None and count <= int(budget["starts"]) * int(budget["per_start"]):
        return "exhaustive", "rule"
    if isinstance(block.get("grid"), dict) and block["grid"]:
        return "global_then_local", "rule"
    return "bounded_local", "rule"


# --- the search ------------------------------------------------------------------------------------------------------


def _reached(value, target, sign):
    """Whether an objective ``value`` reaches ``target`` (a minimised objective at or below it, a maximised one at or
    above it)."""
    return bool(_number(value) and _number(target) and sign * (value - target) <= 0)


class _Search:
    def __init__(self, block, seed, method, rounds=None):
        self.block = block
        self.space = Space(block["design_variables"])
        self.sign = _sign(block)
        self.seed = int(seed)
        self.rng = random.Random(f"fi-optimise|{self.seed}")
        budget = block.get("evaluation_budget") if isinstance(block.get("evaluation_budget"), dict) else {}
        self.starts = max(1, int(budget.get("starts") or 1))
        self.per_start = max(1, int(budget.get("per_start") or 1))
        self.total = self.starts * self.per_start
        requested, _where = (method, "plan") if method else effective_method(block)
        self.method_requested = requested
        self.method_used = requested
        self.method_why = ""
        self.rows = []
        self.cache = {}
        self.used = 0
        self.stopped = None
        self.notes = []
        self.start_log = []
        self.failures_in_a_row = 0
        self.any_ok = False
        self.baseline = self.space.canonical(block["baseline"]["values"])
        # A continued search (see the module docstring): each round asked for, and what it did.
        self.rounds = [r for r in (rounds or []) if isinstance(r, dict)]
        self.round_log = []
        self.target = None
        self.local_stage = "local"

    def _take_scan(self, scan):
        """The coarse scan's evaluated designs (not counted against the budget)."""
        for item in scan or []:
            try:
                design = self.space.canonical(item["design"])
            except (KeyError, TypeError, ValueError):
                continue
            key = self.space.key(design)
            if key in self.cache:
                continue
            judged = judge(self.block, item.get("values"), str(item.get("reason") or ""))
            self.cache[key] = judged
            self.any_ok = self.any_ok or judged["status"] == "ok"
            self._row(design, judged, "scan", None, item.get("elapsed_s"), counted=False)

    # one evaluation

    def _row(self, design, judged, stage, start, elapsed, counted=True):
        row = {"event": "evaluation", "n": len(self.rows) + 1, "stage": stage, "start": start, "design": dict(design),
               "objective": judged["objective"], "constraints": dict(judged["constraints"]),
               "feasible": judged["feasible"], "status": judged["status"],
               "method": ("scan" if stage == "scan" else "bounded_local" if stage == "continued" else
                          self.method_used), "counted": counted,
               "elapsed_s": elapsed if _number(elapsed) else None}
        if judged.get("reason"):
            row["reason"] = judged["reason"]
        self.rows.append(row)
        return row

    def evaluate(self, design, stage, start, cap=None):
        """Evaluate one design (a generator: it yields the request). A design already evaluated is not evaluated again
        and costs nothing; ``cap`` is where the current local search's share of the budget ends."""
        design = self.space.canonical(design)
        key = self.space.key(design)
        if key in self.cache:
            return self.cache[key]
        if self.stopped is not None:
            raise _Stop()
        if cap is not None and self.used >= cap:
            raise _StartDone("budget")
        if self.used >= self.total:
            self.stopped = "budget"
            raise _Stop()
        reply = yield {"kind": "evaluate", "cell": cell_of(self.block, design), "design": dict(design),
                       "stage": stage, "start": start}
        reply = reply if isinstance(reply, dict) else {"failed": "no answer for this design"}
        if reply.get("stop"):
            self.stopped = str(reply["stop"])
            raise _Stop()
        judged = judge(self.block, reply.get("values") if "values" in reply else None, str(reply.get("failed") or ""))
        self.used += 1
        self.cache[key] = judged
        self._row(design, judged, stage, start, reply.get("elapsed_s"))
        if judged["status"] == "ok":
            self.any_ok, self.failures_in_a_row = True, 0
            if self.target is not None and judged["feasible"] and _reached(judged["objective"], self.target, self.sign):
                # A continued search stops as soon as a design that meets every limit reaches the person's target.
                self.stopped = "target"
                raise _Stop()
        else:
            self.failures_in_a_row += 1
            if not self.any_ok and self.failures_in_a_row >= _FAILURES_BEFORE_GIVING_UP:
                self.stopped = "failures"
                raise _Stop()
        return judged

    def rank(self, design):
        return _rank(self.cache[self.space.key(self.space.canonical(design))], self.sign)

    def _f(self, u, stage, start, cap):
        judged = yield from self.evaluate(self.space.design(u), stage, start, cap)
        return _rank(judged, self.sign)

    # starting points

    def _lhs(self, n):
        """``n`` points of a Latin hypercube in the unit cube."""
        if n <= 0:
            return []
        columns = []
        for _ in range(self.space.dims):
            strata = list(range(n))
            self.rng.shuffle(strata)
            columns.append([(s + self.rng.random()) / n for s in strata])
        return [[columns[d][i] for d in range(self.space.dims)] for i in range(n)]

    def _best_designs(self, n, *, exclude=(), stages=None):
        """The best ``n`` distinct designs evaluated so far (feasible ones first), leaving out ``exclude``."""
        seen, out = set(exclude), []
        rows = [r for r in self.rows if r["status"] == "ok" and (stages is None or r["stage"] in stages)]
        for row in sorted(rows, key=lambda r: self.rank(r["design"])):
            key = self.space.key(self.space.canonical(row["design"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(row["design"])
            if len(out) >= n:
                break
        return out

    def _starting_points(self):
        """The baseline, then the best points of the coarse scan, then a Latin hypercube sample: ``starts`` in all."""
        points = [self.space.unit(self.baseline)]
        extra = self._best_designs(self.starts - 1, exclude=[self.space.key(self.baseline)], stages=("scan",))
        points += [self.space.unit(d) for d in extra]
        points += self._lhs(self.starts - len(points))
        return points[: self.starts]

    # the methods

    def run(self):
        try:
            probe, starts = None, None
            library = self.method_used.startswith(("scipy:", "optuna:"))
            if not library and self.method_used not in BUILT_IN:
                self._fall_back(f"{self.method_used} is not a method FI knows")
            # The baseline first, always, at the search's settings; then the coarse scan (not counted in the budget).
            yield from self.evaluate(self.baseline, "baseline", 0)
            if isinstance(self.block.get("grid"), dict) and self.block["grid"]:
                scan = yield {"kind": "scan", "grid": scan_grid(self.block)}
                self._take_scan(scan if isinstance(scan, list) else [])
            if library:
                # A library the quest's environment lacks is said once, before anything else is evaluated.
                starts = self._starting_points()
                probe = yield {"kind": "drive", "spec": self._drive_spec([], starts)}
                if not self._library_usable(probe):
                    self._fall_back(self._library_why(probe))
            if self.method_used == "exhaustive":
                yield from self._exhaustive()
            elif self.method_used.startswith(("scipy:", "optuna:")):
                yield from self._library(probe, starts)
            elif starts is not None:
                yield from self._local_starts(starts, first_cap=self.per_start)
            elif self.method_used == "global_then_local":
                yield from self._global_then_local()
            else:
                yield from self._local_starts(self._starting_points(), first_cap=self.per_start)
        except _Stop:
            pass
        if self.stopped is None:
            # Every starting point converged, or used its share of the budget before it did (``share``).
            self.stopped = ("budget" if self.used >= self.total else
                            "share" if any(s["stopped_because"] == "budget" for s in self.start_log) else "converged")
        for index, spec in enumerate(self.rounds):
            yield from self._round(index, spec)
        return self._outcome()

    # a continued search

    def _best_row(self):
        feasible = [r for r in self.rows if r["status"] == "ok" and r["feasible"]]
        return min(feasible, key=lambda r: (self.sign * r["objective"], r["n"])) if feasible else None

    def _round(self, index, spec):
        """One round of a continued search: ``added`` more evaluations, from the best design so far, toward the target
        when one is given (``search_target``, the person's target moved to the search's own settings, when the check
        at finer settings showed how far apart they are)."""
        added = max(1, int(spec.get("added") or self.per_start))
        target = float(spec["target"]) if _number(spec.get("target")) else None
        aim = float(spec["search_target"]) if _number(spec.get("search_target")) else target
        before = self._best_row()
        log = {"round": index + 1, "added": added, "target": target, "asked": str(spec.get("asked") or "")[:300],
               "from": dict(before["design"]) if before else dict(self.baseline),
               "best_before": before["objective"] if before else None, "evaluations": 0}
        if aim is not None and aim != target:
            log["search_target"] = aim
        # The round spends what it was given, never what an earlier part of the search left unused; the budget the
        # record states is the plan's and every round's.
        budget_after = self.total + added
        if self.stopped in ("time", "failures", "the simulation could not be loaded"):
            log.update(stopped_because="not_run", not_run_because=self.stopped)
        elif aim is not None and before is not None and _reached(before["objective"], aim, self.sign):
            log.update(stopped_because="target")
        else:
            used, starts_before = self.used, len(self.start_log)
            first = 1 + max((r["start"] for r in self.rows if isinstance(r.get("start"), int)), default=-1)
            self.total = self.used + added
            self.stopped, self.target, self.local_stage = None, aim, "continued"
            try:
                # From the best design so far, with the whole added budget; what it leaves when it converges goes to new
                # starting points (a share of ``per_start`` each), so the search does not stay in one valley.
                yield from self._local(self.space.unit(log["from"]), first, self.total)
                left = self.total - self.used
                if left > 0:
                    yield from self._local_starts(self._lhs(max(1, left // self.per_start)), offset=first + 1)
            except _Stop:
                pass
            finally:
                self.target, self.local_stage = None, "local"
            if self.stopped is None:
                mine = self.start_log[starts_before:]
                self.stopped = ("budget" if self.used >= self.total else
                                "share" if any(s["stopped_because"] == "budget" for s in mine) else "converged")
            log.update(evaluations=self.used - used, stopped_because=self.stopped)
        self.total = budget_after
        after = self._best_row()
        log.update(best_after=after["objective"] if after else None,
                   design_after=dict(after["design"]) if after else None,
                   target_reached=bool(target is not None and after is not None
                                       and _reached(after["objective"], target, self.sign)))
        self.round_log.append(log)

    def _fall_back(self, why):
        self.method_why = f"{why}; the built-in bounded_local search was used instead"
        self.notes.append(f"{self.method_requested} was asked for, but {self.method_why}")
        for row in self.rows:  # the baseline, evaluated before the method was known not to run: under the one that did
            if row["method"] == self.method_used:
                row["method"] = "bounded_local"
        self.method_used = "bounded_local"

    def _local_starts(self, points, *, first_cap=None, offset=0):
        for index, u0 in enumerate(points):
            if self.used >= self.total:
                self.stopped = "budget"
                break
            left = len(points) - index
            share = self.per_start if first_cap is not None else max(1, (self.total - self.used) // left)
            first = index == 0 and first_cap is not None
            cap = first_cap if first else self.used + share
            # The first starting point is the baseline, whose evaluation (the search's first) is its own.
            yield from self._local(u0, offset + index, min(cap, self.total),
                                   began=cap - self.per_start if first else None)

    def _local(self, u0, start, cap, began=None):
        began = self.used if began is None else max(0, began)
        try:
            yield from self._nelder_mead(u0, start, cap)
            why = "converged"
        except _StartDone as done:
            why = str(done)
        except _Stop:
            # The whole search stops (its budget or time): this starting point is still recorded.
            self._log_start(start, u0, began, self.stopped or "budget")
            raise
        self._log_start(start, u0, began, why)

    def _log_start(self, start, u0, began, why):
        self.start_log.append({"start": start, "from": self.space.design(u0), "evaluations": self.used - began,
                               "stopped_because": why, "best_objective": self._best_of_start(start)})

    def _best_of_start(self, start):
        rows = [r for r in self.rows if r["start"] == start and r["feasible"]]
        if not rows:
            return None
        return min(rows, key=lambda r: self.sign * r["objective"])["objective"]

    def _nelder_mead(self, u0, start, cap):
        """A Nelder-Mead simplex in the unit cube, every point clipped into it. It compares designs by their rank only
        (feasible by objective, then infeasible by how far past the limits, then failed)."""
        d = self.space.dims
        steps = self.space.steps()
        simplex = [list(u0)]
        for i in range(d):
            p = list(u0)
            p[i] = p[i] + steps[i] if p[i] + steps[i] <= 1.0 else p[i] - steps[i]
            simplex.append(p)
        values = []
        for p in simplex:
            values.append((yield from self._f(p, self.local_stage, start, cap)))
        proposals, most = 0, 20 * self.per_start + 50
        while True:
            order = sorted(range(d + 1), key=lambda k: values[k])
            simplex, values = [simplex[k] for k in order], [values[k] for k in order]
            diameter = max(max(abs(a - b) for a, b in zip(p, simplex[0])) for p in simplex[1:])
            # Converged: the simplex is a ten-thousandth of each range across (or all its points are one design, for
            # whole-numbered variables), or it is a thousandth across and every point gives the same value.
            if diameter < 1e-4 or len({self.space.key(self.space.design(p)) for p in simplex}) == 1:
                raise _StartDone("converged")
            if (all(v[0] == 0 for v in values) and diameter < 1e-3
                    and abs(values[-1][1] - values[0][1]) <= 1e-10 * (1.0 + abs(values[0][1]))):
                raise _StartDone("converged")
            proposals += 1
            if proposals > most:
                raise _StartDone("converged")
            centroid = [sum(p[j] for p in simplex[:-1]) / d for j in range(d)]
            worst = simplex[-1]

            def along(t):
                return [min(1.0, max(0.0, c + t * (c - w))) for c, w in zip(centroid, worst)]

            reflected = along(1.0)
            fr = yield from self._f(reflected, self.local_stage, start, cap)
            if fr < values[0]:
                expanded = along(2.0)
                fe = yield from self._f(expanded, self.local_stage, start, cap)
                simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
                continue
            if fr < values[-2]:
                simplex[-1], values[-1] = reflected, fr
                continue
            if fr < values[-1]:
                contracted = along(0.5)
                fc = yield from self._f(contracted, self.local_stage, start, cap)
                accepted = fc <= fr
            else:
                contracted = along(-0.5)
                fc = yield from self._f(contracted, self.local_stage, start, cap)
                accepted = fc < values[-1]
            if accepted:
                simplex[-1], values[-1] = contracted, fc
                continue
            best = simplex[0]
            for k in range(1, d + 1):
                simplex[k] = [b + 0.5 * (p - b) for b, p in zip(best, simplex[k])]
                values[k] = yield from self._f(simplex[k], self.local_stage, start, cap)

    def _global_then_local(self):
        if not any(r["stage"] == "scan" for r in self.rows):
            yield from self._evolution(max(self.space.dims + 2, self.total // 2))
        points = [self.space.unit(d) for d in self._best_designs(self.starts)] or [self.space.unit(self.baseline)]
        yield from self._local_starts(points)

    def _evolution(self, cap):
        """Differential evolution over the whole range until ``cap`` evaluations are spent (the baseline is one of the
        population)."""
        d = self.space.dims
        size = max(4, min(10 * d, cap // 3))
        population = [self.space.unit(self.baseline)] + self._lhs(size - 1)
        fitness = []
        try:
            for p in population:
                fitness.append((yield from self._f(p, "global", None, cap)))
            proposals, most = 0, 20 * cap + 50
            while True:
                for i in range(size):
                    proposals += 1
                    if proposals > most:
                        return
                    a, b, c = self.rng.sample([k for k in range(size) if k != i], 3)
                    mutant = [population[a][j] + 0.7 * (population[b][j] - population[c][j]) for j in range(d)]
                    mutant = [min(1.0, max(0.0, m)) for m in mutant]
                    keep = self.rng.randrange(d)
                    trial = [mutant[j] if (self.rng.random() < 0.9 or j == keep) else population[i][j] for j in range(d)]
                    ft = yield from self._f(trial, "global", None, cap)
                    if ft <= fitness[i]:
                        population[i], fitness[i] = trial, ft
        except _StartDone:
            return

    def _exhaustive(self):
        combos = self.space.combos()
        if combos is None or len(combos) > self.total:
            why = ("exhaustive needs every variable to be whole-numbered or a choice" if combos is None else
                   f"exhaustive needs all {len(combos)} combinations within the budget of {self.total} evaluations")
            self._fall_back(why)
            yield from self._local_starts(self._starting_points(), first_cap=self.per_start)
            return
        for design in combos:
            yield from self.evaluate(design, "exhaustive", 0)
        self.stopped = "finished"

    # a library's method, run where the library is

    def _drive_spec(self, answers, starts):
        return {"method": self.method_used, "seed": self.seed, "variables": self.space.variables,
                "baseline": dict(self.baseline), "starts": starts, "per_start": self.per_start, "total": self.total,
                "answers": answers}

    @staticmethod
    def _library_usable(reply):
        return isinstance(reply, dict) and ("ask" in reply or "done" in reply)

    def _library_why(self, reply):
        lib = self.method_requested.split(":", 1)[0]
        if isinstance(reply, dict) and reply.get("unavailable"):
            return f"{lib} is not installed in the quest's environment (FI does not install it)"
        why = str((reply or {}).get("error") or "it gave no answer") if isinstance(reply, dict) else "it gave no answer"
        return f"{self.method_requested} could not be run ({why[:200]})"

    def _scalar(self, judged):
        """The one number a library minimises: the objective (made smaller-is-better) for a feasible design, a large
        penalty that grows with the violation for an infeasible one, a larger one for a failed one."""
        base = self.cache.get(self.space.key(self.baseline))
        scale = abs(base["objective"]) if base and _number(base.get("objective")) else 1.0
        big = 1e6 * (1.0 + scale)
        if judged["status"] != "ok":
            return big * 10.0
        value = self.sign * judged["objective"]
        if judged["feasible"]:
            return value
        return big * (1.0 + judged["violation"]) + abs(value)

    def _library(self, reply, starts):
        answers, known = [], set()
        # A design the method asks for again under another point (two points that round to one whole number) costs
        # nothing and does not count against its share; so the number of steps is bounded here instead.
        steps, most = 0, 20 * self.total + 50
        while True:
            steps += 1
            if steps > most:
                self.notes.append(f"{self.method_used} kept asking for designs already evaluated; the search stopped")
                self.stopped = "converged"
                return
            if reply is None:
                reply = yield {"kind": "drive", "spec": self._drive_spec(answers, starts)}
            if isinstance(reply, dict) and reply.get("stop"):
                self.stopped = str(reply["stop"])
                return
            if not isinstance(reply, dict) or reply.get("error") or reply.get("unavailable"):
                self.notes.append(f"{self.method_used} stopped: {self._library_why(reply)}")
                self.stopped = "library_error"
                return
            if "done" in reply:
                self.notes.append(str(reply.get("done"))[:200])
                self.stopped = "budget" if self.used >= self.total else "converged"
                return
            ask = reply["ask"]
            reply = None
            if "u" in ask:
                design = self.space.design(ask["u"])
            else:
                design = self.space.canonical(ask["design"])
                if not self.space.inside(design):
                    design = self.space.design(self.space.unit(design))
            start = int(ask.get("start") or 0)
            before = self.used
            judged = yield from self.evaluate(design, "library", start)
            if ask["key"] not in known:
                known.add(ask["key"])
                answers.append({"key": ask["key"], "start": start, "value": self._scalar(judged),
                                "free": self.used == before})

    # the result

    def _judged_row(self, design):
        key = self.space.key(self.space.canonical(design))
        return next((r for r in self.rows if self.space.key(self.space.canonical(r["design"])) == key), None)

    def _outcome(self):
        feasible = [r for r in self.rows if r["status"] == "ok" and r["feasible"]]
        best = min(feasible, key=lambda r: (self.sign * r["objective"], r["n"])) if feasible else None
        base_row = self._judged_row(self.baseline)
        baseline = {"design": dict(self.baseline), "status": base_row["status"] if base_row else "not_run",
                    "objective": base_row["objective"] if base_row else None,
                    "constraints": dict(base_row["constraints"]) if base_row else {},
                    "feasible": bool(base_row and base_row["feasible"]),
                    **({"reason": base_row["reason"]} if base_row and base_row.get("reason") else {})}
        counted = [r for r in self.rows if r["counted"]]
        return {
            "rows": self.rows, "evaluations": self.used, "scan_evaluations": len(self.rows) - len(counted),
            "budget": self.total, "starts": self.starts, "per_start": self.per_start,
            "stopped_because": self.stopped,
            "budget_ran_out": self.used >= self.total,
            "method_requested": self.method_requested, "method_used": self.method_used, "method_why": self.method_why,
            "notes": list(self.notes), "baseline": baseline, "start_log": self.start_log, "seed": self.seed,
            "failed": sum(1 for r in self.rows if r["status"] != "ok"),
            "infeasible": sum(1 for r in self.rows if r["status"] == "ok" and not r["feasible"]),
            "best": ({"design": dict(best["design"]), "objective": best["objective"],
                      "constraints": dict(best["constraints"]), "feasible": True, "evaluation": best["n"],
                      "start": best["start"], "stage": best["stage"]} if best else None),
            "rounds": list(self.round_log),
        }


def search(block, *, seed, method=None, rounds=None):
    """The search as a generator of requests (see the module docstring); returns the outcome. ``rounds``: the rounds of
    a continued search, in order (each ``{"added": evaluations, "target": a value of the objective or None}``)."""
    return (yield from _Search(block, seed, method, rounds).run())


def grid_cells(grid):
    """Every combination of a grid's values (axes in order), as the trial runner makes them."""
    axes = list(grid.items())
    return [dict(zip((a for a, _ in axes), combo)) for combo in itertools.product(*(v for _, v in axes))]


def run_sync(block, evaluate, *, seed, method=None, scan_step=None, drive_step=None, deadline=None,
             max_evaluations=None, rounds=None):
    """Run the search here, calling ``evaluate(cell) -> dict`` for each design (an exception is a failed evaluation),
    ``scan_step(grid) -> rows`` for the coarse scan (default: ``evaluate`` on each of its cells) and
    ``drive_step(spec)`` (default: :func:`drive` in this process) for a library's method. ``deadline`` is a
    ``time.monotonic()`` time after which no evaluation starts; ``max_evaluations`` stops the search where an earlier
    one was stopped by its time limit (so ``code/run.py`` repeats that search, not a longer one)."""
    gen = search(block, seed=seed, method=method, rounds=rounds)
    done = 0
    names = [str(v["name"]) for v in block["design_variables"]]

    def one(cell):
        started = time.monotonic()
        try:
            values = evaluate(dict(cell))
            if not isinstance(values, dict):
                raise TypeError(f"the simulation returned {type(values).__name__}, not a dict of numbers")
            reply = {"values": {str(k): (float(v) if _number(v) else v) for k, v in values.items()}}
        except Exception as exc:  # noqa: BLE001 -- a failed design is recorded, not raised
            reply = {"failed": f"{type(exc).__name__}: {exc}"[:300]}
        reply["elapsed_s"] = round(time.monotonic() - started, 4)
        return reply

    try:
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                if scan_step is not None:
                    reply = scan_step(request["grid"])
                else:
                    reply = []
                    for cell in grid_cells(request["grid"]):
                        answer = one(cell)
                        reply.append({"design": {n: cell[n] for n in names}, "values": answer.get("values"),
                                      "reason": answer.get("failed", ""), "elapsed_s": answer["elapsed_s"]})
            elif request["kind"] == "evaluate":
                if ((deadline is not None and time.monotonic() >= deadline)
                        or (max_evaluations is not None and done >= max_evaluations)):
                    reply = {"stop": "time"}
                else:
                    reply = one(request["cell"])
                    done += 1
            else:
                reply = (drive_step or drive)(request["spec"])
            request = gen.send(reply)
    except StopIteration as finished:
        return finished.value


# --- the record ------------------------------------------------------------------------------------------------------


def _describe(design):
    return ", ".join(f"{k} = {_fmt(v)}" for k, v in design.items())


def best_design(outcome, block, *, seed, check=None, extra=None):
    """``results/best_design.json``: the best design that meets every limit, the baseline at the same settings, the
    improvement, the method, the evaluations used and why the search stopped. Found at the search's numerical settings
    only: nothing here says the design was checked at finer ones (``checked_at_finer_settings`` is false)."""
    objective = block["objective"]
    sign = _sign(block)
    unit = str(objective.get("unit") or "")
    best, baseline = outcome["best"], outcome["baseline"]
    improvement = None
    if best is not None and _number(baseline.get("objective")):
        value = sign * (baseline["objective"] - best["objective"])
        # The plan's threshold for "better", when it gives one. Without one the rule is "more than the numerical error
        # of the two designs", which only the check at finer settings can apply: not done here, so not decided here.
        tolerance = block.get("improvement_tolerance")
        tol_value = tolerance.get("value") if isinstance(tolerance, dict) else tolerance
        mode = str(tolerance.get("mode") or "absolute") if isinstance(tolerance, dict) else "absolute"
        threshold = None
        if _number(tol_value):
            threshold = tol_value * abs(baseline["objective"]) if mode == "relative" else tol_value
        improvement = {"value": value,
                       "relative": value / abs(baseline["objective"]) if baseline["objective"] else None,
                       "unit": unit, "better": value > 0, "baseline_feasible": bool(baseline["feasible"]),
                       "threshold": ({"value": tol_value, "mode": mode, "in_unit": threshold}
                                     if threshold is not None else None),
                       "beyond_threshold": (value > threshold) if threshold is not None else None}
    target = block.get("target")
    target_value = target.get("value") if isinstance(target, dict) else target
    evaluations = {"search": outcome["evaluations"], "budget": outcome["budget"], "starts": outcome["starts"],
                   "per_start": outcome["per_start"], "scan": outcome["scan_evaluations"],
                   "failed": outcome["failed"], "infeasible": outcome["infeasible"],
                   "budget_ran_out": outcome["budget_ran_out"], "stopped_because": outcome["stopped_because"]}
    u = f" {unit}" if unit else ""
    if best is None and not outcome["rows"] and outcome["stopped_because"] == "time":
        says = "No design was evaluated before the time limit (execution.timeout_s) ran out."
    elif best is None:
        says = (f"No design that meets every limit was found in {outcome['evaluations']} evaluation(s) "
                f"(of {outcome['budget']} allowed).")
    else:
        says = (f"Best design found in {outcome['evaluations']} evaluation(s) (of {outcome['budget']} allowed): "
                f"{_describe(best['design'])}; {objective['quantity']} = {_fmt(best['objective'])}{u}")
        if improvement is not None:
            says += (f", against {_fmt(baseline['objective'])}{u} for the baseline "
                     f"({abs(improvement['value']):.6g}{u} {'better' if improvement['better'] else 'not better'}")
            if improvement["beyond_threshold"] is False and improvement["better"]:
                says += ", less than the plan's threshold for better"
            says += ")"
        says += "."
    rounds = [r for r in outcome.get("rounds") or [] if isinstance(r, dict)]
    if rounds:
        says += (f" The search was continued {len(rounds)} time(s) at the person's request "
                 f"({sum(int(r.get('added') or 0) for r in rounds)} more evaluation(s) allowed).")
    says += (" Found and scored at the search's own numerical settings only; it has not been recomputed at finer "
             "settings.")
    record = {
        "schema": RECORD_SCHEMA,
        "objective": {k: objective.get(k) for k in ("quantity", "direction", "unit", "meaning") if objective.get(k)},
        "best": best,
        "baseline": {**baseline, "source": block["baseline"].get("source")},
        "improvement": improvement,
        "method": {"requested": outcome["method_requested"], "used": outcome["method_used"],
                   "why": outcome["method_why"]},
        "evaluations": evaluations,
        "starts": outcome["start_log"],
        "seed": seed,
        "fixed": dict(block.get("fixed") or {}),
        "search_settings": search_settings(block),
        "check_settings": dict(check) if check is not None else check_settings(block),
        "checked_at_finer_settings": False,
        "notes": outcome["notes"],
        "says": says,
    }
    asked = next((r for r in reversed(rounds) if _number(r.get("target"))), None)
    if asked is not None:
        # The person's latest target (a refine) is the one the result is held to.
        target_value, where = asked["target"], "refine"
    else:
        where = "plan"
    if _number(target_value):
        record["target"] = {"value": target_value, "from": where,
                            "reached": bool(best is not None and sign * (best["objective"] - target_value) <= 0)}
    if rounds:
        record["continued"] = rounds
        record["evaluations"]["planned_budget"] = outcome["starts"] * outcome["per_start"]
        record["evaluations"]["added_budget"] = sum(int(r.get("added") or 0) for r in rounds)
    if extra:
        record.update(extra)
    return plain_zero(json.loads(json.dumps(record, allow_nan=False, default=str)))


def plain_zero(value):
    """``value`` with every negative zero written as 0.0, at any depth: -0.0 (what ``0 - 0.0`` or a product with a
    negative number leaves) reads as a number with a sign when it is printed, and is no different from zero."""
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        return {k: plain_zero(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_zero(v) for v in value]
    return value


def ledger_lines(outcome, block):
    """The ledger: one line saying what the search was, then one line per evaluation."""
    head = {"schema": LEDGER_SCHEMA, "event": "search", "objective": block["objective"],
            "method": outcome["method_used"], "method_requested": outcome["method_requested"],
            "budget": outcome["budget"], "starts": outcome["starts"], "per_start": outcome["per_start"],
            "seed": outcome["seed"], "search_settings": search_settings(block)}
    return [json.dumps(plain_zero(head), default=str)] + [json.dumps(plain_zero(row), default=str) for row in outcome["rows"]]


# --- a library's method, where the library is installed ------------------------------------------------------------


class _Ask(Exception):
    pass


class _Enough(Exception):
    pass


def _keyword_seed(function, seed):
    import inspect

    try:
        names = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {}
    if "rng" in names:
        return {"rng": seed}
    if "seed" in names:
        return {"seed": seed}
    return {}


def _drive_scipy(spec, name, answers, counts):
    try:
        import scipy.optimize as so
    except ImportError:
        return {"unavailable": "scipy"}
    d = len(spec["variables"])
    bounds = [(0.0, 1.0)] * d
    starts, per, total, seed = spec["starts"], int(spec["per_start"]), int(spec["total"]), int(spec["seed"])

    def objective(start, limit):
        def f(x):
            point = [float(v) for v in x]
            key = json.dumps([start, point])
            if key in answers:
                return answers[key]
            if counts.get(start, 0) >= limit:
                raise _Enough()
            raise _Ask({"u": [min(1.0, max(0.0, v)) for v in point], "start": start, "key": key})
        return f

    lname = name.lower()
    try:
        if lname == "differential_evolution":
            so.differential_evolution(objective(0, total), bounds, x0=starts[0],
                                      **_keyword_seed(so.differential_evolution, seed))
        elif lname == "dual_annealing":
            so.dual_annealing(objective(0, total), bounds, x0=starts[0], maxfun=total,
                              **_keyword_seed(so.dual_annealing, seed))
        elif lname == "direct":
            so.direct(objective(0, total), bounds, maxfun=total)
        else:
            method = "Nelder-Mead" if lname == "minimize" else name
            spent = False
            for index, x0 in enumerate(starts):
                try:
                    so.minimize(objective(index, per), x0, method=method, bounds=bounds)
                except _Enough:
                    spent = True
            return {"done": f"scipy.optimize.minimize ({method}) finished from every starting point"
                            + (" (some within their share of the budget only)" if spent else ""), "budget": spent}
        return {"done": f"scipy.optimize.{name} finished"}
    except _Ask as ask:
        return {"ask": ask.args[0]}
    except _Enough:
        return {"done": f"scipy.optimize.{name} used the whole budget", "budget": True}
    except (ValueError, TypeError, AttributeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


_SAMPLERS = {"tpe": "TPESampler", "random": "RandomSampler", "cmaes": "CmaEsSampler", "cma-es": "CmaEsSampler",
             "qmc": "QMCSampler", "gp": "GPSampler", "nsgaii": "NSGAIISampler"}


def _drive_optuna(spec, name, answers, counts):
    try:
        import optuna
    except ImportError:
        return {"unavailable": "optuna"}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cls = getattr(optuna.samplers, _SAMPLERS.get(name.lower(), name), None)
    if cls is None:
        return {"error": f"optuna has no sampler named {name}"}
    try:
        sampler = cls(seed=int(spec["seed"]))
    except TypeError:
        sampler = cls()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(dict(spec["baseline"]))
    paid = counts.get(0, 0)
    for _ in range(20 * int(spec["total"]) + 50):
        if paid >= int(spec["total"]):
            break
        trial = study.ask()
        params = {}
        for v in spec["variables"]:
            kind, vname = _kind(v), str(v["name"])
            if kind == "choice":
                params[vname] = trial.suggest_categorical(vname, list(v["values"]))
            elif kind == "integer":
                params[vname] = trial.suggest_int(vname, int(v["low"]), int(v["high"]))
            else:
                params[vname] = trial.suggest_float(vname, float(v["low"]), float(v["high"]))
        key = json.dumps([0, params], sort_keys=True)
        if key in answers:
            study.tell(trial, answers[key])
            continue
        return {"ask": {"design": params, "start": 0, "key": key}}
    return {"done": "optuna used every trial of the budget", "budget": True}


def drive(spec):
    """One step of a library's method, replayed from the answers so far: ``{"ask": ...}`` (the next design it wants),
    ``{"done": why}``, ``{"unavailable": library}`` or ``{"error": why}``. The method is run afresh each time with the
    same seed and fed the answers already known, so it asks for the next design it has not been told about."""
    method = str(spec.get("method") or "")
    lib, _, name = method.partition(":")
    answers = {a["key"]: a["value"] for a in spec.get("answers") or []}
    counts = {}  # the evaluations each starting point has spent (a design already evaluated was free)
    for a in spec.get("answers") or []:
        if not a.get("free"):
            counts[a["start"]] = counts.get(a["start"], 0) + 1
    if lib == "scipy":
        return _drive_scipy(spec, name, answers, counts)
    if lib == "optuna":
        return _drive_optuna(spec, name, answers, counts)
    return {"error": f"no library named {lib!r}"}


def _main(argv):
    if argv[1:2] != ["--drive"] or len(argv) < 4:
        sys.stderr.write("usage: optimise_search.py --drive <spec.json> <out.json>\n")
        return 2
    with open(argv[2], encoding="utf-8") as fh:
        spec = json.load(fh)
    try:
        reply = drive(spec)
    except BaseException as exc:  # noqa: BLE001 -- reported to FI, which decides
        if isinstance(exc, KeyboardInterrupt):
            raise
        reply = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    reply["nonce"] = spec.get("nonce")
    with open(argv[3], "w", encoding="utf-8") as fh:
        json.dump(reply, fh)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
