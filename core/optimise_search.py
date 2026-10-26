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
* ``{"kind": "evaluate_many", "requests": [...]}`` (with ``batch``): designs chosen before any of their answers is read;
  the answer is a list, one answer per request, in order.
* ``{"kind": "evaluate_stream", "requests": [...]}`` (with ``stream``): the starting points running side by side ask for
  their designs without waiting for each other; each request has an ``id``, and the answer is a list of
  ``(id, answer)`` for the designs that have finished (at least one; the list of requests may be empty while designs run).
  A start goes on as soon as its own designs are answered; the rows are listed in the order of the step-by-step search.
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
* ``global_then_local``: the coarse scan (or, without one, a differential-evolution search with 15% of the budget) over
  the whole range, then the local search from the best points found.
* ``exhaustive``: every combination, when every variable is whole-numbered or a choice and they all fit the budget.
* ``scipy:<name>`` / ``optuna:<sampler>``: the library's method, when the quest's environment has it; otherwise
  ``bounded_local``, and the record says why.

A search whose design changes none of its results stops early (``no_effect``): when ``results`` names the results the
plan gives (besides the objective and the limits), and the first :data:`NO_EFFECT_DESIGNS` designs after the baseline,
which between them moved every number of the design, all returned exactly the baseline's value of every one of them.
That is what a simulation that does not use the design at the settings the search holds looks like; spending the rest of
the budget on it only finds the same numbers again.

A search whose objective does not move stops early too (``flat_objective``): when the first
:data:`NO_EFFECT_DESIGNS` designs after the baseline, which between them moved every number of the design, all gave the
baseline's value of the objective (other results may have moved). In FI's records of real searches, every search whose
objective ever moved did so at the first design after the baseline, and every one that had not moved by then gave the
same value for its whole budget (thousands of evaluations): the search has no direction to follow there. It is not a
fault of the simulation; the best design is the baseline, and the record says why.
The record also names which results beside the objective stayed at the baseline's value in every one of those designs
and which moved. With ``stop_flat`` off (the plan was already revised for it once) the fact is recorded the same way
and the search runs its whole budget.
A search whose objective is counted in whole steps of one of its own numerical settings stops before those two
(``stepped_objective``): when, among the same designs (and the baseline), at least :data:`STEPPED_MIN_VALUES` gave an
objective other than zero and at least :data:`STEPPED_SHARE` of those are whole multiples of the value the search uses
of a setting that is finer when smaller and has the objective's unit (a scan's spacing: a length counted as the number
of scan points inside times their spacing). The record names the setting; FI sends the simulation back to be repaired
once (read between the points of that scan), and with ``stop_stepped`` off the fact is only recorded.
The same stop, with the values in the record, ends a search whose objective moves only between a few values: when those
designs (and the baseline) gave no more than :data:`FEW_OBJECTIVE_VALUES` different values of it, and at least two of
them gave the same value away from the most common one (a single design apart may sit in a narrow region where the
objective moves smoothly). An objective that jumps between a few levels gives the search no direction either.

A design that breaks a limit is recorded and never chosen; a failed evaluation (an exception, a missing or non-finite
number) is recorded and never chosen. The random choices (the sample, the evolution) come from the quest's seed, so the
same seed gives the same search.
"""

from __future__ import annotations

import collections
import copy
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
#: The ``kind`` :func:`judge` gives a failed evaluation whose answer has results but not the objective: a fact about what
#: the simulation returns, not about the design, so a search that failed every evaluation this way says the same of the
#: alternatives it did not search (``every_failure``, :func:`compare`).
NO_OBJECTIVE = "no_objective"
#: Designs after the baseline that must all return the baseline's results before the search stops for ``no_effect``.
#: In FI's records of searches that do use their design, the first design that differs from the baseline was the first
#: or the third one tried; 50 leaves room for a design that does act but only past a threshold the first few miss.
NO_EFFECT_DESIGNS = 50
_SAME_RESULT = 1e-9
#: At most this many different values of the objective among the designs the ``flat_objective`` watch counts (and the
#: baseline) stop the search as one whose objective does not move. In FI's records of searches that found better designs,
#: their first 52 such designs gave 18 to 35 different values; the searches that found nothing gave 1, or 3 (an objective
#: read in whole steps of a setting of the simulation).
FEW_OBJECTIVE_VALUES = 3
#: The fewest values other than zero, among the designs the ``flat_objective`` watch counts and the baseline, and the
#: share of them that are whole multiples of one step setting, for the search to stop as ``stepped_objective``. In FI's
#: records, searches whose objective is a length counted in whole steps of a scan had 100% and 37% of their values on that
#: step; searches whose objective moved smoothly had none.
STEPPED_MIN_VALUES = 5
STEPPED_SHARE = 0.25


#: From how many design numbers, all continuous, the local stage is a CMA-ES (:meth:`_Search._cma_restarts`) instead of
#: Nelder-Mead simplexes: on test functions with known minima (8, 16 and 64 numbers, 600 to 18,000 evaluations) it found
#: the better design in most cases, and it asks for a whole population at a time instead of one design. Fewer numbers
#: were not measured, so they keep the simplex.
CMA_FROM_DIMS = 8
#: Whether the starting points of a CMA-ES local stage are each searched on their own share of the budget, side by side
#: (:meth:`_Search._side_by_side`), as the plan's ``starts`` ask. Off: one CMA-ES spends the budget from the first
#: starting point and begins again from the next ones only when it converges (:meth:`_Search._cma_restarts`). Measured on
#: analytic test functions with a known optimum (a smooth bowl and a multimodal one; 16, 64 and 144 numbers; 100, 300 and
#: 1000 evaluations per start; 3 seeds each; both ``bounded_local`` and ``global_then_local``), the shared-out search found
#: the worse design in 35 of 36 cases and tied in the other (both stuck at the same value); at 144 numbers and 1000
#: evaluations per start the bowl's best was 86 of 100 against 99.7. The search says so in its notes.
CMA_SPLIT = False
#: Whether differential evolution (the global stage, when the coarse scan gave no starting design) asks for a whole
#: generation of trials at a time instead of one trial after another. Steady-state (off) is the search as it was;
#: generational (on) lets the trials run at the same time. Measured on analytic test functions with a known optimum (3, 5
#: and 7 numbers, 100, 300 and 1000 evaluations per start, 3 seeds, a smooth and a multimodal function): the design the
#: whole search ended with was 0.12 of 100 worse on average (worse in 8 of 18 cases, better in 3, equal in 7) with 5 to
#: 10 times fewer waits for answers; over 144 numbers (the recorded plans) it was as good (99.76 against 99.70 on the
#: smooth one, 82.1 against 82.1 on the multimodal one) and the wait for answers was 13 times fewer. On by the
#: maintainer's decision (2026-10-23): a recorded 144-number plan's search, 8 at once, took 36 h of simulation time
#: instead of 97 h, with the same best design.
DE_GENERATIONAL = True
#: The share of the evaluation budget the global stage of ``global_then_local`` spends when there is no coarse scan (never
#: fewer evaluations than the watches of a flat objective count, ``NO_EFFECT_DESIGNS`` after the baseline); the
#: rest goes to the local stage. At equal wall-clock time on test functions with 8, 32 and 144 design numbers, 15% ranked
#: best in 14 of 26 cases and half the budget in 4 of 26.
GLOBAL_SHARE = 0.15
#: The CMA-ES's first step, as a share of each design number's range.
CMA_SIGMA0 = 0.3


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
    if isinstance(value, list):
        leaves = _leaves(value)
        if len(leaves) > 16:  # a long array is shown by its first values and its size
            return "[" + ", ".join(_fmt(x) for x in leaves[:8]) + f", ... ({len(leaves)} values)]"
        return "[" + ", ".join(_fmt(x) for x in value) + "]"
    return str(value)


def _leaves(value):
    return [x for item in value for x in _leaves(item)] if isinstance(value, list) else [value]


# --- array design variables --------------------------------------------------------------------------------------------
# A design variable with a ``shape`` is an array of numbers (``core.optimisation_plan`` checks the block). The search
# moves in one coordinate per element; a design carries the array back in its shape (a nested list), so the simulation,
# the ledger and the record see the array, not its elements.


def _shape(variable):
    shape = variable.get("shape")
    return [int(n) for n in shape] if isinstance(shape, list) else None


def _bound_at(bound, index):
    for i in index:
        if not isinstance(bound, list):
            break
        bound = bound[i]
    return bound


def element_name(name, index):
    return f"{name}[{','.join(str(i) for i in index)}]"


def scalar_variables(variables):
    """The numbers the search changes: a scalar variable as it is, an array variable as one scalar variable per element
    (named ``name[i,j]``, with its own ``low`` and ``high``), in row-major order."""
    out = []
    for v in variables:
        shape = _shape(v)
        if shape is None:
            out.append(v)
            continue
        for index in itertools.product(*[range(n) for n in shape]):
            out.append({"name": element_name(str(v["name"]), index), "kind": _kind(v), "of": str(v["name"]),
                        "index": list(index), "low": _bound_at(v["low"], index), "high": _bound_at(v["high"], index),
                        **({"unit": v["unit"]} if v.get("unit") else {})})
    return out


def _nest(shape, flat):
    """``flat`` (row-major) as a nested list of ``shape``."""
    if len(shape) == 1:
        return list(flat)
    step = math.prod(shape[1:])
    return [_nest(shape[1:], flat[i * step:(i + 1) * step]) for i in range(shape[0])]


def _flatten_to(shape, value):
    """The row-major values of a nested list of ``shape`` (``ValueError`` when it is not of that shape)."""
    if not shape:
        if isinstance(value, list):
            raise ValueError("an array has more levels than its shape")
        return [value]
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"an array must be a list of {shape[0]}")
    return [x for item in value for x in _flatten_to(shape[1:], item)]


# --- the design space ------------------------------------------------------------------------------------------------


class Space:
    """The design variables, and the unit cube the search moves in (one coordinate per variable, 0 to 1)."""

    def __init__(self, variables):
        self.variables = [dict(v) for v in variables]
        self.names = [str(v["name"]) for v in self.variables]
        self.kinds = [_kind(v) for v in self.variables]
        # One coordinate per number: an array variable is one coordinate per element.
        self.scalars = scalar_variables(self.variables)
        self.scalar_kinds = [_kind(v) for v in self.scalars]
        # The coordinates the search moves: an array element whose low equals its high is held at that value.
        self.free = [k for k, (v, kind) in enumerate(zip(self.scalars, self.scalar_kinds))
                     if kind == "choice" or "of" not in v or v["low"] != v["high"]]

    @property
    def dims(self):
        return len(self.free)

    def flat(self, design):
        """The values of a design, one per coordinate (an array variable's elements, row-major)."""
        out = []
        for v in self.variables:
            value = design[str(v["name"])]
            shape = _shape(v)
            out.extend(_flatten_to(shape, value) if shape is not None else [value])
        return out

    def fold(self, flat):
        """A design from one value per coordinate: an array variable is a nested list of its shape."""
        design, at = {}, 0
        for v in self.variables:
            shape = _shape(v)
            if shape is None:
                design[str(v["name"])] = flat[at]
                at += 1
            else:
                n = math.prod(shape)
                design[str(v["name"])] = _nest(shape, flat[at:at + n])
                at += n
        return design

    def element_design(self, flat_by_element):
        """A design from a mapping of coordinate names (``name`` or ``name[i,j]``) to values, as a library method gives it."""
        return self.fold([flat_by_element.get(str(v["name"]), v.get("low")) if "of" in v and v["low"] == v["high"]
                          else flat_by_element[str(v["name"])] for v in self.scalars])

    def value(self, index, u):
        """The value of the ``index``-th coordinate the search moves (one of :attr:`free`) at ``u`` of its range."""
        k = self.free[index]
        v, kind = self.scalars[k], self.scalar_kinds[k]
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
        flat = [v.get("low") for v in self.scalars]  # an element held at its one value stays there
        for i, k in enumerate(self.free):
            flat[k] = self.value(i, u[i])
        return self.fold(flat)

    def canonical(self, design):
        """A design with each value in its variable's type: an int for a whole-numbered one, a float otherwise."""
        flat = []
        for value, kind, v in zip(self.flat(design), self.scalar_kinds, self.scalars):
            if kind == "integer":
                flat.append(int(round(float(value))))
            elif kind == "choice":
                flat.append(next((c for c in v["values"] if c == value or str(c) == str(value)), value))
            else:
                flat.append(min(max(_snap(value), float(v["low"])), float(v["high"])))
        return self.fold(flat)

    def unit(self, design):
        """Where a design is in the unit cube the search moves in (one coordinate per number it changes)."""
        out = []
        flat = self.flat(design)
        for k in self.free:
            value, kind, v = flat[k], self.scalar_kinds[k], self.scalars[k]
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
        try:
            flat = self.flat(design)
        except (KeyError, ValueError, TypeError):
            return False
        for value, kind, v in zip(flat, self.scalar_kinds, self.scalars):
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
        for kind, v in zip(self.scalar_kinds, self.scalars):
            if kind == "choice":
                axes.append(list(v["values"]))
            elif kind == "integer":
                axes.append(list(range(int(v["low"]), int(v["high"]) + 1)))
            else:
                return None
        return [self.fold(list(combo)) for combo in itertools.product(*axes)]

    def count(self):
        total = 1
        for kind, v in zip(self.scalar_kinds, self.scalars):
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
        for k in self.free:
            kind, v = self.scalar_kinds[k], self.scalars[k]
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


def in_each_run(block):
    """The values every evaluation goes through together (``in_each_run``: all of them in one run, as in a measurement),
    each a list; ``{}`` when the block has none."""
    inside = block.get("in_each_run") if isinstance(block.get("in_each_run"), dict) else {}
    return {str(k): list(v) for k, v in inside.items() if isinstance(v, list)}


def cell_of(block, design):
    """What the simulation is called with: the conditions held fixed, the search's numerical settings, the values every
    evaluation goes through together (each list whole), the design."""
    fixed = block.get("fixed") if isinstance(block.get("fixed"), dict) else {}
    return {**{str(k): v for k, v in fixed.items()}, **search_settings(block), **in_each_run(block), **design}


def scan_grid(block):
    """The coarse scan as a grid of cells: every fixed condition and search setting at its one value, each design
    variable over the scan's values (or at the baseline's value when the scan leaves it out). A condition held fixed that
    the scan names is compared over the scan's values: those cells are measured and kept, and none but the cells at the
    value the plan holds can start the search (:func:`scan_rows`). With ``sub_studies`` the scan is a :class:`StudyGrid`:
    each part varies only some of the scan's settings, the others held, as a measurement's sub-studies do. The values
    every evaluation goes through together (:func:`in_each_run`) are not settings of the scan: the trial runner gives each
    cell all of them."""
    grid = block.get("grid") if isinstance(block.get("grid"), dict) else {}
    baseline = block["baseline"]["values"]
    inside = in_each_run(block)
    out = {name: [value] for name, value in cell_of(block, {}).items() if name not in inside}
    for name, values in grid.items():
        if str(name) in out and isinstance(values, list) and values:
            out[str(name)] = list(values)
    for v in block["design_variables"]:
        name = str(v["name"])
        values = grid.get(name)
        out[name] = list(values) if isinstance(values, list) and values else [baseline[name]]
    subs = block.get("sub_studies")
    if isinstance(subs, list) and subs and all(isinstance(s, dict) for s in subs):
        return StudyGrid(out, subs)
    return out


def _same(a, b):
    if _number(a) and _number(b):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def scan_rows(summary, block):
    """The coarse scan's results (a ``fi.trials/v1`` summary: one entry per setting) as evaluated designs."""
    names = [str(v["name"]) for v in block["design_variables"]]
    held = cell_of(block, {})
    compared = [str(k) for k in (block.get("grid") if isinstance(block.get("grid"), dict) else {})
                if str(k) in held]
    out = []
    for cell in (summary or {}).get("cells") or []:
        if cell.get("level") is not None:
            continue  # the numbers one run gave per value it went through: the same design, already its own row
        # A cell at another value of a condition the plan holds fixed is a comparison: it is kept in the trial record,
        # but it is not a design evaluated under the search's conditions, so it never seeds the search.
        if any(not _same(cell.get("cell", {}).get(k), held[k]) for k in compared):
            continue
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
        # Returned per value of what each run compares inside it: the numbers are in the value rows of this setting only.
        # They are given as an evaluation gives them, a number per value, so the objective and each limit's quantity are
        # judged the same way.
        for quantity in [str(block["objective"]["quantity"])] + [str(c["quantity"]) for c in block.get("constraints") or []]:
            if quantity not in values:
                per_value = _per_value(summary, cell, quantity)
                if per_value:
                    values[quantity] = per_value
        out.append({"design": design, "values": values, "reason": "", "elapsed_s": cell.get("elapsed_s")})
    return out


def _per_value(summary, cell, quantity):
    """``{value key: mean}`` of ``quantity`` over the value rows of ``cell`` (the rows the trial record keeps of a run's
    numbers per value), or ``{}`` when it has none."""
    from .in_run import level_key

    out = {}
    for row in (summary or {}).get("cells") or []:
        level = row.get("level")
        if not isinstance(level, dict) or row.get("ok") != row.get("planned"):
            continue
        parent = {k: v for k, v in (row.get("cell") or {}).items() if k not in level}
        vals = ((row.get("metrics") or {}).get(quantity) or {}).get("values") or []
        if parent == cell.get("cell") and vals and all(_number(x) for x in vals):
            out[level_key(level)] = math.fsum(vals) / len(vals)
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


def _per_value_reason(kind, name, block):
    """Why a quantity returned per value of what each run compares inside it cannot be judged: a search needs one number."""
    inside = ", ".join(sorted(in_each_run(block))) or "the values each run compares"
    return (f"the {kind} `{name}` is returned per value of {inside}, but a search needs one number per design "
            f"(the plan's {kind} must name one number)")


def judge(block, values, reason=""):
    """What one evaluation gave: ``status`` (ok / failed), the ``objective``, each constrained quantity, how far past
    its limits the design is (``violation``, 0 when it meets every limit) and whether it is ``feasible``."""
    failed = {"status": "failed", "objective": None, "constraints": {}, "violation": None, "feasible": False,
              "reason": reason or "the simulation reported nothing for this design"}
    if not isinstance(values, dict):
        return failed
    quantity = block["objective"]["quantity"]
    if quantity not in values:
        return {**failed, "kind": NO_OBJECTIVE,
                "reason": f"the simulation returned {', '.join(sorted(map(str, values))[:8]) or 'nothing'} "
                          f"but not the objective `{quantity}`"}
    objective = values[quantity]
    if isinstance(objective, dict):
        return {**failed, "reason": _per_value_reason("objective", quantity, block)}
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
        if isinstance(value, dict):
            return {**failed, "reason": _per_value_reason("constraint", name, block)}
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


def _unit(value):
    """A unit as written, compared without case or spaces (``nm`` and `` NM`` are one unit); empty when none."""
    return "".join(str(value or "").split()).lower()


def _whole_multiple(value, step):
    """Whether ``value`` is a whole number of ``step`` (other than none), up to rounding."""
    k = value / step
    return round(k) != 0 and abs(k - round(k)) <= 1e-6 * max(1.0, abs(k))


def _same_result(a, b):
    """Two values a simulation returned are the same: numbers within a relative ``1e-9`` (two NaN alike), anything else
    equal."""
    if _number(a) and _number(b):
        return abs(float(a) - float(b)) <= _SAME_RESULT * max(abs(float(a)), abs(float(b)))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


class _Search:
    def __init__(self, block, seed, method, rounds=None, results=None, batch=False, stop_flat=True,
                 stop_stepped=True, stream=False, first_reply=None):
        self.block = block
        #: The baseline's answer when it was asked for already (:func:`_first_evaluation`): used, counted and recorded
        #: as the search's first evaluation, not asked for again.
        self.first_reply = first_reply
        self.batch = bool(batch)
        #: Whether the starts running side by side are answered as each design finishes (:meth:`_stream`), not step by step.
        self.stream = bool(stream) and self.batch
        self.stop_flat = bool(stop_flat)
        self.stop_stepped = bool(stop_stepped)
        #: Answers asked for together (:meth:`evaluate_many`), by design key, each used once by :meth:`evaluate`.
        self.prefetched = {}
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
        #: For each row, where it stands in the step-by-step order (``(step, start)``), set only while the starts are
        #: answered as designs finish (:meth:`_stream`), which puts the rows back in that order afterwards.
        self.row_order = []
        self.order = None
        self.cache = {}
        self.used = 0
        #: Evaluations counted per starting point; while the starts run side by side (``own_caps``), each start's share
        #: of the budget is counted on its own evaluations, not on the whole search's.
        self.used_by_start = {}
        self.own_caps = False
        #: While the starts run side by side, each start's current share of the budget: what a start that converged did
        #: not spend is given to the starts still running (:meth:`_give_back`).
        self.shares = {}
        self.stopped = None
        self.notes = []
        self.run_not_used = []  # designs that finished when a stop fired but were not part of the search's record
        self.start_log = []
        self.start_order = []  # where each start's line stands in the step-by-step order, as ``row_order`` does
        self.failures_in_a_row = 0
        self.any_ok = False
        self.baseline = self.space.canonical(block["baseline"]["values"])
        # A continued search (see the module docstring): each round asked for, and what it did.
        self.rounds = [r for r in (rounds or []) if isinstance(r, dict)]
        self.round_log = []
        self.target = None
        self.local_stage = "local"
        # Whether the design changes any result (see the module docstring): the results compared, the baseline's values
        # of them, the coordinates no design has moved yet, and how many designs returned the baseline's values.
        held = {str(block["objective"]["quantity"])} | {str(c.get("quantity")) for c in block.get("constraints") or []
                                                         if isinstance(c, dict)}
        self.held_results = held
        # The results the plan names besides the objective and the limits: each evaluation's row keeps them, so a search
        # continued from FI's record (core/optimise.py) has every design's results as the simulation gave them.
        self.named = sorted({str(r) for r in results or []} - held)
        self.results = sorted({str(r) for r in results or []} | held) if set(results or []) - held else []
        self.base_results = None
        self.unmoved = set(self.space.free)
        self.as_baseline = 0
        self.no_effect = None
        # Whether the objective moves at all (``flat_objective``): the baseline's objective and whether it meets every
        # limit, the coordinates no design with that objective has moved yet, and how many designs gave it. Only designs
        # sampled across the whole space count (the coarse scan's and the global stage's); a local step near the baseline
        # shows nothing of the rest. The watch ends at the first design that gives another value, meets the limits
        # differently or fails.
        self.base_objective = None
        self.base_feasible = None
        self.flat_unmoved = set(self.space.free)
        self.flat_designs = 0
        self.flat_watch = True
        self.flat_objective = None
        # The plan's results beside the objective: their own copy of the names (``results`` is cleared once one moves),
        # the baseline's value of each, and the ones every counted flat design has left at that value.
        self.flat_names = sorted({str(r) for r in results or []} - {str(block["objective"]["quantity"])})
        self.flat_base = None
        self.flat_stayed = set()
        # Whether the objective takes more than a few values over the same designs (the same fact, with ``values``):
        # each value seen, with how many designs gave it and whether they met every limit (a value met both ways is two:
        # the search ranks them apart), and the coordinates none of them has moved yet. A failed design is not counted;
        # the watch ends at the first value past FEW_OBJECTIVE_VALUES.
        self.step_counts = []
        self.step_unmoved = set(self.space.free)
        self.step_designs = 0
        self.step_watch = True
        self.step_base = None
        # Whether the objective is counted in whole steps of one of the search's numerical settings
        # (``stepped_objective``): the settings that can be such a step (finer when smaller, in the objective's unit),
        # how many counted values other than zero are a whole multiple of each, and the coordinates not yet moved.
        unit = _unit(self.block.get("objective", {}).get("unit"))
        self.stepped_settings = {}
        for name, spec in (self.block.get("numerical_settings") or {}).items():
            if (isinstance(spec, dict) and _number(spec.get("search")) and spec["search"] > 0 and unit
                    and str(spec.get("finer") or "smaller") == "smaller" and _unit(spec.get("unit")) == unit):
                self.stepped_settings[str(name)] = float(spec["search"])
        self.stepped_hits = dict.fromkeys(self.stepped_settings, 0)
        self.stepped_nonzero = 0
        self.stepped_designs = 0
        self.stepped_unmoved = set(self.space.free)
        self.stepped_watch = bool(self.stepped_settings)
        self.stepped_objective = None

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
            self._row(design, judged, "scan", None, item.get("elapsed_s"), counted=False, values=item.get("values"))
            self._watch_objective(design, key, judged, "scan", item.get("values"), stop=False)
        # Decided on the whole scan, every setting recorded first: one that gives another value ends the watch.
        self._stepped_stop()
        self._flat_stop()
        self._steps_stop()

    # one evaluation

    def _row(self, design, judged, stage, start, elapsed, counted=True, values=None):
        row = {"event": "evaluation", "n": len(self.rows) + 1, "stage": stage, "start": start, "design": dict(design),
               "objective": judged["objective"], "constraints": dict(judged["constraints"]),
               "feasible": judged["feasible"], "status": judged["status"],
               "method": ("scan" if stage == "scan" else "bounded_local" if stage == "continued" else
                          self.method_used), "counted": counted,
               "elapsed_s": elapsed if _number(elapsed) else None}
        if judged.get("reason"):
            row["reason"] = judged["reason"]
        if judged.get("kind"):
            row["failure"] = judged["kind"]
        named = {n: values[n] for n in self.named if n in values} if isinstance(values, dict) else {}
        if named:
            row["results"] = named
        self.rows.append(row)
        self.row_order.append(self.order)
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
        cap = self._cap(start, cap)
        if cap is not None and (self.used_by_start.get(start, 0) if self.own_caps else self.used) >= cap:
            raise _StartDone("budget")
        if self.used >= self.total:
            self.stopped = "budget"
            raise _Stop()
        if key in self.prefetched:
            reply = self.prefetched.pop(key)
        else:
            reply = yield {"kind": "evaluate", "cell": cell_of(self.block, design), "design": dict(design),
                           "stage": stage, "start": start}
        reply = reply if isinstance(reply, dict) else {"failed": "no answer for this design"}
        if reply.get("stop"):
            self.stopped = str(reply["stop"])
            raise _Stop()
        if key in self.cache:
            # Another start, running beside this one, asked for the same design in the same batch and was answered
            # first: as one after another, the second costs nothing.
            return self.cache[key]
        judged = judge(self.block, reply.get("values") if "values" in reply else None, str(reply.get("failed") or ""))
        self.used += 1
        self.used_by_start[start] = self.used_by_start.get(start, 0) + 1
        self.cache[key] = judged
        self._row(design, judged, stage, start, reply.get("elapsed_s"), values=reply.get("values"))
        self._watch_effect(design, key, reply.get("values") if judged["status"] == "ok" else None)
        self._watch_objective(design, key, judged, stage, reply.get("values"))
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

    def evaluate_many(self, designs, stage, start, cap=None):
        """Evaluate designs chosen before any of their answers is read (a generator): with ``batch``, the ones the
        budget still allows are asked for together, then each is judged in order by :meth:`evaluate`, as one after
        another would be (a design after a stop is evaluated but never judged or recorded)."""
        designs = [self.space.canonical(d) for d in designs]
        if (self.batch or self.own_caps) and self.stopped is None and len(designs) > 1:
            room = self.total - self.used
            cap = self._cap(start, cap)
            if cap is not None:
                room = min(room, cap - (self.used_by_start.get(start, 0) if self.own_caps else self.used))
            todo, keys = [], set()
            for design in designs:
                key = self.space.key(design)
                if key in self.cache or key in keys or key in self.prefetched:
                    continue
                if len(todo) >= room:
                    break
                keys.add(key)
                todo.append(design)
            if len(todo) > 1:
                # Inside starts run side by side, one request: the batch of every start is made from it (_side_by_side).
                replies = yield from self._ask_all([
                    {"kind": "evaluate", "cell": cell_of(self.block, d), "design": dict(d), "stage": stage,
                     "start": start} for d in todo], together=self.batch or self.own_caps)
                for k, design in enumerate(todo):
                    self.prefetched[self.space.key(design)] = replies[k] if k < len(replies) else None
                mine = [self.space.key(d) for d in todo]
            else:
                mine = []
        else:
            mine = []
        out = []
        try:
            for design in designs:
                out.append((yield from self.evaluate(design, stage, start, cap)))
        finally:
            # Only this call's answers: starts running side by side each hold their own.
            for key in mine:
                self.prefetched.pop(key, None)
        return out

    def _fs(self, points, stage, start, cap):
        """:meth:`_f` for points chosen together (:meth:`evaluate_many`)."""
        judged = yield from self.evaluate_many([self.space.design(u) for u in points], stage, start, cap)
        return [_rank(j, self.sign) for j in judged]

    def _watch_effect(self, design, key, values):
        """Stop the search (``no_effect``) once enough designs, moving every number of the design between them, returned
        the baseline's value of every result compared. A design that changes one of them ends the watch."""
        if not self.results or not isinstance(values, dict):
            return
        got = {n: values[n] for n in self.results if n in values}
        if key == self.space.key(self.baseline):
            # Only the objective and the limits back from the simulation: a plateau of the objective is not a sign.
            self.base_results = got if set(got) - self.held_results else None
            if self.base_results is None:
                self.results = []
            return
        if self.base_results is None:
            return
        if set(got) != set(self.base_results) or not all(_same_result(got[n], self.base_results[n]) for n in got):
            self.results = []
            return
        here, base = self.space.flat(design), self.space.flat(self.baseline)
        self.unmoved -= {k for k in self.unmoved if not _same_result(here[k], base[k])}
        self.as_baseline += 1
        if self.as_baseline >= NO_EFFECT_DESIGNS and not self.unmoved:
            self.no_effect = {"designs": self.as_baseline, "results": sorted(self.base_results)}
            self.stopped = "no_effect"
            self.results = []  # the watch has fired: a design evaluated after the stop (the best one's report) is not watched
            raise _Stop()

    def _watch_objective(self, design, key, judged, stage, values=None, stop=True):
        """The watches of the objective: whole steps of a setting first (its record names the setting), then one value
        (its record also names the results that stayed), then a few values."""
        self._watch_stepped(design, key, judged, stage, stop)
        self._watch_flat(design, key, judged, stage, values, stop)
        self._watch_steps(design, key, judged, stage, stop)

    def _watch_flat(self, design, key, judged, stage, values=None, stop=True):
        """Stop the search (``flat_objective``) once enough designs sampled across the space (the coarse scan's and the
        global stage's), moving every number of the design between them, gave the baseline's value of the objective and
        met the limits as it did. A design that gives another value, meets them differently or fails ends the watch.
        ``stop=False`` only counts the design (the coarse scan's, decided once all of it is recorded: :meth:`_flat_stop`).
        ``values``: the results the simulation gave for the design, for the tally of those that stayed at the baseline's."""
        if not self.flat_watch or self.stopped is not None:
            return
        if judged["status"] != "ok" or not _number(judged.get("objective")):
            self.flat_watch = False
            return
        if key == self.space.key(self.baseline):
            self.base_objective, self.base_feasible = float(judged["objective"]), bool(judged["feasible"])
            if isinstance(values, dict):
                self.flat_base = {n: values[n] for n in self.flat_names if n in values}
                self.flat_stayed = set(self.flat_base)
            return
        if self.base_objective is None:
            return
        if not _same_result(judged["objective"], self.base_objective) or bool(judged["feasible"]) != self.base_feasible:
            self.flat_watch = False
            return
        if stage not in ("scan", "global"):
            return
        here, base = self.space.flat(design), self.space.flat(self.baseline)
        self.flat_unmoved -= {k for k in self.flat_unmoved if not _same_result(here[k], base[k])}
        self.flat_designs += 1
        if self.flat_base:
            got = values if isinstance(values, dict) else {}
            self.flat_stayed -= {n for n in self.flat_stayed if n not in got or not _same_result(got[n], self.flat_base[n])}
        if stop:
            self._flat_stop()

    def _flat_stop(self):
        """Record the fact (``flat_objective``) when the watch still holds over enough designs moving every number, and
        stop the search then unless ``stop_flat`` is off (the plan has been revised for it already): the watch ends."""
        if (self.flat_watch and self.stopped is None and self.flat_objective is None
                and self.flat_designs >= NO_EFFECT_DESIGNS and not self.flat_unmoved):
            base = set(self.flat_base or {})
            self.flat_objective = {"designs": self.flat_designs, "objective": self.base_objective,
                                   "feasible": bool(self.base_feasible), "stayed": sorted(self.flat_stayed),
                                   "moved": sorted(base - self.flat_stayed), "stopped": self.stop_flat}
            self.flat_watch = False
            if self.stop_flat:
                self.stopped = "flat_objective"
                raise _Stop()

    def _watch_stepped(self, design, key, judged, stage, stop=True):
        """Count whether the objective of a design sampled across the space (or of the baseline) is a whole multiple of
        each step setting; ``stop=False`` only counts it (:meth:`_stepped_stop` decides)."""
        if not self.stepped_watch or self.stopped is not None:
            return
        if judged["status"] != "ok" or not _number(judged.get("objective")):
            return
        baseline = key == self.space.key(self.baseline)
        if not baseline and stage not in ("scan", "global"):
            return
        value = float(judged["objective"])
        if abs(value) > _SAME_RESULT:
            self.stepped_nonzero += 1
            for name, step in self.stepped_settings.items():
                if _whole_multiple(value, step):
                    self.stepped_hits[name] += 1
        if baseline:
            return
        here, base = self.space.flat(design), self.space.flat(self.baseline)
        self.stepped_unmoved -= {k for k in self.stepped_unmoved if not _same_result(here[k], base[k])}
        self.stepped_designs += 1
        if stop:
            self._stepped_stop()

    def _stepped_stop(self):
        """Record the fact (``stepped_objective``) when enough designs moving every number gave enough values other than
        zero and enough of them sit on the step of one setting; stop the search then unless ``stop_stepped`` is off.
        Decided once, at the same point as the other watches: the watch ends either way."""
        if not (self.stepped_watch and self.stopped is None and self.stepped_designs >= NO_EFFECT_DESIGNS
                and not self.stepped_unmoved):
            return
        self.stepped_watch = False
        name, hits = max(self.stepped_hits.items(), key=lambda item: item[1])
        if self.stepped_nonzero < STEPPED_MIN_VALUES or hits < STEPPED_SHARE * self.stepped_nonzero:
            return
        self.stepped_objective = {"designs": self.stepped_designs, "setting": name, "step": self.stepped_settings[name],
                                  "unit": str(self.block.get("objective", {}).get("unit") or ""), "on_step": hits,
                                  "nonzero": self.stepped_nonzero, "stopped": self.stop_stepped}
        if self.stop_stepped:
            self.stopped = "stepped_objective"
            raise _Stop()

    def _watch_steps(self, design, key, judged, stage, stop=True):
        """Count the objective's value of a design sampled across the space (the coarse scan's and the global stage's)
        or of the baseline; ``stop=False`` only counts it (:meth:`_steps_stop` decides)."""
        if not self.step_watch or self.stopped is not None:
            return
        if judged["status"] != "ok" or not _number(judged.get("objective")):
            return
        baseline = key == self.space.key(self.baseline)
        if not baseline and stage not in ("scan", "global"):
            return
        value, feasible = float(judged["objective"]), bool(judged["feasible"])
        for level in self.step_counts:
            if _same_result(level[0], value) and level[2] == feasible:
                level[1] += 1
                break
        else:
            self.step_counts.append([value, 1, feasible])
            if len(self.step_counts) > FEW_OBJECTIVE_VALUES:
                self.step_watch = False
                return
        if baseline:
            self.step_base = value
            return
        here, base = self.space.flat(design), self.space.flat(self.baseline)
        self.step_unmoved -= {k for k in self.step_unmoved if not _same_result(here[k], base[k])}
        self.step_designs += 1
        if stop:
            self._steps_stop()

    def _steps_stop(self):
        """Record the fact (``flat_objective`` with ``values``, the most common first) when enough designs moving every
        number gave the objective no more than a few values, and two of them gave the same value away from the most
        common one (a step, given again), unless the one-value watch has recorded it already; stop the search then
        unless ``stop_flat`` is off."""
        if (self.step_watch and self.stopped is None and self.flat_objective is None
                and self.step_designs >= NO_EFFECT_DESIGNS and not self.step_unmoved):
            values = sorted((list(level) for level in self.step_counts), key=lambda level: -level[1])
            if not any(level[1] >= 2 for level in values[1:]):
                # Every design away from the common value gave a value of its own: the objective may move smoothly in a
                # region the sampled designs only touched (a window that opens), not in steps. Watched on.
                return
            self.flat_objective = {"designs": self.step_designs, "objective": values[0][0], "values": values,
                                   "start": self.step_base, "stayed": [], "moved": [], "stopped": self.stop_flat}
            self.step_watch = False
            if self.stop_flat:
                self.stopped = "flat_objective"
                raise _Stop()

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
            if (not library and self.method_used in ("bounded_local", "global_then_local") and self.space.dims + 1 > self.per_start
                    and any(_shape(v) is not None for v in self.space.variables)):
                self.notes.append(
                    f"the design has {self.space.dims} numbers to change, so a local search needs {self.space.dims + 1} "
                    f"evaluations just to start; {self.per_start} are allowed from each starting point, so a search may stop "
                    "before it improves anything (a library method, or more evaluations per starting point, fits better)")
            # The baseline first, always, at the search's settings; then the coarse scan (not counted in the budget).
            if self.first_reply is not None:
                self.prefetched[self.space.key(self.baseline)] = self.first_reply
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
        for index, spec in enumerate(self.rounds if self.stopped not in ("no_effect", "flat_objective", "stepped_objective",
                                                                         "scan_unchanged")
                                     else []):
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
        if spec.get("auto"):
            log["by"] = "FI"  # FI continued it itself: the search was still improving when its budget ran out
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
        if self._cma_fits():
            if len(points) > 1 and CMA_SPLIT:
                # Many continuous design numbers: a CMA-ES from each starting point, side by side, each on its share of
                # the budget (what a start that converged leaves goes to the others, :meth:`_give_back`), whichever method
                # placed the starting points. When every start converged before the budget was spent, the rest goes to
                # restarts with a larger population (:meth:`_cma_restarts`).
                yield from self._side_by_side(points, offset, cma=True)
                if self.used < self.total and self.stopped is None:
                    yield from self._cma_restarts(points, offset, first=len(points), lam=2 * self._cma_lam())
                return
            if len(points) > 1:
                note = (f"the design has {self.space.dims} continuous numbers, so the local search is one CMA-ES that spends "
                        f"the budget from the first starting point and begins again from the next only when it converges, "
                        f"not {len(points)} separate searches (on test functions with a known optimum, one run found the "
                        "better design than separate ones in 35 of 36 cases)")
                if note not in self.notes:
                    self.notes.append(note)
            yield from self._cma_restarts(points, offset)
            return
        if len(points) > 1:
            # The starting points do not depend on each other's answers: they run side by side (_side_by_side), however
            # many evaluations run at the same time, so the search is the same on any machine.
            yield from self._side_by_side(points, offset)
            return
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

    def _ask_all(self, requests, together):
        """The answers to ``requests`` (designs chosen before any answer is read), in order: asked for in one request
        when ``together``, else one after another."""
        if len(requests) == 1:
            return [(yield requests[0])]
        if together:
            replies, todo = [], list(requests)
            while todo:
                # While no answer has worked, ask only for as many designs as it takes to give up if all of them fail,
                # and look at the answers before asking for more: a broken simulation costs those, not the whole batch.
                # Once one works (or one is known to), the rest are asked for together, as before.
                step = len(todo)
                if not self._worked(replies):
                    step = min(step, max(1, _FAILURES_BEFORE_GIVING_UP - self.failures_in_a_row - len(replies)))
                if step == 1:
                    got = [(yield todo[0])]
                else:
                    got = yield {"kind": "evaluate_many", "requests": todo[:step]}
                    got = list(got) if isinstance(got, list) else []
                replies.extend(got)
                todo = todo[len(got):] if got else []
                if not self._worked(replies) and self.failures_in_a_row + len(replies) >= _FAILURES_BEFORE_GIVING_UP:
                    break  # the search gives up on the answers so far: it asks for no more
                if any(isinstance(r, dict) and r.get("stop") for r in got):
                    break
            return replies
        out = []
        for request in requests:
            out.append((yield request))
        return out

    def _worked(self, replies):
        """Whether the search has an answer that worked: one it judged already, or one of ``replies``."""
        return self.any_ok or any(
            judge(self.block, r.get("values") if isinstance(r, dict) else None,
                  str(r.get("failed") or "") if isinstance(r, dict) else "")["status"] == "ok" for r in replies)

    def _cap(self, start, cap):
        """Where a local search's share of the budget ends: its ``cap``, or, while the starts run side by side, the share
        it holds now (a start that converged gave its unused evaluations to the others)."""
        if cap is not None and self.own_caps:
            return self.shares.get(start, cap)
        return cap

    def _give_back(self, finished, live):
        """The evaluations the start ``finished`` did not use of its share, shared out evenly among the starts ``live``
        (the lowest numbers take the remainder), so the plan's whole budget is spent across its starts."""
        left = self.shares.get(finished, 0) - self.used_by_start.get(finished, 0)
        if left <= 0 or not live:
            return
        self.shares[finished] -= left
        each, extra = divmod(left, len(live))
        for k, other in enumerate(sorted(live)):
            self.shares[other] = self.shares.get(other, 0) + each + (1 if k < extra else 0)

    def _side_by_side(self, points, offset, cma=False):
        """The local searches from ``points`` at the same time, each on its own share of the budget (``per_start``, or an
        even share of what is left; a start that converges first gives what it did not use to the others): every step,
        each start's next design (or designs) are asked for together in one batch, and each start goes on from its own
        answers exactly as it would alone. A design two starts ask for at
        once is evaluated once. A stop of the whole search (budget, time) ends every start, each recorded."""
        # Each start's share counts what it has used already (the first start's is the baseline's evaluation): an even
        # share of what is left and what the starts used, at most ``per_start``. When it is short of ``per_start``, the
        # remainder goes to the first starts, so none of the budget is left over.
        n = max(1, len(points))
        room = self.total - self.used + sum(self.used_by_start.get(offset + i, 0) for i in range(n))
        share = max(1, min(self.per_start, room // n))
        extra = room - share * n if share < self.per_start else 0
        self.own_caps = True
        self.shares = {offset + i: share + (1 if i < extra else 0) for i in range(n)}
        # Streamed: each start also has a look-ahead copy (:meth:`_look_ahead`), made before any start takes its first step.
        look = ([self._look_ahead(u0, offset + i, i, cma) for i, u0 in enumerate(points)]
                if self.stream and len(points) > 1 else [])
        gens = {i: self._local(u0, offset + i, self.shares[offset + i], cma=cma) for i, u0 in enumerate(points)}
        pending = {}
        try:
            for i, gen in list(gens.items()):
                try:
                    pending[i] = next(gen)
                except StopIteration:
                    del gens[i]
                    self._give_back(offset + i, [offset + j for j in gens])
            while pending and not self.stream:
                asked, where, index = [], [], {}
                for i in sorted(pending):
                    request = pending[i]
                    parts = request["requests"] if request["kind"] == "evaluate_many" else [request]
                    slots = []
                    for part in parts:
                        key = self.space.key(self.space.canonical(part["design"]))
                        if key not in index:  # a design two starts ask for at once is evaluated once
                            index[key] = len(asked)
                            asked.append(part)
                        slots.append(index[key])
                    where.append((i, slots, request["kind"]))
                replies = yield from self._ask_all(asked, together=self.batch)
                pending = {}
                for i, slots, kind in where:
                    got = [replies[k] if k < len(replies) else None for k in slots]
                    answer = got if kind == "evaluate_many" else got[0]
                    try:
                        pending[i] = gens[i].send(answer)
                    except StopIteration:
                        del gens[i]
                        self._give_back(offset + i, [offset + j for j in gens])
            if pending and self.stream:
                yield from self._stream(gens, pending, offset, look)
        except _Stop:
            for i, gen in gens.items():
                try:
                    gen.throw(_Stop())
                except (_Stop, StopIteration):
                    pass
            raise
        finally:
            self.own_caps = False
            self.shares = {}

    def _look_ahead(self, u0, start, number, cma):
        """A light copy of the search that runs the local search of one start alone, for :meth:`_stream` to see what the start
        will ask for next. It is made before any start takes a step, with the random draws the earlier starts take at their
        first step already taken, so it asks for the designs the start asks for; it shares the block, the space and a read
        layer of the cache with the search (its own writes stay in the copy), and what it does is never recorded (a design it asks for that the search does not is only run, and listed apart if the
        search stops). Returns ``[copy, its generator, its pending request or None]``."""
        # Only what this start's search changes is copied. The record (rows, start log, notes, the rounds and the designs run
        # but not used) starts empty, the designs already evaluated are read through from the search's own cache, and the
        # block and the space are shared: with many starts over a large coarse scan, a full copy for each start would hold
        # the scan once per start.
        shared = ("rows", "row_order", "start_log", "start_order", "cache", "block", "space", "notes", "run_not_used",
                  "rounds", "round_log")
        copy_ = copy.copy(self)
        for name, value in vars(self).items():
            if name not in shared:
                setattr(copy_, name, copy.deepcopy(value))
        copy_.rows, copy_.row_order, copy_.start_log, copy_.start_order = [], [], [], []
        copy_.notes, copy_.run_not_used, copy_.rounds, copy_.round_log = [], [], [], []
        copy_.cache = collections.ChainMap({}, self.cache)
        for _ in range(number):
            copy_.rng.randrange(2 ** 32)
        gen = copy_._local(u0, start, copy_.shares[start], cma=cma)
        return [copy_, gen, None]

    def _stream(self, gens, pending, offset, look=()):
        """:meth:`_side_by_side` with every start's designs sent to the pool at once and answered as they finish: every
        design a start asks for goes to the pool when it is asked (``{"kind": "evaluate_stream", "requests": [...]}``,
        each with an ``id``), the driver answers with the ones that finished (``[(id, answer), ...]``, at least one), and a
        place in the pool is refilled the moment a design ends. The answers are used in the order the starts asked, not the
        order the designs finished in: a start goes on once all the designs of its own step are answered and every start
        that asked before it has gone on, so the rows, the budget charged and the design at which a count-threshold stop
        (``no_effect``, ``flat_objective``, ``stepped_objective``) fires depend on the plan alone, whatever the timing. The
        search itself therefore waits for a slow design, but the pool does not: each start also has a look-ahead copy
        (:meth:`_look_ahead`) that goes on as soon as its own answers are in and sends the designs it asks for to the pool
        early, so the search finds them answered when its turn comes. The copy changes nothing that is recorded: a design
        it asked for that the search never asks is only run, and listed apart if the search stops. A design two starts ask
        for is evaluated once and charged to the one that asked first. When a stop fires, designs that had already finished
        but come after it in that order are no part of
        the rows, the budget or the best; they are listed apart in the outcome (``run_not_used``)."""
        steps = {i: 0 for i in gens}
        needs = {}  # start -> the keys of the designs its step asks for, in order
        line = []  # the starts in the order they asked: they are answered in this order, whatever order designs finish in
        answers = {}  # key -> answer
        inflight = set()
        first_row, first_start = len(self.rows), len(self.start_log)
        new = []  # designs asked for and not yet handed to the pool
        asked = {}  # key -> the design asked for
        ahead = {}  # start -> [look-ahead copy, its generator, the request it waits on]
        for number, (copy_, gen, _none) in enumerate(look):
            try:
                ahead[number] = [copy_, gen, next(gen)]
            except Exception:  # noqa: BLE001 -- a look-ahead only helps: whatever ends it, the search goes on without it
                pass

        def look_ahead():
            """Let every look-ahead copy go on as far as the answers in hand take it; the designs it asks for next go to
            the pool at once, so a place in the pool is not idle while the search waits for a slow design."""
            moved = True
            while moved:
                moved = False
                for number in sorted(ahead):
                    copy_, gen, request = ahead[number]
                    parts = request["requests"] if request["kind"] == "evaluate_many" else [request]
                    keys = []
                    for part in parts:
                        key = self.space.key(self.space.canonical(part["design"]))
                        keys.append(key)
                        if key not in answers and key not in inflight:
                            inflight.add(key)
                            asked[key] = part["design"]
                            new.append(dict(part, id=key))
                    if not all(k in answers for k in keys):
                        continue
                    got = [answers[k] for k in keys]
                    try:
                        ahead[number][2] = gen.send(got if request["kind"] == "evaluate_many" else got[0])
                        moved = True
                    except Exception:  # noqa: BLE001 -- over (converged, stopped or its share spent), or a fault: drop it
                        del ahead[number]

        try:
            while pending:
                for i in sorted(pending):
                    if i in needs:
                        continue
                    request = pending[i]
                    parts = request["requests"] if request["kind"] == "evaluate_many" else [request]
                    keys = []
                    for part in parts:
                        key = self.space.key(self.space.canonical(part["design"]))
                        keys.append(key)
                        if key not in answers and key not in inflight and key not in self.cache:
                            inflight.add(key)
                            asked[key] = part["design"]
                            new.append(dict(part, id=key))
                    needs[i] = (keys, request["kind"])
                    line.append(i)
                look_ahead()
                went = False
                # Designs may finish in any order, but a start goes on only once every start that asked before it has:
                # what is recorded, what is charged and where a count-threshold stop fires then depend on the plan alone.
                while line and all(k in answers or k in self.cache for k in needs[line[0]][0]):
                    i = line.pop(0)
                    keys, kind = needs.pop(i)
                    got = [answers.get(k) for k in keys]
                    steps[i] += 1
                    self.order = (steps[i], i)
                    went = True
                    try:
                        pending[i] = gens[i].send(got if kind == "evaluate_many" else got[0])
                    except StopIteration:
                        del gens[i], pending[i]
                        self._give_back(offset + i, [offset + j for j in gens])
                    finally:
                        self.order = None
                    break  # the start that went on may ask for more: those go to the pool before the next start goes on
                if went:
                    continue
                if not new and not inflight:
                    raise RuntimeError("the search waits for a design nobody asked for")
                asked_now, new = new, []
                if not self._worked(list(answers.values())):
                    # As in :meth:`_ask_all`: while no answer has worked, only as many designs go to the pool as it takes
                    # to give up if all of them fail; the others wait until those are answered.
                    held = {n["id"] for n in asked_now}
                    out = len(inflight - held) + sum(1 for k in answers if k not in self.cache)
                    room = _FAILURES_BEFORE_GIVING_UP - self.failures_in_a_row - out
                    room = max(room, 1 if not out else 0)
                    asked_now, new = asked_now[:room], asked_now[room:]
                replies = yield {"kind": "evaluate_stream", "requests": asked_now}
                for key, reply in replies if isinstance(replies, list) else []:
                    answers[key] = reply
                    inflight.discard(key)
        except _Stop:
            # The search stops at the design its request order says. Designs that had already run to the end are not
            # part of its rows, its budget or its best (those do not depend on the timing); they are listed apart so the
            # designs run are all accounted for.
            self.run_not_used = [
                {"design": dict(asked[k]),
                 **({"values": dict(a["values"])} if "values" in a else {"failed": str(a.get("failed"))[:200]})}
                for k, a in answers.items()
                if k in asked and k not in self.cache and isinstance(a, dict) and ("values" in a or "failed" in a)]
            raise
        finally:
            for copy_, gen, _request in list(ahead.values()):
                gen.close()
            # Put this stretch's rows in the order of the step-by-step search, and number them again.
            tail = sorted(range(first_row, len(self.rows)), key=lambda k: (self.row_order[k] or (0, 0), k))
            rows = [self.rows[k] for k in tail]
            orders = [self.row_order[k] for k in tail]
            self.rows[first_row:] = rows
            self.row_order[first_row:] = orders
            for n, row in enumerate(self.rows[first_row:], start=first_row + 1):
                row["n"] = n
            tail = sorted(range(first_start, len(self.start_log)), key=lambda k: (self.start_order[k] or (0, 0), k))
            self.start_log[first_start:] = [self.start_log[k] for k in tail]
            self.start_order[first_start:] = [self.start_order[k] for k in tail]

    def _local(self, u0, start, cap, began=None, cma=False):
        began = self.used if began is None else max(0, began)
        try:
            yield from (self._cma(u0, start, cap) if cma else self._nelder_mead(u0, start, cap))
            why = "converged"
        except _StartDone as done:
            why = str(done)
        except _Stop:
            # The whole search stops (its budget or time): this starting point is still recorded.
            self._log_start(start, u0, began, self.stopped or "budget")
            raise
        self._log_start(start, u0, began, why)

    def _log_start(self, start, u0, began, why):
        used = self.used_by_start.get(start, 0) if self.own_caps else self.used - began
        self.start_log.append({"start": start, "from": self.space.design(u0), "evaluations": used,
                               "stopped_because": why, "best_objective": self._best_of_start(start)})
        self.start_order.append(self.order)

    def _best_of_start(self, start):
        rows = [r for r in self.rows if r["start"] == start and r["feasible"]]
        if not rows:
            return None
        return min(rows, key=lambda r: self.sign * r["objective"])["objective"]

    def _cma_fits(self):
        """Whether a local search here is a CMA-ES: at least :data:`CMA_FROM_DIMS` free design numbers, every one of
        them continuous, and numpy to compute it with."""
        if self.space.dims < CMA_FROM_DIMS:
            return False
        if any(self.space.scalar_kinds[k] != "continuous" for k in self.space.free):
            return False
        try:
            import numpy  # noqa: F401
        except ImportError:
            return False
        return True

    def _cma_lam(self):
        return 4 + int(3 * math.log(self.space.dims))

    def _cma_restarts(self, points, offset, first=0, lam=None):
        """A CMA-ES from the starting point number ``first`` with what is left of the budget; when it converges first, it
        starts again from the next starting point (and from the first again after the last) with twice the population
        (the "IPOP" restarts), until the budget is spent. A search with several starting points runs them side by side
        first (:meth:`_side_by_side`) and comes here only for what they left."""
        lam = lam or self._cma_lam()
        k = first
        while self.used < self.total:
            u0 = points[k % len(points)] if points else self.space.unit(self.baseline)
            began, start = self.used, offset + k
            try:
                yield from self._cma(u0, start, None, lam=lam)
                why = "converged"
            except _StartDone as done:
                why = str(done)
            except _Stop:
                self._log_start(start, u0, began, self.stopped or "budget")
                raise
            self._log_start(start, u0, began, why)
            if self.used == began:
                break  # nothing new was evaluated: every design it asked for was evaluated already
            k += 1
            lam *= 2
        self.stopped = self.stopped or ("budget" if self.used >= self.total else "converged")

    def _cma(self, u0, start, cap, lam=None):
        """A CMA-ES (covariance matrix adaptation evolution strategy) in the unit cube from ``u0``: each generation asks
        for a population of designs together, every point clipped into the cube, and moves its mean, step size and
        shape from their ranks only (feasible by objective, then infeasible by how far past the limits, then failed), as
        the simplex does. The standard settings (Hansen, "The CMA evolution strategy: a tutorial"); its random numbers
        come from the search's own seed, so the search is the same on any machine."""
        import numpy as np

        n = self.space.dims
        rng = np.random.default_rng(self.rng.randrange(2 ** 32))
        lam = int(lam or 4 + int(3 * math.log(n)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / float((weights ** 2).sum())
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        mean = np.clip(np.asarray(u0, dtype=float), 0.0, 1.0)
        sigma = CMA_SIGMA0
        pc, ps = np.zeros(n), np.zeros(n)
        b, d = np.eye(n), np.ones(n)
        c = np.eye(n)
        best, stale, generation = None, 0, 0
        most = 20 * self.per_start + 50
        while True:
            generation += 1
            if generation > most:
                raise _StartDone("converged")
            z = rng.standard_normal((lam, n))
            y = z @ (b * d).T
            x = np.clip(mean + sigma * y, 0.0, 1.0)
            values = yield from self._fs([list(map(float, row)) for row in x], self.local_stage, start, cap)
            order = sorted(range(lam), key=lambda k: values[k])
            if best is None or values[order[0]] < best:
                best, stale = values[order[0]], 0
            else:
                stale += 1
            # The steps actually taken (the clipped points), so a point held at a bound moves the search as it was run.
            steps = (x[order[:mu]] - mean) / sigma
            old = mean
            mean = mean + sigma * (weights @ steps)
            shift = (mean - old) / sigma
            inv_sqrt = b @ np.diag(1 / d) @ b.T
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt @ shift)
            norm_ps = float(np.linalg.norm(ps))
            hsig = norm_ps / math.sqrt(1 - (1 - cs) ** (2 * generation)) / chi_n < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + (math.sqrt(cc * (2 - cc) * mueff) * shift if hsig else 0.0)
            rank_mu = (steps.T * weights) @ steps
            c = ((1 - c1 - cmu) * c + c1 * (np.outer(pc, pc) + (0.0 if hsig else cc * (2 - cc)) * c)
                 + cmu * rank_mu)
            sigma = min(1.0, sigma * math.exp((cs / damps) * (norm_ps / chi_n - 1)))
            c = (c + c.T) / 2
            evals, vecs = np.linalg.eigh(c)
            d = np.sqrt(np.maximum(evals, 1e-20))
            b = vecs
            # Converged: every step is a ten-thousandth of each range (as the simplex's), or many generations in a row
            # found nothing better.
            if sigma * float(d.max()) < 1e-4 or stale > 10 + int(30 * n / lam):
                raise _StartDone("converged")

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
        values = yield from self._fs(simplex, self.local_stage, start, cap)
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
            values[1:] = yield from self._fs(simplex[1:], self.local_stage, start, cap)

    def _global_then_local(self):
        if not any(r["stage"] == "scan" for r in self.rows):
            # At least the designs the watches of a flat objective need (the baseline and NO_EFFECT_DESIGNS others), so what
            # they can detect does not depend on how small a share of a small budget the global stage gets.
            yield from self._evolution(max(self.space.dims + 2, int(self.total * GLOBAL_SHARE), NO_EFFECT_DESIGNS + 1))
        points = [self.space.unit(d) for d in self._best_designs(self.starts)] or [self.space.unit(self.baseline)]
        yield from self._local_starts(points)

    def _evolution(self, cap):
        """Differential evolution over the whole range until ``cap`` evaluations are spent (the baseline is one of the
        population). By default each trial is made from the population as it stands and kept at once when it is no worse
        than the member it was made for, so the trials follow one another; with :data:`DE_GENERATIONAL` the trials of a
        whole generation are made from the population as the generation began and asked for together."""
        d = self.space.dims
        size = max(4, min(10 * d, cap // 3))
        population = [self.space.unit(self.baseline)] + self._lhs(size - 1)
        try:
            fitness = yield from self._fs(population, "global", None, cap)
            proposals, most = 0, 20 * cap + 50
            while True:
                if DE_GENERATIONAL:
                    trials = []
                    for i in range(size):
                        proposals += 1
                        if proposals > most:
                            return
                        trials.append(self._de_trial(population, i, size, d))
                    results = yield from self._fs(trials, "global", None, cap)
                    for i in range(size):
                        if results[i] <= fitness[i]:
                            population[i], fitness[i] = trials[i], results[i]
                    continue
                for i in range(size):
                    proposals += 1
                    if proposals > most:
                        return
                    trial = self._de_trial(population, i, size, d)
                    ft = yield from self._f(trial, "global", None, cap)
                    if ft <= fitness[i]:
                        population[i], fitness[i] = trial, ft
        except _StartDone:
            return

    def _de_trial(self, population, i, size, d):
        """The trial for member ``i``: a mutant of three other members, crossed with member ``i`` (rand/1/bin)."""
        a, b, c = self.rng.sample([k for k in range(size) if k != i], 3)
        mutant = [population[a][j] + 0.7 * (population[b][j] - population[c][j]) for j in range(d)]
        mutant = [min(1.0, max(0.0, m)) for m in mutant]
        keep = self.rng.randrange(d)
        return [mutant[j] if (self.rng.random() < 0.9 or j == keep) else population[i][j] for j in range(d)]

    def _exhaustive(self):
        count = self.space.count()
        combos = self.space.combos() if count is not None and count <= self.total else None
        if combos is None:
            why = ("exhaustive needs every variable to be whole-numbered or a choice" if count is None else
                   f"exhaustive needs all {count} combinations within the budget of {self.total} evaluations")
            self._fall_back(why)
            yield from self._local_starts(self._starting_points(), first_cap=self.per_start)
            return
        yield from self.evaluate_many(list(combos), "exhaustive", 0)
        self.stopped = "finished"

    # a library's method, run where the library is

    def _drive_spec(self, answers, starts):
        free = [self.space.scalars[k] for k in self.space.free]  # an element held at its one value is not given
        flat = self.space.flat(self.baseline)
        return {"method": self.method_used, "seed": self.seed, "variables": free,
                "baseline": {str(self.space.scalars[k]["name"]): flat[k] for k in self.space.free},
                "starts": starts, "per_start": self.per_start, "total": self.total,
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
                design = self.space.canonical(self.space.element_design(ask["design"]))
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
            **({"every_failure": NO_OBJECTIVE}
               if self.stopped == "failures" and self.rows and all(r.get("failure") == NO_OBJECTIVE for r in self.rows)
               else {}),
            "best": ({"design": dict(best["design"]), "objective": best["objective"],
                      "constraints": dict(best["constraints"]), "feasible": True, "evaluation": best["n"],
                      "start": best["start"], "stage": best["stage"]} if best else None),
            "rounds": list(self.round_log),
            **({"run_not_used": {"count": len(self.run_not_used), "designs": list(self.run_not_used)}}
               if self.run_not_used else {}),
            **({"no_effect": dict(self.no_effect)} if self.no_effect else {}),
            **({"flat_objective": dict(self.flat_objective)} if self.flat_objective else {}),
            **({"stepped_objective": dict(self.stepped_objective)} if self.stepped_objective else {}),
        }


def search(block, *, seed, method=None, rounds=None, results=None, batch=False, stop_flat=True, stop_stepped=True,
           stream=False, first_reply=None):
    """The search as a generator of requests (see the module docstring); returns the outcome. ``rounds``: the rounds of
    a continued search, in order (each ``{"added": evaluations, "target": a value of the objective or None}``).
    ``results``: the names of the results the plan gives, watched for a design that changes none of them. ``batch``:
    designs the method chooses before it reads any of their answers (a starting population, a simplex's first corners, a
    shrunk simplex) are asked for together, ``{"kind": "evaluate_many", "requests": [...]}`` answered by a list in their
    order, so a driver can run them at the same time; each answer is then judged in that order exactly as one after
    another, so the search and its record are the same. ``stop_flat``: whether a search whose objective never moves stops
    early (``flat_objective``); off, the fact is recorded and the search runs its budget."""
    return (yield from _Search(block, seed, method, rounds, results, batch=batch, stop_flat=stop_flat,
                               stop_stepped=stop_stepped, stream=stream, first_reply=first_reply).run())


# --- a comparison across named alternatives: one search for each -----------------------------------------------------


def _combinations_in_the_study(block, axes):
    """The combinations of values of ``axes`` (``{axis: [values]}``) that settings of the study have, from the plan's
    ``sub_studies`` (each one varies some settings of the grid and holds the others at one value, a setting it neither
    varies nor holds at its first value), as a set of value tuples in the order of ``axes``; ``None`` when the plan has no
    sub-studies that read, so every combination is a setting of the study."""
    grid = block.get("grid") if isinstance(block.get("grid"), dict) else {}
    subs = block.get("sub_studies")
    if not isinstance(subs, list) or not subs or not all(isinstance(s, dict) for s in subs):
        return None
    if not getattr(study_grid({"grid": grid, "sub_studies": subs}), "sub_studies", None):
        return None  # sub-studies that do not read are not used by the scan either
    found = []
    for sub in subs:
        vary, hold = sub["vary"], sub.get("hold") or {}
        choices = []
        for axis in axes:
            values = list(grid[axis])
            choices.append(values if axis in vary else [hold[axis]] if axis in hold else values[:1])
        found.extend(itertools.product(*choices))
    return found


def compared_levels(block):
    """The alternatives a study compares while it also searches for the best design: the combinations of the names
    listed on the scan (``grid``) for a condition the plan holds fixed (``fixed``). When the plan has ``sub_studies`` the
    study has only the combinations its sub-studies have (an axis one sub-study varies is held by the others), as the
    coarse scan has; the plan's own held combination is always one. Each is a dict ``{condition: name}``;
    the first is the one the plan holds. ``[]`` when the study compares no named condition (a condition compared across
    numbers stays what it was: measured by the scan, held at one value by the search). FI searches each alternative on
    its own (:func:`compare`): the comparison is made between their best designs, not by searching only the first."""
    grid = block.get("grid") if isinstance(block.get("grid"), dict) else {}
    fixed = block.get("fixed") if isinstance(block.get("fixed"), dict) else {}
    axes = {}
    for axis, values in grid.items():
        names = [v for v in (values if isinstance(values, list) else [values])]
        if str(axis) in fixed and names and all(isinstance(v, str) and v.strip() for v in names):
            held = fixed[str(axis)] if isinstance(fixed[str(axis)], str) else None
            ordered = ([held] if held else []) + [v for v in dict.fromkeys(names) if v != held]
            if len(ordered) > 1:
                axes[str(axis)] = ordered
    if not axes:
        return []
    combos = list(itertools.product(*axes.values()))
    in_study = _combinations_in_the_study(block, {a: grid[a] for a in axes})
    if in_study is not None:
        combos = combos[:1] + [c for c in combos[1:] if c in in_study]
    return [dict(zip(axes, combo)) for combo in combos]


def level_count(block):
    """How many searches the plan's block runs: one for each alternative it compares, else one."""
    return max(1, len(compared_levels(block)))


def level_block(block, level):
    """``block`` with the conditions of one alternative held at its names."""
    fixed = dict(block.get("fixed") if isinstance(block.get("fixed"), dict) else {})
    fixed.update(level)
    return {**block, "fixed": fixed}


def level_key(level):
    """A short text naming an alternative (empty for no alternative), for keys of records."""
    return json.dumps(level, sort_keys=True) if level else ""


def level_says(level):
    return ", ".join(f"{k} = {v}" for k, v in level.items())


def compare(block, *, seed, method=None, rounds=None, results=None, batch=False, stop_flat=True, stop_stepped=True,
            stream=False):
    """:func:`search`, once for each alternative the plan compares (:func:`compared_levels`), one after another, each with
    the plan's own design variables, ranges, budget and seed; the comparison is made between the best design of each.
    Every request it yields says its alternative (``"level"``), and a scan request the alternative's block (``"block"``)
    so the scan, which measures every alternative together, is read for the one it starts. Without a compared condition it
    is exactly :func:`search`. The outcome is the one of :func:`search`, over every alternative: the best design is the best
    of the alternatives (it names its ``level``), the baseline is the one of that alternative, and ``levels`` has the best
    design, the baseline and the evaluations of each. An alternative after the first is first asked for a few designs the
    first one evaluated (:func:`_probe_alternative`); when it gives exactly the same results for every one, it is not
    searched (``same_as`` in its entry of ``levels``)."""
    levels = compared_levels(block)
    if not levels:
        return (yield from search(block, seed=seed, method=method, rounds=rounds, results=results, batch=batch,
                                  stop_flat=stop_flat, stop_stepped=stop_stepped, stream=stream))
    outcomes = []
    seen = {}  # the first alternative's answers, by design: what the probe of each later alternative is compared with
    names = compared_names(block, results)
    moving = {}  # the quantities that differ between two calls of the same design, once the first probe has shown them
    baseline_reply = None  # the answer a later alternative's first evaluation already got, handed to its search
    check_first = False  # an alternative before had no objective in any answer: each later one shows its own first
    for number, level in enumerate(levels):
        sub = level_block(block, level)
        baseline_reply = None
        if number > 0 and check_first:
            lone, baseline_reply = yield from _first_evaluation(sub, level, outcomes[0])
            if lone is not None:
                outcomes.append(lone)
                continue
        if number > 0:
            same = yield from _probe_alternative(sub, level, outcomes[0], seen, names, moving,
                                                 repeat=(level_block(block, levels[0]), levels[0]) if number == 1 else None)
            if same is not None:
                outcomes.append(_same_outcome(outcomes[0], levels[0], same))
                continue
        gen = search(sub, seed=seed, method=method, rounds=rounds, results=results, batch=batch, stop_flat=stop_flat,
                     stop_stepped=stop_stepped, stream=stream, first_reply=baseline_reply)
        asked = {}  # a streamed design, by its id
        try:
            request = next(gen)
            while True:
                request = dict(request, level=dict(level))
                if request["kind"] in ("evaluate_many", "evaluate_stream"):
                    request["requests"] = [dict(r, level=dict(level)) for r in request["requests"]]
                if request["kind"] == "evaluate_stream":
                    # A design id carries its alternative: a late answer of one alternative is never taken for the
                    # same design of the next.
                    request["requests"] = [dict(r, id=(number, r["id"])) for r in request["requests"]]
                if request["kind"] == "scan":
                    request["block"] = sub
                reply = yield request
                if request["kind"] == "evaluate_stream" and isinstance(reply, list):
                    reply = [(ident[1], answer) for ident, answer in reply if ident[0] == number]
                if number == 0:
                    _remember(seen, request, reply, asked, names)
                request = gen.send(reply)
        except StopIteration as finished:
            outcomes.append(finished.value)
        if outcomes[-1].get("every_failure"):
            # Every evaluation failed because the simulation's answer had no objective (the outcome's own record of why,
            # :data:`NO_OBJECTIVE`). Whether it has one can depend on the alternative, so each later one is first asked
            # for its baseline alone (:func:`_first_evaluation`): it is searched when that answer has the objective.
            check_first = True
    return merge_outcomes(outcomes, levels, block)


NOT_SEARCHED = "not_searched"
NOT_SEARCHED_BECAUSE = "its first evaluation also returned no objective"


def _first_evaluation(sub, level, first):
    """For an alternative after one whose every answer lacked the objective (a generator): its baseline, evaluated alone.
    When the answer also lacks the objective the alternative is not searched, and its outcome (``not_searched``) holds
    that one evaluation; otherwise ``None``, and the alternative is searched as any other. Returns ``(outcome, reply)``:
    the reply is then the search's own first evaluation (``first_reply``), not asked for again."""
    design = Space(sub["design_variables"]).canonical(sub["baseline"]["values"])
    reply = yield {"kind": "evaluate", "cell": cell_of(sub, design), "design": dict(design), "stage": "baseline",
                   "start": 0, "level": dict(level)}
    reply = reply if isinstance(reply, dict) else {"failed": "no answer for this design"}
    judged = judge(sub, reply.get("values") if "values" in reply else None, str(reply.get("failed") or ""))
    if judged.get("kind") != NO_OBJECTIVE:
        return None, reply
    row = {"event": "evaluation", "n": 1, "stage": "baseline", "start": 0, "design": dict(design), "objective": None,
           "constraints": {}, "feasible": False, "status": "failed", "method": first["method_used"], "counted": True,
           "elapsed_s": reply.get("elapsed_s") if _number(reply.get("elapsed_s")) else None,
           "reason": judged["reason"], "failure": NO_OBJECTIVE}
    return ({
        "rows": [row], "evaluations": 1, "scan_evaluations": 0, "budget": 1, "starts": first["starts"],
        "per_start": first["per_start"], "stopped_because": NOT_SEARCHED, "not_searched": NOT_SEARCHED_BECAUSE,
        "budget_ran_out": False, "method_requested": first["method_requested"], "method_used": first["method_used"],
        "method_why": first["method_why"], "notes": [],
        "baseline": {"design": dict(design), "status": "failed", "objective": None, "constraints": {},
                     "feasible": False, "reason": judged["reason"]},
        "start_log": [], "seed": first["seed"], "failed": 1, "infeasible": 0, "best": None, "rounds": [],
    }, reply)


PROBE_DESIGNS = 8  # the designs an alternative is asked for before it is searched (see :func:`_probe_alternative`)


def _design_key(design):
    return json.dumps(design, sort_keys=True, default=str)


def compared_names(block, results):
    """The results two alternatives are compared on: the objective, every limited quantity and every result the plan names."""
    names = {str(block["objective"]["quantity"])}
    names |= {str(c.get("quantity")) for c in block.get("constraints") or [] if isinstance(c, dict)}
    return sorted(names | {str(r) for r in results or []})


def varying_names(block, names, wanted, again):
    """The quantities of ``names`` that are not the same in two calls of one design (``wanted``, ``again``: the values of
    each): a result the simulation reports that changes by itself from call to call (how long it took) says nothing about the
    alternative it was asked at, so it is left out of the comparison. The objective and every limited quantity are never left
    out: when one of those varies, the alternatives cannot be told apart and each is searched."""
    kept = {str(block["objective"]["quantity"])} | {str(c.get("quantity")) for c in block.get("constraints") or []
                                                      if isinstance(c, dict)}
    return {n for n in names if n not in kept and ((n in wanted) != (n in again) or (n in wanted and wanted[n] != again[n]))}


def probe_answer(sub, values, names):
    """What an alternative gave for one design, for the comparison: the value of each of ``names`` it reported, or ``None``
    when the design was not judged ``ok`` (a failed or missing answer cannot show two alternatives are the same)."""
    if not isinstance(values, dict) or judge(sub, values)["status"] != "ok":
        return None
    return {n: values[n] for n in names if n in values}


def _remember(seen, request, reply, asked, names):
    """Keep, for each design the first alternative's search evaluated, what the simulation gave for ``names``."""
    kind = request["kind"]
    if kind == "evaluate":
        pairs = [(request.get("design"), reply)]
    elif kind == "evaluate_many" and isinstance(reply, list):
        pairs = [(r.get("design"), a) for r, a in zip(request["requests"], reply)]
    elif kind == "evaluate_stream":
        for r in request["requests"]:
            asked[r["id"][1]] = r.get("design")
        pairs = [(asked.get(i), a) for i, a in reply] if isinstance(reply, list) else []
    else:
        return
    for design, answer in pairs:
        if isinstance(design, dict) and isinstance(answer, dict) and isinstance(answer.get("values"), dict):
            seen[_design_key(design)] = {n: answer["values"][n] for n in names if n in answer["values"]}


def _probe_designs(first, seen):
    """Up to :data:`PROBE_DESIGNS` designs the first alternative evaluated and answered: its baseline, then ones spread
    over its run (the first, the best, and evenly spaced ones). Deterministic."""
    chosen, keys = [], set()
    base = first["baseline"]
    if base.get("status") == "ok" and _design_key(base["design"]) in seen:
        chosen.append(dict(base["design"]))
        keys.add(_design_key(base["design"]))
    pool = []
    for row in first["rows"]:
        key = _design_key(row["design"])
        if row["counted"] and row["status"] == "ok" and key in seen and key not in keys:
            keys.add(key)
            pool.append(dict(row["design"]))
    room = PROBE_DESIGNS - len(chosen)
    n = len(pool)
    if n <= room:
        return chosen + pool
    order = []
    if first["best"]:
        order.append(next((i for i, d in enumerate(pool) if _design_key(d) == _design_key(first["best"]["design"])), 0))
    order += [round(i * (n - 1) / (room - 2)) for i in range(room - 1)]  # from the first design to the last
    order = list(dict.fromkeys(order))
    order += [i for i in range(n) if i not in order][:room - len(order)]  # where two of them fell on the same design
    return chosen + [pool[i] for i in order[:room]]


def sizing_designs(block):
    """The designs a size estimate asks of every alternative before any search has run (so the search's own designs are
    not known yet): the baseline and :data:`PROBE_DESIGNS` - 1 designs spread over the space by a fixed rule, the same on
    every run."""
    space = Space(block["design_variables"])
    out = [dict(block["baseline"]["values"])]
    for i in range(PROBE_DESIGNS - 1):
        u = [(0.5 + 0.618034 * (i + 1) + 0.37 * k) % 1.0 for k in range(space.dims)]
        out.append(space.design(u))
    return out


def alternatives_searched(block, levels, replies, repeat, names):
    """How many of ``levels`` the run would search, from the values each gave for the designs of :func:`sizing_designs`
    (``replies``: a list, one entry for each alternative, of the values for each design, ``None`` for one that did not
    answer) and ``repeat`` (the first alternative's first design asked again): by the rule :func:`compare` applies, an
    alternative that gives exactly the first one's values for every design (of the quantities that do not vary between
    calls) is not searched. The first is always searched, and so is an alternative that did not answer every design."""
    first = replies[0]
    if not first or any(v is None for v in first):
        return len(levels)
    stable = [n for n in names if n not in varying_names(block, names, first[0], repeat or {})] if repeat else list(names)
    mine = [probe_answer(level_block(block, levels[0]), v, stable) for v in first]
    if any(m is None for m in mine):
        return len(levels)
    searched = 1
    for number in range(1, len(levels)):
        theirs = replies[number]
        got = ([probe_answer(level_block(block, levels[number]), v, stable) for v in theirs]
               if theirs and all(v is not None for v in theirs) else None)
        if got is None or got != mine:
            searched += 1
    return searched


def _probe_alternative(sub, level, first, seen, names, moving=None, repeat=None):
    """Before an alternative after the first is searched (a generator): ask the simulation for a few designs the first
    alternative evaluated, at this alternative (one ``evaluate_many`` request, answered in order). When every one gives
    exactly the first alternative's value of the objective, every limited quantity and every result the plan names (never
    a timing), the alternative changes nothing the study reads and a search of it would find the same numbers again: the
    rows of the probe are returned and the alternative is not searched. When any differs, or the simulation did not
    answer, ``None``: the alternative is searched as usual, and these evaluations stay outside its budget and its rows
    (a repeated search asks them again). Exact equality: a simulation whose answers vary between calls fails the test and
    is searched, except for a result that varies by itself from call to call: ``repeat`` (``(the first alternative's
    block, its level)``, given with the first probe) asks the first alternative's first probe design once more, outside
    the rows, and a named result that is not the same in the two calls is left out of every comparison (kept in
    ``moving["names"]``; :func:`varying_names`)."""
    probes = _probe_designs(first, seen)
    if len(probes) < 2:
        return None
    requests = [{"kind": "evaluate", "cell": cell_of(sub, d), "design": dict(d), "stage": "probe", "start": 0,
                 "level": dict(level)} for d in probes]
    replies = yield {"kind": "evaluate_many", "requests": requests, "level": dict(level)}
    if repeat is not None:
        again_block, again_level = repeat
        again = yield {"kind": "evaluate", "cell": cell_of(again_block, probes[0]), "design": dict(probes[0]),
                       "stage": "probe", "start": 0, "level": dict(again_level), "repeat": True}
        values = again.get("values") if isinstance(again, dict) else None
        if moving is not None:
            wanted = seen[_design_key(probes[0])]
            moving["names"] = (varying_names(sub, names, wanted, {n: values[n] for n in names if n in values})
                               if isinstance(values, dict) else set())
    used = [n for n in names if n not in (moving or {}).get("names", ())]
    if not isinstance(replies, list) or len(replies) != len(probes):
        return None
    rows = []
    for design, reply in zip(probes, replies):
        if not isinstance(reply, dict) or not isinstance(reply.get("values"), dict):
            return None
        values = reply["values"]
        wanted = seen[_design_key(design)]
        if any((n in wanted) != (n in values) or (n in wanted and values[n] != wanted[n]) for n in used):
            return None
        judged = judge(sub, values)
        if judged["status"] != "ok":
            return None
        rows.append({"event": "evaluation", "n": len(rows) + 1, "stage": "probe", "start": 0, "design": dict(design),
                     "objective": judged["objective"], "constraints": dict(judged["constraints"]),
                     "feasible": judged["feasible"], "status": "ok", "method": first["method_used"], "counted": True,
                     "elapsed_s": reply.get("elapsed_s") if _number(reply.get("elapsed_s")) else None})
        named = {n: values[n] for n in names if n in values and n != str(sub["objective"]["quantity"])
                 and n not in judged["constraints"]}
        if named:
            rows[-1]["results"] = named
    return rows


def _same_outcome(first, first_level, rows):
    """The outcome of an alternative that was not searched because it gave the first one's answers (``same_as``): the
    probe's rows as its own evaluations, the first alternative's best design and baseline. Its budget is the probe."""
    out = {
        "rows": rows, "evaluations": len(rows), "scan_evaluations": 0, "budget": len(rows),
        "starts": first["starts"], "per_start": first["per_start"], "stopped_because": "same_as_first",
        "budget_ran_out": False, "method_requested": first["method_requested"], "method_used": first["method_used"],
        "method_why": first["method_why"], "notes": [], "baseline": first["baseline"], "start_log": [],
        "seed": first["seed"], "failed": 0, "infeasible": sum(1 for r in rows if not r["feasible"]),
        "best": dict(first["best"]) if first["best"] else None, "rounds": [],
        "same_as": dict(first_level), "probes": len(rows),
    }
    for key in ("no_effect", "flat_objective", "stepped_objective"):
        if first.get(key):
            out[key] = dict(first[key])
    return out


def merge_outcomes(outcomes, levels, block):
    """The outcome of a search over every alternative (see :func:`compare`)."""
    sign = _sign(block)
    rows = []
    for outcome, level in zip(outcomes, levels):
        for row in outcome["rows"]:
            rows.append(dict(row, level=dict(level), n=len(rows) + 1))
    offset = 0
    per = []
    for outcome, level in zip(outcomes, levels):
        best = outcome["best"]
        # An alternative that was not searched has the first one's best design, numbered as in the first one's rows.
        shift = 0 if outcome.get("same_as") else offset
        per.append({"level": dict(level), "best": dict(best, evaluation=best["evaluation"] + shift) if best else None,
                    "baseline": outcome["baseline"], "evaluations": outcome["evaluations"],
                    "budget": outcome["budget"], "stopped_because": outcome["stopped_because"],
                    **({"status": NOT_SEARCHED, "not_searched_because": outcome["not_searched"]}
                       if outcome.get("not_searched") else {}),
                    **({"same_as": dict(outcome["same_as"]), "probes": outcome["probes"]}
                       if outcome.get("same_as") else {})})
        offset += len(outcome["rows"])
    winners = [p for p in per if p["best"] is not None]
    win = min(winners, key=lambda p: sign * p["best"]["objective"]) if winners else per[0]
    best = dict(win["best"], level=dict(win["level"])) if win["best"] else None
    # The search's own end is that of the alternatives that were searched and gave results: one that failed every
    # evaluation, or was not searched, is reported by its own entry (``levels``) and does not name the search's end.
    stops = [o["stopped_because"] for o in outcomes if o["stopped_because"] not in ("failures", NOT_SEARCHED)]
    stops = stops or [o["stopped_because"] for o in outcomes]
    stopped = next((s for s in ("time", "the simulation could not be loaded", "failures", "scan_unchanged") if s in stops),
                   "budget" if "budget" in stops else stops[0])
    first = outcomes[0]
    out = {
        "rows": rows, "evaluations": sum(o["evaluations"] for o in outcomes),
        "scan_evaluations": sum(o["scan_evaluations"] for o in outcomes), "budget": sum(o["budget"] for o in outcomes),
        "starts": first["starts"], "per_start": first["per_start"], "levels": per, "level_count": len(outcomes),
        "stopped_because": stopped, "budget_ran_out": any(o["budget_ran_out"] for o in outcomes),
        "method_requested": first["method_requested"], "method_used": first["method_used"],
        "method_why": first["method_why"],
        "notes": list(dict.fromkeys(n for o in outcomes for n in o["notes"])),
        "baseline": win["baseline"],
        "start_log": [dict(s, level=dict(lv)) for o, lv in zip(outcomes, levels) for s in o["start_log"]],
        "seed": first["seed"], "failed": sum(o["failed"] for o in outcomes),
        "infeasible": sum(o["infeasible"] for o in outcomes), "best": best,
        "rounds": [dict(r, level=dict(lv)) for o, lv in zip(outcomes, levels) for r in o["rounds"]],
        "fixed": dict(level_block(block, win["level"])["fixed"]),
    }
    unused = [dict(d, level=dict(lv)) for o, lv in zip(outcomes, levels)
              for d in (o.get("run_not_used") or {}).get("designs", [])]
    if unused:
        out["run_not_used"] = {"count": len(unused), "designs": unused}
    # A stop that is a fact about the simulation (it does not move the objective) is the search's only when every
    # alternative's search found it: one alternative that did move it shows the simulation reaches the objective.
    for key in ("no_effect", "flat_objective", "stepped_objective"):
        if all(o.get(key) for o in outcomes):
            out[key] = dict(first[key])
    return out


def not_searched_says(entry):
    """The plain sentence for an alternative that was not searched (``status`` ``not_searched``) and why."""
    return f"{level_says(entry['level'])} was not searched: {entry.get('not_searched_because') or NOT_SEARCHED_BECAUSE}"


def same_as_says(entry):
    """The plain sentence for an alternative that was not searched because it gave the results of another (no final stop)."""
    n = int(entry.get("probes") or 0)
    return (f"{level_says(entry['level'])} gave the same results as {level_says(entry['same_as'])} for every one of the "
            f"{n} designs tried at both, so the search was not run again for it")


def rehearse(block, *, batch, seed=0):
    """How many designs the search asks for together, request by request after its coarse scan (a list of counts), from
    the search itself run on a stand-in objective with the same ``batch`` as the run: how many designs a method asks for at
    a time (its starting points side by side, a population, a simplex's corners) is how it is written, not what the
    designs give. The stand-in gives each design a fixed objective from its key, so it never looks flat, and every limited
    quantity at its limit, so no design fails; ``None`` for a library method (it runs in the quest's own Python, one step
    at a time) or a block the search cannot run."""
    import hashlib

    if str(block.get("search_method") or "").startswith(("scipy:", "optuna:")):
        return None
    quantity = str((block.get("objective") or {}).get("quantity") or "objective")
    limits = {str(c.get("quantity")): (_parse_limit(c.get("limit")) or ("<=", 0.0))[1]
              for c in block.get("constraints") or [] if isinstance(c, dict)}

    def value(design):
        key = json.dumps(design, sort_keys=True, default=str).encode()
        return {**limits, quantity: int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2 ** 64}

    widths = []
    try:
        gen = search(block, seed=seed, batch=batch, stop_flat=False, stop_stepped=False)
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                rows = [{"design": cell, "values": value(cell)} for cell in grid_cells(request["grid"])]
                request = gen.send(rows)
            elif request["kind"] == "evaluate_many":
                widths.append(len(request["requests"]))
                request = gen.send([{"values": value(r["design"])} for r in request["requests"]])
            elif request["kind"] == "evaluate":
                widths.append(1)
                request = gen.send({"values": value(request["design"])})
            else:
                return None
    except StopIteration:
        return widths
    except (KeyError, TypeError, ValueError, TooManySettings):
        return None


#: The most settings FI lists (or runs) from one grid: each listed setting is a dict, about a kilobyte with its key, so this
#: many is about half a gigabyte; a plan that asks for more is stopped with its count, never cut and never listed.
MAX_LISTED_SETTINGS = 500_000


class TooManySettings(ValueError):
    """A grid has more settings than FI lists. The message is what a person reads."""


class StudyGrid(dict):
    """A protocol's ``grid`` (each setting with every value it takes) together with its ``sub_studies``: separate parts of
    the study that each vary only some of the settings, the others held at one value. Everywhere but in counting and
    listing the settings it is the plain dict of axes, so every reader of the grid keeps working. ``sub_studies``: a list of
    ``{"name", "vary": [axes], "hold": {axis: value}}`` (a setting a sub-study neither varies nor holds is held at its first
    value). ``everywhere``: settings every setting of the study is crossed with (what each run compares inside it)."""

    def __init__(self, grid, sub_studies=(), everywhere=()):
        super().__init__(grid)
        self.sub_studies = [dict(s) for s in sub_studies]
        self.everywhere = tuple(everywhere)


def _value_index(values, wanted):
    for i, value in enumerate(values):
        if value == wanted:
            return i
    return None


def study_grid(protocol):
    """The ``grid`` of a protocol (``{}`` when it has none). With ``sub_studies`` that read (each one varying settings of the
    grid, each held value one of its values) it is a :class:`StudyGrid`, so counting and listing the settings give the union
    of the sub-studies' settings; sub-studies that do not read are left out (the plan check names what is wrong with them)."""
    protocol = protocol if isinstance(protocol, dict) else {}
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    subs = protocol.get("sub_studies")
    if not grid or not isinstance(subs, list) or not subs:
        return grid
    found = []
    for sub in subs:
        vary, hold = (sub.get("vary"), sub.get("hold") or {}) if isinstance(sub, dict) else (None, None)
        if (not isinstance(vary, list) or not vary or not all(isinstance(a, str) and a in grid for a in vary)
                or not isinstance(hold, dict)
                or not all(isinstance(a, str) and a in grid and isinstance(grid[a], (list, tuple))
                           and _value_index(grid[a], v) is not None for a, v in hold.items())):
            return grid
        found.append({"name": str(sub.get("name") or ""), "vary": list(vary), "hold": dict(hold)})
    return StudyGrid(grid, found)


def with_everywhere(grid, inside):
    """``grid`` with the settings ``inside`` (a dict) added, each crossed with every setting of the study; a grid with
    sub-studies stays one."""
    merged = {**grid, **inside}
    if not getattr(grid, "sub_studies", None):
        return merged
    return StudyGrid(merged, grid.sub_studies, tuple(getattr(grid, "everywhere", ())) + tuple(inside))


def with_axes(grid, axes):
    """``axes`` (a dict) as a grid of the same kind as ``grid``: with its sub-studies when it has them."""
    if not getattr(grid, "sub_studies", None):
        return axes
    return StudyGrid(axes, grid.sub_studies, getattr(grid, "everywhere", ()))


def _axis_choices(grid, sub):
    """For one sub-study, the indexes of the values each axis of ``grid`` takes in it (all of them, or the one it is held at)."""
    out = []
    for name, values in grid.items():
        if name in sub["vary"] or name in grid.everywhere:
            out.append(list(range(len(values))))
        else:
            held = _value_index(values, sub["hold"][name]) if name in sub["hold"] else 0
            out.append([held or 0])
    return out


#: Counting the settings that two or more sub-studies share by inclusion-exclusion takes 2^n steps, so it is done for up to this
#: many sub-studies; beyond it :func:`grid_size` lists the settings to count them when the sum of the parts is under the limit
#: FI lists, and otherwise gives that sum (an upper bound: the study is stopped for its size either way).
MAX_EXACT_SUB_STUDIES = 12


def grid_weight(grid, weights=None):
    """The sum, over the settings of a grid, of the product of one weight per axis (``weights``: ``{axis: [a weight for
    each of its values]}``; an axis without one weighs 1 per value, so the sum is the number of settings). For a grid
    with sub-studies it is the sum over the settings in at least one of them (one two share is counted once), worked out
    without listing any: the weight of the settings in all of a set of sub-studies is the product, axis by axis, of the
    weights of the values they all have, and a setting in several sub-studies is taken out of the sum as often as it was
    added. A time estimate that gives each value a cost ratio sums them the same way."""
    names = list(grid)
    each = [list(weights.get(name) or [1] * len(grid[name])) if weights else [1] * len(grid[name]) for name in names]
    subs = getattr(grid, "sub_studies", None)
    if not subs:
        return math.prod(sum(w) for w in each)
    choices = [_axis_choices(grid, sub) for sub in subs]
    if len(subs) > MAX_EXACT_SUB_STUDIES:
        return sum(math.prod(sum(each[k][i] for i in c) for k, c in enumerate(one)) for one in choices)
    total = 0
    for bits in range(1, 1 << len(subs)):
        chosen = [choices[i] for i in range(len(subs)) if bits >> i & 1]
        shared = 1
        for k in range(len(names)):
            shared *= sum(each[k][i] for i in set.intersection(*(set(one[k]) for one in chosen)))
            if not shared:
                break
        total += shared if len(chosen) % 2 else -shared
    return total


def grid_size(grid):
    """How many settings a grid names, counted without listing any: the product of its axes' lengths; for a grid with
    sub-studies, the settings in at least one of them (a setting two of them share is counted once). Always the length of
    :func:`grid_cells`, except above :data:`MAX_EXACT_SUB_STUDIES` sub-studies and more settings than FI lists, where it is the
    sum of the sub-studies' (at least the count, over the limit either way)."""
    subs = getattr(grid, "sub_studies", None)
    total = grid_weight(grid)
    if subs and len(subs) > MAX_EXACT_SUB_STUDIES and total <= MAX_LISTED_SETTINGS:
        return sum(1 for _ in iter_grid_indexes(grid))
    return total


def sub_study_sizes(grid):
    """``[(name, settings)]`` of the sub-studies of a :class:`StudyGrid` (a setting two share is in both), else ``[]``."""
    return [(sub.get("name") or f"sub-study {i}", math.prod(len(c) for c in _axis_choices(grid, sub)))
            for i, sub in enumerate(getattr(grid, "sub_studies", None) or [], start=1)]


def part_cells(grid):
    """``[(name, setting)]``: for each sub-study of a :class:`StudyGrid`, one real setting of it (every setting it varies at its
    middle value, every other at the value it holds), else ``[]``. A part can cost far more than another (what a run of it
    computes differs), so timing a setting of every part is how its cost is known."""
    out = []
    for i, sub in enumerate(getattr(grid, "sub_studies", None) or [], start=1):
        picks = [c[len(c) // 2] for c in _axis_choices(grid, sub)]
        out.append((sub.get("name") or f"sub-study {i}", {name: values[k] for (name, values), k in zip(grid.items(), picks)}))
    return out


def costly_part_cells(grid):
    """``[(name, setting)]``: for each sub-study of a :class:`StudyGrid`, the setting of it that looks the costliest, else ``[]``:
    every setting it varies at its largest number (a count, a size, a number of steps is what a run's work grows with; a
    setting whose values are not numbers stays at its middle value), every other at the value it holds. The middle setting
    of :func:`part_cells` can be far cheaper than the part's biggest one (a sub-study whose settings load many more inputs
    at its large values was timed only at its cheap middle), so a part is timed at both and costed at the dearer."""
    out = []
    for i, sub in enumerate(getattr(grid, "sub_studies", None) or [], start=1):
        picks = []
        for (name, values), choices in zip(grid.items(), _axis_choices(grid, sub)):
            numbers = [(float(values[k]), k) for k in choices
                       if isinstance(values[k], (int, float)) and not isinstance(values[k], bool)]
            picks.append(max(numbers)[1] if len(choices) > 1 and len(numbers) == len(choices) else choices[len(choices) // 2])
        out.append((sub.get("name") or f"sub-study {i}", {name: values[k] for (name, values), k in zip(grid.items(), picks)}))
    return out


def check_listable(grid):
    """Raise :class:`TooManySettings` when listing the whole grid would hold more than :data:`MAX_LISTED_SETTINGS`."""
    size = grid_size(grid)
    if size > MAX_LISTED_SETTINGS:
        raise TooManySettings(
            f"the plan asks for {size:,} settings (every combination of the settings' values); FI does not "
            f"list or run more than {MAX_LISTED_SETTINGS:,} settings. Fewer settings or fewer values for each must be "
            "planned; FI does not choose which to leave out.")


def grid_too_large(grid):
    """``""`` when a protocol's ``grid`` is a size FI lists, else one plain sentence: the count (the same
    :func:`grid_size` that :func:`check_listable` stops on, which the trial runner, the run manifest and the script's own
    list all hold to), each setting with its number of values (the most first), and FI's limit. Only ``grid`` multiplies
    the settings: what every run compares inside it (``in_each_run``) is not in it, and ``constants`` have one value. With
    ``sub_studies`` only the settings of one sub-study multiply: the sentence gives each sub-study's count."""
    if not isinstance(grid, dict) or not grid:
        return ""
    listed = {str(k): (v if isinstance(v, (list, tuple)) else [v]) for k, v in grid.items()}
    if getattr(grid, "sub_studies", None):
        listed = StudyGrid(listed, grid.sub_studies, grid.everywhere)
    counts = {k: len(v) for k, v in listed.items()}
    size = grid_size(listed)
    if size <= MAX_LISTED_SETTINGS:
        return ""
    if getattr(listed, "sub_studies", None):
        parts = ", ".join(f"`{name}` {n:,}" for name, n in sorted(sub_study_sizes(listed), key=lambda kv: -kv[1]))
        return (f"the settings in `protocol.grid` and `protocol.sub_studies` are {size:,}, and FI does not list or run more "
                f"than {MAX_LISTED_SETTINGS:,} (the settings of each sub-study, every combination of the values of the "
                f"settings it varies): {parts}")
    axes = ", ".join(f"`{k}` {n} value{'s' if n != 1 else ''}"
                     for k, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n > 1)
    # The limit before the list: a console line cut at its length still says it.
    return (f"the settings in `protocol.grid` are {size:,}, and FI does not list or run more than {MAX_LISTED_SETTINGS:,} "
            f"(every combination of the values of its {len(counts)} settings, each multiplying every other): {axes}")


def iter_grid_indexes(grid):
    """The settings of :func:`grid_cells` one at a time, each as the index of its value on every axis (axes in order),
    without the size check (a caller that takes only the first few)."""
    subs = getattr(grid, "sub_studies", None)
    if not subs:
        yield from itertools.product(*(range(len(v)) for v in grid.values()))
        return
    seen = set()
    for sub in subs:
        for picked in itertools.product(*_axis_choices(grid, sub)):
            if picked not in seen:
                seen.add(picked)
                yield picked


def iter_grid_cells(grid):
    """The settings of :func:`grid_cells` one at a time, without the size check."""
    axes = list(grid.items())
    for picked in iter_grid_indexes(grid):
        yield {a: v[i] for (a, v), i in zip(axes, picked)}


def first_grid_cells(grid, n):
    """The first ``n`` settings of :func:`grid_cells` (the plan's order), listed lazily: a grid of more settings than FI lists
    is never built whole."""
    return list(itertools.islice(iter_grid_cells(grid), max(1, int(n))))


def grid_cells(grid):
    """Every setting of a grid, as the trial runner makes them: every combination of the values (axes in order), or, for a
    grid with sub-studies, the settings of each sub-study in turn without the ones an earlier sub-study already has."""
    check_listable(grid)
    return list(iter_grid_cells(grid))


def run_sync(block, evaluate, *, seed, method=None, scan_step=None, drive_step=None, deadline=None,
             max_evaluations=None, rounds=None, results=None):
    """Run the search here, calling ``evaluate(cell) -> dict`` for each design (an exception is a failed evaluation),
    ``scan_step(grid) -> rows`` for the coarse scan (default: ``evaluate`` on each of its cells) and
    ``drive_step(spec)`` (default: :func:`drive` in this process) for a library's method. ``deadline`` is a
    ``time.monotonic()`` time after which no evaluation starts; ``max_evaluations`` stops the search where an earlier
    one was stopped by its time limit (so ``code/run.py`` repeats that search, not a longer one)."""
    gen = compare(block, seed=seed, method=method, rounds=rounds, results=results)
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
                    # An alternative's scan is read for its own block (the scan measures every alternative together).
                    reply = (scan_step(request["grid"], block=request["block"]) if "block" in request
                             else scan_step(request["grid"]))
                else:
                    reply = []
                    for cell in grid_cells(request["grid"]):
                        answer = one(cell)
                        reply.append({"design": {n: cell[n] for n in names}, "values": answer.get("values"),
                                      "reason": answer.get("failed", ""), "elapsed_s": answer["elapsed_s"]})
            elif request["kind"] in ("evaluate", "evaluate_many"):
                # ``evaluate_many`` (the probe of an alternative, see :func:`compare`): one after another here.
                replies = []
                for one_request in request["requests"] if request["kind"] == "evaluate_many" else [request]:
                    if ((deadline is not None and time.monotonic() >= deadline)
                            or (max_evaluations is not None and done >= max_evaluations)):
                        replies.append({"stop": "time"})
                    else:
                        replies.append(one(one_request["cell"]))
                        done += 1
                reply = replies if request["kind"] == "evaluate_many" else replies[0]
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
                   "budget_ran_out": outcome["budget_ran_out"], "stopped_because": outcome["stopped_because"],
                   **({"run_not_used": outcome["run_not_used"]["count"]} if outcome.get("run_not_used") else {}),
                   **({"no_effect": outcome["no_effect"]} if outcome.get("no_effect") else {}),
                   **({"flat_objective": outcome["flat_objective"]} if outcome.get("flat_objective") else {}),
                   **({"stepped_objective": outcome["stepped_objective"]} if outcome.get("stepped_objective") else {})}
    u = f" {unit}" if unit else ""
    if best is None and not outcome["rows"] and outcome["stopped_because"] == "time":
        says = "No design was evaluated before the time limit (execution.timeout_s) ran out."
    elif best is None:
        says = (f"No design that meets every limit was found in {outcome['evaluations']} evaluation(s) "
                f"(of {outcome['budget']} allowed).")
        for p in outcome.get("levels") or []:
            if p.get("status") == NOT_SEARCHED:
                says += " " + not_searched_says(p) + "."
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
        if outcome.get("levels"):
            shown = "; ".join(level_says(p["level"]) + (": " + _fmt(p["best"]["objective"]) + u if p["best"]
                                                        else ": no design that meets every limit")
                              for p in outcome["levels"])
            same = [p for p in outcome["levels"] if p.get("same_as")]
            skipped = [p for p in outcome["levels"] if p.get("status") == NOT_SEARCHED]
            says += (f" It is the best of {len(outcome['levels'])} alternatives"
                     f"{', each searched on its own' if not same and not skipped else ''} ({shown}); the winner is "
                     f"{level_says(best['level'])}.")
            for p in same:
                says += " " + same_as_says(p) + "."
            for p in skipped:
                says += " " + not_searched_says(p) + "."
    rounds = [r for r in outcome.get("rounds") or [] if isinstance(r, dict)]
    if rounds:
        by_fi = sum(1 for r in rounds if r.get("by") == "FI")
        who = ("at the person's request" if not by_fi else
               "by FI, because it was still improving when its budget ran out" if by_fi == len(rounds) else
               f"({len(rounds) - by_fi} at the person's request, {by_fi} by FI because it was still improving when its "
               "budget ran out)")
        says += (f" The search was continued {len(rounds)} time(s) {who} "
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
        "fixed": dict(outcome.get("fixed") or block.get("fixed") or {}),
        "search_settings": search_settings(block),
        "check_settings": dict(check) if check is not None else check_settings(block),
        "checked_at_finer_settings": False,
        "notes": outcome["notes"],
        "says": says,
    }
    if outcome.get("levels"):
        record["compared"] = {"conditions": sorted({k for p in outcome["levels"] for k in p["level"]}),
                              "levels": outcome["levels"]}
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
        record["evaluations"]["planned_budget"] = outcome["starts"] * outcome["per_start"] * outcome.get("level_count", 1)
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
    if outcome.get("levels"):
        head["compared"] = [p["level"] for p in outcome["levels"]]
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
