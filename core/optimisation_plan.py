"""The plan of a study that looks for the best design, rather than measuring over settings chosen in advance.

A quest's design says what kind of study it is in ``study_type``:

* ``measure`` (the default): how a result changes, or which of fixed alternatives does better, over settings the plan
  chooses. The protocol has a ``grid``.
* ``find_best_design``: the design (the fin spacing and thickness, the truss cross-section, the controller gain) that
  makes one result as low or as high as possible, within limits. The protocol has an ``optimisation`` block instead of a
  ``grid``: what to minimise or maximise (``objective``), what may change and over what range (``design_variables``),
  what is held fixed (``fixed``), the limits the design must meet (``constraints``), the design to beat (``baseline``),
  the numerical settings the search uses and the finer ones the best design is checked at (``numerical_settings``), how
  many evaluations the search may spend (``evaluation_budget``), the search method (``search_method``), an optional coarse
  scan run first (``grid``), an optional threshold for "better" (``improvement_tolerance``) and an optional ``target``.

This module reads and checks that block, as :mod:`core.metric_spec` does for ``protocol.metrics``: lenient about form,
strict about the numbers. Nothing it returns adds a key the plan did not write (a block read back must hash as it was
written; the frozen protocol's hash depends on it): what FI does when a part is left out (the finer check settings, the
method, the threshold) is computed by the helpers below at the time it is used, and said in ``plan.md``.

It also decides, from the topic's words alone, whether the topic is clear about the kind of study; when it is not, the
clarify step asks the person one plain question (:func:`add_study_type_question`).

The search itself is not run here: the engine runs it (:mod:`core.optimise`). A plan of that kind never runs as a plain
sweep; one the engine cannot search (no block, no budget, one script, a cluster) stops before anything runs
(``core/engine.py::_stop_if_the_search_cannot_start``).
"""

from __future__ import annotations

import math
import re
from typing import Any

STUDY_TYPES = {
    "measure": "measure how a result changes over settings chosen in advance",
    "find_best_design": "find the best design",
}
_STUDY_TYPE_WORDS = {
    "measure": "measure", "measurement": "measure", "sweep": "measure", "scan": "measure",
    "find_best_design": "find_best_design", "find best design": "find_best_design", "find the best design": "find_best_design",
    "best_design": "find_best_design", "best design": "find_best_design", "optimisation": "find_best_design",
    "optimization": "find_best_design", "optimise": "find_best_design", "optimize": "find_best_design",
}

DIRECTIONS = {"minimise": "minimise", "minimize": "minimise", "min": "minimise", "minimum": "minimise",
              "maximise": "maximise", "maximize": "maximise", "max": "maximise", "maximum": "maximise"}
KINDS = ("continuous", "integer", "choice")
BUILT_IN_METHODS = {
    "bounded_local": "a local search inside the ranges, from several starting points",
    "global_then_local": "a search over the whole range first, then a local search from the best point found",
    "exhaustive": "every combination is evaluated (only when the variables take few values)",
}
# A method from a library the quest's environment may have: ``scipy:<function>`` or ``optuna:<sampler>``.
_LIBRARY_METHOD = re.compile(r"^(scipy|optuna):([A-Za-z][A-Za-z0-9_.-]*)$")
_LIMIT = re.compile(r"^\s*(<=|>=|<|>|≤|≥|=<|=>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
_OPS = {"<=": "<=", "<": "<=", "≤": "<=", "=<": "<=", ">=": ">=", ">": ">=", "≥": ">=", "=>": ">="}
# Each level of the finer check must be at least this much finer than the one before it: a ratio near 1 makes the
# numerical error look small only because the two levels are almost the same (Roache's grid-convergence rule).
MIN_REFINEMENT = 1.1
# How many of the best designs the check recomputes at the finer settings, besides the baseline.
CHECK_CANDIDATES = 3
_P = "`protocol.optimisation"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _whole(value: Any) -> bool:
    return _number(value) and float(value).is_integer()


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ""


# --- the kind of study ----------------------------------------------------------------------------------------------


def normalize_study_type(value: Any) -> tuple[str | None, str | None]:
    """``(canonical, None)`` for a ``study_type`` a plan wrote, else ``(None, why)``."""
    key = " ".join(str(value).strip().lower().replace("-", " ").split()) if isinstance(value, str) else ""
    canonical = _STUDY_TYPE_WORDS.get(key) or _STUDY_TYPE_WORDS.get(key.replace(" ", "_"))
    if canonical is None:
        return None, "`study_type` must be `measure` (how a result changes over chosen settings) or `find_best_design`"
    return canonical, None


def repair_study_type(design: Any) -> list[str]:
    """Put a model's draft ``study_type`` right in place (a draft, not a person's edit, which is refused instead): one that
    cannot be read is left out, and ``measure`` beside an ``optimisation`` block becomes ``find_best_design`` (the block
    asks for a search, and a search is never run as a sweep). A sentence per change."""
    if not isinstance(design, dict) or design.get("study_type") is None:
        return []
    canonical, _why = normalize_study_type(design["study_type"])
    if canonical is None:
        written = design.pop("study_type")
        return [f"the plan's `study_type` ({written!r}) was left out: it must be `measure` or `find_best_design`"]
    if canonical == "measure" and has_block(design):
        design["study_type"] = "find_best_design"
        return ["the draft said `study_type: measure` but wrote an `optimisation` block (a search for the best design); "
                "the plan says `find_best_design`. Set `study_type: measure` and give a `grid` if a measurement is meant."]
    return []


def has_block(design: Any) -> bool:
    protocol = design.get("protocol") if isinstance(design, dict) else None
    return isinstance(protocol, dict) and isinstance(protocol.get("optimisation"), dict)


def study_type_of(design: Any) -> str:
    """``find_best_design`` when the design says so or its protocol has an ``optimisation`` block (the safe reading: a
    block that asks for a search is never run as a sweep), else ``measure``."""
    if has_block(design):
        return "find_best_design"
    if isinstance(design, dict) and design.get("study_type") is not None:
        canonical, _why = normalize_study_type(design.get("study_type"))
        return canonical or "measure"
    return "measure"


# --- the block -----------------------------------------------------------------------------------------------------


def kind_of(variable: dict[str, Any]) -> str:
    """A design variable's kind; ``continuous`` when the plan left it out (``choice`` when it lists ``values``)."""
    kind = str(variable.get("kind") or "").strip().lower()
    if kind:
        return kind
    return "choice" if isinstance(variable.get("values"), list) else "continuous"


def _variable(item: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    where = f"{_P}.design_variables` entry {index}"
    if not isinstance(item, dict):
        return None, f"{where} is not a mapping (`name`, `low`, `high`, `kind`)"
    out = dict(item)
    name = _text(out.get("name"))
    if not name:
        return None, f"{where} has no `name` (the name the simulation reads it under)"
    out["name"] = name
    where = f"{_P}.design_variables` `{name}`"
    if out.get("kind") is not None:
        kind = str(out["kind"]).strip().lower()
        if kind not in KINDS:
            return None, f"{where}: `kind` must be `continuous`, `integer` or `choice`"
        out["kind"] = kind
    kind = kind_of(out)
    for key in ("unit", "meaning"):
        if out.get(key) is not None and not isinstance(out[key], (str, int, float)):
            return None, f"{where}: `{key}` must be text"
    if kind == "choice":
        values = out.get("values")
        if not isinstance(values, list) or not values or not all(_number(v) or (isinstance(v, str) and v.strip())
                                                                  for v in values):
            return None, f"{where}: a `choice` lists the `values` it may take (numbers or names)"
        if len({str(v) for v in values}) != len(values):
            return None, f"{where}: a value is listed twice in `values`"
        return out, None
    low, high = out.get("low"), out.get("high")
    if not _number(low) or not _number(high):
        return None, f"{where}: `low` and `high` must be numbers (the range the search may use)"
    if low >= high:
        return None, f"{where}: `low` ({low}) must be below `high` ({high})"
    if kind == "integer" and not (_whole(low) and _whole(high)):
        return None, f"{where}: an `integer` variable has whole-number `low` and `high`"
    return out, None


def _inside(variable: dict[str, Any], value: Any) -> bool:
    kind = kind_of(variable)
    if kind == "choice":
        return any(value == v for v in variable["values"])
    if not _number(value):
        return False
    if kind == "integer" and not _whole(value):
        return False
    return variable["low"] <= value <= variable["high"]


def _limit(value: Any) -> tuple[str, float] | None:
    match = _LIMIT.match(str(value)) if isinstance(value, str) else None
    if match is None:
        return None
    number = float(match.group(2))
    if not math.isfinite(number):
        return None  # "<= 1e999" would be written back as "<= inf", which cannot be read again
    return _OPS[match.group(1)], number


def _fmt(value: Any) -> str:
    if _number(value):
        return str(int(value)) if float(value).is_integer() and abs(value) < 1e15 else repr(float(value))
    return str(value)


def _constraint(item: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    where = f"{_P}.constraints` entry {index}"
    if not isinstance(item, dict):
        return None, f"{where} is not a mapping (`quantity`, `limit`)"
    out = dict(item)
    quantity = _text(out.get("quantity"))
    if not quantity:
        return None, f"{where} has no `quantity` (the name the simulation returns it under)"
    out["quantity"] = quantity
    parsed = _limit(out.get("limit"))
    if parsed is None:
        return None, (f"{where} (`{quantity}`): `limit` must read like \"<= 120\" or \">= 3\" (a bound and a number); "
                      f"{out.get('limit')!r} cannot be read")
    out["limit"] = f"{parsed[0]} {_fmt(parsed[1])}"  # a bound is inclusive: "< 120" is read as "<= 120"
    if out.get("unit") is not None and not isinstance(out["unit"], (str, int, float)):
        return None, f"{where}: `unit` must be text"
    return out, None


def _setting(name: str, item: Any) -> tuple[dict[str, Any] | None, str | None]:
    where = f"{_P}.numerical_settings` `{name}`"
    if _number(item):
        item = {"search": item}
    if not isinstance(item, dict):
        return None, f"{where} must be a mapping with `search` (and `check`, the finer values)"
    out = dict(item)
    search = out.get("search")
    if not _number(search) or search <= 0:
        return None, f"{where}: `search` must be a positive number (the value the search uses)"
    finer = str(out.get("finer") or "smaller").strip().lower()
    if finer not in ("smaller", "larger"):
        return None, f"{where}: `finer` must be `smaller` (a mesh size, a time step) or `larger` (a count of elements)"
    if out.get("finer") is not None:
        out["finer"] = finer
    check = out.get("check")
    if check is None:
        return out, None
    levels = check if isinstance(check, list) else [check]
    if not levels or not all(_number(v) and v > 0 for v in levels):
        return None, f"{where}: `check` must be one or more positive numbers (the finer values)"
    before = search
    for level in levels:
        ratio = before / level if finer == "smaller" else level / before
        if ratio < MIN_REFINEMENT:
            return None, (f"{where}: the check value {_fmt(level)} is not finer than {_fmt(before)} (each level must be "
                          f"at least {MIN_REFINEMENT} times {'smaller' if finer == 'smaller' else 'larger'} than the one "
                          "before it)")
        before = level
    out["check"] = list(levels)
    return out, None


def _whole_at_least_one(value: Any) -> bool:
    return _whole(value) and value >= 1


def _grid(grid: Any, variables: dict[str, dict[str, Any]]) -> str | None:
    if not isinstance(grid, dict) or not grid:
        return f"{_P}.grid` (the coarse scan run first) must map each design variable to the values it takes"
    for axis, values in grid.items():
        if str(axis) not in variables:
            return f"{_P}.grid` scans `{axis}`, which is not one of the design variables"
        values = values if isinstance(values, list) else [values]
        if not values or not all(_inside(variables[str(axis)], v) for v in values):
            return f"{_P}.grid` `{axis}`: every value must lie inside that variable's range"
    return None


def normalize(block: Any) -> tuple[dict[str, Any] | None, str | None]:
    """``(block, None)`` when the optimisation block can be read and checked, else ``(None, why)``."""
    if not isinstance(block, dict):
        return None, f"{_P}` must be a mapping (`objective`, `design_variables`, `baseline`, ...)"
    out: dict[str, Any] = dict(block)
    objective = out.get("objective")
    if not isinstance(objective, dict):
        return None, f"{_P}.objective` must say the `quantity` to minimise or maximise and its `direction`"
    objective = dict(objective)
    if not _text(objective.get("quantity")):
        return None, f"{_P}.objective` has no `quantity` (the name the simulation returns it under)"
    objective["quantity"] = _text(objective["quantity"])
    direction = DIRECTIONS.get(str(objective.get("direction") or "").strip().lower())
    if direction is None:
        return None, f"{_P}.objective.direction` must be `minimise` or `maximise`"
    objective["direction"] = direction
    for key in ("unit", "meaning"):
        if objective.get(key) is not None and not isinstance(objective[key], (str, int, float)):
            return None, f"{_P}.objective.{key}` must be text"
    out["objective"] = objective

    items = out.get("design_variables")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        return None, f"{_P}.design_variables` must list what the search may change (each with a `name` and a range)"
    variables: dict[str, dict[str, Any]] = {}
    fixed_items: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        variable, why = _variable(item, index)
        if variable is None:
            return None, why
        if variable["name"] in variables:
            return None, f"{_P}.design_variables` names `{variable['name']}` twice"
        variables[variable["name"]] = variable
        fixed_items.append(variable)
    out["design_variables"] = fixed_items

    fixed = out.get("fixed")
    if fixed is not None:
        if not isinstance(fixed, dict):
            return None, f"{_P}.fixed` must map each condition held fixed to its value"
        clash = sorted(set(map(str, fixed)) & set(variables))
        if clash:
            return None, f"{_P}.fixed` holds `{clash[0]}` fixed, but it is also a design variable the search may change"

    baseline = out.get("baseline")
    if not isinstance(baseline, dict) or not isinstance(baseline.get("values"), dict):
        return None, (f"{_P}.baseline` must give the design to beat: `values` (one per design variable) and `source` "
                      "(where it comes from)")
    values = {_text(k): v for k, v in baseline["values"].items()}  # a name is read with its spaces collapsed
    out["baseline"] = {**baseline, "values": values}
    missing = [name for name in variables if name not in values]
    if missing:
        return None, f"{_P}.baseline.values` has no value for `{missing[0]}` (the baseline gives every design variable)"
    extra = [str(name) for name in values if str(name) not in variables]
    if extra:
        return None, f"{_P}.baseline.values` names `{extra[0]}`, which is not a design variable"
    for name, variable in variables.items():
        if not _inside(variable, values[name]):
            return None, (f"{_P}.baseline.values` `{name}` = {values[name]!r} is outside the range the search may use "
                          "(or not a value that variable takes)")
    if baseline.get("source") is not None and not isinstance(baseline["source"], (str, int, float)):
        return None, f"{_P}.baseline.source` must be text"

    constraints = out.get("constraints")
    if constraints is not None:
        if isinstance(constraints, dict):
            constraints = [constraints]
        if not isinstance(constraints, list):
            return None, f"{_P}.constraints` must be a list (each a `quantity` and a `limit` such as \"<= 120\")"
        fixed_constraints = []
        for index, item in enumerate(constraints, start=1):
            constraint, why = _constraint(item, index)
            if constraint is None:
                return None, why
            fixed_constraints.append(constraint)
        out["constraints"] = fixed_constraints

    settings = out.get("numerical_settings")
    if settings is not None:
        if not isinstance(settings, dict):
            return None, f"{_P}.numerical_settings` must map each setting (a mesh size, a time step) to its values"
        fixed_settings: dict[str, Any] = {}
        for name, item in settings.items():
            setting, why = _setting(str(name), item)
            if setting is None:
                return None, why
            fixed_settings[str(name)] = setting
        out["numerical_settings"] = fixed_settings

    budget = out.get("evaluation_budget")
    if budget is not None:
        if not isinstance(budget, dict):
            return None, f"{_P}.evaluation_budget` must give `starts` and `per_start` (whole numbers)"
        for key in ("starts", "per_start"):
            if not _whole_at_least_one(budget.get(key)):
                return None, f"{_P}.evaluation_budget.{key}` must be a whole number of at least 1"
        seconds = budget.get("seconds_per_evaluation")
        if seconds is not None and (not _number(seconds) or seconds <= 0):
            return None, f"{_P}.evaluation_budget.seconds_per_evaluation` must be a positive number (an estimate)"
    for key in ("runs_per_evaluation", "check_runs"):
        if out.get(key) is not None and not _whole_at_least_one(out[key]):
            return None, f"{_P}.{key}` must be a whole number of at least 1"

    method = out.get("search_method")
    if method is not None:
        name = str(method).strip() if isinstance(method, str) else ""
        if name not in BUILT_IN_METHODS and not _LIBRARY_METHOD.match(name):
            return None, (f"{_P}.search_method` must be one of {', '.join(BUILT_IN_METHODS)}, or a library method "
                          "written as `scipy:<function>` or `optuna:<sampler>`")
        out["search_method"] = name

    tolerance = out.get("improvement_tolerance")
    if tolerance is not None:
        if isinstance(tolerance, dict):
            value, mode = tolerance.get("value"), str(tolerance.get("mode") or "absolute").strip().lower()
            if not _number(value) or value < 0:
                return None, f"{_P}.improvement_tolerance.value` must be a number of at least 0"
            if mode not in ("absolute", "relative"):
                return None, f"{_P}.improvement_tolerance.mode` must be `absolute` or `relative`"
            if mode == "relative" and value > 1:
                return None, (f"{_P}.improvement_tolerance.value` is a fraction when `mode` is relative (0.05 means 5%); "
                              f"{_fmt(value)} would mean more than 100%")
            if tolerance.get("mode") is not None:
                out["improvement_tolerance"] = {**tolerance, "mode": mode}
        elif not _number(tolerance) or tolerance < 0:
            return None, f"{_P}.improvement_tolerance` must be a number (how much better counts), or leave it out"

    target = out.get("target")
    if target is not None:
        value = target.get("value") if isinstance(target, dict) else target
        if not _number(value):
            return None, f"{_P}.target` must be a number (the value of the objective you hope to reach), or leave it out"

    margin = out.get("constraint_margin")
    if margin is not None and (not _number(margin) or margin < 0):
        return None, f"{_P}.constraint_margin` must be a number of at least 0"

    grid = out.get("grid")
    if grid is not None:
        why = _grid(grid, variables)
        if why:
            return None, why
        out["grid"] = {str(axis): (v if isinstance(v, list) else [v]) for axis, v in grid.items()}
    return out, None


_ENTRY = re.compile(r"entry (\d+)")
_NAMED = re.compile(r"`protocol\.optimisation\.(\w+)` `([^`]+)`")


def repair(block: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """A drafted block with each part that cannot be read left out, and a sentence per part saying so; ``(None, notes)``
    when what is left cannot drive a search (no objective, no design variable, or no usable baseline).

    Left out one at a time: a design variable (and its baseline value and coarse-scan axis with it), a constraint, a
    numerical setting, or an optional part. A ``check`` that is not finer than ``search`` is left out on its own: FI's
    fixed rule then gives the finer values, and the plan says so."""
    if not isinstance(block, dict):
        return None, [f"the optimisation block (`protocol.optimisation`) was not a mapping and was left out"]
    out: dict[str, Any] = dict(block)
    notes: list[str] = []
    for _ in range(64):
        fixed, why = normalize(out)
        if fixed is not None:
            return fixed, notes
        why = why or ""
        part = re.match(r"`protocol\.optimisation\.?(\w*)", why)
        key = part.group(1) if part else ""
        cannot = f"the optimisation block was left out because it cannot drive a search ({why})"
        if key in ("", "objective"):
            return None, notes + [cannot]
        if key == "design_variables":
            items = out["design_variables"] if isinstance(out.get("design_variables"), list) else []
            entry = _ENTRY.search(why)
            named = re.search(r"`protocol\.optimisation\.design_variables` (?:names )?`([^`]+)`", why)
            drop = None
            if entry:
                drop = int(entry.group(1)) - 1
            elif named:
                same = [i for i, v in enumerate(items) if isinstance(v, dict) and _text(v.get("name")) == named.group(1)]
                if f"{_P}.design_variables` names `" in why:
                    drop = same[-1] if same else None
                else:
                    drop = next((i for i in same if _variable(items[i], i + 1)[0] is None), None)
            if drop is None or not (0 <= drop < len(items)) or len(items) <= 1:
                return None, notes + [cannot]
            gone = items[drop]
            name = _text(gone.get("name")) if isinstance(gone, dict) else ""
            out["design_variables"] = items[:drop] + items[drop + 1:]
            if f"{_P}.design_variables` names `" in why:
                notes.append(f"the design variable {name} was listed twice; the second entry was left out")
                continue
            if name and isinstance(out.get("baseline"), dict) and isinstance(out["baseline"].get("values"), dict):
                out["baseline"] = {**out["baseline"], "values": {k: v for k, v in out["baseline"]["values"].items()
                                                                 if str(k) != name}}
            if name and isinstance(out.get("grid"), dict):
                out["grid"] = {k: v for k, v in out["grid"].items() if str(k) != name} or None
                if out["grid"] is None:
                    del out["grid"]
            notes.append(f"a design variable ({name or f'entry {drop + 1}'}) was left out of the optimisation because it "
                         f"could not be read ({why}); put it right in the plan if the search should change it")
            continue
        if key == "fixed" and isinstance(out.get("fixed"), dict) and "also a design variable" in why:
            clash = [k for k in out["fixed"] if _text(k) in {_text(v.get("name")) for v in out.get("design_variables") or []
                                                              if isinstance(v, dict)}]
            out["fixed"] = {k: v for k, v in out["fixed"].items() if k not in clash}
            notes.append(f"{', '.join(map(str, clash))} is a design variable the search may change, so it is not held "
                         "fixed; the other fixed conditions are kept")
            continue
        if key == "baseline":
            extra = re.search(r"names `([^`]+)`, which is not a design variable", why)
            if extra and isinstance(out.get("baseline"), dict) and isinstance(out["baseline"].get("values"), dict):
                out["baseline"] = {**out["baseline"], "values": {k: v for k, v in out["baseline"]["values"].items()
                                                                 if str(k) != extra.group(1)}}
                notes.append(f"the baseline's value of `{extra.group(1)}` was left out: it is not one of the design "
                             "variables (hold it fixed under `fixed` if the simulation needs it)")
                continue
            return None, notes + [f"the optimisation block was left out because its baseline cannot be used ({why})"]
        if key == "constraints" and isinstance(out.get("constraints"), list):
            entry = _ENTRY.search(why)
            index = int(entry.group(1)) - 1 if entry else -1
            if 0 <= index < len(out["constraints"]):
                gone = out["constraints"][index]
                label = _text(gone.get("quantity")) if isinstance(gone, dict) else ""
                out["constraints"] = out["constraints"][:index] + out["constraints"][index + 1:]
                notes.append(f"the limit on {label or f'entry {index + 1}'} was left out of the optimisation because it "
                             f"could not be read ({why}); write it in the plan, or the search will not respect it")
                continue
        if key == "numerical_settings" and isinstance(out.get("numerical_settings"), dict):
            named = _NAMED.search(why)
            name = named.group(2) if named else ""
            settings = dict(out["numerical_settings"])
            if name in settings:
                if ("`check`" in why or "check value" in why) and isinstance(settings[name], dict) and "check" in settings[name]:
                    settings[name] = {k: v for k, v in settings[name].items() if k != "check"}
                    notes.append(f"the finer check values of `{name}` were left out ({why}); FI's fixed rule gives them "
                                 "instead (half, then a quarter of the search value, or twice and four times for a count)")
                else:
                    del settings[name]
                    notes.append(f"the numerical setting `{name}` was left out of the optimisation because it could not "
                                 f"be read ({why}); write it in the plan if the check should refine it")
                out["numerical_settings"] = settings
                continue
        if key in out:
            del out[key]
            notes.append(f"`protocol.optimisation.{key}` was left out of the plan because it could not be read ({why}); "
                         "put it right here if it matters")
            continue
        return None, notes + [f"the optimisation block was left out because it could not be repaired ({why})"]
    return None, notes + ["the optimisation block was left out because it could not be repaired"]


# --- what FI does with what the plan left out ---------------------------------------------------------------------


def check_levels(setting: dict[str, Any]) -> tuple[list[float], str]:
    """``(the finer values the check uses, where they come from)``: the plan's ``check``, or FI's fixed rule (half, then a
    quarter of the search value; twice and four times for a setting where larger is finer)."""
    if isinstance(setting.get("check"), list) and setting["check"]:
        return list(setting["check"]), "plan"
    search = setting["search"]
    if str(setting.get("finer") or "smaller") == "larger":
        levels = [search * 2, search * 4]
    else:
        levels = [search / 2, search / 4]
    if _whole(search) and all(_whole(v) for v in levels):
        levels = [int(v) for v in levels]
    return levels, "rule"


def effective_method(block: dict[str, Any]) -> tuple[str, str]:
    """``(method, where it comes from)``: the plan's ``search_method``, or FI's choice (``exhaustive`` when every
    variable takes few values and all of them fit the budget, ``global_then_local`` with a coarse scan, else
    ``bounded_local``)."""
    method = block.get("search_method")
    if isinstance(method, str) and method:
        return method, "plan"
    variables = block.get("design_variables") or []
    if variables and all(kind_of(v) in ("integer", "choice") for v in variables):
        combos = 1
        for v in variables:
            combos *= len(v["values"]) if kind_of(v) == "choice" else int(v["high"] - v["low"] + 1)
        search, _scan = evaluations(block)
        if search is not None and combos <= search:
            return "exhaustive", "rule"
    if isinstance(block.get("grid"), dict) and block["grid"]:
        return "global_then_local", "rule"
    return "bounded_local", "rule"


def evaluations(block: dict[str, Any]) -> tuple[int | None, int]:
    """``(search evaluations the budget allows, or None when no budget is written; evaluations of the coarse scan)``."""
    budget = block.get("evaluation_budget")
    search = int(budget["starts"]) * int(budget["per_start"]) if isinstance(budget, dict) else None
    scan = 0
    if isinstance(block.get("grid"), dict) and block["grid"]:
        scan = 1
        for values in block["grid"].values():
            scan *= len(values)
    return search, scan


def budget(block: dict[str, Any]) -> dict[str, Any]:
    """The evaluations a plan asks for, worked out when the plan is written (every number FI will count against):

    * ``scan``: the coarse scan's designs; ``search``: ``starts × per_start`` (a limit, not a target);
    * ``check`` (core/optimum_check.py, not taken from the search's budget): (the best ``min(3, starts)`` designs + the
      baseline) × the finer levels, plus two nudges of each variable that takes numbers (continuous or whole-numbered)
      around the best design; at most this many, fewer when a nudge would leave the range or starting points agree;
    * each times ``runs_per_evaluation`` / ``check_runs`` (by default as many as ``runs_per_evaluation``) for a study
      with randomness;
    * ``seconds``: the search's time from the plan's own estimate per evaluation, or ``None`` (not measured yet)."""
    search, scan = evaluations(block)
    budget_block = block.get("evaluation_budget") if isinstance(block.get("evaluation_budget"), dict) else {}
    starts = int(budget_block.get("starts") or 0)
    candidates = min(CHECK_CANDIDATES, starts) if starts else CHECK_CANDIDATES
    settings = block.get("numerical_settings") if isinstance(block.get("numerical_settings"), dict) else {}
    levels = max((len(check_levels(s if isinstance(s, dict) else {"search": s})[0]) for s in settings.values()),
                 default=0) or 1
    continuous = sum(1 for v in block.get("design_variables") or [] if kind_of(v) in ("continuous", "integer"))
    runs = int(block.get("runs_per_evaluation") or 1)
    check_runs = int(block.get("check_runs") or block.get("runs_per_evaluation") or 1)
    check = ((candidates + 1) * levels + 2 * continuous) * check_runs
    seconds_each = budget_block.get("seconds_per_evaluation")
    searched = (search or 0) * runs + scan * runs
    return {
        "scan": scan * runs, "search": search * runs if search is not None else None, "check": check,
        "starts": starts or None, "per_start": budget_block.get("per_start"), "candidates": candidates,
        "levels": levels, "continuous": continuous, "runs_per_evaluation": runs, "check_runs": check_runs,
        "seconds": searched * seconds_each if _number(seconds_each) and search is not None else None,
        "seconds_per_evaluation": seconds_each if _number(seconds_each) else None,
    }


def improvement_rule(block: dict[str, Any]) -> tuple[str, str]:
    """``(the rule for "better than the baseline", where it comes from)``."""
    unit = _text((block.get("objective") or {}).get("unit"))
    tolerance = block.get("improvement_tolerance")
    if isinstance(tolerance, dict) and _number(tolerance.get("value")):
        if str(tolerance.get("mode") or "absolute") == "relative":
            return f"better than the baseline by more than {_fmt(round(tolerance['value'] * 100, 6))}%", "plan"
        return f"better than the baseline by more than {_fmt(tolerance['value'])}{' ' + unit if unit else ''}", "plan"
    if _number(tolerance):
        return f"better than the baseline by more than {_fmt(tolerance)}{' ' + unit if unit else ''}", "plan"
    return "better than the baseline by more than the numerical error of the two designs", "rule"


def missing_parts(design: Any) -> list[str]:
    """What a ``find_best_design`` design lacks before a search could start, in plain words (empty when nothing)."""
    if study_type_of(design) != "find_best_design":
        return []
    if not has_block(design):
        return ["the plan asks for the best design but has no `optimisation` block (what to make as low or as high as "
                "possible, which settings may change and over what range, and the design to beat)"]
    block = design["protocol"]["optimisation"]
    out = []
    if not isinstance(block.get("evaluation_budget"), dict):
        out.append("no evaluation budget (`evaluation_budget`: how many starting points, and how many evaluations each)")
    return out


# --- the section of plan.md ------------------------------------------------------------------------------------------

HEADING = "What is being optimised"


def _flat(text: Any) -> str:
    return " ".join(str(text if text is not None else "").replace("`", "'").split())


def _range(variable: dict[str, Any]) -> str:
    unit = _flat(variable.get("unit"))
    kind = kind_of(variable)
    if kind == "choice":
        return "one of " + ", ".join(_flat(v) for v in variable["values"])
    span = f"{_fmt(variable['low'])} to {_fmt(variable['high'])}{' ' + unit if unit else ''}"
    return span + (" (whole numbers)" if kind == "integer" else "")


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def plan_lines(design: Any) -> list[str]:
    """The readable form of ``study_type`` and ``protocol.optimisation``, for ``plan.md`` above the design block. Nothing
    here is read back. Empty for a design that does not say what kind of study it is."""
    if not isinstance(design, dict):
        return []
    kind = study_type_of(design)
    if kind == "measure":
        if design.get("study_type") is None:
            return []
        return ["## What kind of study this is", "",
                "**A measurement** (`study_type: measure`): how the result changes over the settings in the protocol's "
                "grid. To look for the best design instead, set `study_type: find_best_design` and give an "
                "`optimisation` block in the design below.", ""]
    lines = [f"## {HEADING}", "",
             "> Shown from the design block below (`study_type` and `protocol.optimisation`) as it was when the plan was "
             "written; edit the block, not this section (after an edit, only the block counts). FI runs the search "
             "itself, within the budget below, and records every evaluation; the best design is found at the search's "
             "own numerical settings, then FI evaluates it and the baseline again at the finer check settings below, "
             "nudges it, and compares the starting points. A check that fails is reported as the result and limits the "
             "evidence level; it does not stop the quest.", "",
             "**Kind of study:** find the best design (`study_type: find_best_design`), not a measurement over "
             "settings chosen in advance.", ""]
    if not has_block(design):
        lines += ["- (the plan gives no `optimisation` block: write what to make as low or as high as possible, which "
                  "settings may change and over what range, and the design to beat, in the design below; or set "
                  "`study_type: measure` and give a `grid`)", ""]
        return lines
    block = design["protocol"]["optimisation"]
    objective = block["objective"]
    unit = _flat(objective.get("unit"))
    meaning = _flat(objective.get("meaning"))
    goal = "as low as possible" if objective["direction"] == "minimise" else "as high as possible"
    lines += [f"**Goal:** make {_flat(objective['quantity'])}{f' ({meaning})' if meaning else ''} {goal}"
              f"{f', in {unit}' if unit else ''}.", ""]
    lines += ["**What the search may change:**", ""]
    lines += [f"- {_flat(v['name'])}: {_range(v)}" for v in block["design_variables"]]
    lines.append("")
    fixed = block.get("fixed")
    if isinstance(fixed, dict) and fixed:
        lines += ["**Held fixed:** " + "; ".join(f"{_flat(k)} = {_flat(_fmt(v))}" for k, v in fixed.items()), ""]
    constraints = block.get("constraints") or []
    if constraints:
        margin = block.get("constraint_margin")
        rows = "; ".join(f"{_flat(c['quantity'])} {c['limit'].replace('<=', '≤').replace('>=', '≥')}"
                         f"{' ' + _flat(c.get('unit')) if c.get('unit') else ''}" for c in constraints)
        lines += [f"**Limits every design must meet:** {rows}"
                  f"{f' (with a margin of {_fmt(margin)})' if _number(margin) and margin else ''}. A second goal is "
                  "written as one of these limits: there is one goal.", ""]
    else:
        lines += ["**Limits every design must meet:** none.", ""]
    baseline = block["baseline"]
    values = "; ".join(f"{_flat(k)} = {_flat(_fmt(v))}" for k, v in baseline["values"].items())
    source = _flat(baseline.get("source")) or "(not said: name where this design comes from)"
    lines += [f"**The design to beat (baseline):** {values}. From: {source}", ""]
    rule, where = improvement_rule(block)
    lines += [f"**When a design counts as better:** {rule}"
              f"{' (the plan sets this)' if where == 'plan' else ' (FI’s default: the plan sets no threshold)'}.", ""]
    target = block.get("target")
    value = target.get("value") if isinstance(target, dict) else target
    if _number(value):
        lines += [f"**Your own target:** {_fmt(value)}{' ' + unit if unit else ''}. The result says how close the best "
                  "design comes to it; it is not a condition for success.", ""]
    method, where = effective_method(block)
    library = _LIBRARY_METHOD.match(method)
    described = (BUILT_IN_METHODS.get(method) or
                 (f"the {library.group(2)} method of {library.group(1)}, used only when the quest's environment has "
                  f"{library.group(1)} (otherwise bounded_local); each of its steps starts the quest's Python again, so "
                  "a large budget takes noticeably longer than the evaluations alone" if library else method))
    lines += [f"**How the search runs:** {method}: {described}"
              f"{'' if where == 'plan' else ' (FI’s choice: the plan names no method)'}.", ""]
    grid = block.get("grid")
    if isinstance(grid, dict) and grid:
        axes = " × ".join(f"{_flat(k)} [{', '.join(_flat(_fmt(v)) for v in vals)}]" for k, vals in grid.items())
        lines += [f"**A coarse scan first:** {axes}; plotted, and its best point is where the search goes on from.", ""]
    settings = block.get("numerical_settings") if isinstance(block.get("numerical_settings"), dict) else {}
    if settings:
        lines += ["**Numerical settings (the search's, and the finer ones the best design and the baseline are "
                  "recomputed at, so an improvement that is only a numerical error does not count):**", ""]
        for name, setting in settings.items():
            levels, where = check_levels(setting)
            said = ("the plan's values" if where == "plan" else
                    "FI's fixed rule, because the plan gave none: "
                    + ("twice, then four times the search value" if str(setting.get("finer") or "smaller") == "larger"
                       else "half, then a quarter of the search value"))
            lines.append(f"- {_flat(name)}: the search uses {_fmt(setting['search'])}; the check uses "
                         f"{', then '.join(_fmt(v) for v in levels)} ({said})")
        lines.append("")
    else:
        lines += ["**Numerical settings:** none named. If the simulation has a mesh size, a time step or a tolerance, "
                  "name it under `numerical_settings`: otherwise the best design is only recomputed at the same "
                  "settings, which cannot tell a real improvement from a numerical error.", ""]
    count = budget(block)
    lines += ["**How many evaluations (worked out now, before anything runs):**", ""]
    if count["scan"]:
        lines.append(f"- coarse scan: {count['scan']} evaluations")
    if count["search"] is None:
        lines.append("- search: no evaluation budget is written (`evaluation_budget`: `starts` and `per_start`); the "
                     "search cannot start without one")
    else:
        starts, per_start = int(count["starts"]), int(count["per_start"])
        runs = (f" × {count['runs_per_evaluation']} runs each = {count['search']} runs"
                if count["runs_per_evaluation"] > 1 else "")
        lines.append(f"- search: {starts} starting points × {per_start} = {starts * per_start} evaluations{runs}, "
                     "at most (a limit, not a target)")
    check_runs = f", × {count['check_runs']} fresh runs each" if count["check_runs"] > 1 else ""
    lines.append(f"- check: ({count['candidates']} best designs + the baseline) × {count['levels']} finer "
                 f"level{'s' if count['levels'] != 1 else ''} + 2 × {count['continuous']} nudges around the best design"
                 f"{check_runs} = {count['check']} evaluations at most, at the finer settings (each costs more than a "
                 "search evaluation); not taken from the search's budget, and with its own time limit "
                 "(`execution.timeout_s`)")
    if count["seconds"] is not None:
        lines.append(f"- time: about {_duration(count['seconds'])} for the scan and the search "
                     f"({_fmt(count['seconds_per_evaluation'])} s per evaluation, the plan's own estimate; not measured "
                     "yet)")
    else:
        lines.append("- time per evaluation: not measured yet")
    lines.append("- the scan and the search stop at the run's time limit (`execution.timeout_s`), whatever is left of the "
                 "budget; the results say which came first")
    lines.append("")
    return lines


# --- the question at the clarify step -------------------------------------------------------------------------------

# Words of looking for the best that are enough to ask the person (narrow: "maximum likelihood", "minimum spanning tree"
# or "best response" are not a search for a design) ...
_SEEK = re.compile(
    r"\b(optim(?:al|um|ums|a|i[sz]e[sd]?|i[sz]ing|i[sz]ation)|best(?!\s+(?:response|practice|practices|known|fit)\b)|"
    r"minimi[sz]e[sd]?|minimi[sz]ing|maximi[sz]e[sd]?|maximi[sz]ing)\b|"
    r"最佳|最優|最优|最適|最小化|最大化|優化|优化",
    re.IGNORECASE,
)
# ... and the broader set that is enough to tell the plan how to write a search, should it be one (the plan decides).
_SEEK_BROAD = re.compile(
    _SEEK.pattern + r"|\b(minimum|maximum|lowest|highest|smallest|largest|lightest|cheapest|fastest|coolest)\b|"
    r"最好|最低|最高|最輕|最轻",
    re.IGNORECASE,
)
_FIND = re.compile(
    r"\b(find|choose|select|design|pick|determine|identify)\b[^.?!]{0,80}\b(best|optimal|optimum|minimi[sz]e[sd]?|"
    r"maximi[sz]e[sd]?|minimi[sz]ing|maximi[sz]ing|lowest|highest|smallest|largest|lightest)\b|"
    r"\boptimi[sz]e\s+(the\s+)?\w+|(找出|找到|求出|設計出|设计出|選出|选出)[^。？?!！]{0,30}(最|極|极)|最佳化|最优化",
    re.IGNORECASE,
)
_MEASURE = re.compile(
    r"\b(how\s+(does|do|much|strongly)|effect\s+of|effects\s+of|influence\s+of|impact\s+of|depend(s|ence)?\s+on|"
    r"as\s+a\s+function\s+of|vary|varies|varying|sweep|scan|sensitivity|trade-?off|compare|comparison|versus|vs\.?)\b|"
    r"如何|怎麼|怎么|影響|影响|隨著|随着|變化|变化|比較|比较",
    re.IGNORECASE,
)


def classify_topic(topic: str) -> str:
    """``measure``, ``find_best_design``, or ``ambiguous``, from the topic's words alone.

    A topic with no word of looking for the best (best, optimal, minimise, lowest, 最佳 ...) is a measurement. One that asks
    to find, choose or design the best (or to optimise something) and has no word of measuring (how does, effect of, as a
    function of, compare ...) is a search for the best design. Anything else is ambiguous, and the person is asked."""
    text = topic or ""
    if _FIND.search(text) and not _MEASURE.search(text):
        return "find_best_design"
    if not _SEEK.search(text):
        return "measure"
    return "ambiguous"


# A topic that plainly asks for a search: a verb of choosing and a word that only a search uses ("find the fin spacing
# that minimises ...", "choose the optimal gain"). "Find the lowest error of Simpson's rule" or "determine the best-fit
# exponent" are measurements and do not match.
_PLAIN_SEARCH = re.compile(
    r"\b(find|choose|select|design|pick|determine|identify)\b[^.?!]{0,80}\b(optimal|optimum|optimi[sz]e[sd]?|"
    r"optimi[sz]ing|minimi[sz]e[sd]?|maximi[sz]e[sd]?|minimi[sz]ing|maximi[sz]ing|"
    r"best(?![\s-]+(?:fit|response|practice|practices|known|case)\b))\b|"
    r"(找出|找到|求出|設計出|设计出|選出|选出)[^。？?!！]{0,30}(最佳|最優|最优)|最佳化|最优化",
    re.IGNORECASE,
)


def plainly_seeks_best(topic: str) -> bool:
    """Whether the topic itself plainly asks to find the best design (narrower than :func:`classify_topic`): the plan step
    then writes ``study_type: find_best_design`` into a draft that says nothing of its kind, with a note."""
    text = topic or ""
    return bool(_PLAIN_SEARCH.search(text)) and not _MEASURE.search(text)


def may_seek_best(topic: str) -> bool:
    """Whether the topic has any word of looking for the best (broader than :func:`classify_topic`'s): the plan prompt
    then carries the rules for writing a search for the best design. A topic with none of them is planned as before."""
    return classify_topic(topic) != "measure" or bool(_SEEK_BROAD.search(topic or ""))


LET_THE_PLAN_DECIDE = "let the plan decide"
QUESTION = (
    "Is this study meant to (1) measure how the result changes as the design changes, over settings chosen in advance, "
    "or (2) find the best design: search for the design that makes the result as low or as high as possible, and "
    "compare it with a starting design? Answer 1 or 2 (or leave it, and the plan decides from the topic)."
)


def add_study_type_question(questions: dict[str, Any], topic: str) -> bool:
    """Add the one question about the kind of study to the clarify step's questions, when the topic's words leave it
    open (:func:`classify_topic`). The question goes through the same form as every other clarify question, on every
    interface. ``True`` when it was added."""
    if not isinstance(questions, dict) or classify_topic(topic) != "ambiguous":
        return False
    questions["study_type"] = {"question": QUESTION, "default": LET_THE_PLAN_DECIDE}
    return True


def resolve_answer(value: Any) -> str | None:
    """The kind of study a person's answer names (``1``/``2``, or the words), or ``None`` (the plan decides)."""
    text = " ".join(str(value or "").strip().lower().split())
    if not text or text == LET_THE_PLAN_DECIDE:
        return None
    if re.search(r"(?<!\d)1(?!\d)", text) and re.search(r"(?<!\d)2(?!\d)", text):
        return None  # "1 or 2": no choice made
    if text in ("1", "(1)", "1.") or text.startswith("1 ") or text.startswith("(1)"):
        return "measure"
    if text in ("2", "(2)", "2.") or text.startswith("2 ") or text.startswith("(2)"):
        return "find_best_design"
    canonical, _why = normalize_study_type(text)
    if canonical:
        return canonical
    best = re.search(r"\b(best|optim|minimi|maximi)|最佳|最好|最優|最优", text)
    measure = re.search(r"\b(measure|how|change|sweep|scan)|量測|测量|變化|变化", text)
    if bool(best) == bool(measure):
        return None  # both, or neither: the plan decides rather than a guess
    return "find_best_design" if best else "measure"


def answer_label(value: Any) -> str:
    kind = resolve_answer(value)
    return STUDY_TYPES[kind] if kind else LET_THE_PLAN_DECIDE


def numbers(block: Any) -> list[tuple[float, str]]:
    """Every number the optimisation block fixes, with where it sits (for the check that the plan keeps the numbers the
    topic sets, :func:`core.protocol_check.plan_notes`)."""
    out: list[tuple[float, str]] = []
    if not isinstance(block, dict):
        return out
    for v in block.get("design_variables") or []:
        if isinstance(v, dict):
            for key in ("low", "high"):
                if _number(v.get(key)):
                    out.append((float(v[key]), f"range of {v.get('name')}"))
            for value in v.get("values") or [] if isinstance(v.get("values"), list) else []:
                if _number(value):
                    out.append((float(value), f"values of {v.get('name')}"))
    for name, value in (block.get("fixed") or {}).items() if isinstance(block.get("fixed"), dict) else []:
        if _number(value):
            out.append((float(value), f"fixed {name}"))
    for c in block.get("constraints") or [] if isinstance(block.get("constraints"), list) else []:
        parsed = _limit(c.get("limit")) if isinstance(c, dict) else None
        if parsed:
            out.append((parsed[1], f"limit on {c.get('quantity')}"))
    baseline = block.get("baseline")
    if isinstance(baseline, dict) and isinstance(baseline.get("values"), dict):
        for name, value in baseline["values"].items():
            if _number(value):
                out.append((float(value), f"baseline {name}"))
    budget_block = block.get("evaluation_budget")
    if isinstance(budget_block, dict):
        for key in ("starts", "per_start"):
            if _number(budget_block.get(key)):
                out.append((float(budget_block[key]), f"evaluation budget {key}"))
    target = block.get("target")
    value = target.get("value") if isinstance(target, dict) else target
    if _number(value):
        out.append((float(value), "target"))
    tolerance = block.get("improvement_tolerance")
    value = tolerance.get("value") if isinstance(tolerance, dict) else tolerance
    if _number(value):
        out.append((float(value), "improvement threshold"))
    for axis, values in (block.get("grid") or {}).items() if isinstance(block.get("grid"), dict) else []:
        for value in values if isinstance(values, list) else []:
            if _number(value):
                out.append((float(value), f"coarse scan {axis}"))
    return out
