"""A simulation whose output does not depend on its inputs is broken, not a finding.

Two shapes, both read from what FI itself recorded (never from the script's own claims):

* a search for the best design in which the objective (and every limit's quantity) is the same at every design that
  was evaluated, or in which FI's check found that nudging every design variable changes nothing the simulation returns;
* a measurement over a grid in which every number the simulation returned is the same in every setting.

Either is what a simulation that never reads the values it is given looks like, so FI sends the simulation back to be
repaired, in plain words ("the result does not change when ... change, so the simulation does not use them") and with
no expected value in them. Only the all-identical case counts: a quantity that one setting does not move while
another quantity does move is left alone, and so is a single setting.
"""

from __future__ import annotations

import math
from typing import Any

#: Two numbers are "the same" when they differ by less than this fraction of their size (plus the same amount absolute):
#: numerical noise, not a different answer.
SAME_TOLERANCE = 1e-9
#: A search is read only once it has tried this many designs: two equal numbers prove nothing.
MIN_DESIGNS = 3


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def same(a: float, b: float) -> bool:
    """``a`` and ``b`` agree within numerical noise."""
    return abs(float(a) - float(b)) <= SAME_TOLERANCE * (1.0 + max(abs(float(a)), abs(float(b))))


def _all_same(values: list[float]) -> bool:
    return bool(values) and all(same(v, values[0]) for v in values[1:])


def _names(items: list[str], *, limit: int = 6) -> str:
    shown = [f"`{n}`" for n in items[:limit]]
    return ", ".join(shown) + (f" and {len(items) - limit} more" if len(items) > limit else "")


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def _symptom(quantities: list[str], values: dict[str, float], varied: list[str], count: int, what: str) -> str:
    said = ("it was " + _fmt(values[quantities[0]]) if len(quantities) == 1 and quantities[0] in values
            else "each had one value")
    return (f"the result does not change when {_names(varied)} change, so the simulation does not use "
            f"{'it' if len(varied) == 1 else 'them'}: {_names(quantities)} {'was' if len(quantities) == 1 else 'were'} "
            f"the same at all {count} {what} ({said}). Make the simulation compute its results from the values it is "
            "given, not from a fixed number.")


def flat_search(rows: list[dict[str, Any]], objective: str = "the objective") -> str | None:
    """The plain description of a search whose evaluations all returned the same numbers, else ``None``. ``rows`` are
    the search's evaluations as FI wrote them (``status``, ``design``, ``objective``, ``constraints``); ``objective`` is
    the name the plan gives the quantity."""
    ok = [r for r in rows if isinstance(r, dict) and r.get("status") == "ok" and isinstance(r.get("design"), dict)]
    designs = {repr(sorted((str(k), repr(v)) for k, v in r["design"].items())) for r in ok}
    if len(ok) < MIN_DESIGNS or len(designs) < 2:
        return None
    if not all(_is_number(r.get("objective")) for r in ok):
        return None
    quantities = {objective: [float(r["objective"]) for r in ok]}
    for name in sorted({n for r in ok for n in (r.get("constraints") or {})}):
        column = [r["constraints"][name] for r in ok if isinstance(r.get("constraints"), dict) and name in r["constraints"]]
        if len(column) != len(ok) or not all(_is_number(v) for v in column):
            return None
        quantities[name] = [float(v) for v in column]
    if not all(_all_same(v) for v in quantities.values()):
        return None
    names = sorted({n for r in ok for n in r["design"]})
    varied = [n for n in names if len({repr(r["design"].get(n)) for r in ok}) > 1] or names
    shown = {k: v[0] for k, v in quantities.items()}
    return _symptom(list(quantities), shown, varied, len(ok), "designs the search tried")


def flat_search_check(check: dict[str, Any] | None, block: dict[str, Any] | None) -> str | None:
    """The same symptom as FI's own check of the best design found it: nudging every variable that can be nudged changed
    nothing the simulation returns. ``None`` when the check did not say so for all of them (one that does matter, a
    check that did not finish, or a design with only choices)."""
    if not isinstance(check, dict) or not isinstance(block, dict) or check.get("finished") is False:
        return None
    hood = (check.get("checks") or {}).get("neighbourhood")
    if not isinstance(hood, dict):
        return None
    no_effect = [str(n) for n in hood.get("no_effect") or []]
    if not no_effect:
        return None
    sensitivity = hood.get("sensitivity") or {}
    movable = [str(v.get("name")) for v in block.get("design_variables") or []
               if isinstance(v, dict) and str(v.get("kind") or "continuous") != "choice"]
    if not movable or set(movable) - set(no_effect) or set(sensitivity) - set(no_effect):
        return None
    return (f"the result does not change when {_names(movable)} change, so the simulation does not use "
            f"{'it' if len(movable) == 1 else 'them'}: nudging {'it' if len(movable) == 1 else 'each of them'} "
            "changed nothing the simulation returns. Make the simulation compute its results from the values it is "
            "given, not from a fixed number.")


def flat_sweep(cells: list[Any]) -> str | None:
    """The plain description of a measurement whose every returned number is the same in every setting, else ``None``.
    ``cells`` are :class:`core.trial_runner.CellRun` objects (``cell`` and ``rows``, each row with ``status`` and
    ``values``). A study with one setting, one that varies nothing, or one in which some quantity does change from one
    setting to the next is left alone."""
    by_cell: list[tuple[dict[str, Any], dict[str, list[float]]]] = []
    for c in cells:
        columns: dict[str, list[float]] = {}
        ok = [r for r in getattr(c, "rows", []) if r.get("status") == "ok"]
        if not ok:
            continue
        for r in ok:
            for name, value in (r.get("values") or {}).items():
                if _is_number(value):
                    columns.setdefault(str(name), []).append(float(value))
        by_cell.append((dict(getattr(c, "cell", {}) or {}), columns))
    if len(by_cell) < 2:
        return None
    axes = sorted({a for cell, _ in by_cell for a in cell})
    varied = [a for a in axes if len({repr(cell.get(a)) for cell, _ in by_cell}) > 1]
    if not varied:
        return None
    names = sorted({n for _, cols in by_cell for n in cols})
    if not names:
        return None
    pooled: dict[str, list[float]] = {}
    for _, cols in by_cell:
        if set(cols) != set(names):
            return None
        for n, values in cols.items():
            pooled.setdefault(n, []).extend(values)
    if not all(_all_same(v) for v in pooled.values()):
        return None
    return _symptom(names, {n: v[0] for n, v in pooled.items()}, varied, len(by_cell), "settings")
