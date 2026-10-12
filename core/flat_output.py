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

import ast
import math
from dataclasses import dataclass
from typing import Any, Iterable

from . import optimisation_plan as _plan

#: Two numbers are "the same" when they differ by less than this fraction of their own size: numerical noise, not a
#: different answer. There is no absolute floor: a floor has a unit, and the same quantity written in another unit
#: (a length in metres or in picometres) must be judged the same way. A set of numbers that cannot be told apart by their
#: own size (some near zero beside others that are not, or noise around zero) is not called the same.
SAME_TOLERANCE = 1e-9
#: A search is read only once it has tried this many designs: two equal numbers prove nothing.
MIN_DESIGNS = 3


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def same(a: float, b: float) -> bool:
    """``a`` and ``b`` agree within numerical noise, measured against their own size (so the answer does not change when
    both are written in another unit). Two exact zeros agree; a zero and a number that is not zero do not."""
    a, b = float(a), float(b)
    return abs(a - b) <= SAME_TOLERANCE * max(abs(a), abs(b))


def _varies(values: list[Any]) -> bool:
    """Whether a design variable takes more than one value: numbers must differ by more than numerical noise (the same
    relative tolerance as :func:`same`), anything else by its value."""
    if all(_is_number(v) for v in values):
        return not _all_same([float(v) for v in values])
    return len({repr(v) for v in values}) > 1


def _all_same(values: list[float]) -> bool:
    """Every number is within numerical noise of the others, against the size of the numbers themselves: the spread of the
    set is compared with its largest magnitude, so multiplying all of them by one factor (a change of unit) never changes
    the answer. A set that is not all the same size to that tolerance is not called the same, and nothing is decided
    from it."""
    if not values:
        return False
    return max(values) - min(values) <= SAME_TOLERANCE * max(abs(v) for v in values)


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


def flat_search(rows: list[dict[str, Any]], objective: str = "the objective", *, source: str | None = None,
                expected: set[str] | frozenset[str] = frozenset()) -> str | None:
    """The plain description of a search whose evaluations all returned the same objective AND whose simulation never lets
    a design variable reach what it returns (evidence from its code, :func:`unread_settings`), else ``None``: the same
    objective at every design tried is otherwise only an observation (a plateau, or an objective the plan declares constant),
    and a finite search cannot prove the objective does not depend on the design. ``rows`` are the search's evaluations as
    FI wrote them (``status``, ``design``, ``objective``, ``constraints``); ``objective`` is the name the plan gives the
    quantity; ``expected`` the names the plan expects to stay put."""
    ok = [r for r in rows if isinstance(r, dict) and r.get("status") == "ok" and isinstance(r.get("design"), dict)]
    names = sorted({n for r in ok for n in r["design"]})
    varied = [n for n in names if _varies([r["design"].get(n) for r in ok])]
    if len(ok) < MIN_DESIGNS or not varied:
        return None
    if not all(_is_number(r.get("objective")) for r in ok):
        return None
    # The objective alone: a quantity that does not depend on the design is broken even when a limit's quantity varies.
    quantities = {objective: [float(r["objective"]) for r in ok]}
    if not _all_same(quantities[objective]):
        return None
    shown = {k: v[0] for k, v in quantities.items()}
    if objective in expected or not _ignores(source, varied, [objective]):
        return None
    return _symptom(list(quantities), shown, varied, len(ok), "designs the search tried")


def _ignores(source: str | None, keys: list[str], quantities: list[str] | None = None) -> bool:
    """Whether the code of ``source`` shows that the numbers could not have responded: none of the settings ``keys`` reaches
    anything it returns, or every one of the ``quantities`` is built from no setting at all."""
    if source is None or not keys:
        return False
    unread = unread_settings(source, keys)
    if unread is not None and set(unread) == set(keys):
        return True
    if quantities:
        unused = unused_outputs(source, quantities)
        return unused is not None and set(unused) == set(quantities)
    return False


def flat_search_check(check: dict[str, Any] | None, block: dict[str, Any] | None, *,
                      source: str | None = None) -> str | None:
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
    # Judged on the numbers the check really nudged: an array larger than the check's sample has elements it never moved,
    # and those are not evidence either way; a variable with no nudged number is left out of the claim.
    probed = hood.get("probed")
    movable, variables = [], []
    for v in block.get("design_variables") or []:
        if not isinstance(v, dict) or str(v.get("kind") or "continuous") == "choice":
            continue
        names = ([_plan.element_name(str(v.get("name")), i) for i in _plan.indices(_plan.shape_of(v))]
                 if _plan.is_array(v) else [str(v.get("name"))])
        if probed is not None:
            names = [n for n in names if n in {str(p) for p in probed}]
            if not names:
                continue  # a variable none of whose numbers was nudged: nothing is claimed about it, the others are judged
        movable.extend(names)
        variables.append(str(v.get("name")))  # the code reads a variable by its name, an array as a whole
    if not movable or set(movable) - set(no_effect) or set(sensitivity) - set(no_effect):
        return None
    if not _ignores(source, variables):
        return None
    return (f"the result does not change when {_names(movable)} change, so the simulation does not use "
            f"{'it' if len(movable) == 1 else 'them'}: nudging {'it' if len(movable) == 1 else 'each of them'} "
            "changed nothing the simulation returns. Make the simulation compute its results from the values it is "
            "given, not from a fixed number.")


@dataclass(frozen=True)
class Observed:
    """A measurement in which every number the simulation returned is the same in every setting. This is an observation, not
    a finding about the code: a conserved or invariant quantity, a plateau and a setting with no effect look exactly like
    this. See :func:`flat_sweep` for when it counts as a fault."""

    names: list[str]
    values: dict[str, float]
    varied: list[str]
    count: int
    what: str = "settings"

    def note(self) -> str:
        """The plain record of the observation, for the run's records and the writer: what did not change, and that a finite
        set of settings cannot show that there is no effect."""
        shown = (f" (it was {_fmt(self.values[self.names[0]])})"
                 if len(self.names) == 1 and self.names[0] in self.values else "")
        return (f"{_names(self.names)} {'was' if len(self.names) == 1 else 'were'} the same at all {self.count} "
                f"{self.what} tried{shown}, whatever {_names(self.varied)} {'was' if len(self.varied) == 1 else 'were'} set "
                "to. That is what a result that does not depend on the setting looks like, and also what a conserved or "
                "invariant quantity, a plateau or a true absence of effect looks like. FI found nothing in the code that "
                "ignores the setting, so this is reported as observed: a finite set of settings cannot show that there is "
                "no effect elsewhere, so it must not be written up as proof of none.")


def observed_flat_sweep(cells: list[Any]) -> Observed | None:
    """The observation behind :func:`flat_sweep`: ``cells`` are :class:`core.trial_runner.CellRun` objects (``cell`` and
    ``rows``, each row with ``status`` and ``values``). ``None`` for a study with one setting, one that varies nothing, or
    one in which some quantity does change from one setting to the next (or whose numbers cannot be told apart by their
    size)."""
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
    return Observed(names, {n: v[0] for n, v in pooled.items()}, varied, len(by_cell))


def flat_sweep(cells: list[Any], *, source: str | None = None,
               expected: set[str] | frozenset[str] = frozenset()) -> str | None:
    """The plain description of a measurement whose every returned number is the same in every setting AND whose simulation
    does not read the setting that was varied (evidence from its code, :func:`unread_settings`), else ``None``. Only that
    pair is a fault the simulation must be repaired for. Without ``source`` (or when the code does read the settings, or
    cannot be read with certainty) the same numbers are an observation, not a fault: :func:`observed_flat_sweep` has it.
    ``expected``: names the plan itself expects to stay put (an invariant, a symmetry): they are never a fault."""
    seen = observed_flat_sweep(cells)
    if seen is None:
        return None
    names = [n for n in seen.names if n not in expected]
    if not names or source is None:
        return None
    if not _ignores(source, seen.varied, names):
        return None
    return _symptom(names, seen.values, seen.varied, seen.count, "settings")


# ---- evidence from the code: what the simulation reads ---------------------------------------------------------------
#
# A result that did not change is a symptom, and several things cause it. The code says which: if the entry function never
# lets a setting reach what it returns, the setting is ignored whatever the numbers were. The reading is of the text
# (``ast``, nothing is run) and errs towards "it reads it": anything it cannot follow (a setting passed to another
# function, stored somewhere, looked up by name at run time) leaves the question undecided, so a fault is claimed only
# when the code clearly shows it.

ENTRIES = ("run_cell", "run_trial")
#: Names that look a value up by name at run time, which this reading cannot follow.
_DYNAMIC = frozenset({"eval", "exec", "locals", "vars", "globals", "getattr", "setattr", "__import__"})
#: Calls whose result is made from their arguments and that do nothing else: a setting given to one of them reaches the
#: result, and nowhere else.
_PURE_ROOTS = frozenset({"float", "int", "abs", "min", "max", "sum", "len", "round", "range", "list", "tuple", "dict", "set",
                         "str", "bool", "sorted", "enumerate", "zip", "reversed", "pow", "divmod", "math", "np", "numpy",
                         "scipy", "cmath"})
_FILLS = ("append", "extend", "add", "update", "insert", "setdefault", "appendleft")
_WHOLE = "@"  # the settings mapping itself, before one key is taken from it
_UNKNOWN = "\0unknown"


def _entry(tree: ast.Module) -> ast.FunctionDef | None:
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ENTRIES]
    return found[0] if len(found) == 1 else None


def _root(node: ast.AST) -> str:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else ""


class _Flow:
    """Which setting keys can reach each name and each returned value of the entry function (flow-insensitive: a name
    assigned anywhere carries what any assignment gave it, and a branch's test reaches what is assigned or returned in it).
    A key set holding ``_WHOLE`` stands for every key."""

    def __init__(self, func: ast.FunctionDef) -> None:
        params = [a.arg for a in [*func.args.posonlyargs, *func.args.args]]
        self.ok = bool(params)
        self.env: dict[str, set[str]] = {params[0]: {_WHOLE}} if params else {}
        self.func = func
        self.escaped = False  # a setting went somewhere this reading does not follow
        self.control: set[str] = set()  # what decides whether the function returns early or raises
        self.returns: list[tuple[ast.expr, set[str]]] = []
        self.dict_keys: dict[str, dict[str, set[str]]] = {}  # a local mapping built with literal names
        self.out_keys: dict[str, set[str]] = {}
        self.unknown_out = False
        for node in ast.walk(func):
            if isinstance(node, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await, ast.AsyncFunctionDef)):
                self.ok = False
            if isinstance(node, ast.Name) and node.id in _DYNAMIC:
                self.ok = False
        if not self.ok:
            return
        for _ in range(10):
            before = self._snapshot()
            self.returns, self.dict_keys, self.out_keys, self.unknown_out = [], {}, {}, False
            self._block(func.body, set())
            if before == self._snapshot():
                break

    def _snapshot(self) -> tuple[Any, ...]:
        return (sorted((k, sorted(v)) for k, v in self.env.items()), self.escaped, sorted(self.control))

    # -- the settings an expression is made from
    def deps(self, node: ast.AST | None) -> set[str]:
        if node is None:
            return set()
        if isinstance(node, ast.Name):
            return set(self.env.get(node.id, ()))
        if isinstance(node, ast.Subscript):
            base = self.deps(node.value)
            if base == {_WHOLE} and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                return {node.slice.value}
            return base | self.deps(node.slice)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.ClassDef)):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            return set().union(*[self.env.get(n, set()) for n in names]) if names else set()
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                self._bind(gen.target, self.deps(gen.iter))
        out: set[str] = set()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                out |= self.deps(child)
            elif isinstance(child, ast.comprehension):
                out |= self.deps(child.iter)
                for cond in child.ifs:
                    out |= self.deps(cond)
            elif isinstance(child, ast.keyword):
                out |= self.deps(child.value)
        return out

    def _call(self, node: ast.Call) -> set[str]:
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "get" and node.args and self.deps(func.value) == {_WHOLE}
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            out = {node.args[0].value}
            for a in node.args[1:]:
                out |= self.deps(a)
            return out
        out = self.deps(func)
        for a in node.args:
            out |= self.deps(a.value if isinstance(a, ast.Starred) else a)
        for k in node.keywords:
            out |= self.deps(k.value)
        root = _root(func)
        if out and root not in _PURE_ROOTS and root not in self.env:
            self.escaped = True  # a setting handed to a function this reading does not look into
        return out

    def _bind(self, target: ast.AST, deps: set[str]) -> None:
        if isinstance(target, ast.Name):
            self.env.setdefault(target.id, set()).update(deps)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                self._bind(t, deps)
        elif isinstance(target, ast.Starred):
            self._bind(target.value, deps)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            base = target.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name):
                self.env.setdefault(base.id, set()).update(deps)
            elif deps:
                self.escaped = True

    def _block(self, body: list[ast.stmt], control: set[str]) -> None:
        for stmt in body:
            self._stmt(stmt, control)

    def _stmt(self, node: ast.stmt, control: set[str]) -> None:
        if isinstance(node, ast.Assign):
            value = self.deps(node.value) | control
            for t in node.targets:
                self._bind(t, value)
                self._note_dict(t, node.value, control)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if node.value is not None:
                self._bind(node.target, self.deps(node.value) | control)
                self._note_dict(node.target, node.value, control)
        elif isinstance(node, ast.Return):
            if node.value is not None:
                self.returns.append((node.value, set(control)))
                self._note_return(node.value, control)
        elif isinstance(node, ast.Expr):
            self._expression_statement(node.value)
        elif isinstance(node, ast.Assert):
            self.control |= self.deps(node.test)
        elif isinstance(node, (ast.If, ast.While)):
            test = self.deps(node.test)
            self._block(node.body, control | test)
            self._block(node.orelse, control | test)
            if test and any(isinstance(n, (ast.Return, ast.Raise, ast.Break, ast.Continue))
                            for part in (node.body, node.orelse) for s in part for n in ast.walk(s)):
                self.control |= test
        elif isinstance(node, ast.For):
            it = self.deps(node.iter)
            self._bind(node.target, it | control)
            self._block(node.body, control | it)
            self._block(node.orelse, control | it)
            if it and any(isinstance(n, (ast.Return, ast.Raise, ast.Break)) for s in node.body for n in ast.walk(s)):
                self.control |= it
        elif isinstance(node, ast.With):
            for item in node.items:
                got = self.deps(item.context_expr)
                if item.optional_vars is not None:
                    self._bind(item.optional_vars, got | control)
                elif got:
                    self.escaped = True
            self._block(node.body, control)
        elif isinstance(node, ast.Try):
            for part in (node.body, node.orelse, node.finalbody, *[h.body for h in node.handlers]):
                self._block(part, control)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            self.env.setdefault(node.name, set()).update(self.deps(node))
        elif isinstance(node, (ast.Pass, ast.Break, ast.Continue, ast.Delete, ast.Import, ast.ImportFrom, ast.Raise)):
            return
        else:
            self.ok = False  # a statement this reading does not know

    def _expression_statement(self, value: ast.expr) -> None:
        """A call made for its effect: on a local container it only fills it (the container then carries what it was given);
        anything else that is given a setting may store it where this reading cannot see."""
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name) \
                and value.func.value.id in self.env and value.func.attr in _FILLS:
            given = set()
            for a in value.args:
                given |= self.deps(a)
            for k in value.keywords:
                given |= self.deps(k.value)
            self.env[value.func.value.id].update(given)
            return
        if self.deps(value):
            self.escaped = True

    # -- which settings reach each returned name (for a returned mapping built with literal names)
    def _note_dict(self, target: ast.AST, value: ast.AST, control: set[str]) -> None:
        if isinstance(target, ast.Name) and isinstance(value, ast.Dict):
            table = self.dict_keys.setdefault(target.id, {})
            for k, v in zip(value.keys, value.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    table.setdefault(k.value, set()).update(self.deps(v) | control)
                else:
                    table[_UNKNOWN] = {_WHOLE}
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            table = self.dict_keys.setdefault(target.value.id, {})
            if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                table.setdefault(target.slice.value, set()).update(self.deps(value) | control)
            else:
                table[_UNKNOWN] = {_WHOLE}

    def _note_return(self, value: ast.AST, control: set[str]) -> None:
        if isinstance(value, ast.Dict):
            for k, v in zip(value.keys, value.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    self.out_keys.setdefault(k.value, set()).update(self.deps(v) | control)
                else:
                    self.unknown_out = True
        elif isinstance(value, ast.Name) and value.id in self.dict_keys:
            for k, v in self.dict_keys[value.id].items():
                if k == _UNKNOWN:
                    self.unknown_out = True
                else:
                    self.out_keys.setdefault(k, set()).update(v | control)
        else:
            self.unknown_out = True

    def reached(self) -> set[str]:
        """The settings that can decide anything the function returns, or whether it returns."""
        out = set(self.control)
        for value, control in self.returns:
            out |= self.deps(value) | control
        return out


def _flow(source: str) -> _Flow | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    func = _entry(tree)
    if func is None:
        return None
    flow = _Flow(func)
    return flow if flow.ok and not flow.escaped and flow.returns else None


def unread_settings(source: str, keys: Iterable[str]) -> list[str] | None:
    """The settings among ``keys`` that never reach what the simulation's entry function returns, read from its text, or
    ``None`` when that cannot be said with certainty (no single ``run_cell`` / ``run_trial``, a setting handed to another
    function or stored, a name looked up at run time)."""
    flow = _flow(source)
    if flow is None:
        return None
    reached = flow.reached()
    if _WHOLE in reached:
        return []
    return [k for k in keys if k not in reached]


def unused_outputs(source: str, names: Iterable[str]) -> list[str] | None:
    """The returned quantities among ``names`` whose value is built from no setting, read from the text of the entry
    function: a returned mapping whose names are written out, with that name's value made from nothing the function was
    given. ``None`` when the code cannot be followed with certainty (a returned value that is not such a mapping, a name
    computed at run time, a setting handed to another function)."""
    flow = _flow(source)
    if flow is None or flow.unknown_out or not flow.out_keys:
        return None
    return [n for n in names if n in flow.out_keys and not (flow.out_keys[n] | flow.control)]
