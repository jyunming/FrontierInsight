"""A model's numbers come from the plan or from its equations, not from the model's author.

The functions that implement the plan's equations (the ones labelled ``# E1`` ..., and every function of the model's
package) are read with ``ast`` (they are never run). A number written into such a function is fine when the plan gives
it (a fixed setting, a value in an equation or in the topic, a grid value) or when it is a plain numeric that no plan
would state: the allow-list below. A number the plan never gives, in the argument of a smooth function (a centre and a
width inside ``exp``, a decay length, a rolloff), or three or more of them in one function, is a model typed in instead of
computed: a surrogate shaped to give the numbers the author expected. It is reported, and only that, so the check stays
conservative: one stray tolerance in a function is left alone.

The allow-list (stated here and nowhere else; none of it is specific to a field):

* zero and the integers up to 12 (indices, exponents, counts, halving), the integers that are powers of two or of ten
  (grid sizes, unit scales), and ``0.5``, ``0.25``, ``0.75``, ``0.125``, ``1.5``;
* any power of ten from ``1e-15`` to ``1e15`` (unit conversions, guards against dividing by zero, tolerances);
* ``60``, ``180``, ``360``, ``3600``, ``86400`` (angle and time conversions);
* ``pi`` and its simple multiples and fractions (``2 pi``, ``pi / 2``, ``pi / 4``, ``pi / 180``, ``180 / pi``), ``e``, the
  square roots of 2 and 3 and their inverses.

A number the plan gives is matched by value (the same number, or that number as a percent) against every number in the
plan's text, its protocol and the topic. The plan's text is not given to the repair: the request names the code's own
numbers only.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any

from . import code_layout as _layout
from . import grid_keys as _grid_keys
from . import oracle_check as _oracle

#: A smooth function's argument is where a typed shape (a centre, a width, a decay length) sits.
_SMOOTH = frozenset({"exp", "expm1", "tanh", "erf", "erfc", "sinh", "cosh", "log", "log10", "log2", "log1p", "arctan",
                     "atan", "sinc", "sigmoid", "sin", "cos", "tan", "arcsin", "asin", "arccos", "acos"})
#: Calls whose arguments are sizes and positions, not model numbers.
_SHAPE_CALLS = frozenset({"range", "zeros", "ones", "empty", "full", "eye", "reshape", "arange", "tile", "repeat",
                          "enumerate", "len", "default_rng", "seed", "Random", "RandomState", "round", "linspace",
                          "meshgrid", "identity", "diag", "roll", "pad", "squeeze", "expand_dims", "axis", "sum", "mean"})
_UNIT_NUMBERS = frozenset({60.0, 180.0, 360.0, 3600.0, 86400.0, 0.5, 0.25, 0.75, 0.125, 1.5})
_CONSTANTS = [math.pi, 2 * math.pi, math.pi / 2, math.pi / 4, 4 * math.pi, math.pi / 180, 180 / math.pi, math.e,
              math.sqrt(2), math.sqrt(3), 1 / math.sqrt(2), 1 / math.sqrt(3)]
#: Distinct untraceable numbers in one function that make it a clear case on their own.
MANY = 3
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Typed:
    """One function with numbers typed into it that the plan does not give."""

    script: str
    function: str
    line: int
    values: tuple[float, ...]
    shape: tuple[float, ...]

    def says(self, supplied: list[str] | None = None) -> str:
        shown = ", ".join(_fmt(v) for v in self.values[:8]) + (" and more" if len(self.values) > 8 else "")
        return f"{self.script} line {self.line}: `{self.function}` has {shown} written into it"


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def plan_numbers(text: str) -> set[float]:
    """Every number written in ``text`` (a plan, a protocol, a topic)."""
    out: set[float] = set()
    for m in _NUMBER.finditer(text or ""):
        try:
            out.add(abs(float(m.group(0))))
        except ValueError:
            continue
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def simple(value: float) -> bool:
    """Whether ``value`` is a plain numeric that needs no source (the allow-list in the module docstring)."""
    v = abs(float(value))
    if v == 0.0:
        return True
    if v.is_integer():
        n = int(v)
        if n <= 12 or n & (n - 1) == 0 or math.log10(v).is_integer():
            return True
    if math.isfinite(v) and v > 0 and abs(math.log10(v) - round(math.log10(v))) < 1e-12 and abs(round(math.log10(v))) <= 15:
        return True
    return any(_close(v, u) for u in _UNIT_NUMBERS) or any(_close(v, c) for c in _CONSTANTS)


def traceable(value: float, numbers: set[float]) -> bool:
    """Whether the plan gives ``value`` (as written, or as a percent of what is written)."""
    v = abs(float(value))
    return any(_close(v, n) or _close(v * 100.0, n) or _close(v / 100.0, n) for n in numbers)


def _model_functions(files: dict[str, str], protocol: dict[str, Any] | None) -> dict[tuple[str, str], str]:
    """``{(script, qualified function name): "model" | "script"}``: every function of the model's package and every function
    labelled with an equation of the plan is the model; the simulation's other functions are read for shapes only."""
    ids = _oracle.generating_equations(protocol)
    out: dict[tuple[str, str], str] = {}
    packages = {p.split("/")[0] for p in files if "/" in p}
    for script, text in files.items():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        in_package = script.split("/")[0] in packages and "/" in script and not script.endswith("__init__.py")
        labelled: set[str] = set()
        for _line, label, fn in _layout._labels(text):
            if fn and (any(_oracle._labelled_in(eid, label) for eid in ids) if ids else re.search(r"(?<![\w.])E\d+\b", label)):
                labelled.add(fn)
        for _s, _e, name in _layout._functions(tree):
            kind = "model" if in_package or name in labelled else "script"
            if kind == "script" and not script.split("/")[-1] in ("simulate.py",):
                continue
            out[(script, name)] = kind
    return out


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        f = node.func
        return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    return ""


def _in_smooth_call(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    cur = parents.get(id(node))
    while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        if _call_name(cur) in _SMOOTH:
            return True
        cur = parents.get(id(cur))
    return False


def _literals(fn: ast.AST, parents: dict[int, ast.AST]) -> list[tuple[float, int, bool]]:
    """``(value, line, inside a smooth function's argument)`` for every number written in the function, leaving out indices
    and the sizes and positions given to array and loop calls."""
    out: list[tuple[float, int, bool]] = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Subscript) and node is parent.slice:
            continue
        if isinstance(parent, (ast.Slice, ast.Tuple)) and isinstance(parents.get(id(parent)), ast.Subscript):
            continue
        if isinstance(parent, ast.UnaryOp):
            parent = parents.get(id(parent))
        if isinstance(parent, ast.Call) and _call_name(parent) in _SHAPE_CALLS:
            continue
        if isinstance(parent, ast.keyword) and parent.arg in ("axis", "dtype", "ndmin", "ddof", "seed", "size", "num", "maxiter"):
            continue
        out.append((float(node.value), int(getattr(node, "lineno", 0)), _in_smooth_call(node, parents)))
    return out


def find(files: dict[str, str], protocol: dict[str, Any] | None, plan_text: str = "") -> list[Typed]:
    """The functions with typed-in numbers the plan does not give (a clear case: a number inside a smooth function's
    argument, or three or more distinct numbers). ``plan_text``: the plan, its protocol and the topic, as text."""
    numbers = plan_numbers(plan_text)
    scope = _model_functions(files, protocol)
    out: list[Typed] = []
    for script, text in files.items():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        parents = _parents(tree)
        qualified = {name: (s, e) for s, e, name in _layout._functions(tree)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = next((q for q, (s, e) in qualified.items() if q.split(".")[-1] == node.name and s <= node.lineno <= e), node.name)
            kind = scope.get((script, name))
            if kind is None:
                continue
            found = [(v, ln, sm) for v, ln, sm in _literals(node, parents) if not simple(v) and not traceable(v, numbers)]
            values = tuple(dict.fromkeys(v for v, ln, sm in found))
            shape = tuple(dict.fromkeys(v for v, ln, sm in found if sm))
            clear = bool(shape) or (kind == "model" and len(values) >= MANY)
            if clear and values:
                out.append(Typed(script=script, function=name, line=int(node.lineno), values=values, shape=shape))
    return sorted(out, key=lambda t: (t.script, t.line))


def plan_text_of(protocol: dict[str, Any] | None, design: dict[str, Any] | None) -> str:
    """The text the plan's numbers are read from: the context the engine gathered (plan.md, the topic) and the protocol and
    design as they are."""
    import json

    parts = []
    if isinstance(design, dict):
        parts.append(str(design.get("_text") or ""))
        parts.append(json.dumps({k: v for k, v in design.items() if k != "_text"}, default=str))
    parts.append(json.dumps(protocol or {}, default=str))
    return "\n".join(parts)


def problems(files: dict[str, str], protocol: dict[str, Any] | None, design: dict[str, Any] | None = None) -> list[Typed]:
    return find(files, protocol, plan_text_of(protocol, design))


def directive(found: list[Typed], protocol: dict[str, Any] | None) -> str:
    lines = []
    for t in found[:6]:
        lines.append(f"- {t.says()}" + (f" (inside the argument of a smooth function: {', '.join(_fmt(v) for v in t.shape[:4])})"
                                        if t.shape else ""))
    return (
        "FI read the code before running it, and the model's functions have numbers typed into them that the plan never "
        "gives:\n" + "\n".join(lines)
        + "\n\nThese look like a shape written into the code (a centre, a width, a decay length, a rolloff) rather than "
        "computed. Compute each from the model's equations, or state it in the plan as a fixed setting with its source and "
        "take it from there. Do not replace one typed number with another, and change nothing else."
    )


def plain_note(found: list[Typed], protocol: dict[str, Any] | None) -> str:
    return ("the model's functions have numbers typed into them that the plan does not give ("
            + "; ".join(t.says() for t in found[:3]) + ")")


def writer_note(found: list[Typed], protocol: dict[str, Any] | None) -> str:
    where = "; ".join(t.says() for t in found[:3])
    return (f"The model's code has numbers written into it that the plan does not give ({where}). Say plainly, in the "
            "limitations, that the result depends on numbers that are not in the plan and were not derived from the model's "
            "equations, and do not describe the model as computed from the plan alone.")


def gap(found: list[Typed], protocol: dict[str, Any] | None) -> str:
    return ("the model's code has numbers typed into it that the plan does not give "
            f"({found[0].says()}), so the result rests on a model the plan does not describe")


def record(found: list[Typed], protocol: dict[str, Any] | None) -> dict[str, Any]:
    return {"functions": [{"script": t.script, "function": t.function, "line": t.line, "numbers": list(t.values),
                           "in_smooth_call": list(t.shape)} for t in found], "gap": gap(found, protocol) if found else ""}


# ---- does a headline result rest on it? ---------------------------------------------------------------------------------


def _called(fn: ast.AST) -> set[str]:
    return {_call_name(n) for n in ast.walk(fn) if isinstance(n, ast.Call) and _call_name(n)}


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def tainted_outputs(files: dict[str, str], found: list[Typed]) -> set[str]:
    """The keys ``run_cell`` / ``run_trial`` return whose value comes, through assignments and calls, from a function in
    ``found``."""
    functions: dict[str, ast.AST] = {}
    for text in files.values():
        try:
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.setdefault(node.name, node)
        except (SyntaxError, ValueError):
            continue
    affected = {t.function.split(".")[-1] for t in found}
    grew = True
    while grew:
        grew = False
        for name, fn in functions.items():
            if name not in affected and _called(fn) & affected:
                affected.add(name)
                grew = True
    keys: set[str] = set()
    for entry in ("run_cell", "run_trial"):
        fn = functions.get(entry)
        if fn is None:
            continue
        tainted: set[str] = set()
        for _ in range(8):
            before = len(tainted)
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and (_called(node.value) & affected or _names_in(node.value) & tainted):
                    tainted |= {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if len(tainted) == before:
                break
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                for k, v in zip(node.value.keys, node.value.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and (
                            _called(v) & affected or _names_in(v) & tainted):
                        keys.add(k.value)
    return keys


def headline_hit(found: list[Typed], files: dict[str, str], protocol: dict[str, Any] | None,
                 design: dict[str, Any] | None = None) -> bool:
    """Whether a quantity the plan calls the main one is computed through a function in ``found``."""
    from . import code_review as _review

    heads = {str(r.get("name") or "").strip().lower() for r in _review.requirements(design or {}, protocol or {})
             if r.get("headline") and str(r.get("name") or "").strip()}
    if not heads:
        return False
    outputs = {k.lower() for k in tainted_outputs(files, found)}
    return any(h in o or o in h for h in heads for o in outputs)


code_files = _grid_keys.code_files
