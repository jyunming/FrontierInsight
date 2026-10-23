"""A second implementation must be a second implementation.

A check of kind ``second_implementation`` says that two independent ways of computing one quantity agree. When the code
"compares" two values that are one operation on the same data written two ways (``sum(a)`` against
``einsum("i->", a)``), the difference is zero by construction and the agreement shows nothing. This reads the code with
``ast`` (it is never run), finds every comparison of two values (``abs(a - b)``, ``(a - b) ** 2``, ``norm(a - b)``,
``allclose(a, b)``, ``isclose(a, b)``), follows each side back through the names assigned once and through functions
whose only statement returns an expression, and flags a comparison whose two sides reduce to the same expression.

Two spellings of one reduction are the same (``sum`` / ``fsum`` / ``.sum()`` / ``add.reduce`` / a single-operand
``einsum`` with no output indices; the same for ``mean``, ``max``, ``min``, ``prod``), so is a function copied under another
name. Two different functions that compute the same quantity by different routes are not the same: only a comparison that
cannot differ is flagged. The check applies only to a plan that declares a ``second_implementation`` check.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any

from . import grid_keys as _grid_keys
from . import oracle_check as _oracle
from .stop_power import READ

#: What a fault found here rests on (core/stop_power.py): FI's reading of the script's text.
SOURCE = READ

_ALIASES = frozenset({"np", "numpy", "math", "scipy", "linalg", "jnp", "torch"})
_FAMILY = {
    "sum": "SUM", "fsum": "SUM", "nansum": "SUM", "add.reduce": "SUM",
    "mean": "MEAN", "average": "MEAN", "nanmean": "MEAN",
    "max": "MAX", "amax": "MAX", "nanmax": "MAX", "min": "MIN", "amin": "MIN", "nanmin": "MIN",
    "prod": "PROD", "nanprod": "PROD",
}
#: Keywords that do not change what is computed.
_IGNORED_KEYWORDS = frozenset({"dtype", "out", "optimize", "keepdims", "casting", "order"})
_DIFFERENCE_CALLS = frozenset({"allclose", "isclose", "assert_allclose", "assert_array_almost_equal", "array_equal"})
_NORMS = frozenset({"norm", "fabs", "abs"})
_DEPTH = 6


@dataclass(frozen=True)
class Tautology:
    """One comparison of two values that are the same expression."""

    script: str
    line: int
    function: str
    left: str
    right: str
    expression: str

    def says(self, supplied: list[str] | None = None) -> str:
        return (f"{self.script} line {self.line} (in `{self.function}`): compares `{self.left}` with `{self.right}`, which are "
                f"the same operation on the same data, so it cannot differ")


def declared(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The protocol's checks of kind ``second_implementation``."""
    return [o for o in _oracle.declared(protocol) if _oracle.kind_of(o) == "second_implementation"]


def _short(node: ast.AST) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # noqa: BLE001
        return "?"
    return text if len(text) <= 60 else text[:57] + "..."


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _callee(node: ast.Call) -> str:
    """The function's name without its library prefix (``np.linalg.norm`` -> ``norm``; ``np.add.reduce`` -> ``add.reduce``)."""
    parts = [p for p in _dotted(node.func).split(".") if p]
    while parts and parts[0] in _ALIASES:
        parts = parts[1:]
    return ".".join(parts[-2:] if parts[-2:] == ["add", "reduce"] else parts[-1:])


class _Functions:
    """The functions defined in the code, to read through the ones that only return an expression."""

    def __init__(self, trees: dict[str, ast.Module]) -> None:
        self.by_name: dict[str, ast.FunctionDef] = {}
        for tree in trees.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.by_name.setdefault(node.name, node)

    @staticmethod
    def single_return(fn: ast.FunctionDef) -> ast.expr | None:
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        return body[0].value if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None else None

    @staticmethod
    def fingerprint(fn: ast.FunctionDef) -> str:
        """A copy of a function under another name has the same fingerprint (its name and parameter names left out)."""
        params = {a.arg: f"p{i}" for i, a in enumerate(fn.args.args)}
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]

        class Rename(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
                return ast.copy_location(ast.Name(id=params.get(node.id, node.id), ctx=node.ctx), node)

        dumped = "|".join(ast.dump(Rename().visit(ast.parse(ast.unparse(s)).body[0])) for s in body)
        return hashlib.sha1(dumped.encode("utf-8")).hexdigest()[:12]


class _Canon:
    """An expression reduced to a form in which two spellings of one operation are equal."""

    def __init__(self, functions: _Functions, assigned: dict[str, ast.expr], changing: set[str]) -> None:
        self.functions, self.assigned, self.changing = functions, assigned, changing

    def __call__(self, node: ast.AST, depth: int = 0, bound: dict[str, str] | None = None) -> str:
        bound = bound or {}
        if depth > _DEPTH:
            return "?"
        if isinstance(node, ast.Name):
            if node.id in bound:
                return bound[node.id]
            if node.id in self.assigned and depth < _DEPTH:
                return self(self.assigned[node.id], depth + 1, bound)
            if node.id in self.changing:
                # A name the function sets more than once holds different values at different places: two uses are two values.
                return f"${node.id}@{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"
            return f"${node.id}"
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Call):
            return self.call(node, depth, bound)
        if isinstance(node, ast.BinOp):
            return f"({self(node.left, depth + 1, bound)} {type(node.op).__name__} {self(node.right, depth + 1, bound)})"
        if isinstance(node, ast.UnaryOp):
            return f"({type(node.op).__name__} {self(node.operand, depth + 1, bound)})"
        if isinstance(node, ast.Subscript):
            return f"{self(node.value, depth + 1, bound)}[{self(node.slice, depth + 1, bound)}]"
        if isinstance(node, ast.Attribute):
            return f"{self(node.value, depth + 1, bound)}.{node.attr}"
        if isinstance(node, (ast.Tuple, ast.List)):
            return "[" + ",".join(self(e, depth + 1, bound) for e in node.elts) + "]"
        return ast.dump(node)

    def call(self, node: ast.Call, depth: int, bound: dict[str, str]) -> str:
        name = _callee(node)
        args = [self(a, depth + 1, bound) for a in node.args]
        keywords = sorted(f"{k.arg}={self(k.value, depth + 1, bound)}" for k in node.keywords
                          if k.arg and k.arg not in _IGNORED_KEYWORDS)
        # ``a.sum()``: the method form of a reduction of ``a``.
        if isinstance(node.func, ast.Attribute) and node.func.attr in _FAMILY and not _dotted(node.func).split(".")[0] in _ALIASES:
            return f"{_FAMILY[node.func.attr]}({self(node.func.value, depth + 1, bound)}{''.join(',' + k for k in keywords)})"
        if name == "einsum" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            spec = node.args[0].value.replace(" ", "")
            if "->" in spec and spec.split("->")[1] == "" and len(node.args) == 2 and "," not in spec:
                return f"SUM({args[1]}{''.join(',' + k for k in keywords if not k.startswith('axis'))})"
        if name in _FAMILY:
            return f"{_FAMILY[name]}({','.join(args)}{''.join(',' + k for k in keywords)})"
        fn = self.functions.by_name.get(name) if isinstance(node.func, ast.Name) else None
        if fn is not None and depth < _DEPTH:
            returned = self.functions.single_return(fn)
            params = [a.arg for a in fn.args.args]
            if returned is not None and len(params) >= len(args):
                inner = {p: a for p, a in zip(params, args)}
                return self(returned, depth + 1, inner)
            return f"user:{self.functions.fingerprint(fn)}({','.join(args)}{''.join(',' + k for k in keywords)})"
        return f"{name}({','.join(args)}{''.join(',' + k for k in keywords)})"


def _once_assigned(fn: ast.AST) -> tuple[dict[str, ast.expr], set[str]]:
    """``(the names a function assigns exactly once, in a straight line (not in a branch or a loop), with their value; the
    names it sets more than once or in a loop)``."""
    counts: dict[str, int] = {}
    straight: dict[str, ast.expr] = {}
    top = {id(s) for s in getattr(fn, "body", [])}
    for node in ast.walk(fn):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.comprehension, ast.NamedExpr)):
            targets = [node.target]
        elif isinstance(node, (ast.With,)):
            targets = [i.optional_vars for i in node.items if i.optional_vars is not None]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    counts[n.id] = counts.get(n.id, 0) + 1
        if isinstance(node, ast.Assign) and id(node) in top and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            straight[node.targets[0].id] = node.value
    return ({n: v for n, v in straight.items() if counts.get(n) == 1}, {n for n, c in counts.items() if c > 1 or n not in straight})


def _sides(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """The two sides a comparison node compares, else ``None``."""
    if isinstance(node, ast.Call):
        name = _callee(node)
        if name in _DIFFERENCE_CALLS and len(node.args) >= 2:
            return node.args[0], node.args[1]
        if name in _NORMS and len(node.args) == 1:
            inner = node.args[0]
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub):
                return inner.left, inner.right
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and isinstance(node.left, ast.BinOp)
            and isinstance(node.left.op, ast.Sub)):
        return node.left.left, node.left.right
    return None


def tautologies(files: dict[str, str]) -> list[Tautology]:
    """Every comparison in ``files`` whose two sides reduce to the same expression. Never raises."""
    trees: dict[str, ast.Module] = {}
    for script, text in files.items():
        try:
            trees[script] = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
    functions = _Functions(trees)
    out: list[Tautology] = []
    for script, tree in trees.items():
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            canon = _Canon(functions, *_once_assigned(fn))
            seen: set[int] = set()
            for node in ast.walk(fn):
                sides = _sides(node)
                if sides is None or id(node) in seen:
                    continue
                left, right = canon(sides[0]), canon(sides[1])
                # Two bare names are two things the function was given or built elsewhere: only a comparison whose sides
                # were built from something is read.
                if left == right and not left.startswith("$") and left not in ("?",) and "(" in left:
                    seen.add(id(node))
                    out.append(Tautology(script=script, line=int(getattr(node, "lineno", 0)), function=fn.name,
                                         left=_short(sides[0]), right=_short(sides[1]), expression=_short(node)))
    return sorted(out, key=lambda t: (t.script, t.line))


def problems(files: dict[str, str], protocol: dict[str, Any] | None, design: dict[str, Any] | None = None) -> list[Tautology]:
    """The tautological comparisons in ``files``, for a plan that declares a second-implementation check (else none)."""
    if not declared(protocol):
        return []
    return tautologies(files)


def directive(found: list[Tautology], protocol: dict[str, Any] | None) -> str:
    lines = "\n".join(f"- {t.says()}" for t in found[:6])
    names = ", ".join(f"`{o.get('name')}`" for o in declared(protocol)[:3])
    return (
        "FI read the code before running it, and a comparison of two implementations compares a value with itself:\n"
        + lines + f"\n\nThe plan's check {names} says two independent ways of computing one quantity agree. Two spellings of one "
        "operation on the same data always agree, so this shows nothing. Write the second implementation as its own function "
        "that computes the compared quantity by the other method the plan names, from the same inputs, without calling or "
        "restating the first, and compare what the two return. Change nothing else."
    )


def plain_note(found: list[Tautology], protocol: dict[str, Any] | None) -> str:
    return ("the code's comparison of two implementations compares a value with itself ("
            + "; ".join(t.says() for t in found[:2]) + ")")


def writer_note(found: list[Tautology], protocol: dict[str, Any] | None) -> str:
    return ("The code's check that two implementations agree compares a value with itself through shared code "
            f"({'; '.join(t.says() for t in found[:2])}), so their agreement is not independent validation. Do not present it "
            "as validation; say plainly, in the limitations, that the two implementations were not independent.")


def gap(found: list[Tautology], protocol: dict[str, Any] | None) -> str:
    return (f"the second-implementation check compares a value with itself through shared code ({found[0].says()}), so it is "
            "not independent validation")


def record(found: list[Tautology], protocol: dict[str, Any] | None) -> dict[str, Any]:
    return {"comparisons": [{"script": t.script, "line": t.line, "function": t.function, "left": t.left, "right": t.right}
                            for t in found], "gap": gap(found, protocol) if found else ""}


code_files = _grid_keys.code_files
