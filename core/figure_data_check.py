"""A figure must be drawn from the run's results, not from numbers typed into the plotting code.

``typed_series`` reads a script with ``ast`` (it is never run) and finds every plotting call whose data is a literal
list or tuple of three or more numbers, or a name that is bound only to such a literal: ``ax.plot(xs, [4, 5, 6, 8])`` with
``xs`` a literal list. Axis limits and ticks, reference lines (``axhline``, ``axvline``, a line through two points, a
line ``y = x``), annotations and the colour, size and style arguments are not data and are left alone.

A script with such a call is sent back once or twice to draw the figure from the run's saved results. When it still has
one, the figures it saves are not used: ``figures_to_drop`` says which (the one the call is saved into, when the
script's own ``savefig`` names it, otherwise every figure the script saves), and the paper says so in its limitations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Methods that draw data: the arguments they are given are what the figure shows.
PLOT_METHODS = frozenset({
    "plot", "scatter", "bar", "barh", "errorbar", "fill_between", "fill_betweenx", "step", "stairs", "stem", "hist",
    "semilogx", "semilogy", "loglog", "imshow", "matshow", "contour", "contourf", "pcolormesh", "pcolor", "hexbin",
    "boxplot", "violinplot", "pie", "polar", "plot_surface", "plot_wireframe", "scatter3D", "plot3D", "stackplot",
    "lineplot", "scatterplot", "barplot", "histplot", "boxplot", "violinplot", "heatmap", "pointplot", "regplot",
})
#: Keywords that carry data (everything else, colour, size, label, alpha, style, bins, levels, is not data).
DATA_KEYWORDS = frozenset({
    "x", "y", "z", "height", "width", "bottom", "left", "y1", "y2", "x1", "x2", "yerr", "xerr", "data", "X", "Y", "Z", "C",
    "weights", "values",
})
_ARRAY_MAKERS = frozenset({"array", "asarray", "asanyarray", "list", "tuple"})
MIN_NUMBERS = 3
#: What FI reads from a script and a redraw of its figures must therefore keep.
KEEP_MARKS = ("RESULT_JSON", "FI_ORACLE", "FI_REPLICATE_SEED", "FI_TRIALS", "FI_BEST_DESIGN", "FI_RAW_DIR", "FI_OPTIMISATION",
              "def run_cell", "def run_trial", "def oracle", ".savefig(")
_LINE_METHODS = frozenset({"plot", "step", "semilogx", "semilogy", "loglog", "scatter"})


@dataclass
class TypedSeries:
    """One plotting call that draws numbers typed into the code."""

    script: str
    line: int
    call: str
    argument: str
    count: int
    #: File names the figure is saved as, when the script's own ``savefig`` names it; ``None`` when that cannot be told.
    saved_as: set[str] | None = None

    def says(self) -> str:
        return (f"{self.script} line {self.line}: {self.call}() draws {self.argument}, {self.count} numbers typed into "
                "the code, not the run's results")


@dataclass
class _Scope:
    node: ast.AST
    bindings: dict[str, list[ast.AST | None]] = field(default_factory=dict)


def _numeric(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _numeric(node.operand)
    return False


def _literal_count(node: ast.AST) -> int:
    """How many numbers a literal list or tuple holds (nested ones counted), or 0 when anything in it is not a number
    typed into the code."""
    if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)) and node.args:
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        if name in _ARRAY_MAKERS and all(k.arg == "dtype" for k in node.keywords):
            return _literal_count(node.args[0])
        return 0
    if isinstance(node, (ast.List, ast.Tuple)):
        total = 0
        for item in node.elts:
            if _numeric(item):
                total += 1
            elif isinstance(item, (ast.List, ast.Tuple)):
                inner = _literal_count(item)
                if inner == 0:
                    return 0
                total += inner
            else:
                return 0
        return total
    return 0


def _flat_values(node: ast.AST) -> list[float] | None:
    if isinstance(node, ast.Call) and node.args:
        return _flat_values(node.args[0])
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    out: list[float] = []
    for item in node.elts:
        if _numeric(item):
            try:
                out.append(float(ast.literal_eval(item)))
            except (ValueError, TypeError):
                return None
        else:
            inner = _flat_values(item)
            if inner is None:
                return None
            out.extend(inner)
    return out


def _scope_bindings(scope: ast.AST) -> dict[str, list[ast.AST | None]]:
    """Every way each name is bound inside ``scope`` (not inside a function or class nested in it): the assigned value
    when it is a plain ``name = value``, ``None`` for any other binding (a loop, an unpacking, an argument, ``+=``)."""
    out: dict[str, list[ast.AST | None]] = {}

    def add(name: str, value: ast.AST | None) -> None:
        out.setdefault(name, []).append(value)

    def targets(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            add(node.id, None)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                targets(item)
        elif isinstance(node, ast.Starred):
            targets(node.value)

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = scope.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, *([a.vararg] if a.vararg else []), *([a.kwarg] if a.kwarg else [])]:
            add(arg.arg, None)
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                add(node.name, None)
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    add(target.id, node.value)
                else:
                    targets(target)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
                add(node.target.id, node.value)
            else:
                targets(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    targets(item.optional_vars)
        elif isinstance(node, ast.NamedExpr):
            targets(node.target)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                add((alias.asname or alias.name).split(".")[0], None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name, None)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                add(name, None)
        elif isinstance(node, ast.comprehension):
            targets(node.target)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _typed_name(name: str, scopes: list[dict[str, list[ast.AST | None]]]) -> int:
    """The number of numbers a name holds when every binding of it (in the innermost scope that binds it) is a literal."""
    for bindings in scopes:
        if name in bindings:
            values = bindings[name]
            counts = [(_literal_count(v) if v is not None else 0) for v in values]
            return min(counts) if counts and all(c >= MIN_NUMBERS for c in counts) else 0
    return 0


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else "ax"
        return f"{base}.{func.attr}"
    return getattr(func, "id", "")


def _saves_after(scope_body: list[ast.stmt], line: int) -> set[str] | None:
    """The file names ``savefig`` is given in the same block, after ``line`` (the first one), or ``None`` when no
    ``savefig`` in it names a file with a string the code holds."""
    best: tuple[int, set[str] | None] | None = None
    for stmt in scope_body:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "savefig"
                    and node.lineno >= line and node.args):
                names = {Path(c.value).name for c in ast.walk(node.args[0])
                         if isinstance(c, ast.Constant) and isinstance(c.value, str) and "." in c.value}
                if best is None or node.lineno < best[0]:
                    best = (node.lineno, names or None)
    return best[1] if best else None


def typed_series(source: str, script: str = "experiment.py") -> list[TypedSeries]:
    """The plotting calls in ``source`` that draw numbers typed into the code. Never raises: a script that does not parse
    has none to report (it fails on its own)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    module_bindings = _scope_bindings(tree)
    found: list[TypedSeries] = []

    def visit(scope: ast.AST, chain: list[dict[str, list[ast.AST | None]]], body: list[ast.stmt]) -> None:
        own = _scope_bindings(scope) if scope is not tree else module_bindings
        scopes = [own, *chain] if scope is not tree else [own]
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(node, scopes, node.body)
                continue
            if isinstance(node, (ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PLOT_METHODS:
                _read_call(node, scopes, body)
            stack.extend(ast.iter_child_nodes(node))

    def literal_of(arg: ast.AST, scopes: list[dict[str, list[ast.AST | None]]]) -> tuple[int, list[float] | None]:
        if isinstance(arg, ast.Name):
            n = _typed_name(arg.id, scopes)
            values = None
            for bindings in scopes:
                if arg.id in bindings:
                    only = bindings[arg.id]
                    values = _flat_values(only[0]) if only and only[0] is not None else None
                    break
            return n, values
        n = _literal_count(arg)
        return (n, _flat_values(arg)) if n >= MIN_NUMBERS else (0, None)

    def _read_call(call: ast.Call, scopes: list[dict[str, list[ast.AST | None]]], body: list[ast.stmt]) -> None:
        data: list[tuple[str, ast.AST]] = []
        for i, arg in enumerate(call.args):
            data.append((f"argument {i + 1}", arg))
        for kw in call.keywords:
            if kw.arg in DATA_KEYWORDS:
                data.append((f"`{kw.arg}=`", kw.value))
        typed = [(label, arg, *literal_of(arg, scopes)) for label, arg in data]
        typed = [t for t in typed if t[2] >= MIN_NUMBERS]
        if not typed:
            return
        # The settings along the horizontal axis written out (``ax.plot([1, 2, 4], results["y"])``) are labels for numbers
        # that come from the run: only the numbers drawn against them have to be the run's.
        if [t[0] for t in typed] == ["argument 1"] and len(call.args) >= 2 and not any(
                literal_of(a, scopes)[0] >= MIN_NUMBERS for a in call.args[1:]):
            return
        # A line through the same typed numbers on both axes (y = x) is a reference line, not a result.
        if (call.func.attr in _LINE_METHODS and [t[0] for t in typed[:2]] == ["argument 1", "argument 2"]  # type: ignore[union-attr]
                and typed[0][3] is not None and typed[0][3] == typed[1][3]):
            return
        label, _arg, count, _values = typed[-1]
        found.append(TypedSeries(script=script, line=call.lineno, call=_call_name(call), argument=label, count=count,
                                 saved_as=_saves_after(body, call.lineno)))

    visit(tree, [], tree.body)
    return sorted(found, key=lambda s: s.line)


def figures_to_drop(findings: list[TypedSeries], saved: list[str]) -> list[str]:
    """The saved figure files a script's typed-in series may be in: those the call's own ``savefig`` names, and every saved
    figure when a call's file cannot be told (never fewer than the figure it is in)."""
    if not findings:
        return []
    names = {Path(n).name for n in saved}
    drop: set[str] = set()
    for f in findings:
        if f.saved_as is None:
            return sorted(names)
        hit = {n for n in names if n in f.saved_as}
        if not hit:
            return sorted(names)  # the file the call is saved to is not among this run's figures: cannot tell which
        drop |= hit
    return sorted(drop)


def directive(findings: list[TypedSeries]) -> str:
    """What to tell a repair about the figures drawn from typed-in numbers: plain, naming each call and line."""
    lines = [f"- {f.says()}" for f in findings[:12]]
    return (
        "A FIGURE SHOWS NUMBERS TYPED INTO THE CODE: a figure must be drawn from what the run computed, so a reader can "
        "trust that it shows the results. These plotting calls draw literal lists of numbers:\n" + "\n".join(lines)
        + "\n\nDraw each of these figures from the run's saved results instead: read the values from the raw results files "
        "the simulation wrote (or from the values the analysis already holds), and pass those to the plotting call. Do "
        "not replace one typed list with another, and do not change anything else (the analysis, the printed "
        "RESULT_JSON, the other figures). Reference lines (a limit, a target, y = x) and axis limits may stay as they "
        "are. Return the whole corrected script."
    )


def plain_note(removed: list[str], where: list[TypedSeries]) -> str:
    """The sentence for the paper's limitations when figures were left out, and for run.log."""
    names = ", ".join(f"figures/{n}" for n in removed) or "the figure(s) it saves"
    calls = "; ".join(f.says() for f in where[:4])
    return (f"{names} {'was' if len(removed) == 1 else 'were'} left out: the code that drew "
            f"{'it' if len(removed) == 1 else 'them'} used numbers typed into the code rather than the run's results "
            f"({calls}).")


def record(removed: list[str], where: list[TypedSeries]) -> dict[str, Any]:
    return {"removed_figures": removed, "calls": [f.says() for f in where], "note": plain_note(removed, where)}
