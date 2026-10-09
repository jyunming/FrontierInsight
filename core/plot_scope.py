"""What of a script is plotting and what is computation, read from its text.

A repair of a figure may change how the figure is drawn; it may not change a number the study computes. This reads a script
with ``ast`` (it is never run) and splits it:

* a **drawing statement** is one that draws (a call to a plotting method such as ``plot`` or ``savefig``, or anything that
  uses ``plt``, a figure or an axes, or a call of a function that only draws);
* **plot data** is an assignment whose names are read only by drawing statements (``ys = [4, 5, 6]`` followed by
  ``ax.plot(xs, ys)``): it feeds the drawing and nothing else;
* everything else is **computation**: the simulation's functions, what is returned, what is printed as a result.

:func:`computation_print` is the script with the first two taken out, and :func:`computation_changes` names what a rewrite
changed of the third. A rewrite whose computation is the same syntax tree (comments, spacing and the drawing do not count)
changed only the plotting. New functions and new imports that rebind no name the script had are additions, not changes. A
script that does not parse has no reading: any change to it is a change. The reading errs towards "computation": what it
cannot tell is never taken for drawing.
"""

from __future__ import annotations

import ast
from typing import Any, Iterator

from .figure_data_check import PLOT_METHODS

#: Method names that draw or finish a figure without being a data method.
_DRAW_EXTRA = frozenset({"savefig", "show", "close", "subplots", "subplot", "figure", "legend", "colorbar", "tight_layout",
                         "suptitle", "title", "xlabel", "ylabel", "xlim", "ylim", "grid", "text", "annotate", "axhline",
                         "axvline", "axhspan", "axvspan", "clf", "cla", "gca", "gcf", "twinx", "twiny", "subplots_adjust",
                         "set_xscale", "set_yscale", "set_title", "set_xlabel", "set_ylabel", "set_xlim", "set_ylim",
                         "set_xticks", "set_yticks", "tick_params", "set_aspect", "set_size_inches", "add_subplot",
                         "add_axes", "xticks", "yticks", "axis", "margins"})
#: Names that are the drawing library or a figure by convention.
_ROOTS = frozenset({"plt", "pyplot", "matplotlib", "mpl", "pylab", "sns", "seaborn", "fig", "figure", "ax", "axes", "axs",
                    "ax1", "ax2", "ax3", "cax", "gs"})
_DRAW_MODULES = ("matplotlib", "pylab", "seaborn", "mpl_toolkits")
_COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.FunctionDef,
             ast.AsyncFunctionDef, ast.ClassDef)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _is_draw_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return all(a.name.split(".")[0] in _DRAW_MODULES for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] in _DRAW_MODULES
    return False


def _bound(stmt: ast.stmt) -> set[str]:
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}


def _header(stmt: ast.stmt) -> Iterator[ast.AST]:
    """The nodes of a compound statement that are not in its body: its test, iterable, context, decorators and defaults."""
    for name, value in ast.iter_fields(stmt):
        if name in ("body", "orelse", "finalbody", "handlers"):
            continue
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, ast.AST):
                yield from ast.walk(item)


def _statements(body: list[ast.stmt]) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for s in body:
        out.append(s)
        for name in ("body", "orelse", "finalbody"):
            inner = getattr(s, name, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                out.extend(_statements(inner))
        for h in getattr(s, "handlers", None) or []:
            out.extend(_statements(h.body))
    return out


class _Reading:
    """The statements of one script sorted into drawing, plot data and computation."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.plot_vars: set[str] = set()
        self.plot_fns: set[str] = set()
        self.drawing: set[int] = set()  # id() of the statements that draw
        self.data: set[int] = set()  # id() of the statements that are plot data
        self.everything = _statements(tree.body)
        for _ in range(8):  # a figure, then what is built from it, then a function that only draws, then its callers
            before = (len(self.plot_vars), len(self.plot_fns), len(self.drawing), len(self.data))
            for s in self.everything:
                if not isinstance(s, _COMPOUND) and not _is_draw_import(s) and id(s) not in self.drawing \
                        and self._draws(s):
                    self.drawing.add(id(s))
                    self.plot_vars |= _bound(s)
            self._find_data()
            for s in self.everything:
                if isinstance(s, _FUNCTIONS) and s.name not in self.plot_fns and self._only_draws(s):
                    self.plot_fns.add(s.name)
            if before == (len(self.plot_vars), len(self.plot_fns), len(self.drawing), len(self.data)):
                break

    def _draws(self, node: ast.AST) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and (n.attr in PLOT_METHODS or n.attr in _DRAW_EXTRA):
                return True
            if isinstance(n, ast.Name) and (n.id in _ROOTS or n.id in self.plot_vars):
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.plot_fns:
                return True
        return False

    def _only_draws(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """A function that draws and does nothing else: it has a drawing statement, returns nothing, changes no global, and
        every other simple statement of it is plot data."""
        if any(isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom)) for n in ast.walk(fn)):
            return False
        if any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn)):
            return False
        inner = _statements(fn.body)
        if not any(id(s) in self.drawing for s in inner):
            return False
        for s in inner:
            if isinstance(s, _COMPOUND) or id(s) in self.drawing or id(s) in self.data:
                continue
            if isinstance(s, (ast.Pass, ast.Import, ast.ImportFrom)) or (
                    isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)):
                continue
            return False
        return True

    def _find_data(self) -> None:
        """Assignments whose names are read nowhere but by drawing statements (or by other plot data)."""
        readers: dict[str, list[ast.stmt]] = {}
        for s in self.everything:
            for n in (_header(s) if isinstance(s, _COMPOUND) else ast.walk(s)):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    readers.setdefault(n.id, []).append(s)
        grew = True
        while grew:
            grew = False
            for s in self.everything:
                if not isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)) or id(s) in self.drawing \
                        or id(s) in self.data:
                    continue
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                if any(isinstance(n, (ast.Attribute, ast.Subscript)) for t in targets for n in ast.walk(t)):
                    continue  # it changes an object that something else may hold
                names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                reads = [r for n in names for r in readers.get(n, []) if r is not s]
                if names and reads and all(id(r) in self.drawing or id(r) in self.data for r in reads) \
                        and not any(r is s for n in names for r in readers.get(n, [])):
                    self.data.add(id(s))
                    grew = True

    # -- the computation, as comparable text
    def _dump(self, node: Any) -> str:
        """``node`` as text with its drawing statements and plot data left out (at any depth); positions are not in it."""
        if isinstance(node, ast.AST):
            if isinstance(node, ast.stmt) and (id(node) in self.drawing or id(node) in self.data):
                return "-"
            return type(node).__name__ + "(" + ",".join(f"{k}={self._dump(v)}" for k, v in ast.iter_fields(node)) + ")"
        if isinstance(node, list):
            return "[" + ",".join(self._dump(v) for v in node) + "]"
        return repr(node)

    def computation(self) -> tuple[list[tuple[str, str]], dict[str, str]]:
        """``([(what it is, its text) for each top-level item that computes], {name: what an import binds it to})``."""
        items: list[tuple[str, str]] = []
        imports: dict[str, str] = {}
        for s in self.tree.body:
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                if _is_draw_import(s):
                    continue
                for alias in s.names:
                    name = alias.asname or alias.name.split(".")[0]
                    imports[name] = (f"{s.module or ''}.{alias.name}" if isinstance(s, ast.ImportFrom) else alias.name)
            elif isinstance(s, _FUNCTIONS) and s.name in self.plot_fns:
                continue
            elif id(s) in self.drawing or id(s) in self.data:
                continue
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
                continue  # a docstring
            else:
                label = (f"function {s.name}" if isinstance(s, _FUNCTIONS) else f"class {s.name}" if isinstance(s, ast.ClassDef)
                         else "the module: " + _first_line(s))
                items.append((label, self._dump(s)))
        return items, imports


def _first_line(s: ast.stmt) -> str:
    try:
        return ast.unparse(s).splitlines()[0][:60]
    except Exception:  # noqa: BLE001 -- a label is only for the message
        return type(s).__name__


def computation_print(source: str) -> str | None:
    """A text that is the same for two sources exactly when their computation is the same, or ``None`` when the source does
    not parse."""
    try:
        reading = _Reading(ast.parse(source))
    except (SyntaxError, ValueError, RecursionError):
        return None
    items, imports = reading.computation()
    return repr((sorted(items), sorted(imports.items())))


def computation_changes(old: str, new: str) -> list[str]:
    """What ``new`` changed of the computation of ``old``, in words; empty when only the plotting changed (or a function, class
    or import was added without rebinding a name ``old`` had). A source that does not parse is a change."""
    try:
        before, after = _Reading(ast.parse(old)), _Reading(ast.parse(new))
    except (SyntaxError, ValueError, RecursionError):
        return ["the script does not parse"]
    old_items, old_imports = before.computation()
    new_items, new_imports = after.computation()
    remaining = list(new_items)
    problems: list[str] = []
    for item in old_items:
        if item in remaining:
            remaining.remove(item)
        else:
            problems.append(f"{item[0]} was changed or removed")
    old_labels = {label for label, _text in old_items}
    gone = {p[: -len(" was changed or removed")] for p in problems}
    for label, _text in remaining:
        is_new_definition = label.startswith(("function ", "class ")) and label not in old_labels
        if not is_new_definition and label not in gone:
            problems.append(f"{label} was added")
    for name, target in old_imports.items():
        if new_imports.get(name) != target:
            problems.append(f"the import `{name}` was changed or removed")
    return sorted(set(problems))
