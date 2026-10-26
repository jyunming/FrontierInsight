"""Is each number the simulation reports computed through the model's package?

The plan's model is written as a package of functions (one per equation and per step). The simulation (``run_cell`` /
``run_trial``) should compute what it reports by calling them. A metric that is returned as a number written into the
code, or computed without a call to any function of the package, is a number the tested functions do not stand behind.

Reads syntax trees only. Returns plain sentences (warnings); nothing here stops a quest: the engine records them as a gap
in the evidence record and QUALITY.md (core/run_checks.py). A metric whose value cannot be told from the syntax (a
dictionary built in a loop, a ``**`` unpacking) is not reported on.

What it is for: a ``run_trial`` that returns the plan's true value plus noise, with a standard error that matches the noise,
and never calls the model passes the failure-rate, calibration, seed and typed-result checks. Only the question "is this
number computed through the model's functions?" tells that code from an honest one. It reads syntax, so one call into the
package that does not change the value (a package call multiplied by 0) defeats it; the equation tests and the hidden check
are what stand behind that case.

Pure: no model calls, no engine state.
"""

from __future__ import annotations

import ast
from typing import Any

ENTRY_POINTS = ("run_cell", "run_trial")
#: Where the engine writes what this check found, under ``.fi/``.
RECORD = "step_coverage.json"
#: Methods that put a value into a list, set or mapping the metric is later computed from.
_GROWS = {"append", "extend", "add", "update", "insert"}


def _metric_names(metrics: Any) -> list[str]:
    out: list[str] = []
    if isinstance(metrics, dict):
        metrics = [metrics]
    for m in metrics if isinstance(metrics, list) else []:
        name = (m.get("id") or m.get("name")) if isinstance(m, dict) else m
        if isinstance(name, str) and name.strip() and name.strip() not in out:
            out.append(name.strip())
    return out


def _called_names(node: ast.AST) -> set[str]:
    """The simple name of everything called inside ``node`` (``f(...)`` and ``module.f(...)`` both give ``f``)."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                out.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                out.add(n.func.attr)
    return out


def _loaded_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _shallow(nodes: list[ast.AST]):
    """Every node of ``nodes`` and below, not entering the body of a nested function, class or lambda (the nested scope
    itself is yielded)."""
    stack = list(reversed(nodes))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, _SCOPES):
            stack.extend(reversed(list(ast.iter_child_nodes(n))))


def _params(node: ast.AST) -> set[str]:
    args = getattr(node, "args", None)
    if not isinstance(args, ast.arguments):
        return set()
    names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
    names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
    return set(names)


class _Calls:
    """Which calls in the simulation go to the model's package.

    With ``modules`` (the top-level names of the package's folders) a call counts only when its target is bound to the
    package by an import: ``from pkg.mod import f`` then ``f(...)``, ``import pkg.mod as m`` / ``from pkg import mod`` then
    ``mod.f(...)``. Names are resolved scope by scope: a name the simulation binds itself (a ``def``, a class, an
    assignment, a parameter, another import) is its own inside the function that binds it and every function nested in it,
    whatever the package calls its functions, so an unrelated local ``update`` is not the package's ``update``. Without
    ``modules`` a call counts when its simple name is a function name of the package (the older, looser reading)."""

    def __init__(self, tree: ast.Module, package: set[str], modules: set[str] | None) -> None:
        self.package = package
        self.modules = modules
        self._hits: dict[int, str] = {}
        if modules is not None:
            self._scope(tree.body, set(), set(), set())

    def _bindings(self, body: list[ast.AST], params: set[str]) -> tuple[set[str], set[str], set[str]]:
        """``(names bound plainly, names bound to a package function, names bound to a package module)`` in one scope."""
        assert self.modules is not None
        plain, funcs, aliases = set(params), set(), set()
        for n in _shallow(body):
            if isinstance(n, _SCOPES):
                if not isinstance(n, ast.Lambda):
                    plain.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                plain.add(n.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                plain.add(n.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    bound = a.asname or a.name.split(".")[0]
                    (aliases if a.name.split(".")[0] in self.modules else plain).add(bound)
            elif isinstance(n, ast.ImportFrom):
                from_package = n.level == 0 and (n.module or "").split(".")[0] in self.modules
                for a in n.names:
                    bound = a.asname or a.name
                    if not from_package:
                        plain.add(bound)
                    elif a.name == "*":
                        funcs.update(self.package)
                    elif a.name in self.package:
                        funcs.add(bound)
                    else:
                        aliases.add(bound)
        # A name bound both to the package and plainly in one scope is the scope's own.
        return plain, funcs - plain, aliases - plain

    def _scope(self, body: list[ast.AST], params: set[str], funcs: set[str], aliases: set[str]) -> None:
        plain, new_funcs, new_aliases = self._bindings(body, params)
        funcs = (funcs - plain) | new_funcs
        aliases = (aliases - plain) | new_aliases
        for n in _shallow(body):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name) and f.id in funcs:
                    self._hits[id(n)] = f.id
                elif isinstance(f, ast.Attribute) and f.attr in self.package:
                    root = f.value
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in aliases:
                        self._hits[id(n)] = f.attr
            if isinstance(n, _SCOPES):
                inner = [n.body] if isinstance(n, ast.Lambda) else list(n.body)
                self._scope(inner, _params(n), funcs, aliases)

    def to_package(self, node: ast.AST) -> set[str]:
        """The package functions called inside ``node``."""
        if self.modules is None:
            return _called_names(node) & self.package
        return {self._hits[id(n)] for n in ast.walk(node) if isinstance(n, ast.Call) and id(n) in self._hits}


def _reaching(simulation: ast.Module, calls: _Calls) -> set[str]:
    """The simulation's own functions that call a function of the package, directly or through another of its functions."""
    functions = _functions(simulation)
    reaching = {name for name, fn in functions.items() if calls.to_package(fn)}
    changed = True
    while changed:
        changed = False
        for name, fn in functions.items():
            if name not in reaching and _called_names(fn) & reaching:
                reaching.add(name)
                changed = True
    return reaching


def _string_key(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _returned_values(fn: ast.AST) -> tuple[dict[str, list[ast.AST]], dict[str, list[ast.AST]], bool]:
    """``(metric -> the expressions it is given, variable -> the expressions assigned to it, whether some part of what is
    returned cannot be read from the syntax)``."""
    assigned: dict[str, list[ast.AST]] = {}
    metrics: dict[str, list[ast.AST]] = {}
    dicts: set[str] = set()  # names bound to a dict literal / dict(...)
    unreadable = False

    def dict_items(node: ast.AST) -> list[tuple[str, ast.AST]] | None:
        nonlocal unreadable
        if isinstance(node, ast.Dict):
            items = []
            for k, v in zip(node.keys, node.values):
                key = _string_key(k)
                if key is None:
                    unreadable = True
                else:
                    items.append((key, v))
            return items
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            if node.args:
                unreadable = True
            if any(kw.arg is None for kw in node.keywords):
                unreadable = True
            return [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
        return None

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            # Every name a statement binds depends on its whole right-hand side: ``a, b = f(x)`` (each of a and b comes
            # from f), ``x = y = f()``, and ``d[key] = v`` (what ``d`` holds comes from v, whatever the key).
            for target in node.targets:
                if isinstance(target, (ast.Tuple, ast.List)):
                    for t in ast.walk(target):
                        if isinstance(t, ast.Name):
                            assigned.setdefault(t.id, []).append(node.value)
                elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    assigned.setdefault(target.value.id, []).append(node.value)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assigned.setdefault(target.id, []).append(node.value)
                items = dict_items(node.value)
                if items is not None:
                    dicts.add(target.id)
                    for key, value in items:
                        metrics.setdefault(key, []).append(value)
            elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                  and target.value.id in dicts):
                key = _string_key(target.slice)
                if key is None:
                    unreadable = True
                else:
                    metrics.setdefault(key, []).append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assigned.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.For):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    assigned.setdefault(t.id, []).append(node.iter)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.attr in _GROWS):
            assigned.setdefault(node.func.value.id, []).extend(node.args)
        elif isinstance(node, ast.Return) and node.value is not None:
            items = dict_items(node.value)
            if items is not None:
                for key, value in items:
                    metrics.setdefault(key, []).append(value)
    return metrics, assigned, unreadable


def _expand(expr: ast.AST, assigned: dict[str, list[ast.AST]]) -> list[ast.AST]:
    """``expr`` and, one name at a time, what each variable it reads was assigned (every variable once)."""
    seen: set[str] = set()
    out = [expr]
    queue = [expr]
    while queue:
        for name in _loaded_names(queue.pop()):
            if name not in seen and name in assigned:
                seen.add(name)
                out.extend(assigned[name])
                queue.extend(assigned[name])
    return out


def package_module_names(sources: dict[str, str]) -> list[str]:
    """The top-level folder names of the package files (``{"pkg/model.py": text}``), sorted."""
    return sorted({path.replace("\\", "/").split("/")[0] for path in sources if "/" in path.replace("\\", "/")})


def package_function_names(sources: dict[str, str]) -> list[str]:
    """The names of the functions defined at the top of each file of the model's package (``{path: text}``), sorted."""
    out: set[str] = set()
    for text in sources.values():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        out.update(n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return sorted(out)


def check(simulation_source: str, package_functions: Any, metrics: Any, *, only_returned: bool = False,
          package_modules: Any = None) -> list[str]:
    """Plain warnings, one per protocol metric that the simulation's ``run_cell`` / ``run_trial`` returns but does not compute
    through the package: ``package_functions`` are the names of the functions of the model's package, ``metrics`` the
    protocol's metrics (ids or ``{"id": ...}`` entries). Empty when each is reached, or when nothing can be told.
    With ``only_returned`` a metric the entry point does not return at all is left alone (the analysis script may compute it
    from the trials); only the ones it returns are held to the package. ``package_modules`` (the top-level names of the
    package's folders) makes a call count only when it is bound to the package by an import and not shadowed by a name the
    simulation defines itself; left out, a call counts by its function name alone."""
    names = _metric_names(metrics)
    package = {str(f) for f in (package_functions or [])}
    if not names or not package:
        return []
    try:
        tree = ast.parse(simulation_source)
    except (SyntaxError, ValueError):
        return []
    functions = _functions(tree)
    entries = [functions[n] for n in ENTRY_POINTS if n in functions]
    if not entries:
        return []
    modules = {str(m) for m in package_modules} if package_modules is not None else None
    calls = _Calls(tree, package, modules)
    reaching = _reaching(tree, calls)
    found: dict[str, list[ast.AST]] = {}
    expanded: dict[str, list[ast.AST]] = {}
    unreadable = False
    for fn in entries:
        metrics_here, assigned, bad = _returned_values(fn)
        unreadable = unreadable or bad
        for key, values in metrics_here.items():
            found.setdefault(key, []).extend(values)
            for v in values:
                expanded.setdefault(key, []).extend(_expand(v, assigned))
    warnings: list[str] = []
    entry_names = " / ".join(fn.name for fn in entries)
    for name in names:
        if name not in found:
            if not unreadable and not only_returned:
                warnings.append(f"the metric `{name}` is not among the numbers `{entry_names}` returns, so nothing in the "
                                "simulation computes it")
            continue
        parts = expanded[name]
        if any(calls.to_package(p) or _called_names(p) & reaching for p in parts):
            continue
        if all(_literal_only_expr(p) for p in found[name]):
            warnings.append(f"the metric `{name}` is returned as a number written into the code, not computed")
        else:
            warnings.append(f"the metric `{name}` is computed without calling any function of the model's package "
                            f"({', '.join(sorted(package)[:4])}{', ...' if len(package) > 4 else ''}), so the tested "
                            "functions do not stand behind it")
    return warnings


def _literal_only_expr(node: ast.AST) -> bool:
    """A value built from literals alone (no name read, no call)."""
    for n in ast.walk(node):
        if isinstance(n, (ast.Name, ast.Call, ast.Attribute, ast.Subscript)):
            return False
    return True
