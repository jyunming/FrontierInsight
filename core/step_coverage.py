"""Is each number the simulation reports computed through the model's package?

The plan's model is written as a package of functions (one per equation and per step). The simulation (``run_cell`` /
``run_trial``) should compute what it reports by calling them. A metric that is returned as a number written into the
code, or computed without a call to any function of the package, is a number the tested functions do not stand behind.

Reads syntax trees only. Returns plain sentences (warnings); nothing here stops a quest. A metric whose value cannot be
told from the syntax (a dictionary built in a loop, a ``**`` unpacking) is not reported on.

Pure: no model calls, no engine state.
"""

from __future__ import annotations

import ast
from typing import Any

ENTRY_POINTS = ("run_cell", "run_trial")
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


def _reaching(simulation: ast.Module, package: set[str]) -> set[str]:
    """The simulation's own functions that call a function of the package, directly or through another of its functions."""
    functions = _functions(simulation)
    reaching = {name for name, fn in functions.items() if _called_names(fn) & package}
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


def check(simulation_source: str, package_functions: Any, metrics: Any) -> list[str]:
    """Plain warnings, one per protocol metric that the simulation's ``run_cell`` / ``run_trial`` returns but does not compute
    through the package: ``package_functions`` are the names of the functions of the model's package, ``metrics`` the
    protocol's metrics (ids or ``{"id": ...}`` entries). Empty when each is reached, or when nothing can be told."""
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
    reaching = _reaching(tree, package)
    through = package | reaching
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
            if not unreadable:
                warnings.append(f"the metric `{name}` is not among the numbers `{entry_names}` returns, so nothing in the "
                                "simulation computes it")
            continue
        parts = expanded[name]
        if any(_called_names(p) & through for p in parts):
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
