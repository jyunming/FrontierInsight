"""A static look at one step function: does it compute anything from its inputs?

A step of the method (``S<n>`` in the plan) is written by a model and tested by a worked example when the plan gave one.
A function that returns a number written into it, or never reads a single parameter, can pass an example by luck (the
example's answer typed in) and still compute nothing. This reads the function's syntax tree only, never runs it, and
returns one plain sentence, or ``""`` when nothing is wrong.

Pure: no model calls, no engine state.
"""

from __future__ import annotations

import ast

_PLACEHOLDER = "not written yet"


def _find(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    names = [x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]]
    names += [x.arg for x in (a.vararg, a.kwarg) if x is not None]
    return names


def _own_returns(fn: ast.AST) -> list[ast.Return]:
    """The function's own ``return`` statements, not those of a function or class defined inside it."""
    found: list[ast.Return] = []
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Return):
            found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _is_number(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_number(node.operand)
    return False


def _literal_only(node: ast.AST | None) -> bool:
    """A number written out, or a tuple, list or mapping of numbers written out."""
    if _is_number(node):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(_literal_only(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.values) and all(k is not None and _literal_only(v) for k, v in zip(node.keys, node.values))
    return False


def check(source: str, name: str) -> str:
    """``""`` when the function ``name`` in ``source`` may compute something, else a plain reason: it returns a number
    written into it, or it never reads any of its parameters. A function that is only a placeholder, one that does not
    parse, or one that does not exist is not judged here (the import check says that)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ""
    fn = _find(tree, name)
    if fn is None:
        return ""
    if any(isinstance(n, ast.Raise) and _PLACEHOLDER in ast.dump(n) for n in ast.walk(fn)):
        return ""
    returns = _own_returns(fn)
    if returns and all(_literal_only(r.value) for r in returns):
        return (f"`{name}` returns a number written into the code, whatever its inputs are; it must compute its result "
                "from its parameters")
    params = _params(fn)
    if not params:
        return ""
    # a nested function reads the parameter through its closure, so the whole body counts
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if "locals" in used or "vars" in used:
        return ""  # reads them all by name; not decidable here
    if not used & set(params):
        return (f"`{name}` never reads any of its parameters ({', '.join(params)}), so its result cannot depend on its "
                "inputs; it must compute its result from them")
    return ""
