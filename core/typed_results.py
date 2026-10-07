"""A number the simulation reports must be computed, not typed into the code.

``typed_results`` reads the simulation with ``ast`` (it is never run) and finds, in the functions FI calls for results
(``run_trial``, ``run_cell`` and ``oracle`` of the trial contract, ``core/trial_runner.py``), every returned key whose
value is a number written into the code: ``{"count": 0.0}``, or a name that is only ever assigned such a number in a
straight line of the function and never changed after (``p = 0.0 ... return {"p": p}``). The same value is then
reported as a measurement for every trial, which nothing was measured to say.

The check is conservative: when it cannot be sure it flags nothing. A value built from the function's inputs
(``cell[...]``, the trial number, the seed), from a call, from an attribute or a subscript, a name assigned more than
once, assigned inside a branch or a loop, changed by ``+=`` or by a loop, or a dictionary that is built up in ways the
check cannot follow (``update``, ``**``, a call that is given it) is not flagged, and a function that returns something
the check cannot read (a call's result) is left alone as a whole. A key that is also returned with a computed value
in another ``return`` is not flagged. Two kinds of key are exempt: the plan's own fixed settings, the names of the
grid's settings and the thresholds (values a simulation echoes back because the plan fixed them, not results), and a
module-level constant returned under its own name (``LIMIT`` as ``limit``: a setting echoed back). A module constant
returned under another name, a number made by ``float(0)`` or ``np.float64(0.0)``, and a call to a helper whose only
statement returns a number are typed in like a literal. A counter the code really computes (``n = 0; for ...: n += 1``)
is computed, not typed, and passes.

A simulation with such a key is sent back once or twice to compute it from the simulation or to leave it out. When it
still has one, that quantity is left out of the results the paper may use (:func:`strip_keys`), and the paper says so
in its limitations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from . import figure_data_check as _figdata

#: The functions FI calls for results (see ``core/trial_runner.py``).
ENTRIES = ("run_trial", "run_cell", "oracle")
#: Calls whose result is a constant when every argument is: the builtins, and the scalar constructors of numpy
#: (``np.float64(0.0)``, ``numpy.int32(3)``, or one imported by name).
_PURE = frozenset({"float", "int", "abs", "round", "min", "max"})
_SCALARS = frozenset({
    "float64", "float32", "float16", "float_", "double", "single", "longdouble", "int8", "int16", "int32", "int64", "intp",
    "int_", "intc", "longlong", "uint8", "uint16", "uint32", "uint64", "uintp", "uint", "ubyte", "short", "ushort",
})
_NUMPY = frozenset({"np", "numpy"})
#: What FI reads from a simulation and a repair of it must therefore keep.
KEEP_MARKS = ("RESULT_JSON", "FI_ORACLE", "FI_REPLICATE_SEED", "FI_TRIALS", "FI_RAW_DIR", "def run_cell", "def run_trial",
              "def oracle")


@dataclass
class TypedResult:
    """One returned key whose value is a number typed into the code."""

    script: str
    function: str
    key: str
    line: int
    value: str

    def says(self) -> str:
        return f"{self.script} line {self.line}: `{self.key}` is returned as {self.value}, a number typed into the code"


def _number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def _functions(tree: ast.Module, names: tuple[str, ...]) -> list[ast.FunctionDef]:
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]


def _own_nodes(fn: ast.AST):
    """Every node of ``fn``'s body, not those of a function, class or lambda nested in it."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _module_names(tree: ast.Module) -> dict[str, ast.AST]:
    """What a function may name without defining it and still hold a number written into the code: a module-level name
    bound once to a number expression (``LIMIT = 5.0``), and a module-level function with no parameters whose only
    statement is ``return <number expression>`` (looked up as ``"name()"``). A helper that takes an argument, or computes from anything, is not followed."""
    out: dict[str, ast.AST] = {}
    bindings = _figdata._scope_bindings(tree)
    for name, values in bindings.items():
        if len(values) == 1 and values[0] is not None and _constant(values[0], out):
            out[name] = values[0]
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name in ENTRIES:
            continue
        a = node.args
        if a.posonlyargs or a.args or a.kwonlyargs or a.vararg or a.kwarg or node.decorator_list:
            continue
        body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None \
                and _constant(body[0].value, out):
            out[f"{node.name}()"] = body[0].value
    return out


def _straight_constants(
    fn: ast.FunctionDef, outer: dict[str, ast.AST] | None = None,
) -> tuple[dict[str, ast.AST], set[str]]:
    """The names of ``fn`` that are bound only by plain ``name = <number expression>`` statements in the top level of its
    body (never in a branch, a loop or a ``try``), all to the same value, and nowhere else (not an argument, a loop
    variable, ``+=``, a ``with`` target, an import ...). Order matters not: a name given the same number every time it is
    bound holds that number whenever it is read after its first binding."""
    every = _figdata._scope_bindings(fn)
    straight: dict[str, list[ast.AST]] = {}
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    straight.setdefault(target.id, []).append(stmt.value)
    # The module's constants and helpers, except a name the function binds itself.
    found: dict[str, ast.AST] = {k: v for k, v in (outer or {}).items() if k not in every}
    outer_names = set(found)
    # A value built from names that are themselves constants (``a = 2.0; b = a * 3``) is read in order.
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            values = straight.get(name, [])
            if name in found or len(values) != len(every.get(name, [])):
                continue
            if all(_constant(v, found) for v in values) and len({ast.dump(v) for v in values}) == 1:
                found[name] = values[0]
    return found, outer_names


def _constant(node: ast.AST, names: dict[str, ast.AST]) -> bool:
    """Whether ``node`` is a number written into the code: a number, arithmetic on such numbers, ``float(...)``-like calls
    on them, or a name in ``names`` (the constants of the function)."""
    if _number(node):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _constant(node.operand, names)
    if isinstance(node, ast.BinOp):
        return _constant(node.left, names) and _constant(node.right, names)
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        if isinstance(func, ast.Name):
            if f"{func.id}()" in names and not node.args:
                return True  # a helper whose only statement returns a number written into the code
            if (func.id in _PURE or func.id in _SCALARS) and node.args:
                return all(_constant(a, names) for a in node.args)
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in _NUMPY
              and func.attr in _SCALARS and node.args):
            return all(_constant(a, names) for a in node.args)
    return False


def _text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- only for the sentence
        return "a number"


class _Unreadable(Exception):
    """The function returns something the check cannot follow: nothing in it is flagged."""


def _returned_entries(fn: ast.FunctionDef) -> list[tuple[str, ast.AST, bool]]:
    """``(key, value expression, written in a straight line of the function)`` for every key every ``return`` of ``fn`` gives
    in a dictionary the check can follow. Raises :class:`_Unreadable` for a return it cannot follow."""
    parents: dict[int, ast.AST] = {}
    for node in _own_nodes(fn):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    top_level = {id(s) for s in fn.body}
    out: list[tuple[str, ast.AST, bool]] = []
    returns = [n for n in _own_nodes(fn) if isinstance(n, ast.Return)]
    if not returns:
        raise _Unreadable
    bindings = _figdata._scope_bindings(fn)

    def from_dict(node: ast.Dict, straight: bool) -> None:
        for k, v in zip(node.keys, node.values):
            if k is None:  # ``**other``: it may override or add any key
                raise _Unreadable
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                raise _Unreadable
            out.append((k.value, v, straight))

    def from_call(node: ast.Call) -> None:
        if node.args:
            raise _Unreadable
        for kw in node.keywords:
            if kw.arg is None:
                raise _Unreadable
            out.append((kw.arg, kw.value, True))

    def is_dict_call(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"

    for ret in returns:
        value = ret.value
        if isinstance(value, ast.Dict):
            from_dict(value, True)
        elif is_dict_call(value):
            from_call(value)  # type: ignore[arg-type]
        elif isinstance(value, ast.Name):
            name = value.id
            made = bindings.get(name, [])
            if len(made) != 1 or not (isinstance(made[0], ast.Dict) or is_dict_call(made[0])):
                raise _Unreadable
            owner = next((s for s in _own_nodes(fn) if isinstance(s, ast.Assign) and s.value is made[0]), None)
            straight = owner is not None and id(owner) in top_level
            if isinstance(made[0], ast.Dict):
                from_dict(made[0], straight)
            else:
                from_call(made[0])  # type: ignore[arg-type]
            # Every other way the name is used must be a read or a write of one key with a string.
            for node in _own_nodes(fn):
                if not (isinstance(node, ast.Name) and node.id == name):
                    continue
                parent = parents.get(id(node))
                if isinstance(parent, ast.Return) or (isinstance(parent, ast.Assign) and parent.value is made[0]):
                    continue
                if (isinstance(parent, ast.Assign) and node in parent.targets):
                    continue
                if (isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.slice, ast.Constant)
                        and isinstance(parent.slice.value, str)):
                    holder = parents.get(id(parent))
                    key = parent.slice.value
                    if isinstance(parent.ctx, ast.Store):
                        if isinstance(holder, ast.Assign) and parent in holder.targets and len(holder.targets) == 1:
                            out.append((key, holder.value, id(holder) in top_level))
                        else:
                            # ``+=``, an unpacking or a loop target: the key's value is computed.
                            out.append((key, ast.Call(func=ast.Name(id="_computed", ctx=ast.Load()), args=[], keywords=[]),
                                        False))
                    elif isinstance(parent.ctx, ast.Del):
                        raise _Unreadable
                    continue
                raise _Unreadable  # passed on, updated, spread, iterated ...: the check cannot follow it
        else:
            raise _Unreadable
    return out


def typed_results(source: str, script: str = "simulate.py", *, exempt: set[str] | frozenset[str] = frozenset(),
                  entries: tuple[str, ...] = ENTRIES) -> list[TypedResult]:
    """The returned keys of ``source`` whose value is a number typed into the code (see the module docstring). ``exempt``:
    key names that are the plan's own fixed settings, the grid's settings or the thresholds. Never raises: a script that does
    not parse has none to report (it fails on its own)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: list[TypedResult] = []
    outer = _module_names(tree)
    for fn in _functions(tree, entries):
        try:
            returned = _returned_entries(fn)
        except _Unreadable:
            continue
        names, module_names = _straight_constants(fn, outer)
        by_key: dict[str, list[tuple[ast.AST, bool]]] = {}
        written_straight: dict[str, list[bool]] = {}
        for key, value, straight in returned:
            by_key.setdefault(key, []).append((value, straight))
            written_straight.setdefault(key, []).append(straight)
        for key, occurrences in by_key.items():
            if key in exempt:
                continue
            # A module constant returned under its own name (``LIMIT`` as `limit`) is a setting echoed back, not a result.
            if any(isinstance(v, ast.Name) and v.id in module_names and v.id.lower() == key.lower() for v, _ in occurrences):
                continue
            if not all(_constant(value, names) for value, _ in occurrences):
                continue
            # One value everywhere it is written, and never written in a branch or a loop (which would make it depend on one).
            if len({_resolved(value, names) for value, _ in occurrences}) != 1:
                continue
            if not all(straight for straight in written_straight[key]):
                continue
            first = min(occurrences, key=lambda o: getattr(o[0], "lineno", 0))[0]
            found.append(TypedResult(script=script, function=fn.name, key=key, line=int(getattr(first, "lineno", fn.lineno)),
                                     value=_text(first)))
    return sorted(found, key=lambda r: (r.line, r.key))


def _resolved(node: ast.AST, names: dict[str, ast.AST]) -> str:
    """The text of a constant expression with its constant names written out, so two spellings of one number agree."""
    if isinstance(node, ast.Name) and node.id in names:
        return _resolved(names[node.id], names)
    if isinstance(node, ast.BinOp):
        return f"({_resolved(node.left, names)}{type(node.op).__name__}{_resolved(node.right, names)})"
    if isinstance(node, ast.UnaryOp):
        return f"({type(node.op).__name__}{_resolved(node.operand, names)})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return f"{node.func.id}({','.join(_resolved(a, names) for a in node.args)})"
    return ast.dump(node)


def directive(findings: list[TypedResult]) -> str:
    """What to tell a repair about results typed into the code: plain, naming each key and line."""
    lines = [f"- {f.says()}" for f in findings[:12]]
    keys = ", ".join(f"`{f.key}`" for f in findings[:12])
    return (
        "A RESULT IS A NUMBER TYPED INTO THE CODE: what the simulation returns is reported as measured, so each returned "
        "value must be computed by the simulation from its inputs, never written into the code. These returned values are "
        "numbers written into the code:\n" + "\n".join(lines)
        + f"\n\nFor each of {keys}: compute it from the simulation (the state it ends in, the events it counted, the "
        "quantity it measured), or remove it from what is returned if the simulation does not compute it. If the true value "
        "is a constant that the simulation computes, compute it; if it is a setting, return it under the setting's own name. "
        "Do not replace one "
        "typed number with another, and do not change anything else (the model's equations, the other returned values, the "
        "signatures, the printed RESULT_JSON). Return the whole corrected script."
    )


def plain_note(keys: list[str], where: list[TypedResult]) -> str:
    """The sentence for the paper's limitations and for run.log when quantities were left out."""
    names = ", ".join(f"`{k}`" for k in keys)
    calls = "; ".join(f.says() for f in where[:4])
    return (f"{names} {'was' if len(keys) == 1 else 'were'} left out of the results: the simulation returned "
            f"{'a number' if len(keys) == 1 else 'numbers'} typed into the code rather than computed ({calls}).")


def record(keys: list[str], where: list[TypedResult]) -> dict[str, Any]:
    return {"removed_quantities": keys, "calls": [f.says() for f in where], "note": plain_note(keys, where)}


def strip_keys(value: Any, keys: set[str] | frozenset[str]) -> Any:
    """``value`` (a parsed RESULT_JSON) without any mapping entry named in ``keys``, at every depth."""
    if isinstance(value, dict):
        return {k: strip_keys(v, keys) for k, v in value.items() if str(k) not in keys}
    if isinstance(value, list):
        return [strip_keys(v, keys) for v in value]
    return value
