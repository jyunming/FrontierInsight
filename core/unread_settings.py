"""Which settings of the grid the code reads: the one reader FI asks.

FI calls the simulation once per setting of ``protocol.grid``. A grid key the code never reads changes nothing: each of its
values runs the same simulation again, with other seeds. A real plan (first-order kinetics, rounds s1 and r13) kept five
temperatures in the grid beside the noise level and the number of time points while ``run_trial`` built its own five
temperatures: 125 settings ran where the study was 25, and the paper said there was no replication.

Public API (small and kept stable; other modules call only these two):

* :func:`setting_reads` ``(files) -> (names, "")`` or ``(None, why)``: every setting name the code reads, the simulation
  and the analysis together. ``files`` is ``{path relative to code/: text}`` (``core.grid_keys.code_files``).
* :func:`unread_axes` ``(files, protocol) -> (axes, "")`` or ``([], why)``: the grid's settings with two values or more
  that nothing reads.

The code is read with ``ast`` (never run). It answers only when it is certain: any use it cannot follow means "this could
read any name", and then nothing is known (``None``, with the reason). Calling a used setting unread would take it out of
the plan, which is worse than running it again.

What is followed, in the simulation (from the first parameter of ``run_trial`` / ``run_cell``):

* a read by a literal name: ``cell["x"]``, ``cell.get("x", d)``, ``cell.pop("x")``, ``"x" in cell``, in any scope inside the
  function, a ``lambda`` or a nested ``def`` included (a nested scope reads the same setting);
* another name for it: ``c = cell``, ``c = dict(cell)``, ``c = cell.copy()`` (not a ``global`` / ``nonlocal`` name);
* the setting handed whole, by position (with no ``*args`` before it) or by keyword, to a function these files define and
  that has no decorator: ``f(cell)`` when ``f`` is not imported from elsewhere and not bound again at module level,
  ``self.f(cell)``, ``Model().f(cell)`` for a class defined here, and ``mod.f(cell)`` for a module of these files.

In the analysis (any file): FI's record of a setting, ``rec["cell"]`` / ``rec.get("cell")``, and names assigned from it,
followed the same way; a string ``"cell.x"`` (a column of ``pandas.json_normalize``) is a read of ``x``.

Anything else ends the reading with the reason: ``**cell``, ``*args`` before the setting, a computed name, a loop over it,
``cell.items()``, a call to code these files do not define, a method called on a class (``Model.f(m, cell)``) or on an
object FI cannot place, a decorated function, a function name bound again at module level, storing the setting on an object
or in a container or as a default value, a read of FI's key of a setting (``rec["key"]``, which names every setting), and
an ``eval`` / ``exec`` / ``locals`` / ``vars`` in a function that holds the setting or in an analysis file that reads FI's
record of one (``globals()`` cannot reach the setting, which never becomes a module-level name).
"""

from __future__ import annotations

import ast
from typing import Any

from .grid_keys import ENTRIES

_Fn = ast.FunctionDef | ast.AsyncFunctionDef
_Scope = ast.FunctionDef | ast.AsyncFunctionDef | ast.Module
#: Calls on the setting that read one literal name.
_HOLDER_GETS = frozenset({"get", "pop", "setdefault"})
#: Calls that can reach any name by a string: the code cannot be read for which settings it uses.
_OPAQUE = frozenset({"eval", "exec", "locals", "vars", "__import__"})
#: The field of FI's per-setting record that holds the setting, and the one that names it as a string.
_RECORD_CELL, _RECORD_KEY = "cell", "key"


def _const_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _local_modules(files: dict[str, str]) -> set[str]:
    """The dotted module names of these files (``pkg/model.py`` -> ``pkg.model`` and ``pkg``)."""
    out: set[str] = set()
    for rel in files:
        parts = rel.replace("\\", "/").removesuffix(".py").split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for i in range(1, len(parts) + 1):
            out.add(".".join(parts[:i]))
    return out


def _imports(tree: ast.Module, local: set[str]) -> dict[str, bool]:
    """Each name an import binds in this file, and whether it comes from these files (``True``) or from elsewhere."""
    out: dict[str, bool] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound = a.asname or a.name.split(".")[0]
                out[bound] = (a.asname and a.name or a.name.split(".")[0]) in local
        elif isinstance(node, ast.ImportFrom):
            here = node.level > 0 or (node.module or "") in local
            for a in node.names:
                out[a.asname or a.name] = here
    return out


def _module_bound(tree: ast.Module) -> dict[str, int]:
    """How many times each name is bound at module level (a ``def``, a ``class``, an assignment, a loop or ``with``
    target), outside every function and class body."""
    out: dict[str, int] = {}
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = out.get(node.name, 0) + 1
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out[node.id] = out.get(node.id, 0) + 1
        stack.extend(ast.iter_child_nodes(node))
    return out


def _is_record_cell(node: ast.AST) -> bool:
    """``x["cell"]`` or ``x.get("cell")``: FI's record of one setting, as the analysis reads it."""
    if isinstance(node, ast.Subscript) and _const_str(node.slice) == _RECORD_CELL:
        return True
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and bool(node.args) and _const_str(node.args[0]) == _RECORD_CELL)


def _is_record_key(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript) and _const_str(node.slice) == _RECORD_KEY:
        return True
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and bool(node.args) and _const_str(node.args[0]) == _RECORD_KEY)


class _Unknown(Exception):
    """A use the reader cannot follow: what the code reads is not known."""


def setting_reads(files: dict[str, str], entries: tuple[str, ...] = ENTRIES, *,
                  analysis: bool = True) -> tuple[set[str] | None, str]:
    """``(every setting name the code reads, "")``, or ``(None, why)`` when that cannot be known for sure. ``files``: the
    code's files as ``{path relative to code/: text}``, the simulation and the analysis (not the tests). ``analysis=False``:
    what the simulation reads alone (from ``run_trial`` / ``run_cell``), not the analysis's reads of FI's records."""
    trees: dict[str, ast.Module] = {}
    for name, text in files.items():
        try:
            trees[name] = ast.parse(text)
        except (SyntaxError, ValueError):
            return None, f"{name} does not parse"
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if _is_record_key(node):
                return None, (f"{name} reads FI's key of a setting (line {node.lineno}), which names every setting")
    local = _local_modules(files)
    defs: dict[str, list[_Fn]] = {}
    classes: set[str] = set()
    binds_of: dict[int, dict[str, bool]] = {}  # id(function or module) -> the imports of its file
    rebound: set[str] = set()  # a function name bound again at module level, or declared global somewhere
    for tree in trees.values():
        binds = _imports(tree, local)
        binds_of[id(tree)] = binds
        for bound, times in _module_bound(tree).items():
            if times > 1:
                rebound.add(bound)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(node.name, []).append(node)
                binds_of[id(node)] = binds
            elif isinstance(node, ast.ClassDef):
                classes.add(node.name)
            elif isinstance(node, ast.Global):
                rebound.update(node.names)

    def params(fn: _Scope) -> list[str]:
        if isinstance(fn, ast.Module):
            return []
        return [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]

    def callees(func: ast.AST, fn: _Scope) -> tuple[list[_Fn], bool]:
        """``(the functions of these files the call reaches, whether a method is called on an object)``; raises when the
        call may reach code FI cannot read."""
        binds = binds_of.get(id(fn), {})
        if isinstance(func, ast.Name):
            if binds.get(func.id) is False:
                raise _Unknown(f"{func.id}() is imported from elsewhere")
            if func.id in rebound:
                raise _Unknown(f"the name {func.id} is bound again at module level, so FI cannot tell which function it is")
            if func.id in params(fn) or (not isinstance(fn, ast.Module) and any(
                    isinstance(n, ast.Name) and n.id == func.id and isinstance(n.ctx, ast.Store) for n in ast.walk(fn))):
                raise _Unknown(f"{func.id} is a name the function binds itself; it may hold any function")
            found = defs.get(func.id)
            if not found:
                raise _Unknown(f"{func.id}() is not defined in the code")
            return found, False
        if not isinstance(func, ast.Attribute):
            raise _Unknown("it is handed to a call FI cannot read")
        recv = func.value
        if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Name) and recv.func.id in classes \
                and binds.get(recv.func.id) is not False:
            pass  # ``Model().f(cell)``
        elif isinstance(recv, ast.Name) and recv.id in ("self", "cls"):
            pass
        elif isinstance(recv, ast.Name) and binds.get(recv.id) is True:
            found = defs.get(func.attr)
            if not found:
                raise _Unknown(f"{recv.id}.{func.attr}() is not defined in the code")
            return found, False  # a function of a module of these files
        elif isinstance(recv, ast.Name) and recv.id in classes:
            raise _Unknown(f"{recv.id}.{func.attr}() is a method called on its class, with the object passed by hand")
        else:
            raise _Unknown(f".{func.attr}() is called on an object FI cannot place")
        found = defs.get(func.attr)
        if not found:
            raise _Unknown(f".{func.attr}() is not defined in the code")
        return found, True

    def follow(scope: _Scope, holders: set[str], keys: set[str], todo: list[tuple[_Scope, str]]) -> None:
        """Every use of the setting in ``scope``: the names in ``holders`` and (anywhere) FI's record of a setting."""
        parent: dict[int, ast.AST] = {}
        for node in ast.walk(scope):
            for child in ast.iter_child_nodes(node):
                parent[id(child)] = node
        if not isinstance(scope, ast.Module) or any(_is_record_cell(n) for n in ast.walk(scope)):
            # Where the setting is held, a lookup by a string can reach it. (``globals()`` cannot: the setting is never a
            # module-level name, since storing it outside its function ends the reading.)
            opaque = next((n for n in ast.walk(scope) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id in _OPAQUE), None)
            if opaque is not None:
                raise _Unknown(f"{opaque.func.id}() (line {opaque.lineno}) can read any name")  # type: ignore[attr-defined]
        shared = {n for node in ast.walk(scope) if isinstance(node, (ast.Global, ast.Nonlocal)) for n in node.names}
        holders = set(holders)
        grew = True
        while grew:  # names that hold the setting too (``c = cell``, ``c = dict(cell)``, ``c = rec["cell"]``)
            grew = False
            for node in ast.walk(scope):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id not in holders
                        and (_same_setting(node.value, holders) or _is_record_cell(node.value))):
                    if node.targets[0].id in shared:
                        raise _Unknown(f"{node.targets[0].id} holds the setting outside its function (line {node.lineno})")
                    holders.add(node.targets[0].id)
                    grew = True
        for node in ast.walk(scope):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(_RECORD_CELL + "."):
                keys.add(node.value[len(_RECORD_CELL) + 1:])  # a column of pandas.json_normalize over FI's records
            if isinstance(node, ast.Name) and node.id in holders and isinstance(node.ctx, ast.Load):
                label = node.id
            elif _is_record_cell(node):
                label = "FI's record of a setting"
            else:
                continue
            use(node, label, parent, scope, keys, todo)

    def use(node: ast.AST, label: str, parent: dict[int, ast.AST], scope: _Scope, keys: set[str],
            todo: list[tuple[_Scope, str]]) -> None:
        line = getattr(node, "lineno", "?")
        up = parent.get(id(node))
        if isinstance(up, ast.Subscript) and up.value is node:
            key = _const_str(up.slice)
            if key is None:
                raise _Unknown(f"{label}[...] is read by a name FI cannot see (line {line})")
            keys.add(key)
            return
        if isinstance(up, ast.Attribute) and up.value is node:
            call = parent.get(id(up))
            if not (isinstance(call, ast.Call) and call.func is up):
                raise _Unknown(f"{label}.{up.attr} is used as a value (line {line})")
            if up.attr in _HOLDER_GETS:
                key = _const_str(call.args[0]) if call.args else None
                if key is None:
                    raise _Unknown(f"{label}.{up.attr}(...) reads a name FI cannot see (line {line})")
                keys.add(key)
                return
            if up.attr == "copy" and not call.args and _assigned_whole(call, parent):
                return  # another name for it, followed above
            raise _Unknown(f"{label}.{up.attr}(...) may read any of its names (line {line})")
        if isinstance(up, ast.Compare) and up.comparators and up.comparators[0] is node and len(up.ops) == 1 \
                and isinstance(up.ops[0], (ast.In, ast.NotIn)):
            key = _const_str(up.left)
            if key is None:
                raise _Unknown(f"a name FI cannot see is looked for in {label} (line {line})")
            keys.add(key)
            return
        if isinstance(up, ast.Assign) and up.value is node and len(up.targets) == 1 and isinstance(up.targets[0], ast.Name):
            return  # another name for it, followed above
        if isinstance(up, ast.Call) and up.func is not node:
            if isinstance(up.func, ast.Name) and up.func.id == "dict" and len(up.args) == 1 and not up.keywords \
                    and _assigned_whole(up, parent):
                return
            name = up.func.id if isinstance(up.func, ast.Name) else up.func.attr if isinstance(up.func, ast.Attribute) else "a call"
            if any(isinstance(a, ast.Starred) for a in up.args) or any(k.arg is None for k in up.keywords):
                raise _Unknown(f"{label} is handed to {name}() beside *args or **kwargs, so FI cannot tell which parameter "
                               f"it is (line {line})")
            try:
                targets, method = callees(up.func, scope)
            except _Unknown as e:
                raise _Unknown(f"{label} is handed to {name}(), which FI cannot follow: {e} (line {line})") from None
            for target in targets:
                if target.decorator_list:
                    raise _Unknown(f"{label} is handed to {name}(), which is decorated: the decorator may read any name "
                                   f"(line {line})")
                names = params(target)
                if node in up.args:
                    at = up.args.index(node) + (1 if method and names and names[0] in ("self", "cls") else 0)
                    if at >= len(names):
                        raise _Unknown(f"{label} is handed to {name}() past its named parameters (line {line})")
                    todo.append((target, names[at]))
                else:
                    word = next((k.arg for k in up.keywords if k.value is node), None)
                    if word is None or word not in names + [a.arg for a in target.args.kwonlyargs]:
                        raise _Unknown(f"{label} is handed to {name}() in a way FI cannot follow (line {line})")
                    todo.append((target, word))
            return
        raise _Unknown(f"{label} is used in a way FI cannot follow (line {line})")

    starts = [(fn, params(fn)[0]) for e in entries for fn in defs.get(e, []) if params(fn)]
    if not starts:
        return None, "the simulation defines no run_trial or run_cell that takes the setting"
    keys: set[str] = set()
    todo: list[tuple[_Scope, str]] = list(starts)
    done: set[tuple[int, str]] = set()
    try:
        for tree in trees.values() if analysis else ():  # the analysis: FI's record of a setting, wherever it is read
            follow(tree, set(), keys, todo)
        while todo:
            fn, holder = todo.pop()
            if (id(fn), holder) in done:
                continue
            done.add((id(fn), holder))
            if fn.decorator_list and (fn, holder) not in starts:
                raise _Unknown(f"{fn.name}() is decorated: the decorator may read any name")
            follow(fn, {holder}, keys, todo)
    except _Unknown as e:
        return None, str(e)
    return keys, ""


def _same_setting(value: ast.AST, holders: set[str]) -> bool:
    if isinstance(value, ast.Name):
        return value.id in holders
    if isinstance(value, ast.Call) and not value.keywords:
        if isinstance(value.func, ast.Attribute) and value.func.attr == "copy" and not value.args:
            return isinstance(value.func.value, ast.Name) and value.func.value.id in holders
        if isinstance(value.func, ast.Name) and value.func.id == "dict" and len(value.args) == 1:
            return isinstance(value.args[0], ast.Name) and value.args[0].id in holders
    return False


def _assigned_whole(call: ast.AST, parent: dict[int, ast.AST]) -> bool:
    up = parent.get(id(call))
    return isinstance(up, ast.Assign) and up.value is call and len(up.targets) == 1 and isinstance(up.targets[0], ast.Name)


def _words(name: str) -> set[str]:
    return {w for w in str(name).lower().replace("-", "_").split("_") if w}


def unread_axes(files: dict[str, str], protocol: dict[str, Any] | None) -> tuple[list[str], str]:
    """``(the settings of protocol.grid with two values or more that nothing in the code reads, "")``, or ``([], why)`` when
    that is not certain. Nothing is named when the code reads none of the swept settings (a reading that finds every one
    unused is likelier wrong than the code), or when the code reads a name that is like an unread setting but is not one
    of the plan's (``T_K`` for ``T``, ``sigma_noise`` for ``sigma``): that is a misspelt read, a fault of the code."""
    protocol = protocol if isinstance(protocol, dict) else {}
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    many = [str(axis) for axis, values in grid.items() if isinstance(values, list) and len(values) >= 2]
    if not many:
        return [], ""
    read, why = setting_reads(files)
    if read is None:
        return [], why
    unread = [axis for axis in many if axis not in read]
    if len(unread) == len(many):
        return [], "the code reads none of the grid's swept settings by name"
    from .grid_keys import supplied_names

    _axes, supplied = supplied_names(protocol)
    for name in sorted(read - supplied - set(grid)):
        like = [axis for axis in unread if _words(name) & _words(axis)]
        if like:
            return [], (f"the code reads `{name}`, which is not a setting of the plan but is like `{like[0]}`: a misspelt "
                        "read is a fault of the code, not a setting to take out")
    return unread, ""
