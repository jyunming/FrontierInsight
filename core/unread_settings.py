"""A setting of the grid the simulation never reads.

FI calls the simulation once per setting of ``protocol.grid``. A grid key the simulation never reads changes nothing: each
of its values runs the same simulation again, with other seeds. A real plan (first-order kinetics, rounds s1 and r13) kept
five temperatures in the grid beside the noise level and the number of time points while ``run_trial`` built its own five
temperatures: 125 settings ran where the study was 25, and the paper said there was no replication.

This reads the simulation with ``ast`` (it is never run) and follows the setting (the first parameter of ``run_trial`` /
``run_cell``) through every use. It answers only when it is certain: any use it cannot follow means "this could read any
name", and then nothing is known and nothing is taken out. Removing a setting the simulation does read would change the
experiment without a word, which is worse than running it again.

What is followed:

* a read by a literal name: ``cell["x"]``, ``cell.get("x", d)``, ``cell.pop("x")``, ``"x" in cell``, in any scope inside the
  function, a ``lambda`` or a nested ``def`` included (a nested scope reads the same setting);
* another name for it: ``c = cell``, ``c = dict(cell)``, ``c = cell.copy()`` (not a ``global`` / ``nonlocal`` name);
* the setting handed whole to a function these files define: ``f(cell)`` when ``f`` is not imported from elsewhere,
  ``self.f(cell)``, ``Model().f(cell)`` / ``Model.f(cell)`` for a class defined here, and ``mod.f(cell)`` for a module of
  these files; followed into that function's parameter.

Anything else (``**cell``, a computed name, a loop over it, ``cell.items()``, a call to code these files do not define,
a call on an object FI cannot place, storing it on an object or in a container, a default value of a parameter) ends the
reading with the reason.
"""

from __future__ import annotations

import ast
from typing import Any

from .grid_keys import ENTRIES

_Fn = ast.FunctionDef | ast.AsyncFunctionDef
#: Calls on the setting that read one literal name.
_HOLDER_GETS = frozenset({"get", "pop", "setdefault"})


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


def setting_reads(files: dict[str, str], entries: tuple[str, ...] = ENTRIES) -> tuple[set[str] | None, str]:
    """``(every name the simulation reads from its setting, "")``, or ``(None, why)`` when that cannot be known for sure.

    ``files``: the simulation's own files as ``{relative path: text}`` (the script FI calls and the package it uses, not
    the analysis or the tests)."""
    trees: dict[str, ast.Module] = {}
    for name, text in files.items():
        try:
            trees[name] = ast.parse(text)
        except (SyntaxError, ValueError):
            return None, f"{name} does not parse"
    local = _local_modules(files)
    defs: dict[str, list[_Fn]] = {}
    classes: set[str] = set()
    imported: dict[int, dict[str, bool]] = {}  # id(function) -> the imports of its file
    for tree in trees.values():
        binds = _imports(tree, local)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(node.name, []).append(node)
                imported[id(node)] = binds
            elif isinstance(node, ast.ClassDef):
                classes.add(node.name)

    def params(fn: _Fn) -> list[str]:
        return [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]

    def callees(func: ast.AST, fn: _Fn) -> list[_Fn] | None:
        """The functions of these files a call can reach, or ``None`` when it may reach code FI cannot read."""
        binds = imported.get(id(fn), {})
        if isinstance(func, ast.Name):
            if binds.get(func.id) is False:  # imported from elsewhere, even if a function here has the same name
                return None
            if func.id in params(fn) or any(isinstance(n, ast.Name) and n.id == func.id and isinstance(n.ctx, ast.Store)
                                            for n in ast.walk(fn)):
                return None  # a name the function binds itself: it may hold any function
            return defs.get(func.id) or None
        if not isinstance(func, ast.Attribute):
            return None
        recv = func.value
        if isinstance(recv, ast.Call):
            recv = recv.func  # ``Model().f(cell)``
            known = isinstance(recv, ast.Name) and recv.id in classes and binds.get(recv.id) is not False
        elif isinstance(recv, ast.Name):
            known = (recv.id in ("self", "cls") or binds.get(recv.id) is True
                     or (recv.id in classes and binds.get(recv.id) is not False))
        else:
            known = False
        return (defs.get(func.attr) or None) if known else None

    starts = [(fn, params(fn)[0]) for e in entries for fn in defs.get(e, []) if params(fn)]
    if not starts:
        return None, "the simulation defines no run_trial or run_cell that takes the setting"
    keys: set[str] = set()
    todo = list(starts)
    done: set[tuple[int, str]] = set()
    while todo:
        fn, holder = todo.pop()
        if (id(fn), holder) in done:
            continue
        done.add((id(fn), holder))
        parent: dict[int, ast.AST] = {}
        for node in ast.walk(fn):
            for child in ast.iter_child_nodes(node):
                parent[id(child)] = node
        shared = {n for node in ast.walk(fn) if isinstance(node, (ast.Global, ast.Nonlocal)) for n in node.names}
        holders = {holder}
        grew = True
        while grew:  # names that hold the setting too (``c = cell``, ``c = dict(cell)``, ``c = cell.copy()``)
            grew = False
            for node in ast.walk(fn):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id not in holders and _same_setting(node.value, holders)):
                    if node.targets[0].id in shared:
                        return None, f"{node.targets[0].id} holds the setting outside {fn.name}() (line {node.lineno})"
                    holders.add(node.targets[0].id)
                    grew = True
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Name) and node.id in holders and isinstance(node.ctx, ast.Load)):
                continue
            up = parent.get(id(node))
            if isinstance(up, ast.Subscript) and up.value is node:
                key = _const_str(up.slice)
                if key is None:
                    return None, f"{node.id}[...] is read by a name FI cannot see (line {node.lineno})"
                keys.add(key)
                continue
            if isinstance(up, ast.Attribute) and up.value is node:
                call = parent.get(id(up))
                if not (isinstance(call, ast.Call) and call.func is up):
                    return None, f"{node.id}.{up.attr} is used as a value (line {node.lineno})"
                if up.attr in _HOLDER_GETS:
                    key = _const_str(call.args[0]) if call.args else None
                    if key is None:
                        return None, f"{node.id}.{up.attr}(...) reads a name FI cannot see (line {node.lineno})"
                    keys.add(key)
                    continue
                if up.attr == "copy" and not call.args and _assigned_whole(call, parent):
                    continue  # another name for it, followed above
                return None, f"{node.id}.{up.attr}(...) may read any of its names (line {node.lineno})"
            if isinstance(up, ast.Compare) and up.comparators and up.comparators[0] is node and len(up.ops) == 1 \
                    and isinstance(up.ops[0], (ast.In, ast.NotIn)):
                key = _const_str(up.left)
                if key is None:
                    return None, f"a name FI cannot see is looked for in {node.id} (line {node.lineno})"
                keys.add(key)
                continue
            if isinstance(up, ast.Assign) and up.value is node and len(up.targets) == 1 and isinstance(up.targets[0], ast.Name):
                continue  # another name for it, followed above
            if isinstance(up, ast.Call) and up.func is not node:
                if isinstance(up.func, ast.Name) and up.func.id == "dict" and len(up.args) == 1 and not up.keywords \
                        and _assigned_whole(up, parent):
                    continue
                name = up.func.id if isinstance(up.func, ast.Name) else up.func.attr if isinstance(up.func, ast.Attribute) else "a call"
                targets = callees(up.func, fn)
                if not targets:
                    return None, f"{node.id} is handed to {name}(), which FI cannot read (line {node.lineno})"
                for target in targets:
                    names = params(target)
                    if node in up.args:
                        at = up.args.index(node)
                        # A method called on an object or a class: its first parameter is that object or class.
                        if isinstance(up.func, ast.Attribute) and names and names[0] in ("self", "cls"):
                            at += 1
                        if at >= len(names):
                            return None, f"{node.id} is handed to {name}() past its named parameters (line {node.lineno})"
                        todo.append((target, names[at]))
                    else:
                        word = next((k.arg for k in up.keywords if k.value is node), None)
                        if word is None or word not in names + [a.arg for a in target.args.kwonlyargs]:
                            return None, f"{node.id} is handed to {name}() in a way FI cannot follow (line {node.lineno})"
                        todo.append((target, word))
                continue
            return None, f"{node.id} is used in a way FI cannot follow (line {node.lineno})"
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


def unread_axes(files: dict[str, str], protocol: dict[str, Any] | None) -> tuple[list[str], str]:
    """``(the settings of protocol.grid with two values or more that the simulation never reads, "")``, or ``([], why)``
    when that is not certain. Nothing is named when the simulation reads none of the grid's settings with two values or
    more: a reading that finds every swept setting unused is far likelier wrong than the simulation."""
    grid = protocol.get("grid") if isinstance(protocol, dict) and isinstance(protocol.get("grid"), dict) else {}
    many = [str(axis) for axis, values in grid.items() if isinstance(values, list) and len(values) >= 2]
    if not many:
        return [], ""
    read, why = setting_reads(files)
    if read is None:
        return [], why
    unread = [axis for axis in many if axis not in read]
    if len(unread) == len(many):
        return [], "the simulation reads none of the grid's swept settings by name"
    return unread, ""
