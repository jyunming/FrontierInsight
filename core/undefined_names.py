"""A name the code uses that nothing in the code defines, found before the code runs (no model reads anything here).

A generated script that uses a name it never defines or imports (a class written in one file and used in another without
an import, a helper left out) fails with a NameError on the first line that reaches it. The small run finds that only after
an install and about half a minute, so this reads the code first: :mod:`symtable` resolves every name the way Python does
(function and class scopes, comprehensions, globals), and a name that is used as a global and is bound at module level
nowhere, nor a built-in, is a fault. A file with a ``from x import *`` is not read (its names are not known from the text).

Only the names a Python run would look up are judged, so a name used in a branch that never runs is still a fault of the
code, and a name the code binds in any way Python binds it (assignment, import, def, class, for, with, except, a
``global`` declaration in a function that assigns it) is defined.
"""

from __future__ import annotations

import ast
import builtins
import symtable
from dataclasses import dataclass
from typing import Any

from . import grid_keys as _grid_keys
from .stop_power import READ

#: What a name nothing binds rests on (:mod:`core.stop_power`): a reading of the code, so it never ends a quest.
SOURCE = READ

#: Names the module always has, besides the built-ins.
_MODULE_NAMES = frozenset({"__file__", "__name__", "__doc__", "__spec__", "__loader__", "__package__", "__builtins__",
                           "__annotations__", "__debug__", "__path__", "__cached__", "__class__"})
_BUILTIN = frozenset(dir(builtins)) | _MODULE_NAMES


@dataclass(frozen=True)
class Undefined:
    script: str
    name: str
    line: int
    function: str  # the scope the name is used in: "<module>" or the function's name

    def says(self, supplied: list[str] | None = None) -> str:
        # ``supplied`` is unused; every static check's finding takes it, so the gate calls them all alike.
        where = "at module level" if self.function == "<module>" else f"in `{self.function}`"
        return f"`{self.name}` ({self.script}, first used on line {self.line}, {where})"


def _star_import(tree: ast.Module) -> bool:
    return any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree))


def _tables(table: symtable.SymbolTable) -> list[symtable.SymbolTable]:
    out = [table]
    for child in table.get_children():
        out.extend(_tables(child))
    return out


def _defined(top: symtable.SymbolTable) -> set[str]:
    """The names the module has: bound at module level, or bound in a scope that declares it ``global``."""
    out: set[str] = set()
    for table in _tables(top):
        for sym in table.get_symbols():
            if not (sym.is_assigned() or sym.is_imported() or sym.is_namespace()):
                continue
            if table is top or sym.is_declared_global():
                out.add(sym.get_name())
    return out


def _unbound_uses(top: symtable.SymbolTable, defined: set[str], script: str, lines: dict[str, int]) -> list[Undefined]:
    out: list[Undefined] = []
    for table in _tables(top):
        scope = "<module>" if table is top else table.get_name()
        for sym in table.get_symbols():
            name = sym.get_name()
            # A name used here that its scope does not bind (not a local, a parameter or a free name of an enclosing
            # scope) is looked up as a global: the module's, then the built-ins. (``is_global`` is true of locals too in
            # some Python versions, so it is not used.)
            bound_here = sym.is_local() or sym.is_free() or sym.is_parameter()
            if sym.is_referenced() and not bound_here and name not in defined and name not in _BUILTIN:
                out.append(Undefined(script, name, lines.get(name, 0), scope))
    return out


def _run_time_names(tree: ast.AST) -> tuple[set[str], bool]:
    """The names a script sets on a module at run time (``module.NAME = ...``, ``setattr(module, "NAME", ...)``), and
    whether it does so in a way the text does not show (a ``setattr`` with a computed name, ``exec``, ``globals()``): then
    the names the code may use are not all in the text, and nothing is judged."""
    names: set[str] = set()
    opaque = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "setattr" and len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    names.add(node.args[1].value)
                else:
                    opaque = True
            elif node.func.id in ("exec", "globals"):
                opaque = True
    return names, opaque


def _lines(tree: ast.AST) -> dict[str, int]:
    """The line of each name's first read in the file (symtable gives no lines)."""
    lines: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            lines[node.id] = min(node.lineno, lines.get(node.id, node.lineno))
    return lines


def _read(script: str, text: str) -> tuple[ast.Module, symtable.SymbolTable] | None:
    try:
        return ast.parse(text), symtable.symtable(text, script, "exec")
    except (SyntaxError, ValueError):
        return None


def problems(files: dict[str, str], protocol: dict[str, Any] | None = None,
             design: dict[str, Any] | None = None) -> list[Undefined]:
    """The names used but never defined, in the scripts of ``files`` (``{script name: text}``). A name counts as defined when
    any script of the folder binds it at module level, or sets it on a module at run time (scripts import one another, and
    a package's constants are often set by the script that calls it). A script that does not parse, that holds a
    ``from ... import *``, or that sets names at run time in a way the text does not show is not judged."""
    read = {script: r for script, text in files.items() if (r := _read(script, text)) is not None}
    defined: set[str] = set()
    for tree, top in read.values():
        names, opaque = _run_time_names(tree)
        if opaque:  # a name set at run time that the text does not show may be any of the names used: nothing is judged
            return []
        defined |= names | _defined(top)
    out: list[Undefined] = []
    for script, (tree, top) in read.items():
        if _star_import(tree):
            continue
        out.extend(_unbound_uses(top, defined | _MODULE_NAMES, script, _lines(tree)))
    return out


def undefined_in(script: str, text: str) -> list[Undefined]:
    """The names one script uses that nothing in it defines (:func:`problems` for a folder of one script)."""
    return problems({script: text})


def directive(found: list[Undefined], protocol: dict[str, Any] | None) -> str:
    lines = "\n".join(f"- {u.says()}" for u in found[:6])
    return ("FI read the code before running it, and it uses a name that nothing defines or imports:\n" + lines
            + "\n\nDefine each of these in the script where it is used, or import it from the module that defines it, "
            "as the code's other parts do. Change nothing else.")


def plain_note(found: list[Undefined], protocol: dict[str, Any] | None) -> str:
    return "the code uses a name that nothing defines or imports (" + "; ".join(u.says() for u in found[:2]) + ")"


def gap(found: list[Undefined], protocol: dict[str, Any] | None) -> str:
    return f"the code would fail with a NameError where it uses {found[0].says()}"


def record(found: list[Undefined], protocol: dict[str, Any] | None) -> dict[str, Any]:
    return {"names": [{"script": u.script, "name": u.name, "line": u.line, "function": u.function} for u in found],
            "gap": gap(found, protocol) if found else ""}


code_files = _grid_keys.code_files
