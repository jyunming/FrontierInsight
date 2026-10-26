"""What a repair of a script gives back, as the script it makes.

A repair of ``simulate.py`` (or the analysis) used to be asked for the whole file. A reply that came back cut short lost
the helpers it left out, a stand-in word replaced the script, and a result name changed in passing made a check that had
passed fail again. Now a repair is asked for the **functions** to replace or add; FI puts them in the file with the same
splice a repair of a finding uses (:func:`core.finding_repair.splice`) and leaves every other byte as it was. A reply
that is a whole script (in `code`) is still taken whole. In both cases the one contract check
(:func:`core.trial_runner.repair_breaks_contract`) judges the result, which every repair path asks.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from . import finding_repair as _finding


@dataclass
class Resolved:
    """``code`` is the script the reply makes (``""`` when it makes none), ``kind`` how (``whole`` / ``functions`` /
    ``none``), ``replaced`` the functions put in (``functions`` only) and ``why`` the reason when nothing was made."""
    code: str = ""
    kind: str = "none"
    replaced: list[str] = field(default_factory=list)
    why: str = ""


def resolve(old: str, code: object = "", functions: object = "") -> Resolved:
    """The script that a repair reply makes from ``old``. ``functions`` is the reply's functions to replace or add (their
    whole ``def``, any number); ``code`` is its script. A ``code`` alone is the whole script and is returned as it is, for
    the contract check to judge (which refuses a stand-in word, a lost entry point and a function that is called but no
    longer defined). Both given: the functions are spliced AND what the whole script adds at module level (imports and
    constants the file lacks, and any function of it that differs from the file's) comes with them, so nothing the reply
    wrote is lost; when the two give one function differently, nothing is made and the reason says so."""
    fn_text = functions if isinstance(functions, str) else ""
    code_text = code if isinstance(code, str) else ""
    if fn_text.strip() and code_text.strip():
        return _both(old, code_text, fn_text)
    if fn_text.strip():
        return _splice(old, fn_text)
    if not code_text.strip():
        return Resolved(why="the reply held neither functions nor a script")
    return Resolved(code=code_text, kind="whole")


def _splice(old: str, text: str) -> Resolved:
    new, replaced, why = _finding.splice(old, text, None, None, limit=None)
    if new is None:
        return Resolved(why=why)
    return Resolved(code=new, kind="functions", replaced=replaced)


def _parsed(text: str) -> tuple[ast.Module | None, dict[str, ast.AST]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None, {}
    return tree, {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _stored(node: ast.AST) -> set[str]:
    return {t.id for t in ast.walk(node) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}


def _both(old: str, code: str, functions: str) -> Resolved:
    block = next((m.group(1) for m in _finding._FENCE.finditer(functions) if "def " in m.group(1)), functions)
    f_tree, f_defs = _parsed(block)
    c_tree, c_defs = _parsed(code)
    o_tree, o_defs = _parsed(old)
    if f_tree is None or c_tree is None or o_tree is None:
        return Resolved(why="the reply's `functions` and `code` are not both valid Python")
    for name in f_defs.keys() & c_defs.keys():
        if ast.dump(f_defs[name]) != ast.dump(c_defs[name]):
            return Resolved(why=f"the reply gives `{name}` in `functions` and in `code` and they differ; give it once")
    sources = ((c_tree, code), (f_tree, block))
    imports = [ast.get_source_segment(text, n) or "" for tree, text in sources for n in tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    changed = [ast.get_source_segment(code, n) or "" for name, n in c_defs.items()
               if name not in f_defs and (name not in o_defs or ast.dump(n) != ast.dump(o_defs[name]))]
    pieces = [*dict.fromkeys(i for i in imports if i), block.rstrip("\n"), *changed]
    made = _splice(old, "\n\n\n".join(pieces) + "\n")
    if made.kind != "functions":
        return made
    return _carry_units(old, made, sources)


def _unit_key(node: ast.stmt) -> tuple | None:
    """What a top-level statement is called, so that the same unit of the old file and of the reply can be paired: a class by
    its name, an assignment by the names it binds, an ``if`` by its test (the ``if __name__`` block). Anything else (a bare
    call, a loop) has no name: it is added when the file lacks it."""
    if isinstance(node, ast.ClassDef):
        return ("class", node.name)
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        names = _stored(node)
        return ("assign", tuple(sorted(names))) if names else None
    if isinstance(node, ast.If):
        return ("if", ast.dump(node.test))
    return None


def _segment(lines: list[str], node: ast.stmt) -> str:
    start = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
    return "".join(lines[start - 1:node.end_lineno])


def _carry_units(old: str, made: Resolved, sources: tuple) -> Resolved:
    """Every top-level unit of the reply that is not a function or an import and that the file lacks or that the reply
    changes (a class, a constant, the ``if __name__`` block, a bare statement) is in the result: put in place of the file's
    own unit of the same name, else added at the end. Decorators go with their class. When two units of the file share one
    name the reply cannot say which one it means: nothing is made."""
    current = made.code
    for tree, text in sources:
        lines = text.splitlines(keepends=True)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # a docstring or a bare value
            now = ast.parse(current)
            if any(ast.dump(n) == ast.dump(node) for n in now.body):
                continue
            key = _unit_key(node)
            same = [n for n in now.body if key is not None and _unit_key(n) == key]
            body = _segment(lines, node).rstrip("\n") + "\n"
            if len(same) > 1:
                return Resolved(why=f"the file has more than one top-level unit like `{key[1]}`; the reply cannot say which it changes")
            cur = current.splitlines(keepends=True)
            if same:
                start = min([same[0].lineno, *[d.lineno for d in getattr(same[0], "decorator_list", [])]])
                cur[start - 1:same[0].end_lineno] = [body]
                current = "".join(cur)
            else:
                current = current.rstrip("\n") + "\n\n\n" + body
    try:
        ast.parse(current)
    except SyntaxError as e:
        return Resolved(why=f"the file would not be valid Python after the reply's classes and constants ({e.msg})")
    made.code = current
    return made
