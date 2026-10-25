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
    taken: set[str] = set()
    for n in o_tree.body:
        taken |= _stored(n) if isinstance(n, (ast.Assign, ast.AnnAssign)) else (
            {n.name} if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else set())
    extra = []
    for tree, text in sources:
        for n in tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and getattr(n, "value", None) is not None:
                names = _stored(n)
                if names and not names & taken:
                    extra.append(ast.get_source_segment(text, n) or "")
                    taken |= names
    if extra:
        made.code = made.code.rstrip("\n") + "\n\n\n" + "\n".join(extra) + "\n"
    return made
