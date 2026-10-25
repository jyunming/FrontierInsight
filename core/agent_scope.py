"""A repair round of the coding agent touches only the faulty part of the code.

A repair round in agent mode (:meth:`AgentBuildMixin._reflect_by_agent`, the fix rounds before the run) lets the agent edit
the files of ``code/`` in place. When the failure names a place, FI names it to the agent as the scope of the round: the
function the traceback ends in, and the functions that call it directly (a fix may need them to pass a value on). After the
round, an edit to an existing function outside that scope is put back (the fix inside the scope is kept) and logged in
plain words; an existing file outside the scope that the round changed is put back whole. Files and functions the round
*adds* are kept. A failure that names no place in the code (an error raised by a library, a run that printed nothing)
has no scope, and the round is not limited.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .agent_coder import AgentCoder, AgentResult, TreeSnapshot, is_scratch

_FRAME = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')


@dataclass
class RepairScope:
    """``files``: file under ``code/`` -> the top-level names (functions, classes) the round may change, or ``None`` when
    the whole file is in scope (the failure is at the top level of the file)."""
    files: dict[str, set[str] | None] = field(default_factory=dict)

    def describe(self) -> str:
        parts = []
        for rel, names in sorted(self.files.items()):
            parts.append(f"`{rel}`" + (": " + ", ".join(f"`{n}`" for n in sorted(names)) if names else ""))
        return ("\n\nThe failure is in " + "; ".join(parts) + ". Change only that code (and, where the fix needs it, a "
                "function that calls it). FI puts back an edit to any other existing function or file, so leave the "
                "rest as it is.")


def _units(tree: ast.Module) -> dict[str, ast.stmt]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _start(node: ast.stmt) -> int:
    return min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])


def _segment(lines: list[str], node: ast.stmt) -> str:
    return "".join(lines[_start(node) - 1: node.end_lineno])


def _rel_in_code(path: str, quest_root: Path) -> str:
    """The path of a traceback frame relative to ``code/``; ``""`` for a frame outside the quest's code."""
    text = path.replace("\\", "/")
    root = str(quest_root).replace("\\", "/").rstrip("/") + "/"
    if text.casefold().startswith(root.casefold()):
        text = text[len(root):]
    elif "/code/" in text:
        text = text[text.rfind("/code/") + 1:]
    if not text.startswith("code/") or "site-packages" in text or "/.venv/" in f"/{text}":
        return ""
    return text[len("code/"):]


def scope_from_traceback(text: str, quest_root: Path) -> RepairScope | None:
    """The scope a failure's traceback names: the top-level function (or class) holding its last frame in the quest's code,
    and the functions that call it directly. ``None`` when the text holds no such frame."""
    from .finding_repair import direct_callers

    code = quest_root / "code"
    last: tuple[str, int, str] | None = None
    for m in _FRAME.finditer(text or ""):
        rel = _rel_in_code(m.group(1), quest_root)
        if rel:
            last = (rel, int(m.group(2)), m.group(3) or "")
    if last is None:
        return None
    rel, line, _where = last
    path = code / rel
    try:
        source = path.read_text(encoding="utf-8")
        units = _units(ast.parse(source))
    except (OSError, SyntaxError, ValueError):
        return RepairScope({rel: None})
    name = next((n for n, node in units.items() if _start(node) <= line <= (node.end_lineno or line)), "")
    if not name:
        return RepairScope({rel: None})  # the top level of the file (or a line no function holds)
    scope = RepairScope({rel: {name}})
    for other, names in direct_callers(code, rel, name).items():
        scope.files.setdefault(other, set())
        if scope.files[other] is not None:
            scope.files[other].update(names)  # type: ignore[union-attr]
    return scope


def enforce_scope(scope: RepairScope, snap: TreeSnapshot, code_dir: Path) -> list[str]:
    """Put back what the round changed outside ``scope``; one plain sentence for each thing put back."""
    said: list[str] = []
    for rel in snap.changed(code_dir):
        if is_scratch(rel) or not rel.endswith(".py") or rel not in snap.files:
            continue  # scratch folders, other file kinds and files the round added are not limited
        if rel in scope.files and scope.files[rel] is None:
            continue
        old_bytes = snap.files[rel][1]
        path = code_dir / rel
        if old_bytes is None:
            continue  # too large for the snapshot to hold: cannot be put back
        if rel not in scope.files or not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(old_bytes)
            said.append(f"{rel} was " + ("removed" if rel in scope.files else "changed, and the failure is not in it")
                        + "; FI put it back")
            continue
        allowed = scope.files[rel] or set()
        fixed, names = _put_back_functions(old_bytes.decode("utf-8", "replace"), path.read_bytes().decode("utf-8", "replace"), allowed)
        if names:
            path.write_bytes(fixed.encode("utf-8"))
            said.append(f"{', '.join(names)} in {rel} {'was' if len(names) == 1 else 'were'} changed, and the failure is not "
                        "in " + ("it" if len(names) == 1 else "them") + "; FI put " + ("it" if len(names) == 1 else "them")
                        + " back")
    return said


def _put_back_functions(old: str, new: str, allowed: set[str]) -> tuple[str, list[str]]:
    """``new`` with every top-level function or class of ``old`` that is not in ``allowed`` and differs (or is gone) as it
    was in ``old``; and the names it put back. A file that does not parse in either form is returned as it is."""
    try:
        old_units, new_tree = _units(ast.parse(old)), ast.parse(new)
    except (SyntaxError, ValueError):
        return new, []
    new_units = _units(new_tree)
    old_lines, new_lines = old.splitlines(keepends=True), new.splitlines(keepends=True)
    differ = [n for n, node in old_units.items() if n not in allowed
              and (n not in new_units or _segment(new_lines, new_units[n]) != _segment(old_lines, node))]
    if not differ:
        return new, []
    out = new_lines[:]
    for n in sorted(differ, key=lambda n: _start(new_units[n]) if n in new_units else 10 ** 9, reverse=True):
        text = _segment(old_lines, old_units[n])
        if n in new_units:
            out[_start(new_units[n]) - 1: new_units[n].end_lineno] = [text]
        else:
            if out and not out[-1].endswith("\n"):
                out[-1] += "\n"
            out.extend(["\n\n", text])
    return "".join(out), differ


async def scoped_continue(coder: AgentCoder, feedback: str, scope: RepairScope | None,
                          log: logging.Logger | None = None) -> AgentResult:
    """:meth:`AgentCoder.continue_` for one repair round limited to ``scope`` (``None``: not limited)."""
    if scope is None or not scope.files:
        return await coder.continue_(feedback)
    code_dir = coder.code_dir
    snap = await asyncio.to_thread(TreeSnapshot.take, code_dir)
    result = await coder.continue_(feedback + scope.describe())
    for line in await asyncio.to_thread(enforce_scope, scope, snap, code_dir):
        if log is not None:
            log.warning("[repair] the coding agent's round went beyond the failing code: %s", line)
    return result

