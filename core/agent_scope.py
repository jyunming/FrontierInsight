"""A repair round of the coding agent touches only the faulty part of the code.

A repair round in agent mode (:meth:`AgentBuildMixin._reflect_by_agent`, the fix rounds before the run) lets the agent edit
the files of ``code/`` in place. When the failure names a place, FI names it to the agent as the scope of the round: the
function (or method) the traceback ends in, the functions that call it directly and the functions it calls directly (a fix
may need either to pass a value on or to change what is passed). After the round:

* an edit to an existing function or file outside that scope is not kept silently. When the agent says in its last message
  that the fix needs it (a line ``NEEDS: <names> - <why>``), it is kept and logged. Otherwise it is put back and the agent
  is sent back ONCE with the names ("you changed X, Y outside the failing part; keep them unless the fix needs them, and
  say why"); what is still outside after that and not claimed as needed is put back for good;
* a put-back is kept only when every file it touched still compiles; when it does not, the put-back is undone and logged;
* a round that leaves a file that is not valid Python is a failed round: the whole ``code/`` folder is put back.

Files and functions the round *adds* are kept. A failure that names no place in the code (an error raised by a library, a
run that printed nothing) has no scope, and the round is not limited. Only ``.py`` files are limited.
"""
from __future__ import annotations

import ast
import asyncio
import dataclasses
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .agent_coder import AgentCoder, AgentResult, TreeSnapshot, is_scratch, parse_failures

_FRAME = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')
_NEEDS = re.compile(r"(?im)^\s*NEEDS:\s*(.+)$")


@dataclass
class RepairScope:
    """``files``: file under ``code/`` -> the names (``function`` or ``Class.method``) the round may change, or ``None`` when
    the whole file is in scope (the failure is at the top level of the file)."""
    files: dict[str, set[str] | None] = field(default_factory=dict)

    def describe(self) -> str:
        parts = []
        for rel, names in sorted(self.files.items()):
            parts.append(f"`{rel}`" + (": " + ", ".join(f"`{n}`" for n in sorted(names)) if names else ""))
        return ("\n\nThe failure is in " + "; ".join(parts) + " (the function it is in, the functions that call it and the "
                "functions it calls). Change only that code. If the fix really needs a change anywhere else, make it and "
                "end your last message with a line `NEEDS: <names> - <why>`; without that line FI puts an edit to any "
                "other existing function or file back.")


@dataclass
class Violations:
    """What a round changed outside the scope: whole files, and units (``function`` or ``Class.method``) of scope files."""
    files: list[str] = field(default_factory=list)
    units: dict[str, list[str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.files or self.units)

    def names(self) -> list[str]:
        return [*self.files, *[f"{n} in {rel}" for rel, ns in self.units.items() for n in ns]]


# -- the places in a file ----------------------------------------------------------------------------------------------------


def _units(tree: ast.Module) -> dict[str, ast.stmt]:
    """Every top-level function (``name``), every top-level class (``Class``, present or not) and every method
    (``Class.method``)."""
    out: dict[str, ast.stmt] = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = n
        elif isinstance(n, ast.ClassDef):
            out[n.name] = n
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{n.name}.{m.name}"] = m
    return out


def _start(node: ast.stmt) -> int:
    return min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])


def _segment(lines: list[str], node: ast.stmt) -> str:
    return "".join(lines[_start(node) - 1: node.end_lineno])


def _is_class(node: ast.stmt) -> bool:
    return isinstance(node, ast.ClassDef)


def _rel_in_code(path: str, quest_root: Path) -> str:
    """The path of a traceback frame relative to this quest's ``code/`` folder; ``""`` for any other file (a library, a
    folder that merely has a ``code`` part in its name)."""
    code = (quest_root / "code").resolve()
    text = path.replace("\\", "/")
    if text.startswith("/work/"):  # the container's view of the quest folder
        candidate = quest_root / text[len("/work/"):]
    else:
        candidate = Path(path)
    try:
        return candidate.resolve().relative_to(code).as_posix()
    except (OSError, ValueError):
        return ""


def _called_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _callees(code: Path, rel: str, unit: str, node: ast.stmt) -> dict[str, set[str]]:
    """The functions and methods of the study's code (its tests and FI's skill copies left out) that ``node`` calls by name."""
    wanted = _called_names(node)
    out: dict[str, set[str]] = {}
    for path in sorted(code.rglob("*.py")):
        other = path.relative_to(code)
        if any(p in ("tests", "__pycache__") for p in other.parts[:-1]) or other.parts[0] == "skills":
            continue
        try:
            units = _units(ast.parse(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError, ValueError):
            continue
        for name, n in units.items():
            if _is_class(n) or (other.as_posix() == rel and name == unit):
                continue
            if name.rsplit(".", 1)[-1] in wanted:
                out.setdefault(other.as_posix(), set()).add(name)
    return out


def scope_from_traceback(text: str, quest_root: Path) -> RepairScope | None:
    """The scope a failure's traceback names: the function or method holding its last frame in the quest's code, the
    functions that call it directly and the functions it calls. ``None`` when the text holds no such frame."""
    from .finding_repair import direct_callers

    code = quest_root / "code"
    last: tuple[str, int] | None = None
    for m in _FRAME.finditer(text or ""):
        rel = _rel_in_code(m.group(1), quest_root)
        if rel:
            last = (rel, int(m.group(2)))
    if last is None:
        return None
    rel, line = last
    try:
        units = _units(ast.parse((code / rel).read_text(encoding="utf-8")))
    except (OSError, SyntaxError, ValueError):
        return RepairScope({rel: None})
    inside = [(n, node) for n, node in units.items() if _start(node) <= line <= (node.end_lineno or line)]
    methods = [(n, node) for n, node in inside if "." in n]
    pick = methods[0] if methods else next(((n, node) for n, node in inside if not _is_class(node)), None)
    if pick is None:
        return RepairScope({rel: None})  # the top level of the file, or a class body (a line no function holds)
    name, node = pick
    scope = RepairScope({rel: {name}})
    found = direct_callers(code, rel, name.rsplit(".", 1)[-1])
    found_callees = _callees(code, rel, name, node)
    for table in (found, found_callees):
        for other, names in table.items():
            if scope.files.get(other, set()) is not None:
                scope.files.setdefault(other, set()).update(names)  # type: ignore[union-attr]
    return scope


# -- what a round changed outside the scope -----------------------------------------------------------------------------------


def _allowed(name: str, allowed: set[str]) -> bool:
    """Whether the scope (or what the agent claims as needed) lets the round change ``name``; a class (present or not) is
    allowed when one of its methods is."""
    return (name in allowed or name.rsplit(".", 1)[-1] in allowed
            or any(a.startswith(name + ".") for a in allowed))


def _parse_needs(message: str) -> tuple[set[str], str]:
    """The names a ``NEEDS: a, b - why`` line of the agent's last message claims, and what it says."""
    names: set[str] = set()
    said: list[str] = []
    for m in _NEEDS.finditer(message or ""):
        head = m.group(1).partition(" - ")[0]
        names.update(t.strip(" `'\".") for t in head.split(",") if t.strip(" `'\"."))
        said.append(m.group(1).strip())
    return names, "; ".join(said)


def _claimed(name: str, claimed: set[str]) -> bool:
    bare = name.rsplit(".", 1)[-1]
    return any(c == name or c == bare or c == name.split(".", 1)[0] or c.endswith("/" + name) for c in claimed)


def find_violations(scope: RepairScope, snap: TreeSnapshot, code_dir: Path, claimed: set[str]) -> Violations:
    """What the round changed outside ``scope`` (and not claimed as needed): existing ``.py`` files and, in a scope file,
    existing functions and methods. Files and functions the round added are not limited."""
    out = Violations()
    for rel in snap.changed(code_dir):
        if is_scratch(rel) or not rel.endswith(".py") or rel not in snap.files:
            continue
        old_bytes = snap.files[rel][1]
        if old_bytes is None or _claimed(rel, claimed) or (rel in scope.files and scope.files[rel] is None):
            continue
        path = code_dir / rel
        if rel not in scope.files:
            out.files.append(rel)
            continue
        if not path.is_file():
            out.files.append(rel)
            continue
        names = _differing_units(old_bytes.decode("utf-8", "replace"), path.read_bytes().decode("utf-8", "replace"),
                                 {*(scope.files[rel] or ()), *claimed})
        if names:
            out.units[rel] = names
    return out


def _differing_units(old: str, new: str, allowed: set[str]) -> list[str]:
    """The units of ``old`` outside ``allowed`` that are gone from ``new`` or differ in it (a class: only if it is gone)."""
    try:
        old_units, new_units = _units(ast.parse(old)), _units(ast.parse(new))
    except (SyntaxError, ValueError):
        return []
    old_lines, new_lines = old.splitlines(keepends=True), new.splitlines(keepends=True)
    gone_classes = {n for n, node in old_units.items() if _is_class(node) and n not in new_units}
    out = []
    for n, node in old_units.items():
        if _allowed(n, allowed) or n.split(".", 1)[0] in gone_classes and "." in n:
            continue
        if n not in new_units:
            out.append(n)
        elif not _is_class(node) and _segment(new_lines, new_units[n]) != _segment(old_lines, node):
            out.append(n)
    return out


def _put_back_units(old: str, new: str, names: list[str]) -> str:
    """``new`` with each unit of ``names`` as it is in ``old``: a changed one in its place, a deleted one at the place it had
    (after the unit that came before it, else before the one that came after, else at the end of its module or class)."""
    old_tree = ast.parse(old)
    old_units = _units(old_tree)
    old_lines = old.splitlines(keepends=True)
    text = new
    # 1. the ones that are there, bottom up so earlier line numbers stay right
    new_units = _units(ast.parse(text))
    lines = text.splitlines(keepends=True)
    here = sorted((n for n in names if n in new_units), key=lambda n: _start(new_units[n]), reverse=True)
    for n in here:
        node = new_units[n]
        lines[_start(node) - 1: node.end_lineno] = [_segment(old_lines, old_units[n])]
    text = "".join(lines)
    # 2. the ones that are gone, in their old order
    for n in sorted((n for n in names if n not in new_units), key=lambda n: _start(old_units[n])):
        text = _insert_unit(text, old_units, old_lines, n)
    return text


def _insert_unit(text: str, old_units: dict[str, ast.stmt], old_lines: list[str], name: str) -> str:
    cur = _units(ast.parse(text))
    owner = name.split(".", 1)[0] if "." in name else ""
    siblings = [n for n in old_units if (n.split(".", 1)[0] == owner and "." in n)] if owner else         [n for n in old_units if "." not in n]
    siblings.sort(key=lambda n: _start(old_units[n]))
    body = _segment(old_lines, old_units[name])
    lines = text.splitlines(keepends=True)
    gap = ["\n"] if owner else ["\n", "\n"]
    before = [n for n in siblings if _start(old_units[n]) < _start(old_units[name]) and n in cur]
    after = [n for n in siblings if _start(old_units[n]) > _start(old_units[name]) and n in cur]
    if before:
        at = (cur[before[-1]].end_lineno or 0)
        lines[at:at] = [*gap, body]
    elif after:
        at = _start(cur[after[0]]) - 1
        lines[at:at] = [body, *gap]
    elif owner and owner in cur:
        at = cur[owner].end_lineno or len(lines)
        lines[at:at] = [*gap, body]
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.extend([*gap, body])
    return "".join(lines)


def _compile_problems(code_dir: Path, rels: list[str]) -> list[str]:
    bad = []
    for rel in rels:
        path = code_dir / rel
        if rel.endswith(".py") and path.is_file():
            try:
                compile(path.read_bytes(), rel, "exec")
            except (SyntaxError, ValueError):
                bad.append(rel)
    return bad


def put_back(v: Violations, snap: TreeSnapshot, code_dir: Path) -> tuple[list[str], bool]:
    """Put back what ``v`` names; ``(one plain sentence for each thing put back, whether it was kept)``. When a file the
    put-back touched no longer compiles, the whole put-back is undone and nothing is said to be put back."""
    touched = [*v.files, *v.units]
    before: dict[str, bytes | None] = {rel: (code_dir / rel).read_bytes() if (code_dir / rel).is_file() else None
                                       for rel in touched}
    said: list[str] = []
    for rel in v.files:
        data = snap.files[rel][1]
        if data is not None:
            (code_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (code_dir / rel).write_bytes(data)
            said.append(f"{rel} was changed outside the failing part; FI put it back")
    for rel, names in v.units.items():
        old = (snap.files[rel][1] or b"").decode("utf-8", "replace")
        new = (code_dir / rel).read_bytes().decode("utf-8", "replace")
        try:
            fixed = _put_back_units(old, new, names)
        except (SyntaxError, ValueError):
            continue
        (code_dir / rel).write_bytes(fixed.encode("utf-8"))
        said.append(f"{', '.join(names)} in {rel} {'was' if len(names) == 1 else 'were'} changed outside the failing part; "
                    "FI put " + ("it" if len(names) == 1 else "them") + " back")
    if _compile_problems(code_dir, touched):
        for rel, data in before.items():
            if data is None:
                (code_dir / rel).unlink(missing_ok=True)
            else:
                (code_dir / rel).write_bytes(data)
        return [], False
    return said, True


# -- the round -----------------------------------------------------------------------------------------------------------------


async def scoped_continue(coder: AgentCoder, feedback: str, scope: RepairScope | None,
                          log: logging.Logger | None = None) -> AgentResult:
    """:meth:`AgentCoder.continue_` for one repair round limited to ``scope`` (``None``: not limited)."""
    if scope is None or not scope.files:
        return await coder.continue_(feedback)
    code_dir = coder.code_dir

    def say(level: int, text: str, *args: object) -> None:
        if log is not None:
            log.log(level, "[repair] " + text, *args)

    snap = await asyncio.to_thread(TreeSnapshot.take, code_dir)
    result = await coder.continue_(feedback + scope.describe())
    changed = [c for c in await asyncio.to_thread(snap.changed, code_dir) if not is_scratch(c)]
    bad = await asyncio.to_thread(parse_failures, code_dir, changed)
    if bad:
        await asyncio.to_thread(snap.restore, code_dir)
        say(logging.WARNING, "the coding agent's round left %s as invalid Python, so FI undid the whole round",
            ", ".join(bad[:3]))
        return dataclasses.replace(result, ok=False, error=f"{', '.join(bad[:3])} is not valid Python")
    claimed, why = _parse_needs(result.message)
    if claimed:
        say(logging.INFO, "the coding agent says the fix needs %s (%s)", ", ".join(sorted(claimed)), why)
    v = await asyncio.to_thread(find_violations, scope, snap, code_dir, claimed)
    if not v:
        return result
    said, kept = await asyncio.to_thread(put_back, v, snap, code_dir)
    if not kept:
        say(logging.WARNING, "the coding agent's round changed %s outside the failing part; putting that back would leave "
            "the code not compiling, so FI kept it", ", ".join(v.names()))
        return result
    for line in said:
        say(logging.WARNING, "the coding agent's round went beyond the failing code: %s", line)
    # Once, the agent is told what FI put back and may say that the fix needs it.
    again = (f"FI put back your changes to {', '.join(v.names())}, which are outside the failing part "
             f"({scope.describe().strip().splitlines()[0]}). Keep them as they are unless the fix needs them. If it does, "
             "make them again and end your last message with a line `NEEDS: <names> - <why>`; if it does not, there is "
             "nothing more to do.")
    mid = await asyncio.to_thread(TreeSnapshot.take, code_dir)
    second = await coder.continue_(again)
    if second.exhausted or not second.ok:
        say(logging.INFO, "the coding agent was not asked again (%s); the put-back stands",
            "the rounds allowed are used" if second.exhausted else "its round did not finish")
        await asyncio.to_thread(mid.restore, code_dir)
        return result
    changed = [c for c in await asyncio.to_thread(mid.changed, code_dir) if not is_scratch(c)]
    bad = await asyncio.to_thread(parse_failures, code_dir, changed)
    if bad:
        await asyncio.to_thread(mid.restore, code_dir)
        say(logging.WARNING, "the coding agent's second round left %s as invalid Python; FI undid it", ", ".join(bad[:3]))
        return result
    claimed2, why2 = _parse_needs(second.message)
    claimed |= claimed2
    if claimed2:
        say(logging.INFO, "the coding agent says the fix needs %s (%s)", ", ".join(sorted(claimed2)), why2)
    v2 = await asyncio.to_thread(find_violations, scope, snap, code_dir, claimed)
    if v2:
        said2, kept2 = await asyncio.to_thread(put_back, v2, snap, code_dir)
        for line in said2:
            say(logging.WARNING, "the coding agent's round went beyond the failing code: %s", line)
        if not kept2:
            say(logging.WARNING, "putting back %s would leave the code not compiling, so FI kept it", ", ".join(v2.names()))
    return second
