"""A setting the code reads must be a setting the study has.

The simulation is called once per cell of the protocol's grid, and the analysis reads each cell back from FI's record
(``entry["cell"]``). A read of a name the grid does not have gives the default silently (``cell.get("x", 80.0)``), so one
misspelt name (the grid says ``rate``, the code reads ``rate_hz``) makes a whole axis collapse to one value with no
error anywhere. This reads the code with ``ast`` (it is never run) and finds every read of a setting by a name that is
not among the names the plan gives: the grid's axes, the settings the plan fixes, and the keys of the cases its
checks run at.

What is read as "a setting":

* a parameter named ``cell`` / ``setting`` / ``settings`` (or the first parameter of ``run_cell`` / ``run_trial``, the trial
  contract's entry points, ``core/trial_runner.py``), and a name assigned from one;
* anything read as ``<expression>["cell"]`` or ``<expression>.get("cell")``: how the analysis reads a cell out of FI's
  per-cell record.

What counts as a read: ``setting["k"]`` and ``setting.get("k"[, default])`` with a literal ``k``. Names written in a
chain of fallbacks (``cell.get("a", cell.get("b", 1.0))`` or ``cell.get("a") or cell.get("b")``) are one read that is
satisfied by any one of them, so a code that accepts several spellings is flagged only when none of them is a name
the study has. The fields of FI's per-cell record itself (``cell``, ``key``, ``metrics`` ...) are not settings. The check
is conservative: a read with a fallback value of a name that is like none of the study's settings is an optional setting
the study never sets, and is left alone; a computed key, ``in`` tests, ``**`` spreads and a setting passed on to a function are not followed.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Names that hold one setting (one cell of the grid).
SETTING_NAMES = frozenset({"cell", "setting", "settings"})
#: The trial contract's entry points: their first parameter is the setting.
ENTRIES = ("run_cell", "run_trial")
#: The fields of the per-cell record FI writes for the analysis (``raw/trials.json``): not settings.
RECORD_FIELDS = frozenset({"cell", "key", "planned", "ok", "failed", "metrics", "trials", "rows", "values"})
_GET = frozenset({"get", "pop", "setdefault"})


@dataclass(frozen=True)
class Read:
    """One read of a setting: the names it accepts (a chain of fallbacks is one read), where, and its default."""

    script: str
    line: int
    keys: tuple[str, ...]
    default: str
    fallback: bool = False
    planned: int | None = None  # set for a fixed count of runs per setting (:func:`counts`): what the protocol says

    def says(self, supplied: list[str]) -> str:
        if self.planned is not None:
            return (f"{self.script} line {self.line}: fixes the number of runs per setting at {self.default} "
                    f"(`{self.keys[0]}`); the protocol runs {self.planned}")
        named = " or ".join(f"`{k}`" for k in self.keys)
        fallback = f" (and gives {self.default} when it is not there)" if self.default else ""
        have = ", ".join(f"`{n}`" for n in supplied[:8]) or "none"
        return f"{self.script} line {self.line}: reads the setting {named}{fallback}; the study's settings are {have}"


def _const_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _text(node: ast.AST | None) -> str:
    try:
        return ast.unparse(node)[:40] if node is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def _is_record_get(node: ast.AST) -> bool:
    """``<x>["cell"]`` or ``<x>.get("cell"[, d])``: the cell of one entry of FI's per-cell record."""
    if isinstance(node, ast.Subscript):
        return _const_str(node.slice) == "cell"
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and bool(node.args) and _const_str(node.args[0]) == "cell")


def _setting_like(node: ast.AST, names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    if _is_record_get(node):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "copy":
            return _setting_like(node.func.value, names)
        if isinstance(node.func, ast.Name) and node.func.id == "dict" and len(node.args) == 1 and not node.keywords:
            return _setting_like(node.args[0], names)
    return False


def _read_of(node: ast.AST, names: set[str]) -> tuple[str, ast.AST | None] | None:
    """``(the key, the default expression)`` when ``node`` reads a setting by a literal key, else ``None``."""
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and _setting_like(node.value, names):
        key = _const_str(node.slice)
        return (key, None) if key is not None else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _GET and node.args
            and _setting_like(node.func.value, names)):
        key = _const_str(node.args[0])
        if key is None:
            return None
        return key, node.args[1] if len(node.args) > 1 else None
    return None


def _scopes(tree: ast.Module) -> list[tuple[ast.AST, set[str]]]:
    """Each function with the names in it that hold a setting, and the module body with its own."""
    out: list[tuple[ast.AST, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            names = {a for a in args if a in SETTING_NAMES}
            if node.name in ENTRIES and args:
                names.add(args[0])
            out.append((node, names))
    out.append((tree, set(SETTING_NAMES)))
    return out


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of ``scope`` that are not inside a nested function (those have their own scope)."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        found.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return found


def reads(source: str, script: str) -> list[Read]:
    """Every read of a setting by a literal name in ``source`` (see the module docstring). Never raises: a script that
    does not parse has none to report (it fails on its own)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: dict[int, tuple[ast.AST, str, ast.AST | None]] = {}
    for scope, names in _scopes(tree):
        names = set(names)
        nodes = sorted(_own_nodes(scope), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
        # A name assigned from a setting holds one too (``c = entry["cell"]``), in the order the code assigns them.
        for node in nodes:
            if isinstance(node, ast.Assign) and _setting_like(node.value, names):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.target, ast.Name):
                if node.target.id in SETTING_NAMES:
                    names.add(node.target.id)
        for node in nodes:
            hit = _read_of(node, names)
            if hit is not None and hit[0] not in RECORD_FIELDS:
                found[id(node)] = (node, hit[0], hit[1])
    # A chain of fallbacks is one read: the default of one is another read, and ``a or b`` joins both.
    parent: dict[int, int] = {i: i for i in found}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (node, _key, default) in list(found.items()):
        if default is not None and id(default) in found:
            parent[root(i)] = root(id(default))
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            ids = [id(v) for v in node.values if id(v) in found]
            for other in ids[1:]:
                parent[root(other)] = root(ids[0])
    groups: dict[int, list[int]] = {}
    for i in found:
        groups.setdefault(root(i), []).append(i)
    out: list[Read] = []
    for members in groups.values():
        members.sort(key=lambda i: (getattr(found[i][0], "lineno", 0), getattr(found[i][0], "col_offset", 0)))
        keys = tuple(dict.fromkeys(found[i][1] for i in members))
        last_default = next((found[i][2] for i in reversed(members) if found[i][2] is not None
                             and id(found[i][2]) not in found), None)
        first = found[members[0]][0]
        out.append(Read(script=script, line=int(getattr(first, "lineno", 0)), keys=keys, default=_text(last_default),
                        fallback=last_default is not None))
    return sorted(out, key=lambda r: (r.line, r.keys))


def _case_keys(value: Any, into: set[str]) -> None:
    """Every key of every ``case`` / ``cases`` the plan's checks give, at any depth."""
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k) in ("case", "cases"):
                cases = v if isinstance(v, list) else [v]
                for c in cases:
                    if isinstance(c, dict):
                        into.update(str(n) for n in c)
            _case_keys(v, into)
    elif isinstance(value, list):
        for v in value:
            _case_keys(v, into)


def supplied_names(protocol: dict[str, Any] | None, design: dict[str, Any] | None = None) -> tuple[list[str], set[str]]:
    """``(the grid's axes, every name a setting may have)``: the axes, the settings the plan fixes, and the keys of the
    cases its checks run at."""
    from . import oracle_forms as _forms

    protocol = protocol if isinstance(protocol, dict) else {}
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    axes = [str(k) for k in grid]
    allowed = set(axes)
    allowed.update(_forms.fixed_settings_of_design({**(design or {}), "protocol": protocol}))
    _case_keys(protocol, allowed)
    return axes, allowed


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name).lower()) if len(t) >= 3}


def _near(key: str, supplied: list[str] | set[str]) -> list[str]:
    """The supplied names that look like ``key``: one contains the other, they share a word of three letters or more
    (``rate`` and ``rate_hz``), or they are spelt alike."""
    low = key.lower()
    pool = sorted(supplied)
    contained = [s for s in pool if (len(s) >= 3 and s.lower() in low) or (len(low) >= 3 and low in s.lower())]
    shared = [s for s in pool if _tokens(s) & _tokens(key)]
    close = difflib.get_close_matches(key, pool, n=2, cutoff=0.75)
    return list(dict.fromkeys([*contained, *shared, *close]))


#: A name that says it counts trials or runs (``N_TRIALS``, ``num_runs``, ``TRIALS_PER_SETTING``). Replicates, repeats and
#: repetitions are what an analysis counts for its own resampling or seeds, so they are not read as the protocol's count.
_COUNT_NAME = re.compile(r"^(?:(?:n|num|number)_?(?:of_)?)?(?:trials|runs)(?:_per_(?:setting|cell|condition))?$", re.IGNORECASE)


def counts(source: str, script: str, planned: int) -> list[Read]:
    """The module-level whole-number literals in ``source`` that a name saying it counts trials or runs holds, other than
    ``planned``. Never raises: a script that does not parse has none to report."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    out: list[Read] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, int) and not isinstance(value.value, bool)):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and _COUNT_NAME.match(target.id) and value.value != planned:
                out.append(Read(script=script, line=int(node.lineno), keys=(target.id,), default=str(value.value),
                                planned=planned))
    return out


def _planned_runs(protocol: dict[str, Any] | None) -> int | None:
    runs = protocol.get("runs_per_setting") if isinstance(protocol, dict) else None
    if isinstance(runs, bool) or not isinstance(runs, (int, float)) or runs != int(runs) or runs < 1:
        return None
    return int(runs)


def problems(files: dict[str, str], protocol: dict[str, Any] | None, design: dict[str, Any] | None = None) -> list[Read]:
    """The reads, in ``files`` (``{script name: text}``), of a setting the study does not have, and each number of runs per
    setting a script fixes that the protocol sets otherwise. Nothing about the first when the plan has no grid (there is
    nothing to compare with)."""
    out: list[Read] = []
    planned = _planned_runs(protocol)
    if planned is not None and any(re.search(r"^\s*def\s+run_trial\s*\(", text, re.MULTILINE) for text in files.values()):
        for script, text in files.items():
            out.extend(counts(text, script, planned))
    axes, allowed = supplied_names(protocol, design)
    if not axes:
        return out
    for script, text in files.items():
        for r in reads(text, script):
            if set(r.keys) & allowed:
                continue
            # A name with a fallback value that is like none of the study's is an optional setting the study never
            # sets, not a misspelt one: only a read that cannot be answered, or one that looks like a study setting
            # spelt another way, is a fault.
            if r.fallback and not any(_near(k, allowed) for k in r.keys):
                continue
            out.append(r)
    return out


def directive(found: list[Read], protocol: dict[str, Any] | None) -> str:
    """What to tell a repair: plain, naming each read, its line and the names the study really has. The instruction is the
    last thing in it (the repair path keeps the tail of a long message)."""
    axes, _allowed = supplied_names(protocol)
    lines = []
    for r in found[:8]:
        if r.planned is not None:
            lines.append(f"- {r.says(axes)}")
            continue
        hint = [h for h in _near(r.keys[0], axes)][:2]
        lines.append(f"- {r.says(axes)}" + (f" (did you mean {' or '.join(f'`{h}`' for h in hint)}?)" if hint else ""))
    tail = []
    if any(r.planned is None for r in found):
        tail.append(
            "A read of a setting that is not there silently gives its default, so every row would carry the same value "
            "and no error would show. Read each setting by the name the study gives it, with no fallback value")
    if any(r.planned is not None for r in found):
        tail.append(
            "FI runs the trials the protocol states and hands the analysis every one of them; a number fixed in the code "
            "makes the analysis reject the record. Take the number of runs from the trials FI supplies (count them), "
            "not from a constant")
    return (
        "FI read the code before running it, and "
        + ("it fixes a number of runs that the protocol sets otherwise" if all(r.planned is not None for r in found)
           else "it reads a setting the study does not have" if all(r.planned is None for r in found)
           else "it does not match the study")
        + ":" + chr(10) + chr(10).join(lines)
        + chr(10) * 2 + ". ".join(tail) + ", and change nothing else."
    )


def plain_note(found: list[Read], protocol: dict[str, Any] | None) -> str:
    """One sentence for the stop and for run.log."""
    axes, _allowed = supplied_names(protocol)
    names = "; ".join(r.says(axes) for r in found[:3])
    if all(r.planned is not None for r in found):
        return f"the code fixes a number of runs that the protocol sets otherwise, so its analysis would not accept the trials ({names})"
    return f"the code reads a setting the study does not have, so a whole axis would not vary ({names})"


def record(found: list[Read], protocol: dict[str, Any] | None) -> dict[str, Any]:
    axes, _allowed = supplied_names(protocol)
    return {"settings_of_the_study": axes, "reads": [{"script": r.script, "line": r.line, "names": list(r.keys),
                                                      "default": r.default,
                                                      **({"protocol_runs": r.planned} if r.planned is not None else {})}
                                                     for r in found]}


def code_files(code_dir: Path) -> dict[str, str]:
    """The scripts and package files of ``code/`` that run the study, not their tests, as ``{relative path: text}``."""
    out: dict[str, str] = {}
    code_dir = Path(code_dir)
    if not code_dir.is_dir():
        return out
    for path in sorted(code_dir.rglob("*.py")):
        rel = path.relative_to(code_dir)
        if "__pycache__" in rel.parts or "tests" in rel.parts or re.match(r"test_", path.name) or ".venv" in rel.parts:
            continue
        try:
            out[rel.as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return out


def own_scope_nodes(fn: ast.AST) -> list[ast.AST]:
    """The nodes of a function that are its own: its statements, without the bodies of the functions, classes and lambdas
    defined inside it, except a nested function the function itself calls (by name) from its own scope."""
    nodes: list[ast.AST] = []
    nested: dict[str, ast.AST] = {}
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested[node.name] = node
            continue
        if isinstance(node, (ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    used: set[str] = set()
    grew = True
    while grew:
        grew = False
        called = {n.func.id for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for name, inner in nested.items():
            if name in called and name not in used:
                used.add(name)
                nodes.extend(ast.walk(inner))
                grew = True
    return nodes
