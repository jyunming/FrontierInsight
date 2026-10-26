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

A loop or comprehension variable called ``cell`` is not one of these when what it walks is a list the script builds itself from
its own summaries (``cells.append(result)``, :func:`_fresh_summaries`, a strict allow-list: a list filled only with new dict literals); a list FI supplies, a
parameter or anything it cannot see through is still read as settings.

What counts as a read: ``setting["k"]`` and ``setting.get("k"[, default])`` with a literal ``k``. Names written in a
chain of fallbacks (``cell.get("a", cell.get("b", 1.0))`` or ``cell.get("a") or cell.get("b")``) are one read that is
satisfied by any one of them, so a code that accepts several spellings is flagged only when none of them is a name
the study has. The fields of FI's per-cell record itself (``cell``, ``key``, ``metrics`` ...) are not settings. The check
is conservative: a read with a fallback value of a name that is like none of the study's settings is an optional setting
the study never sets, and is left alone; a computed key, ``in`` tests, ``**`` spreads and a setting passed on to a function are not followed.

A constant the plan itself defines is not a setting the study lacks. ``cell.get("baseline_rate", 80.0)`` reads a name the grid does
not have, and its only likeness to the swept ``rate`` is a shared word; when the plan states that very number as the value of that name (a structured field, or one short clause such as
the name's words "of 80") (``a declared baseline rate of 80``), the fallback IS the plan's constant (:func:`plan_defined`)
and is left alone. A name spelt almost like a swept setting (``rat``) is still a collapsed axis, whatever the plan says. And a repair
never turns the plan's constant into the swept setting: :func:`substituted` finds that, and a repair that does it is not kept.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stop_power import READ

#: What a fault found here rests on (core/stop_power.py): FI's reading of the script's text.
SOURCE = READ

#: Names that hold one setting (one cell of the grid).
SETTING_NAMES = frozenset({"cell", "setting", "settings"})
#: The trial contract's entry points: their first parameter is the setting.
ENTRIES = ("run_cell", "run_trial")
#: The fields of the per-cell record FI writes for the analysis (``raw/trials.json``): not settings.
RECORD_FIELDS = frozenset({"cell", "key", "planned", "ok", "failed", "metrics", "trials", "rows", "values"})
#: The fields FI adds to the record of a study that compares values inside each run (core/in_run.py): names only such a
#: study has.
IN_RUN_RECORD_FIELDS = frozenset({"level", "compared_in_each_run"})
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


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _params(fn: ast.AST) -> list[str]:
    args = fn.args  # type: ignore[attr-defined]
    return [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs,
                            *(x for x in (args.vararg, args.kwarg) if x is not None))]


def _locals(fn: ast.AST) -> set[str]:
    """The names a function binds itself (its parameters, and names it assigns that it does not declare ``nonlocal``
    or ``global``): inside it, they are not the enclosing function's."""
    own = set(_params(fn))
    declared: set[str] = set()
    nodes = _own_nodes(fn)
    # a comprehension's variable is the comprehension's own, not the function's
    looped = {id(n) for c in nodes if isinstance(c, ast.comprehension) for n in ast.walk(c.target)}
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in looped:
            own.add(node.id)
        elif isinstance(node, (ast.Nonlocal, ast.Global)):
            declared.update(node.names)
        elif isinstance(node, _FUNCTIONS[:2]) or isinstance(node, ast.ClassDef):
            own.add(node.name)
    return own - declared


def _grown(scope: ast.AST, names: set[str]) -> set[str]:
    """``names`` with every name ``scope`` assigns from a setting (``c = cell``, ``c = entry["cell"]``) and every loop
    variable called ``cell`` over anything but the script's own summaries: the names that hold a setting in it."""
    names = set(names)
    nodes = sorted(_own_nodes(scope), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    for node in nodes:
        if isinstance(node, ast.Assign) and _setting_like(node.value, names):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id in SETTING_NAMES:
            if not _fresh_summaries(node.iter, nodes):
                names.add(node.target.id)
    return names


def _given_settings(fn: ast.Lambda, outer: ast.AST, holders: set[str], parent: dict[int, ast.AST]) -> set[str]:
    """The parameters of the lambda ``fn`` (in ``outer``, whose names ``holders`` hold a setting) that are handed a
    setting: by a call of it with one (``(lambda c: ...)(cell)``, or ``f(cell)`` after ``f = lambda c: ...`` in
    ``outer``), or by a default (``lambda t, c=cell: ...``). A lambda's parameter is not a setting by its name: the
    ``cell`` of ``max(rows, key=lambda cell: cell["score"])`` is one of the rows."""
    positional = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
    given = {a.arg for a, d in zip(fn.args.args[::-1], fn.args.defaults[::-1]) if _setting_like(d, holders)}
    given |= {a.arg for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None and _setting_like(d, holders)}
    up = parent.get(id(fn))
    calls = [up] if isinstance(up, ast.Call) and up.func is fn else []
    if isinstance(up, ast.Assign) and up.value is fn and len(up.targets) == 1 and isinstance(up.targets[0], ast.Name):
        name = up.targets[0].id
        calls += [n for n in _own_nodes(outer) if isinstance(n, ast.Call) and _is_name(n.func, name)]
    for call in calls:
        given |= {p for p, arg in zip(positional, call.args) if _setting_like(arg, holders)}  # type: ignore[union-attr]
        given |= {k.arg for k in call.keywords if k.arg and _setting_like(k.value, holders)}  # type: ignore[union-attr]
    return given


def _scopes(tree: ast.Module) -> list[tuple[ast.AST, set[str]]]:
    """Each function and lambda with the names in it that hold a setting, and the module body with its own.

    A function or lambda defined inside another function sees the names of the enclosing one that it does not bind
    itself, so a setting read in a closure or a callback (``lambda: cell["sigma"]``, ``solve_ivp(lambda t, y:
    -cell["k"] * y, ...)``, ``def model(x): return x * c["a"]``) is a read of the function it is in; so is one in a
    lambda of the module body. A function defined at the top of the module does not see the module's names (as
    before). A lambda's own parameter holds a setting only when the lambda is handed one (:func:`_given_settings`)."""
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    def enclosing(node: ast.AST) -> ast.AST | None:
        up = parent.get(id(node))
        while up is not None and not isinstance(up, _FUNCTIONS):
            up = parent.get(id(up))
        if up is None and isinstance(node, ast.Lambda):
            return tree
        return up

    held: dict[int, set[str]] = {id(tree): _grown(tree, set(SETTING_NAMES))}
    out: list[tuple[ast.AST, set[str]]] = []
    for node in ast.walk(tree):  # breadth first: an enclosing function before the ones inside it
        if not isinstance(node, _FUNCTIONS):
            continue
        args = _params(node)
        outer = enclosing(node)
        if isinstance(node, ast.Lambda):
            names = _given_settings(node, outer, held.get(id(outer), set()), parent) if outer is not None else set()
        else:
            names = {a for a in args if a in SETTING_NAMES}
            if node.name in ENTRIES and args:
                names.add(args[0])
        if outer is not None:
            mine = _locals(node)
            names |= {n for n in held.get(id(outer), set()) if n not in mine}
        held[id(node)] = _grown(node, names)
        out.append((node, names))
    out.append((tree, set(SETTING_NAMES)))
    return out


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of ``scope`` that are not inside a nested function or lambda (those have their own scope)."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        found.append(node)
        if isinstance(node, _FUNCTIONS):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return found


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
_LOOKS = frozenset({"values", "keys", "items", "get", "copy"})


def _assignments_of(name: str, nodes: list[ast.AST]) -> list[ast.AST] | None:
    """The values ``name`` is assigned in ``nodes``, or ``None`` when it is also bound or changed some other way (``+=``, a
    loop variable, ``:=``, ``global``, a tuple target, a ``with`` / ``import`` name)."""
    values: list[ast.AST] = []
    for node in nodes:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            if node.value is not None:  # ``cells: list[dict] = []`` binds like ``cells = []``; ``cells: list`` binds nothing
                values.append(node.value)
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if len(node.targets) != 1:
                return None
            values.append(node.value)
        elif (isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.comprehension, ast.NamedExpr))
              and any(isinstance(t, ast.Name) and t.id == name for t in ast.walk(node.target))):
            return None
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return None
        elif isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
                                                   for t in node.targets for n in ast.walk(t)):
            return None  # a tuple target
        elif isinstance(node, (ast.With, ast.AsyncWith)) and any(
                o.optional_vars is not None and any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(o.optional_vars))
                for o in node.items):
            return None
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == name for a in node.names):
            return None
    return values


def _fresh_dict(value: ast.AST, nodes: list[ast.AST]) -> bool:
    """Whether ``value`` is positively a new summary dict: a dict literal with no ``**`` entry, or a name assigned exactly once
    in this scope from one and never changed with ``.update`` / ``|=``."""
    if isinstance(value, ast.Dict):
        return all(k is not None for k in value.keys)
    if not isinstance(value, ast.Name):
        return False
    got = _assignments_of(value.id, nodes)
    if got is None or len(got) != 1 or not (isinstance(got[0], ast.Dict) and all(k is not None for k in got[0].keys)):
        return False
    name = value.id
    for n in nodes:
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and _is_name(n.func.value, name) and n.func.attr not in _LOOKS:
                return False
            handed = [*n.args, *(k.value for k in n.keywords)]
            appended = isinstance(n.func, ast.Attribute) and n.func.attr == "append"
            if any(isinstance(a, ast.Starred) and _is_name(a.value, name) for a in n.args):
                return False
            if not appended and any(_is_name(a, name) for a in handed):
                return False  # handed to a call that may change it
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            if any(isinstance(sub, ast.Subscript) and _is_name(sub.value, name) for t in targets for sub in ast.walk(t)):
                return False  # ``name[k] = v``, ``name[k] += v``, ``del name[k]``: its contents change
    return True


def _fresh_summaries(iterable: ast.AST, nodes: list[ast.AST], depth: int = 0) -> bool:
    """Whether ``iterable`` is a local list of the script's own summaries, which a loop variable called ``cell`` then walks
    without holding a setting. A strict allow-list: it must be positively known, and anything else is read as settings.
    Recognised: a name assigned once as an empty ``[]`` / ``{}`` and filled only by ``.append(x)`` / ``name[k] = x`` with at
    least one fill, every ``x`` a fresh summary dict (:func:`_fresh_dict`); or a name assigned once from a list comprehension
    that keeps (does not change) the items of such a list (``[c for c in cells if ...]``). A parameter, FI's record, a call, a
    subscript, an ``extend``, a ``dict(...)`` or a value of unknown origin among the fills is not recognised."""
    while True:  # ``sorted(x)``, ``list(x)``, ``x.values()`` walk the same items
        if (isinstance(iterable, ast.Call) and len(iterable.args) == 1 and not iterable.keywords
                and isinstance(iterable.func, ast.Name) and iterable.func.id in ("sorted", "list", "tuple", "reversed")):
            iterable = iterable.args[0]
        elif (isinstance(iterable, ast.Call) and not iterable.args and not iterable.keywords
              and isinstance(iterable.func, ast.Attribute) and iterable.func.attr == "values"):
            iterable = iterable.func.value
        else:
            break
    if not isinstance(iterable, ast.Name) or depth > 4:
        return False
    name = iterable.id
    assigned = _assignments_of(name, nodes)
    if assigned is None or len(assigned) != 1:
        return False
    fills: list[ast.AST] = []
    for node in nodes:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _is_name(node.func.value, name):
            if node.func.attr in _LOOKS:
                continue
            if node.func.attr != "append" or len(node.args) != 1 or node.keywords:
                return False
            fills.append(node.args[0])
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) and _is_name(t.value, name) for t in node.targets):
            fills.append(node.value)
        elif isinstance(node, (ast.AugAssign, ast.Delete)) and any(
                isinstance(n, ast.Name) and n.id == name for n in ast.walk(node)):
            return False
    value = assigned[0]
    if isinstance(value, ast.ListComp):
        if fills or len(value.generators) != 1:
            return False
        g = value.generators[0]
        return (isinstance(g.target, ast.Name) and _is_name(value.elt, g.target.id)
                and _fresh_summaries(g.iter, nodes, depth + 1))
    empty = (isinstance(value, ast.List) and not value.elts) or (isinstance(value, ast.Dict) and not value.keys)
    return empty and bool(fills) and all(_fresh_dict(f, nodes) for f in fills)


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


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
        # A name assigned from a setting holds one too (``c = entry["cell"]``), in the order the code assigns them. A loop
        # or comprehension variable called ``cell`` holds one, unless it walks a list of the script's own summaries
        # (:func:`_fresh_summaries`): then it is not one inside that loop or comprehension, whatever the outer scope
        # makes of the name (the module body treats ``cell`` as a setting throughout).
        shadow: dict[int, set[str]] = {}

        def shade(inner: list[ast.AST], drop: set[str], add: set[str]) -> None:
            for n in inner:
                shadow[id(n)] = (shadow.get(id(n), names) - drop) | add

        for node in nodes:
            if isinstance(node, ast.Assign) and _setting_like(node.value, names):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id in SETTING_NAMES:
                if not _fresh_summaries(node.iter, nodes):
                    names.add(node.target.id)
        # Each loop or comprehension that binds the name sets it for its own nodes: dropped over the script's own
        # summaries, a setting holder over anything else. Outer ones come first (``nodes`` is in source order), so the
        # innermost binding wins.
        for node in nodes:
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id in SETTING_NAMES:
                fresh = _fresh_summaries(node.iter, nodes)
                inner = [n for stmt in (*node.body, *node.orelse) for n in ast.walk(stmt)]
                shade(inner, {node.target.id} if fresh else set(), set() if fresh else {node.target.id})
            elif isinstance(node, _COMPREHENSIONS):
                drop = {g.target.id for g in node.generators if isinstance(g.target, ast.Name)
                        and g.target.id in SETTING_NAMES and _fresh_summaries(g.iter, nodes)}
                add = {g.target.id for g in node.generators if isinstance(g.target, ast.Name)
                       and g.target.id in SETTING_NAMES} - drop
                if drop or add:
                    shade(list(ast.walk(node)), drop, add)
        for node in nodes:
            hit = _read_of(node, shadow.get(id(node), names))
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
    """``(the study's axes, every name a setting may have)``: the axes (the grid's, and what each run compares inside it,
    which every run's cell carries as a list), the settings the plan fixes, and the keys of the cases its checks run at."""
    from . import in_run as _in_run
    from . import oracle_forms as _forms

    protocol = protocol if isinstance(protocol, dict) else {}
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    inside = _in_run.of_study(protocol) or {}
    axes = [str(k) for k in grid] + [str(k) for k in inside if str(k) not in grid]
    allowed = set(axes) | (set(IN_RUN_RECORD_FIELDS) if inside else set())
    allowed.update(_forms.fixed_settings_of_design({**(design or {}), "protocol": protocol}))
    _case_keys(protocol, allowed)
    return axes, allowed


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name).lower()) if len(t) >= 3}


def _alike(key: str, supplied: list[str] | set[str]) -> bool:
    """Whether ``key`` is spelt like a supplied name: one contains the other, or they differ by a letter or two. Sharing a
    word is not enough (``baseline_rate`` and ``rate`` are two quantities)."""
    low = key.lower()
    pool = sorted(supplied)
    return (any((len(s) >= 3 and s.lower() in low) or (len(low) >= 3 and low in s.lower()) for s in pool)
            or bool(difflib.get_close_matches(key, pool, n=1, cutoff=0.75)))


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


_RUNNER_WORDS = ("run_trial", "run_cell", "simulate")
_RUNNER_KEYWORDS = frozenset({"runs_per_setting", "n_trials", "num_trials", "trials", "n_runs", "num_runs", "runs"})


def _mentions(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _is_length(node: ast.AST) -> bool:
    """``len(x)``, ``x.shape[0]``, ``x.size``: the number of items something holds."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
        return True
    if isinstance(node, ast.Attribute) and node.attr in ("size", "shape"):
        return True
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape"


def _mentions_trial_source(node: ast.AST) -> bool:
    """Whether a node's text mentions FI_TRIALS, trials.json, ledger, or calls load_trials/read_trials."""
    # Check via unparsed text for all keywords (handles string constants like environ["FI_TRIALS"])
    text = ast.unparse(node)
    if any(kw in text for kw in ["FI_TRIALS", "trials.json", "ledger"]):
        return True

    # Check for function calls with load_trials or read_trials in the name
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func_name = ""
            if isinstance(n.func, ast.Name):
                func_name = n.func.id
            elif isinstance(n.func, ast.Attribute):
                func_name = n.func.attr
            if any(kw in func_name.lower() for kw in ["load_trials", "read_trials"]):
                return True

    return False


def _is_trial_records_length(node: ast.AST, tree: ast.Module) -> bool:
    """Whether a length expression measures the trial records (not a resample or seed count).

    Returns True for len(X), X.shape[0], X.size, X.shape when X satisfies:
    (a) X is a name containing one of: trials, trial, records, record, rows, results, runs (whole word, case-insensitive)
    (b) X is a name assigned from an expression mentioning FI_TRIALS, trials.json, ledger, load_trials, or read_trials,
        OR assigned inside a function whose name or parameters contain trial-related keywords
    (c) The expression itself mentions FI_TRIALS
    """
    if not _is_length(node):
        return False

    # Extract the variable being measured
    var_node = None
    var_name = None

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" and node.args:
        var_node = node.args[0]
    elif isinstance(node, ast.Attribute) and node.attr in ("size", "shape"):
        var_node = node.value
    elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
        var_node = node.value.value

    if var_node is None:
        return False

    # Extract variable name if it's a simple Name node
    if isinstance(var_node, ast.Name):
        var_name = var_node.id

    # Check rule (a): name contains trial-related keywords
    if var_name:
        keywords = {"trials", "trial", "records", "record", "rows", "results", "runs"}
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = set(re.split(r"[^a-z0-9]+", var_name.lower()))
        if tokens & keywords:
            return True

    # Check rule (c): the expression itself mentions FI_TRIALS
    if _mentions_trial_source(var_node):
        return True

    # Build set of functions related to trial records (for rule b extension)
    trial_related_funcs = set()
    func_keywords = {"trial", "trials"}
    for node_in_tree in ast.walk(tree):
        if isinstance(node_in_tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check function name
            func_tokens = set(re.split(r"[^a-z0-9]+", node_in_tree.name.lower()))
            if func_tokens & func_keywords:
                trial_related_funcs.add(node_in_tree.name)
            # Check parameter names
            for arg in node_in_tree.args.args + node_in_tree.args.posonlyargs + node_in_tree.args.kwonlyargs:
                arg_tokens = set(re.split(r"[^a-z0-9]+", arg.arg.lower()))
                if arg_tokens & func_keywords:
                    trial_related_funcs.add(node_in_tree.name)

    # Check rule (b): variable assigned from expression mentioning trial-source keywords,
    # or assigned inside a trial-related function
    if var_name:
        for stmt in ast.walk(tree):
            value = None
            target_names = []

            if isinstance(stmt, ast.Assign):
                value = stmt.value
                target_names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                value = stmt.value
                if isinstance(stmt.target, ast.Name):
                    target_names = [stmt.target.id]
            else:
                continue

            if var_name not in target_names or value is None:
                continue

            # Check if directly mentions trial source
            if _mentions_trial_source(value):
                return True

            # Check if inside a trial-related function
            for func_node in ast.walk(tree):
                if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if func_node.name in trial_related_funcs:
                        # Check if stmt is inside this function
                        for sub_stmt in ast.walk(func_node):
                            if sub_stmt is stmt:
                                return True

    return False


def _calls_runner(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ""
            if any(w in f.lower() for w in _RUNNER_WORDS):
                return True
    return False


def _controls_trial_records(tree: ast.Module, name: str) -> bool:
    """Whether the constant ``name`` decides how many trial records are produced or expected: the bound of a loop whose body
    calls the simulation, a length of the trial records compared with it (an assertion or a check of the records received), or the count passed
    to a runner by keyword. Used only as the size of a resample, a count of seeds or an inner iteration count, it is not."""
    for n in ast.walk(tree):
        if isinstance(n, (ast.For, ast.AsyncFor)) and _mentions(n.iter, name) and any(_calls_runner(b) for b in n.body):
            return True
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)) and any(
                _mentions(g.iter, name) for g in n.generators) and _calls_runner(
                    n.elt if not isinstance(n, ast.DictComp) else ast.Tuple(elts=[n.key, n.value], ctx=ast.Load())):
            return True  # the same loop written as a comprehension
        if isinstance(n, ast.Compare):
            parts = [n.left, *n.comparators]
            if any(isinstance(p, ast.Name) and p.id == name for p in parts) and any(_is_trial_records_length(p, tree) for p in parts):
                return True
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg and kw.arg.lower() in _RUNNER_KEYWORDS and _mentions(kw.value, name):
                    return True
    return False


def counts(source: str, script: str, planned: int) -> list[Read]:
    """The module-level whole-number literals in ``source`` that a name saying it counts trials or runs holds, other than
    ``planned``, and that the script uses to decide how many trial records are produced or expected (see
    :func:`_controls_trial_records`). Never raises: a script that does not parse has none to report."""
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
            if (isinstance(target, ast.Name) and _COUNT_NAME.match(target.id) and value.value != planned
                    and _controls_trial_records(tree, target.id)):
                out.append(Read(script=script, line=int(node.lineno), keys=(target.id,), default=str(value.value),
                                planned=planned))
    return out


def _planned_runs(protocol: dict[str, Any] | None) -> int | None:
    runs = protocol.get("runs_per_setting") if isinstance(protocol, dict) else None
    if isinstance(runs, bool) or not isinstance(runs, (int, float)) or runs != int(runs) or runs < 1:
        return None
    return int(runs)


_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_WINDOW = 60


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _distinct_words(key: str, supplied: list[str] | set[str]) -> list[str]:
    """The words of ``key`` that no supplied name has: what makes it another quantity (``baseline`` of
    ``baseline_rate``, beside ``rate``)."""
    have: set[str] = set()
    for name in supplied:
        have |= _tokens(name)
    return sorted(_tokens(key) - have)


_BREAK = re.compile(r"[;,\n()\[\]{}]|\.(?:\s|$)|\b(?:but|while|whereas|and|with|which|when|where)\b")
_BINDER = re.compile(r"(?:\b(?:of|is|are|at|to|equals|equal to|set to|fixed at)\b|=|:)\s*(?:about\s+|approximately\s+)?$")
_UNIT = re.compile(r"^\s*([A-Za-z%/][\w/%.^*-]*)\s+")
#: Words allowed between the last word of a name and the ``of`` / ``is`` / ``=`` that gives its value.
_BOUND_WORDS = 2


def _structured_value(node: Any, key: str, value: float, allowed: set[str]) -> bool:
    """Whether the plan's structured fields (the protocol, the design) hold ``key: value``: a mapping whose key is the name
    (case and punctuation ignored) and whose value is that number, or an entry that names it (``name`` / ``id``) and gives
    it as its ``value``."""
    want = re.sub(r"[^a-z0-9]", "", key.lower())

    def same(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and abs(float(x) - value) <= 1e-9 * max(1.0, abs(value))

    def walk(n: Any) -> bool:
        if isinstance(n, dict):
            for k, v in n.items():
                if re.sub(r"[^a-z0-9]", "", str(k).lower()) == want and (same(v) or (
                        isinstance(v, dict) and same(v.get("value")))):
                    return True
            label = re.sub(r"[^a-z0-9]", "", str(n.get("name") or n.get("id") or "").lower())
            if label == want and same(n.get("value")):
                return True
            return any(walk(v) for v in n.values())
        if isinstance(n, (list, tuple)):
            return any(walk(v) for v in n)
        return False

    return walk(node)


def _word_end(text: str, word: str) -> int:
    """Where the last whole-word occurrence of ``word`` in ``text`` ends (``-1`` when it is not there as a word: ``rate`` is
    not in ``accelerate``)."""
    ends = [m.end() for m in re.finditer(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text)]
    return ends[-1] if ends else -1


def _stated_as_value(low: str, words: list[str], value: float) -> bool:
    """Whether the prose states ``value`` AS the value of the thing the ``words`` name, in one short clause: ``<words> of 80``,
    ``<words> = 80``, ``<words> is 80``, or ``80 <unit> <words>``; no other number between them and no clause break
    (a comma, a semicolon, a bracket, ``with`` / ``and`` / ``but`` ...)."""
    for m in _NUMBER.finditer(low):
        n = _number(m.group(0))
        if n is None or abs(n - value) > 1e-9 * max(1.0, abs(value)):
            continue
        before = low[max(0, m.start() - _WINDOW): m.start()]
        cuts = list(_BREAK.finditer(before))
        clause = before[cuts[-1].end():] if cuts else before
        binder = _BINDER.search(clause.rstrip() + " ")
        if all(_word_end(clause, w) >= 0 for w in words) and binder and not _NUMBER.search(clause):
            # The binder must follow the name itself: at most a few words (``reference line dose of 80``) between the
            # last of its words and ``of`` / ``is`` / ``=``. "the rate is estimated from a sample of 80" binds 80 to the
            # sample, not to the rate.
            last = max(_word_end(clause, w) for w in words)
            between = clause[last:binder.start()].split()
            if len(between) <= _BOUND_WORDS:
                return True
        after = low[m.end(): m.end() + _WINDOW]
        cut = _BREAK.search(after)
        tail = after[:cut.start()] if cut else after
        unit = _UNIT.match(tail + " ")
        if unit and all(_word_end(tail[unit.end():], w) >= 0 for w in words) and not _NUMBER.search(tail) and len(tail.split()) <= 6:
            return True
    return False


def plan_defined(r: Read, allowed: set[str], plan_text: str, structured: Any = None) -> bool:
    """Whether a read with a fixed fallback is the plan's own constant: the fallback is a plain number, no name read is spelt
    like a setting of the study, and the plan states that number AS the value of that name, either (a) in a structured field
    whose key is the name (the protocol's fixed settings, constants or thresholds, an optimisation ``fixed`` block, an
    assumption given as ``key: value``) or (b) in prose, the name's distinguishing words (words the study's settings do
    not have) and the number bound in one short clause (:func:`_stated_as_value`). A number that merely stands near
    those words is not enough. Only then is the fallback the plan's constant and not a swept setting under another name."""
    if not r.fallback:
        return False
    value = _number(r.default)
    if value is None:
        return False
    low = (plan_text or "").lower()
    for key in r.keys:
        if _alike(key, allowed):
            continue
        if structured is not None and _structured_value(structured, key, value, allowed):
            return True
        words = _distinct_words(key, allowed)
        if words and low and _stated_as_value(low, words, value):
            return True
    return False


def _unwrap(node: ast.AST) -> ast.AST:
    """``float(x)`` / ``int(x)`` / ``np.float64(x)`` -> ``x`` (one level)."""
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if name in ("float", "int", "float64", "float32", "int64", "abs"):
            return node.args[0]
    return node


def _setting_assigns(source: str) -> dict[tuple[str, str], tuple[tuple[str, ...], str, int]]:
    """``{(function, name): (the setting names it reads, the fallback, line)}`` for each ``name = setting.get(...)`` /
    ``setting["k"]`` assignment (a chain of fallbacks is one read; the fallback is its last default)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    out: dict[tuple[str, str], tuple[tuple[str, ...], str, int]] = {}
    for scope, names in _scopes(tree):
        names = set(names)
        label = getattr(scope, "name", "<module>")
        for node in sorted(_own_nodes(scope), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
            if isinstance(node, ast.Assign) and _setting_like(node.value, names):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
                continue
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                continue
            keys: list[str] = []
            cur: ast.AST | None = _unwrap(node.value)
            default = ""
            while cur is not None:
                hit = _read_of(cur, names)
                if hit is None:
                    default = _text(cur) if keys else ""
                    break
                keys.append(hit[0])
                cur = _unwrap(hit[1]) if hit[1] is not None else None
            if keys:
                out[(label, node.targets[0].id)] = (tuple(dict.fromkeys(keys)), default, int(node.lineno))
    return out


def substituted(old_files: dict[str, str], new_files: dict[str, str], protocol: dict[str, Any] | None,
                design: dict[str, Any] | None = None) -> list[str]:
    """What a repair did that it must not: where ``old_files`` read a constant the plan itself defines (a fixed reference
    value with a fallback, :func:`plan_defined`) into a name, ``new_files`` read a setting the study sweeps into the same
    name of the same function. The plan's constant is not the swept setting; the fault is the fallback, and it is mended by
    keeping the constant (read from the plan, or defined once under its own name). One sentence each; empty when none."""
    from . import typed_constants as _constants

    axes, allowed = supplied_names(protocol, design)
    if not axes:
        return []
    plan_text = _constants.plan_text_of(protocol, design)
    found: list[str] = []
    for script, text in old_files.items():
        if script not in new_files:
            continue
        after = _setting_assigns(new_files[script])
        for (fn, name), (keys, default, line) in _setting_assigns(text).items():
            if set(keys) & allowed or not plan_defined(Read(script, line, keys, default, fallback=bool(default)), allowed, plan_text, (protocol, design)):
                continue
            new = after.get((fn, name))
            if new is not None and set(new[0]) & set(axes) and not (set(new[0]) - allowed):
                found.append(f"{script} line {line}: `{name}` was the plan's constant ({default}) and now reads the swept setting "
                             + " or ".join(f"`{k}`" for k in new[0]))
    return found


def rejects(old_files: dict[str, str], new_files: dict[str, str], protocol: dict[str, Any] | None,
            design: dict[str, Any] | None = None) -> str:
    """Why a repair of these files is not kept (:func:`substituted`), else ``""``."""
    got = substituted(old_files, new_files, protocol, design)
    return ("it replaced a constant the plan defines with a setting the study sweeps (" + "; ".join(got[:2]) + ")") if got else ""


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
    from . import typed_constants as _constants

    plan_text = _constants.plan_text_of(protocol, design)
    for script, text in files.items():
        for r in reads(text, script):
            if set(r.keys) & allowed:
                continue
            if plan_defined(r, allowed, plan_text, (protocol, design)):
                continue  # the plan's own constant, with its fallback: not a setting the study lacks
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
            "and no error would show. Read each setting by the name the study gives it, with no fallback value. If the "
            "name is a fixed value the PLAN itself defines (a reference value, not a setting the study varies), it is not "
            "a setting: define it once under its own name in the code with the value the plan gives, and never replace it "
            "with a setting the study sweeps")
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
        # code/skills/: FI's copy of the selected skills' files (Engine._skill_files_to_code), the skill's code.
        if "__pycache__" in rel.parts or "tests" in rel.parts or re.match(r"test_", path.name) or ".venv" in rel.parts \
                or rel.parts[0] == "skills":
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
