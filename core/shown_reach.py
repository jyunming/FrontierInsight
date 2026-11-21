"""What a fault a failing test showed does to the study's results, read from FI's own records instead of from the words of the
reader's finding (the evidence gate, ``Engine._shown_fault_outcome``).

The gate used to look the finding's quantity up in the run's printed results. FI's frame writes the results of a study in two
places: the trial-level numbers the run prints, and what ``derive_<id>`` works out afterwards, which lives only in
``data/results/derived.json``. A finding about a derived result was therefore "not found" and stopped the paper, as did a
finding about a figure and about a helper no result reaches. Each fact is
now read where FI holds it:

* the failing test says which function it calls (``derive_<id>``, ``figure_<id>``, the trial, a helper);
* the plan's contract (``needs/FROZEN_PROTOCOL.json``) lists the declared results, the standard error and interval ends of a
  result with a true value (``known_value``), and the plan's first dependent variable names the main result;
* FI's record of the trials (``raw/trials.json``) lists the names a trial returned;
* FI's record of its analysis (``data/results/derived.json``) lists what each derived result was worked out from and the file
  of each figure;
* the code itself (:mod:`core.code_reach`) says which result's function reaches a helper.

The answer for one finding is an :class:`Effect`: a stop (the main result is among what the fault reaches, or nothing would be left), or the
results to leave out, the plan items not done and the figures to drop. ``None`` when the records cannot place the finding:
the gate then judges it as before.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import code_review as _code_review


@dataclass
class Effect:
    """What one shown fault does: ``stop`` (a reason; the paper has nothing to report), else the declared results to leave out
    (``left_out``), the plan items the code does not return (``not_done``) and the figures to drop (``figures``, ids)."""

    stop: str = ""
    left_out: list[str] = field(default_factory=list)
    not_done: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    plan_text: bool = False  # the finding is about the plan's own text, not a computed number
    probes: list[str] = field(default_factory=list)  # names the test looked for that the plan does not declare
    why: str = ""
    extra_keys: list[str] = field(default_factory=list)  # printed keys to leave out that no declared result owns
    named: str = ""  # the quantity the finding is about, among the left-out ones
    cannot: bool = False  # FI cannot tell what the code reaches: the records place nothing
    empty: bool = False  # the stop is that leaving out what it changes leaves no result at all
    by_trial: bool = False  # a helper reached through the trial: the call graph ties it to every result of the trial, to none in particular
    unreached: bool = False  # the helper is reached by no function of a result or a figure (:meth:`Reach._helper_effect`)


def union(a: "Effect", b: "Effect") -> "Effect":
    """The effect of two placements of one fault: a stop in either is a stop, and what either leaves out is left out."""
    def both(x: list[str], y: list[str]) -> list[str]:
        return list(dict.fromkeys([*x, *y]))

    return Effect(stop=a.stop or b.stop, left_out=both(a.left_out, b.left_out), not_done=both(a.not_done, b.not_done),
                  figures=both(a.figures, b.figures), plan_text=False, probes=both(a.probes, b.probes),
                  why=a.why or b.why, extra_keys=both(a.extra_keys, b.extra_keys),
                  named=a.named or b.named, empty=a.empty or b.empty)


_BUILTIN_NAMES = frozenset(dir(builtins)) - {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint"}
#: Methods of strings and mappings: a call to one is not a call into the study's own code.
_PLAIN_METHODS = frozenset({"get", "items", "keys", "values", "join", "lower", "upper", "strip", "split", "format", "startswith",
                            "endswith", "count", "index", "append", "extend", "copy"})


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# ---- the failing test ---------------------------------------------------------------------------------------------------
@dataclass
class TestShape:
    """What a finding's test source does, read from its syntax: the functions it calls, the string keys it reads from what it
    got back (``exact``; ``prefix`` for an f-string key such as ``f"k_hat_{i}"``). What its assertions say (a value wrong, a
    value missing) is not read: FI holds no record that tells them apart."""

    calls: set[str]
    exact: set[str]
    prefix: set[str]
    frame: set[str] = field(default_factory=set)  # constants of FI's analysis frame it reads (``derive.ROLES``)
    defaulted: set[str] = field(default_factory=set)  # keys read by ``.get``/``.pop`` with a default (the default may be compared)
    #: ``(module, name)`` of each ``base.name(...)`` call whose base is a name the test imports (``module`` is the top-level module
    #: it came from), and the names of the method calls on any other value (a string, a mapping, a local): only the first kind
    #: can reach the study's own code. ``calls`` still holds every bare name.
    module_calls: set[tuple[str, str]] = field(default_factory=set)
    method_calls: set[str] = field(default_factory=set)
    bare_calls: set[str] = field(default_factory=set)  # the names called as ``name(...)``
    #: method calls whose receiver the test's own code does not prove to be a plain value (a parameter, a fixture, an alias of a
    #: module, ``getattr(...)``, an unresolved name): judged by their name against the study's functions, as a bare name is
    unproven_calls: set[str] = field(default_factory=set)


#: Builtins whose result is a plain value (a number, a string, a container), whatever they are given.
_VALUE_BUILTINS = frozenset({"str", "int", "float", "bool", "len", "list", "tuple", "set", "frozenset", "dict", "sorted", "repr", "abs",
                             "min", "max", "sum", "round", "range", "enumerate", "zip", "reversed", "any", "all", "bytes"})


def _value_reader(tree: ast.AST, bound: dict[str, str], constants: set[str]) -> Callable[[ast.AST], bool]:
    """A function that says whether an expression of the test is a plain value, proven from the test's own code: a literal, a value read from one of FI's
    analysis-frame constants (``derive.RESULTS``), a name that is such a value on every assignment the test makes, or the result of a plain builtin or a plain method of such a value. A
    parameter (a pytest fixture too), an import that is not a frame constant, a ``with`` target, a name bound by ``def`` or
    ``class``, and a name with no assignment in the test are not. Scope is ignored: one unsafe binding anywhere makes the name
    unsafe."""
    values: dict[str, list[ast.AST | None]] = {}

    def bind(target: ast.AST, value: ast.AST | None) -> None:
        if isinstance(target, ast.Name):
            values.setdefault(target.id, []).append(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                bind(t, value)
        elif isinstance(target, ast.Starred):
            bind(target.value, None)
        # an attribute or a subscript target binds no name

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                bind(t, n.value)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            bind(n.target, n.value if isinstance(n, ast.AugAssign) or n.value is not None else None)
        elif isinstance(n, ast.NamedExpr):
            bind(n.target, n.value)
        elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
            bind(n.target, n.iter)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            bind(n.optional_vars, None)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(n, ast.Lambda):
                values.setdefault(n.name, []).append(None)
            a = n.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, *([a.vararg] if a.vararg else []), *([a.kwarg] if a.kwarg else [])]:
                values.setdefault(arg.arg, []).append(None)
        elif isinstance(n, ast.ClassDef):
            values.setdefault(n.name, []).append(None)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            values.setdefault(n.name, []).append(None)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                values.setdefault((al.asname or al.name).split(".")[0], []).append(None)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            for nm in n.names:
                values.setdefault(nm, []).append(None)

    safe = set(values)

    def plain(node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp, ast.SetComp,
                             ast.DictComp, ast.GeneratorExp)):
            return True
        if isinstance(node, ast.Name):
            if node.id in bound:
                return node.id in constants  # a frame constant imported by name (``from derive import RESULTS``)
            return node.id in values and node.id in safe
        if isinstance(node, ast.Attribute):
            chain: ast.AST = node
            first = ""
            while isinstance(chain, ast.Attribute):
                first = chain.attr
                chain = chain.value
            return isinstance(chain, ast.Name) and chain.id in bound and first in constants
        if isinstance(node, ast.Subscript):
            return plain(node.value)
        if isinstance(node, ast.BinOp):
            return plain(node.left)
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                return f.id in _VALUE_BUILTINS and f.id not in values and f.id not in bound
            if isinstance(f, ast.Attribute):
                return f.attr in _PLAIN_METHODS and plain(f.value)
        return False

    while True:
        drop = {nm for nm in safe if not all(plain(v) for v in values[nm])}
        if not drop:
            break
        safe -= drop
    return plain


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def read_test(src: str) -> TestShape | None:
    """The shape of a finding's test source; ``None`` when it does not parse."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    calls: set[str] = set()
    exact: set[str] = set()
    prefix: set[str] = set()
    defaulted: set[str] = set()
    module_calls: set[tuple[str, str]] = set()
    method_calls: set[str] = set()
    bare_calls: set[str] = set()
    unproven_calls: set[str] = set()
    from . import analysis_frame as _frame

    constants = {n for n in _frame.FRAME_NAMES if n.isupper()}
    bound: dict[str, str] = {}  # a name the test imports -> the top-level module it comes from
    origin: dict[str, str] = {}  # a name imported under another name (``from m import f as g``) -> ``f``
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                bound[a.asname or a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                # a relative import names its module after the dot (``from . import model``, ``from .model import f``)
                source = (n.module or a.name) if n.level else (n.module or "")
                bound[a.asname or a.name] = source.split(".")[0]
                if a.asname:
                    origin[a.asname] = a.name
    is_value = _value_reader(tree, bound, constants)

    def attribute_call(f: ast.Attribute) -> None:
        base: ast.AST = f.value
        first = ""
        while isinstance(base, ast.Attribute):
            first = base.attr
            base = base.value
        if isinstance(base, ast.Name) and base.id in bound:
            if first in constants or (not first and base.id in constants):
                method_calls.add(f.attr)  # a method of a value FI wrote from the plan (``derive.RESULTS.get``)
            else:
                module_calls.add((bound[base.id], f.attr))
        elif is_value(f.value):
            method_calls.add(f.attr)
        else:
            unproven_calls.add(f.attr)

    def key(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            exact.add(node.value)
        elif isinstance(node, ast.JoinedStr):
            head = ""
            for v in node.values:
                if isinstance(v, ast.Constant):
                    head += str(v.value)
                else:
                    break
            if head:
                prefix.add(head)

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = _call_name(n)
            if name:
                calls.add(name)
            if isinstance(n.func, ast.Attribute):
                attribute_call(n.func)
            elif not isinstance(n.func, ast.Name):
                unproven_calls.add("<computed callee>")  # ``getattr(m, "f")()``, ``fns["f"]()``: what is called is not read
            else:
                bare_calls.add(n.func.id)
                if n.func.id in origin:
                    bare_calls.add(origin[n.func.id])  # called under an alias: the name it has in its own module counts too
                    calls.add(origin[n.func.id])
            if isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "pop") and n.args:
                key(n.args[0])
                given = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "default"), None)
                if given is not None and not (isinstance(given, ast.Constant) and given.value is None):
                    defaulted.add(str(n.args[0].value) if isinstance(n.args[0], ast.Constant) else "")
        elif isinstance(n, ast.Subscript):
            key(n.slice)
        elif isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
            key(n.left)
    frame = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in constants}
    frame |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names if a.name in constants}
    # a body named without being called (`derive.figure_fig1.__code__`) is still the body the test is about
    calls |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr.startswith(("derive_", "figure_"))}
    return TestShape(calls, exact, prefix, frame, defaulted, module_calls, method_calls, bare_calls, unproven_calls)


#: Calls an assertion may make on what it tests without comparing a value: whether something is there, how many there are.
_LOOK_CALLS = frozenset({"any", "all", "bool", "len"})


def _failed_asserts_only_look(src: str, failures: Any, looked: set[str]) -> bool:
    """Whether every assertion that failed (``failures``: its line and the names that held nothing there, recorded by FI as the
    test ran, :data:`core.finding_repair.CAPTURE`) only looks for the names in ``looked``: each comparison in it is a lookup of
    one of them (``"x" in r``, ``r.get("x") is None``, ``r["x"] == v``) or compares only names that held nothing; what it tests
    for truth is such a comparison, a name that held nothing, a lookup, or ``any``/``all``/``bool``/``len`` of them. A name that
    held nothing counts only when every value the test gives it is made by looking for one of the names (a lookup, or a
    comprehension one of whose conditions has a lookup of the name as a conjunct): a list emptied by a value filter may be
    empty because a value differs. These are accepted shapes of the test's syntax; any other shape keeps the stop. A failed
    assertion that compares any other value (``r == {...}``, ``v == 2``, ``check(r)``) may be about a value that differs.
    Read from the test's syntax at the line FI recorded, never from its message."""
    if not isinstance(failures, list) or not failures:
        return False
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return False
    names = {_norm(n) for n in looked}
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]

    def name_of(node: ast.AST) -> str:
        return _norm(node.value) if isinstance(node, ast.Constant) and isinstance(node.value, str) else ""

    def lookup(node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            return name_of(node.slice) in names
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "pop")
                and bool(node.args) and name_of(node.args[0]) in names)

    def member(c: ast.AST) -> bool:
        return (isinstance(c, ast.Compare) and len(c.ops) == 1 and isinstance(c.ops[0], (ast.In, ast.NotIn))
                and name_of(c.left) in names)

    def filters_by_name(value: ast.AST) -> bool:
        # what a name that held nothing was made from: a lookup of a looked-for name, or a comprehension that keeps only what
        # one of those names is found in (a conjunct of one of its conditions); anything else may be empty for a value
        if lookup(value):
            return True
        if not isinstance(value, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return False
        for gen in value.generators:
            for cond in gen.ifs:
                parts = cond.values if isinstance(cond, ast.BoolOp) and isinstance(cond.op, ast.And) else [cond]
                if any(member(x) or lookup(x) for x in parts):
                    return True
        return False

    made: dict[str, list[ast.AST]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                for leaf in ast.walk(t):
                    if isinstance(leaf, ast.Name):
                        made.setdefault(leaf.id, []).append(n.value if isinstance(t, ast.Name) else t)
        elif isinstance(n, ast.arg):
            made.setdefault(n.arg, []).append(n)
        else:
            # any other way a name is bound (augmented, annotated, a loop, a with, a walrus) is not a filter on the names
            bound = ([n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.For, ast.NamedExpr, ast.comprehension))
                     else [i.optional_vars for i in n.items if i.optional_vars is not None] if isinstance(n, ast.With)
                     else [])
            for target in bound:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name):
                        made.setdefault(leaf.id, []).append(n)
    #: the names whose every value in the test is made by looking for a looked-for name
    kept_by_name = {k for k, values in made.items() if values and all(filters_by_name(v) for v in values)}

    for failure in failures:
        line = failure.get("line") if isinstance(failure, dict) else None
        empty = {str(x) for x in (failure.get("empty") or [])} if isinstance(failure, dict) else set()
        node = next((a for a in asserts if isinstance(line, int) and a.lineno <= line <= (a.end_lineno or a.lineno)), None)
        if node is None:
            return False

        def nothing(x: ast.AST) -> bool:
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in _LOOK_CALLS:
                return bool(x.args) and all(nothing(a) or lookup(a) for a in x.args)
            return isinstance(x, ast.Constant) or (isinstance(x, ast.Name) and x.id in empty and x.id in kept_by_name)

        for c in (n for n in ast.walk(node.test) if isinstance(n, ast.Compare)):
            operands = [c.left, *c.comparators]
            if not (member(c) or any(lookup(o) for o in operands) or all(nothing(o) for o in operands)):
                return False

        def truth(x: ast.AST) -> bool:
            if isinstance(x, ast.BoolOp):
                return all(truth(v) for v in x.values)
            if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.Not):
                return truth(x.operand)
            if isinstance(x, ast.Compare) or lookup(x) or nothing(x):
                return True
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in _LOOK_CALLS:
                return bool(x.args) and all(truth(a.elt) if isinstance(a, (ast.GeneratorExp, ast.ListComp, ast.SetComp))
                                            else truth(a) for a in x.args)
            return False

        if not truth(node.test):
            return False
    return True


# ---- the records ------------------------------------------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _trial_names(root: Path) -> set[str]:
    """Every name a trial returned, from FI's record of the trials; empty before the first run."""
    data = _read_json(root / "raw" / "trials.json")
    cells = data.get("cells") if isinstance(data, dict) else None
    return {str(k) for c in (cells if isinstance(cells, list) else []) if isinstance(c, dict)
            for k in (c.get("metrics") if isinstance(c.get("metrics"), dict) else {})}


def _function_names(code: Path, rel: str) -> dict[str, ast.AST]:
    try:
        tree = ast.parse((code / rel).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return {}
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _unwritten(fn: ast.AST | None) -> bool:
    """A function not written yet (absent, or its body only raises): nothing can be read from it."""
    if fn is None:
        return True
    body = [s for s in getattr(fn, "body", []) if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    return not body or all(isinstance(s, ast.Raise) for s in body)


def _strings_of(fn: ast.AST) -> tuple[set[str], set[str]]:
    """The string constants a function names (exact) and the heads of its f-strings."""
    exact: set[str] = set()
    prefix: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            exact.add(n.value)
        elif isinstance(n, ast.JoinedStr):
            head = ""
            for v in n.values:
                if isinstance(v, ast.Constant):
                    head += str(v.value)
                else:
                    break
            if head:
                prefix.add(head)
    return exact, prefix


def _used_name(node: ast.AST) -> str:
    """The name of a function a node uses: called (``f(x)``, ``m.f(x)``), or only mentioned (``g = f``, ``{"k": f}``, ``h(f)``,
    ``from m import f as g``); ``""`` for any other node."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return node.attr
    if isinstance(node, ast.alias):
        return node.name
    return ""


def _called_from(fns: dict[str, ast.AST], start: str) -> set[str]:
    """The names used by function ``start`` and by the functions of the same file it uses, in turn."""
    seen, todo, called = {start}, [start], set()
    while todo:
        for n in ast.walk(fns[todo.pop()]):
            used = _used_name(n)
            if used:
                called.add(used)
                if used in fns and used not in seen:
                    seen.add(used)
                    todo.append(used)
    return called


def _printed_exactly(result: Any, wanted: set[str]) -> set[str]:
    """The keys of the printed results, at any depth, that are one of the names ``wanted`` (normalised, the whole name: no
    prefix, no longer or shorter form) and hold a finite number, or a list or mapping with one. NaN and infinity are no value."""
    import math

    def holds(v: Any) -> bool:
        if isinstance(v, bool) or v is None:
            return False
        if isinstance(v, (int, float)):
            return math.isfinite(v)
        if isinstance(v, dict):
            return any(holds(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return any(holds(x) for x in v)
        return False

    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if _norm(k) in wanted and holds(v):
                    out.add(str(k))
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(result)
    return out


class Reach:
    """The records of one quest, read once, and the question put to them: what does the fault a finding's test showed do to
    the results (:meth:`effect`)?"""

    def __init__(self, quest_root: Path, protocol: Any, headline: list[str], declared: list[str] | None = None) -> None:
        from . import analysis_frame as _frame
        from . import contract as _contract

        self.root = Path(quest_root)
        self.code = self.root / "code"
        self.contract = None
        if isinstance(protocol, dict) and protocol:
            try:
                self.contract = _contract.from_design({"protocol": protocol})
            except Exception:  # noqa: BLE001 - a plan that cannot be read places nothing
                self.contract = None
        c = self.contract
        self.ids: list[str] = []
        self.labels: dict[str, str] = {}
        self.owner: dict[str, str] = {}  # norm(name a trial returns for a result with a true value) -> the result's id
        self.estimate: set[str] = set()  # the norms of those names that are the result's own estimate
        self.counted: set[str] = set()  # results whose coverage and bias FI counts itself
        if c is not None and getattr(c, "form", "") == "contract" and (c.results or c.derived):
            for r in c.results:
                self.ids.append(r.id)
                self.labels[r.id] = r.label
                kv = r.known_value if isinstance(r.known_value, dict) else {}
                for n in [kv.get("standard_error") or kv.get("variance"), *(kv.get("interval") or ())]:
                    if isinstance(n, str) and n:
                        self.owner[_norm(n)] = r.id
                if isinstance(kv.get("estimate"), str) and kv.get("estimate"):
                    self.owner[_norm(kv["estimate"])] = r.id
                    self.estimate.add(_norm(kv["estimate"]))
            for d in c.derived:
                if d.id not in self.ids:
                    self.ids.append(d.id)
                    self.labels[d.id] = d.label
            try:
                self.counted = set(_frame._known_values(protocol, c))
            except Exception:  # noqa: BLE001
                self.counted = set()
        elif declared:
            self.ids = list(dict.fromkeys(str(n) for n in declared))
        self.by_norm = {_norm(i): i for i in self.ids}
        self.headline = _code_review.headline_ids(headline, self.ids, self.labels, _contract.family_labels(protocol))
        self.returned = {_norm(n) for n in _trial_names(self.root)}
        derived = _read_json(self.root / _frame.DERIVED_PATH)
        derived = derived if isinstance(derived, dict) else {}
        items = derived.get("derived") if isinstance(derived.get("derived"), dict) else {}
        self.inputs: dict[str, set[str]] = {str(k): {str(x) for x in (v.get("inputs") or [])} for k, v in items.items()
                                            if isinstance(v, dict)}
        self.raw_inputs = {k: {_norm(x) for x in v} for k, v in self.inputs.items()}  # as the analysis record lists them
        self.derived_ok = {str(k) for k, v in items.items() if isinstance(v, dict) and (v.get("status") == "ok" or v.get("by_setting"))}
        #: The declared results FI's record of the analysis lists as not worked out (``not_computed``), whatever their status.
        self.not_computed = {str(x.get("id")) for x in derived.get("not_computed") or [] if isinstance(x, dict) and x.get("id")}
        self.headline_raw = list(headline)
        #: Whether a fault reaching the main result stops the paper. Off for a fault no rerun could place: leaving the result
        #: out is all FI may do then, and only an empty paper stops (:meth:`effect_of_changes` is judged by a rerun).
        self.headline_stops = True
        self.result: Any = None
        self.figure_files: dict[str, str] = {str(o["id"]): str(o.get("file") or "") for o in derived.get("figures") or []
                                             if isinstance(o, dict) and o.get("id")}
        #: Per figure file, the results the figure drew (``drawn`` of its record, else what it shows).
        self.figure_drawn: dict[str, set[str]] = {}
        for o in derived.get("figures") or []:
            if isinstance(o, dict) and o.get("file") and isinstance(o.get("declared"), dict):
                said = o["declared"].get("drawn") if o["declared"].get("drawn") is not None else o["declared"].get("shows")
                self.figure_drawn[str(o["file"])] = {str(x) for x in said or []}
        self._read_bodies()
        self._reach: dict[str, set[tuple[str, str]] | None] = {}

    # -- what the code of derive.py reads ---------------------------------------------------------------------------------
    def _read_bodies(self) -> None:
        """Add to the dependencies of each derived result what its function in ``derive.py`` reads: a result's id as a string
        (also one held by a module constant), another ``derive_<id>`` it calls, and the same of the functions of that file it
        calls in turn. Note the results whose function is not written yet: what they are worked out from cannot be read."""
        try:
            tree = ast.parse((self.code / "derive.py").read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
            tree = None
        fns = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} if tree else {}
        consts: dict[str, str] = {}
        for node in (tree.body if tree else []):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                consts.update({t.id: node.value.value for t in node.targets if isinstance(t, ast.Name)})
        self.unreadable: set[str] = set()
        self.body_calls: dict[str, set[str]] = {}  # id -> the names its function (and the file's functions it calls) calls
        self.body_reads: dict[str, tuple[set[str], set[str]]] = {}  # id -> (strings it names, heads of its f-strings), normalised
        for i in self.ids:
            fn = fns.get(f"derive_{i}")
            if fn is None:
                continue
            if _unwritten(fn):
                self.unreadable.add(i)
                continue
            scope, todo = [fn], [fn]
            while todo:  # the functions of this file it uses, and those they use
                for n in ast.walk(todo.pop()):
                    used = _used_name(n)
                    if used in fns and fns[used] not in scope:
                        scope.append(fns[used])
                        todo.append(fns[used])
            exact: set[str] = set()
            prefix: set[str] = set()
            called: set[str] = set()
            for f in scope:
                e, p = _strings_of(f)
                exact |= e
                prefix |= p
                for n in ast.walk(f):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in consts:
                        exact.add(consts[n.id])
                    if _used_name(n).startswith(("derive_", "figure_")):
                        called.add(_used_name(n))  # called, aliased, kept in a dict or passed on: it is used
            read = {self.by_norm[_norm(s)] for s in exact if _norm(s) in self.by_norm}
            read |= {j for p in prefix for j in self.ids if _norm(p) and _norm(j).startswith(_norm(p))}
            read |= {self.by_norm[_norm(c[len("derive_"):])] for c in called
                     if c.startswith("derive_") and _norm(c[len("derive_"):]) in self.by_norm}
            self.inputs.setdefault(i, set()).update(read - {i})
            self.body_reads[i] = ({_norm(x) for x in exact}, {_norm(x) for x in prefix if _norm(x)})
            self.body_calls[i] = called
        self.derive_fns = set(fns)

    def _reading(self, name: str, prefix: bool) -> set[str]:
        """The derived results whose function (or whose record in ``derived.json``) reads the trial's name ``name``."""
        n = _norm(name)
        out: set[str] = set()
        if not n:
            return out
        for i in self.ids:
            exact, heads = self.body_reads.get(i, (set(), set()))
            named = exact | self.raw_inputs.get(i, set())
            if (any(x.startswith(n) for x in named) if prefix else n in named) or any(n.startswith(h) for h in heads):
                out.add(i)
        return out

    def _closure(self, ids: set[str]) -> set[str]:
        """``ids`` and every result worked out from one of them, directly or through others."""
        out = set(ids)
        grew = True
        while grew:
            grew = False
            for k, ins in self.inputs.items():
                if k not in out and ins & out:
                    out.add(k)
                    grew = True
        return out

    def trial_ids(self) -> set[str]:
        """The declared results a trial returns (every one when FI holds no record of the trials)."""
        if not self.returned:
            return set(self.ids)
        return {i for i in self.ids if _norm(i) in self.returned}

    def _holds(self, i: str) -> bool:
        """Whether the run holds a value for the declared result ``i``: worked out in ``derived.json``, returned by a trial, or
        printed."""
        return (i in self.derived_ok or _norm(i) in self.returned
                or (self.result is not None and bool(_code_review.named_in_results(i, self.result))))

    def _valued(self, i: str) -> bool:
        """Whether the run holds a value for the declared result ``i`` that FI's record of the analysis does not list as not
        worked out."""
        return self._holds(i) and i not in self.not_computed

    # -- the question -----------------------------------------------------------------------------------------------------
    def effect(self, record: dict[str, Any], issue: dict[str, Any], result: Any = None) -> Effect | None:
        """The effect of the fault a finding's failing test showed; ``None`` when the records cannot place the finding.
        ``result``: what the run printed (to find the keys that hold a result)."""
        if not self.ids:
            return None
        self.result = result
        rel = str(issue.get("test") or "")
        try:
            src = (self.root / rel).read_text(encoding="utf-8") if rel else ""
        except OSError:
            return None
        shape = read_test(src) if src else None
        if shape is None:
            return None
        function = str(issue.get("function") or "")
        last = function.split(".")[-1]
        calls = shape.calls
        derive_calls = sorted({c[len("derive_"):] for c in calls if c.startswith("derive_")}
                              | ({last[len("derive_"):]} if last.startswith("derive_") else set()))
        figure_calls = sorted({c[len("figure_"):] for c in calls if c.startswith("figure_")}
                              | ({last[len("figure_"):]} if last.startswith("figure_") else set()))
        entries = frozenset(self._trial_entries())
        trial_call = bool(calls & entries) or last in entries
        parts: list[Effect | None] = []
        if derive_calls or figure_calls:
            parts.append(self._derive_effect(derive_calls, figure_calls))
            for f in figure_calls:  # a figure's function that a result's function (or a trial) calls is a helper of that result
                parts.append(self._figure_used_by(f"figure_{f}"))
        if trial_call:
            got = self._trial_effect(shape)
            if got.probes and not got.stop:
                # Names the test looked for in what the trial returns and the plan does not declare. What is not done is the name
                # the test itself reads (FI's own source); the reader's word for the quantity places nothing, whatever result
                # the run holds under that word.
                got.not_done = got.probes[:1]
            parts.append(got)
        if not parts and shape.frame and self._plain(shape):
            # The frame is written by FI from the plan and a repair does not change it: a failing assertion about it says the
            # plan differs from what the reader wanted, never that a number the code computed is wrong.
            return Effect(plan_text=True, why="its test reads only text FI wrote from the plan (" + ", ".join(sorted(shape.frame)) + ")")
        file = str(issue.get("file") or "").replace("\\", "/")
        if last and file and not file.endswith("derive.py") and not last.startswith(("derive_", "figure_")) and last not in entries:
            parts.append(self._helper_effect(file, function))
        known = [p for p in parts if p is not None]
        if not known:
            # No function of the study placed it: a key the test reads literally that is a declared result the run holds is the
            # result the test is about (the key is the plan's own name, not a reading of what the asserts mean).
            held = {i for i in self._test_ids(shape) if self._holds(i)}
            if not held:
                return None
            known = [self._judge(Effect(), held, "its test reads " + ", ".join(sorted(held)))]
        if all(p.by_trial for p in known):
            # Only the call graph ties a helper of the trial to the results, and it ties it to every result the trial makes, not to
            # one (the fault may change one result or all): that is no placement by itself. A key the test reads literally that is
            # a declared result the run holds does place it; otherwise the records leave it unplaced and the gate stops for it.
            held = {i for i in self._test_ids(shape) if self._holds(i)}
            if not held:
                return None
            known = [self._judge(Effect(), held, "its test reads " + ", ".join(sorted(held)))]
        merged = self._merge(known)
        return None if merged is None else self._printed_key(self._settle(merged, record, issue), record, issue, result)

    def _printed_key(self, eff: Effect, record: dict[str, Any], issue: dict[str, Any], result: Any) -> Effect:
        """A finding the call graph places on no result is not released when the run prints a value under the quantity it
        names. "No function reaches it" is a guess from the call graph (a call it cannot follow, a key a helper builds); the
        printed key is FI's own record. The key is left out of the results, with the note, as for any quantity no function
        places; a quantity the run prints no number for is released as before."""
        if not eff.unreached or eff.stop or eff.left_out or eff.figures or eff.not_done or eff.plan_text or eff.probes                 or eff.extra_keys or eff.cannot or result is None:
            return eff
        wanted = {_norm(n) for n in _code_review.quantity_names(record, issue)} - {""}
        keys = sorted(_printed_exactly(result, wanted))
        if keys:
            eff.extra_keys = keys
            eff.why += "; the run prints a value for " + ", ".join(keys[:4]) + ", so it is left out, not released"
            main = {_norm(i) for i in self.headline} | {_norm(h) for h in self.headline_raw}
            hit = [k for k in keys if _norm(k) in main]
            if hit and self.headline_stops:  # the same main-result rule as for a result a function reaches (:meth:`_judge`)
                eff.stop = f"the main result {', '.join(hit)} is among what it changes (the run prints a value for it)"
        return eff

    def reads_held_key(self, issue: dict[str, Any], result: Any) -> bool:
        """Whether the finding's failing test reads, by a literal key, a result the run reports a value for (a declared result
        the run holds, or a printed key that holds a number): then the test itself places the finding on that key."""
        if not self.ids:
            return False
        self.result = result
        rel = str(issue.get("test") or "")
        try:
            shape = read_test((self.root / rel).read_text(encoding="utf-8")) if rel else None
        except OSError:
            return False
        if shape is None:
            return False
        return (any(self._holds(i) for i in self._test_ids(shape))
                or any(bool(_code_review.named_in_results(n, result)) for n in shape.exact if _norm(n)))

    def absent_names(self, issue: dict[str, Any], result: Any, held: Any) -> list[str]:
        """The names a finding's failing test looks for that FI saw the code never give it, when that is all the failure can
        be about and FI's own records hold no value under them: the plan items not done (each the name the test itself reads),
        else ``[]``. ``held`` is what FI recorded when it ran the test again (``TestRun.held``,
        :data:`core.finding_repair.CAPTURE`): the names of every record the study's code returned to the test and of the
        records the test held when its assertion failed.

        Every one of these must hold, from FI's records alone (never the finding's words): FI listed all of it; the code
        returned something to the test, and each value was a record of names (a bare number a test compares may be a wrong
        value); the test reads at least one name literally and none with a default; it reads no key the run holds a value for;
        each assertion that failed only looks for those names (:func:`_failed_asserts_only_look`); no name the code gave the
        test, no name a trial returned and no printed key holds the name the test looks for (a name
        that contains it counts as holding it); no derived result FI holds a value for reads it; it is not the main result,
        and the run holds every main result."""
        if not self.ids or not isinstance(held, dict) or held.get("complete") is not True:
            return []
        returned = [r for r in held.get("returned") or [] if isinstance(r, dict)]
        if not returned or not all(r.get("record") is True for r in returned):
            return []
        rel = str(issue.get("test") or "")
        try:
            src = (self.root / rel).read_text(encoding="utf-8") if rel else ""
        except OSError:
            return []
        shape = read_test(src) if src else None
        if shape is None or shape.defaulted:
            return []
        looked = sorted({(n, False) for n in shape.exact if _norm(n)} | {(n, True) for n in shape.prefix if _norm(n)})
        if not looked or self.reads_held_key(issue, result):
            return []
        if not _failed_asserts_only_look(src, held.get("failures"), {n for n, prefix in looked if not prefix}):
            return []
        printed: set[str] = set()  # the printed keys that hold a number (a key printed as null with its reason holds none)

        def walk(x: Any) -> bool:
            if isinstance(x, bool):
                return False
            if isinstance(x, (int, float)):
                return True
            found = False
            if isinstance(x, dict):
                for k, v in x.items():
                    if walk(v):
                        printed.add(_norm(k))
                        found = True
            elif isinstance(x, (list, tuple)):
                for v in x:
                    found = walk(v) or found
            return found

        walk(result)
        seen = {_norm(n) for n in held.get("names") or []} | self.returned | printed
        seen.discard("")
        main = {_norm(h) for h in self.headline}
        if any(not self._valued(h) for h in self.headline):
            return []
        for name, prefix in looked:
            n = _norm(name)
            if any(n in s for s in seen) or n in main:
                return []
            if any(self._valued(i) for i in self.ids if _norm(i) == n):
                return []
            if any(self._valued(i) for i in self._reading(name, prefix)):
                return []
        return [name + ("*" if prefix else "") for name, prefix in looked]

    def effect_of_changes(self, names: list[str]) -> Effect:
        """The effect of a repair a rerun of the study showed to change numbers: ``names`` are the results whose reported
        numbers moved (a declared result's id, or a name a trial returns). The results they are, and every result worked out
        from one of them, are left out; the paper stops when the main result is among them or nothing would be left. A name
        that is no declared result (a standard error, an interval end, a count of failed trials) is left out of the printed
        results as it is (``extra_keys``), with the derived results that read it."""
        touched: set[str] = set()
        extra: list[str] = []
        for n in names:
            if _norm(n) in self.by_norm:
                touched.add(self.by_norm[_norm(n)])
                continue
            # A name a trial returns (a standard error, an interval end) that is no declared result: that number is left out as
            # it is, with the derived results that read it. The rerun showed which numbers moved, so the result it belongs to
            # is not left out with it (`_key_ids` takes every result of the trial for a standard error, as FI cannot tell then).
            touched |= self._reading(n, False)
            extra.append(n)
            owner = self.owner.get(_norm(n))
            if owner in self.counted:
                # FI counts the coverage (and, for the estimate, the bias) of a result with a true value from these numbers itself
                extra.append(f"{owner}_coverage")
                if _norm(n) in self.estimate:
                    extra.append(f"{owner}_bias")
        out = Effect()
        if touched:
            out = self._judge(out, touched, "the study run again with a repair reports other numbers for " + ", ".join(sorted(touched))[:120])
        else:
            out.why = "the study run again with a repair reports other numbers for " + ", ".join(extra)[:120]
        out.extra_keys = sorted(set(extra))
        return out

    def figures_drawing(self, ids: list[str], keys: list[str]) -> list[str]:
        """The files of the figures that draw a result left out: one of ``ids`` (results) or ``keys`` (printed names), or a result
        worked out from one of those keys. A figure whose record says it drew none of them is not listed."""
        gone = {_norm(x) for x in [*ids, *keys] if x}
        out: list[str] = []
        for file, drawn in self.figure_drawn.items():
            if any(_norm(d) in gone or (self.raw_inputs.get(d, set()) & gone) for d in drawn):
                out.append(file)
        return sorted(out)

    def unmoved(self, metrics: list[str], analysis_only: bool = False) -> list[str]:
        """The declared results a rerun that changed ``metrics`` (its own record of the numbers that moved) did not show moved.
        ``analysis_only``: the rerun ran the analysis on the recorded trials, so a result a trial returns could not move and
        is not among them."""
        moved = self.changed_keys(metrics)
        return [i for i in self.ids if i not in moved and not (analysis_only and _norm(i) in self.returned)]

    def changed_keys(self, metrics: list[str]) -> set[str]:
        """The printed keys whose recorded value a rerun showed moved: ``metrics`` are the metric names of the rerun record's own
        changed keys (``rerun_check.metric_of``), each as the declared result's id when it is one. Nothing is added to them (not
        the interval, coverage or bias keys of the result, not the results worked out from it): the record says nothing of those."""
        return {self.by_norm.get(_norm(n), n) for n in metrics if n}

    def _figure_used_by(self, fn: str) -> Effect:
        """The results whose function calls the figure's function ``fn`` by name (a figure's function a result is worked out
        with is a helper of that result), or the trial when a trial entry of the study calls it. Not the call graph: FI's own
        files call every figure's function to draw it, which says nothing of the results."""
        touched = {i for i, called in self.body_calls.items() if fn in called}
        for rel in self._study_files():
            names = _function_names(self.code, rel)
            for e in self._trial_entries():
                if e in names and fn in _called_from(names, e):
                    touched.update(self.trial_ids() or set(self.ids))
        return self._judge(Effect(), touched, f"{fn} is called by the function of " + ", ".join(sorted(touched))) if touched else Effect()

    def _study_files(self) -> list[str]:
        """The study's own Python files (not the tests, not FI's frame), relative to ``code/``, except ``derive.py``."""
        return sorted(r for r in (p.relative_to(self.code).as_posix() for p in self.code.rglob("*.py"))
                      if not any(part == "tests" or part.startswith("fi_") for part in r.split("/")[:-1])
                      and not r.rsplit("/", 1)[-1].startswith(("test_", "fi_")) and r != "derive.py")

    def _plain(self, shape: TestShape) -> bool:
        """Whether every call a test makes is a builtin, a plain string or mapping method on a value of its own, or a call into a
        module that is not the study's: no call into the study's own code. A method call on a value (``text.split(",")``) is
        never a call into the study, whatever the study's functions are named; a bare name or a call on an imported name
        counts when the study defines or is that name."""
        defined = set(self.derive_fns)
        modules = {"derive"}
        for rel in self._study_files():
            defined |= set(_function_names(self.code, rel))
            parts = rel.split("/")
            modules.add(parts[0][:-3] if len(parts) == 1 else parts[0])
        bare = shape.bare_calls | shape.unproven_calls
        if shape.calls - bare - shape.method_calls - {n for _m, n in shape.module_calls}:
            return False  # a name the test uses without a call this reading cannot place (``derive.figure_x.__code__``)
        return (all((c in _BUILTIN_NAMES or c in _PLAIN_METHODS) and c not in defined for c in bare)
                and all(c in _PLAIN_METHODS or c in _BUILTIN_NAMES for c in shape.method_calls)
                and all(m not in modules and (c in _PLAIN_METHODS or c in _BUILTIN_NAMES) and c not in defined
                        for m, c in shape.module_calls))

    def _merge(self, parts: list[Effect]) -> Effect | None:
        """The union of what several parts of one test place (a stop in any is a stop); ``None`` when one cannot tell."""
        if any(p.cannot for p in parts):
            return None
        out = Effect()
        for p in parts:
            out.left_out = list(dict.fromkeys([*out.left_out, *p.left_out]))
            out.not_done = list(dict.fromkeys([*out.not_done, *p.not_done]))
            out.figures = list(dict.fromkeys([*out.figures, *p.figures]))
            out.probes = list(dict.fromkeys([*out.probes, *p.probes]))
            out.named = out.named or p.named
            out.by_trial = out.by_trial or p.by_trial
            out.unreached = out.unreached or p.unreached
        out.stop = "; ".join(dict.fromkeys(p.stop for p in parts if p.stop))
        out.why = "; ".join(dict.fromkeys(p.why for p in parts if p.why))
        if not out.stop and out.left_out and not (set(self.ids) - set(out.left_out)):
            out.stop, out.empty = "leaving out what it changes leaves no result", True
        return out

    def _settle(self, eff: Effect, record: dict[str, Any], issue: dict[str, Any]) -> Effect:
        """A part "not done" that the run holds a value for is not left in the results: its value is left out, with what is
        worked out from it, or the paper stops when it is the main result or feeds it. Only a quantity the run holds no value
        for at all stays a part not done."""
        if eff.stop or not eff.not_done:
            return eff
        touched = set(eff.left_out)
        keep: list[str] = []
        for n in eff.not_done:
            # the declared result the name is (its id or its plan label); a name that only contains an id's words is another quantity
            ids = {i for i in self.ids if _norm(i) == _norm(n) or (self.labels.get(i) and _norm(self.labels[i]) == _norm(n))}
            held = {i for i in ids if self._holds(i)}
            keys = _code_review.named_in_results(n, self.result) if self.result is not None else set()
            if not held and not keys:
                keep.append(n)
                continue
            touched |= held
            eff.named = eff.named or (sorted(held)[0] if held else n)
            if keys - {k for i in held for k in self.printed_keys([i], self.result)}:
                if not held and _code_review.about_headline(record, issue, self.result, self.headline_raw):
                    eff.stop = f"a test of the code failed on `{n}`, which the run holds a value for and may be the main result"
                    return eff
                eff.extra_keys = sorted({*eff.extra_keys, *keys})
        eff.not_done = keep
        if touched - set(eff.left_out):
            return self._judge(eff, touched, eff.why)
        return eff

    def _trial_entries(self) -> list[str]:
        """The functions whose return is a trial's results (or the truth they are compared with)."""
        from . import code_reach as _reach

        return ["trial", *_reach.HARNESS_ENTRIES]

    # -- derive_<id> and figure_<id> --------------------------------------------------------------------------------------
    def _derive_effect(self, derive_calls: list[str], figure_calls: list[str]) -> Effect:
        out = Effect()
        left: set[str] = set()
        for x in derive_calls:
            if _norm(x) in self.by_norm:
                left.add(self.by_norm[_norm(x)])
            else:
                out.probes.append(x)  # a function for a result the plan does not declare: no declared result
        for x in figure_calls:
            out.figures.append(x)
        return self._judge(out, left, f"derive_{'/'.join(derive_calls)}" if derive_calls else "")

    def _judge(self, out: Effect, touched: set[str], what: str) -> Effect:
        """Leave out what the fault reaches (the results touched and every result worked out from one of them); stop when the
        main result is among them, or nothing would be left."""
        reached = self._closure(touched)
        out.left_out = sorted(reached)
        out.named = out.named or (sorted(touched)[0] if touched else "")
        hit = sorted(reached & self.headline) if self.headline_stops else []
        if hit:
            out.stop = f"the main result {', '.join(hit)} is among what it changes ({what})"
        elif reached and not (set(self.ids) - reached):
            out.stop, out.empty = "leaving out what it changes leaves no result", True
        out.why = what
        return out

    # -- a test that calls the trial --------------------------------------------------------------------------------------
    def _key_ids(self, name: str, prefix: bool) -> set[str]:
        """The declared results a key a test reads stands for in what the trial returns: the result of that name, or the result
        an interval end or a standard error belongs to (and, for those, every result of the trial: FI cannot tell which one
        they feed)."""
        n = _norm(name)
        trial = self.trial_ids()
        match = [i for i in self.ids if (_norm(i).startswith(n) if prefix else _norm(i) == n) and (i in trial or not self.returned)]
        if match:
            return set(match)
        owned = {k: o for k, o in self.owner.items() if (k.startswith(n) if prefix else k == n)}
        if not owned:
            return set()
        out = set(owned.values())
        if any(k not in self.estimate for k in owned):
            out |= trial
        return out

    def _test_ids(self, shape: TestShape) -> set[str]:
        """The declared results the string keys a test reads literally stand for (``exact``; ``prefix`` for an f-string key)."""
        out: set[str] = set()
        for name in shape.exact:
            out |= self._key_ids(name, False) if _norm(name) else set()
        for name in shape.prefix:
            out |= self._key_ids(name, True) if _norm(name) else set()
        return out

    def _trial_effect(self, shape: TestShape) -> Effect:
        out = Effect()
        touched: set[str] = set()
        trial = self.trial_ids()
        head_in_trial = bool(self.headline & trial)
        stop: list[str] = []

        def undeclared_returned(name: str, prefix: bool) -> None:
            feeds = self._reading(name, prefix)
            touched.update(feeds)  # a derived result's function reads it: that result, and what is worked out from it
            if head_in_trial:
                stop.append(f"the trial returns `{name}`, which the plan does not declare, in the function that returns the main result")
            elif not feeds:
                out.probes.append(name + ("*" if prefix else ""))

        def read(name: str, prefix: bool) -> None:
            n = _norm(name)
            if not n:
                return
            ids = self._key_ids(name, prefix)
            if ids:
                touched.update(ids)
            elif any((r.startswith(n) if prefix else r == n) for r in self.returned):
                undeclared_returned(name, prefix)
            else:
                out.probes.append(name + ("*" if prefix else ""))

        for k in sorted(shape.exact):
            read(k, False)
        for p in sorted(shape.prefix):
            read(p, True)
        if not shape.exact and not shape.prefix:
            touched = set(trial)
        if stop:
            out.left_out = sorted(touched)
            out.stop = "; ".join(stop)
            return out
        return self._judge(out, touched, "the trial")

    # -- a helper outside derive_/figure_/the trial -----------------------------------------------------------------------
    def _entry_reach(self, rel: str, fn: str) -> set[tuple[str, str]] | None:
        from . import code_reach as _reach

        key = f"{rel}::{fn}"
        if key not in self._reach:
            self._reach[key] = _reach.reached(self.code, scripts=[], harness=None, start=[(rel, fn)])
        return self._reach[key]

    def _helper_effect(self, file: str, function: str) -> Effect | None:
        """Which results' functions reach the helper (the call graph of :mod:`core.code_reach`): the results of those
        functions are what a fault in it changes; a helper no result's function reaches changes none. ``None`` when the
        function is not found or no entry was found (the gate then judges as before); an ``Effect`` with ``cannot`` when
        FI cannot tell what the code reaches."""
        from . import code_reach as _reach

        judged = _reach._judged(self.code / file, function)
        if judged is None:
            return None
        entries: list[tuple[str, str, str, str]] = []  # (kind, id or "", file, function)
        for i in self.ids:
            if f"derive_{i}" in self.derive_fns:
                entries.append(("derive", i, "derive.py", f"derive_{i}"))
        derive_figs = sorted(f for f in self.derive_fns if f.startswith("figure_"))
        entries += [("figure", f[len("figure_"):], "derive.py", f) for f in derive_figs]
        # Every function FI's harness calls for a trial (`trial`, `run_trial`, `run_cell`, `oracle`), in any file of the study
        # including `experiment.py`, which may hold the whole study.
        for rel in self._study_files():
            names = _function_names(self.code, rel)
            for e in self._trial_entries():
                if e in names:
                    entries.append(("trial", "", rel, e))
        if not entries:
            return None  # no entry to ask: nothing is known of what reaches it
        out = Effect()
        touched: set[str] = set()
        reached_by: list[str] = []
        by_trial, by_result = False, False
        for kind, ident, rel, fn in entries:
            got = self._entry_reach(rel, fn)
            if got is None:
                return Effect(cannot=True, why="FI cannot tell what the code reaches")
            if (file, judged) in got or (rel, fn) == (file, judged):
                reached_by.append(fn)
                if kind == "derive":
                    touched.add(ident)
                    by_result = True
                elif kind == "figure":
                    out.figures.append(ident)
                    by_result = True
                else:
                    touched.update(self.trial_ids() or set(self.ids))
                    by_trial = True
        out.by_trial = by_trial and not by_result  # the trial is the only link: the call graph names no result in particular
        if not reached_by:
            out.why = "no function of a result or a figure reaches it, so it changes no reported result"
            out.unreached = True
            return out
        return self._judge(out, touched, "reached by " + ", ".join(sorted(set(reached_by))[:4]))

    def touched_by(self, file: str, function: str) -> list[str] | None:
        """The declared results a fault in ``function`` of ``file`` changes (:meth:`_helper_effect`: the results of the
        functions that reach it, and every result worked out from them); ``[]`` for a function no result reaches; ``None`` when
        the function is not found or FI cannot tell what the code reaches."""
        eff = self._helper_effect(file.replace("\\", "/"), function) if self.ids and file and function else None
        return None if eff is None or eff.cannot else list(eff.left_out)

    # -- the names to leave out of the printed results ----------------------------------------------------------------------
    def printed_keys(self, ids: list[str], result: Any, unmoved: Any = ()) -> list[str]:
        """The keys of the printed results that hold the declared results ``ids``: the id itself, the keys that are its
        intervals (:func:`core.code_review.named_in_results`) and the coverage and bias FI counts for it. ``unmoved``: declared
        results a rerun of the study compared and found unchanged (:meth:`unmoved`); one of them is never taken as a key of
        ``ids`` by the likeness of the names. Without a rerun the names still group them."""
        keys: set[str] = set()
        apart = [u for u in unmoved or () if u not in ids]
        for i in ids:
            keys.add(i)
            keys |= _code_review.named_in_results(i, result, apart)
            if i in self.counted:
                keys |= {f"{i}_coverage", f"{i}_bias"}
        return sorted(keys)
