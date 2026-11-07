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
from typing import Any

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
            if isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "pop") and n.args:
                key(n.args[0])
        elif isinstance(n, ast.Subscript):
            key(n.slice)
        elif isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
            key(n.left)
    from . import analysis_frame as _frame

    constants = {n for n in _frame.FRAME_NAMES if n.isupper()}
    frame = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in constants}
    frame |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names if a.name in constants}
    # a body named without being called (`derive.figure_fig1.__code__`) is still the body the test is about
    calls |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr.startswith(("derive_", "figure_"))}
    return TestShape(calls, exact, prefix, frame)


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
        self.headline_raw = list(headline)
        #: Whether a fault reaching the main result stops the paper. Off for a fault no rerun could place: leaving the result
        #: out is all FI may do then, and only an empty paper stops (:meth:`effect_of_changes` is judged by a rerun).
        self.headline_stops = True
        self.result: Any = None
        self.figure_files: dict[str, str] = {str(o["id"]): str(o.get("file") or "") for o in derived.get("figures") or []
                                             if isinstance(o, dict) and o.get("id")}
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
                # Names the test looked for in what the trial returns and the plan does not declare: what is not done is the
                # plan's quantity the finding is about, not the probed names (they are not the plan's words).
                got.not_done = _code_review.quantity_names(record, issue)[:1] or got.probes[:1]
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
        merged = self._merge(known)
        return None if merged is None else self._settle(merged, record, issue)

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

    def changed_keys(self, names: list[str], result: Any) -> set[str]:
        """The printed keys whose recorded value a rerun showed changed: those of the declared results among ``names`` and the
        names that are no declared result as they are. The results worked out from them (``effect_of_changes`` leaves those out
        too) are not among them: that they read a changed number does not show their own number changed."""
        declared = [self.by_norm[_norm(n)] for n in names if _norm(n) in self.by_norm]
        extra = [n for n in names if _norm(n) not in self.by_norm]
        keys = set(self.printed_keys(declared, result)) | set(extra)
        for n in extra:
            owner = self.owner.get(_norm(n))
            if owner in self.counted:
                keys.add(f"{owner}_coverage")
                if _norm(n) in self.estimate:
                    keys.add(f"{owner}_bias")
        return keys

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
        """Whether every call a test makes is a builtin, a plain string or mapping method, or reads a frame constant: no call
        into the study's own code."""
        defined = set(self.derive_fns)
        for rel in self._study_files():
            defined |= set(_function_names(self.code, rel))
        return all((c in _BUILTIN_NAMES or c in _PLAIN_METHODS) and c not in defined for c in shape.calls)

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
        for kind, ident, rel, fn in entries:
            got = self._entry_reach(rel, fn)
            if got is None:
                return Effect(cannot=True, why="FI cannot tell what the code reaches")
            if (file, judged) in got or (rel, fn) == (file, judged):
                reached_by.append(fn)
                if kind == "derive":
                    touched.add(ident)
                elif kind == "figure":
                    out.figures.append(ident)
                else:
                    touched.update(self.trial_ids() or set(self.ids))
        if not reached_by:
            out.why = "no function of a result or a figure reaches it, so it changes no reported result"
            return out
        return self._judge(out, touched, "reached by " + ", ".join(sorted(set(reached_by))[:4]))

    # -- the names to leave out of the printed results ----------------------------------------------------------------------
    def printed_keys(self, ids: list[str], result: Any) -> list[str]:
        """The keys of the printed results that hold the declared results ``ids``: the id itself, the keys that are its
        intervals (:func:`core.code_review.named_in_results`) and the coverage and bias FI counts for it."""
        keys: set[str] = set()
        for i in ids:
            keys.add(i)
            keys |= _code_review.named_in_results(i, result)
            if i in self.counted:
                keys |= {f"{i}_coverage", f"{i}_bias"}
        return sorted(keys)
