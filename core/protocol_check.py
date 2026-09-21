"""Does the experiment use the protocol the plan froze?

The plan (``plan.md``) holds a ``protocol`` block in its design: the grid the experiment sweeps, how many runs each setting
gets, the thresholds it applies, how randomness is seeded, how uncertainty is estimated and what would count as support.
A quest whose topic named its numbers ("for R0 in {0.9, 1.5, 3.0} and N in {100, 1000, 5000}, 300 runs each") had
nothing that held the experiment to them: one run drew its grid at {0.8, 1.2, 2.0, 3.0} and was accepted with the top
score, because no step compared the code with the topic and the only later check (:mod:`core.goal_coverage`) reads the
topic's prose after the paper is written and is advice only.

This module holds the parts that need no model and no engine state:

* :func:`check` compares the numbers a protocol fixes with what an experiment script contains, and names each difference;
* :func:`plan_notes` compares the protocol with the numbers the topic sets, when the plan is written.

A difference is reported only on evidence that a person would accept, because a difference stops the quest:

* a **grid** axis is a list, a tuple, a numeric range or a ``for`` loop over literals that the script names after the axis
  (``R0_LIST`` for ``R0``), or, when no name says so, that shares at least half its values with the axis. It differs when
  none of the matching lists holds exactly the protocol's values. A list assigned under ``if os.environ.get("FI_PILOT")``
  is the engine's smoke test, which discards its numbers, and is not read;
* **runs per setting** differs when the script names counts of runs (``NUM_RUNS``, ``n_samples``, ``TRIALS``...) and none of
  them is the protocol's;
* a **threshold** differs when its value appears nowhere in the script (as itself, or as a percentage of itself).

A protocol key the script gives no evidence about is not a difference: the check finds a script that contradicts the
plan, and does not claim to prove a script that agrees with it.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .goal_coverage import _generated, _literal, asked_numbers, code_numbers

_RUN_WORDS = {
    "runs", "run", "replicates", "replicate", "reps", "trials", "trial", "samples", "sample", "realizations",
    "realisations", "simulations", "simulation", "sims", "repeats",
}
_PILOT = "FI_PILOT"
_NUMERIC_CALLS = ("linspace", "logspace", "geomspace", "arange", "range")


@dataclass(frozen=True)
class Mismatch:
    """One way the script differs from the protocol."""

    kind: str  # "grid", "runs" or "threshold"
    name: str  # the axis, or the threshold, as the protocol names it
    expected: list[float]
    found: list[float] = field(default_factory=list)  # what the script holds under a matching name
    where: str = ""  # "R0_LIST in experiment.py, line 166"

    def message(self) -> str:
        want = _fmt(self.expected)
        if self.kind == "grid":
            if not self.found:
                return f"the protocol fixes {self.name} at {want}, and the script contains no such list (missing {self.where})"
            extra = [v for v in self.found if not _has(self.expected, v)]
            left_out = [v for v in self.expected if not _has(self.found, v)]
            parts = []
            if extra:
                parts.append(f"adds {_fmt(extra)}")
            if left_out:
                parts.append(f"leaves out {_fmt(left_out)}")
            return (
                f"the protocol fixes {self.name} at {want}; {self.where} is {_fmt(self.found)}"
                + (f": it {' and '.join(parts)}" if parts else "")
            )
        if self.kind == "runs":
            return (
                f"the protocol fixes the runs per setting at {want}; the script sets {_fmt(self.found)} "
                f"({self.where})"
            )
        return f"the protocol fixes {self.name} at {want}, and that number appears nowhere in the script"


def _fmt(values: list[float]) -> str:
    def one(v: float) -> str:
        return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else f"{v:g}"

    return "[" + ", ".join(one(v) for v in values) + "]" if len(values) != 1 else one(values[0])


def _has(values: list[float] | set[float], target: float) -> bool:
    return any(math.isclose(target, v, rel_tol=1e-9, abs_tol=1e-12) for v in values)


def _same_set(a: list[float], b: list[float]) -> bool:
    return all(_has(b, v) for v in a) and all(_has(a, v) for v in b)


def _tokens(name: str) -> list[str]:
    # R0_LIST, r0Values, N_values -> ["r0", "list"], ["r0", "values"] (camel case is split at a lower-to-upper step)
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


# --- reading a script ------------------------------------------------------------------------------------------------


@dataclass
class _Found:
    name: str
    values: list[float]
    line: int
    script: str


def _numeric_sequence(node: ast.AST) -> list[float] | None:
    """The values of a literal list, tuple or set of numbers, of ``np.array([...])`` and the like, or of a numeric range
    call with literal arguments."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_literal(e) for e in node.elts]
        return [v for v in values if v is not None] if values and all(v is not None for v in values) else None
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if name in _NUMERIC_CALLS:
            args = [_literal(a) for a in node.args]
            if args and all(a is not None for a in args):
                values = _generated(name, [a for a in args if a is not None])
                return values or None
        if name in ("array", "asarray", "list", "tuple", "sorted", "float64") and node.args:
            return _numeric_sequence(node.args[0])
    return None


def _mentions_pilot(test: ast.AST) -> bool:
    return any(isinstance(n, ast.Constant) and n.value == _PILOT for n in ast.walk(test)) or any(
        isinstance(n, ast.Name) and n.id == _PILOT for n in ast.walk(test)
    )


class _Reader(ast.NodeVisitor):
    """Collects the numeric lists and the scalar counts a script assigns, outside any ``FI_PILOT`` branch."""

    def __init__(self, script: str) -> None:
        self.script = script
        self.lists: list[_Found] = []
        self.scalars: list[_Found] = []

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if _mentions_pilot(node.test):
            # The branch taken when FI_PILOT is set is the smoke test; the other one is the real run.
            negated = any(isinstance(n, (ast.Not, ast.NotEq)) for n in ast.walk(node.test))
            for child in node.body if negated else node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def _record(self, name: str, value: ast.AST, line: int) -> None:
        seq = _numeric_sequence(value)
        if seq is not None:
            self.lists.append(_Found(name, seq, line, self.script))
            return
        lit = _literal(value)
        if lit is not None:
            self.scalars.append(_Found(name, [lit], line, self.script))
        elif isinstance(value, ast.Dict):
            for key, val in zip(value.keys, value.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    self._record(key.value, val, line)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record(target.id, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._record(node.target.id, node.value, node.lineno)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        if isinstance(node.target, ast.Name):
            seq = _numeric_sequence(node.iter)
            if seq is not None:
                self.lists.append(_Found(node.target.id, seq, node.lineno, self.script))
        self.generic_visit(node)


def _read(scripts: dict[str, str]) -> tuple[list[_Found], list[_Found], set[float]]:
    lists: list[_Found] = []
    scalars: list[_Found] = []
    numbers: set[float] = set()
    for name, source in scripts.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            numbers |= code_numbers(source)
            continue
        reader = _Reader(name)
        reader.visit(tree)
        lists += reader.lists
        scalars += reader.scalars
        numbers |= code_numbers(source)
    return lists, scalars, numbers


# --- the protocol --------------------------------------------------------------------------------------------------


def protocol_numbers(protocol: dict[str, Any] | None) -> list[tuple[float, str]]:
    """Every number a protocol fixes, with where it sits: ``(0.9, "grid R0")``."""
    out: list[tuple[float, str]] = []
    if not isinstance(protocol, dict):
        return out
    grid = protocol.get("grid")
    if isinstance(grid, dict):
        for axis, values in grid.items():
            for v in values if isinstance(values, list) else []:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append((float(v), f"grid {axis}"))
    runs = protocol.get("runs_per_setting")
    if isinstance(runs, (int, float)) and not isinstance(runs, bool):
        out.append((float(runs), "runs per setting"))
    thresholds = protocol.get("thresholds")
    if isinstance(thresholds, dict):
        for name, v in thresholds.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((float(v), f"threshold {name}"))
    return out


def check(protocol: dict[str, Any] | None, scripts: dict[str, str]) -> list[Mismatch]:
    """The ways ``scripts`` (file name -> source) contradict ``protocol``; empty when they do not."""
    if not isinstance(protocol, dict) or not scripts:
        return []
    lists, scalars, numbers = _read(scripts)
    out: list[Mismatch] = []

    grid = protocol.get("grid")
    for axis, values in (grid.items() if isinstance(grid, dict) else []):
        want = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)] if isinstance(values, list) else []
        if not want:
            continue
        axis_tokens = _tokens(str(axis))
        named = [f for f in lists if axis_tokens and set(axis_tokens) <= set(_tokens(f.name))]
        candidates = named or [
            f for f in lists
            if len(f.values) >= 2 and sum(_has(f.values, v) for v in want) >= max(1, len(want) // 2)
        ]
        if candidates:
            if not any(_same_set(f.values, want) for f in candidates):
                best = max(candidates, key=lambda f: sum(_has(want, v) for v in f.values))
                out.append(Mismatch("grid", str(axis), want, best.values, f"{best.name} in {best.script}, line {best.line}"))
        else:
            absent = [v for v in want if not _has(numbers, v)]
            if absent:
                out.append(Mismatch("grid", str(axis), want, [], _fmt(absent)))

    runs = protocol.get("runs_per_setting")
    if isinstance(runs, (int, float)) and not isinstance(runs, bool):
        counts = [f for f in scalars if set(_tokens(f.name)) & _RUN_WORDS and float(f.values[0]).is_integer()]
        if counts and not any(math.isclose(f.values[0], float(runs)) for f in counts):
            first = counts[0]
            out.append(Mismatch(
                "runs", "runs_per_setting", [float(runs)], sorted({f.values[0] for f in counts}),
                f"{first.name} in {first.script}, line {first.line}",
            ))

    thresholds = protocol.get("thresholds")
    for name, v in (thresholds.items() if isinstance(thresholds, dict) else []):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            forms = (float(v), float(v) * 100.0, float(v) / 100.0)
            if not any(_has(numbers, form) for form in forms):
                out.append(Mismatch("threshold", str(name), [float(v)]))
    return out


def plan_notes(topic: str, protocol: dict[str, Any] | None) -> list[str]:
    """What the plan leaves out of the numbers the topic sets: one sentence each, for the plan's *Checks already made*."""
    asked = asked_numbers(topic or "")
    if not asked:
        return []
    fixed = protocol_numbers(protocol)
    if not fixed:
        return [
            "The topic sets numbers (" + ", ".join(a.text for a in asked[:6])
            + "), but the plan has no protocol block, so nothing checks the experiment against them."
        ]
    values = [v for v, _where in fixed]
    notes = []
    for a in asked:
        held = _has(values, a.value) or (
            a.scaled and any(math.isclose(abs(a.value), abs(v) * 10 ** k, rel_tol=1e-6) for v in values for k in range(-9, 10))
        )
        if not held:
            notes.append(f"The topic sets {a.text} ({a.where}), and the protocol does not contain it.")
    return notes
