"""Range and unit assertions on a run's results, declared by its design.

Why this is not just a wider degeneracy check
=============================================
``_is_degenerate_result`` fires only when *every* numeric leaf is within
1e-12 of zero. That catches a simulation that produced nothing. It cannot
catch the far more common failure: a run that produced plausible-looking
numbers from wrong physics — a unit error, a factor of two, a sign flip.
Those exit 0, parse fine, and sail through every downstream gate.

The missing ingredient is not a cleverer heuristic. It is knowledge of
what the outputs are *allowed* to be, and only the design stage has that:
it chose the working point, so it knows k1 lives in roughly [0.25, 0.5],
that a CD is positive, that a normalised contrast cannot exceed 1.

So the design declares the bounds and this module enforces them, in code.
An assertion the design did not make is not checked — silence here means
"nothing was claimed", never "everything is fine".

Where the bounds come from later
================================
When a quest calls a trusted simulation skill rather than generating its
physics from scratch, the skill carries its own valid domain and can
supply these assertions without the design having to restate them. This
module takes assertions from a list; it does not care who wrote them.

Why a value inside its bounds can still fail
============================================
A bound that rejects a result invites the script to stop producing it. A
real quest showed exactly that: an integrator diverged, the gate rejected the
huge error, and the regenerated script read ``rmse = 10.0  # Cap at
assertion max`` -- divergence laundered into a number the gate cannot tell
from a genuine RMSE of 10. So a value sitting exactly on a non-zero bound is
rejected when the script also caps values at that constant. Both halves are
required: a perfect score of 1.0 is a real result, and ``np.clip(p, 0, 1)``
inside a softmax is real code; only the two together are a capped number.
Zero bounds are exempt, because real zeros and ``max(0, x)`` are both
everywhere.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any

# Assertion paths are matched against the flattened result_json keys used
# by ``numeric_oracle.flatten_numbers`` — "metrics.contrast", "cd_nm",
# "sweep[0].nils". A trailing ".*" or a bare name matches any leaf whose
# final segment equals it, so a design can say "cd_nm" without knowing how
# deeply the script nested it.
_LEAF = re.compile(r"[^.\[\]]+$")


@dataclass(frozen=True)
class Assertion:
    """One declared bound on a result value."""

    path: str
    min: float | None = None
    max: float | None = None
    unit: str = ""
    reason: str = ""

    def describe(self) -> str:
        if self.min is not None and self.max is not None:
            rng = f"[{self.min:g}, {self.max:g}]"
        elif self.min is not None:
            rng = f">= {self.min:g}"
        elif self.max is not None:
            rng = f"<= {self.max:g}"
        else:
            rng = "finite"
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.path} must be {rng}{unit}"


@dataclass(frozen=True)
class Violation:
    path: str
    value: float
    assertion: Assertion
    # "out_of_range", or "clamped": on a bound the script caps values at.
    kind: str = "out_of_range"

    def describe(self) -> str:
        why = f" ({self.assertion.reason})" if self.assertion.reason else ""
        if self.kind == "clamped":
            return (
                f"{self.path} = {self.value:g} sits exactly on a bound "
                f"({self.assertion.describe()}){why}, and the script caps "
                f"values at {self.value:g}: a clamped number, not a measurement"
            )
        return (
            f"{self.path} = {self.value:g} violates "
            f"{self.assertion.describe()}{why}"
        )


# Calls that pin a value to a constant: builtins, numpy/torch/tf spellings.
_CLAMP_CALLS = frozenset({
    "min", "max", "clip", "clamp", "clip_by_value", "minimum", "maximum",
    "fmin", "fmax", "where", "nan_to_num",
})


def _literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) \
            and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _literal(node.operand)
        if v is not None:
            return -v if isinstance(node.op, ast.USub) else v
    return None


def clamp_constants(code: str) -> set[float]:
    """Constants the script pins values to.

    Collected from the three shapes a cap takes: a literal argument to a
    min/max/clip-style call, a literal branch of a conditional expression
    (``10.0 if diverged else rmse``), and a literal assigned inside an ``if``
    (``if rmse > 1e6: rmse = 10.0``). A name bound to a literal anywhere in
    the file counts as that literal, so ``CAP = 10.0 ... min(x, CAP)`` is
    seen too. Unparseable code yields nothing rather than raising.
    """
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError):
        return set()
    names: dict[str, float] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            v = _literal(node.value)
            if v is not None:
                names[node.targets[0].id] = v

    def value(node: ast.AST) -> float | None:
        v = _literal(node)
        if v is None and isinstance(node, ast.Name):
            v = names.get(node.id)
        return v

    out: set[float] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else "")
            if name in _CLAMP_CALLS:
                for arg in (*node.args, *(k.value for k in node.keywords)):
                    v = value(arg)
                    if v is not None:
                        out.add(v)
        elif isinstance(node, ast.IfExp):
            for branch in (node.body, node.orelse):
                v = value(branch)
                if v is not None:
                    out.add(v)
        elif isinstance(node, ast.If):
            for stmt in (*node.body, *node.orelse):
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                    v = value(stmt.value)
                    if v is not None:
                        out.add(v)
    return out


def _pinned(value: float, a: Assertion, clamps: set[float]) -> bool:
    for bound in (a.min, a.max):
        if bound is None or bound == 0:
            continue
        if math.isclose(value, bound, rel_tol=1e-9) and any(
            math.isclose(c, bound, rel_tol=1e-9) for c in clamps
        ):
            return True
    return False


def parse_assertions(design: Any) -> list[Assertion]:
    """Read ``result_assertions`` off a design spec.

    Tolerant by construction: this is LLM output, and a malformed entry
    must be dropped rather than raise. A design that declares nothing
    yields nothing, which is the correct no-op.
    """
    if not isinstance(design, dict):
        return []
    raw = design.get("result_assertions")
    if not isinstance(raw, list):
        return []

    out: list[Assertion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue

        def _num(key: str) -> float | None:
            v = item.get(key)
            if isinstance(v, bool) or v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if math.isfinite(f) else None

        lo, hi = _num("min"), _num("max")
        if lo is None and hi is None:
            continue  # an assertion that asserts nothing
        if lo is not None and hi is not None and lo > hi:
            continue  # inverted bounds are a typo, not a constraint
        out.append(
            Assertion(
                path=path,
                min=lo,
                max=hi,
                unit=str(item.get("unit") or "").strip(),
                reason=str(item.get("reason") or "").strip(),
            )
        )
    return out


def _matches(assertion_path: str, leaf_path: str) -> bool:
    """Does an assertion path address this flattened result path?

    Exact match wins. Otherwise the assertion may name just the final
    segment ("cd_nm" matching "sweep[2].cd_nm"), so a design does not have
    to predict the script's nesting.
    """
    if assertion_path == leaf_path:
        return True
    a = assertion_path.rstrip(".*")
    if a == leaf_path:
        return True
    m = _LEAF.search(leaf_path)
    return bool(m and m.group(0) == a)


def violations(
    result_json: Any, assertions: list[Assertion], *, code: str = "",
) -> list[Violation]:
    """Every declared bound the result actually breaks.

    With ``code``, a value exactly on a non-zero bound that the script also
    caps at is reported as ``kind="clamped"`` (see the module docstring).
    """
    if not assertions:
        return []
    from core.numeric_oracle import flatten_numbers

    leaves = list(flatten_numbers(result_json))
    if not leaves:
        return []

    clamps = clamp_constants(code) if code else set()
    out: list[Violation] = []
    for a in assertions:
        for path, value in leaves:
            if not _matches(a.path, path):
                continue
            if a.min is not None and value < a.min:
                out.append(Violation(path, value, a))
            elif a.max is not None and value > a.max:
                out.append(Violation(path, value, a))
            elif clamps and _pinned(value, a, clamps):
                out.append(Violation(path, value, a, kind="clamped"))
    return out


def check_design(result_json: Any, design: Any, *, code: str = "") -> list[Violation]:
    """Convenience: parse a design's assertions and apply them."""
    return violations(result_json, parse_assertions(design), code=code)
