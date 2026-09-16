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

Why several values on a bound fail
==================================
A computation that goes wrong often returns the trivial answer rather than a
wild one: a root finder that settles on the solution at the starting state, a
sentinel, a guard branch. That answer tends to be the edge of the legal range,
where a range check passes it. A real quest's deterministic model reported a
final size of 0.0 at two reproduction numbers and 1.0 at a third, every one
inside the declared [0, 1]. One value on a bound is ordinary. The same quantity
exactly on one bound in ``AT_BOUND_MIN_SETTINGS`` or more settings is sent back
for repair, zero bounds included, and the repair is told to change the code
only if the computation is what put it there. Zeros are compared for every
kind here: a value of exactly 0 below a positive ``min`` is out of range.

Why a correct zero must not be sent back
========================================
The rule above was right about trivial answers and wrong about correct ones,
and every catch it made came with a false alarm that cost a repair round. Below
the epidemic threshold nothing takes off: at R0 = 0.9 with N = 5000, zero of 300
trajectories crossing a threshold is simply the answer. Runs were sent back to
"fix" it — one lost its replicates, one returned ``null`` and crashed its own
plot, one spent 77,702 tokens and a skipped replication pass — and a model told
that a correct number is wrong will hide it rather than defend it.

Three things separate a correct zero from a trivial one, and all three are
decided from the results themselves:

*One setting is counted once.* A sweep reported under two groupings — once by
R0 and once by N — lists the same cell twice, so a single zero looked like the
two settings the rule needs. Settings are identified by the coordinates in
their path rather than by the path string, so the same cell reached by a
different route is one setting.

*A zero below the critical point is expected.* Where a path names the
reproduction number (or an enclosing object carries it as a field), a zero at
R0 < 1 is what the theory predicts, not an anomaly. This applies to a zero
bound only: a *one* below threshold is excused by nothing, and the run that
reported a final size of 1.0 at R0 = 0.9 was genuinely wrong.

*Zeros confined to one level of one coordinate are a regime, not a bug.* Where
no name identifies the control parameter, the shape still does: if every zero
shares one coordinate value, the quantity is non-zero at the other levels, and
the zeros are a minority of the sweep, that is a threshold effect. A quantity
that is zero at *every* setting gets no such excuse — that is the trivial
answer this gate exists to catch, and ``numeric_oracle`` reports it too.

What survives is reported together with the zeros that were recognised, so a
repair is told what has already been accounted for instead of being asked to
change it.
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
    # "out_of_range"; "clamped": on a bound the script caps values at;
    # "at_bound": exactly on one bound in several settings.
    kind: str = "out_of_range"
    # The result paths an ``at_bound`` violation covers.
    paths: tuple[str, ...] = ()
    # Values on the same bound that were recognised as expected, as
    # (path, reason), so a repair is not asked to change them.
    explained: tuple[tuple[str, str], ...] = ()

    def describe(self) -> str:
        why = f" ({self.assertion.reason})" if self.assertion.reason else ""
        if self.kind == "at_bound":
            shown = ", ".join(self.paths[:3]) + (", …" if len(self.paths) > 3 else "")
            note = ""
            if self.explained:
                seen = ", ".join(
                    f"{p} ({r})" for p, r in self.explained[:3]
                ) + (", …" if len(self.explained) > 3 else "")
                note = (
                    f". {len(self.explained)} further value(s) on this bound are "
                    f"expected and were not counted: {seen} — leave those as they are"
                )
            return (
                f"{self.path} equals its bound {self.value:g} in {len(self.paths)} "
                f"settings ({shown}); {self.assertion.describe()}{why}. Several "
                f"results exactly on a bound are what a computation returning a "
                f"trivial answer looks like{note}"
            )
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


# How many settings of one quantity must sit exactly on the same bound before
# the run is sent back (see "Why several values on a bound fail" above). One is
# ordinary: a probability of 0 at one working point is often the answer.
AT_BOUND_MIN_SETTINGS = 2

# Below this the epidemic dies out, so a zero is the prediction rather than a
# failure. The value is the critical point of the branching-process
# approximation and is not tunable: R0 = 1 is where the behaviour changes.
CRITICAL_R0 = 1.0

# One path segment that is purely a number — a sweep coordinate rather than a
# field name.
_NUMERIC_SEG = re.compile(r"^-?\d+(?:\.\d+)?$")

# The reproduction number written into a path: ``by_R0.0.9``, ``by_r0_0.9``,
# ``by_R0_N.R0_0.9_N_100``. Matched against the whole path, because splitting on
# "." cannot tell the key "0.9" from the two keys "0" and "9".
_R0_IN_PATH = re.compile(r"(?:^|[._\[])r_?0[._]?(\d+(?:\.\d+)?)", re.IGNORECASE)

# The same quantity carried as a field of an enclosing object, which is how a
# result keyed by cell rather than by coordinate records it.
_R0_FIELDS = ("R0", "r0", "R_0", "r_0", "reproduction_number",
              "basic_reproduction_number")

# A grouping key that LEADS with R0 introduces it as the first coordinate that
# follows, even when the key is compound and the value is packed: ``by_R0_N``
# then ``0.9_100`` means R0 = 0.9, N = 100. Without this the name and its value
# sit two separators apart and neither the path match above nor the positional
# rule can see the coordinate at all.
_R0_GROUP_KEY = re.compile(r"^by[_-]?r_?0(?:[_-]|$)", re.IGNORECASE)
_LEADING_NUMBER = re.compile(r"^(\d+(?:\.\d+)?)")


def _walk_leaves(
    obj: Any, tokens: tuple[tuple[str, Any], ...] = (),
) -> Any:
    """Every numeric leaf as ``(tokens, dotted_path, value)``.

    Same values and same dotted paths as ``numeric_oracle.flatten_numbers``
    with ``keep_zero=True`` — a test pins that — but the tokens keep the
    key boundaries the dotted form loses. ``by_N.100.0.9`` is three keys
    (N = 100, R0 = 0.9) and reading it back off the string cannot say so.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            yield tokens, _dotted(tokens), float(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_leaves(v, tokens + (("k", str(k)),))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_leaves(v, tokens + (("i", i),))


def _dotted(tokens: tuple[tuple[str, Any], ...]) -> str:
    out = ""
    for kind, val in tokens:
        if kind == "i":
            out += f"[{val}]"
        else:
            out = f"{out}.{val}" if out else str(val)
    return out or "<root>"


def _coords(tokens: tuple[tuple[str, Any], ...]) -> list[str]:
    """The sweep coordinates of a leaf, in order: every numeric key and list
    index on the way to it. The leaf's own name is not a coordinate."""
    out: list[str] = []
    for kind, val in tokens[:-1]:
        s = str(val)
        if kind == "i" or _NUMERIC_SEG.match(s):
            out.append(s)
    return out


def _setting_key(tokens: tuple[tuple[str, Any], ...]) -> frozenset:
    """Which cell of the sweep a leaf belongs to.

    Each field name is paired with the coordinate values that follow it, and
    the pairs are unordered, so ``by_R0.0.9.by_N.5000`` and
    ``by_N.5000.by_R0.0.9`` — the same cell reported under two groupings — are
    one setting. Pairing rather than sorting the bare segments matters:
    ``cells[1].takeoff_events.2.0`` and ``cells[2].takeoff_events.1.0`` hold
    the same segments and are different cells.
    """
    pairs: list[tuple[str, str]] = []
    name: str | None = None
    vals: list[str] = []
    for kind, val in tokens[:-1]:
        s = str(val)
        if kind == "i" or _NUMERIC_SEG.match(s):
            vals.append(s)
            continue
        if name is not None or vals:
            pairs.append((name or "", ".".join(vals)))
        name, vals = s, []
    if name is not None or vals:
        pairs.append((name or "", ".".join(vals)))
    return frozenset(pairs)


def _r0_for(
    tokens: tuple[tuple[str, Any], ...], dotted: str, root: Any,
) -> float | None:
    """The reproduction number governing this leaf, or None if nothing says.

    Read from the path when it names R0, otherwise from the nearest enclosing
    object that carries it as a field.
    """
    m = _R0_IN_PATH.search(dotted)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    keys = [str(v) for kind, v in tokens[:-1] if kind == "k"]
    for name, nxt in zip(keys, keys[1:]):
        if _R0_GROUP_KEY.match(name):
            lead = _LEADING_NUMBER.match(nxt)
            if lead:
                try:
                    return float(lead.group(1))
                except ValueError:
                    pass
    node = root
    found: float | None = None
    for kind, val in tokens[:-1]:
        try:
            node = node[val] if kind == "i" else node[str(val)]
        except (KeyError, IndexError, TypeError):
            return found
        if isinstance(node, dict):
            for key in _R0_FIELDS:
                v = node.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    found = float(v)
    return found


def _confined_position(
    flagged: list[list[str]], others: list[list[str]],
) -> int | None:
    """The coordinate position every flagged setting shares and at least one
    unflagged setting does not — the signature of a regime boundary rather
    than a broken computation. None when there is no such position."""
    if not flagged or not others:
        return None
    width = min(len(c) for c in flagged + others)
    for pos in range(width):
        level = {c[pos] for c in flagged}
        if len(level) != 1:
            continue
        if any(c[pos] not in level for c in others):
            return pos
    return None


def violations(
    result_json: Any, assertions: list[Assertion], *, code: str = "",
) -> list[Violation]:
    """Every declared bound the result actually breaks.

    With ``code``, a value exactly on a non-zero bound that the script also
    caps at is reported as ``kind="clamped"`` (see the module docstring). A
    quantity exactly on one bound in ``AT_BOUND_MIN_SETTINGS`` or more settings
    is reported once, as ``kind="at_bound"``.
    """
    if not assertions:
        return []
    leaves = list(_walk_leaves(result_json))
    if not leaves:
        return []

    clamps = clamp_constants(code) if code else set()
    out: list[Violation] = []
    for a in assertions:
        matched: list[tuple[tuple, str, float]] = []
        on_bound: dict[float, list[tuple[tuple, str]]] = {}
        for tokens, path, value in leaves:
            if not _matches(a.path, path):
                continue
            matched.append((tokens, path, value))
            if a.min is not None and value < a.min:
                out.append(Violation(path, value, a))
            elif a.max is not None and value > a.max:
                out.append(Violation(path, value, a))
            elif clamps and _pinned(value, a, clamps):
                out.append(Violation(path, value, a, kind="clamped"))
            else:
                bound = next(
                    (b for b in (a.min, a.max)
                     if b is not None and math.isclose(value, b, rel_tol=1e-9, abs_tol=1e-12)),
                    None,
                )
                if bound is not None:
                    on_bound.setdefault(bound, []).append((tokens, path))
        for bound, entries in on_bound.items():
            v = _at_bound(a, bound, entries, matched, result_json)
            if v is not None:
                out.append(v)
    return out


def _at_bound(
    a: Assertion,
    bound: float,
    entries: list[tuple[tuple, str]],
    matched: list[tuple[tuple, str, float]],
    root: Any,
) -> Violation | None:
    """The ``at_bound`` signal for one quantity and one bound, after the zeros
    that are expected have been taken out of it. See the module docstring."""
    # The same cell reported under two groupings is one setting.
    by_key: dict[frozenset, tuple[tuple, str]] = {}
    for tokens, path in entries:
        by_key.setdefault(_setting_key(tokens), (tokens, path))
    settings = list(by_key.values())
    explained: list[tuple[str, str]] = []

    if bound == 0:
        all_settings: set[frozenset] = {_setting_key(t) for t, _, _ in matched}
        minority = len(settings) * 2 < len(all_settings)

        keep: list[tuple[tuple, str]] = []
        for tokens, path in settings:
            r0 = _r0_for(tokens, path, root)
            if r0 is not None and r0 < CRITICAL_R0:
                explained.append((path, f"R0 = {r0:g} is below the epidemic "
                                        f"threshold, so 0 is expected"))
            else:
                keep.append((tokens, path))

        # No name identified the control parameter, but the shape can: zeros
        # confined to one level of one coordinate are a regime boundary.
        if keep and minority:
            flagged_keys = {_setting_key(t) for t, _ in keep}
            pos = _confined_position(
                [_coords(t) for t, _ in keep],
                [_coords(t) for t, _, v in matched
                 if v != 0.0 and _setting_key(t) not in flagged_keys],
            )
            if pos is not None:
                for _, path in keep:
                    explained.append(
                        (path, f"every 0 sits at the same value of sweep "
                               f"coordinate {pos + 1}, and the quantity is "
                               f"non-zero at the other levels")
                    )
                keep = []
        settings = keep

    if len(settings) < AT_BOUND_MIN_SETTINGS:
        return None
    return Violation(
        a.path, bound, a, kind="at_bound",
        paths=tuple(p for _, p in settings),
        explained=tuple(explained),
    )


def check_design(result_json: Any, design: Any, *, code: str = "") -> list[Violation]:
    """Convenience: parse a design's assertions and apply them."""
    return violations(result_json, parse_assertions(design), code=code)
