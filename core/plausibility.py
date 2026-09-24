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
Which bounds carry this signal at all is decided two sections below.

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

Why a value on its bound is not outside it
==========================================
``1 - 1/3`` is 0.6666666666666667 and ``2/3`` is 0.6666666666666666. A design
that declared ``max`` = 2/3 and a script that computed ``1 - 1/R0`` agreed
about the answer and disagreed in the last bit, and an exact ``>`` called that
a violation: "branching_probability = 0.666667 violates [0, 0.666667]", the
same number printed on both sides of the word. The repair believed it, emitted
``null`` for every boundary value with a flag reading
``"exact_theoretical_boundary"`` — it knew the values were right — and the
reference line vanished from two panels of the figure that existed to show it.
A value equal to a bound to within ``_BOUND_REL_TOL`` IS on that bound, which
is the comparison the at-bound and clamped rules already used, and a value on a
bound is never also outside it. One equality, asked once: a result cannot be
too large by an amount no measurement can resolve.

Why a bound the design derived is attainable
============================================
"Exactly on a bound" is evidence about the computation only when the bound is a
number a fault can return by accident. That is true of 0 and 1 — nothing and
everything, which is what a saturating probability, a normalisation that
divides a quantity by itself, an empty count and a guard branch all produce —
and of any constant the script itself writes down, which is where a bracket
endpoint, a sentinel and a cap come from. It is not true of a bound the design
derived for this experiment: the design above got its 2/3 from the very
formula the run evaluates ("max(0, 1 - 1/R0) ranges from 0 to 2/3 for the
specified R0 values"), and nothing returns 0.6666666666666667 by accident. A
result equal to such a bound is the design's own prediction, attained where the
design said it would be, and is left alone — unless the quantity sits on that
bound at *every* setting the assertion matched, which is a quantity that never
varied with the sweep at all (a loop that captured one cell's value for all of
them looks exactly like this) and is the trivial answer this gate exists to
catch, whatever the bound happens to be.

Why a tiny value is not on 0
============================
The tolerance that keeps a value on its bound from reading as outside it is
absolute near 0 (``_BOUND_ABS_TOL``), so every value smaller than it was also
counted as *sitting on* 0. A live quest's one-sided Wilcoxon p-values were
8.5e-28 and similar (z about 10.9 over 183 pairs, the right answer), and the
at-bound rule reported "equals its bound 0 in 11 settings": two repairs were
spent on a correct script, each saying so. What a fault returns at 0 is an
exact 0 (an empty count, a guard branch, a saturated or underflowed
probability), and a non-zero value, however small, was computed from the data.
So on a bound of 0 only an exact 0 joins the at-bound count; a tiny value is
still inside the range, as before, and is simply not evidence of a trivial
answer.

Why a bounded quantity may not disappear
========================================
An assertion whose path matches nothing checks nothing, silently. That is
common from the start (a sweep of 83 quests found 63 assertions a design and its
script had named differently from the first run) and is not flagged: nothing
says which result key was meant. What is flagged is a quantity an earlier run
DID report that a later one leaves out altogether: a repair that renames or
deletes a bounded key leaves the gate nothing to check, and says nothing about
why. A key reported as ``null`` is not missing: that is the repair prompt's own
answer for a value the method cannot give, with a flag saying why. Both cases
seen in live quests were of that kind (a diverging Euler error set to null with
a "diverged" flag, and a p-value set to null and reported as its log10 after
the at-bound rule had wrongly sent it back twice), so neither is flagged.

Why a count of failures may sit at 0
====================================
0 in every setting is what a trivial computation returns, and it is also the
right answer for a count of failed trials when none failed. A live quest's
``failed_run_count`` was 0 in all nine settings, its run manifest listed no
failed trial, and the at-bound rule sent the correct script back twice, until
the repair hid the zeros as nulls. A name cannot tell the two apart; the
protocol can. A quantity the protocol's ``failure_policy`` names, whose 0 the
run's own manifest does not contradict, is exempt (the engine passes it as
``zero_expected``). Any other 0 is held to the rule as before.
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
    # "at_bound": exactly on one bound in several settings; "missing": an
    # earlier run reported this bounded quantity and this one does not.
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
        if self.kind == "missing":
            return (
                f"{self.path} is no longer in the results, though an earlier run reported it and the design bounds it "
                f"({self.assertion.describe()}){why}. Renaming or dropping a bounded quantity leaves its bound with "
                f"nothing to check: report it again under the name `{self.path}`, with the value the run computes "
                f"(a value outside the range is reported as it is; one the method cannot give is `null` with a flag "
                f"saying why, never left out)"
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


# A value this close to a bound is ON that bound. The at-bound and clamped
# rules already compared this way; the range test did not, which is how a
# result one bit above its declared maximum became a violation of it.
_BOUND_REL_TOL = 1e-9
_BOUND_ABS_TOL = 1e-12

# The bounds a fault can return without doing any physics: nothing, and
# everything.
_TRIVIAL_BOUNDS = (0.0, 1.0)


def numeric_literals(code: str) -> set[float]:
    """Every number the script writes down.

    A bound that appears here is a value the code can produce without
    computing anything: a bracket endpoint (``brentq(f, 0.0, 1.0)``), a
    sentinel, a cap, an initial value. A bound that appears nowhere in the
    script had to be arrived at. Unparseable code yields nothing rather than
    raising, which makes the rule no harsher on a script this cannot read.

    A number that appears only inside an ``assert`` is the script CHECKING for
    that value, not able to produce it, so it does not count. The implement step
    copies the design's declared bounds into the script as asserts
    (``assert 0.0 <= p <= 0.6666666667``); counting those made every derived
    bound look like a constant the code wrote down, and a result that legitimately
    reached the design's own predicted maximum in several settings was sent back
    for repair. A number the script also uses anywhere else still counts.
    """
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError):
        return set()
    checks = {
        id(inner)
        for node in ast.walk(tree) if isinstance(node, ast.Assert)
        for inner in ast.walk(node)
    }
    out: set[float] = set()
    for node in ast.walk(tree):
        if id(node) in checks:
            continue
        v = _literal(node)
        if v is not None:
            out.add(v)
    return out


def _trivial_bound(bound: float, literals: set[float]) -> bool:
    """Could a broken computation land on this bound by accident?

    True for 0 and 1, and for any constant the script writes down. False for
    a bound the design derived for this experiment, where a result equal to
    the bound is the prediction rather than a coincidence. See "Why a bound
    the design derived is attainable" in the module docstring.
    """
    if bound in _TRIVIAL_BOUNDS:
        return True
    return any(
        math.isclose(bound, c, rel_tol=_BOUND_REL_TOL, abs_tol=_BOUND_ABS_TOL)
        for c in literals
    )


def _pinned(value: float, a: Assertion, clamps: set[float]) -> bool:
    for bound in (a.min, a.max):
        if bound is None or bound == 0:
            continue
        if math.isclose(value, bound, rel_tol=1e-9) and any(
            math.isclose(c, bound, rel_tol=1e-9) for c in clamps
        ):
            return True
    return False


def _on_bound(value: float, a: Assertion) -> float | None:
    """The declared bound this value sits on, or None.

    A bound is a number with a precision, and a value that equals it to that
    precision is on it — never outside it. ``min`` is tried first so that an
    assertion whose bounds coincide reports the one a reader would name. See
    "Why a value on its bound is not outside it" in the module docstring.
    """
    for bound in (a.min, a.max):
        if bound is not None and math.isclose(
            value, bound, rel_tol=_BOUND_REL_TOL, abs_tol=_BOUND_ABS_TOL,
        ):
            return bound
    return None


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

# A range [0, m] with m no larger than this describes a tolerance rather than a
# scale, and its lower bound is where a correct answer lands. A residual of the
# equation a script just solved is the case seen: an exact 0.0 at six settings
# sent a quest through three repairs (14% of its tokens) that rewrote a correct
# solver to avoid returning a perfect residual.
TOLERANCE_MAX = 1e-3

# Below this the epidemic dies out, so a zero is the prediction rather than a
# failure. The value is the critical point of the branching-process
# approximation and is not tunable: R0 = 1 is where the behaviour changes.
CRITICAL_R0 = 1.0

# One path segment that is purely a number — a sweep coordinate rather than a
# field name.
_NUMERIC_SEG = re.compile(r"^-?\d+(?:\.\d+)?$")

# The reproduction number written into a path: ``by_R0.0.9``, ``by_r0_0.9``,
# ``by_R0_N.R0_0.9_N_100``, ``strata.R0=0.9,N=100``. Matched against the whole
# path, because splitting on "." cannot tell the key "0.9" from the two keys
# "0" and "9". A stratum label spells the coordinate with "=" and packs the
# sweep into one key, which leaves no numeric segment for the positional rule
# either — so without this spelling nothing at all identifies the control
# parameter and three correct zeros below the threshold went back for repair.
_R0_IN_PATH = re.compile(r"(?:^|[._\[])r_?0[._=]?(\d+(?:\.\d+)?)", re.IGNORECASE)

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


def _present_paths(obj: Any, tokens: tuple[tuple[str, Any], ...] = ()) -> Any:
    """The dotted path of every leaf, whatever it holds: a number, ``null``, NaN, a string, a flag. A key reported as
    ``null`` is still reported (the honest answer for a value the method cannot give)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _present_paths(v, tokens + (("k", str(k)),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _present_paths(v, tokens + (("i", i),))
    else:
        yield _dotted(tokens)


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
    result_json: Any, assertions: list[Assertion], *, code: str = "", zero_expected: Any = (),
) -> list[Violation]:
    """Every declared bound the result actually breaks.

    A value equal to a bound, to the precision a bound carries, sits on that
    bound rather than outside it. With ``code``, a value on a non-zero bound
    that the script also caps at is reported as ``kind="clamped"`` (see the
    module docstring). A quantity on one bound in ``AT_BOUND_MIN_SETTINGS`` or
    more settings is reported once, as ``kind="at_bound"``, when the bound is
    one a fault could have returned.
    """
    if not assertions:
        return []
    leaves = list(_walk_leaves(result_json))
    if not leaves:
        return []

    clamps = clamp_constants(code) if code else set()
    literals = numeric_literals(code) if code else set()
    out: list[Violation] = []
    for a in assertions:
        matched: list[tuple[tuple, str, float]] = []
        on_bound: dict[float, list[tuple[tuple, str]]] = {}
        for tokens, path, value in leaves:
            if not _matches(a.path, path):
                continue
            matched.append((tokens, path, value))
            bound = _on_bound(value, a)
            if bound is None:
                if (a.min is not None and value < a.min) or (
                    a.max is not None and value > a.max
                ):
                    out.append(Violation(path, value, a))
            elif clamps and _pinned(value, a, clamps):
                out.append(Violation(path, value, a, kind="clamped"))
            elif bound != 0 or value == 0:
                # On a bound of 0 only an exact 0 is the trivial answer; see "Why a tiny value is not on 0".
                on_bound.setdefault(bound, []).append((tokens, path))
        for bound, entries in on_bound.items():
            if bound == 0 and a.path in zero_expected:
                continue  # 0 is this quantity's answer: see "Why a count of failures may sit at 0"
            v = _at_bound(a, bound, entries, matched, result_json, literals)
            if v is not None:
                out.append(v)
    return out


def _at_bound(
    a: Assertion,
    bound: float,
    entries: list[tuple[tuple, str]],
    matched: list[tuple[tuple, str, float]],
    root: Any,
    literals: set[float],
) -> Violation | None:
    """The ``at_bound`` signal for one quantity and one bound, after the values
    that are expected have been taken out of it. See the module docstring."""
    # The same cell reported under two groupings is one setting.
    by_key: dict[frozenset, tuple[tuple, str]] = {}
    for tokens, path in entries:
        by_key.setdefault(_setting_key(tokens), (tokens, path))
    settings = list(by_key.values())
    explained: list[tuple[str, str]] = []
    all_settings: set[frozenset] = {_setting_key(t) for t, _, _ in matched}

    # A bound the design derived for this experiment, reached at some of the
    # settings and not at others, is the design's own prediction: the quantity
    # did vary with the sweep and arrived where the design said it would.
    if not _trivial_bound(bound, literals) and len(by_key) < len(all_settings):
        return None

    if bound == 0:
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
    # A range that ends just above 0 is a tolerance: a residual, a discretisation
    # error, an invariant's drift. The design asked for "as close to 0 as this
    # gets", so a 0 in several settings is the best answer, not a trivial one.
    if bound == 0 and a.min == 0 and a.max is not None and 0 < a.max <= TOLERANCE_MAX:
        return None
    return Violation(
        a.path, bound, a, kind="at_bound",
        paths=tuple(p for _, p in settings),
        explained=tuple(explained),
    )


def bounded_paths(result_json: Any, design: Any) -> list[str]:
    """The assertion paths this result reports at least one number for."""
    leaves = [path for _tokens, path, _value in _walk_leaves(result_json)]
    return [a.path for a in parse_assertions(design) if any(_matches(a.path, p) for p in leaves)]


def check_design(
    result_json: Any, design: Any, *, code: str = "", seen: Any = (), zero_expected: Any = (),
) -> list[Violation]:
    """Parse a design's assertions and apply them.

    ``seen`` holds the assertion paths an earlier run of this quest reported a number for (``bounded_paths``). One of
    them that this result no longer carries at all, not even as ``null``, is a ``kind="missing"`` violation: see "Why
    a bounded quantity may not disappear" in the module docstring. ``zero_expected`` holds the assertion paths whose 0
    is the answer (see "Why a count of failures may sit at 0").
    """
    assertions = parse_assertions(design)
    out = violations(result_json, assertions, code=code, zero_expected={str(p) for p in zero_expected or ()})
    earlier = {str(p) for p in seen or ()}
    if earlier:
        present = list(_present_paths(result_json))
        out += [
            Violation(a.path, math.nan, a, kind="missing")
            for a in assertions
            if a.path in earlier and not any(_matches(a.path, p) for p in present)
        ]
    return out
