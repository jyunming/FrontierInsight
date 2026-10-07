"""How a check against a known answer (an oracle, :mod:`core.oracle_check`) is written, so that the plan and the
simulation cannot mean two different numbers by it.

Two live quests failed on the same kind of mistake. A power-conservation check expected 1 (the ratio of output to input)
while the simulation measured 0 (the violation): the same physics in two representations, and the repair blamed the
check. A second quest expected a truncated 0.367879 within 1e-12, and step-size limits as the answers at a finite step.
The rule that an invariant's value is its worst violation, expected 0, existed only in a prompt. This module is where the
engine holds it:

* **One numeric form per kind.** :data:`VIOLATION_KINDS` (a conserved quantity or other invariant, a symmetry or
  scaling law, an independent second implementation): the number is the worst absolute (or declared relative)
  violation and the expected value is 0. :data:`QUANTITY_KINDS` (a special or limiting case, a published benchmark
  value, a convergence rate): the number is the quantity itself, with its expected value and tolerance.
  :func:`enforce` rewrites a check that does not fit when the rewrite cannot change its verdict (a violation-kind check
  whose number is a formula and whose expected value is not 0 becomes ``abs((formula) - expected)`` expecting 0), and
  otherwise returns a precise request for the plan.
* **How the number is computed** is part of the protocol: an oracle's ``measure`` is a formula of the names the
  simulation's ``run_cell``/``run_trial`` returns (``abs(P_out - P_in) / P_in``), in the small language of
  :func:`evaluate` (numbers, the returned names, ``+ - * / ** %``, and the functions in :data:`FUNCTIONS`), parsed with
  :mod:`ast` and never run as code. A ``measure`` that is one returned name is the formula of that name, so every
  oracle written before this keeps working. The engine applies the formula to what the simulation returned on the
  oracle's case (:func:`core.trial_runner.measure_oracles`): the script never decides the representation.
* **A test run of the checks before the study** (:func:`mismatch`): the oracle gate's first measurement, before the
  protocol is frozen, is read for a number whose size says the plan and the simulation mean different things by the
  check (a conservation check expecting 1 that measures about 0, a value orders of magnitude from its expected one), so
  the plan is asked once to look at the definition before any repair of the script is spent on it.
"""

from __future__ import annotations

import ast
import keyword
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Callable

from . import oracle_check as _oracle

#: The kinds whose number is the worst violation of a rule, expected to be 0.
VIOLATION_KINDS = ("invariant", "symmetry", "second_implementation")
#: The kinds whose number is the quantity itself, compared with a known value.
QUANTITY_KINDS = ("special_case", "published_value", "convergence_rate")

MAX_FORMULA_CHARS = 400
_MAX_NODES = 150


def _variadic(fn: Callable[..., float]) -> Callable[..., float]:
    def call(*args: float) -> float:
        if not args:
            raise ValueError("it is given nothing")
        return float(fn(args))
    return call


#: The functions a formula may call, and nothing else.
FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs, "min": _variadic(min), "max": _variadic(max), "sum": _variadic(math.fsum),
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "hypot": math.hypot,
    "floor": math.floor, "ceil": math.ceil,
}
#: The names a formula may use that are not returned by the simulation.
CONSTANTS: dict[str, float] = {"pi": math.pi}


def _ellipk(m: float) -> float:
    """The complete elliptic integral of the first kind in SciPy's convention, ``scipy.special.ellipk(m)`` with the
    parameter ``m = k**2``: ``pi / (2 * AGM(1, sqrt(1 - m)))`` (no SciPy needed)."""
    if not m < 1:
        raise ValueError("ellipk(m) needs m < 1")
    a, b = 1.0, math.sqrt(1.0 - m)
    for _ in range(64):
        if abs(a - b) <= 1e-16 * a:
            break
        a, b = (a + b) / 2, math.sqrt(a * b)
    return math.pi / (2 * a)


#: Functions only FI's own working-out of a derivation may use (core/oracle_triage.py::calculate), never a check's
#: formula: a plan writes its expected value with them, a simulation's returned names are never passed through them.
SPECIAL_FUNCTIONS: dict[str, Callable[..., float]] = {"ellipk": _ellipk}

_BINARY: dict[type, Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
    # math.pow, not **: a negative base to a fractional power is an error here, never a complex number.
    ast.Div: lambda a, b: a / b, ast.Pow: math.pow, ast.Mod: lambda a, b: a % b,
}
_UNARY: dict[type, Callable[[float], float]] = {ast.USub: lambda a: -a, ast.UAdd: lambda a: +a}
_WHAT: dict[type, str] = {
    ast.Attribute: "a dot (an attribute such as x.y)", ast.Subscript: "an index (such as x[0])",
    ast.Compare: "a comparison", ast.BoolOp: "and/or", ast.IfExp: "if/else", ast.Lambda: "a lambda",
    ast.List: "a list", ast.Tuple: "a tuple", ast.Dict: "a mapping", ast.Set: "a set",
    ast.ListComp: "a comprehension", ast.GeneratorExp: "a comprehension", ast.Starred: "a *",
    ast.NamedExpr: "an assignment",
}

FUNCTION_LIST = ", ".join(sorted(FUNCTIONS))
#: How a formula may be written, for the sentences a person and the plan read.
LANGUAGE = (
    "numbers, the names the simulation function returns, + - * / ** %, parentheses and the functions "
    f"{FUNCTION_LIST} (and the constant pi)"
)


def _tree(text: Any, functions: dict[str, Callable[..., float]] | None = None) -> tuple[ast.Expression | None, str]:
    """The parsed formula, or ``(None, why it cannot be read)``. ``functions``: the functions it may call (default
    :data:`FUNCTIONS`)."""
    functions = FUNCTIONS if functions is None else functions
    text = str(text if text is not None else "").strip()
    if not text:
        return None, "it is empty"
    if len(text) > MAX_FORMULA_CHARS:
        return None, f"it is longer than {MAX_FORMULA_CHARS} characters"
    if re.search(r"[\n\r#\x00]", text):
        return None, "it runs over more than one line or holds a comment"
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        return None, f"it is not a formula ({e.msg})"
    except (ValueError, RecursionError, MemoryError):
        return None, "it is not a formula"
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_NODES:
        return None, "it is too long a formula"
    calls = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
    for node in nodes:
        if isinstance(node, (ast.Expression, ast.Load)) or type(node) in _BINARY or type(node) in _UNARY:
            continue
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            op = node.op
            if type(op) not in _BINARY and type(op) not in _UNARY:
                return None, f"it uses the operator {type(op).__name__}, which a formula here may not"
            continue
        if isinstance(node, ast.Name):
            if id(node) in calls and node.id not in functions:
                return None, f"it calls {node.id}(), which is not one of {FUNCTION_LIST}"
            if id(node) not in calls and node.id in functions:
                return None, f"it uses the function {node.id} without calling it"
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return None, f"it holds {node.value!r}, which is not a number"
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return None, "it calls something that is not one of the functions a formula may use"
            if node.keywords:
                return None, f"it calls {node.func.id}() with a named argument"
            continue
        return None, f"it uses {_WHAT.get(type(node), type(node).__name__)}, which a formula here may not"
    return tree, ""


def problem(text: Any) -> str | None:
    """Why ``text`` cannot be read as a formula, or ``None`` when it can."""
    tree, why = _tree(text)
    return None if tree is not None else why


def names(text: Any, functions: dict[str, Callable[..., float]] | None = None) -> list[str]:
    """The names a formula takes from what the simulation returns (not its functions or constants), in order; empty
    when it cannot be read (with ``functions``, the functions it may call, as :func:`evaluate` reads it)."""
    tree, _ = _tree(text, functions)
    if tree is None:
        return []
    calls = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in calls and n.id not in CONSTANTS]
    found.sort(key=lambda n: (n.lineno, n.col_offset))  # as written, left to right
    return list(dict.fromkeys(n.id for n in found))


def is_name(text: Any) -> bool:
    """Whether ``text`` is one returned name alone (the form every oracle had before formulas)."""
    tree, _ = _tree(text)
    return tree is not None and isinstance(tree.body, ast.Name)


@dataclass
class Evaluated:
    """What a formula gave on one set of returned values: ``value`` (finite), or why not."""
    value: float | None = None
    unreadable: str = ""
    missing: list[str] = field(default_factory=list)
    problem: str = ""


def evaluate(text: Any, values: dict[str, Any], *, special: bool = False) -> Evaluated:
    """``text`` computed from ``values`` (what the simulation returned). Never runs code: the parsed tree is walked
    here, and each constant is a float, so no exponent can grow an integer without bound. ``special``: FI's own
    working-out of a derivation, which may also call :data:`SPECIAL_FUNCTIONS`."""
    functions = {**FUNCTIONS, **SPECIAL_FUNCTIONS} if special else FUNCTIONS
    tree, why = _tree(text, functions)
    if tree is None:
        return Evaluated(unreadable=why)
    missing = [n for n in names(text, functions)
               if not isinstance(values.get(n), (int, float)) or isinstance(values.get(n), bool)]
    if missing:
        return Evaluated(missing=missing)

    def real(result: Any) -> float:
        if isinstance(result, complex) or isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError("it gives a number that is not real")
        return float(result)

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return real(node.value)
        if isinstance(node, ast.Name):
            return real(CONSTANTS[node.id] if node.id in CONSTANTS and node.id not in values else values[node.id])
        if isinstance(node, ast.UnaryOp):
            return real(_UNARY[type(node.op)](walk(node.operand)))
        if isinstance(node, ast.BinOp):
            return real(_BINARY[type(node.op)](walk(node.left), walk(node.right)))
        if isinstance(node, ast.Call):
            return real(functions[node.func.id](*[walk(a) for a in node.args]))  # type: ignore[union-attr]
        raise ValueError(f"cannot compute {type(node).__name__}")

    try:
        value = walk(tree)
        finite = math.isfinite(value)
    except ZeroDivisionError:
        return Evaluated(problem="it divides by zero")
    except OverflowError:
        return Evaluated(problem="the number is too large")
    except (ValueError, TypeError, RecursionError, KeyError) as e:
        return Evaluated(problem=f"it cannot be computed ({e!r})")
    if not finite:
        return Evaluated(problem=f"it gives {value}")
    return Evaluated(value=value)


# --- one numeric form per kind -----------------------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt(value: float) -> str:
    return f"{value:g}" if abs(value) >= 1e-4 or value == 0 else f"{value:.3g}"


def _literal(value: float) -> str:
    """``value`` as a formula writes it, exactly (``repr`` of the float, without a trailing ``.0``)."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def written_precision(value: Any) -> tuple[int, float] | None:
    """``(significant digits, half a unit of the last one)`` of ``value`` as written in the plan (the shortest form
    that reads back as the same float), or ``None`` for 0 or a value that is not a number."""
    number = _num(value)
    if number is None or number == 0 or number.is_integer():
        return None  # a whole number (101325, 299792458) is exact as written
    if Fraction(number).denominator <= 2 ** 20:
        return None  # a short binary fraction (0.015625 = 1/64) is exact as written
    try:
        dec = Decimal(repr(number)).normalize()
    except InvalidOperation:
        return None
    digits = len(dec.as_tuple().digits)
    exponent = dec.as_tuple().exponent
    if not isinstance(exponent, int):
        return None
    return digits, 0.5 * 10.0 ** exponent


#: A value written with at least this many significant digits and fewer than a float holds reads as a truncated
#: decimal (0.367879 for exp(-1)), not an exact one (0.5, 2, 0.25).
_TRUNCATED_DIGITS = (5, 13)


def _criteria_on(protocol: dict[str, Any] | None, name: str, *, value_only: bool) -> list[str]:
    """The criteria that read this check (only those that take its measured value itself, ``use: value``, with
    ``value_only``): a rewrite would change the number they are judged on."""
    items = protocol.get("criteria") if isinstance(protocol, dict) else None
    out = []
    for c in items if isinstance(items, list) else []:
        if (isinstance(c, dict) and str(c.get("oracle") or "").strip().lower() == name.strip().lower()
                and (not value_only or str(c.get("use") or "").strip().lower() == "value")):
            out.append(str(c.get("name") or "?"))
    return out


# A returned key written with hyphens (``l2-ratio``) reads as a subtraction: FI cannot tell the two apart.
_HYPHEN_KEY_RE = re.compile(r"[A-Za-z_][\w]*(?:-[\w]+)+")


@dataclass
class Enforced:
    """What :func:`enforce` made of a protocol's checks: the checks (rewritten where that was safe), a sentence for
    each rewrite, and a request for the plan for each check that needs one."""
    oracles: list[dict[str, Any]]
    rewrites: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)


def _describe(kind: str | None) -> str:
    return _oracle.KINDS.get(kind or "", "check")


def enforce(protocol: dict[str, Any] | None, *, fi_runs: bool = True) -> Enforced:
    """Hold each declared check to its kind's numeric form (see the module docstring). A check of an unrecognised
    kind is left alone (the plan's notes already say its kind is none of the six). ``fi_runs``: FI runs the simulation
    on each check's case itself and computes its formula (two scripts, the trial contract); when it does not, the
    script reports each check's number by name, so a rewritten formula would not be applied and nothing is rewritten."""
    oracles = [dict(o) for o in _oracle.declared(protocol)]
    out = Enforced(oracles=oracles)
    names_seen = [str(o["name"]).strip().lower() for o in oracles]
    for oracle in oracles:
        name = str(oracle["name"]).strip()
        kind = _oracle.kind_of(oracle)
        expected = _num(oracle.get("expected"))
        tolerance = _num(oracle.get("tolerance"))
        relative = str(oracle.get("tolerance_mode") or "").strip().lower() == "relative"
        raw_measure = oracle.get("measure")
        measure = raw_measure.strip() if isinstance(raw_measure, str) else ""
        has_case = isinstance(oracle.get("case"), dict)
        if measure and has_case and not is_name(measure) and problem(measure) and re.search(r"[()+\-*/%]", measure):
            out.requests.append(
                f"The check {name!r} computes its number as `{measure}`, which cannot be read ({problem(measure)}): write "
                f"`measure` with {LANGUAGE}."
            )
            continue
        if measure and has_case and problem(measure) is None and not names(measure):
            out.requests.append(
                f"The check {name!r} computes its number as `{measure}`, which takes nothing the simulation returns, so "
                "it would pass or fail without the simulation: write `measure` from the names the simulation returns."
            )
            continue
        if kind in VIOLATION_KINDS and expected is not None and expected != 0:
            what = _describe(kind)
            formula = has_case and measure and not is_name(measure) and problem(measure) is None
            users = _criteria_on(protocol, name, value_only=not relative)
            new = (f"abs(({measure}) - {_literal(expected)})" if expected > 0
                   else f"abs(({measure}) + {_literal(-expected)})") + (f" / {_literal(abs(expected))}" if relative else "")
            why_not = (
                "FI does not run this quest's checks itself (the script reports each number), so a formula would not be "
                "applied" if not fi_runs else
                f"its `measure` `{measure}` may be one returned name written with hyphens" if formula and _HYPHEN_KEY_RE.fullmatch(measure) else
                "two checks have this name" if names_seen.count(name.lower()) > 1 else
                "the rewritten formula would be too long" if formula and problem(new) else ""
            )
            if formula and not users and not why_not and tolerance is not None and tolerance >= 0:
                oracle.update(measure=new, expected=0, tolerance=tolerance)
                oracle["tolerance_mode"] = "absolute"
                out.rewrites.append(
                    f"The check {name!r} is {what}, so its number is the worst violation and it expects 0. It was "
                    f"written expecting {_fmt(expected)} for `{measure}`; FI rewrote it as `{new}`, expecting 0 within "
                    f"{_fmt(tolerance)}{' (the same tolerance, now on the relative violation)' if relative else ''}. "
                    "Its verdict is the same as before."
                )
                continue
            if users:
                how = (f"the criteria {', '.join(repr(u) for u in users)} are judged on its number, so FI did not "
                       "rewrite it (change those criteria too, so they say the same thing of the new number)")
            elif why_not:
                how = why_not
            elif not has_case:
                how = ("it has no `case`, so the script's own oracle() decides how the number is computed; give it a "
                       "`case` and write `measure` as the formula of the violation")
            elif not measure or is_name(measure):
                how = (f"its `measure` `{measure or '(none)'}` does not say whether that number is the quantity "
                       f"(which should come out {_fmt(expected)}) or already its violation (which should come out 0)")
            else:
                how = "its tolerance cannot be read"
            out.requests.append(
                f"The check {name!r} is {what}: its number must be the worst violation, 0 when the rule holds, but it "
                f"expects {_fmt(expected)} and {how}. Write `measure` as a formula of the names the simulation returns "
                f"that gives the violation (for example `abs({measure if is_name(measure) else 'ratio'} - "
                f"{_literal(expected)})` when that name is the quantity that should be {_fmt(expected)}), set "
                "`expected: 0` and an absolute `tolerance`."
            )
            continue
        if kind == "convergence_rate" and expected is not None and expected <= 0:
            out.requests.append(
                f"The check {name!r} is a convergence rate, so its number is the observed order itself and it expects the "
                f"method's order (for example 4 for classical RK4), not {_fmt(expected)}: write `measure` as the formula "
                "of the observed order and `expected` as the order the method should show."
            )
            continue
        precision = written_precision(expected)
        if precision is not None and tolerance is not None:
            digits, half = precision
            limit = tolerance * abs(expected) if relative else tolerance  # type: ignore[arg-type]
            if _TRUNCATED_DIGITS[0] <= digits < _TRUNCATED_DIGITS[1] and 0 <= limit < half:
                out.requests.append(
                    f"The check {name!r} expects {_literal(expected)}, a value written to {digits} significant digits "  # type: ignore[arg-type]
                    f"(so known only to about ±{_fmt(half)}), but its tolerance is {_fmt(limit)}: a correct simulation "
                    "would fail it. Give the expected value to full precision (worked out to 15 digits in its "
                    "`reference`), or a tolerance no tighter than the precision it is written to."
                )
    return out


#: The section of plan.md that says what FI did to the checks before anything ran (prose, never read back).
HEADING = "A second look at the checks against known answers"


def apply_to_plan(text: str, *, fi_runs: bool = True) -> tuple[str, list[str], list[str]]:
    """``(text, rewrites, requests)``: ``text`` (a plan.md) with its design block's checks held to their kinds' forms
    (:func:`enforce`), a sentence per rewrite, and a request per check the plan has to change itself. The block is
    edited as written; a plan whose block cannot be read is returned as it is."""
    from . import plan as _plan

    found: dict[str, Enforced] = {}

    def change(block: dict[str, Any]) -> dict[str, Any] | None:
        protocol = block.get("protocol")
        if not isinstance(protocol, dict) or not isinstance(protocol.get("oracles"), list):
            return None
        enforced = enforce(protocol, fi_runs=fi_runs)
        found["e"] = enforced
        if not enforced.rewrites:
            return None
        by_name = {str(o["name"]).strip(): o for o in enforced.oracles}
        items = []
        for item in protocol["oracles"]:
            if isinstance(item, dict) and str(item.get("name") or "").strip() in by_name:
                new = by_name[str(item["name"]).strip()]
                item = {**item, **{k: new[k] for k in ("measure", "expected", "tolerance", "tolerance_mode") if k in new}}
            items.append(item)
        return {**block, "protocol": {**protocol, "oracles": items}}

    edited = _plan.edit_design_block(text, change)
    if edited is not None:
        edited = _plan.refresh_model_section(edited)
    enforced = found.get("e")
    if enforced is None:
        return text, [], []
    return (edited if edited is not None else text), (enforced.rewrites if edited is not None else []), enforced.requests


_SHOWN = ("kind", "measure", "case", "expected", "expected_formula", "tolerance", "tolerance_mode")
_TOLD = {"check": "what it checks is worded differently", "reference": "where its expected value comes from changed"}


def describe_changes(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    """One plain sentence per check the plan added, removed or changed between two versions (by name)."""
    out = []
    for name, new in after.items():
        old = before.get(name)
        if old is None:
            out.append(f"the check {name!r} was added (expects {new.get('expected')}, computed as "
                       f"`{new.get('measure') or 'the script’s own value'}`)")
            continue
        parts = [f"{k} {old.get(k)!r} → {new.get(k)!r}" for k in _SHOWN if old.get(k) != new.get(k)]
        parts += [said for k, said in _TOLD.items() if old.get(k) != new.get(k)]
        if parts:
            out.append(f"the check {name!r} changed: " + "; ".join(parts))
    out += [f"the check {name!r} was removed" for name in before if name not in after]
    return out


def request(requests: list[str], *, last: bool = True) -> str:
    """One request to the plan for every check that is not in its kind's form (``last``: it ends the request, so it
    says what else may change; not when another part follows that says it for both)."""
    return (
        "Each check against a known answer has one numeric form, fixed by its kind: a conserved quantity or invariant, "
        "a symmetry or scaling law, or a second implementation is measured as the worst violation and expects 0; a "
        "special or limiting case, a published value, or a convergence rate is measured as the quantity itself and "
        "expects its known value. These checks do not fit:\n"
        + "\n".join(f"- {r}" for r in requests)
        + ("\nChange only these checks (and a criterion that reads one of them, when its number changes meaning), and "
           "nothing else in the plan." if last else "")
    )


# --- the expected value, computed by FI itself, before anything runs -----------------------------------------------------
#
# A plan that writes an expected value from memory can be wrong in a way no check of its written steps can see (a real
# plan wrote K(0.0670) = 1.654 where the elliptic integral is 1.5981, and every multiplication around it was right). So
# each check may carry ``expected_formula``: the expected value as ONE formula, which FI computes itself, at full
# precision, with the calculator the checks' own formulas use. Before the protocol is frozen and before any run, when no
# measured value exists, a formula that disagrees with ``expected`` by more than the check's own tolerance is put to the
# plan once; if they still disagree FI uses the formula's value (the plan's own working, computed exactly), never a
# measured one.

#: How an ``expected_formula`` is written, generated from the calculator's own function tables so the two cannot drift.
EXPECTED_FORMULA_LANGUAGE = (
    "the expected value as one formula FI can compute itself: numbers, + - * / ** %, parentheses, pi, the functions "
    f"{FUNCTION_LIST}, and {', '.join(f'{n}(m)' for n in sorted(SPECIAL_FUNCTIONS))} with m = k**2 (SciPy's "
    "convention, so ellipk(sin(a/2)**2) for amplitude a), and the settings of the check's `case` by name; angles in "
    "radians (30 degrees is 30*pi/180)"
)


def case_numbers(oracle: dict[str, Any]) -> dict[str, float]:
    """The settings of ``oracle``'s ``case`` that are plain numbers, by name (what a formula may use)."""
    case = oracle.get("case") if isinstance(oracle.get("case"), dict) else {}
    return {str(k): float(v) for k, v in case.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def _plain_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def fixed_settings(protocol: dict[str, Any] | None) -> dict[str, float]:
    """The plan's fixed numeric settings by name, from the structured places a plan states them: ``protocol.thresholds``,
    a ``protocol.grid`` parameter with one value and ``protocol.optimisation.baseline.values`` (a design's controls given
    as ``{name, value}`` are added by :func:`fixed_settings_of_design`). Prose is never read. A name given two different
    numbers is dropped: it cannot be told which is meant."""
    found: dict[str, float] = {}
    clash: set[str] = set()

    def take(name: Any, value: Any) -> None:
        number = _plain_number(value)
        key = str(name).strip()
        if number is None or not key:
            return
        if key in found and found[key] != number:
            clash.add(key)
        found.setdefault(key, number)

    if isinstance(protocol, dict):
        thresholds = protocol.get("thresholds")
        for name, value in (thresholds.items() if isinstance(thresholds, dict) else []):
            take(name, value)
        for key in ("fixed", "constants", "controls"):
            stated = protocol.get(key)
            for name, value in (stated.items() if isinstance(stated, dict) else []):
                take(name, value)
        grid = protocol.get("grid")
        for name, values in (grid.items() if isinstance(grid, dict) else []):
            if isinstance(values, list) and len(values) == 1:
                take(name, values[0])
        block = protocol.get("optimisation")
        baseline = block.get("baseline") if isinstance(block, dict) else None
        values = baseline.get("values") if isinstance(baseline, dict) else None
        for name, value in (values.items() if isinstance(values, dict) else []):
            take(name, value)
    return {k: v for k, v in found.items() if k not in clash}


def fixed_settings_of_design(design: dict[str, Any] | None) -> dict[str, float]:
    """:func:`fixed_settings` of a design, with the controls of its ``variables`` given as ``{name, value}``."""
    if not isinstance(design, dict):
        return {}
    protocol = design.get("protocol") if isinstance(design.get("protocol"), dict) else None
    found = fixed_settings(protocol)
    variables = design.get("variables")
    controls = variables.get("controls") if isinstance(variables, dict) else None
    for item in (controls if isinstance(controls, list) else []):
        if isinstance(item, dict) and str(item.get("name") or "").strip() and _plain_number(item.get("value")) is not None:
            found.setdefault(str(item["name"]).strip(), float(item["value"]))
    return found


def _kept(name: str) -> str:
    """``name`` as the calculator can hold it: a word Python reserves (``lambda``) takes a trailing underscore."""
    return f"{name}_" if keyword.iskeyword(name) and name not in ("True", "False", "None") else name


def _renamed(text: str) -> str:
    return re.sub(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z_0-9]*)(?![A-Za-z0-9_])",
                  lambda m: _kept(m.group(1)), text)


def _formula_result(oracle: dict[str, Any], fixed: dict[str, float] | None) -> tuple[float | None, str, list[str], str]:
    r"""``(value, why not, names it uses that nothing sets, the formula as it was read)``. A formula is read as written;
    only one that cannot be read as written is read again with its LaTeX spelled as the calculator writes it
    (``\lambda`` / ``\lambda`` -> the name ``lambda``, ``\frac``, ``\cdot``, ``\sqrt{}``, ``^``). A name is
    set by the check's ``case``, else by the plan's fixed settings (:func:`fixed_settings`); never guessed."""
    text = oracle.get("expected_formula")
    if text is None or (isinstance(text, str) and not text.strip()):
        return None, "missing", [], ""
    if isinstance(text, bool) or not isinstance(text, (str, int, float)):
        return None, "it is not a formula", [], ""
    text = str(text)
    values = {**(fixed or {}), **case_numbers(oracle)}
    result = evaluate(text, values, special=True)
    used = text
    if result.unreadable:
        from .numeric_oracle import latex_to_arithmetic

        again = latex_to_arithmetic(text, names=True).strip()
        if again and again != text:
            held = _renamed(again)
            held_values = {_kept(k): v for k, v in values.items()}
            second = evaluate(held, held_values, special=True)
            if not second.unreadable:
                result, used = second, again
                if second.missing:
                    second.missing = [m[:-1] if m.endswith("_") and keyword.iskeyword(m[:-1]) else m
                                      for m in second.missing]
    if result.value is not None:
        return result.value, "", [], used
    if result.missing:
        return (None, f"it uses {', '.join(result.missing)}, which neither the check's case nor the plan's fixed "
                      "settings give as a number", list(result.missing), used)
    return None, result.unreadable or result.problem or "it cannot be computed", [], used


def formula_value(oracle: dict[str, Any], fixed: dict[str, float] | None = None) -> tuple[float | None, str]:
    """``(value, "")`` the check's ``expected_formula`` computes, or ``(None, why not)``; why is ``"missing"`` when the
    check has none. ``fixed``: the plan's fixed numeric settings (:func:`fixed_settings`), used for a name the check's
    ``case`` does not set."""
    value, why, _missing, _used = _formula_result(oracle, fixed)
    return value, why


#: Said to the plan, and in plan.md, about a formula that can be read two ways.
AMBIGUOUS_WHY = ("it can be read two ways: write it so it can only be read one way (angles in radians with pi, log10 or "
                 "ln named, ellipk(k**2))")


def formula_findings(protocol: dict[str, Any] | None, fixed: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """What FI's own computation of each check's expected value found, for the checks FI may correct that can be judged
    (a number ``expected`` and a ``tolerance``): ``{"name", "state", ...}`` with ``state`` ``agrees``, ``missing``,
    ``unusable`` (``why``), ``ambiguous`` (a disagreeing formula that can be read two ways: never applied) or ``differs``
    (``value``, ``expected``). A check whose number is a violation, expecting 0
    by its form, has no formula to compute. ``fixed``: the plan's fixed settings a formula may use by name (default: those
    of ``protocol``, :func:`fixed_settings`); an ``unusable`` finding names the ``names`` nothing sets."""
    from . import oracle_triage as _triage

    if fixed is None:
        fixed = fixed_settings(protocol)

    found: list[dict[str, Any]] = []
    for oracle in _oracle.declared(protocol):
        expected, limit, _mode = _oracle.limit_of(oracle)
        if expected is None or limit is None or not _triage.correctable(oracle):
            continue
        name = str(oracle["name"]).strip()
        value, why, unset, used = _formula_result(oracle, fixed)
        if value is None:
            found.append({"name": name, "state": "missing" if why == "missing" else "unusable", "why": why,
                          "formula": str(oracle.get("expected_formula") or ""), "names": unset})
        elif abs(value - expected) > limit and not math.isclose(value, expected, rel_tol=1e-12, abs_tol=0.0):
            formula = str(oracle.get("expected_formula"))
            if _triage.ambiguous(used or formula):
                # A formula that can be read two ways (sin(30) for 30 degrees, a bare log, ellipk of a modulus) is never
                # used to change `expected`: FI's reading may not be the plan's.
                found.append({"name": name, "state": "ambiguous", "value": value, "expected": expected,
                              "formula": formula, "why": AMBIGUOUS_WHY})
            else:
                found.append({"name": name, "state": "differs", "value": value, "expected": expected,
                              "formula": formula})
        else:
            found.append({"name": name, "state": "agrees", "value": value, "expected": expected})
    return found


def _digits(value: float) -> str:
    return f"{value:.12g}"


def formula_request(findings: list[dict[str, Any]], *, last: bool = True, fixed: dict[str, float] | None = None) -> str:
    """The part of the one request to the plan about the expected values FI computed itself: a check with no usable
    formula, and one whose formula gives another number than its ``expected`` (both numbers shown: no measured value
    exists yet). ``""`` when every check agrees."""
    lines = []
    for f in findings:
        if f["state"] == "missing":
            lines.append(f"- {f['name']!r}: give `expected_formula`.")
        elif f["state"] == "unusable" and f.get("names"):
            lines.append(f"- {f['name']!r}: its `expected_formula` `{f['formula']}` uses {', '.join(f['names'])}, which "
                         "are neither settings of this check's `case` nor fixed settings of the plan. Give the value of "
                         "each of these names (add it to the check's `case`), or write the formula with numbers.")
        elif f["state"] == "unusable":
            lines.append(f"- {f['name']!r}: its `expected_formula` `{f['formula']}` cannot be computed ({f['why']}); give "
                         "one that can.")
        elif f["state"] == "ambiguous":
            lines.append(f"- {f['name']!r}: its `expected_formula` `{f['formula']}` {f['why']}; FI reads it as "
                         f"{_digits(f['value'])} while `expected` says {_digits(f['expected'])}. Give it in a form that "
                         "can only be read one way, and the corrected `expected` if the formula is right.")
        elif f["state"] == "differs":
            lines.append(f"- {f['name']!r}: FI computed your `expected_formula` `{f['formula']}`: it gives "
                         f"{_digits(f['value'])}, but `expected` says {_digits(f['expected'])}. Give the corrected "
                         "`expected` and `expected_formula`.")
    if not lines:
        return ""
    return (
        "FI works out each expected value itself before anything runs, from a formula the plan gives: "
        f"{EXPECTED_FORMULA_LANGUAGE}. A name in a formula must be a setting of that check's `case`"
        + (f" or one of the plan's fixed settings ({', '.join(f'{k} = {_digits(v)}' for k, v in sorted(fixed.items()))})"
           if fixed else " (the plan states no fixed setting by name)")
        + f"; any other name cannot be computed. `expected` must be the number that formula gives, to full precision (a special "
        "function's value written from memory is the usual mistake: let the formula compute it).\n" + "\n".join(lines)
        + ("\nChange only these checks, and nothing else in the plan." if last else ""))


def apply_formulas(text: str) -> tuple[str, list[dict[str, Any]]]:
    """``(text, corrections)``: ``text`` (a plan.md) with each check's ``expected`` set to what its ``expected_formula``
    computes where the two still disagree by more than the check's own tolerance; ``tolerance``, ``tolerance_mode``,
    ``case`` and ``measure`` are never touched. Each correction is ``{"name", "was", "now", "formula"}``."""
    from . import plan as _plan

    done: list[dict[str, Any]] = []

    def change(block: dict[str, Any]) -> dict[str, Any] | None:
        protocol = block.get("protocol")
        if not isinstance(protocol, dict) or not isinstance(protocol.get("oracles"), list):
            return None
        wrong = {f["name"]: f for f in formula_findings(protocol, fixed_settings_of_design(block))
                 if f["state"] == "differs"}
        if not wrong:
            return None
        items = []
        for item in protocol["oracles"]:
            name = str(item.get("name") or "").strip() if isinstance(item, dict) else ""
            if name in wrong:
                f = wrong[name]
                done.append({"name": name, "was": f["expected"], "now": f["value"], "formula": f["formula"]})
                item = {**item, "expected": f["value"]}
            items.append(item)
        return {**block, "protocol": {**protocol, "oracles": items}}

    edited = _plan.edit_design_block(text, change)
    if edited is None or not done:
        return text, []
    return _plan.refresh_model_section(edited), done


# --- a test run of the checks before the study -------------------------------------------------------------------------

#: How far apart (a factor) a value and a non-zero expected value must be to read as a different definition.
_ORDERS = 100.0


def mismatch(oracle: dict[str, Any], value: Any, formula_problem: str = "", *, engine: bool = True,
             broad: bool = True) -> str | None:
    """A sentence when the number a check measured on its test run says the plan and the simulation mean different
    things by it (its definition, not the simulation, looks wrong), or ``None``. A value within its tolerance is never
    a mismatch; one that is simply outside it is the simulation's to explain (a repair), not this. ``broad``: also
    name the reciprocal and a count of the case (:func:`unit_multiple`), which the card says but the test run, whose
    sentences go to the plan as a request, does not."""
    name = str(oracle.get("name") or "?").strip()
    raw_measure = oracle.get("measure")
    measure = raw_measure.strip() if isinstance(raw_measure, str) else ""
    if formula_problem and engine:
        return (f"the check {name!r} computes its number as `{measure}`, and on its case {formula_problem}: the formula "
                "or the case does not fit what the simulation returns")
    number = _num(value)
    expected, limit, _mode = _oracle.limit_of(oracle)
    if number is None or expected is None or limit is None or abs(number - expected) <= limit:
        return None
    kind = _oracle.kind_of(oracle)
    # A formula FI computed states the representation (the plan chose it); a bare name or the script's own number does not.
    stated = engine and bool(measure) and not is_name(measure) and problem(measure) is None
    outer_abs = stated and _outer_call(measure) == "abs"
    tight = limit < abs(expected) / 10 if expected != 0 else True
    if kind in VIOLATION_KINDS and expected != 0 and tight and abs(number) <= max(limit, 1e-12):
        return (f"the check {name!r} is {_describe(kind)} and expects {_fmt(expected)}, but its test run measured "
                f"{_fmt(number)}: that reads as the violation (0 when the rule holds) where the plan expects the "
                "quantity itself (a ratio of 1, say)")
    if kind in VIOLATION_KINDS and expected == 0 and not outer_abs and abs(number - 1.0) <= max(limit, 1e-12):
        return (f"the check {name!r} is {_describe(kind)} and expects 0 (its worst violation), but its test run measured "
                f"{_fmt(number)}: that reads as a ratio that is kept at 1, not as a violation")
    multiple = unit_multiple(oracle, number, broad=broad)
    if multiple:
        return multiple
    if expected != 0 and number == 0:
        return (f"the check {name!r} expects {_fmt(expected)}, but its test run measured exactly 0: either the simulation "
                "returns another quantity than the one the check means, or it fails to compute it; which one must follow "
                "from the check's `reference`, not from this number")
    if expected != 0 and number != 0:
        ratio = abs(number / expected)
        if ratio >= _ORDERS or ratio <= 1 / _ORDERS:
            orders = abs(math.log10(ratio))
            return (f"the check {name!r} expects {_fmt(expected)}, but its test run measured {_fmt(number)}, about "
                    f"{orders:.0f} orders of magnitude away: either a unit or a representation differs (a per cent and a "
                    "fraction, a sum and a mean, a step-size limit and the value at a finite step) or the simulation is "
                    "wrong; which one must follow from the check's `reference`, not from this number")
        if (number > 0) != (expected > 0) and abs(abs(number) - abs(expected)) <= limit:
            return (f"the check {name!r} expects {_fmt(expected)}, but its test run measured {_fmt(number)}: the same "
                    "size with the other sign, so either the simulation or the check has the sign the other way; the "
                    "check's sign must follow from its `reference`, not from this number")
    return None


#: The factors one representation of a quantity differs from another by, and how a person says each.
_UNIT_FACTORS: tuple[tuple[float, str, str], ...] = (
    (100.0, "100", "a per cent and a fraction"),
    (1e3, "1000", "two units a thousand apart (milli- and the unit, or the unit and kilo-)"),
    (1e6, "10^6", "two units a million apart (micro- and the unit, or the unit and mega-)"),
    (1e9, "10^9", "two units a billion apart (nano- and the unit, or the unit and giga-)"),
    (2 * math.pi, "2π", "an angular frequency and a frequency (radians and cycles)"),
)


#: Case keys that count things a total sums over (a total and a mean differ by one of them).
_COUNT_KEY = re.compile(r"^(n|N|num|count|size|samples?|trials?|runs?|reps?|replicates?|agents?|particles?|customers?|"
                        r"nodes?|individuals?|people|population|members?|items?)$|^(n|num)_|_count$|^n[A-Z]")


def unit_multiple(oracle: dict[str, Any], value: Any, *, broad: bool = True) -> str | None:
    """A sentence when a measured value that fails its check would pass it after one of the usual changes of
    representation: times or divided by 100 (a per cent and a fraction), 1000, 10^6 or 10^9 (a unit prefix), 2π (an
    angular frequency and a frequency), the reciprocal (a rate and a time), or a whole-number count of the check's own
    case (a total and a mean over N; ``broad`` only, with the reciprocal). ``None`` otherwise. Only when the converted
    value lands within the check's own tolerance (never on a ratio that is merely close to a factor), the gap is at
    least ten tolerances (a factor the tolerance can tell apart), and never for an expected value of 0. Said, never a
    verdict: the check still fails."""
    number = _num(value)
    expected, limit, _mode = _oracle.limit_of(oracle)
    if number is None or expected is None or limit is None or number == 0 or expected == 0:
        return None
    if abs(number - expected) < 10 * limit:
        return None
    name = str(oracle.get("name") or "?").strip()

    def fits(x: float) -> bool:
        return abs(x - expected) <= limit

    factors = list(_UNIT_FACTORS)
    case = oracle.get("case")
    for key, n in (case.items() if isinstance(case, dict) and broad else []):
        count = _num(n)
        if (count is not None and count >= 2 and float(count).is_integer() and _COUNT_KEY.search(str(key))
                and count not in (f for f, _t, _d in factors)):
            factors.append((count, f"{key}={_fmt(count)}", f"a total and a mean over the case's {key}={_fmt(count)}"))
    head = f"the check {name!r} expects {_fmt(expected)}, but its test run measured {_fmt(number)}"
    for factor, shown, what in factors:
        if fits(number / factor):
            return (f"{head}, which is {shown} times the expected value: it looks like {what} (divided by {shown} it "
                    "would pass); which side is right must follow from the check's `reference`, not from this number")
        if fits(number * factor):
            return (f"{head}, which is 1/{shown} of the expected value: it looks like {what} (times {shown} it would "
                    "pass); which side is right must follow from the check's `reference`, not from this number")
    if broad and abs(abs(expected) - 1) > limit and fits(1 / number):
        return (f"{head}, which is the reciprocal of the expected value: it looks like a rate and a time (or a quantity "
                "and its inverse); which side is right must follow from the check's `reference`, not from this number")
    return None


def _outer_call(text: str) -> str:
    """The function the whole formula is an argument of (``abs`` for ``abs(a - b)``), or ``""``."""
    tree, _ = _tree(text)
    body = tree.body if tree is not None else None
    return body.func.id if isinstance(body, ast.Call) and isinstance(body.func, ast.Name) else ""


def mismatches(oracles: list[dict[str, Any]], checks: list[dict[str, Any]] | None) -> list[str]:
    """The :func:`mismatch` sentences for the checks of one run of the oracle gate (``checks`` as the gate records
    them: name, value, ``measured_by``, and ``formula_problem`` for a formula FI computed that gave no number; a check
    the script reported is read for its value only, never for text of its own)."""
    by_name = {str(c.get("name") or "").strip().lower(): c for c in checks or [] if isinstance(c, dict)}
    out = []
    for oracle in oracles:
        check = by_name.get(str(oracle.get("name") or "").strip().lower())
        if check is None:
            continue
        engine = check.get("measured_by") == "engine"
        why = mismatch(oracle, check.get("value"), str(check.get("formula_problem") or "") if engine else "",
                       engine=engine, broad=False)
        if why:
            out.append(why)
    return out


def passes_on(oracle: dict[str, Any], returned: Any, case: Any = None) -> bool | None:
    """Whether ``oracle`` (as the plan now states it) passes on the values the simulation returned at the test run
    on ``case`` (``None`` when that cannot be told: no values kept, a check now run on another case, a formula they do
    not answer, no numbers to judge by). A change to a check that makes the test run's own number pass is shown to the
    person as exactly that."""
    if not isinstance(returned, dict) or not returned or oracle.get("case") != case:
        return None
    measure = oracle.get("measure")
    if not isinstance(measure, str) or not measure.strip():
        return None
    measure = measure.strip()
    value = _num(returned.get(measure)) if measure in returned else evaluate(measure, returned).value
    expected, limit, _mode = _oracle.limit_of(oracle)
    if value is None or expected is None or limit is None:
        return None
    return abs(value - expected) <= limit


def dry_run_request(found: list[str]) -> str:
    """The one request to the plan after a test run of the checks showed a definition mismatch."""
    return (
        "A test run of the checks against known answers, made before the study, measured numbers whose size says the "
        "plan and the simulation mean different things by these checks:\n"
        + "\n".join(f"- {f[0].upper()}{f[1:]}." for f in found)
        + "\nFor each, look at the check's definition: the quantity it measures, its units and representation, and how "
        "its number is computed (`measure`, a formula of the names the simulation returns). If the definition was "
        "wrong, correct it and keep the kind's form (the worst violation expecting 0 for an invariant, a symmetry or a "
        "second implementation; the quantity itself for the others), and say in `reference` how the expected value "
        "follows. If the definition is right, leave the check exactly as it is: the simulation is then what gets "
        "repaired. Never set an expected value to the number measured here, and never make a check pass by changing "
        "its case or by computing its number from nothing the simulation returns. Change only these checks (and a "
        "criterion that reads one of them, when its number changes meaning), and nothing else in the plan."
    )
