"""A small test per equation of the plan's model, written by FI from the plan's own worked example, before the code runs.

The plan's model lists equations (``protocol.model.equations``). A plan whose headline quantity is defined by the wrong
equation is implemented faithfully by the code, and no check against a known answer that the same plan wrote can see it.
So for each equation that the code must compute as a function of named inputs (``role`` ``generates`` or ``analyses``)
the plan gives ONE worked example:

    "example": {"inputs": {"k": 2.0, "t": 0.5},          # the function's inputs, by name, as numbers
                "expected_formula": "exp(-k * t)"}       # its output, as a formula FI computes itself

or, when the equation has no closed form for its output, ``"example": {"untestable": "<why>"}``. FI computes the output
with its own calculator (:func:`core.oracle_forms.formula_value`; the formula may use only the example's inputs, never
a guessed or remembered number), and writes the test itself (:func:`test_source`, template code, never model-written):
``tests/test_equations.py`` imports the function that implements the equation (the one labelled ``# E<n>``, found by
:func:`locate`) and calls it with the inputs.

The contract with the code: the function is a top-level function whose parameters are named exactly as the example's
inputs and which returns the equation's output as one number. The outline step names one function per equation
(``implements``), so the contract is known before any body is written.

**The tolerance is a rule, not a guess.** A closed form agrees within a relative 1e-6 (and, only for an output that is exactly
0, an absolute 1e-12: the size of floating-point round-off). When the plan states a numerical method for the equation
(``"method"``), the plan's own numerical setting decides (``"tolerance"`` and ``"tolerance_mode"``, required with a
method); a stated tolerance without a method can only make the test tighter. FI never invents a looser one.

**An equation whose output is a random draw** (``"example": {"random": true, ...}``, the plan's own flag) is tested on the distribution, not on one call: ``expected_formula`` is the distribution's MEAN for those
inputs (and ``variance_formula``, when the plan states it, its variance). The test calls the function ``DRAWS`` times
with fixed seeds (a numpy generator seeded once and passed as the parameter ``rng``, or ``seed`` = 0, 1, 2, ... when the
function takes that), and holds the mean of the draws to ``SIGMAS`` standard errors of the stated mean (the standard
error is sigma / sqrt(n): sigma from the stated variance, else the draws' own spread) and, when the variance is stated,
the draws' variance to ``SIGMAS`` standard errors of a variance (from the draws' fourth moment). When the draws do not
vary at all, the output is deterministic and the strict rule above applies unchanged: a deterministic equation is
never judged by the statistical rule. A random equation whose plan gives no usable mean is not tested this way, and
says so; it is not failed.

A failed test is reported by its function and equation only (:func:`failure_message`): the expected value is never put
into a request to the model that writes code.

**``untestable`` is not a way out.** The equation the simulation generates its data with (``role`` ``generates``) may be
declared ``untestable`` only when something else checks it: an entry of the protocol's ``oracles`` of kind
``special_case`` (a limit), ``invariant`` (a conservation), ``symmetry`` or ``second_implementation`` whose ``check`` or
``reference`` names the equation (:func:`property_check_for`). Without one the equation is ``unprotected``: the plan is
asked once (in its one request about the checks) for a worked example or such a check, and when none comes the record
says plainly that the core equation is not tested at all (:func:`write_note`), which the paper's limitations repeat.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from . import equation_measure as _measure
from . import oracle_check as _oracle
from . import oracle_forms as _forms

#: The equations the code computes as functions (the plan's ``role``): the ones that get a worked example.
ROLES = ("generates", "analyses")
#: A closed form agrees within this relative tolerance; an output of exactly 0, within ``ABSOLUTE_FLOOR``.
RELATIVE = 1e-6
ABSOLUTE_FLOOR = 1e-12
#: A random equation: how many fixed-seed draws, the fewest that still count, the most time they may take, and how many
#: standard errors the draws' mean (and variance) may be from the stated one (core/equation_measure.py).
DRAWS = _measure.DRAWS
MIN_DRAWS = _measure.MIN_DRAWS
TIME_LIMIT_S = _measure.TIME_LIMIT_S
SIGMAS = _measure.SIGMAS
#: The parameters of the function that FI supplies to a random equation (never taken from the example's inputs).
RNG_PARAMS = _measure.RNG_PARAMS
SEED_PARAM = "seed"
TEST_NAME = "test_equations.py"
TESTS_DIR = "tests"
TEST_PATH = f"{TESTS_DIR}/{TEST_NAME}"
#: What the tests FI writes into ``code/tests/`` compare with (expected values, tolerances, the worked examples' inputs)
#: is never written into ``code/``: a coding agent that works in ``code/`` could read it there and special-case it. It
#: lives in the quest's ``.fi/checks/`` (beside ``code/``, outside it), where a person's run of the test file reads it.
#: FI's own verdict never comes from that file (:func:`run_and_judge`): the process that calls the functions is given
#: the inputs only, and FI compares what they return with these values in its own process.
CHECKS_DIR = ".fi/checks"
#: In ``.fi/checks/``: each equation's worked example (inputs, expected output, tolerance, variance), by equation id.
VALUES_NAME = "equation_values.json"
#: Files an older FI left in ``code/tests/`` that hold expected values (:func:`remove_stale`): removed on load.
STALE_NAMES = ("equation_numbers.json", "equation_values.json", "oracle_values.json")
#: The fields of a case that hold values a test compares with: kept out of the test file.
_HIDDEN = ("inputs", "expected", "rel", "abs", "variance")
#: The line the test file prints (with ``--json``), one JSON list of ``{"id", "function", "status", "detail"}``.
MARK = "EQUATION_TESTS:"

OK, MISMATCH, CONTRACT, ERROR = "ok", "mismatch", "contract", "error"
#: The function returned its outputs by name (a mapping) for a formula that defines several names, and the plan's
#: example gives one value: the test cannot tell which output the value is for. The equation is not tested before the
#: run (:func:`cannot_judge_note`, said in the limitations); it is never a failure that stops the study.
OUTPUTS = "outputs"
#: The kinds of check of the protocol that count as another check of an equation declared untestable.
PROPERTY_KINDS = ("special_case", "invariant", "symmetry", "second_implementation")

#: What the plan is told about the example (the design prompt, and the one request to the plan).
EXAMPLE_RULE = (
    "Each equation with role `generates` or `analyses` carries one worked example, `\"example\": {\"inputs\": "
    "{\"<name>\": <number>, ...}, \"expected_formula\": \"<the equation's output for those inputs, as a formula>\"}`. The "
    "code will have a function that implements the equation, with one parameter for each name in `inputs` (named exactly "
    "so) returning the equation's output as one number. FI computes `expected_formula` itself (the same formula language "
    "as a check's `expected_formula`, using only the names in `inputs` and numbers) and tests that function before the "
    "study runs; so give inputs that make the equation's terms matter (no zero or one that hides a factor), and compute "
    "the output from the equation as it is written, not from the answer you expect. A closed form is held to a "
    "relative 1e-6. When the equation is computed by a numerical method (a step size, a solver tolerance), add `\"method\"` "
    "(what it is) and `\"tolerance\"` with `\"tolerance_mode\"` (`relative` or `absolute`): that tolerance is then "
    "the test's. When the equation has no closed form for its output, write `\"example\": {\"untestable\": \"<why>\"}` "
    "instead; FI then says plainly that it is not tested before the run. When the equation's output is a random draw "
    "(a count or value drawn from a distribution), add `\"random\": true`: then `expected_formula` is the distribution's "
    "MEAN for those inputs (and, if you can state it, add `\"variance_formula\"`, its variance); the function takes one "
    "more parameter after the inputs, `rng` (a numpy random generator) or `seed`, draws with it, and FI calls it many "
    "times with fixed seeds and compares the mean (and variance) of the draws with yours, within five standard errors. "
    "When the formula defines several named outputs (`a = ...; b = ...`), the function returns them by name (a dict with "
    "those keys) and `expected_formula` is a mapping from each name the formula defines to its formula, one for every "
    "name.")


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def is_random(equation: dict[str, Any]) -> bool:
    """Whether the equation's output is a random draw: the plan marked its example (or the equation) ``"random": true``.
    Only that record decides. The formula's words never do (a ``~ N(0, 1)`` that defines a term of the formula, or the
    ``gamma(d)`` of a coherence function, is not a statement that the example's output is a draw: an example whose inputs
    already fix the output is a plain function of them, and the draw belongs to the step that generates its inputs); the
    plan is told to flag a random output (:data:`EXAMPLE_RULE`), and an unflagged draw fails the exact test, which puts
    the example back to the plan with that rule."""
    example = equation.get("example")
    unset = (None, False, "", 0, "false", "False")
    if isinstance(example, dict) and example.get("random") not in unset:
        return True
    return equation.get("random") not in unset  # the plan wrote the flag beside the example instead of inside it


_DEFINES = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)")


def _statements(formula: str) -> list[str]:
    """The formula split on ``;`` and line breaks, but only outside brackets (``f(a; b)`` stays one statement)."""
    out: list[str] = []
    depth, start = 0, 0
    text = str(formula or "")
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch in ";\n" and depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return [s for s in out if s.strip()]


def outputs(formula: str) -> list[str]:
    """The names a formula defines, in the order they are first defined, read from its text alone. The formula is split
    into statements on ``;`` and line breaks (outside brackets); a statement that starts with a bare name and a single
    ``=`` (``Ea = ...``) defines that name. A statement whose left side is anything else (``ln(k) = ...``, ``c(t) = ...``,
    an inequality, ``x == y``, a clause such as ``where x = 1``) is a relation or a remark. A name defined twice counts
    once.

    The text alone cannot tell an output from a term defined on the way (``s = (a+b+c)/2; area = sqrt(s*...)``), so
    these names never decide by themselves that an equation has several outputs: the plan says so by giving
    ``expected_formula`` as a mapping (its keys must be names the formula defines, every one of them), and the function
    says so by returning its outputs by name (:data:`OUTPUTS`, a record of the run, not a guess)."""
    seen: list[str] = []
    for statement in _statements(formula):
        m = _DEFINES.match(statement)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def is_multi_output(formula: str) -> bool:
    """Whether the formula defines several names (:func:`outputs`): only then may an example be a mapping, or a function
    that returns its outputs by name be told the example must give one value for each."""
    return len(outputs(formula)) >= 2


def eligible(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The plan's equations that get a worked example: an id, a formula and a role of ``generates`` or ``analyses``."""
    model = protocol.get("model") if isinstance(protocol, dict) else None
    view = _oracle.model_view(model)
    out: list[dict[str, Any]] = []
    for eq in view["equations"] if view else []:
        if (isinstance(eq, dict) and _text(eq.get("id")) and _text(eq.get("formula"))
                and _text(eq.get("role")).lower() in ROLES):
            out.append(eq)
    return out


def _tolerance(example: dict[str, Any]) -> tuple[float, float, str, str]:
    """``(relative, absolute, method, problem)`` for one example: the rule above. ``problem`` is non-empty when a
    stated method has no usable tolerance."""
    method = _text(example.get("method"))
    stated = _number(example.get("tolerance"))
    mode = _text(example.get("tolerance_mode")).lower() or "relative"
    if method:
        if stated is None or stated <= 0 or mode not in ("relative", "absolute"):
            return RELATIVE, ABSOLUTE_FLOOR, method, (
                "it names a numerical method but no tolerance of its own (`tolerance`, a positive number, and "
                "`tolerance_mode`: relative or absolute); the method's own setting decides, FI does not guess it")
        return (stated, ABSOLUTE_FLOOR, method, "") if mode == "relative" else (0.0, stated, method, "")
    if stated is not None and stated > 0 and mode == "relative":
        return min(RELATIVE, stated), ABSOLUTE_FLOOR, "", ""  # a stated tolerance can only make the test tighter
    return RELATIVE, ABSOLUTE_FLOOR, "", ""


def _stated_variance(example: dict[str, Any], inputs: dict[str, Any]) -> float | None:
    """The variance the plan states for a random equation (``variance_formula``, computed by FI from the example's inputs),
    or ``None`` when it states none or it cannot be computed or is not a positive number: the draws' own spread is then
    used, and nothing is claimed about the variance."""
    formula = example.get("variance_formula")
    if formula is None or not _text(formula):
        return None
    value, _why, _unset, _used = _forms._formula_result({"expected_formula": formula, "case": inputs}, None)
    return float(value) if value is not None and math.isfinite(float(value)) and float(value) > 0 else None


def property_check_for(eid: str, protocol: dict[str, Any] | None) -> str:
    """The name of a check of the protocol that is another check of equation ``eid``: an entry of ``oracles`` of kind
    special case (a limit), invariant (a conservation), symmetry or second implementation whose name, check or reference
    names the equation. ``""`` when there is none."""
    for o in _oracle.declared(protocol):
        if _oracle.kind_of(o) not in PROPERTY_KINDS:
            continue
        text = " ".join(_text(x) for x in (o.get("name"), o.get("check"), _oracle.reference_of(o)))
        if _oracle._labelled_in(eid, text):
            return _text(o.get("name"))[:80] or "a check of the plan"
    return ""


def example_rows(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per equation that should have a worked example: ``{"id", "formula", "state", "why", "inputs",
    "expected", "rel", "abs", "method", "random", "variance"}``. ``state``: ``ok`` (FI computed the output), ``missing`` (no example),
    ``unusable`` (``why``: the example cannot be used, never repaired by guessing), ``untestable`` (the plan says
    the equation has no closed form for its output, and why; for an equation the simulation generates its data with
    another check of it must exist: ``covered_by`` names it) or ``unprotected`` (untestable, and nothing else checks it)."""
    rows: list[dict[str, Any]] = []
    for eq in eligible(protocol):
        found = outputs(str(eq.get("formula") or ""))
        names = found if len(found) >= 2 else []  # the names a mapping example must cover, and a by-name return may hold
        row = {"id": _text(eq["id"]), "formula": _text(eq["formula"]), "state": "missing",
               "why": "the plan gave no worked example", "inputs": {}, "expected": None,
               "rel": RELATIVE, "abs": ABSOLUTE_FLOOR, "method": "", "random": is_random(eq), "variance": None,
               "outputs": names}
        rows.append(row)
        example = eq.get("example")
        if example is None or (isinstance(example, str) and not example.strip()):
            continue
        if not isinstance(example, dict):
            row.update(state="unusable", why="its `example` must be a mapping with `inputs` and `expected_formula`")
            continue
        reason = _text(example.get("untestable"))
        if reason:
            row.update(state="untestable", why=reason)
            if _text(eq.get("role")).lower() == "generates":
                covered = property_check_for(row["id"], protocol)
                if covered:
                    row["covered_by"] = covered
                else:
                    row["state"] = "unprotected"
            continue
        inputs = example.get("inputs")
        good = {str(k).strip(): _number(v) for k, v in inputs.items()} if isinstance(inputs, dict) else {}
        if row["random"]:  # FI supplies the generator and the seed; they are not inputs of the example
            good = {k: v for k, v in good.items() if k not in (*RNG_PARAMS, SEED_PARAM)}
        if not good or any(v is None or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k) for k, v in good.items()):
            row.update(state="unusable", why="`inputs` must be a mapping of names (that are valid parameter names) to numbers")
            continue
        formula = example.get("expected_formula")
        if isinstance(formula, dict):  # the plan says the equation has several outputs: one formula for each
            if names:
                _multi_output_row(row, example, good, names)
            else:
                row.update(state="unusable", why=(
                    "its `expected_formula` is a mapping, but the formula defines at most one name (`name = ...`); give "
                    "`expected_formula` as one formula, or write each output in the formula as `name = ...`"))
            continue
        if formula is None or not _text(formula):
            row.update(state="unusable", why=(
                "its output is a random draw, so it cannot be held to one exact value, and the plan gave inputs but no "
                "`expected_formula` (the distribution's mean)"
                if row["random"] else "it gave inputs but no `expected_formula`"))
            continue
        value, why, unset, _used = _forms._formula_result({"expected_formula": formula, "case": good}, None)
        if value is None:
            row.update(state="unusable", why=(
                f"its `expected_formula` uses {', '.join(unset)}, which are not among its `inputs`" if unset
                else f"its `expected_formula` cannot be computed ({why})")
                + (" (for a random draw it is the distribution's mean)" if row["random"] else ""))
            continue
        rel, abs_tol, method, problem = _tolerance(example)
        if problem:
            row.update(state="unusable", why=problem)
            continue
        if float(value) != 0.0 and abs_tol == ABSOLUTE_FLOOR:
            abs_tol = 0.0  # the round-off floor is only for an output that is exactly 0: a small output is held to its size
        row.update(state="ok", why="", inputs={k: float(v) for k, v in good.items()}, expected=float(value), rel=rel,
                   abs=abs_tol, method=method)
        if row["random"]:
            row["variance"] = _stated_variance(example if "variance_formula" in example else eq, good)
            continue
        own = equation_value(row["formula"], row["inputs"])
        if own is not None:
            row["equation_value"] = own
            gap = abs(own - float(value))
            # Agrees: code computing the equation would pass the test. Disagrees: beyond the test's bar and round-off
            # too (an output near 0 is held to its size by the test, but two ways of working it out may differ there).
            if gap <= max(rel * abs(float(value)), abs_tol):
                row["equation_agrees"] = True
            elif gap > max(rel * abs(float(value)), ABSOLUTE_FLOOR):
                row["equation_agrees"] = False
            if row.get("equation_agrees") is False:
                # Code that follows the equation would fail this example: the example, not the code, is wrong.
                row.update(state="unusable", why=(
                    f"its `expected_formula` gives {float(value):.10g} for its inputs, but the equation as written gives "
                    f"{own:.10g} for the same inputs (FI computed both)"))
    return rows


_DEFINITION = re.compile(r"^\s*[A-Za-z_]\w*\s*(\([^()=]*\))?\s*=(?![=<>])(.*)$", re.S)


def equation_value(formula: str, inputs: dict[str, float]) -> float | None:
    """What the equation as written gives for ``inputs``, computed by FI, or ``None`` when its text cannot be read that
    way for sure: only one statement that defines one name (``k = A*exp(-Ea/(R*T))`` or ``k(T) = ...``) whose right side
    FI's calculator reads from the inputs alone, with the functions it reads in prose (a plan's equation is prose, not the
    language FI states for ``expected_formula``), and that cannot be read two ways (a bare ``log``, an angle with no
    unit). A relation, a distribution, several definitions or index notation are not read (in the suite's rounds, 35 of 90
    worked examples could be read this way, and none disagreed with its equation)."""
    from . import oracle_triage as _triage

    statements = _statements(formula)
    if len(statements) != 1:
        return None
    m = _DEFINITION.match(statements[0])
    if m is None:
        return None
    right = m.group(2).strip()
    if not right or re.search(r"[=<>~]", right) or _triage.ambiguous(right):
        return None
    # `f(T) = ...`: each argument the left side names must be used on the right, or the text says two things.
    used = set(_forms.names(right, {**_forms.FUNCTIONS, **_forms.PROSE_FUNCTIONS}))
    if m.group(1) and any(a.strip() and a.strip() not in used for a in m.group(1).strip("()").split(",")):
        return None
    return _forms.evaluate(right, inputs, special=_forms.PROSE_FUNCTIONS).value


def _multi_output_row(row: dict[str, Any], example: dict[str, Any], good: dict[str, float], names: list[str]) -> None:
    """Judge the example of an equation whose formula defines several named outputs (:func:`outputs`): ``expected_formula``
    must be a mapping from every output's name to its formula, each computed by FI from the example's inputs. The row's
    ``expected`` is then ``{name: number}`` and its ``abs`` is ``{name: absolute tolerance}`` (the round-off floor only for
    an output that is exactly 0). A random equation with several outputs is not tested this way: it must be split."""
    listed = ", ".join(names)
    if row["random"]:
        row.update(state="unusable", why=f"its output is a random draw and the formula defines several outputs ({listed}); "
                                         "a random equation is tested one output at a time, so split it into one equation per output")
        return
    formula = example.get("expected_formula")
    given = {str(k).strip(): v for k, v in formula.items()}
    absent = [n for n in names if n not in given or given[n] is None or not _text(given[n])]
    extra = [k for k in given if k not in names]
    if absent or extra:
        parts = ([f"no formula for {', '.join(absent)}"] if absent else []) + (
            [f"{', '.join(extra)} is not an output of the formula" if len(extra) == 1 else
             f"{', '.join(extra)} are not outputs of the formula"] if extra else [])
        row.update(state="unusable", why=f"its `expected_formula` must give a formula for each output ({listed}): "
                                         + "; ".join(parts))
        return
    values: dict[str, float] = {}
    for n in names:
        value, why, unset, _used = _forms._formula_result({"expected_formula": given[n], "case": good}, None)
        if value is None:
            row.update(state="unusable", why=(
                f"its `expected_formula` for {n} uses {', '.join(unset)}, which are not among its `inputs`" if unset
                else f"its `expected_formula` for {n} cannot be computed ({why})"))
            return
        values[n] = float(value)
    rel, abs_tol, method, problem = _tolerance(example)
    if problem:
        row.update(state="unusable", why=problem)
        return
    floors = {n: (0.0 if v != 0.0 and abs_tol == ABSOLUTE_FLOOR else abs_tol) for n, v in values.items()}
    row.update(state="ok", why="", inputs={k: float(v) for k, v in good.items()}, expected=values, rel=rel, abs=floors,
               method=method)


def request(rows: list[dict[str, Any]], *, last: bool = True) -> str:
    """The part of the one request to the plan that asks for the worked examples a plan lacks (``""`` when none)."""
    lines = []
    for r in rows:
        if r["state"] == "missing":
            lines.append(f"- {r['id']} (`{r['formula']}`): give its `example`.")
        elif r["state"] == "unusable":
            lines.append(f"- {r['id']} (`{r['formula']}`): its `example` cannot be used: {r['why']}.")
        elif r["state"] == "unprotected":
            lines.append(f"- {r['id']} (`{r['formula']}`): the plan says it has no closed form for its output ({r['why']}), "
                         "and nothing else checks it. Either give it a worked example, or add to the protocol's `oracles` "
                         f"a check against a known answer that names {r['id']} in its `check` or `reference`: a limit "
                         "(`special_case`), a conserved quantity (`invariant`), a `symmetry` or an independent second "
                         "implementation (`second_implementation`).")
    if not lines:
        return ""
    return ("FI tests each equation of the model before the study runs, from a worked example the plan gives. "
            + EXAMPLE_RULE + "\n" + "\n".join(lines)
            + ("\nChange only these equations' `example`, and nothing else in the plan." if last else ""))


#: How a failed example is put to the plan, in plain words (:func:`rederive_request`): never the number the code gave.
_COULD_NOT_MEET = {
    MISMATCH: "code written from the formula computes another number than the example says",
    CONTRACT: "the function written from the formula cannot take the example's inputs (its parameters are not the "
              "example's input names)",
    ERROR: "the function written from the formula failed when it was called with the example's inputs",
}


def repairable_by_example(result: dict[str, Any]) -> bool:
    """Whether a failed equation test may be a fault of the worked example (so the plan is asked about it): a wrong number,
    inputs the function cannot take, or an error when called. Not a function that cannot be found or imported."""
    status = result.get("status")
    if status in (MISMATCH, CONTRACT):
        return True
    detail = _text(result.get("detail"))
    return status == ERROR and not re.match(r"(?:ImportError|ModuleNotFoundError|SyntaxError|AttributeError)\b", detail)


def example_as_written(protocol: dict[str, Any] | None, eid: str) -> Any:
    """The ``example`` the plan wrote for equation ``eid`` (``None`` when there is none)."""
    for eq in eligible(protocol):
        if _text(eq["id"]) == eid:
            return eq.get("example")
    return None


def rederive_request(items: list[dict[str, Any]]) -> str:
    """The one request to the plan that asks it to work a worked example out again from its equation: for each item
    (``id``, ``formula``, ``example`` as written, ``why``: a key of :data:`_COULD_NOT_MEET` or a sentence) the equation, the
    example as written and the kind of failure. It never holds the value the code computed, nor the code."""
    lines = []
    for it in items:
        why = _COULD_NOT_MEET.get(it.get("why"), _text(it.get("why")))
        written = json.dumps(it.get("example"), ensure_ascii=False, default=str) if it.get("example") is not None else "none"
        names = [n for n in (it.get("names") or []) if n not in (*RNG_PARAMS, SEED_PARAM) and not n.startswith("*")]
        fixed = (f" The function for it is already planned with the parameters ({', '.join(names)}): give the example's "
                 "`inputs` under exactly those names." if names else "")
        lines.append(f"- {it['id']} (`{it['formula']}`): the example you gave, `{written}`, could not be met: {why}.{fixed}")
    return ("FI tested each equation of the model before the study ran, from the worked example the plan gave. For the "
            "equations below the example could not be met by code that follows the equation as written, so the example "
            "may be the fault. Work each one out again from its equation as it is written: `inputs` are that equation's own "
            "variables (the names its formula uses, which are the parameters of the function that computes it; for an "
            "equation with several named outputs, the names the output formulas use), each a "
            "number, and `expected_formula` is the equation's output for exactly those inputs. " + EXAMPLE_RULE + "\n"
            + "\n".join(lines) + "\nChange only these equations' `example`, and nothing else in the plan.")


def unsupplied(required: list[str], inputs: Any, *, random: bool = False) -> list[str]:
    """The parameters without a default that the test would not give a value: not one of the example's ``inputs`` (and, for a
    random equation, not ``rng``/``random_state``/``generator``/``seed``, which FI supplies). The test file applies the same
    rule (``_run`` in :data:`_TEMPLATE`, which cannot import this module); a test checks that the two agree."""
    given = set(inputs)
    if random:
        given |= {*RNG_PARAMS, SEED_PARAM}
    return [n for n in required if n not in given]


def parameter_problems(rows: list[dict[str, Any]],
                       params: dict[str, tuple[list[str], list[str]]]) -> dict[str, str]:
    """For each equation whose example is usable and whose function is already named with its parameters (``params``: the
    equation's id to ``(every parameter, the parameters without a default)`` the outline fixed): why the test would fail
    on names alone, in plain words. That is exactly the test's own contract: every input of the example must be a
    parameter, and a parameter without a default must be an input (for a random equation ``rng`` or ``seed`` is FI's);
    a function that takes ``*args`` or ``**kwargs`` is not judged. Only names are compared, nothing is guessed."""
    out: dict[str, str] = {}
    for r in rows:
        names, required = params.get(r["id"]) or (None, [])
        if r["state"] != "ok" or names is None or any(n.startswith("*") for n in names):
            continue
        want = set(r["inputs"])
        if not want <= set(names) or unsupplied(required, want, random=bool(r.get("random"))):
            out[r["id"]] = (f"the function for it is planned with the parameters ({', '.join(names)}), but the example's "
                            f"inputs are ({', '.join(sorted(want))}); they must be the same names")
    return out


def with_examples(protocol: dict[str, Any] | None, examples: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """``protocol`` with the worked example of each equation in ``examples`` (id to ``{"formula", "example"}``) replaced,
    for an equation whose formula is still the one the example was worked out from. Nothing else changes; the protocol
    given is not modified."""
    model = protocol.get("model") if isinstance(protocol, dict) else None
    if not examples or not isinstance(model, dict):
        return protocol
    key = next((k for k in _oracle._EQUATION_KEYS if _oracle.equation_items(model.get(k))), None)
    if key is None:
        return protocol
    items = [dict(e) for e in _oracle.equation_items(model[key])]
    changed = False
    for eq in items:
        new = examples.get(_text(eq.get("id")))
        if new and _text(eq.get("formula")) == _text(new.get("formula")) and eq.get("example") != new.get("example"):
            eq["example"] = new["example"]
            changed = True
    return {**protocol, "model": {**model, key: items}} if changed else protocol


def cannot_judge_note(eid: str) -> str:
    """The plain sentence for an equation the test could not judge (:data:`OUTPUTS`)."""
    return (f"equation {eid} is not tested before the run: its function returns several outputs by name and the plan's "
            "worked example gives one value, so FI cannot tell which output the value is for")


def untested_notes(rows: list[dict[str, Any]], located: dict[str, dict[str, str] | None]) -> list[str]:
    """One plain sentence for each equation that will not be tested before the run, and why."""
    notes = []
    for r in rows:
        if r["state"] == "missing":
            notes.append(f"equation {r['id']} is not tested before the run: the plan gave no worked example for it")
        elif r["state"] == "unusable":
            notes.append(f"equation {r['id']} is not tested before the run: its worked example cannot be used "
                         f"({r['why']})")
        elif r["state"] == "unprotected":
            notes.append(f"equation {r['id']} is not tested at all: the plan says it has no closed form for its output "
                         f"({r['why']}) and no limit, symmetry, conservation or second implementation checks it")
        elif r["state"] == "untestable":
            notes.append(f"equation {r['id']} is not tested before the run: the plan says it has no closed form "
                         f"for its output ({r['why']})"
                         + (f"; the check '{r['covered_by']}' tests it instead" if r.get("covered_by") else ""))
        elif not located.get(r["id"]):
            notes.append(f"equation {r['id']} is not tested before the run: no function of the code that can be "
                         "imported is labelled with it (`# " + r["id"] + "`)")
    return notes


# --- finding the function that implements an equation -----------------------------------------------------------------


def locate(protocol: dict[str, Any] | None, sources: dict[str, str]) -> dict[str, dict[str, Any] | None]:
    """For each equation that gets an example: the importable top-level function labelled ``# E<n>`` (a comment on or
    above it, or in its docstring), as ``{"module", "function", "file", "static"}`` (``static``: FI's own reading of
    it, :func:`static_facts`); ``None`` when there is none. ``sources``:
    ``{"<package>/model.py": text, "simulate.py": text}`` (a script that runs when imported, ``experiment.py``, is
    never one of them). A package is preferred to ``simulate.py``."""
    from . import code_layout as _layout

    labelled = {rel: _layout._labels(text) for rel, text in sorted(sources.items(), key=lambda kv: (kv[0] == "simulate.py", kv[0]))}
    out: dict[str, dict[str, str] | None] = {}
    for eq in eligible(protocol):
        eid = _text(eq["id"])
        found = None
        for rel, items in labelled.items():
            for _line, text, fn in items:
                if fn and "." not in fn and _oracle._labelled_in(eid, text):
                    found = {"module": module_of(rel), "function": fn, "file": rel,
                             "static": static_facts(sources[rel], fn)}
                    break
            if found:
                break
        out[eid] = found
    return out


def module_of(rel: str) -> str:
    """The import name of a file of ``code/`` (``pkg/model.py`` -> ``pkg.model``; ``pkg/__init__.py`` -> ``pkg``)."""
    path = rel.replace("\\", "/")
    path = path[:-3] if path.endswith(".py") else path
    path = path[: -len("/__init__")] if path.endswith("/__init__") else path
    return path.replace("/", ".")


def cases(rows: list[dict[str, Any]], located: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    """The tests to write: one per equation whose example is ``ok`` and whose function is found."""
    out = []
    for r in rows:
        where = located.get(r["id"])
        if r["state"] == "ok" and where:
            out.append({"id": r["id"], "formula": r["formula"], "module": where["module"], "function": where["function"],
                        "inputs": r["inputs"], "expected": r["expected"], "rel": r["rel"], "abs": r["abs"],
                        "method": r["method"], "random": bool(r.get("random")), "variance": r.get("variance"),
                        "outputs": list(r.get("outputs") or []), "static": where.get("static")})
    return out


# --- the test file FI writes -------------------------------------------------------------------------------------------

_TEMPLATE = '''"""A small test per equation of the plan's model. Written by Frontier Insight (never by the model that wrote the code)
from the plan's own worked example: the inputs are the plan's, and the output is computed by FI from the plan's formula.

    python -m pytest tests/test_equations.py        (or: python tests/test_equations.py)

Each test imports the function labelled with the equation (`# E1`) and calls it with the example's inputs, by name.
The inputs and the output are not in this folder: Frontier Insight writes them into the quest's
`.fi/checks/{values_name}` (next to `code/`) once the quest's code has run for the last time. Run the tests inside the quest folder, or name the folder
that holds that file: python tests/test_equations.py --checks <folder>.

This file is a copy for a person. Frontier Insight does not take its verdict from it: it calls each function in a
process of its own that is given only the function and its inputs, and compares the number it returns with the
expected value in its own process.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _checks_dir():
    if "--checks" in sys.argv[:-1]:
        return Path(sys.argv[sys.argv.index("--checks") + 1])
    return Path(__file__).resolve().parent.parent.parent / ".fi" / "checks"


VALUES_NAME = {values_name!r}


def _values():
    try:
        data = json.loads((_checks_dir() / VALUES_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {{}}
    return data if isinstance(data, dict) else {{}}


_VALUES = _values()
CASES = [{{**c, **_VALUES[c["id"]]}} if isinstance(_VALUES.get(c["id"]), dict) else c for c in json.loads({cases!r})]

# --- the same measure and judge Frontier Insight runs (core/equation_measure.py), here in one process ---
@@MEASURE@@


def _run(case):
    """(status, detail, value, tolerance, expected): status ok | mismatch | contract | outputs | error."""
    if any(k not in case for k in ("inputs", "expected", "rel", "abs")):
        # Never a pass: without FI's values the test has nothing to compare with.
        return ("error", "the worked example this test compares with is not here (Frontier Insight keeps it in the "
                "quest's .fi/checks/" + VALUES_NAME + ", outside code/); run the tests inside the quest folder",
                None, None, None)
    case = dict(case, names=list(case["expected"]) if isinstance(case["expected"], dict) else None)
    return judge(case, measure(case))


def _check(index):
    case = CASES[index]
    status, detail, value, tol, want = _run(case)
    assert status == "ok", (case["id"] + " (" + case["function"] + "): " + status + " " + detail
                            + (" got " + repr(value) + ", the worked example gives " + repr(want)
                               + (", the tolerance is " + repr(tol) if tol is not None else "")
                               if value is not None else ""))
'''


def test_source(case_list: list[dict[str, Any]]) -> str:
    """``tests/test_equations.py`` for ``case_list`` (:func:`cases`): a test per equation, and a runner. The file holds
    each equation's id, formula and function; the example's inputs, output and tolerance (:func:`values_of`) are read
    when it runs from the quest's ``.fi/checks/`` (:func:`write_values`), never written into it."""
    public = [{k: v for k, v in c.items() if k not in _HIDDEN and k != "static"} for c in case_list]
    text = _TEMPLATE.format(cases=json.dumps(public, default=str), values_name=VALUES_NAME)
    text = text.replace("@@MEASURE@@", _measure_body())
    names = []
    for i, c in enumerate(case_list):
        name = "test_" + re.sub(r"[^a-z0-9]+", "_", str(c["id"]).lower()).strip("_")
        while name in names:
            name += "_"
        names.append(name)
        text += f"\n\ndef {name}():\n    {('equation ' + str(c['id']) + ': ' + str(c['formula']))!r}\n    _check({i})\n"
    text += '''

if __name__ == "__main__":
    only = []
    if "--only" in sys.argv:
        for arg in sys.argv[sys.argv.index("--only") + 1:]:
            if arg.startswith("--"):
                break
            only.append(arg)
    results = []
    for case in CASES:
        if only and case["id"] not in only:
            continue
        status, detail, value, tol, want = _run(case)
        results.append({"id": case["id"], "function": case["function"], "status": status, "detail": detail})
    if "--json" in sys.argv:
        print("''' + MARK + ''' " + json.dumps(results))
        sys.exit(0)
    for r in results:
        print("ok  " if r["status"] == "ok" else "FAIL", r["id"], r["function"], r["detail"])
    sys.exit(1 if any(r["status"] != "ok" for r in results) else 0)
'''
    return text


def _measure_body() -> str:
    """The text of :mod:`core.equation_measure` without its module docstring (what the test file and the command embed)."""
    source = Path(_measure.__file__).read_text(encoding="utf-8")
    end = source.index('"""', 3) + 3
    return source[end:].strip("\n") + "\n"


def measure_spec(case_list: list[dict[str, Any]], *, nonce: str, only: list[str] | None = None,
                 code: str = "code") -> dict[str, Any]:
    """What the process that calls the functions is given: per case the module, the function, the example's inputs and
    how the output is reported (one number, named outputs, or draws). Never the expected value, the tolerance or the
    variance, and no path to them."""
    picked = [c for c in case_list if not only or str(c["id"]) in only]
    return {"nonce": nonce, "code": code,
            "cases": [{"id": str(c["id"]), "module": c["module"], "function": c["function"],
                       "inputs": dict(c.get("inputs") or {}), "random": bool(c.get("random")),
                       "names": list(c["expected"]) if isinstance(c.get("expected"), dict) else None,
                       "outputs": list(c.get("outputs") or [])} for c in picked]}


def measure_argv(case_list: list[dict[str, Any]], *, nonce: str, only: list[str] | None = None,
                 code: str = "code") -> list[str]:
    """The arguments after ``python`` that measure ``case_list`` (:func:`measure_spec`): the measuring code itself
    (``-c``, FI's own text, nothing written into the quest) and the spec. Run from the folder that holds ``code``."""
    return ["-c", _measure_body() + "\nmain()\n", json.dumps(measure_spec(case_list, nonce=nonce, only=only, code=code),
                                                             default=str)]


def new_nonce() -> str:
    import secrets

    return secrets.token_hex(12)


#: What a failed test says when the measuring process printed no result (it stopped early, was ended, ran past its time
#: or printed more than one result line): a failure of the test, never "not tested".
NO_RESULT = ("the run that calls the function gave no result FI can take (it stopped early, was ended, ran past its "
             "time, or printed more than one result)")


def _returns_by_name(fn: Any) -> bool:
    """Whether FI's reading of ``fn`` (an ``ast`` function) finds it returning its outputs by name: a dict, or an object
    of a class (a namedtuple, a dataclass), directly or through a name assigned one."""
    import ast

    def named(node: Any) -> bool:
        if isinstance(node, (ast.Dict, ast.DictComp)):
            return True
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            return name in ("dict", "OrderedDict", "namedtuple") or name[:1].isupper()
        return False

    nodes, stack = [], list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))
    held = {t.id for n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and named(n.value)
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    return any(isinstance(n, ast.Return) and n.value is not None
               and (named(n.value) or (isinstance(n.value, ast.Name) and n.value.id in held)) for n in nodes)


def static_facts(source: str, function: str) -> dict[str, Any] | None:
    """FI's own reading of the top-level function ``function`` in ``source``, made in FI's process before anything
    runs: its parameters (``{"name", "default", "kind"}``), whether it takes ``*args`` / ``**kwargs``, and whether it
    returns its outputs by name. ``None`` when it cannot be read (no such function, a decorator that may change the
    signature, text that does not parse)."""
    import ast

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function),
              None)
    if fn is None or fn.decorator_list:
        return None
    a = fn.args
    positional = [*a.posonlyargs, *a.args]
    firsts = len(positional) - len(a.defaults)
    params = [{"name": p.arg, "default": i >= firsts, "kind": "positional_only" if p in a.posonlyargs else "positional"}
              for i, p in enumerate(positional)]
    params += [{"name": p.arg, "default": d is not None, "kind": "keyword_only"} for p, d in zip(a.kwonlyargs, a.kw_defaults)]
    return {"params": params, "varargs": a.vararg is not None, "varkw": a.kwarg is not None,
            "by_name": _returns_by_name(fn)}


def _contract_here(case: dict[str, Any]) -> str:
    """Why the function cannot be called with the example's inputs, by FI's own reading (:func:`static_facts`): the
    same rule the measuring process applies, decided here so nothing it prints can claim it. ``""`` when it can, or when
    FI could not read the function."""
    static = case.get("static")
    if not isinstance(static, dict):
        return ""
    names = {p["name"] for p in static["params"]}
    inputs = list(case.get("inputs") or {})
    if not static["varkw"]:
        missing = [n for n in inputs if n not in names]
        if missing:
            return "the function must take " + ", ".join(missing) + " as arguments, named so"
    if static["varargs"] or static["varkw"]:
        return ""
    given = set(inputs) | ((set(RNG_PARAMS) | {SEED_PARAM}) if case.get("random") else set())
    required = [p["name"] for p in static["params"] if not p["default"] and p["name"] not in given]
    if required:
        return ("the parameter " + ", ".join(required) + (" has" if len(required) == 1 else " have")
                + " no default and is not one of the example's inputs; give it a default or take it out of this function")
    return ""


def _judge_here(case: dict[str, Any], measured: Any) -> tuple[str, str, Any, Any, Any]:
    """:func:`equation_measure.judge`, with the statuses that mean "FI could not judge this" decided by FI alone: a
    contract from FI's own reading of the function, and outputs returned by name only when the formula defines several
    and FI's reading finds the function returning them by name. The measuring process saying either is otherwise a
    failure (``error``)."""
    want = case["expected"]
    contract = _contract_here(case)
    if contract:
        return CONTRACT, contract, None, None, want
    status = measured.get("status") if isinstance(measured, dict) else None
    if status == CONTRACT:
        return (ERROR, "the run that calls the function said it could not be called with the example's inputs, which "
                "FI's own reading of the function does not confirm", None, None, want)
    if status == OUTPUTS:
        static = case.get("static")
        if len(case.get("outputs") or []) >= 2 and isinstance(static, dict) and static.get("by_name"):
            return OUTPUTS, str(measured.get("detail") or ""), None, None, want
        return (ERROR, "the run that calls the function said it returns its outputs by name, which FI's own reading "
                "of the function does not confirm", None, None, want)
    return _measure.judge(case, measured)


def judged(case_list: list[dict[str, Any]], stdout: str, *, nonce: str,
           only: list[str] | None = None) -> list[dict[str, Any]]:
    """FI's verdict on each case, from what the measuring process printed (its one line with ``nonce``): ``{"id",
    "function", "status", "detail", "got", "expected", "tolerance"}`` per case, judged here, in FI's own process, with
    the expected value and the tolerance that process never had. No such line, or more than one, is a failure of every
    case (:data:`NO_RESULT`), never "not tested"."""
    found: list[Any] = []
    head = _measure.MEASURED_MARK + " " + nonce + " "
    for line in (stdout or "").splitlines():
        if line.startswith(head):
            try:
                found.append(json.loads(line[len(head):]))
            except ValueError:
                found.append(None)
    taken = found[0] if len(found) == 1 and isinstance(found[0], dict) else None
    out = []
    for c in case_list:
        if only and str(c["id"]) not in only:
            continue
        case = {**c, "names": list(c["expected"]) if isinstance(c.get("expected"), dict) else None}
        if taken is None:
            contract = _contract_here(case)
            status, detail, got, tol, want = (CONTRACT if contract else ERROR), (contract or NO_RESULT), None, None, c["expected"]
        else:
            status, detail, got, tol, want = _judge_here(case, taken.get(str(c["id"])))
        out.append({"id": str(c["id"]), "function": c["function"], "status": status, "detail": detail,
                    "got": got, "expected": want, "tolerance": tol})
    return out


async def run_and_judge(run: Any, case_list: list[dict[str, Any]], *, only: list[str] | None = None,
                        **run_kwargs: Any) -> tuple[list[dict[str, Any]], int, str]:
    """Measure ``case_list`` with ``run(argv, **run_kwargs) -> (returncode, stdout, stderr)`` (python's arguments, run
    from the quest folder or a copy of it) and judge it here: ``(results, returncode, stderr)``."""
    nonce = new_nonce()
    rc, out, err = await run(measure_argv(case_list, nonce=nonce, only=only), **run_kwargs)
    return judged(case_list, out, nonce=nonce, only=only), rc, err


def parse_results(stdout: str) -> list[dict[str, Any]] | None:
    """The results the test file printed (the last ``EQUATION_TESTS:`` line), or ``None`` when there is none."""
    found = None
    for line in (stdout or "").splitlines():
        if line.startswith(MARK):
            try:
                value = json.loads(line[len(MARK):])
            except ValueError:
                continue
            if isinstance(value, list):
                found = [r for r in value if isinstance(r, dict)]
    return found


def failure_message(result: dict[str, Any], formula: str) -> str:
    """What a repair is told about a failed equation test: the function and the equation, never the expected value (and
    never the worked example's inputs)."""
    fn, eid, status = result.get("function"), result.get("id"), result.get("status")
    head = f"The function `{fn}` implements equation {eid} of the model (`{formula}`)."
    if status == MISMATCH:
        return (head + " FI tested it against a worked example it computed itself from the plan's formula for this "
                "equation, and the function's result does not agree. Check the function against the equation term by "
                "term (factors, signs, which quantity is divided or multiplied by which) and make it compute exactly "
                "this equation." + (f" ({result.get('detail')})" if result.get("detail") else ""))
    if status == CONTRACT:
        return head + f" FI could not call it: {result.get('detail')}. The parameters are named as the equation's inputs."
    return head + f" Calling it with the equation's inputs raised: {result.get('detail')}"


def checks_dir(code_dir: Path) -> Path:
    """Where FI keeps what its tests in ``code_dir/tests/`` compare with: ``<quest>/.fi/checks/``, beside ``code/`` and
    outside it (where the test files look unless a person names another folder with ``--checks``)."""
    return Path(code_dir).resolve().parent / CHECKS_DIR


def values_of(case_list: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """What each equation's test compares with, by equation id: the example's inputs, output, tolerance and variance."""
    return {str(c["id"]): {k: c[k] for k in _HIDDEN if k in c} for c in case_list}


def write_values(code_dir: Path, case_list: list[dict[str, Any]]) -> Path | None:
    """Write :func:`values_of` into :func:`checks_dir` (removing it when there is no case); the file written."""
    path = checks_dir(code_dir) / VALUES_NAME
    if not case_list:
        path.unlink(missing_ok=True)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values_of(case_list), default=str, indent=1) + "\n", encoding="utf-8")
    return path


#: A test file of FI's that reads its values from outside ``code/`` holds this line; one without it is from an older FI
#: that wrote the expected values into the file itself.
_READS_OUTSIDE = "VALUES_NAME = "


def remove_stale(code_dir: Path) -> list[str]:
    """Remove what an older FI left in ``code/tests/`` with expected values in it: the value files
    (:data:`STALE_NAMES`) and FI's test files written before their values moved out of ``code/`` (FI writes them again
    in the current form). The paths removed, relative to ``code_dir``. Nothing else is touched."""
    tests = Path(code_dir) / TESTS_DIR
    removed: list[str] = []
    if not tests.is_dir():
        return removed
    for name in STALE_NAMES:
        path = tests / name
        if path.is_file():
            path.unlink(missing_ok=True)
            removed.append(f"{TESTS_DIR}/{name}")
    for name in (TEST_NAME, "test_oracles.py"):
        path = tests / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
        except OSError:
            continue
        if text is not None and "Frontier Insight" in text and _READS_OUTSIDE not in text:
            path.unlink(missing_ok=True)
            removed.append(f"{TESTS_DIR}/{name}")
    return removed


def failure_reason(result: dict[str, Any]) -> str:
    """Why one equation test failed, for the run log and the record (never for a request to the model: it holds the
    expected value): the expected value, the value got and the tolerance, or the error the call raised. ``result`` is a
    result of :func:`judged`."""
    status = result.get("status")
    got, want, tol = result.get("got"), result.get("expected"), result.get("tolerance")
    detail = _text(result.get("detail"))
    if status == MISMATCH and isinstance(got, dict) and isinstance(want, dict):
        tols = tol if isinstance(tol, dict) else {}
        parts = []
        for n, w in want.items():
            g, t = got.get(n), tols.get(n)
            if g is None:
                parts.append(f"{n}: got nothing usable, expected {w:.10g}")
            elif not (isinstance(t, (int, float)) and abs(g - w) <= t):
                parts.append(f"{n}: got {g:.10g}, expected {w:.10g}" + (f", tolerance {t:.3g}" if isinstance(t, (int, float)) else ""))
        return "; ".join(parts) + (f" ({detail})" if detail else "")
    if status == MISMATCH and got is not None and want is not None:
        text = f"got {got:.10g}, expected {want:.10g}" + (f", tolerance {tol:.3g}" if isinstance(tol, (int, float)) else "")
        return text + (f" ({detail})" if detail else "")
    return f"{status}: {detail}" if detail else str(status or "failed")


def write(code_dir: Path, case_list: list[dict[str, Any]]) -> Path | None:
    """Write ``tests/test_equations.py`` into ``code_dir`` (removing it when there is no case); the test file written.
    What it compares with is not on disk while the quest's code can still run: this removes the quest's
    ``.fi/checks/`` copy, FI judges with the values it holds (:func:`run_and_judge`), and the person's copy is written
    once the last run is over (:func:`write_values`, from ``code_layout.write_check_values``)."""
    path = Path(code_dir) / TEST_PATH
    (checks_dir(code_dir) / VALUES_NAME).unlink(missing_ok=True)
    if not case_list:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(test_source(case_list), encoding="utf-8")
    return path


def untested_core(protocol: dict[str, Any] | None) -> list[str]:
    """The ids of the equations the simulation generates its data with that nothing tests (an untestable equation with no
    other check), as the plan stands now."""
    return [r["id"] for r in example_rows(protocol) if r["state"] == "unprotected"]


def write_note(protocol: dict[str, Any] | None) -> str:
    """For the write prompt: the core equations nothing tests, to be said plainly in the limitations. ``""`` when none."""
    ids = untested_core(protocol)
    if not ids:
        return ""
    return ("The equation" + ("s " if len(ids) > 1 else " ") + ", ".join(ids) + (" that generate" if len(ids) > 1 else " that generates")
            + " the study's data " + ("are" if len(ids) > 1 else "is")
            + " not tested at all: the plan says it has no closed form for its output and gave no limit, symmetry, "
              "conservation or second implementation to check it with. Say so plainly in the limitations, and do not "
              "present the quantities that depend on it as verified.")
