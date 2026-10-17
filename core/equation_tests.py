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

**An equation whose output is a random draw** (``"example": {"random": true, ...}``, or a formula that names a
distribution) is tested on the distribution, not on one call: ``expected_formula`` is the distribution's MEAN for those
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

from . import oracle_check as _oracle
from . import oracle_forms as _forms

#: The equations the code computes as functions (the plan's ``role``): the ones that get a worked example.
ROLES = ("generates", "analyses")
#: A closed form agrees within this relative tolerance; an output of exactly 0, within ``ABSOLUTE_FLOOR``.
RELATIVE = 1e-6
ABSOLUTE_FLOOR = 1e-12
#: A random equation: how many fixed-seed draws, the fewest that still count, the most time they may take, and how many
#: standard errors the draws' mean (and variance) may be from the stated one.
DRAWS = 1000
MIN_DRAWS = 100
TIME_LIMIT_S = 60.0
SIGMAS = 5.0
#: The parameters of the function that FI supplies to a random equation (never taken from the example's inputs).
RNG_PARAMS = ("rng", "random_state", "generator")
SEED_PARAM = "seed"
TEST_NAME = "test_equations.py"
TESTS_DIR = "tests"
TEST_PATH = f"{TESTS_DIR}/{TEST_NAME}"
#: The line the test file prints (with ``--json``), one JSON list of ``{"id", "function", "status", "detail"}``.
MARK = "EQUATION_TESTS:"

OK, MISMATCH, CONTRACT, ERROR = "ok", "mismatch", "contract", "error"
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
    "times with fixed seeds and compares the mean (and variance) of the draws with yours, within five standard errors.")


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


#: A formula that names a distribution or says its value is drawn: the equation's output is then a random draw.
_DRAW_WORDS = re.compile(
    r"\b(?:poisson|binomial|bernoulli|gaussian|normal|uniform|exponential|gamma|beta|lognormal|geometric|weibull|"
    r"multinomial)\s*\(|\b(?:drawn|sampled) from\b|\brandom (?:draw|sample|variate|variable)\b", re.IGNORECASE)


def is_random(equation: dict[str, Any]) -> bool:
    """Whether the equation's output is a random draw: the plan marked its example ``"random": true``, or the equation's
    formula names a distribution (``Poisson(...)``) or says its value is drawn. Conservative on purpose: a deterministic
    function that is taken for a random one is still held to the strict rule (see the module text), so nothing is loosened."""
    example = equation.get("example")
    if isinstance(example, dict) and example.get("random") not in (None, False, "", 0, "false", "False"):
        return True
    return bool(_DRAW_WORDS.search(_text(equation.get("formula"))))


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
        row = {"id": _text(eq["id"]), "formula": _text(eq["formula"]), "state": "missing",
               "why": "the plan gave no worked example", "inputs": {}, "expected": None,
               "rel": RELATIVE, "abs": ABSOLUTE_FLOOR, "method": "", "random": is_random(eq), "variance": None}
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
        if formula is None or not _text(formula):
            row.update(state="unusable", why=(
                "its output is a random draw, and it gave inputs but no `expected_formula` (the distribution's mean)"
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
            row["variance"] = _stated_variance(example, good)
    return rows


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


def locate(protocol: dict[str, Any] | None, sources: dict[str, str]) -> dict[str, dict[str, str] | None]:
    """For each equation that gets an example: the importable top-level function labelled ``# E<n>`` (a comment on or
    above it, or in its docstring), as ``{"module", "function", "file"}``; ``None`` when there is none. ``sources``:
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
                    found = {"module": module_of(rel), "function": fn, "file": rel}
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


def cases(rows: list[dict[str, Any]], located: dict[str, dict[str, str] | None]) -> list[dict[str, Any]]:
    """The tests to write: one per equation whose example is ``ok`` and whose function is found."""
    out = []
    for r in rows:
        where = located.get(r["id"])
        if r["state"] == "ok" and where:
            out.append({"id": r["id"], "formula": r["formula"], "module": where["module"], "function": where["function"],
                        "inputs": r["inputs"], "expected": r["expected"], "rel": r["rel"], "abs": r["abs"],
                        "method": r["method"], "random": bool(r.get("random")), "variance": r.get("variance")})
    return out


# --- the test file FI writes -------------------------------------------------------------------------------------------

_TEMPLATE = '''"""A small test per equation of the plan's model. Written by Frontier Insight (never by the model that wrote the code)
from the plan's own worked example: the inputs are the plan's, and the output is computed by FI from the plan's formula.

    python -m pytest tests/test_equations.py        (or: python tests/test_equations.py)

Each test imports the function labelled with the equation (`# E1`) and calls it with the example's inputs, by name.
"""
import importlib
import inspect
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES = json.loads({cases!r})
DRAWS, MIN_DRAWS, TIME_LIMIT_S, SIGMAS = {draws!r}, {min_draws!r}, {limit!r}, {sigmas!r}
RNG_PARAMS = {rng_params!r}


def _run_random(case, fn, params):
    """A random equation: DRAWS calls with fixed seeds; the mean of the draws is held to SIGMAS standard errors of the
    stated mean (and the variance, when it is stated, to SIGMAS standard errors of a variance). Draws that do not vary at
    all are a deterministic output: the strict rule applies to it, as for any other equation."""
    import random
    import time

    random.seed(0)
    try:
        import numpy
        numpy.random.seed(0)
        rng = numpy.random.default_rng(12345)
    except Exception:  # noqa: BLE001
        rng = random.Random(12345)
    rng_name = next((n for n in RNG_PARAMS if n in params), None)
    values = []
    started = time.monotonic()
    for i in range(DRAWS):
        kwargs = dict(case["inputs"])
        if rng_name:
            kwargs[rng_name] = rng
        if "seed" in params and "seed" not in kwargs:
            kwargs["seed"] = i
        try:
            result = fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            return "error", (type(e).__name__ + ": " + str(e))[:300], None
        try:
            draw = float(result)
        except (TypeError, ValueError):
            return "mismatch", "it returned something that is not one number", None
        if not math.isfinite(draw):
            return "mismatch", "a draw was not a finite number", None
        values.append(draw)
        if len(values) >= MIN_DRAWS and time.monotonic() - started > TIME_LIMIT_S:
            break
    n = len(values)
    if n < MIN_DRAWS:
        return "error", "the function is too slow to be drawn from " + str(MIN_DRAWS) + " times in the time allowed", None
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    want = case["expected"]
    if max(values) == min(values):
        if abs(mean - want) > max(case["rel"] * abs(want), case["abs"]):
            return "mismatch", "", mean
        return "ok", "", mean
    variance = case.get("variance")
    sigma = math.sqrt(variance) if variance else math.sqrt(m2 * n / (n - 1))
    if abs(mean - want) > max(SIGMAS * sigma / math.sqrt(n), case["rel"] * abs(want), case["abs"]):
        return "mismatch", "the mean of " + str(n) + " draws with fixed seeds is further than " + str(int(SIGMAS)) + " standard errors from the distribution's mean the plan states", mean
    if variance:
        m4 = sum((v - mean) ** 4 for v in values) / n
        spread = math.sqrt(max(m4 - m2 * m2, 0.0) / n)
        if abs(m2 * n / (n - 1) - variance) > SIGMAS * spread:
            return "mismatch", "the variance of " + str(n) + " draws with fixed seeds is further than " + str(int(SIGMAS)) + " standard errors from the variance the plan states", mean
    return "ok", "", mean


def _run(case):
    """(status, detail, value): status ok | mismatch | contract | error. The detail never holds the expected value."""
    try:
        fn = getattr(importlib.import_module(case["module"]), case["function"])
    except Exception as e:  # noqa: BLE001
        return "error", (type(e).__name__ + ": " + str(e))[:300], None
    params = inspect.signature(fn).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        missing = [n for n in case["inputs"] if n not in params]
        if missing:
            return "contract", "the function must take " + ", ".join(missing) + " as arguments, named so", None
    if case.get("random"):
        return _run_random(case, fn, params)
    try:
        value = fn(**case["inputs"])
    except Exception as e:  # noqa: BLE001
        return "error", (type(e).__name__ + ": " + str(e))[:300], None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "mismatch", "it returned something that is not one number", None
    want = case["expected"]
    if not math.isfinite(value) or abs(value - want) > max(case["rel"] * abs(want), case["abs"]):
        return "mismatch", "", value
    return "ok", "", value


def _check(index):
    case = CASES[index]
    status, detail, value = _run(case)
    assert status == "ok", (case["id"] + " (" + case["function"] + "): " + status + " " + detail
                            + (" got " + repr(value) + ", the worked example gives " + repr(case["expected"])
                               if value is not None else ""))
'''


def test_source(case_list: list[dict[str, Any]]) -> str:
    """``tests/test_equations.py`` for ``case_list`` (:func:`cases`): a test per equation, and a runner."""
    text = _TEMPLATE.format(cases=json.dumps(case_list, default=str), draws=DRAWS, min_draws=MIN_DRAWS,
                            limit=TIME_LIMIT_S, sigmas=SIGMAS, rng_params=RNG_PARAMS)
    names = []
    for i, c in enumerate(case_list):
        name = "test_" + re.sub(r"[^a-z0-9]+", "_", str(c["id"]).lower()).strip("_")
        while name in names:
            name += "_"
        names.append(name)
        text += f"\n\ndef {name}():\n    {('equation ' + str(c['id']) + ': ' + str(c['formula']))!r}\n    _check({i})\n"
    text += '''

if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
    results = []
    for case in CASES:
        if only and case["id"] not in only:
            continue
        status, detail, _value = _run(case)
        results.append({"id": case["id"], "function": case["function"], "status": status, "detail": detail})
    if "--json" in sys.argv:
        print("''' + MARK + ''' " + json.dumps(results))
        sys.exit(0)
    for r in results:
        print("ok  " if r["status"] == "ok" else "FAIL", r["id"], r["function"], r["detail"])
    sys.exit(1 if any(r["status"] != "ok" for r in results) else 0)
'''
    return text


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


def write(code_dir: Path, case_list: list[dict[str, Any]]) -> Path | None:
    """Write ``tests/test_equations.py`` into ``code_dir`` (removing it when there is no case); the file written."""
    path = Path(code_dir) / TEST_PATH
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
