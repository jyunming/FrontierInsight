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

**The tolerance is a rule, not a guess.** A closed form agrees within a relative 1e-6 (and, for an output that is exactly
0, an absolute 1e-12: the size of floating-point round-off). When the plan states a numerical method for the equation
(``"method"``), the plan's own numerical setting decides (``"tolerance"`` and ``"tolerance_mode"``, required with a
method); a stated tolerance without a method can only make the test tighter. FI never invents a looser one.

A failed test is reported by its function and equation only (:func:`failure_message`): the expected value is never put
into a request to the model that writes code.
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
TEST_NAME = "test_equations.py"
TESTS_DIR = "tests"
TEST_PATH = f"{TESTS_DIR}/{TEST_NAME}"
#: The line the test file prints (with ``--json``), one JSON list of ``{"id", "function", "status", "detail"}``.
MARK = "EQUATION_TESTS:"

OK, MISMATCH, CONTRACT, ERROR = "ok", "mismatch", "contract", "error"

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
    "instead; FI then says plainly that it is not tested before the run.")


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


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


def example_rows(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per equation that should have a worked example: ``{"id", "formula", "state", "why", "inputs",
    "expected", "rel", "abs", "method"}``. ``state``: ``ok`` (FI computed the output), ``missing`` (no example),
    ``unusable`` (``why``: the example cannot be used, never repaired by guessing) or ``untestable`` (the plan says
    the equation has no closed form for its output, and why)."""
    rows: list[dict[str, Any]] = []
    for eq in eligible(protocol):
        row = {"id": _text(eq["id"]), "formula": _text(eq["formula"]), "state": "missing",
               "why": "the plan gave no worked example", "inputs": {}, "expected": None,
               "rel": RELATIVE, "abs": ABSOLUTE_FLOOR, "method": ""}
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
            continue
        inputs = example.get("inputs")
        good = {str(k).strip(): _number(v) for k, v in inputs.items()} if isinstance(inputs, dict) else {}
        if not good or any(v is None or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k) for k, v in good.items()):
            row.update(state="unusable", why="`inputs` must be a mapping of names (that are valid parameter names) to numbers")
            continue
        formula = example.get("expected_formula")
        if formula is None or not _text(formula):
            row.update(state="unusable", why="it gave inputs but no `expected_formula`")
            continue
        value, why, unset, _used = _forms._formula_result({"expected_formula": formula, "case": good}, None)
        if value is None:
            row.update(state="unusable", why=(
                f"its `expected_formula` uses {', '.join(unset)}, which are not among its `inputs`" if unset
                else f"its `expected_formula` cannot be computed ({why})"))
            continue
        rel, abs_tol, method, problem = _tolerance(example)
        if problem:
            row.update(state="unusable", why=problem)
            continue
        row.update(state="ok", why="", inputs={k: float(v) for k, v in good.items()}, expected=float(value), rel=rel,
                   abs=abs_tol, method=method)
    return rows


def request(rows: list[dict[str, Any]], *, last: bool = True) -> str:
    """The part of the one request to the plan that asks for the worked examples a plan lacks (``""`` when none)."""
    lines = []
    for r in rows:
        if r["state"] == "missing":
            lines.append(f"- {r['id']} (`{r['formula']}`): give its `example`.")
        elif r["state"] == "unusable":
            lines.append(f"- {r['id']} (`{r['formula']}`): its `example` cannot be used: {r['why']}.")
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
        elif r["state"] == "untestable":
            notes.append(f"equation {r['id']} is not tested before the run: the plan says it has no closed form "
                         f"for its output ({r['why']})")
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
                        "method": r["method"]})
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
    text = _TEMPLATE.format(cases=json.dumps(case_list, default=str))
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
                "this equation.")
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
