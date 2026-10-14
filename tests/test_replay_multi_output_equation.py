"""Replay of a run that stopped on an equation whose formula defines two outputs (a kinetics parameter-recovery study).

The plan's equation E4 defined `Ea_hat` and `ln_A_hat`. Each time the plan was asked for the worked example it picked a
different statement of the formula (the fitted line, or `Ea_hat`, or `ln_A_hat`) as "the equation's output as one number",
the correct code returned a dict, FI's test did `float(value)` on it, and the last failure was a raw `TypeError` for a
parameter (`covariance`) the example does not give. Now the formula's own text says it has several outputs, the example
is a mapping from each output's name to its formula, the test reads each output from the return by name, and a
parameter with no default that the example does not give is a named contract failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import equation_tests as eqt

FORMULA = "ln(k_i) = alpha + beta*(1/T_i - mean(1/T)); Ea_hat=-R*beta; ln_A_hat=alpha+Ea_hat*mean(1/T)/R"
V13_EXAMPLE = {"inputs": {"alpha": 4.0, "beta": -1000.0, "R": 8.314462618, "mean_invT": 0.002},
               "expected_formula": "alpha+(-R*beta)*mean_invT/R"}
V10_EXAMPLE = {"inputs": {"alpha": 4.0, "beta": -1000.0, "T_i": 400.0, "mean_invT": 0.002},
               "expected_formula": "alpha+beta*(1/T_i-mean_invT)"}
PER_OUTPUT = {"inputs": V13_EXAMPLE["inputs"],
              "expected_formula": {"Ea_hat": "-R*beta", "ln_A_hat": "alpha+(-R*beta)*mean_invT/R"}}

# The function as the run's code wrote it (it is correct): a required `covariance`, and a dict of four outputs.
R12_FUNCTION = '''
import math
import numpy as np


# E4
def arrhenius_parameters(alpha: float, beta: float, covariance: np.ndarray,
                         mean_invT: float, R: float) -> dict:
    """Transform centred-line parameters to Arrhenius parameters."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError("covariance must be a 2 by 2 matrix for alpha and beta.")
    if not (np.isfinite(mean_invT) and np.isfinite(R) and R > 0.0):
        raise ValueError("mean_invT and positive R must be finite.")

    # Ea = -R beta and ln(A) = alpha + Ea mean(1/T) / R
    #                         = alpha - beta mean(1/T).
    ea_hat = -R * beta
    ln_a_hat = alpha - beta * mean_invT

    jacobian = np.array([[0.0, -R], [1.0, -mean_invT]])
    parameter_covariance = jacobian @ covariance @ jacobian.T
    denominator = math.sqrt(
        parameter_covariance[0, 0] * parameter_covariance[1, 1]
    )
    correlation = (
        float(parameter_covariance[0, 1] / denominator)
        if denominator > 0.0 else float("nan")
    )

    return {
        "Ea_hat": float(ea_hat),
        "ln_A_hat": float(ln_a_hat),
        "covariance": parameter_covariance,
        "correlation": correlation,
    }
'''
# With a default for the covariance, skipping that part when it is not given.
DEFAULTED = '''
def arrhenius_parameters(alpha, beta, mean_invT, R, covariance=None):
    ea_hat = -R * beta
    ln_a_hat = alpha - beta * mean_invT
    out = {"Ea_hat": float(ea_hat), "ln_A_hat": float(ln_a_hat)}
    if covariance is not None:
        out["covariance"] = covariance
    return out
'''
TUPLE = '''
def arrhenius_parameters(alpha, beta, mean_invT, R, covariance=None):
    return (-R * beta, alpha - beta * mean_invT)
'''
SIGN_ERROR = '''
def arrhenius_parameters(alpha, beta, mean_invT, R, covariance=None):
    return {"Ea_hat": -R * beta, "ln_A_hat": alpha + beta * mean_invT}
'''
NAMEDTUPLE = '''
from collections import namedtuple
Out = namedtuple("Out", "Ea_hat ln_A_hat")


def arrhenius_parameters(alpha, beta, mean_invT, R):
    return Out(-R * beta, alpha - beta * mean_invT)
'''


def _protocol(example: Any, formula: str = FORMULA, eid: str = "E4") -> dict[str, Any]:
    return {"model": {"summary": "m", "equations": [
        {"id": eid, "formula": formula, "role": "analyses", "source": "derivation", "example": example}]}}


def _run(tmp_path: Path, source: str, example: Any, formula: str = FORMULA) -> dict[str, Any]:
    rows = eqt.example_rows(_protocol(example, formula))
    case_list = eqt.cases(rows, {"E4": {"module": "m", "function": next(
        line.split("(")[0][4:] for line in source.splitlines() if line.startswith("def ")), "file": "m.py"}})
    assert case_list, rows
    (tmp_path / "m.py").write_text(source, encoding="utf-8")
    (tmp_path / "t.py").write_text(eqt.test_source(case_list), encoding="utf-8")
    done = subprocess.run([sys.executable, "-B", "t.py", "--json"], cwd=tmp_path, capture_output=True, text=True)
    results = eqt.parse_results(done.stdout)
    assert results, done.stderr
    return results[0]


def test_the_formula_text_says_it_has_two_outputs_and_a_single_string_example_is_unusable_in_plain_words() -> None:
    assert eqt.outputs(FORMULA) == ["Ea_hat", "ln_A_hat"] and eqt.is_multi_output(FORMULA)
    for example in (V13_EXAMPLE, V10_EXAMPLE):
        row = eqt.example_rows(_protocol(example))[0]
        assert row["state"] == "unusable"
        assert "Ea_hat" in row["why"] and "ln_A_hat" in row["why"] and "mapping" in row["why"]
        text = eqt.request([row])
        assert "Ea_hat, ln_A_hat" in text and "several named outputs" in text


def test_a_mapping_example_is_computed_per_output_and_must_cover_every_output() -> None:
    row = eqt.example_rows(_protocol(PER_OUTPUT))[0]
    assert row["state"] == "ok" and row["outputs"] == ["Ea_hat", "ln_A_hat"]
    assert row["expected"]["Ea_hat"] == pytest.approx(8314.462618) and row["expected"]["ln_A_hat"] == pytest.approx(6.0)
    assert set(row["abs"]) == {"Ea_hat", "ln_A_hat"} and all(v == 0.0 for v in row["abs"].values())
    fewer = {**PER_OUTPUT, "expected_formula": {"Ea_hat": "-R*beta"}}
    more = {**PER_OUTPUT, "expected_formula": {**PER_OUTPUT["expected_formula"], "correlation": "alpha"}}
    uses_unknown = {**PER_OUTPUT, "expected_formula": {"Ea_hat": "-R*beta", "ln_A_hat": "alpha+Q"}}
    for example, words in ((fewer, "no formula for ln_A_hat"), (more, "correlation is not an output"),
                           (uses_unknown, "uses Q")):
        bad = eqt.example_rows(_protocol(example))[0]
        assert bad["state"] == "unusable" and words in bad["why"], bad["why"]


def test_a_random_multi_output_equation_is_unusable_and_says_to_split_it() -> None:
    row = eqt.example_rows(_protocol({**PER_OUTPUT, "random": True}))[0]
    assert row["state"] == "unusable" and "split" in row["why"] and "Ea_hat, ln_A_hat" in row["why"]


def test_the_runs_function_with_a_required_covariance_is_a_contract_failure_naming_it_not_a_raw_error(tmp_path: Path) -> None:
    result = _run(tmp_path, R12_FUNCTION, PER_OUTPUT)
    assert result["status"] == "contract" and "covariance" in result["detail"] and "no default" in result["detail"]
    assert "TypeError" not in result["detail"]
    assert eqt.repairable_by_example(result)


def test_a_function_that_gives_the_covariance_a_default_passes(tmp_path: Path) -> None:
    assert _run(tmp_path, DEFAULTED, PER_OUTPUT)["status"] == "ok"
    assert _run(tmp_path, NAMEDTUPLE, PER_OUTPUT)["status"] == "ok"  # an attribute counts as a named output


def test_a_function_returning_a_tuple_is_told_to_return_the_outputs_by_name(tmp_path: Path) -> None:
    result = _run(tmp_path, TUPLE, PER_OUTPUT)
    assert result["status"] == "contract" and "Ea_hat" in result["detail"] and "ln_A_hat" in result["detail"]


def test_a_sign_error_in_one_output_is_a_mismatch_naming_that_output_and_never_the_expected_value(tmp_path: Path) -> None:
    result = _run(tmp_path, SIGN_ERROR, PER_OUTPUT)
    assert result["status"] == "mismatch" and "ln_A_hat" in result["detail"] and "Ea_hat" not in result["detail"]
    for number in ("6.0", "8314.46", "6.0000"):
        assert number not in result["detail"]
    message = eqt.failure_message(result, FORMULA)
    assert "ln_A_hat" in message and "8314" not in message


def test_the_numbers_of_a_multi_output_failure_are_per_output_for_the_log(tmp_path: Path) -> None:
    rows = eqt.example_rows(_protocol(PER_OUTPUT))
    case_list = eqt.cases(rows, {"E4": {"module": "m", "function": "arrhenius_parameters", "file": "m.py"}})
    (tmp_path / "m.py").write_text(SIGN_ERROR, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text(eqt.test_source(case_list), encoding="utf-8")
    done = subprocess.run([sys.executable, "-B", "tests/t.py", "--json"], cwd=tmp_path, capture_output=True, text=True)
    result = eqt.parse_results(done.stdout)[0]
    numbers = eqt.load_numbers(tmp_path)["E4"]
    reason = eqt.failure_reason({**result, **numbers})
    assert reason.startswith("ln_A_hat: got 2, expected 6"), reason
    assert "Ea_hat" not in reason  # only the output that disagrees


def test_single_output_equations_behave_as_before(tmp_path: Path) -> None:
    for formula in ("c = c0*exp(-k*t)", "c0*exp(-k*t)", "c(t) = c0*exp(-k*t)", "c = c0*exp(-k*t) where k = 0.5",
                    "c == c0", "c <= c0"):
        assert not eqt.is_multi_output(formula), formula
    example = {"inputs": {"c0": 2.0, "k": 0.5, "t": 1.0}, "expected_formula": "c0*exp(-k*t)"}
    row = eqt.example_rows(_protocol(example, "c = c0*exp(-k*t)"))[0]
    assert row["state"] == "ok" and row["outputs"] == [] and isinstance(row["expected"], float)
    good = "import math\ndef c_of(c0, k, t):\n    return c0*math.exp(-k*t)\n"
    wrong = "import math\ndef c_of(c0, k, t):\n    return c0*math.exp(k*t)\n"
    assert _run(tmp_path, good, example, "c = c0*exp(-k*t)")["status"] == "ok"
    assert _run(tmp_path, wrong, example, "c = c0*exp(-k*t)")["status"] == "mismatch"
    # a formula with no `=` at all
    row = eqt.example_rows(_protocol(example, "c0*exp(-k*t)"))[0]
    assert row["state"] == "ok" and row["outputs"] == []
    assert _run(tmp_path, good, example, "c0*exp(-k*t)")["status"] == "ok"
    # a single-output example given as a mapping stays unusable, as before
    assert eqt.example_rows(_protocol({**example, "expected_formula": {"c": "c0"}}, "c = c0"))[0]["state"] == "unusable"


def test_a_required_parameter_the_example_does_not_give_is_the_same_rule_at_outline_time_and_in_the_test(tmp_path: Path) -> None:
    example = {"inputs": {"c0": 2.0, "k": 0.5}, "expected_formula": "c0*k"}
    rows = eqt.example_rows(_protocol(example, "y = c0*k"))
    cases = [
        ("def f(c0, k): return c0*k\n", [], ["c0", "k"], ["c0", "k"]),
        ("def f(c0, k, extra): return c0*k\n", [], ["c0", "k", "extra"], ["c0", "k", "extra"]),
        ("def f(c0, k, extra=1): return c0*k\n", [], ["c0", "k", "extra"], ["c0", "k"]),
        ("def f(c0, k, *, flag): return c0*k\n", [], ["c0", "k", "flag"], ["c0", "k", "flag"]),
        ("def f(c0, k, **kw): return c0*k\n", [], ["c0", "k", "**kw"], ["c0", "k"]),
        ("def f(c0, k, *a): return c0*k\n", [], ["c0", "k", "*a"], ["c0", "k"]),
    ]
    for source, _unused, names, required in cases:
        planned = bool(eqt.parameter_problems(rows, {"E4": (names, required)}))
        (tmp_path / "m.py").write_text(source, encoding="utf-8")
        case_list = eqt.cases(rows, {"E4": {"module": "m", "function": "f", "file": "m.py"}})
        (tmp_path / "t.py").write_text(eqt.test_source(case_list), encoding="utf-8")
        done = subprocess.run([sys.executable, "-B", "t.py", "--json"], cwd=tmp_path, capture_output=True, text=True)
        status = eqt.parse_results(done.stdout)[0]["status"]
        assert planned == (status == "contract"), (source, planned, status)


@pytest.mark.parametrize("formula,expected", [
    ("a = x + 1; b = x - 1", ["a", "b"]),
    ("a = x + 1\nb = x - 1", ["a", "b"]),
    ("a = x + 1; a = x + 2", ["a"]),
    ("f(a; b) = x", []),
    ("y = x; where z = 3", ["y"]),
    ("y = a if b == c else d", ["y"]),
    ("a <= b; c >= d; e != f", []),
    ("p(t) = a; q(t) = b", []),
    # a term defined after the statement that uses it is not an output (r3d duv); one used later still is (r12 E4)
    ("gamma = 2*J1(u)/u; u = 2*pi*r*NA/lam; gamma(0)=1", ["gamma"]),
    ("Ea_hat=-R*beta; ln_A_hat=alpha+Ea_hat*m/R", ["Ea_hat", "ln_A_hat"]),
    ("y = x_u + 1; u = 2", ["y", "u"]),
    ("", []),
])
def test_the_detector_is_conservative(formula: str, expected: list[str]) -> None:
    assert eqt.outputs(formula) == expected
