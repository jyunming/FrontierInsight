"""A small test per equation of the plan's model. Written by Frontier Insight (never by the model that wrote the code)
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

CASES = json.loads('[{"id": "E1", "formula": "Z_i = effect_size*is_signal + epsilon_i, where epsilon_i~N(0,1).", "module": "fdr_bonferroni_vs_bh.model", "function": "draw_z", "inputs": {"effect_size": 2.5, "is_signal": 1.0}, "expected": 2.5, "rel": 1e-06, "abs": 0.0, "method": "", "random": true, "variance": 1.0}, {"id": "E3", "formula": "Reject under Bonferroni when p_i <= alpha/m.", "module": "fdr_bonferroni_vs_bh.model", "function": "bonferroni_threshold", "inputs": {"alpha": 0.05, "m": 20.0}, "expected": 0.0025, "rel": 1e-06, "abs": 0.0, "method": "", "random": false, "variance": null}, {"id": "E4", "formula": "For ordered p-values p_(1)<=...<=p_(m), BH selects k=max{j:p_(j)<=alpha*j/m} and rejects p_(1),...,p_(k).", "module": "fdr_bonferroni_vs_bh.model", "function": "bh_critical_value", "inputs": {"alpha": 0.05, "k": 4.0, "m": 20.0}, "expected": 0.01, "rel": 1e-06, "abs": 0.0, "method": "", "random": false, "variance": null}, {"id": "E5", "formula": "FDR_BH = pi0*alpha under independent tests, with pi0=1-pi1.", "module": "fdr_bonferroni_vs_bh.model", "function": "bh_fdr_reference", "inputs": {"pi1": 0.2, "alpha": 0.05}, "expected": 0.04000000000000001, "rel": 1e-06, "abs": 0.0, "method": "", "random": false, "variance": null}]')
DRAWS, MIN_DRAWS, TIME_LIMIT_S, SIGMAS = 1000, 100, 60.0, 5.0
RNG_PARAMS = ('rng', 'random_state', 'generator')
NUMBERS_NAME = 'equation_numbers.json'


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
            return "error", (type(e).__name__ + ": " + str(e))[:300], None, None
        try:
            draw = float(result)
        except (TypeError, ValueError):
            return "mismatch", "it returned something that is not one number", None, None
        if not math.isfinite(draw):
            return "mismatch", "a draw was not a finite number", None, None
        values.append(draw)
        if len(values) >= MIN_DRAWS and time.monotonic() - started > TIME_LIMIT_S:
            break
    n = len(values)
    if n < MIN_DRAWS:
        return "error", "the function is too slow to be drawn from " + str(MIN_DRAWS) + " times in the time allowed", None, None
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    want = case["expected"]
    if max(values) == min(values):
        tol = max(case["rel"] * abs(want), case["abs"])
        if abs(mean - want) > tol:
            return "mismatch", "", mean, tol
        return "ok", "", mean, tol
    variance = case.get("variance")
    sigma = math.sqrt(variance) if variance else math.sqrt(m2 * n / (n - 1))
    tol = max(SIGMAS * sigma / math.sqrt(n), case["rel"] * abs(want), case["abs"])
    if abs(mean - want) > tol:
        return "mismatch", "the mean of " + str(n) + " draws with fixed seeds is further than " + str(int(SIGMAS)) + " standard errors from the distribution's mean the plan states", mean, tol
    if variance:
        m4 = sum((v - mean) ** 4 for v in values) / n
        spread = math.sqrt(max(m4 - m2 * m2, 0.0) / n)
        if abs(m2 * n / (n - 1) - variance) > SIGMAS * spread:
            return "mismatch", "the variance of " + str(n) + " draws with fixed seeds is further than " + str(int(SIGMAS)) + " standard errors from the variance the plan states", m2 * n / (n - 1), SIGMAS * spread, variance
    return "ok", "", mean, tol


def _run(case):
    """(status, detail, value, tolerance, expected): status ok | mismatch | contract | error. The detail never holds the
    expected value (it goes to the model that repairs the function); the last three are for the record and the log."""
    want = case["expected"]
    try:
        fn = getattr(importlib.import_module(case["module"]), case["function"])
    except Exception as e:  # noqa: BLE001
        return "error", (type(e).__name__ + ": " + str(e))[:300], None, None, want
    params = inspect.signature(fn).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        missing = [n for n in case["inputs"] if n not in params]
        if missing:
            return "contract", "the function must take " + ", ".join(missing) + " as arguments, named so", None, None, want
    if case.get("random"):
        out = _run_random(case, fn, params)
        return (*out, want) if len(out) == 4 else out
    try:
        value = fn(**case["inputs"])
    except Exception as e:  # noqa: BLE001
        return "error", (type(e).__name__ + ": " + str(e))[:300], None, None, want
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "mismatch", "it returned something that is not one number", None, None, want
    tol = max(case["rel"] * abs(want), case["abs"])
    if not math.isfinite(value) or abs(value - want) > tol:
        return "mismatch", "", value, tol, want
    return "ok", "", value, tol, want


def _check(index):
    case = CASES[index]
    status, detail, value, tol, want = _run(case)
    assert status == "ok", (case["id"] + " (" + case["function"] + "): " + status + " " + detail
                            + (" got " + repr(value) + ", the worked example gives " + repr(want)
                               + (", the tolerance is " + repr(tol) if tol is not None else "")
                               if value is not None else ""))


def test_e1():
    'equation E1: Z_i = effect_size*is_signal + epsilon_i, where epsilon_i~N(0,1).'
    _check(0)


def test_e3():
    'equation E3: Reject under Bonferroni when p_i <= alpha/m.'
    _check(1)


def test_e4():
    'equation E4: For ordered p-values p_(1)<=...<=p_(m), BH selects k=max{j:p_(j)<=alpha*j/m} and rejects p_(1),...,p_(k).'
    _check(2)


def test_e5():
    'equation E5: FDR_BH = pi0*alpha under independent tests, with pi0=1-pi1.'
    _check(3)


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
    results, numbers = [], {}
    for case in CASES:
        if only and case["id"] not in only:
            continue
        status, detail, value, tol, want = _run(case)
        results.append({"id": case["id"], "function": case["function"], "status": status, "detail": detail})
        numbers[case["id"]] = {"got": value, "expected": want, "tolerance": tol}
    if "--json" in sys.argv:
        # The numbers (value got, expected value, tolerance) are for the run log, in a file of their own: this line is
        # read by code that may report to the model that writes the function, and it never carries the expected value.
        try:
            Path(__file__).with_name(NUMBERS_NAME).write_text(json.dumps(numbers), encoding="utf-8")
        except OSError:
            pass
        print("EQUATION_TESTS: " + json.dumps(results))
        sys.exit(0)
    for r in results:
        print("ok  " if r["status"] == "ok" else "FAIL", r["id"], r["function"], r["detail"])
    sys.exit(1 if any(r["status"] != "ok" for r in results) else 0)
