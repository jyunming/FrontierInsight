"""Measure an equation's function in one process, judge it in another (core/equation_tests.py). measure() runs where the
quest's code is imported: it gets the function, its inputs and how to report the output, and returns what the function
computed, never compared with anything. judge() runs in FI's own process: it alone holds the expected value and the
tolerance. Standard library only and no annotations: this text is also run whole as `python -c` and embedded in
code/tests/test_equations.py, a person's copy of the same test."""
import importlib
import inspect
import json
import math
import os
import sys
import time

DRAWS, MIN_DRAWS, TIME_LIMIT_S, SIGMAS = 1000, 100, 60.0, 5.0
RNG_PARAMS = ("rng", "random_state", "generator")
MEASURED_MARK = "EQUATION_MEASURED:"


def _err(e):
    return (type(e).__name__ + ": " + str(e))[:300]


def _unsupplied(params, inputs, random):
    """The parameters without a default that the example gives no value."""
    if any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in params.values()):
        return []
    given = set(inputs) | (set(RNG_PARAMS) | {"seed"} if random else set())
    return [n for n, p in params.items() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and n not in given]


def _by_name(result):
    """Whether the function returned its outputs by name: a mapping, a namedtuple or a dataclass."""
    return (hasattr(result, "keys") and hasattr(result, "__getitem__")) or (
        isinstance(result, tuple) and hasattr(result, "_fields")) or (
        hasattr(type(result), "__dataclass_fields__") and not isinstance(result, type))


def _several(value):
    """Whether ``value`` holds several items (a tuple, a list, a mapping, an array of more than one number)."""
    try:
        return len(value) >= 2
    except BaseException:  # noqa: BLE001
        return False


def _output(result, name):
    """The output called ``name`` of what the function returned by name; LookupError when there is none."""
    if not _by_name(result):
        raise LookupError(name)
    try:
        if hasattr(result, "keys"):
            return result[name]
        fields = result._fields if isinstance(result, tuple) else type(result).__dataclass_fields__
        if name not in fields:
            raise LookupError(name)
        return getattr(result, name)
    except Exception:  # noqa: BLE001
        raise LookupError(name)


def _draws(fn, inputs, params):
    import random

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
        kwargs = dict(inputs)
        if rng_name:
            kwargs[rng_name] = rng
        if "seed" in params and "seed" not in kwargs:
            kwargs["seed"] = i
        try:
            result = fn(**kwargs)
        except BaseException as e:  # noqa: BLE001
            return {"status": "error", "detail": _err(e)}
        try:
            draw = float(result)
        except BaseException:  # noqa: BLE001
            return {"status": "not_number", "detail": "it returned something that is not one number"}
        if not math.isfinite(draw):
            return {"status": "nonfinite", "detail": "a draw was not a finite number"}
        values.append(draw)
        if len(values) >= MIN_DRAWS and time.monotonic() - started > TIME_LIMIT_S:
            break
    if len(values) < MIN_DRAWS:
        return {"status": "error",
                "detail": "the function is too slow to be drawn from " + str(MIN_DRAWS) + " times in the time allowed"}
    return {"status": "measured", "draws": values}


def measure(case):
    """What the function computes on the case's inputs: {"status": "measured", "value" | "values" + "absent" | "draws"},
    else {"status": "error" | "contract" | "outputs" | "not_number" | "nonfinite", "detail"}. ``case`` holds id, module,
    function, inputs, random, names (the outputs of a formula that defines several) and outputs; nothing to compare with."""
    try:
        fn = getattr(importlib.import_module(case["module"]), case["function"])
    except BaseException as e:  # noqa: BLE001
        return {"status": "error", "detail": _err(e)}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError) as e:
        return {"status": "error", "detail": _err(e)}
    inputs = dict(case.get("inputs") or {})
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        missing = [n for n in inputs if n not in params]
        if missing:
            return {"status": "contract", "detail": "the function must take " + ", ".join(missing) + " as arguments, named so"}
    required = _unsupplied(params, inputs, case.get("random"))
    if required:
        return {"status": "contract", "detail": "the parameter " + ", ".join(required) + (" has" if len(required) == 1 else " have")
                + " no default and is not one of the example's inputs; give it a default or take it out of this function"}
    names = case.get("names")
    if names:
        try:
            result = fn(**inputs)
        except BaseException as e:  # noqa: BLE001
            return {"status": "error", "detail": _err(e)}
        values, absent = {}, []
        for n in names:
            try:
                got = _output(result, n)
            except LookupError:
                absent.append(n)
                continue
            try:
                values[n] = float(got)
            except BaseException:  # noqa: BLE001 -- an array of several numbers, a method, a tensor: not one number
                values[n] = None
        return {"status": "measured", "values": values, "absent": absent}
    if case.get("random"):
        return _draws(fn, inputs, params)
    try:
        value = fn(**inputs)
    except BaseException as e:  # noqa: BLE001
        return {"status": "error", "detail": _err(e)}
    try:
        return {"status": "measured", "value": float(value)}
    except BaseException:  # noqa: BLE001
        if len(case.get("outputs") or []) >= 2 and _by_name(value):
            return {"status": "outputs", "detail": "the function returns its outputs by name and the worked example gives one value"}
        return {"status": "not_number", "detail": "it returned several values, not one number" if _several(value)
                else "it returned something that is not one number"}


def _number(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _judge_draws(case, draws, want):
    values = [_number(v) for v in draws] if isinstance(draws, list) else []
    if not values or any(v is None or not math.isfinite(v) for v in values):
        return "error", "the draws could not be read", None, None, want
    n = len(values)
    if n < MIN_DRAWS:
        return "error", "the function is too slow to be drawn from " + str(MIN_DRAWS) + " times in the time allowed", None, None, want
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if max(values) == min(values):
        # Draws that do not vary at all are a deterministic output: the strict rule applies.
        tol = max(case["rel"] * abs(want), case["abs"])
        return ("mismatch" if abs(mean - want) > tol else "ok"), "", mean, tol, want
    variance = case.get("variance")
    sigma = math.sqrt(variance) if variance else math.sqrt(m2 * n / (n - 1))
    tol = max(SIGMAS * sigma / math.sqrt(n), case["rel"] * abs(want), case["abs"])
    if abs(mean - want) > tol:
        return ("mismatch", "the mean of " + str(n) + " draws with fixed seeds is further than " + str(int(SIGMAS))
                + " standard errors from the distribution's mean the plan states", mean, tol, want)
    if variance:
        m4 = sum((v - mean) ** 4 for v in values) / n
        spread = math.sqrt(max(m4 - m2 * m2, 0.0) / n)
        if abs(m2 * n / (n - 1) - variance) > SIGMAS * spread:
            return ("mismatch", "the variance of " + str(n) + " draws with fixed seeds is further than " + str(int(SIGMAS))
                    + " standard errors from the variance the plan states", m2 * n / (n - 1), SIGMAS * spread, variance)
    return "ok", "", mean, tol, want


def _judge_multi(case, measured, want):
    names = list(want)
    values = measured.get("values") if isinstance(measured.get("values"), dict) else {}
    absent = [n for n in names if n in (measured.get("absent") or []) or n not in values]
    if absent:
        return ("mismatch", "return the outputs by name: " + ", ".join(names) + " (a dict with these keys); no output called "
                + ", ".join(absent), None, None, want)
    got, tols, bad = {}, {}, []
    for n in names:
        absolute = case["abs"][n] if isinstance(case["abs"], dict) else case["abs"]
        tols[n] = max(case["rel"] * abs(want[n]), absolute)
        got[n] = _number(values.get(n))
        if got[n] is None or not math.isfinite(got[n]) or abs(got[n] - want[n]) > tols[n]:
            bad.append(n)
    if bad:
        return "mismatch", "the output " + ", ".join(bad) + (" does" if len(bad) == 1 else " do") + " not agree", got, tols, want
    return "ok", "", got, tols, want


def judge(case, measured):
    """(status, detail, value, tolerance, expected) of one case: status ok | mismatch | contract | outputs | error. The
    detail never holds the expected value (it goes to the model that repairs the function)."""
    want = case["expected"]
    if not isinstance(measured, dict):
        return "error", "the run that calls the function gave no result for it", None, None, want
    status = measured.get("status")
    if status in ("error", "contract", "outputs"):
        return status, str(measured.get("detail") or ""), None, None, want
    if status in ("not_number", "nonfinite"):
        return "mismatch", str(measured.get("detail") or ""), None, None, want
    if status != "measured":
        return "error", "the run that calls the function gave a result FI cannot read", None, None, want
    if isinstance(want, dict):
        return _judge_multi(case, measured, want)
    if case.get("random"):
        return _judge_draws(case, measured.get("draws"), want)
    value = _number(measured.get("value"))
    if value is None:
        return "mismatch", "it returned something that is not one number", None, None, want
    tol = max(case["rel"] * abs(want), case["abs"])
    if not math.isfinite(value) or abs(value - want) > tol:
        return "mismatch", "", value, tol, want
    return "ok", "", value, tol, want


def main():
    """``python -c <this text> main() <spec>``: measure every case of the spec and print one line, MEASURED_MARK, the
    spec's nonce and the measurements by id. What the code prints goes to stderr, so it never passes for that line."""
    spec = json.loads(sys.argv[-1])
    out = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, os.path.abspath(spec.get("code") or "code"))
    found = {}
    for case in spec.get("cases") or []:
        try:
            found[str(case["id"])] = measure(case)
        except BaseException as e:  # noqa: BLE001
            found[str(case.get("id"))] = {"status": "error", "detail": _err(e)}
    out.write(MEASURED_MARK + " " + str(spec.get("nonce") or "") + " " + json.dumps(found) + "\n")
    out.flush()
