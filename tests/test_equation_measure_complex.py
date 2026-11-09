"""A formula whose value is a complex number is measured as one (core/equation_measure.py), not refused as "not one number".
A real worked example agrees with a complex value only when the imaginary part is within the same tolerance of zero.
Neutral functions only."""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from core import equation_measure as em

MODULE = '''
import cmath, math
import numpy as np

def real_fn(x):
    return x * 2.0

def zero_imag(x):
    return complex(-math.sqrt(x), 0.0)

def small_imag(x):
    return complex(-math.sqrt(x), 1e-12)

def real_imag(x):
    return complex(0.1, 0.5)

def np_scalar(x):
    return np.complex128(-math.sqrt(x) + 0j)

def np_scalar_imag(x):
    return np.complex128(0.1 + 0.5j)

def np_one(x):
    return np.array([complex(-math.sqrt(x), 0.0)])

def np_two(x):
    return np.array([1 + 0j, 2 + 0j])

def named(x):
    return {"a": complex(x, 0.0), "b": complex(x, 0.7)}

def draws_ok(x, rng):
    return complex(x + rng.normal(), 0.0)

def draws_imag(x, rng):
    return complex(x + rng.normal(), 0.5)
'''


@pytest.fixture()
def mod(tmp_path: Path):
    name = "cplx_fixture_mod"
    (tmp_path / (name + ".py")).write_text(textwrap.dedent(MODULE), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        yield name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()


def _run(mod: str, fn: str, want, **extra):
    case = {"id": "E1", "module": mod, "function": fn, "inputs": {"x": 0.06}, "rel": 1e-6, "abs": 1e-9, "expected": want, **extra}
    return em.judge(case, em.measure(case)), em.measure(case)


def test_a_complex_value_with_no_imaginary_part_matches_a_real_example(mod) -> None:
    want = -(0.06 ** 0.5)
    for fn in ("zero_imag", "np_scalar", "np_one"):
        (status, detail, value, _tol, _want), measured = _run(mod, fn, want)
        assert measured["status"] == "measured", (fn, measured)
        assert status == "ok" and value == pytest.approx(want), (fn, detail)


def test_a_real_imaginary_part_against_a_real_example_is_a_mismatch_that_names_it(mod) -> None:
    for fn in ("real_imag", "np_scalar_imag"):
        (status, detail, value, _tol, _want), _m = _run(mod, fn, 0.1)
        assert status == "mismatch" and "imaginary part 0.5" in detail, (fn, detail)
        assert value == pytest.approx(0.1)  # the real part agrees; the record shows it, the detail names the rest


def test_an_imaginary_part_within_the_tolerance_of_zero_agrees(mod) -> None:
    (status, *_), _m = _run(mod, "small_imag", -(0.06 ** 0.5), abs=1e-9)
    assert status == "ok"
    (status, *_), _m = _run(mod, "small_imag", -(0.06 ** 0.5), abs=1e-15, rel=0.0)
    assert status == "mismatch"


def test_a_measurement_with_a_complex_value_is_json_safe(mod) -> None:
    import json
    json.dumps(_run(mod, "real_imag", 0.1)[1])


def test_a_real_function_and_a_wrong_value_behave_as_before(mod) -> None:
    (status, _d, value, _t, _w), measured = _run(mod, "real_fn", 0.12)
    assert status == "ok" and "imag" not in measured and value == pytest.approx(0.12)
    (status, detail, value, _t, _w), _m = _run(mod, "real_fn", 0.5)
    assert status == "mismatch" and detail == "" and value == pytest.approx(0.12)
    # several numbers are still not one number
    (status, detail, *_), measured = _run(mod, "np_two", 1.0)
    assert measured["status"] == "not_number" and status == "mismatch" and "several" in detail


def test_named_outputs_with_a_complex_value_are_compared_by_part(mod) -> None:
    case = {"id": "E1", "module": mod, "function": "named", "inputs": {"x": 2.0}, "rel": 1e-6, "abs": 1e-9,
            "names": ["a", "b"], "expected": {"a": 2.0, "b": 2.0}}
    status, detail, *_ = em.judge(case, em.measure(case))
    assert status == "mismatch" and "b: " in detail and "imaginary part 0.7" in detail and "a: " not in detail


def test_complex_draws_agree_with_a_real_distribution_only_when_the_imaginary_parts_are_zero(mod) -> None:
    case = {"id": "E1", "module": mod, "function": "draws_ok", "inputs": {"x": 3.0}, "rel": 1e-6, "abs": 1e-9,
            "random": True, "variance": 1.0, "expected": 3.0}
    assert em.judge(case, em.measure(case))[0] == "ok"
    case = {**case, "function": "draws_imag"}
    status, detail, *_ = em.judge(case, em.measure(case))
    assert status == "mismatch" and "imaginary part 0.5" in detail
