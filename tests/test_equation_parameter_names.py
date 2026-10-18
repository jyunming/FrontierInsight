"""A tested function's parameters, set by FI when only their spelling differs (core/function_steps.py
``match_example_names`` / ``rename_parameters``, Engine._match_example_names).

Replay of the shape of suite round d8 (a model of 18 functions, written in one block). The outline fixed
``def top(Q, spread, wl)`` for an equation whose worked example has the inputs Q, spread and wl; the code was written with
``def top(q, spread, wl)``. The equation test calls the function by the example's names, so it failed on names alone; the
repair of that one function was refused twice because the reply named ``spread`` ``spread_max`` (the equation's own word);
the plan was then asked for the example again and renamed the input, so the test and the repair guard named different
parameters. FI knows both names: it sets the one the example gives."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import function_steps as fs
from tests.test_equation_gate import GOOD, OUTLINE, PROTOCOL, _engine, _plan_md, _Writer

# Neutral stand-ins for the real pair: the code's `m` for the example's `M`, and a reply that spells `c` as `c_max`.
SIGNATURE = "def quality(M, w, c) -> float:"
WRITTEN = "# E2\ndef quality(m, w, c):\n    return m * w / c\n"


def test_a_name_that_differs_only_in_capitals_is_set_to_the_examples() -> None:
    new, renames = fs.match_example_names(WRITTEN, "quality", ["M", "w", "c"])
    assert renames == {"m": "M"}
    assert new == "# E2\ndef quality(M, w, c):\n    return M * w / c\n"


def test_one_name_left_on_each_side_is_paired_and_two_are_not() -> None:
    new, renames = fs.match_example_names("def f(a, b):\n    return a + b\n", "f", ["a", "b_max"])
    assert renames == {"b": "b_max"} and new == "def f(a, b_max):\n    return a + b_max\n"
    # Two unmatched on each side: which goes with which is a guess FI does not make.
    assert fs.match_example_names("def f(a, b):\n    return a - b\n", "f", ["x", "y"]) == (None, {})
    # More names wanted than the function has (an input the function does not take): nothing to rename into.
    assert fs.match_example_names("def f(a):\n    return a\n", "f", ["a", "x"]) == (None, {})
    # A parameter the function has and the example does not give stays: it is for the test to report, not to rename.
    assert fs.match_example_names("def f(a, t):\n    return a * t\n", "f", ["a"]) == (None, {})


def test_a_setting_with_a_default_and_fis_own_parameters_are_never_renamed() -> None:
    assert fs.match_example_names("def f(x, scale=2.0):\n    return x * scale\n", "f", ["X"]) == (
        "def f(X, scale=2.0):\n    return X * scale\n", {"x": "X"})
    assert fs.match_example_names("def f(x, scale=2.0):\n    return x * scale\n", "f", ["x", "gain"]) == (None, {}), \
        "a setting with a default is not paired with an input the function lacks"
    new, renames = fs.match_example_names("def f(x, rng):\n    return x + rng.random()\n", "f", ["X"], random=True)
    assert renames == {"x": "X"} and "rng" in new


def test_a_rename_that_could_change_the_code_is_not_made() -> None:
    clash = "def f(a, b):\n    c = a * b\n    return c\n"  # `c` is already used inside
    assert fs.match_example_names(clash, "f", ["a", "c"]) == (None, {})
    inner = "def f(a, b):\n    g = lambda b: b + 1\n    return a + g(1)\n"
    assert fs.match_example_names(inner, "f", ["a", "b_max"]) == (None, {})
    # Some code calls the function by the old name: renaming would break it.
    caller = "def g():\n    return f(a=1, b=2)\n"
    assert fs.match_example_names("def f(a, b):\n    return a + b\n", "f", ["a", "b_max"], callers=[caller]) == (None, {})
    assert fs.match_example_names("def f(a, b):\n    return a + b\n", "f", ["a", "b_max"], callers=["f(1, 2)\n"])[1] == {"b": "b_max"}


def test_the_names_in_text_that_is_not_code_are_left_alone() -> None:
    src = 'def f(m, w):\n    """m is the order."""\n    # m again\n    s = "m"\n    return m * w\n'
    new, renames = fs.match_example_names(src, "f", ["M", "w"])
    assert renames == {"m": "M"}
    assert new == 'def f(M, w):\n    """m is the order."""\n    # m again\n    s = "m"\n    return M * w\n'


def test_a_reply_that_only_spells_a_parameter_otherwise_is_used_with_the_fixed_name() -> None:
    reply = "def quality(M, w, c_max):\n    return M * w / c_max\n"
    new, why = fs.replace_function(GOOD["frequency"] + "\n\n" + WRITTEN, "quality", reply, signature=SIGNATURE, equation="E2")
    assert why == "" and new is not None
    assert "def quality(M, w, c):" in new and "return M * w / c" in new and "c_max" not in new


@pytest.mark.parametrize("reply", [
    "def quality(a, b, c):\n    return a * b / c\n",                  # three names left: a guess
    "def quality(M, w, c, k):\n    return M * w / c\n",               # one more parameter than fixed
    "def quality(M, w):\n    return M * w\n",                         # one fewer
])
def test_a_reply_with_other_parameters_is_still_refused(reply: str) -> None:
    new, why = fs.replace_function(WRITTEN, "quality", reply, signature=SIGNATURE, equation="E2")
    assert new is None and "must be exactly (M, w, c)" in why


@pytest.mark.asyncio
async def test_the_gate_sets_the_examples_name_and_a_wrong_formula_still_fails(tmp_path: Path) -> None:
    """The record's shape in the engine: the example says `m` and the code says `mass`; no model is asked, the file is
    changed, and the test then judges the value (code that computes the wrong thing still fails)."""
    protocol = {"grid": {"c": [0.5, 1.0]}, "model": {"summary": "s", "equations": [
        PROTOCOL["model"]["equations"][0],
        {**PROTOCOL["model"]["equations"][1], "example": {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5},
                                                          "expected_formula": "m * w / c"}}]}}
    for body, passes in (("mass * w / c", True), ("w / c", False)):
        engine = _engine(tmp_path / ("good" if passes else "bad"))
        _plan_md(engine, protocol)
        code = engine.quest_root / "code"
        (code / "spring").mkdir(parents=True)
        (code / "spring" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
        (code / "spring" / "model.py").write_text(
            GOOD["frequency"] + f"\n\n# E2\ndef quality(mass, w, c):\n    return {body}\n", encoding="utf-8")
        (code / "simulate.py").write_text("from spring import model\n\n\ndef run_cell(cell):\n    return {}\n", encoding="utf-8")
        engine._client = _Writer({"quality": ["# E2\ndef quality(m, w, c):\n    return w / c\n"]})
        state = {"design": {"protocol": protocol}, "implement_outline": OUTLINE}
        patch = await engine._equation_gate(state, None, None)
        model = (code / "spring" / "model.py").read_text(encoding="utf-8")
        assert "def quality(m, w, c):" in model, "FI set the example's name in the written code"
        if passes:
            assert patch is None, patch['exec_result']['stderr_tail']
        else:
            assert patch is not None and "quality" in patch["exec_result"]["stderr_tail"], "a wrong value still fails its test"
