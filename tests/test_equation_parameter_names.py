"""A tested function's parameters, set by FI when only their spelling differs (core/function_steps.py
``match_example_names`` / ``rename_parameters``, Engine._match_example_names).

Replay of the shape of suite round d8 (a model of 18 functions, written in one block). The outline fixed
``def top(Q, spread, wl)`` for an equation whose worked example has the inputs Q, spread and wl; the code was written with
``def top(q, spread, wl)``. The equation test calls the function by the example's names, so it failed on names alone; the
repair of that one function was refused twice because the reply named ``spread`` ``spread_max`` (the equation's own word);
the plan was then asked for the example again and renamed the input, so the test and the repair guard named different
parameters. FI knows both names and renames only when they differ in capitals or underscores (the one pairing with
no second reading); a parameter for another quantity is never renamed into passing, it goes to the repair."""

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


def test_only_capitals_and_underscores_are_paired() -> None:
    new, renames = fs.match_example_names("def f(a, rate_max):\n    return a + rate_max\n", "f", ["a", "RateMax"])
    assert renames == {"rate_max": "RateMax"} and new == "def f(a, RateMax):\n    return a + RateMax\n"
    # One name left on each side but a different word (another quantity): not renamed; the repair takes it.
    assert fs.match_example_names("def f(mass, k):\n    return mass * k\n", "f", ["mass", "length"]) == (None, {})
    assert fs.pair_parameter_names(["k"], ["length"]) is None
    # Two parameters that differ from one wanted name only in case: no single reading, none is made.
    assert fs.pair_parameter_names(["a", "A"], ["a_"]) is None
    assert fs.match_example_names("def f(a, b):\n    return a - b\n", "f", ["x", "y"]) == (None, {})
    # More names wanted than the function has (an input the function does not take): nothing to rename into.
    assert fs.match_example_names("def f(a):\n    return a\n", "f", ["a", "x"]) == (None, {})
    # A parameter the function has and the example does not give stays: it is for the test to report, not to rename.
    assert fs.match_example_names("def f(a, t):\n    return a * t\n", "f", ["a"]) == (None, {})


def test_a_setting_with_a_default_and_fis_own_parameters_are_never_renamed() -> None:
    assert fs.match_example_names("def f(x, scale=2.0):\n    return x * scale\n", "f", ["X"]) == (
        "def f(X, scale=2.0):\n    return X * scale\n", {"x": "X"})
    assert fs.match_example_names("def f(x, scale=2.0):\n    return x * scale\n", "f", ["x", "Gain"]) == (None, {}), \
        "a setting with a default is not paired with an input the function lacks"
    new, renames = fs.match_example_names("def f(x, rng):\n    return x + rng.random()\n", "f", ["X"], random=True)
    assert renames == {"x": "X"} and "rng" in new


def test_a_rename_that_could_change_the_code_is_not_made() -> None:
    clash = "def f(a, b):\n    c = a * b\n    return c\n"  # `c` is already used inside
    assert fs.match_example_names(clash, "f", ["a", "c"]) == (None, {})
    inner = "def f(a, b):\n    g = lambda b: b + 1\n    return a + g(1)\n"
    assert fs.match_example_names(inner, "f", ["a", "B"]) == (None, {})
    # Some code calls the function by the old name: renaming would break it.
    caller = "def g():\n    return f(a=1, b=2)\n"
    assert fs.match_example_names("def f(a, b):\n    return a + b\n", "f", ["a", "B"], callers=[caller]) == (None, {})
    assert fs.match_example_names("def f(a, b):\n    return a + b\n", "f", ["a", "B"], callers=["f(1, 2)\n"])[1] == {"b": "B"}


def test_the_names_in_text_that_is_not_code_are_left_alone() -> None:
    src = 'def f(m, w):\n    """m is the order."""\n    # m again\n    s = "m"\n    return m * w\n'
    new, renames = fs.match_example_names(src, "f", ["M", "w"])
    assert renames == {"m": "M"}
    assert new == 'def f(M, w):\n    """m is the order."""\n    # m again\n    s = "m"\n    return M * w\n'


def test_a_reply_that_only_spells_a_parameter_otherwise_is_used_with_the_fixed_name() -> None:
    reply = "def quality(M, w, C):\n    return M * w / C\n"
    new, why = fs.replace_function(GOOD["frequency"] + "\n\n" + WRITTEN, "quality", reply, signature=SIGNATURE, equation="E2")
    assert why == "" and new is not None
    assert "def quality(M, w, c):" in new and "return M * w / c" in new and "C" not in new


@pytest.mark.parametrize("reply", [
    "def quality(a, b, c):\n    return a * b / c\n",                  # three names left: a guess
    "def quality(M, w, c, k):\n    return M * w / c\n",               # one more parameter than fixed
    "def quality(M, w):\n    return M * w\n",                         # one fewer
    "def quality(M, w, c_max):\n    return M * w / c_max\n",         # another word: the equation's own
])
def test_a_reply_with_other_parameters_is_still_refused(reply: str) -> None:
    new, why = fs.replace_function(WRITTEN, "quality", reply, signature=SIGNATURE, equation="E2")
    assert new is None and "must be exactly (M, w, c)" in why


@pytest.mark.asyncio
async def test_the_gate_sets_the_examples_capitals_and_a_wrong_formula_still_fails(tmp_path: Path) -> None:
    """The record's shape in the engine: the example says `m` and the code says `M`; no model is asked, the file is
    changed, and the test then judges the value (code that computes the wrong thing still fails)."""
    protocol = {"grid": {"c": [0.5, 1.0]}, "model": {"summary": "s", "equations": [
        PROTOCOL["model"]["equations"][0],
        {**PROTOCOL["model"]["equations"][1], "example": {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5},
                                                          "expected_formula": "m * w / c"}}]}}
    for body, passes in (("M * w / c", True), ("w / c", False)):
        engine = _engine(tmp_path / ("good" if passes else "bad"))
        _plan_md(engine, protocol)
        code = engine.quest_root / "code"
        (code / "spring").mkdir(parents=True)
        (code / "spring" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
        (code / "spring" / "model.py").write_text(
            GOOD["frequency"] + f"\n\n# E2\ndef quality(M, w, c):\n    return {body}\n", encoding="utf-8")
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


@pytest.mark.asyncio
async def test_a_parameter_for_another_quantity_is_repaired_not_renamed_into_passing(tmp_path: Path) -> None:
    """`quality(mass, w, c)` has a parameter for a quantity the example does not give, and its body gives the example's
    value. Renaming `mass` to `m` would pass the example while the callers still hand it the other quantity; FI sends
    the function to the one-function repair instead."""
    engine = _engine(tmp_path)
    _plan_md(engine)
    code = engine.quest_root / "code"
    (code / "spring").mkdir(parents=True)
    (code / "spring" / "__init__.py").write_text("from . import model"+chr(10), encoding="utf-8")
    (code / "spring" / "model.py").write_text(GOOD["frequency"] + chr(10)*2 + "# E2" + chr(10) + "def quality(mass, w, c):" + chr(10)
                                              + "    return mass * w / c" + chr(10), encoding="utf-8")
    (code / "simulate.py").write_text("def run_cell(cell):" + chr(10) + "    return {}" + chr(10), encoding="utf-8")
    writer = _Writer({"quality": ["# E2" + chr(10) + "def quality(m, w, c):" + chr(10) + "    return m * w / c" + chr(10)]})
    engine._client = writer
    patch = await engine._equation_gate({"design": {"protocol": PROTOCOL}, "implement_outline": OUTLINE}, None, None)
    assert patch is None and len(writer.prompts) == 1, "the function was repaired by the model, not renamed"
