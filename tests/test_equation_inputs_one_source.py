"""A function that implements a plan equation takes the example's inputs as its parameters, set by FI when it reads the
outline (core/function_steps.py ``example_signature`` / ``align_outline``). Neutral model: a falling body."""

from __future__ import annotations

import ast
from typing import Any

from core import equation_tests as eqt
from core import function_steps as fs


def _protocol(example: dict[str, Any] | None, eid: str = "E1") -> dict[str, Any]:
    eq: dict[str, Any] = {"id": eid, "formula": "h = g * t**2 / 2", "role": "generates", "source": "derivation"}
    if example is not None:
        eq["example"] = example
    return {"model": {"summary": "a falling body", "equations": [eq]}}


GOOD = {"inputs": {"g": 9.8, "t": 2.0}, "expected_formula": "g * t**2 / 2"}


def _row(example: dict[str, Any] | None = GOOD) -> dict[str, Any]:
    return eqt.example_rows(_protocol(example))[0]


def _names(sig: str) -> list[str]:
    return fs._param_names(ast.parse(sig + "\n    pass").body[0])  # type: ignore[arg-type]


def test_other_names_become_the_examples_names_and_annotations_are_kept() -> None:
    new = fs.example_signature("def drop(gravity: float, seconds: float) -> float:", _row())
    assert _names(new) == ["g", "t"] and new.endswith("-> float:")
    new = fs.example_signature("def drop(t: float, gravity: float) -> float:", _row())
    assert _names(new) == ["t", "g"] and "t: float" in new  # the outline's order for a name it had


def test_names_that_already_agree_are_left_exactly_as_written() -> None:
    sig = "def drop(g: float = 9.8, t: float = 1.0) -> float:"
    assert fs.example_signature(sig, _row()) == sig
    extra = "def drop(g, t, terms=50):"  # a setting with a default is not a clash
    assert fs.example_signature(extra, _row()) == extra


def test_a_setting_with_a_default_is_kept_and_a_required_extra_is_dropped() -> None:
    new = fs.example_signature("def drop(gravity, secs, terms=50, mass):".replace(", mass", ""), _row())
    assert _names(new) == ["g", "t", "terms"] and new.endswith("terms=50):")
    new = fs.example_signature("def drop(g, t, mass):", _row())
    assert _names(new) == ["g", "t"]


def test_rng_only_for_an_example_the_plan_flagged_random() -> None:
    assert _names(fs.example_signature("def drop(gravity, rng):", _row())) == ["g", "t"]
    flagged = _row({**GOOD, "random": True})
    assert _names(fs.example_signature("def drop(gravity, secs):", flagged)) == ["g", "t", "rng"]
    assert _names(fs.example_signature("def drop(gravity, secs, seed):", flagged)) == ["g", "t", "seed"]


def test_an_example_that_is_not_usable_and_a_star_signature_are_not_touched() -> None:
    sig = "def drop(gravity, secs):"
    assert fs.example_signature(sig, _row(None)) == sig  # no example
    assert fs.example_signature(sig, _row({"untestable": "no closed form"})) == sig
    star = "def drop(*args, **kw):"
    assert fs.example_signature(star, _row()) == star


def _outline(sig: str = "def drop(gravity, seconds) -> float:") -> dict[str, Any]:
    return {"scaffold": "import numpy as np\n\n\ndef drop(gravity, seconds) -> float:\n    raise NotImplementedError\n\n\n"
                        "def main():\n    drop(1, 2)\n",
            "functions": [{"name": "drop", "signature": sig, "one_line_purpose": "h", "implements": "E1", "depends_on": []}]}


def test_the_outline_read_gives_the_functions_the_examples_parameters() -> None:
    protocol = _protocol(GOOD)
    specs, notes = fs.model_functions(_outline(), protocol)
    assert [s.params() for s in specs] == [["g", "t"]], notes
    outline = _outline()
    aligned, changed = fs.align_outline(outline, protocol)
    assert aligned["functions"][0]["signature"] == "def drop(g, t) -> float:"
    assert "def drop(g, t) -> float:\n    raise NotImplementedError" in aligned["scaffold"]
    assert len(changed) == 1 and "drop (E1)" in changed[0]
    assert outline == _outline(), "the outline given is not changed"
    assert fs.align_outline(aligned, protocol) == (aligned, []), "reading it again changes nothing"


def test_the_clash_that_asked_the_plan_to_rename_its_example_no_longer_arises() -> None:
    row = _row()
    for sig in ("def drop(gravity, seconds):", "def drop(g):", "def drop(g, t, mass):"):
        old = fs.Spec("drop", sig, "", "E1")
        assert eqt.parameter_problems([row], {"E1": (old.params(), old.required_params())}), sig
        new = fs.Spec("drop", fs.example_signature(sig, row), "", "E1")
        assert not eqt.parameter_problems([row], {"E1": (new.params(), new.required_params())}), sig
