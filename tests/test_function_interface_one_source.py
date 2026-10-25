"""One source for a function's interface (core/equation_tests.py, core/function_steps.py, core/code_layout.py,
core/oracle_check.py, core/run_estimate.py, core/second_impl.py).

A function's interface used to have several sources that disagreed: the outline's signature, the plan's worked example,
the code's own parameters, and a `# E<n>` comment. Equation tests and code repairs then fought over them. Each test below
builds the disagreement from a neutral model (a mass on a spring, a cooling cup) and shows the one source winning, and
each has the planted fault that must still be caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

from core import code_layout as cl, equation_measure as em, equation_tests as eqt, function_steps as fs
from core import oracle_check as oc, run_estimate as re_, second_impl as si
from tests.test_equation_gate import GOOD, OUTLINE, PROTOCOL

OUTLINE_NAMES = {"E1": "frequency", "E2": "quality"}


# --- the outline's function is the one source of which function an equation test calls; no label is looked for ----------


def test_an_equation_the_outline_names_a_function_for_is_never_looked_for_by_label() -> None:
    renamed = GOOD["frequency"] + "\n\n# E2\ndef q_factor(m, w, c):\n    return m * w / c\n"
    sources = {"spring/model.py": renamed}
    located = eqt.locate(PROTOCOL, sources, OUTLINE_NAMES)
    assert located["E1"]["function"] == "frequency"
    assert located["E2"] is None, "the outline's `quality` is not in the code; the label on `q_factor` is not a second source"
    # Known positive: an equation the outline names no function for is still found by its label.
    assert eqt.locate(PROTOCOL, sources, {"E1": "frequency"})["E2"]["function"] == "q_factor"
    rows = eqt.example_rows(PROTOCOL)
    notes = eqt.untested_notes(rows, located, OUTLINE_NAMES)
    assert notes == ["equation E2 is not tested before the run: the code has no function `quality`, the one planned for it"]


def test_the_outline_names_the_function_so_no_label_is_asked_for_and_none_is_missing() -> None:
    bare = {"spring/model.py": "def frequency(k, m):\n    return (k / m) ** 0.5\n\n\ndef quality(m, w, c):\n    return m * w / c\n"}
    assert oc.unlabelled_equations(PROTOCOL, bare) == ["E1", "E2"], "without the outline: labels are missing, as before"
    assert oc.unlabelled_equations(PROTOCOL, bare, OUTLINE_NAMES) == []
    assert oc.label_gaps(PROTOCOL, bare, "model.py", OUTLINE_NAMES) == []
    # Known positive: a function the outline names that the code does not define still needs its label.
    assert oc.unlabelled_equations(PROTOCOL, bare, {"E1": "frequency", "E2": "q_factor"}) == ["E2"]
    rows = cl.equation_map(PROTOCOL, bare, OUTLINE_NAMES)
    assert [r["function"] for r in rows] == ["frequency", "quality"]


def test_the_layout_check_does_not_ask_for_labels_where_the_outline_names_the_functions(tmp_path: Path) -> None:
    code = tmp_path / "code"
    (code / "spring").mkdir(parents=True)
    (code / "spring" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (code / "spring" / "model.py").write_text(
        "def frequency(k, m):\n    return (k / m) ** 0.5\n\n\ndef quality(m, w, c):\n    return m * w / c\n", encoding="utf-8")
    (code / "simulate.py").write_text("from spring import model\n", encoding="utf-8")
    without = [p for p in cl.check(code, PROTOCOL, "spring") if "not labelled" in p]
    assert without and "E1, E2" in without[0]
    named = cl.check(code, PROTOCOL, "spring", OUTLINE_NAMES)
    assert not [p for p in named if "not labelled" in p or "planned for equation" in p]
    absent = cl.check(code, PROTOCOL, "spring", {"E1": "frequency", "E2": "q_factor"})
    assert any("planned for equation E2" in p and "not in code/spring/" in p for p in absent)


# --- a repair is held to ONE interface: the outline's where the test cannot call the code yet, else the code's ------------

# The code was first written with the example's parameter NAMES in another ORDER than the signature built from the example
# (the shape of a recorded round: the tests pass, and every whole-file repair was refused on order alone).
REORDERED = "def frequency(m, k):\n    return (k / m) ** 0.5\n\n\ndef quality(c, w, m):\n    return m * w / c\n"
OUTLINE_SIGS = {"frequency": "def frequency(*, k, m) -> float:", "quality": "def quality(*, m, w, c) -> float:"}


def _cases(source: str) -> list[dict]:
    sources = {"spring/model.py": source}
    return eqt.cases(eqt.example_rows(PROTOCOL), eqt.locate(PROTOCOL, sources, OUTLINE_NAMES))


def test_code_the_test_can_already_call_keeps_its_own_def_even_when_only_the_order_differs() -> None:
    sources = {"spring/model.py": REORDERED}
    held = fs.held_signatures(OUTLINE_SIGS, sources, _cases(REORDERED))
    assert held == {"frequency": "def frequency(m, k):", "quality": "def quality(c, w, m):"}
    # The whole-file repair the quest needed (a fix elsewhere, the parameters as written) is no longer refused.
    assert fs.changed_signatures(REORDERED.replace("** 0.5", "**0.5"), held) == []
    # Old: held to the outline's order, the same repair was refused for both functions.
    assert len(fs.changed_signatures(REORDERED, OUTLINE_SIGS)) == 2


def test_code_the_test_cannot_call_is_held_to_the_outlines_names_and_a_renaming_repair_is_still_refused() -> None:
    wrong_names = "def frequency(spring_k, m):\n    return (spring_k / m) ** 0.5\n\n\ndef quality(m, w, c):\n    return m * w / c\n"
    sources = {"spring/model.py": wrong_names}
    held = fs.held_signatures(OUTLINE_SIGS, sources, _cases(wrong_names))
    assert held["frequency"] == OUTLINE_SIGS["frequency"], "the test calls `k`: the repair must give it"
    assert held["quality"] == "def quality(m, w, c):"
    # Known positive: a repair that renames or drops a parameter of a function the test calls is still refused.
    renamed = wrong_names.replace("def quality(m, w, c)", "def quality(m, w, damping)").replace("m * w / c", "m * w / damping")
    refused = fs.changed_signatures(renamed, held)
    assert len(refused) == 2 and any("`quality`" in r for r in refused)


def test_a_function_the_code_does_not_define_keeps_the_outlines_line() -> None:
    held = fs.held_signatures(OUTLINE_SIGS, {"spring/model.py": "def other():\n    pass\n"}, [])
    assert held == OUTLINE_SIGS


# --- the pilot reads a run_trial that is a whole cell --------------------------------------------------------------------------

CELL_AS_TRIAL = (
    "def run_trial(cell, trial_id=0, seed=0):\n    if isinstance(cell, dict):\n        return run_cell(cell)\n    return {'x': 1.0}\n\n\n"
    "def run_cell(cell):\n    return {'x': sum(run_trial(1.0, i, i)['x'] for i in range(1000)) / 1000}\n")
CELL_FROM_TRIALS = (
    "def run_trial(cell, trial_id=0, seed=0):\n    return {'x': 1.0}\n\n\n"
    "def run_cell(cell):\n    return {'x': sum(run_trial(cell, i, i)['x'] for i in range(1000)) / 1000}\n")


def test_the_pilot_names_a_run_trial_that_returns_the_whole_cell() -> None:
    found = re_.pilot_problems([], {}, source=CELL_AS_TRIAL, stochastic=True)
    assert [p.kind for p in found.problems] == ["aggregate"] and "ONE trial" in found.problems[0].says
    # Old behaviour (nothing said before the run) when the study does not run trials, or the reading is not asked for.
    assert re_.pilot_problems([], {}, source=CELL_AS_TRIAL, stochastic=False).problems == []
    assert re_.pilot_problems([], {}, source=CELL_AS_TRIAL).problems == []
    # Known negative: a run_cell that is built from run_trial is how a cell is made, never a finding.
    assert re_.pilot_problems([], {}, source=CELL_FROM_TRIALS, stochastic=True).problems == []


# --- the second version is compared on inputs inside the function's domain ------------------------------------------------------


def _spec(example: dict) -> tuple[fs.Spec, dict]:
    protocol = {"model": {"equations": [{"id": "E1", "formula": "phi = z * (n - sqrt(n^2 - (w f)^2))", "role": "generates",
                                        "example": example}]}}
    return fs.Spec(name="phase", signature="def phase(z, f, w, n):", purpose="phase", equation="E1"), protocol


EXAMPLE = {"inputs": {"z": 100.0, "f": 0.005, "w": 200.0, "n": 1.4}, "expected_formula": "z*(n - sqrt(n**2 - (w*f)**2))"}


def test_inputs_outside_the_domain_of_the_examples_own_formula_are_not_built() -> None:
    spec, protocol = _spec(EXAMPLE)
    base = si.example_inputs(spec, protocol)
    old = si.build_inputs(base)
    new = si.build_inputs(base, defined=si.domain_of(spec, protocol, base))
    outside = [c for c in old if c["w"] * c["f"] > c["n"]]
    assert outside, "scaling every input by 2 pushes w*f (x4) past n (x2): the old inputs include such a case"
    assert not [c for c in new if c["w"] * c["f"] > c["n"]] and new[0] == base and len(new) >= 3


def test_a_wrong_version_inside_the_domain_still_disagrees_on_the_kept_inputs() -> None:
    import math

    spec, protocol = _spec(EXAMPLE)
    base = si.example_inputs(spec, protocol)
    kept = si.build_inputs(base, defined=si.domain_of(spec, protocol, base))

    def right(z, f, w, n):
        return z * (n - math.sqrt(n * n - (w * f) ** 2))

    def planted(z, f, w, n):  # drops the refractive index from the root
        return z * (n - math.sqrt(1 - (w * f) ** 2 / n))

    rows = [{"main": right(**c), "ref": planted(**c)} for c in kept]
    assert si.judge_rows([{"main": r["main"], "ref": r["ref"]} for r in rows], kept, si.REL)["status"] == si.DISAGREE
    same = [{"main": right(**c), "ref": right(**c)} for c in kept]
    assert si.judge_rows(same, kept, si.REL)["status"] == si.AGREE


def test_no_domain_is_applied_when_the_formula_cannot_be_read_from_the_example() -> None:
    spec, protocol = _spec({"inputs": EXAMPLE["inputs"], "expected_formula": "z*(n - sqrt(n**2 - (w*f)**2)) + unknown_term"})
    assert si.domain_of(spec, protocol, EXAMPLE["inputs"]) is None, "unknown_term is not an input: the formula is not read"
    assert si.domain_of(spec, protocol, None) is None
    spec2, protocol2 = _spec({"inputs": EXAMPLE["inputs"], "untestable": "no closed form"})
    assert si.domain_of(spec2, protocol2, EXAMPLE["inputs"]) is None


# --- the example declares the outputs; the reason a worked example could not be met is the exact one ---------------------------

AND_FORM = {"id": "E1", "formula": "Ea_hat = -R*beta and ln_A_hat = alpha + Ea_hat*x/R", "role": "analyses",
            "example": {"inputs": {"R": 8.0, "beta": 2.0, "alpha": 1.0, "x": 0.5},
                        "expected_formula": {"Ea_hat": "-R*beta", "ln_A_hat": "alpha + (-R*beta)*x/R"}}}


def _rows(equation: dict) -> dict:
    return {r["id"]: r for r in eqt.example_rows({"model": {"equations": [equation]}})}


def test_a_mapping_declares_the_outputs_when_the_formula_joins_them_with_and() -> None:
    row = _rows(AND_FORM)["E1"]
    assert row["state"] == "ok" and row["outputs"] == ["Ea_hat", "ln_A_hat"]
    assert row["expected"] == {"Ea_hat": -16.0, "ln_A_hat": 1.0 + (-16.0) * 0.5 / 8.0}
    # Old: the text defines one name, so the mapping was refused as unusable.
    assert eqt.is_multi_output(AND_FORM["formula"]) is False


def test_a_mapping_that_is_not_two_outputs_is_still_refused_and_one_formula_stays_one_output() -> None:
    one_key = {**AND_FORM, "example": {**AND_FORM["example"], "expected_formula": {"Ea_hat": "-R*beta"}}}
    assert _rows(one_key)["E1"]["state"] == "unusable"
    scalar = {**AND_FORM, "example": {**AND_FORM["example"], "expected_formula": "-R*beta"}}
    row = _rows(scalar)["E1"]
    assert row["state"] == "ok" and row["outputs"] == [] and isinstance(row["expected"], float)
    # The keys of a mapping must be the names a formula that does define several names defines.
    named = {**AND_FORM, "formula": "a = x + 1; b = x + 2", "example": {"inputs": {"x": 1.0}, "expected_formula": {"p": "x", "q": "x"}}}
    assert _rows(named)["E1"]["state"] == "unusable"


def test_a_function_that_returns_several_values_for_one_expected_value_says_so(tmp_path: Path) -> None:
    (tmp_path / "pair_module.py").write_text("def pair(x):\n    return (x, 2 * x)\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        measured = em.measure({"id": "E1", "module": "pair_module", "function": "pair", "inputs": {"x": 1.0}, "random": False,
                               "names": None, "outputs": []})
    finally:
        sys.path.remove(str(tmp_path))
    assert measured["status"] == "not_number" and "several values" in measured["detail"]
    status, detail, *_ = em.judge({"expected": 1.0, "rel": 1e-6, "abs": 0.0}, measured)
    assert status == eqt.MISMATCH and "several values" in detail
    assert "mapping from each output's name" in eqt.could_not_meet(status, detail)
    assert "mapping" not in eqt.could_not_meet(eqt.MISMATCH, "")


def test_a_parameter_the_example_gives_no_value_for_is_told_apart_from_names_that_differ() -> None:
    missing = eqt.could_not_meet(eqt.CONTRACT, "the parameter covariance has no default and is not one of the example's inputs; "
                                              "give it a default or take it out of this function")
    differ = eqt.could_not_meet(eqt.CONTRACT, "the function must take T_i as arguments, named so")
    assert "gives no value for" in missing and "every input the formula uses" in missing
    assert "parameters are not the example's input names" in differ
    request = eqt.rederive_request([{"id": "E4", "formula": "y = a*b", "example": {"inputs": {"a": 1}}, "why": eqt.CONTRACT,
                                     "detail": "the parameter c has no default and is not one of the example's inputs"}])
    assert "gives no value for" in request


# --- the plan's equation and the example are compared by the one name the plan is told to write ---------------------------


def test_a_reserved_word_symbol_is_the_input_with_a_trailing_underscore() -> None:
    inputs = {"lambda_": 200.0, "sigma": 0.5, "NA": 1.0}
    assert abs(eqt.equation_value("d = 0.61*lambda/(sigma*NA)", inputs) - 0.61 * 200.0 / 0.5) < 1e-9
    assert eqt.equation_value("d = 0.61*lambda/(sigma*NA)", {"lambda_nm": 200.0, "sigma": 0.5, "NA": 1.0}) is None
    equation = {"id": "E1", "formula": "d = 0.61*lambda/(sigma*NA)", "role": "generates",
                "example": {"inputs": inputs, "expected_formula": "3.8317059702*lambda_/(2*pi*sigma*NA)"}}
    row = _rows(equation)["E1"]
    # The example (0.6098) and the equation (0.61) are one number rounded two ways (0.61 carries about 0.8% of
    # rounding): FI computes both, accepts the example, and the test bar grows to cover both.
    assert row["state"] == "ok" and row["equation_agrees"] is True and row["rounded_form"] is True
    # An example beyond what the rounding explains is two different numbers: refused, with both values stated.
    wrong = {**equation, "example": {"inputs": inputs, "expected_formula": "0.8*lambda_/(sigma*NA)"}}
    assert _rows(wrong)["E1"]["state"] == "unusable" and "the equation as written gives" in _rows(wrong)["E1"]["why"]
    good = {**equation, "example": {"inputs": inputs, "expected_formula": "0.61*lambda_/(sigma*NA)"}}
    assert _rows(good)["E1"]["state"] == "ok" and _rows(good)["E1"]["equation_agrees"] is True
