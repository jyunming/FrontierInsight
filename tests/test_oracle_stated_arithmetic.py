"""FI computes the sums a plan works out in its own words (a check's reference or derivation) before anything runs
(core/oracle_forms.py::stated_arithmetic, reported by formula_findings and put to the plan in the one request that
already goes back about the checks). Fake models only: nothing here calls a real model."""

from __future__ import annotations

from core import oracle_forms as of

# The real check of round r13: the plan's hand arithmetic said 3.448570123, the sum is 3.45187.
R13 = {"name": "arrhenius_example_rate", "kind": "special_case", "expected": 0, "expected_formula": "0",
       "tolerance": 1e-9, "tolerance_mode": "absolute",
       "reference": "derivation: E2 gives k=100000000*exp(-50000/(8.314462618*350))=3.448570123e0 s^-1, so "
                    "abs(k_E2-3.448570123)/3.448570123=0 at the stated comparison precision.",
       "case": {"noise_sigma_over_c0": 0, "n_time": 9}, "measure": "abs(k_E2_example - 3.448570123) / 3.448570123"}


def _arithmetic(*oracles):
    return [f for f in of.formula_findings({"oracles": list(oracles)}) if f["state"] == "arithmetic"]


def test_the_r13_sum_is_caught_with_both_numbers():
    found = _arithmetic(R13)
    assert len(found) == 1
    assert found[0]["name"] == "arrhenius_example_rate"
    assert found[0]["expression"] == "100000000*exp(-50000/(8.314462618*350))"
    assert found[0]["stated"] == "3.448570123e0"
    assert abs(found[0]["value"] - 3.4518687) < 1e-6
    ask = of.formula_request(found)
    assert "100000000*exp(-50000/(8.314462618*350)) = 3.448570123e0" in ask and "3.4518687033" in ask


def test_a_correct_sum_is_not_flagged():
    good = {**R13, "reference": "derivation: k=100000000*exp(-50000/(8.314462618*350))=3.45186870 s^-1; "
                                "max(k)/min(k) = 1/0.1 = 10; 1/3 = 0.333; sqrt(2)=1.41421356"}
    assert _arithmetic(good) == []


def test_numbers_rounded_on_the_way_are_not_a_slip():
    # 2.006 * (2/pi) * 1.654 = 2.1123, written 2.1118: the factors are only good to their written digits.
    rounded = {"name": "p", "reference": "T = 2.006 * (2/pi) * 1.654 = 2.1118"}
    assert _arithmetic(rounded) == []


def test_an_expression_with_a_name_is_not_read():
    named = {"name": "n", "reference": "so k_E2*2 = 7 and x + 1 = 5, but 2 * 3 = 6 and a <= 5, 1+1=>2"}
    assert _arithmetic(named) == []


def test_a_whole_number_is_exact_as_written():
    assert [f["expression"] for f in _arithmetic({"name": "w", "reference": "2 * 3 = 7"})] == ["2 * 3"]


def test_a_sum_in_a_text_field_that_is_not_a_working_out_is_not_read():
    # Only the reference (and a derivation) is a working-out; a wrong sum in the check's name or a note is not judged.
    in_note = {"name": "n", "note": "2 * 3 = 7", "reference": "[1]"}
    in_name = {"name": "2 * 3 = 7", "reference": "[1]"}
    assert _arithmetic(in_note) == [] and _arithmetic(in_name) == []
    assert [f["expression"] for f in _arithmetic({"name": "d", "derivation": "2 * 3 = 7"})] == ["2 * 3"]
