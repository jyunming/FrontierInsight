"""Every number the paper prints must trace to something the run computed.

The numeric oracle looks for a number sitting NEAR a result without matching
it, and deliberately ignores one that is far from every result. The statistics
claim check asks what a number IS. Neither asks where a number came from, and
a graded paper shipped through the gap: its Table 1 printed eight cells that
match no value the run computed, no replicate, and none of the three-seed
aggregates.

The decisive fact these tests are built around is that a paper does NOT print
the contents of RESULT_JSON — it prints the MEAN OVER THE SEEDS and that
mean's interval, neither of which any single seed recorded. Measured over that
paper, checking against RESULT_JSON alone makes all eleven of its CORRECT
headline numbers untraceable too. So the seed aggregates are part of what a
number may trace to, and every guard below exists to keep the check silent on
prose that is fine.

Each guard comes as a fire / don't-fire pair. The don't-fire half is the one
that matters: a false positive costs a whole rewrite, and FI has already paid
for checks that cried wolf.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.engine import (
    Engine,
    _TEXT_ONLY_HITS,
    _hit_name,
    _hits_need_only_a_rewrite,
    _review_sends_the_experiment_back,
)
from core.number_provenance import _quantile, check

# A small stand-in for a real run: seed 0's RESULT_JSON.
RESULTS = {
    "deterministic_final_size": 0.5828,
    "by_N": {
        "100": {"outbreak_probability": 0.3848, "conditioned_mean": 0.5600},
        "1000": {"outbreak_probability": 0.3300, "conditioned_mean": 0.5538},
    },
    "final_sizes": [0.11, 0.22, 0.33],
}

# What the engine computes ACROSS the seeds and hands to the paper. 0.3337 and
# its bounds appear nowhere in RESULTS: this is the aggregate a paper reports.
INTERVALS = {
    "by_N.1000.outbreak_probability": {
        "mean": 0.3337, "ci_lower": 0.3126, "ci_upper": 0.3548,
    },
}
AGGREGATE = {
    "by_N.1000.outbreak_probability": {
        "mean": 0.3337, "std": 0.0075, "n": 3, "min": 0.3260, "max": 0.3412,
    },
}


def _tokens(report) -> list[str]:
    return [f.token for f in report.findings]


def _check(text: str, **kw):
    kw.setdefault("result_json", RESULTS)
    kw.setdefault("intervals", INTERVALS)
    kw.setdefault("aggregate", AGGREGATE)
    return check(text, **kw)


# --- the defect this check exists for -----------------------------------------

def test_a_number_the_run_never_computed_is_flagged() -> None:
    """The observed Table 1 cell. 0.7093 matches no result, no replicate and
    no aggregate; the run's own value for that cell was 0.5600."""
    report = _check("| N = 100 | 0.2974 (0.2721 to 0.3227) | 0.7093 (0.6934 to 0.7252) |")
    assert "0.7093" in _tokens(report)
    msg = report.findings[0].message
    assert "nothing in this run accounts for it" in msg


def test_the_finding_names_the_number_and_not_a_result_path() -> None:
    """There IS no result path — that is the point of the check. The message
    must therefore not invent one."""
    report = _check("The conditioned mean was 0.7093 in that cell.")
    assert report.findings[0].value == pytest.approx(0.7093)
    assert report.findings[0].kind == "untraceable_number"


# --- a value the run DID compute ----------------------------------------------

def test_a_value_in_the_results_is_not_flagged() -> None:
    report = _check("The outbreak probability was 0.3848 at N = 100.")
    assert _tokens(report) == []


def test_rounding_to_the_papers_own_precision_is_accepted() -> None:
    """A computed 0.5828 printed as 0.583 is the same number."""
    for written in ("0.583", "0.5828"):
        report = _check(f"The deterministic final size is {written}.")
        assert _tokens(report) == [], written


def test_a_fraction_printed_as_a_percentage_is_accepted() -> None:
    """0.3848 in the results, "38.48%" in the prose."""
    report = _check("The outbreak probability was 38.48% at N = 100.")
    assert _tokens(report) == []


def test_a_complement_is_accepted() -> None:
    """A paper may print 1 - p for a probability p the run computed."""
    report = _check("The epidemic died out in 0.6152 of runs.")
    assert _tokens(report) == []


def test_truncation_is_accepted_as_well_as_rounding() -> None:
    """0.5828... written 0.582 is a truncation, not a number from nowhere.
    A rounding comparison alone would call it untraceable."""
    report = _check("The deterministic final size is 0.582.")
    assert _tokens(report) == []


# --- the aggregate over the seeds: without this the check flags correct prose --

def test_the_mean_over_the_seeds_is_what_a_paper_reports() -> None:
    """0.3337 is in no replicate and no RESULT_JSON — only in the aggregate.
    This is the single fact that makes the check usable rather than a machine
    for flagging correct papers."""
    report = _check("The probability stabilised at 0.3337 as N grew.")
    assert _tokens(report) == []


def test_a_confidence_bound_is_accepted() -> None:
    report = _check("The probability was 0.3337 (95% CI 0.3126 to 0.3548).")
    assert _tokens(report) == []


def test_a_standard_error_is_accepted() -> None:
    """A paper prints "SE 0.0043" beside a mean; that is the SD over root n,
    which the aggregate carries only as a standard deviation."""
    se = 0.0075 / (3 ** 0.5)
    report = _check(f"The probability was 0.3337 (SE {se:.4f}, n = 3).")
    assert _tokens(report) == []


def test_without_the_seed_aggregates_correct_prose_would_be_flagged() -> None:
    """The measurement that justifies including them: the SAME sentence, with
    the aggregates withheld, is flagged. If this ever stops being true the
    aggregates have stopped mattering and the guard can be reconsidered."""
    report = check(
        "The probability stabilised at 0.3337 as N grew.", result_json=RESULTS,
    )
    assert "0.3337" in _tokens(report)


# --- the dominant false positive: a range is not a negative number ------------

def test_a_range_written_with_a_dash_is_not_a_negative_number() -> None:
    """"95% CI 0.3126--0.3548" tokenizes as -0.3548, a value no run computes.
    Measured over the graded papers this one artifact produced the majority of
    all flags, in six runs whose numbers were fine."""
    for dash in ("--", "-", "–", "—"):
        report = _check(f"The probability was 0.3337 (95% CI 0.3126{dash}0.3548).")
        assert _tokens(report) == [], dash


def test_a_genuine_negative_number_still_reads_as_negative() -> None:
    """The range guard keys on a DIGIT before the dash, so a negative value
    introduced by an operator keeps its sign and is still checked."""
    report = _check("The effect size was d = -240.55 for that contrast.")
    assert "-240.55" in _tokens(report)


# --- a number is read the way a reader reads it -------------------------------
#
# Sixteen numbers the check flagged across the stored papers were the number
# tokenizer's reading and not the paper's: a typeset minus dropped, a range dash
# taken for a sign, a comma list taken for one number, an identifier and a year
# taken for results. Each rule below is tested on the sentence it was found in,
# and each comes with the control that must STILL be flagged, because a rule that
# stopped flagging real numbers would be worse than the artifact it removed.

MINUS = chr(0x2212)  # U+2212, spelt as a code point so it is not mistaken for "-"
EM_DASH = chr(0x2014)

# The values the real sentences below refer to. Every one of the signed values is
# negative in the run: the paper's sign is right, and a reading that drops it
# reports a correct number as one the run never produced.
SIGNED = {
    "no_opc_pullback_k130": -38.61,
    "rule_opc_pullback_k130": -44.61,
    "rule_opc_pullback_k160": -51.61,
    "effective_gain_pct": -74.5868,
    "conditional_minus_deterministic": -0.0084,
    "analytic_monte_carlo_percent_error": {"CAR": 85.522718, "MOR": 68.108593},
    "gap": {"high": 0.5936, "low": 0.4797},
    "loss_pct": 20.55,
    "n_major": [107, 104, 94],
    "cell": {"n_major": 106, "mean": 0.353, "lo": 0.301, "hi": 0.409},
}


def _signed(text: str, **kw):
    kw.setdefault("result_json", SIGNED)
    kw.setdefault("intervals", None)
    kw.setdefault("aggregate", None)
    return check(text, **kw)


def _read(text: str) -> list[tuple[float, str]]:
    from core.number_provenance import paper_numbers

    return [(value, token) for value, token, _ctx in paper_numbers(text)]


@pytest.mark.parametrize("sentence", [
    f"The no-OPC condition produced a constant pullback of {MINUS}38.61 nm across "
    "all k1 values from 0.30 to 0.60.",
    "The rule-based OPC condition produced even larger pullback than no-OPC, "
    f"ranging from {MINUS}44.61 nm (k1 = 0.30) to {MINUS}51.61 nm (k1 = 0.60).",
    f"- NA=0.55 produced larger mean printed CD (reported effective_gain_pct = {MINUS}74.5868).",
    f"| 1.5 | 1,000 | 106/300 | 0.353 (0.301 to 0.409) | {MINUS}0.0084 |",
])
def test_a_typeset_minus_is_one_number_with_its_sign(sentence: str) -> None:
    """The observed sentences. U+2212 was read as no sign at all, so a U+2212 before 38.61 was
    checked as 38.61 and reported as a number the run never computed, when the
    run holds -38.61."""
    assert _tokens(_signed(sentence)) == []


def test_a_typeset_minus_is_read_with_its_sign() -> None:
    assert _read(f"a pullback of {MINUS}38.61 nm") == [(-38.61, "-38.61")]


def test_the_sign_is_compared_in_both_directions() -> None:
    """Controls. A number printed without the run's sign, and one printed with a
    sign the run does not have, are both still numbers the run never produced."""
    assert "38.61" in _tokens(_signed("The pullback was 38.61 nm."))
    only_positive = {"no_opc_pullback_k130": 38.61}
    report = _signed(f"The pullback was {MINUS}38.61 nm.", result_json=only_positive)
    assert _tokens(report) == ["-38.61"]


def test_a_number_beside_a_typeset_minus_is_still_checked() -> None:
    """Control: a fabricated value next to a real one, both signed. Only the
    fabricated one is reported."""
    report = _signed(
        f"The pullback went from {MINUS}38.61 nm to {MINUS}77.77 nm as k1 grew."
    )
    assert _tokens(report) == ["-77.77"]


@pytest.mark.parametrize("sentence", [
    "a pullback of (-38.61 nm) across k1",
    "a pullback = -38.61 nm across k1",
    "a pullback of -38.61 nm across k1",
])
def test_a_hyphen_minus_after_a_bracket_an_equals_or_a_space_is_a_sign(
    sentence: str,
) -> None:
    """The ASCII forms were already read with their sign; pinned so the range
    rewrite next to them can never take one for a range."""
    assert _read(sentence) == [(-38.61, "-38.61")]
    assert _tokens(_signed(sentence)) == []


@pytest.mark.parametrize("spacing", [" ", ""])
def test_a_typeset_minus_after_a_number_is_a_difference_not_a_sign(spacing: str) -> None:
    """A U+2212 between two numbers is a difference, exactly as a hyphen is, and
    the minus is folded BEFORE the range dash is looked for so that it is read
    the same way. Folded after, a U+2212 stuck to the second number would invent
    -0.4797 (a number the run does not hold) where the hyphen, and the reading
    before the fold, both gave 0.4797."""
    text = f"The two levels are 0.5936 {MINUS}{spacing}0.4797 apart."
    assert _read(text) == [(0.5936, "0.5936"), (0.4797, "0.4797")]
    assert _tokens(_signed(text)) == []


def test_an_em_dash_before_a_number_is_punctuation_not_a_minus() -> None:
    """Control for what is deliberately NOT folded: the dash in "the loss
    — 20.55% at N = 100 — is small" is punctuation, and reading it as a minus
    would report the run's own 20.55 as a value it does not hold."""
    text = f"The loss {EM_DASH}20.55% at N = 100{EM_DASH} is small."
    assert _read(text) == [(20.55, "20.55")]
    assert _tokens(_signed(text)) == []


def test_a_design_and_a_paper_that_both_print_a_typeset_minus_agree() -> None:
    """The design's own text is read the same way. Folded on one side only, a
    declared -18.5 would be read as +18.5 and the paper's -18.5 reported."""
    design = {"method": f"Evaluate the slope at x = {MINUS}18.5 nm."}
    text = f"The slope was taken at {MINUS}18.5 nm."
    assert _tokens(_signed(text, design=design)) == []
    assert "-18.5" in _tokens(_signed(text))


def test_a_range_after_a_percent_sign_is_not_a_negative_number() -> None:
    """"sat 68.108593%-85.522718% below" — the dash after the percent sign
    ended a number, and 85.522718 was read as -85.522718. The run holds
    +85.522718. The sign of "-0.175906" in the same sentence follows a space
    and is kept."""
    text = (
        "The Monte Carlo estimator produced much flatter exponents, -0.175906 "
        "for CAR, and sat 68.108593%-85.522718% below the analytic values."
    )
    assert [v for v, _ in _read(text)] == [-0.175906, 68.108593, 85.522718]
    assert "85.522718" not in _tokens(_signed(text))
    assert "-0.175906" in _tokens(_signed(text)), "a sign after a space stays a sign"


def test_the_upper_bound_of_a_percent_range_is_still_checked() -> None:
    """Control: the range is read as two numbers, and a bound the run does not
    hold is reported as itself, not as its negation."""
    report = _signed("The estimator sat 68.108593%-85.999999% below the analytic values.")
    assert _tokens(report) == ["85.999999"]


@pytest.mark.parametrize("held", ["-18.5", "18.5"])
@pytest.mark.parametrize("sign", ["\\pm ", "\xb1", "\xb1 ", "+/-", "+/- ", "+-"])
def test_a_number_after_a_plus_minus_sign_is_either_sign(sign: str, held: str) -> None:
    """"nominal dark-line edges, ±18.5 nm from line center" names two edges.
    The run holds -18.5 (the design states it) and 18.5 was reported. It must
    clear whichever sign the run holds: written "+/-18.5" the tokenizer itself
    reads a minus, which only clears a run that holds the negative."""
    design = {"method": f"The target-CD edge sits at x = {held} nm."}
    text = f"NILS was computed at the nominal edges, \\({sign}18.5\\) nm from line center."
    assert _tokens(_signed(text, design=design)) == []


def test_a_plus_minus_number_the_run_does_not_hold_is_still_flagged() -> None:
    """Control: ±22.75 traces at neither sign. And a plain 18.5 with no
    plus-minus in front of it still needs the run to hold +18.5."""
    design = {"method": "The target-CD edge sits at x = -18.5 nm."}
    assert "22.75" in _tokens(_signed("edges at \xb122.75 nm", design=design))
    assert "18.5" in _tokens(_signed("the edge sat at 18.5 nm", design=design))


def test_a_comma_list_with_a_short_group_is_a_list_not_a_thousands_number() -> None:
    """"n_major=107,104,94": the tokenizer read the head, 107,104, as 107104
    and then 94. A separator is followed by exactly three digits, so this is a
    list. The run holds 107 and 104."""
    text = "with \\(n_{\\mathrm{major}}=107,104,94\\)."
    assert [v for v, _ in _read(text)] == [107.0, 104.0, 94.0]
    assert _tokens(_signed(text)) == []


def test_a_list_with_a_fabricated_member_is_still_flagged() -> None:
    """Control: after the split the members are checked one by one, so the
    invented 105 is reported as itself and 104 is not."""
    report = _signed("with \\(n_{major}=105,104,94\\).")
    assert _tokens(report) == ["105"]


def test_a_thousands_separator_is_still_a_thousands_separator() -> None:
    """1,000 is 1000; 1,000, 2,000 is two numbers; 3, 4, 5 is three. None of
    the readings the tokenizer already had is touched."""
    assert _read("with 1,250 realizations") == [(1250.0, "1,250")]
    assert _read("cells of 1,250, 2,250 runs") == [(1250.0, "1,250"), (2250.0, "2,250")]
    assert _read("N in {105, 204, 307}") == [(105.0, "105"), (204.0, "204"), (307.0, "307")]
    assert _read("1,000,000 elements") == []  # one number, of one significant digit
    assert _read("10,793,366 sales") == [(10793366.0, "10,793,366")]


def test_a_run_of_three_digit_groups_is_left_as_one_number() -> None:
    """Known limit, pinned. "n_major=202,206,189" is a list, but its digits are
    a well-formed thousands run and nothing in the text says otherwise, so it is
    still one number, and still flagged when the run does not hold 202206189."""
    text = "with \\(n_{major}=202,206,189\\)."
    assert _read(text) == [(202206189.0, "202,206,189")]
    assert _tokens(_signed(text)) == ["202,206,189"]


@pytest.mark.parametrize("gap", [" ", chr(0xA0)])
def test_an_identifier_after_an_acronym_is_not_a_result(gap: str) -> None:
    """"emphasised in the SPIE 11609 review" cites a proceedings volume. The
    gap is a space or a no-break space, which is what a typeset paper has."""
    text = (
        "This is consistent with the low-k1 imaging-enhancement pillar emphasised "
        f"in the SPIE{gap}11609 review [6]."
    )
    assert _read(text) == []
    assert _tokens(_signed(text)) == []


@pytest.mark.parametrize("sentence", [
    "The review reports 11609 runs.",
    "The SPIE review reports 11609 runs.",
    "AUC 0.7093 in that cell.",
    "Set N = 12345 for the sweep.",
    "XPS 193 eV peaks were used.",
])
def test_a_number_that_only_resembles_an_identifier_is_still_checked(
    sentence: str,
) -> None:
    """Controls: no acronym, an acronym before a decimal, one capital and an
    operator, an acronym before a three-digit integer. Each is still a number
    the run never produced."""
    assert _tokens(_signed(sentence)) != [], sentence


@pytest.mark.parametrize("apostrophe", ["'", chr(0x2019)])
def test_a_year_in_a_possessive_attribution_is_not_a_result(apostrophe: str) -> None:
    """"Ernst Abbe's 1873 description of coherent image formation": 1873 is
    outside the 1900-2099 year guard. Straight or curly apostrophe."""
    text = f"Ernst Abbe{apostrophe}s 1873 description of coherent image formation."
    assert _read(text) == []
    assert _tokens(_signed(text)) == []


@pytest.mark.parametrize("sentence", [
    "The grid held 1873 samples.",
    "The model's 1873 samples were drawn once.",
    "Control (1873) and treated (1902) groups differed.",
    "Abbe's 18730 samples.",
    "Sweden's 2431 registrations were counted.",
])
def test_a_count_in_the_same_range_is_still_checked(sentence: str) -> None:
    """Controls: a count is recognised as a count by what stands before it. No
    name, a lower-case noun, a parenthesised label, a five-digit number, and a
    possessive name before a number that is not in the years a work is dated."""
    assert _tokens(_signed(sentence)) != [], sentence


# --- numbers that are not measurements ----------------------------------------

def test_a_year_is_not_a_measurement() -> None:
    report = _check("The model follows Kermack and McKendrick 1927 throughout.")
    assert _tokens(report) == []


def test_a_quantity_written_with_a_thousands_separator_is_not_a_year() -> None:
    """"1,920" is 1920 of something, not the year 1920: it was skipped as a year and so never checked, while the
    run's own value was 1849. A year is written without a separator."""
    result = {"cells": {"a": {"spread": 1849.0, "n": 12.0}}, "per_trial": [1900.0 + i * 0.7 for i in range(400)]}
    wrong = check("The spread fell from 1,920 to 12.0 units.", result_json=result)
    assert _tokens(wrong) == ["1,920"]
    right = check("The spread fell from 1,849 to 12.0 units.", result_json=result)
    assert _tokens(right) == []
    assert _tokens(check("The model follows Kermack and McKendrick 1927 throughout.", result_json=result)) == []


def test_a_version_number_is_not_a_measurement() -> None:
    """"numpy 1.11.4" would otherwise be read as the 3-significant-digit
    measurement 1.11."""
    report = _check("Simulations used numpy 1.11.4 and scipy 1.14.1.")
    assert _tokens(report) == []


def test_a_random_seed_is_not_a_measurement() -> None:
    """A seed is chosen in the code and computed by nothing, so it appears in
    no result. This was the only number flagged in a paper whose numbers the
    grader passed."""
    report = _check("Child streams came from SeedSequence(20260916) per cell.")
    assert _tokens(report) == []


def test_a_citation_marker_is_not_a_measurement() -> None:
    report = _check("This follows from the branching approximation [6, 7].")
    assert _tokens(report) == []


def test_a_figure_or_section_reference_is_not_a_measurement() -> None:
    report = _check("Figure 3.14 and Section 2.11 present the comparison.")
    assert _tokens(report) == []


def test_a_low_precision_number_is_never_asked_to_trace() -> None:
    """The load-bearing guard. "300 runs", "3 seeds", "R0 = 1.5" and "95% CI"
    are rhetorical, conventional or stratum labels; requiring them to trace
    anywhere is how a check starts flagging ordinary sentences."""
    report = _check(
        "We ran 300 realizations across 3 seeds for R0 = 1.5 and N = 5000, "
        "reporting a 95% CI at the 0.05 level over 2 regimes."
    )
    assert _tokens(report) == []


def test_a_low_precision_number_written_to_four_decimals_is_still_checked() -> None:
    """0.0068 carries two significant digits but four decimals: the run either
    computed it or it did not. The real defect included four such bounds."""
    report = _check("The minor-extinction size was 0.0076 (0.0068 to 0.0084).")
    assert "0.0068" in _tokens(report)


# --- derivations --------------------------------------------------------------

def test_arithmetic_the_paper_writes_out_is_accepted() -> None:
    """A paper showing its working prints intermediate values that exist
    nowhere in the results and should not: the derivation is on the page."""
    report = _check(
        "This gives 0.3848 * 0.5600 = 0.2155 for the pooled contribution."
    )
    assert _tokens(report) == []


def test_a_table_row_difference_between_two_results_is_accepted() -> None:
    """A table whose last column is a difference of two columns beside it.
    There is no operator, so both operands must themselves be real results."""
    report = _check("| 1.5 | 100 | 0.5828 | 0.5600 | 0.0228 |")
    assert _tokens(report) == []


def test_two_numbers_that_are_not_results_do_not_explain_a_third() -> None:
    """The strict half of the derivation rule. Without an operator on the page
    the operands must be values the run computed — otherwise any three numbers
    standing near each other would explain away a fourth and the check would
    go blind."""
    report = _check("| 0.7093 | 0.1111 | 0.8204 |")
    assert "0.8204" in _tokens(report)


# --- provenance outside the results -------------------------------------------

def test_a_number_the_user_set_in_the_configuration_is_accepted() -> None:
    """A population size or replicate count comes from the run config, not
    from anything the experiment computed.

    This test said 2700 first, and that made it **vacuous**: 2700 carries two
    significant digits, so the precision floor dropped it before ``config``
    was ever consulted and the assertion held with no config at all. 1250
    survives the floor, so the configuration really is what clears it — which
    the first assertion below now proves.
    """
    text = "Each cell used 1250 independent realizations."
    assert "1250" in _tokens(_check(text)), "without the config this must fire"
    report = _check(text, config={"topic": "Run 1250 realizations per setting."})
    assert _tokens(report) == []


def test_a_reference_value_the_design_declared_is_accepted() -> None:
    """One graded paper prints "as design-time reference values, tau_det =
    0.5828" for a constant the design named and the run never re-derived."""
    text = "As a design-time reference value, tau_det = 0.9405 for R0 = 3.0."
    assert "0.9405" in _tokens(_check(text)), "without the design this must fire"
    report = _check(
        text, design={"method": "Compare against tau_det = 0.9405 at R0 = 3.0."},
    )
    assert _tokens(report) == []


def test_a_figures_own_numbers_are_accepted() -> None:
    """An axis limit printed as a count IS a defect, but stat_claims recomputes
    and names it. Flagging it here too would report one mistake twice."""
    report = _check(
        "The modal bin holds 231 of the runs.",
        figure_records={"fig2.png": {"axes": [{"ylim": [0, 231.0]}]}},
    )
    assert _tokens(report) == []


def test_an_effect_size_from_the_comparison_aggregate_is_accepted() -> None:
    """Cohen's d is computed FOR the paper and is absent from RESULT_JSON, so
    it has nowhere else to trace to."""
    report = _check(
        "The contrast gave Cohen's d = -240.55 at N = 5000.",
        comparison_stats={"effect_sizes": [{"cohens_d": -240.55}]},
    )
    assert _tokens(report) == []


# --- a run with nothing to check against --------------------------------------

def test_a_run_that_recorded_no_results_goes_quiet() -> None:
    """A crashed run whose last RESULT_JSON is empty must not have every
    number in its paper flagged. Several existing engine tests drive the review
    node with an empty result_json and assert must_flag_hits == []."""
    report = check("The conditioned mean was 0.7093 in that cell.", result_json={})
    assert report.skipped and not report.findings
    assert report.skip_reason == "the run recorded no numeric results"


def test_an_empty_array_is_not_a_recorded_result() -> None:
    """The zero length of an empty array would otherwise count as "the run
    computed something" and re-open the whole paper to flagging."""
    report = check(
        "The conditioned mean was 0.7093.", result_json={"final_sizes": []},
        replicates=[],
    )
    assert report.skipped


def test_an_empty_paper_is_skipped() -> None:
    report = _check("   ")
    assert report.skipped and report.skip_reason == "paper text is empty"


def test_the_audit_counts_what_it_checked() -> None:
    report = _check("The conditioned mean was 0.7093 in that cell.")
    body = report.to_dict()
    assert body["paper_numbers_checked"] >= 1
    assert body["traceable_values"] > 0
    assert body["untraceable_numbers"] == 1
    assert body["findings"][0]["token"] == "0.7093"


def test_one_finding_per_distinct_value_and_the_list_is_capped() -> None:
    """A table of wrong cells must not bury every other review finding."""
    text = " ".join(f"value {0.7001 + i / 10000:.4f} here." for i in range(30))
    report = _check(text)
    assert len(report.findings) <= 8
    assert report.untraceable > len(report.findings)
    # ...but the writer must still be told how many more there are, or it
    # fixes the eight it can see and the next round flags the rest, which
    # spends the iteration budget a round at a time.
    assert "A further" in report.findings[-1].message


# --- wiring: the hit routes to a rewrite, never to a re-run -------------------

def test_the_hit_name_is_text_only() -> None:
    hit = "unsourced_number: the paper prints 0.7093, and nothing accounts for it"
    assert _hit_name(hit) == "unsourced_number"
    assert _hit_name(hit) in _TEXT_ONLY_HITS
    assert _hits_need_only_a_rewrite([hit])


def test_the_hit_does_not_send_the_experiment_back() -> None:
    """The experiment ran and recorded its results; the paper quotes a figure
    none of them holds. That is a problem with the prose, so the route is
    ``write`` and the experiment is left alone."""
    review = {
        "verdict": "accept",
        "must_flag_hits": [
            "unsourced_number: the paper prints 0.7093, and nothing in this "
            "run accounts for it"
        ],
        "weaknesses": [], "suggestions": [], "blocking": "",
    }
    state = {"code": "print(1)", "result_json": RESULTS}
    assert _review_sends_the_experiment_back(review, state) == ""  # type: ignore[arg-type]


def _route_engine():
    from types import SimpleNamespace

    return SimpleNamespace(
        config=SimpleNamespace(
            engine=SimpleNamespace(max_iterations=4, review_loop=False),
            pauses=SimpleNamespace(review="off"),
        ),
        _log=SimpleNamespace(info=lambda *a, **k: None),
    )


def test_the_route_is_rewrite_not_revise_or_re_execute() -> None:
    state = {
        "review": {"verdict": "accept", "must_flag_hits": [
            "unsourced_number: the paper prints 0.7093"]},
        "iteration": 0, "code": "print(1)",
    }
    assert Engine._route_after_review(_route_engine(), state) == "rewrite"  # type: ignore[arg-type]


def _review_engine(tmp_path: Path) -> Engine:
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
        ProviderConfig,
    )
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(clarify_mode="off", review_loop=True, max_iterations=2),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


@pytest.mark.parametrize("panel", [False, True])
def test_review_forces_the_hit_even_when_the_reviewer_accepts(
    tmp_path: Path, panel: bool,
) -> None:
    """Panel mode once skipped the numeric oracle entirely, so asking for more
    reviewers meant fewer checks. Both paths run this one."""
    import asyncio

    eng = _review_engine(tmp_path)
    if panel:
        eng.config.engine.review_panel = ["methodologist", "statistician"]

    async def accepting(prompt, *, node=None):  # noqa: ANN001, ARG001
        if node == "review_moderator":
            return json.dumps({"rationale": "both accept"})
        return json.dumps({"verdict": "accept", "score": 5,
                           "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text(
        "The conditioned mean final size was 0.7093 at N = 100.",
        encoding="utf-8",
    )
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "result_json": RESULTS, "figure_records": {},
    }))
    review = patch["review"]
    assert any(str(h).startswith("unsourced_number") for h in review["must_flag_hits"])
    assert patch["iteration"] == 1, "a forced hit spends an iteration"
    assert _review_sends_the_experiment_back(review, {"code": "x"}) == ""  # type: ignore[arg-type]
    assert eng._route_after_review(  # type: ignore[arg-type]
        {"review": review, "iteration": 1, "code": "x"}
    ) == "rewrite"


def test_the_audit_is_written_even_when_the_paper_is_clean(tmp_path: Path) -> None:
    """A clean run must leave evidence the check RAN, so silence never has to
    be read as "it was skipped"."""
    import asyncio

    eng = _review_engine(tmp_path)

    async def accepting(prompt, *, node=None):  # noqa: ANN001, ARG001
        return json.dumps({"verdict": "accept", "score": 5,
                           "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("The outbreak probability was 0.3848 at N = 100.",
                     encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "result_json": RESULTS, "figure_records": {},
    }))
    assert patch["review"]["must_flag_hits"] == []
    audit = tmp_path / "paper" / "provenance_audit.json"
    assert audit.is_file()
    assert json.loads(audit.read_text(encoding="utf-8"))["ok"] is True


# --- powers of ten and the engine's oracle checks ------------------------------------------------------------------

@pytest.mark.parametrize("written", ["1.25×10⁻⁵", "1.25×10^-5", "1.25 × 10^{-5}", "1.25·10⁻⁵"])
def test_a_power_of_ten_is_read_the_way_a_reader_reads_it(written: str) -> None:
    """A live paper's "1.25×10⁻⁵" was read as 1.25 and flagged as a number the run never computed."""
    from core.number_provenance import paper_numbers

    assert [v for v, _t, _c in paper_numbers(f"the drift was {written} here")] == [pytest.approx(1.25e-05)]


def test_the_live_papers_power_of_ten_is_not_flagged() -> None:
    line = "the undamped energy-conservation oracle failed marginally (1.25×10⁻⁵ vs tolerance 1×10⁻⁶)."
    assert check(line, result_json={"energy_drift": 1.249999926e-05, "other": 0.731}).findings == []


def test_a_number_the_engines_oracle_check_measured_is_accounted_for() -> None:
    """The oracle pre-check's values are numbers this run computed, though not in RESULT_JSON."""
    line = "The grid-refinement ratio was 2.0412 before the main run."
    judged = [{"name": "self convergence", "value": 2.0412473807, "expected": 2, "limit": 0.2, "passed_by_engine": True}]
    assert [f.token for f in check(line, result_json={"other": 0.731}).findings] == ["2.0412"]
    assert check(line, result_json={"other": 0.731}, oracle_checks=judged).findings == []


def test_a_comma_list_of_three_digit_values_the_run_holds_is_a_list_not_one_number() -> None:
    """A real paper's grid "N in 100,250,500" was read as 100250500 and flagged; the rewrite then cut the grid."""
    paper = "We ran N in 100,250,500 and saw a peak of 0.4121. A total of 1,050 runs."
    held = check(paper, result_json={"N_grid": [100, 250, 500], "peak": 0.4121, "total": 1050})
    assert held.untraceable == 0
    # A member the run does not hold keeps the whole run flagged, and a leading-zero group is a thousands group.
    missing = check(paper, result_json={"N_grid": [100, 500], "peak": 0.4121, "total": 1050})
    assert [f.token for f in missing.findings] == ["100,250,500"]
    assert check("A total of 1,050 runs, peak 0.4121.",
                                   result_json={"a": [1, 50], "peak": 0.4121}).untraceable == 1


def test_a_comma_list_whose_first_member_is_negative_is_still_a_list() -> None:
    assert check("Offsets -100,250,500 and peak 0.4121.",
                 result_json={"o": [-100, 250, 500], "peak": 0.4121}).untraceable == 0


def test_two_signs_in_front_of_a_comma_run_are_not_a_list_member() -> None:
    from core.number_provenance import _a_list_the_run_holds

    class Held:
        def find(self, value: float, _decimals: int) -> float:
            return value

    assert _a_list_the_run_holds("-100,250", Held()) and not _a_list_the_run_holds("+-100,250", Held())


# --- a number rounded to significant figures, and a change of unit prefix -----

SD_RUN = {"ea_sd_j_mol": 2326.660954282277}


@pytest.mark.parametrize("written", ["2330", "2326.66", "2327"])
def test_an_integer_rounded_to_significant_figures_is_the_value_it_was_rounded_from(written: str) -> None:
    """2330 (+/- 82) J/mol is 2326.66 to three significant digits; compared at the units it read as 2327 != 2330."""
    assert _tokens(check(f"The SD of Ea was {written} J/mol.", result_json=SD_RUN)) == []


def test_an_integer_rounded_to_the_tens_still_has_to_round_from_the_run_value() -> None:
    """2316.0 rounds to 2320 at the tens, not 2330: the check is as strict as the precision the paper wrote."""
    assert _tokens(check("The SD of Ea was 2330 J/mol.", result_json={"ea_sd_j_mol": 2316.0})) == ["2330"]


def test_trailing_zeros_compare_at_the_hundreds_for_two_zeros() -> None:
    assert _tokens(check("It took 20300 steps.", result_json={"steps": 20326.0})) == []
    assert _tokens(check("It took 20300 steps.", result_json={"steps": 20246.0})) == ["20300"]


def test_a_number_with_one_significant_digit_is_still_not_checked() -> None:
    """"3000" carries no information about where it came from, as before: not a year, not hedged, not checked."""
    report = check("We drew 3000 samples.", result_json={"x": 1.5})
    assert _tokens(report) == [] and report.paper_numbers == 0


def test_a_unit_prefix_change_by_a_thousand_is_not_accepted() -> None:
    """2326.66 J/mol written 2.33 kJ/mol is deliberately flagged: accepting x/÷1000 raised coincidental matches."""
    assert _tokens(check("The SD of Ea was 2.33 kJ/mol.", result_json=SD_RUN)) == ["2.33"]
    assert _tokens(check("The SD of Ea was 2330 J/mol.", result_json={"ea_sd_kj_mol": 2.32666})) == ["2330"]


def test_a_rounded_integer_is_never_scaled_up_by_a_hundred() -> None:
    """2330 x 100 must not match an unrelated 233,450 in the run."""
    assert _tokens(check("The SD of Ea was 2330 J/mol.", result_json={"x": 233450.0})) == ["2330"]


def test_a_rounded_integer_is_not_scaled_down_at_a_coarser_precision() -> None:
    """233000 / 100 = 2330 must not be compared at the tens, where it would match 2326.66."""
    assert _tokens(check("The study took 233000 steps.", result_json=SD_RUN)) == ["233000"]
    assert _tokens(check("The SD of Ea was 2330 J/mol.", result_json={"x": 23.28})) == ["2330"]


def test_a_rounded_integer_is_not_matched_by_truncation() -> None:
    """Nobody truncates 2326.66 to 2320: 2320 is flagged (it would widen the window to [2315, 2330))."""
    assert _tokens(check("The SD of Ea was 2320 J/mol.", result_json=SD_RUN)) == ["2320"]

# --- raw trial values and luck: a match must mean something ---------------------

def _draws(n: int, mod: int = 1009, mul: int = 7919, lo: float = 0.1, span: float = 0.9) -> list[float]:
    """``n`` distinct-looking, unordered values in [lo, lo + span): a stand-in for raw per-trial draws."""
    return [lo + span * ((i * mul) % mod) / mod for i in range(n)]


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    mean = math.fsum(vals) / len(vals)
    return mean, math.sqrt(math.fsum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _raw_run(n: int = 400) -> dict:
    return {"summary_rate": 0.7391, "trial_values": _draws(n)}


def test_one_raw_trial_value_does_not_account_for_a_number() -> None:
    """A paper reports what was computed from trials, never one trial picked at random: equal to one of 400 draws is luck."""
    raw = _raw_run()
    pick = round(raw["trial_values"][57], 4)
    assert sum(1 for v in raw["trial_values"] if round(v, 4) == pick) == 1
    assert _tokens(check(f"The measured score was {pick}.", result_json=raw)) == [str(pick)]


def test_a_statistic_of_the_raw_trials_is_accounted_for() -> None:
    raw = _raw_run()
    vals = raw["trial_values"]
    mean, sd = _mean_sd(vals)
    ordered = sorted(vals)
    median = ordered[len(ordered) // 2 - 1] / 2 + ordered[len(ordered) // 2] / 2
    text = (f"The mean score was {mean:.4f} (SD {sd:.4f}), ranging from {ordered[0]:.4f} to {ordered[-1]:.4f}, "
            f"and the median was {median:.4f}.")
    assert _tokens(check(text, result_json=raw)) == []


def test_a_percentile_or_an_interval_the_run_did_not_store_is_not_made_up() -> None:
    """Only mean, median, SD, min and max are derived. A 95th percentile or a 1.96-SEM interval is a result when the
    run stored it, and is not a target the check invents (each invented target is another way to match by luck)."""
    vals = _raw_run()["trial_values"]
    ordered = sorted(vals)
    q95 = _quantile(ordered, 95.0)
    mean, sd = _mean_sd(vals)
    upper = mean + 1.96 * sd / math.sqrt(len(vals))
    run = _raw_run()
    for number in (round(q95, 4), round(upper, 4)):
        assert _tokens(check(f"The score reached {number}.", result_json=run)) == [str(number)]
    stored = {**run, "p95": round(q95, 4)}
    assert _tokens(check(f"The score reached {round(q95, 4)}.", result_json=stored)) == []


def test_a_derived_statistic_is_never_the_operand_of_a_derivation() -> None:
    """A table row whose last cell is the difference of two results is accepted when both are stored results; two
    statistics FI derived from raw trials are not stored results, so their difference is not explained away."""
    vals = _raw_run()["trial_values"]
    mean, _sd = _mean_sd(vals)
    low = min(vals)
    diff = mean - low
    row = f"| {mean:.4f} | {low:.4f} | {diff:.4f} |"
    assert _tokens(check(row, result_json=_raw_run())) == [f"{diff:.4f}"]
    stored = {"summary_rate": 0.7391, "a_result": round(mean, 4), "b_result": round(low, 4)}
    assert _tokens(check(row, result_json=stored)) == []


def test_every_element_of_a_short_list_is_still_a_result() -> None:
    """A sweep or a per-seed list is printed one entry at a time; only a long column is raw data."""
    short = {"summary_rate": 0.7391, "by_setting": _draws(150)}
    pick = round(short["by_setting"][17], 4)
    assert _tokens(check(f"The score at the eighteenth setting was {pick}.", result_json=short)) == []


def test_records_are_pooled_within_a_group_never_across_groups() -> None:
    """A treatment mean is the mean of the treatment records. Pooling every record into one column made it no one's
    statistic, and an honest subgroup mean was flagged."""
    rows = []
    for i in range(300):
        rows.append({"arm": "control", "y": 0.2 + 0.3 * ((i * 7919) % 1009) / 1009})
        rows.append({"arm": "treatment", "y": 0.6 + 0.3 * ((i * 4513) % 1013) / 1013})
    run = {"summary_rate": 0.7391, "rows": rows}
    treat = [r["y"] for r in rows if r["arm"] == "treatment"]
    control = [r["y"] for r in rows if r["arm"] == "control"]
    t_mean, c_mean = _mean_sd(treat)[0], _mean_sd(control)[0]
    text = f"The treatment arm averaged {t_mean:.4f} and the control arm {c_mean:.4f}."
    assert _tokens(check(text, result_json=run)) == []
    pick = round(treat[10], 4)
    assert _tokens(check(f"The treatment arm reached {pick}.", result_json=run)) == [str(pick)]


def _crowd(prefix: str) -> dict:
    """400 distinct values between 0.1 and 0.9: they take about 44% of the 900 numbers ``0.xyz`` can be, so a 3-digit
    number equals one of them on luck alone (chance > 0.2) and only the words around it can decide."""
    return {f"{prefix}_{i}": round(0.1 + i * 0.00203, 3) for i in range(400)}


def test_a_match_that_names_nothing_in_a_crowded_run_is_luck() -> None:
    crowd = _crowd("gain")
    value = crowd["gain_137"]
    assert _tokens(check(f"The gain was {value}.", result_json=crowd)) == []
    assert _tokens(check(f"The widget was {value}.", result_json=crowd)) == [str(value)]


def test_the_words_every_result_has_do_not_name_a_quantity() -> None:
    """``rate``, ``score``, ``error``, ``mean``(``s``), ``time`` appear in nearly every sentence and result name; a stop
    list applied to the 4-letter stem stops each with its inflections."""
    crowd = _crowd("error_rate")
    value = crowd["error_rate_137"]
    assert _tokens(check(f"The mean error rate scores were {value}.", result_json=crowd)) == [str(value)]
    crowd = _crowd("calibration_rate")
    value = crowd["calibration_rate_137"]
    assert _tokens(check(f"The calibration was {value}.", result_json=crowd)) == []


def test_only_the_sentence_of_the_number_names_it() -> None:
    crowd = _crowd("gain")
    value = crowd["gain_137"]
    assert _tokens(check(f"The gain was high. The widget was {value}.", result_json=crowd)) == [str(value)]
    assert _tokens(check(f"The widget gain was {value}. The other was fine.", result_json=crowd)) == []


def test_a_configured_or_declared_number_is_never_luck() -> None:
    crowd = _crowd("gain")
    value = crowd["gain_137"]
    assert _tokens(check(f"The widget was {value}.", result_json=crowd, config={"setting": value})) == []
    assert _tokens(check(f"The widget was {value}.", result_json=crowd, design={"parameters": {"x": value}})) == []


def test_luck_is_counted_per_sign_and_includes_truncation_keys() -> None:
    from core.number_provenance import Traceable

    negatives = Traceable({f"v{i}": -round(0.1 + i * 0.00203, 3) for i in range(400)})
    assert negatives.chance(-0.5, 3) > 0.2          # the crowd is negative
    assert negatives.chance(0.5, 3) == 0.0          # a positive number meets none of it
    plain = Traceable({f"v{i}": 0.1 + i * 0.0023 for i in range(200)})
    # 200 values round to 200 keys and truncate to 200 more: both are accepted matches, so both are counted
    assert plain.chance(0.5, 3) > len(set(round(v, 3) for v in plain.values.values())) / 900.0
    assert plain.chance(0.0004, 3) == 1.0           # a number that rounds to zero says nothing about its source


def test_a_long_list_with_a_non_number_in_it_does_not_break_the_walk() -> None:
    mixed = {"summary_rate": 0.7391, "trial_values": _draws(300) + [None, "n/a"], "also": [True] * 300}
    assert _tokens(check("The summary rate was 0.7391.", result_json=mixed)) == []


def test_a_million_raw_values_are_checked_in_seconds_and_correctly() -> None:
    import time

    n = 1_000_000
    vals = [((i * 7919) % 100003) / 100003 for i in range(n)]
    big = {"summary_rate": 0.7391, "trial_values": vals}
    mean, sd = _mean_sd(vals)
    text = f"The score was 0.7391, the mean {mean:.4f} and the SD {sd:.4f}."
    started = time.time()
    report = check(text, result_json=big)
    took = time.time() - started
    assert report.ok and report.paper_numbers == 3 and report.untraceable == 0
    assert took < 60
    assert report.traceable_values < 1000
    wrong = check("The score was 0.7391 and the mean 0.4123.", result_json=big)
    assert _tokens(wrong) == ["0.4123"]


# --- second review pass ---------------------------------------------------------

def test_a_long_column_with_few_values_keeps_them_and_a_fabricated_number_is_flagged() -> None:
    coded = {"summary_rate": 0.7391, "level": [0.1234] * 100 + [0.5678] * 100 + [0.9012] * 100}
    assert _tokens(check("The coded level 0.5678 was used.", result_json=coded)) == []
    assert _tokens(check("The coded level 0.5679 was used.", result_json=coded)) == ["0.5679"]


def test_a_long_sorted_column_does_not_keep_its_elements() -> None:
    """A cumulative sum, sorted errors or arrival times are in order and still the result of a computation over
    thousands of values: each element is not a target (and not an operand of a derivation)."""
    cumulative = {"summary_rate": 0.7391, "cumulative": [0.1 + i * 0.00371 for i in range(300)]}
    pick = round(cumulative["cumulative"][57], 4)  # 0.3115
    assert _tokens(check(f"The point at index 57 was {pick}.", result_json=cumulative)) == [str(pick)]
    mean, _sd = _mean_sd(cumulative["cumulative"])
    assert _tokens(check(f"The mean position was {mean:.4f}.", result_json=cumulative)) == []
    assert _tokens(check(f"The mean position was {mean + 0.0007:.4f}.", result_json=cumulative)) != []


def test_a_stratum_label_or_an_array_length_is_gated_like_any_result() -> None:
    """Only the configuration and the design are told to the run; a stratum label and a length are results, so in a
    crowded run an unnamed match to one is luck."""
    crowd = _crowd("gain")
    labelled = {**crowd, "by_alpha.0.713.metric": 1.5}
    assert labelled and _tokens(check("The widget was 0.713.", result_json=labelled)) == ["0.713"]


def test_a_group_field_holding_a_list_in_some_record_does_not_crash() -> None:
    rows = [{"arm": "a", "y": ((i * 7919) % 1009) / 1009} for i in range(300)]
    rows[5]["arm"] = ["a", "b"]
    rows[6]["arm"] = {"x": 1}
    run = {"summary_rate": 0.7391, "rows": rows}
    report = check("The summary rate was 0.7391 and a made-up value 0.1234 appeared.", result_json=run)
    assert not report.skipped and _tokens(report) == ["0.1234"]


def test_a_small_group_has_its_own_statistics() -> None:
    """Two arms of 150 records each (a long list, small groups): an arm mean is a result of that arm."""
    rows = []
    for i in range(150):
        rows.append({"arm": "control", "y": 0.2 + 0.3 * ((i * 7919) % 1009) / 1009})
        rows.append({"arm": "treatment", "y": 0.6 + 0.3 * ((i * 4513) % 1013) / 1013})
    run = {"summary_rate": 0.7391, "rows": rows}
    t_mean, t_sd = _mean_sd([r["y"] for r in rows if r["arm"] == "treatment"])
    c_mean = _mean_sd([r["y"] for r in rows if r["arm"] == "control"])[0]
    text = f"The treatment arm averaged {t_mean:.4f} (SD {t_sd:.4f}) and the control arm {c_mean:.4f}."
    assert _tokens(check(text, result_json=run)) == []
    assert _tokens(check(f"The treatment arm averaged {t_mean + 0.0011:.4f}.", result_json=run)) != []


def test_a_small_group_of_a_long_list_of_records_keeps_its_elements_and_a_fabrication_is_flagged() -> None:
    rows = [{"arm": "a", "y": 0.2 + 0.3 * ((i * 7919) % 1009) / 1009} for i in range(300)]
    rows += [{"arm": "b", "y": 0.7 + 0.01 * i} for i in range(5)]
    run = {"summary_rate": 0.7391, "rows": rows}
    assert _tokens(check("Arm b reached 0.7300 at its fourth record.", result_json=run)) == []
    assert _tokens(check("Arm b reached 0.7350 at its fourth record.", result_json=run)) == ["0.7350"]


def test_a_sentence_soft_wrapped_over_lines_is_read_whole() -> None:
    crowd = _crowd("gain")
    value = crowd["gain_137"]
    wrapped = f"The measured gain\nof the widget under\nstudy was\n{value} in the end."
    assert _tokens(check(wrapped, result_json=crowd)) == []
    separate = f"The gain was large.\n\nThe widget was {value}."
    assert _tokens(check(separate, result_json=crowd)) == [str(value)]
    row = f"The gain is tabulated below.\n| widget | {value} |"
    assert _tokens(check(row, result_json=crowd)) == [str(value)]


def test_a_statistic_whose_leaf_names_nothing_is_named_by_its_ancestor() -> None:
    """``calibration.error.@mean``: ``error`` names nothing, so the sentence has to be able to name it by ``calibration``."""
    run = {"calibration": {"error": _draws(400)}, **_crowd("gain")}
    mean, _sd = _mean_sd(run["calibration"]["error"])
    assert _tokens(check(f"The calibration averaged {mean:.3f}.", result_json=run)) == []
    assert _tokens(check(f"The widget averaged {mean:.3f}.", result_json=run)) == [f"{mean:.3f}"]


def test_two_hundred_thousand_records_are_pooled_correctly() -> None:
    import time

    rows = [{"arm": "a" if i % 2 else "b", "y": ((i * 7919) % 100003) / 100003, "z": (i * 31 % 977) / 977}
            for i in range(200_000)]
    run = {"summary_rate": 0.7391, "rows": rows}
    b_mean = _mean_sd([r["y"] for r in rows if r["arm"] == "b"])[0]
    started = time.time()
    report = check(f"The b arm averaged {b_mean:.4f}, the rate 0.7391.", result_json=run)
    assert time.time() - started < 60
    assert not report.skipped and report.ok and report.paper_numbers == 2 and report.traceable_values < 1000
    fabricated = check(f"The b arm averaged {b_mean + 0.0009:.4f}, the rate 0.7391.", result_json=run)
    assert not fabricated.skipped and _tokens(fabricated) == [f"{b_mean + 0.0009:.4f}"]


# --- rounding a value that sits exactly on a tie, a short name, a LaTeX separator -----------------------------------

_TIES = {"cells": {"a": {"coverage": {"estimate": 0.8125}}, "b": {"coverage": {"estimate": 0.9175}}}}


@pytest.mark.parametrize("written", ["0.813", "0.812", "0.8125"])
def test_a_tie_is_rounded_either_way(written: str) -> None:
    """325/400 = 0.8125 is 0.813 rounded half-up (as people do) and 0.812 half-even; both are roundings of it."""
    assert _tokens(check(f"OLS coverage was {written}.", result_json=_TIES)) == []


@pytest.mark.parametrize("written", ["0.918", "0.917", "0.9175"])
def test_a_tie_stored_just_below_in_binary_is_rounded_either_way(written: str) -> None:
    """0.9175 is stored as 0.91749999...: code prints 0.917, a person reads a tie and writes 0.918."""
    assert _tokens(check(f"Weighted coverage was {written}.", result_json=_TIES)) == []


@pytest.mark.parametrize("written", ["0.811", "0.814", "0.815", "0.916", "0.919"])
def test_a_number_beyond_half_a_unit_of_its_last_place_is_still_flagged(written: str) -> None:
    assert _tokens(check(f"Coverage was {written}.", result_json=_TIES)) == [written]


def test_a_derived_tie_is_rounded_either_way() -> None:
    run = {"hits": 325.0, "trials": 400.0}
    assert _tokens(check("325/400 = 0.813 coverage.", result_json=run)) == []
    assert _tokens(check("325/400 = 0.812 coverage.", result_json=run)) == []
    assert _tokens(check("325/400 = 0.815 coverage.", result_json=run)) == ["0.815"]


def test_a_two_letter_abbreviation_in_capitals_names_a_quantity() -> None:
    """709 is 709.449 rounded to the unit; in a run holding a crowd of 3-digit integers it is also a number a match
    would reach by luck, so the words decide. "SD" is the only word that says it is the spread; it is two letters."""
    crowd = {f"filler_{i}": float(101 + 3 * i) for i in range(220)}  # none is 709, 220 of the 900 numbers 1xx..9xx
    run = {"filler": crowd, "ols": {"Ea_estimate_sd_J_mol": {"estimate": 709.449}}, "start": {"Ea_sd": {"estimate": 1849.0}}}
    ok = check("At 8% noise the OLS SD declined from 1,849 to 709 units.", result_json=run)
    assert _tokens(ok) == []
    # without the name it is luck, and a wrong number is still flagged
    assert _tokens(check("At 8% noise the OLS spread declined from 1,849 to 709 units.", result_json=run)) == ["709"]
    assert _tokens(check("At 8% noise the OLS SD declined from 1,849 to 712 units.", result_json=run)) == ["712"]
    assert _tokens(check("At 8% noise the OLS SD declined from 1,920 to 709 units.", result_json=run)) == ["1,920"]


def test_a_latex_thousands_separator_is_one_number() -> None:
    run = {"truth": {"Ea_true": 24122.394677383}, "other": 1.5}
    assert _tokens(check(r"The target was \(E_a=24{,}122.394677\ \mathrm{J}\).", result_json=run)) == []
    assert _tokens(check(r"The target was \(E_a=24{,}199.394677\ \mathrm{J}\).", result_json=run)) == ["24,199.394677"]
