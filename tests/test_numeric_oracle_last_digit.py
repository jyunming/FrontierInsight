"""The numeric oracle's ``near_miss`` is a last-digit slip and nothing wider.

It used to be any number within a quarter of a result. A result dictionary holds
hundreds of numbers, so replayed over the 136 stored quests with results it made 197
findings in 59 quests (most 37), and most were not copying errors: bounds of an
interval, sums worked by hand, settings, siblings in a list of similar values. What
the stored quests do contain is the truncation or the mistyped last digit (2.12 for
2.1259, which rounds to 2.13; 0.184 for 0.18346, which rounds to 0.183), so that is
all ``near_miss`` reports now, and six rules keep the rest of the noise out. Each rule
has a control below that must still be reported.
"""

from __future__ import annotations

import pytest

from core import numeric_oracle as no


def _flagged(paper: str, results: dict, **kw) -> list[tuple[str, float]]:
    """``(kind, paper value)`` of every finding, for compact assertions."""
    return [(f.kind, f.paper_value) for f in no.check(paper, results, **kw).findings]


# --- what a last-digit slip is ----------------------------------------------


@pytest.mark.parametrize(
    "paper,result",
    [
        ("It fell to 2.12 nm.", 2.1259),  # truncated: 2.1259 rounds to 2.13
        ("It rose to 0.795.", 0.7938),  # one above 0.794
        ("The delta was 0.0019 here.", 0.00198056),  # truncated: rounds to 0.0020
        ("- LogisticRegression (baseline): ROC AUC = 0.9120.", 0.91205),  # rounds to 0.9121
        ("the MEEF is measured at 2.00 at best focus", 2.0053),  # rounds to 2.01
    ],
)
def test_a_number_one_last_digit_from_the_correct_rounding_is_reported(paper, result) -> None:
    got = _flagged(paper, {"m": result})
    assert [kind for kind, _ in got] == ["near_miss"], got


@pytest.mark.parametrize(
    "paper,result",
    [
        ("It fell to 2.11 nm.", 2.1259),  # two digits from 2.13
        ("Contrast reached 0.812 here.", 0.79),  # the old 25% band
        ("It fell to 2.2 nm.", 2.1259),  # one decimal: too coarse to be a slip
        ("There were 13 runs.", 12.4),  # an integer
    ],
)
def test_anything_wider_is_not_reported(paper, result) -> None:
    assert _flagged(paper, {"m": result}) == []


def test_a_finding_says_what_the_result_rounds_to() -> None:
    """``off by 0.2%`` reads as harmless; what a reader can act on is the rounding."""
    (finding,) = no.check("It fell to 2.12 nm.", {"meef": 2.1259}).findings
    assert "rounds to 2.13, not 2.12" in finding.describe()
    (zeros,) = no.check("The MEEF is 2.00 at best focus", {"meef": 2.0053}).findings
    assert "rounds to 2.01, not 2.00" in zeros.describe()
    (swapped,) = no.check("NILS reached 2.41", {"nils": 2.14}).findings
    assert "digits transposed" in swapped.describe()


# --- a slip inside an array is still a slip ---------------------------------


_SWEEP = {
    "coherence_sweep": {
        # the sweep axis holds 0.2 and 0.45: short numbers that must not clear a nearby one
        "sigma": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        "dipole_nils": [
            1.643130076445232e-14, 0.17089707604122265, 0.7886303334996332,
            0.6756091505287749, 0.4598949849860378, 0.3467047444093169,
            0.27560368925453693, 0.22234566737171177, 0.18346429992891075,
        ],
    }
}


def test_a_slip_of_one_point_of_a_nine_point_sweep_is_reported() -> None:
    """The case a stored quest contains: 0.18346 rounds to 0.183 and the paper says
    0.184, and the stored results hold a 0.2 beside it (the sweep axis). Leaving out
    the elements of an array lost this one, and so did letting the 0.2 clear it as a
    shorter rounding of 0.184."""
    report = no.check("causing NILS to drop to 0.184 at sigma = 0.45", _SWEEP)
    assert [(f.kind, f.paper_value, f.result_path) for f in report.findings] == [
        ("near_miss", 0.184, "coherence_sweep.dipole_nils[8]")
    ]


def test_a_transposition_of_a_point_of_an_array_is_reported() -> None:
    assert _flagged("the value was 0.45 there", {"a": [0.54, 0.9]}) == [("transposed", 0.45)]


# --- a citation number is not a measurement ---------------------------------


def test_a_number_in_a_citation_bracket_is_not_read() -> None:
    results = {"n_runs": 87.0}
    assert _flagged("as shown in [78] and in [3, 78].", results) == []
    # the same digits outside a bracket are a transposition
    assert _flagged("after 78 runs", results) == [("transposed", 78.0)]


# --- an identifier is not a result to slip from -----------------------------


def test_an_identifier_or_an_index_is_not_a_result() -> None:
    assert _flagged("the value was 7.50", {"trial_index": 7.4949}) == []
    assert _flagged("the value was 7.50", {"value": 7.4949}) == [("near_miss", 7.5)]
    assert not no._is_eligible_near_result("runs[3].seed")
    assert not no._is_eligible_near_result("meta.n_files")
    assert no._is_eligible_near_result("runs[3].nils")


# --- a result the paper also prints right -----------------------------------


def test_a_number_beside_a_result_the_paper_states_right_is_another_quantity() -> None:
    results = {"cutoff_k1": 0.3149}
    # 0.30 is one digit from 0.31, but the paper prints 0.31: it is a setting, a bound, a sibling
    assert _flagged("The cutoff was 0.31 at k_1 = 0.30.", results) == []
    # without the right value beside it, the 0.30 reads as the slip it may be
    assert _flagged("At k_1 = 0.30 the cutoff was reached.", results) == [("near_miss", 0.3)]


def test_a_coarser_statement_of_the_result_does_not_count() -> None:
    """``about 1.5`` is a summary of 1.4549, not a statement of it as 1.45."""
    assert _flagged("The mean was 1.46 (about 1.5).", {"m": 1.4549}) == [("near_miss", 1.46)]


# --- a repeating-decimal constant is a theoretical value --------------------


def test_a_repeating_decimal_constant_is_not_a_slip() -> None:
    results = {"final_size": 0.3339}
    assert _flagged("the deterministic limit is 0.333", results) == []
    assert _flagged("the mean was 0.335", results) == [("near_miss", 0.335)]


@pytest.mark.parametrize(
    "value,token,expected",
    [
        (0.333, "0.333", True), (0.667, "0.667", True), (0.167, "0.167", True),
        (0.143, "0.143", True), (0.33, "0.33", False),  # two decimals: could be a measurement
        (0.335, "0.335", False), (0.5, "0.500", False), (0.25, "0.250", False),
    ],
)
def test_which_numbers_are_repeating_constants(value, token, expected) -> None:
    assert no._is_repeating_constant(value, token) is expected


# --- a shorter stored rounding ----------------------------------------------


def test_a_stored_short_number_is_not_a_slip_of_the_longer_one() -> None:
    assert _flagged("the probability was 0.669", {"p": 0.67}) == []
    # a long stored number is a computation, not somebody's rounding
    assert _flagged("the probability was 0.669", {"p": 0.6704}) == [("near_miss", 0.669)]


def test_a_short_number_elsewhere_in_the_results_clears_nothing() -> None:
    """Only the result a number is read against can be its rounding: a stored 0.2 is
    a sigma or a threshold, and reading it as a rounding of every number in
    [0.15, 0.25) hid the slip in ``test_a_slip_of_one_point_of_a_nine_point_sweep``."""
    assert _flagged("the level was 0.184", {"sigma": 0.2, "nils": 0.18346}) == [("near_miss", 0.184)]


# --- a setting the run was given --------------------------------------------


def test_a_declared_setting_is_not_a_slip_but_a_declared_transposition_is() -> None:
    declared = [1.5]
    results = {"m": 1.4949}
    assert _flagged("At R_0 = 1.50 we converged.", results) == [("near_miss", 1.5)]
    assert _flagged("At R_0 = 1.50 we converged.", results, declared=declared) == []
    assert _flagged("NILS reached 2.41", {"nils": 2.14}, declared=[2.41]) == [("transposed", 2.41)]


# --- documented -------------------------------------------------------------


def test_the_rule_is_documented() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    assert "last-digit" in (repo / "docs" / "capabilities-reference.md").read_text(encoding="utf-8")
    assert "last-digit" in (repo / "docs" / "features.md").read_text(encoding="utf-8")
    assert no.MIN_SLIP_DECIMALS == 2
