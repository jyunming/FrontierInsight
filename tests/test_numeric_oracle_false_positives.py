"""The numeric oracle's false positives, and why they stopped forcing a rewrite.

A real quest produced five flags in one pass, every one of them spurious, and
they forced the non-bypassable revise path. The rewrite that followed left the
paper measurably worse than the draft it replaced: 11 of 11 claims grounded
became 9 of 11.

Two parser faults produced them, both pinned below:

* ``_CONTEXT_SKIP`` was anchored ``(?:...|doi|...)\\s*$``, so the guard matched
  ``DOI 10.5281`` but not ``DOI: 10.5281`` -- and every reference list writes
  the colon. Three DOI registrant prefixes were reported as contradicted
  measurements.
* LaTeX scientific notation was read as its mantissa. A paper stating
  ``1.21 \\times 10^{-2}`` was reported as "paper says 1.21", which then landed
  within 25% of an unrelated result value of 0.957 and was flagged.

The fixes remove this instance. The *exposure* is structural -- the check is a
regex over prose -- which is why the findings are now reported rather than
enforced.

Four more classes turned up when the check was replayed over every stored
quest (311 near-misses in 76 quests):

* a confidence LEVEL -- the ``95`` in ``95% CI`` -- reported against a count of
  93 or a population of 100 (``TestConfidenceLevel``);
* a SETTING the topic or the design gave the run -- ``R0 = 1.5`` printed beside
  an outcome of 1.33, a 13.5 nm wavelength, ``gamma = 1.0`` -- reported against
  whatever result happened to be within 25% (``TestDeclaredSettings``);
* a MINUS written as U+2212, read without its sign, so ``-0.114`` was compared
  as 0.114 (``TestUnicodeMinus``);
* ``97.5th`` read as ``97``, the integer part of a number nobody wrote
  (``TestPercentileRank``).

Each class has controls that must STILL be reported: the check is only worth
having if a paper that prints a result slightly wrong is still caught, so the
tests below pin the silence and the catch together.
"""

from __future__ import annotations

import pytest

from core import numeric_oracle as no
from core.numeric_oracle import extract_paper_numbers


def _values(text: str) -> list[float]:
    return [v for v, _tok, _ctx in extract_paper_numbers(text)]


def _flagged(paper: str, results: dict, **kw) -> list[tuple[str, float]]:
    """``(kind, paper value)`` of every finding, for compact assertions."""
    return [(f.kind, f.paper_value) for f in no.check(paper, results, **kw).findings]


# --- the skip guard vs. punctuation -----------------------------------------

@pytest.mark.parametrize("sep", ["", ":", ".", " -", " ="])
def test_doi_prefix_is_never_a_measurement(sep: str) -> None:
    assert _values(f"Source. DOI{sep} 10.5281/zenodo.20981306. Next.") == []


@pytest.mark.parametrize(
    "prefix", ["page:", "p.", "Figure:", "Table:", "Section:", "arXiv:", "ISBN:"],
)
def test_structural_references_are_skipped_with_a_separator(prefix: str) -> None:
    assert _values(f"see {prefix} 10.25 of the manual") == []


def test_the_three_dois_from_the_observed_run() -> None:
    """The exact strings that were flagged."""
    text = (
        "1. Numerical Integration. DOI: 10.1088/978-1-6817-4976-1ch4. "
        "2. Runge-Kutta 4th Order. DOI: 10.1016/b978-1-78548-177-2.50012-2. "
        "3. Mass-Spring-Damper System. DOI: 10.5281/zenodo.20981306."
    )
    got = _values(text)
    assert not [v for v in got if 10.0 <= v < 10.6], f"DOI prefixes leaked: {got}"


# --- LaTeX scientific notation ----------------------------------------------

def test_latex_times_notation_keeps_its_exponent() -> None:
    assert _values(r"error of $9.65 \times 10^{-6}$ overall") == [
        pytest.approx(9.65e-6)
    ]


def test_the_observed_1_21_case() -> None:
    """`1.21 \\times 10^{-2}` must not be read as 1.21 -- that is the value
    that was compared against an energy drift of 0.957 and flagged at 20.9%."""
    text = r"Forward Euler produced errors of $2.89 \times 10^{-4}$ and $1.21 \times 10^{-2}$"
    got = _values(text)
    assert got == [pytest.approx(2.89e-4), pytest.approx(1.21e-2)]
    assert not any(abs(v - 1.21) < 1e-9 for v in got)


@pytest.mark.parametrize("op", [r"\times", r"\cdot", "×"])
def test_exponent_forms(op: str) -> None:
    assert _values(f"about ${{3.5}} {op} 10^{{3}} steps".replace("{3.5}", "3.5")) == [
        pytest.approx(3500.0)
    ]


def test_positive_exponent_and_braceless_forms() -> None:
    assert _values(r"$4.2 \times 10^{+3}$") == [pytest.approx(4200.0)]
    assert _values(r"$4.2 \times 10^3$") == [pytest.approx(4200.0)]


# --- the checks that must still fire ----------------------------------------

def test_an_ordinary_measurement_is_still_extracted() -> None:
    assert _values("The measured RMSE was 0.4217 across all runs.") == [
        pytest.approx(0.4217)
    ]


def test_plain_e_notation_still_works() -> None:
    assert _values("an error of 9.65e-6 overall") == [pytest.approx(9.65e-6)]


def test_a_number_after_an_unrelated_colon_is_kept() -> None:
    """The separator relaxation must not swallow real claims: only the listed
    structural keywords suppress, not every colon in the paper."""
    assert _values("Result: 0.4217 was observed.") == [pytest.approx(0.4217)]


# --- a confidence level is not a result -------------------------------------


class TestConfidenceLevel:
    """``95`` in ``95% CI`` is within 25% of every result between 72 and 126.

    It was the most frequent false finding across the stored quests: reported
    against a count of 93, a population of 100, an upper bound of 97.6.
    """

    @pytest.mark.parametrize(
        "phrase",
        [
            "the outbreak probability was low (95% CI 0.036 to 0.058)",
            "probabilities were reported with 95% Wilson intervals",
            "Wilson 95% intervals summarise each threshold",
            "a 95% bootstrap confidence interval was used",
            "a 95 % credible interval was used",
            r"the $95\%$ CI excluded zero",
            "the 95%-CI excluded zero",
        ],
    )
    def test_the_level_is_not_a_number_to_check(self, phrase: str) -> None:
        assert 95.0 not in _values(phrase)
        # 93 is a count the run computed; a level beside it is not a near-miss.
        assert _flagged(phrase, {"n_major": 93.0}) == []

    def test_the_bounds_of_the_interval_are_still_checked(self) -> None:
        """Only the level is cleared. The numbers the interval brackets are
        results, and a wrong one is still a near-miss."""
        got = _flagged("It was 0.347 (95% CI 0.32-0.37).", {"p": 0.3459})
        assert ("near_miss", 0.347) in got
        assert all(value != 95.0 for _kind, value in got)

    def test_a_percentage_result_is_not_a_level(self) -> None:
        """``87% of runs`` is a measurement: a percent sign alone clears
        nothing, so 87 against a stored 78 is still reported."""
        got = _flagged("87% of runs converged.", {"converged_pct": 78.0})
        assert ("transposed", 87.0) in got

    def test_empirical_coverage_is_not_a_level(self) -> None:
        """``94% of the intervals covered the truth`` is a coverage RESULT
        that merely names intervals; ``of`` between the two stops the level
        rule from reading it as a level."""
        text = "87% of the confidence intervals covered the true value."
        assert ("transposed", 87.0) in _flagged(text, {"coverage_pct": 78.0})


# --- a setting the run was given is not a result ----------------------------

_DESIGN = {
    "hypothesis": "the deterministic limit is 0.9403",
    "variables": {
        "independent": [
            "R0 (0.9, 1.5, 3.0)",
            "population size N (100, 1000, 5000)",
        ],
        "controls": ["recovery rate gamma = 1", "300 runs per setting"],
    },
    "method": "A final size of at least 10% of N counts as a major outbreak.",
    "expected_outcome": "predict a crossover near 28 nm and P(outbreak) about 0.58",
    "result_assertions": [{"path": "p", "min": 0.01, "max": 0.97}],
}


class TestDeclaredSettings:
    """``R0 = 1.5`` beside an outcome of 1.33 is the paper quoting its setup."""

    def test_the_setup_is_declared_and_the_predictions_are_not(self) -> None:
        got = set(
            no.declared_numbers(_DESIGN, "Compare models for R0 in {0.9, 1.5, 3.0} at 13.5 nm.")
        )
        # topic, variables (list strings), method; a percentage also as a fraction
        assert {0.9, 1.5, 3.0, 100.0, 1000.0, 5000.0, 300.0, 13.5, 10.0, 0.1} <= got
        # what the author expected, and the bounds on results, are not settings
        assert not ({28.0, 0.58, 0.9403, 0.97, 0.01} & got)

    def test_a_setting_beside_a_result_one_digit_off_is_not_a_near_miss(self) -> None:
        paper = "At R_0 = 1.50 the simulations converged."
        results = {"mean_final_size": 1.4949}  # rounds to 1.49, one digit from the printed 1.50
        # without the settings the 1.50 reads as a last-digit slip of 1.4949
        assert _flagged(paper, results) == [("near_miss", 1.5)]
        assert _flagged(paper, results, declared=no.declared_numbers(_DESIGN, "")) == []

    def test_the_topic_alone_declares_its_grid(self) -> None:
        declared = no.declared_numbers(None, "For R0 in {0.9, 1.5, 3.0} run 300 times.")
        assert _flagged("At R_0 = 1.50 we converged.", {"m": 1.4949}) == [("near_miss", 1.5)]
        assert _flagged("At R_0 = 1.50 we converged.", {"m": 1.4949}, declared=declared) == []

    def test_a_number_that_merely_resembles_a_setting_is_still_checked(self) -> None:
        """Declared 1.5 clears 1.5 and 1.50, not 1.46 -- the paper's own
        precision decides."""
        declared = no.declared_numbers(_DESIGN, "")
        assert ("near_miss", 1.46) in _flagged(
            "The mean was 1.46.", {"m": 1.4549}, declared=declared,
        )

    def test_a_prediction_printed_as_a_result_is_still_reported(self) -> None:
        """The design's ``expected_outcome`` says P(outbreak) about 0.58; the run
        measured 0.5749, which rounds to 0.57. A paper that prints the prediction
        as the finding is the mistake this check is for, so the design's
        predictions are not settings."""
        declared = no.declared_numbers(_DESIGN, "")
        got = _flagged(
            "The outbreak probability was 0.58.",
            {"p_outbreak": 0.5749},
            declared=declared,
        )
        assert ("near_miss", 0.58) in got

    def test_a_transposition_of_a_setting_is_still_reported(self) -> None:
        """Same digits in another order is the signal least likely to be a
        coincidence, so a declared number does not silence it."""
        declared = no.declared_numbers({"variables": {"controls": ["target NILS 2.41"]}}, "")
        assert _flagged("NILS reached 2.41.", {"nils": 2.14}, declared=declared) == [
            ("transposed", 2.41)
        ]

    def test_no_settings_means_no_change(self) -> None:
        assert _flagged("At R_0 = 1.50 we converged.", {"m": 1.4949}, declared=None) == [
            ("near_miss", 1.5)
        ]
        assert _flagged("At R_0 = 1.50 we converged.", {"m": 1.4949}, declared=[]) == [
            ("near_miss", 1.5)
        ]

    @pytest.mark.parametrize("design", [None, {}, "not a dict", {"variables": None}])
    def test_a_malformed_design_declares_nothing(self, design) -> None:
        assert no.declared_numbers(design, None) == []


# --- a minus written as U+2212 keeps its sign -------------------------------


_MINUS = chr(0x2212)  # U+2212 MINUS SIGN, not the ASCII hyphen


class TestUnicodeMinus:
    def test_a_signed_number_is_not_compared_as_its_magnitude(self) -> None:
        """0.0963 sits 1.2% from the unrelated positive result 0.0975; -0.0963
        does not."""
        paper = f"The conditional bias was {_MINUS}0.0963 at N = 100."
        assert _flagged(paper, {"p_lower": 0.0975}) == []

    def test_the_ascii_minus_was_already_read_that_way(self) -> None:
        paper = "The conditional bias was -0.0963 at N = 100."
        assert _flagged(paper, {"p_lower": 0.0975}) == []

    def test_a_negative_number_is_still_checked_against_a_negative_result(self) -> None:
        """-0.1147 rounds to -0.115, and the paper prints -0.114. Keeping the
        sign must not stop a genuine negative near-miss."""
        paper = f"The difference was {_MINUS}0.114 at N = 100."
        assert _flagged(paper, {"diff": -0.1147}) == [("near_miss", -0.114)]


# --- a percentile rank is not its integer part ------------------------------


class TestPercentileRank:
    def test_97_5th_is_not_read_as_97(self) -> None:
        text = "ranges are the empirical 2.5th and 97.5th percentiles"
        assert _values(text) == []
        # 98 is a stored count; 97 was reported against it before
        assert _flagged(text, {"n_boot": 98.0}) == []

    def test_a_decimal_without_a_suffix_is_still_read_whole(self) -> None:
        assert _values("97.5 was observed") == [pytest.approx(97.5)]
        assert _values("a dose of 37.4 nm") == [pytest.approx(37.4)]

    def test_a_number_cut_off_from_a_dotted_one_is_not_read(self) -> None:
        assert _values("released as 3.4.1 last year") == []


# --- what must never stop being reported ------------------------------------


@pytest.mark.parametrize(
    "paper,results,expected",
    [
        # truncated, not rounded: 2.1259 rounds to 2.13 (observed in a stored quest)
        (
            "the MEEF only slightly degrades to 2.12",
            {"meef_at_100nm": 2.1259},
            ("near_miss", 2.12),
        ),
        # last digit off: 0.18346 rounds to 0.183 (observed in a stored quest)
        (
            "causing NILS to drop to 0.184 at sigma 0.45",
            {"dipole_nils": [0.18346]},
            ("near_miss", 0.184),
        ),
        # the classic transposition
        ("NILS reached 2.41", {"nils": 2.14}, ("transposed", 2.41)),
    ],
)
def test_a_genuine_near_miss_is_still_reported(paper, results, expected) -> None:
    declared = no.declared_numbers(_DESIGN, "Compare models for R0 in {0.9, 1.5, 3.0}.")
    assert expected in _flagged(paper, results, declared=declared)
