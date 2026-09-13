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
"""

from __future__ import annotations

import pytest

from core.numeric_oracle import extract_paper_numbers


def _values(text: str) -> list[float]:
    return [v for v, _tok, _ctx in extract_paper_numbers(text)]


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
