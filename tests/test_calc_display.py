"""A ``{{calc:...}}`` value in scientific notation reads without the trailing zeros of its mantissa. No model is called."""

from __future__ import annotations

from core import paper_values


def test_a_small_value_has_no_trailing_zeros() -> None:
    assert paper_values._calc_text(1.5e-5) == "1.5e-05"
    assert paper_values._calc_text(1e-5) == "1e-05"
    assert paper_values._calc_text(1.2345e-9) == "1.235e-09"


def test_ordinary_values_are_unchanged() -> None:
    assert paper_values._calc_text(3.18245) == "3.182"
    assert paper_values._calc_text(0.9356) == "0.9356"
    assert paper_values._calc_text(400.0) == "400"
