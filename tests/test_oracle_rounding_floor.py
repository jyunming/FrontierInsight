"""A check's tolerance cannot ask for more than floating-point arithmetic gives.

Replays the frozen protocol of a real quest (a multiple-testing study): the plan set an absolute tolerance of 0 on the
largest difference between computed and analytic p-values, and correct code measured 1.1e-16 (rounding in erfc). The
check failed on every run, was marked unconfirmed, and the paper called its own correct numbers provisional. The same
protocol held two tolerance-0 checks on counts, which must keep failing on a count of 1.
"""

from __future__ import annotations

from core import oracle_check as oc

# Copied from needs/FROZEN_PROTOCOL.json of that quest.
PVALUE = {"name": "pvalue_bounds", "kind": "special_case", "expected": 0.0, "tolerance": 0.0, "tolerance_mode": "absolute",
          "case": {"z": [-1.96, 0.0, 1.96]}, "measure": "pvalue_exact_error"}
NESTING = {"name": "rejection_nesting", "kind": "invariant", "expected": 0.0, "tolerance": 0.0, "tolerance_mode": "absolute",
           "case": {"m": 20, "pi1": 0.2}, "measure": "rejection_nesting_violation"}


def _verdicts(values: dict[str, float]) -> dict[str, bool | None]:
    reported = {"checks": [{"name": k, "value": v} for k, v in values.items()]}
    return {j["name"]: j["passed_by_engine"] for j in oc.judged([PVALUE, NESTING], reported)}


def test_a_tolerance_of_0_on_a_computed_value_passes_rounding_and_a_count_off_by_one_still_fails() -> None:
    assert _verdicts({"pvalue_bounds": 1.1102230246251565e-16, "rejection_nesting": 0.0}) == {
        "pvalue_bounds": True, "rejection_nesting": True}
    assert _verdicts({"pvalue_bounds": 1e-6, "rejection_nesting": 1.0}) == {
        "pvalue_bounds": False, "rejection_nesting": False}


def test_the_floor_scales_with_the_expected_value_and_leaves_ordinary_tolerances_alone() -> None:
    limit = oc.limit_of({"expected": 1e6, "tolerance": 0.0, "tolerance_mode": "absolute"})[1]
    assert limit is not None and 1e6 * 2.2e-16 < limit < 1e6 * 1e-13
    assert oc.limit_of({"expected": 2.0, "tolerance": 0.01, "tolerance_mode": "absolute"})[1] == 0.01
    assert oc.limit_of({"expected": 2.0, "tolerance": 0.01, "tolerance_mode": "relative"})[1] == 0.02
    # A relative tolerance around 0 still cannot be judged.
    assert oc.limit_of({"expected": 0.0, "tolerance": 0.01, "tolerance_mode": "relative"})[1] is None
