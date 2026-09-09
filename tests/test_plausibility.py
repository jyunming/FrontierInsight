"""Tests for ``core.plausibility`` — bounds the design declared, checked in code.

The gap this fills: ``_is_degenerate_result`` fires only when every metric is
~0, so it catches a run that produced nothing and misses a run that produced
the wrong thing convincingly. A unit error, a factor of two or a sign flip
exits 0, parses fine, and passes every downstream gate because all of them are
a language model reading text.

Two properties matter, and the second as much as the first: a declared bound
must be enforced, and an *undeclared* one must never be invented. Silence has
to mean "nothing was claimed".
"""
from __future__ import annotations

import pytest

from core import plausibility as pl
from core.engine import _assertion_violations


def _design(*assertions: dict) -> dict:
    return {"hypothesis": "h", "result_assertions": list(assertions)}


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def test_value_below_min_is_a_violation() -> None:
    d = _design({"path": "cd_nm", "min": 0, "reason": "a CD is positive"})
    v = pl.check_design({"cd_nm": -3.2}, d)
    assert len(v) == 1
    assert v[0].value == -3.2
    assert "positive" in v[0].describe()


def test_value_above_max_is_a_violation() -> None:
    d = _design({"path": "contrast", "min": 0, "max": 1,
                 "reason": "normalised contrast cannot exceed 1"})
    v = pl.check_design({"contrast": 1.8}, d)
    assert len(v) == 1
    assert "contrast" in v[0].describe()


def test_value_inside_the_band_passes() -> None:
    d = _design({"path": "k1", "min": 0.25, "max": 0.5})
    assert pl.check_design({"k1": 0.31}, d) == []


def test_bounds_are_inclusive() -> None:
    d = _design({"path": "contrast", "min": 0, "max": 1})
    assert pl.check_design({"contrast": 1.0}, d) == []
    assert pl.check_design({"contrast": 0.0}, d) == []


def test_assertion_matches_at_any_nesting_depth() -> None:
    """A design names the metric; it cannot predict how the script nests it."""
    d = _design({"path": "cd_nm", "min": 0})
    v = pl.check_design({"sweep": [{"cd_nm": -1.0}, {"cd_nm": 37.0}]}, d)
    assert len(v) == 1
    assert v[0].path == "sweep[0].cd_nm"


def test_exact_path_also_works() -> None:
    d = _design({"path": "metrics.contrast", "max": 1})
    v = pl.check_design({"metrics": {"contrast": 2.0}}, d)
    assert len(v) == 1 and v[0].path == "metrics.contrast"


# ---------------------------------------------------------------------------
# Silence when nothing was claimed
# ---------------------------------------------------------------------------


def test_no_assertions_means_no_violations() -> None:
    assert pl.check_design({"anything": -999.0}, {"hypothesis": "h"}) == []


def test_unmatched_path_is_not_a_violation() -> None:
    """An assertion about a metric the run never emitted is not a failure —
    it is an assertion with nothing to say."""
    d = _design({"path": "nils", "min": 1.0})
    assert pl.check_design({"contrast": 0.4}, d) == []


@pytest.mark.parametrize("bad", [
    {"path": "", "min": 0},                       # no path
    {"path": "x"},                                # asserts nothing
    {"path": "x", "min": 5, "max": 1},            # inverted bounds
    {"path": "x", "min": "not a number"},         # unparseable
    {"path": "x", "min": float("nan")},           # non-finite
    "not even a dict",
])
def test_malformed_assertions_are_dropped_not_raised(bad) -> None:
    """This is LLM output. A malformed entry must be ignored, never fatal."""
    assert pl.parse_assertions({"result_assertions": [bad]}) == []


def test_non_dict_design_is_handled() -> None:
    assert pl.parse_assertions(None) == []
    assert pl.parse_assertions("design") == []
    assert pl.parse_assertions({"result_assertions": "nope"}) == []


def test_booleans_are_not_measurements() -> None:
    d = _design({"path": "converged", "min": 2})
    assert pl.check_design({"converged": True}, d) == []


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def test_engine_helper_surfaces_violations() -> None:
    state = {
        "result_json": {"cd_nm": -3.2},
        "design": _design({"path": "cd_nm", "min": 0}),
    }
    assert len(_assertion_violations(state)) == 1


def test_engine_helper_is_silent_without_assertions() -> None:
    state = {"result_json": {"cd_nm": -3.2}, "design": {"hypothesis": "h"}}
    assert _assertion_violations(state) == []


def test_engine_helper_never_raises(monkeypatch) -> None:
    """A defect in the checker must not stall a quest."""
    from core import plausibility

    monkeypatch.setattr(
        plausibility, "check_design",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _assertion_violations({"result_json": {}, "design": {}}) == []


def test_missing_state_keys_are_safe() -> None:
    assert _assertion_violations({}) == []


# ---------------------------------------------------------------------------
# The design prompt must actually ask for this
# ---------------------------------------------------------------------------


def test_design_prompt_requests_assertions_and_warns_against_hypothesis_bounds() -> None:
    """The field is worthless if the design bounds results to what it *expects* —
    the run could then never disagree with the hypothesis."""
    from pathlib import Path

    md = (Path(__file__).resolve().parent.parent / "agents" / "design.md").read_text(
        encoding="utf-8"
    )
    # Collapse whitespace: the guidance is prose and wraps, so asserting on a
    # literal substring would break on a reflow rather than on a real change.
    flat = " ".join(md.split())
    assert "result_assertions" in flat
    assert "never what you *expect* to happen" in flat
    assert "bounding a result to your hypothesis" in flat
