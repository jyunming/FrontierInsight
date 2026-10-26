"""A quantity the design says can honestly sit on its bound is not sent back for it (G14)."""
from __future__ import annotations

from core import degenerate_results as dg
from core import plausibility as pl
from core import run_estimate as est


def _sweep(**values: float) -> dict:
    return {"by_n": {name: {"fail_rate": v} for name, v in values.items()}}


def _design(**extra) -> dict:
    return {"result_assertions": [{"path": "fail_rate", "min": 0, "max": 1, **extra}]}


def test_declared_honest_bound_is_not_at_bound() -> None:
    """Old code: a failure rate of 0 in every setting was always sent back as at_bound, even when the design said
    0 is a possible honest answer, so the repair loop ran on a correct script."""
    result = _sweep(a=0.0, b=0.0, c=0.0)
    assert pl.check_design(result, _design(may_equal_bound=True)) == []


def test_declared_honest_bound_also_covers_the_upper_bound() -> None:
    assert pl.check_design(_sweep(a=1.0, b=1.0), _design(may_equal_bound=True)) == []


def test_undeclared_bound_still_fires() -> None:
    """Known positive: without the declaration the same result is still caught."""
    v = pl.check_design(_sweep(a=0.0, b=0.0, c=0.0), _design())
    assert [x.kind for x in v] == ["at_bound"]


def test_declared_honest_bound_still_rejects_a_value_outside_the_range() -> None:
    v = pl.check_design(_sweep(a=0.0, b=0.0, c=1.5), _design(may_equal_bound=True))
    assert [x.kind for x in v] == [] or all(x.kind != "at_bound" for x in v)
    assert any(x.value == 1.5 for x in v)


def test_the_declaration_must_be_a_real_true() -> None:
    assert pl.parse_assertions(_design(may_equal_bound="false"))[0].may_equal_bound is False
    assert pl.parse_assertions(_design(may_equal_bound=True))[0].may_equal_bound is True


_OBS = [({"n": n}, {"a_rate": 0.0, "b_rate": 0.0}) for n in (1, 2) for _ in range(4)]
_KINDS = {"a_rate": "proportion", "b_rate": "proportion"}
_SRC = "def run_trial(cell, rng):\n    return {'a_rate': 0.0, 'b_rate': 0.0}\n"


def test_declared_metrics_are_exempt_from_the_pilot_stuck_rule() -> None:
    """Old code: two rare-event proportions that were 0 in a small pilot were sent back as stuck, with no way for the
    plan to say 0 is an honest answer."""
    protocol = {"metrics": [{"id": "a_rate", "kind": "proportion", "may_equal_bound": True},
                            {"id": "b_rate", "kind": "proportion", "may_equal_bound": True}]}
    exempt = dg.expected_constants(protocol)
    assert {"a_rate", "b_rate"} <= exempt
    pilot = est.pilot_problems(_OBS, _KINDS, source=_SRC, exempt=exempt)
    assert not [p for p in pilot.problems if p.kind == "stuck"]


def test_undeclared_stuck_proportions_are_still_found() -> None:
    """Known positive: the same pilot with nothing declared still reports both as stuck."""
    pilot = est.pilot_problems(_OBS, _KINDS, source=_SRC)
    assert sorted(p.metric for p in pilot.problems if p.kind == "stuck") == ["a_rate", "b_rate"]
