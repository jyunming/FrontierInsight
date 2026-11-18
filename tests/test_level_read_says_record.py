"""A coverage or bias read at a value of a factor compared inside each run says what FI's record holds when it holds nothing
there, instead of an empty answer the analysis cannot tell from "no trials": a result recorded once for the setting, a factor
the runs did not compare, a value they did not record, or trials at that value that recorded no interval. Where the record
does hold the trials at that value, the read is as before. Neutral made-up plan (the fixtures of tests/test_fx339_contract.py)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_fx339_contract import _cell, _metric, _table


def _at(noise: float, channel: int, est_a: list[float], *, interval: bool = True) -> dict[str, Any]:
    """The trials of setting ``noise`` at ``channel`` (a factor compared inside each run)."""
    m: dict[str, Any] = {"est_a": _metric(est_a)}
    if interval:
        m |= {"lo_a": _metric([v - 1.0 for v in est_a]), "hi_a": _metric([v + 1.0 for v in est_a])}
    return {"cell": {"noise": noise, "channel": channel}, "level": {"channel": channel}, "key": f"noise={noise},channel={channel}",
            "planned": len(est_a), "ok": len(est_a), "failed": 0, "metrics": m}


def _unknown(t: Any) -> type:
    return type(t).__init__.__globals__["UnknownResult"]


def test_where_the_record_holds_the_trials_at_that_value_the_read_is_as_before() -> None:
    t = _table([_cell(0.1, [2.0, 2.0], [5.0, 5.0]), _at(0.1, 0, [1.5, 2.5, 4.0]), _at(0.1, 1, [2.0, 2.0, 2.0])])
    assert t.coverage("est_a", "noise=0.1", channel=0) == [1.0, 1.0, 0.0]
    assert t.coverage("est_a", "noise=0.1", channel=1) == [1.0, 1.0, 1.0]
    assert t.bias("est_a", "noise=0.1", channel=0) == pytest.approx(8 / 3 - 2.0)
    assert t.bias_dropped("est_a", "noise=0.1", channel=0) == 0


def test_a_result_recorded_once_for_the_setting_says_so() -> None:
    """Old code: ``[]`` and ``None``, no word, and the analysis gave nothing for the headline coverage."""
    t = _table([_cell(0.1, [1.5, 2.5, 4.0], [5.0, 5.0, 5.0])])
    for read in (t.coverage, t.bias, t.bias_dropped):
        with pytest.raises(_unknown(t), match=r"recorded once for setting noise=0.1, not for each value of channel.*without `channel=0`"):
            read("est_a", "noise=0.1", channel=0)
    assert t.coverage("est_a", "noise=0.1") == [1.0, 1.0, 0.0]  # without the value: as before


def test_a_factor_the_runs_did_not_compare_is_named_with_the_ones_they_did() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [2.0]), _at(0.1, 1, [2.0])])
    with pytest.raises(_unknown(t), match=r"did not compare `factor`.*the factors they compared are: `channel`"):
        t.coverage("est_a", "noise=0.1", factor="0")


def test_a_value_the_runs_did_not_record_lists_the_ones_they_did() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [2.0]), _at(0.1, 1, [2.0])])
    with pytest.raises(_unknown(t), match=r"no trials at channel=5; the values they recorded are: channel=0, channel=1"):
        t.bias("est_a", "noise=0.1", channel=5)


def test_trials_at_that_value_that_recorded_no_interval_or_no_value_say_what_they_recorded() -> None:
    """Old code: ``[]`` for the coverage and ``None`` for the bias."""
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [2.0, 2.1], interval=False)])
    with pytest.raises(_unknown(t), match=r"at channel=0 recorded no interval \(lo_a, hi_a\) for `est_a`.*they recorded: est_a"):
        t.coverage("est_a", "noise=0.1", channel=0)
    assert t.bias("est_a", "noise=0.1", channel=0) == pytest.approx(0.05)  # the value is there, only the interval is not
    with pytest.raises(_unknown(t), match=r"at channel=0 recorded no value of `est_b`.*they recorded: est_a"):
        t.bias("est_b", "noise=0.1", channel=0)


def test_trials_that_recorded_the_result_but_no_finite_value_keep_the_plain_answer() -> None:
    """A value recorded as not finite is a trial left out of the mean (``bias_dropped`` counts it), not a missing record."""
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [float("nan"), float("nan")])])
    assert t.bias("est_a", "noise=0.1", channel=0) is None
    assert t.bias_dropped("est_a", "noise=0.1", channel=0) == 2


def _two(noise: float, channel: int, arm: str, est_a: list[float]) -> dict[str, Any]:
    c = _at(noise, channel, est_a)
    c["cell"]["arm"], c["level"]["arm"] = arm, arm
    return c


def test_runs_that_compare_two_factors_together_ask_for_a_value_of_each() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _two(0.1, 0, "x", [2.0]), _two(0.1, 1, "x", [2.0])])
    with pytest.raises(_unknown(t), match=r"compare `arm`, `channel` together: give a value of each \(channel=0\)"):
        t.coverage("est_a", "noise=0.1", channel=0)
    assert t.coverage("est_a", "noise=0.1", channel=0, arm="x") == [1.0]


def test_a_value_given_as_text_where_the_record_holds_a_number_says_so() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 2, [2.0])])
    with pytest.raises(_unknown(t), match=r"recorded channel=2 as a number, and it was given as text: pass it as the record holds it \(channel=2\)"):
        t.coverage("est_a", "noise=0.1", channel="2")


def test_an_interval_recorded_where_the_true_value_is_not_finite_is_no_missing_interval() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [2.0])])
    t._known["est_a"] = {"interval": ["lo_a", "hi_a"], "truth": float("nan")}
    assert t.coverage("est_a", "noise=0.1", channel=0) == []


def test_a_result_the_setting_never_recorded_is_not_said_to_be_recorded_once() -> None:
    t = _table([_cell(0.1, [2.0], [5.0])])
    t._settings["noise=0.1"]["metrics"].pop("est_b")
    with pytest.raises(_unknown(t), match=r"recorded no value compared inside them, and no `est_b`; they recorded: est_a"):
        t.bias("est_b", "noise=0.1", channel=0)
