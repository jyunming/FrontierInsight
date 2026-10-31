"""The run check reads each result where FI itself records it: a number per value of one thing a run compares inside it
is recorded under that value alone, and a mean the analysis prints without a list is backed by FI's trials when it equals
their mean. No model is called."""

from __future__ import annotations

from collections import Counter

from core import run_manifest as rm

PROTOCOL = {
    "grid": {"m": [10, 20]},
    "in_each_run": {"T": [1.0, 2.0], "var": ["a", "b"]},
    "runs_per_setting": 4,
    "metrics": [{"id": "kT", "kind": "mean"}, {"id": "kv", "kind": "mean"}],
}


def _record() -> dict[str, dict[str, Counter]]:
    out: dict[str, dict[str, Counter]] = {}
    for m in (10, 20):
        for t in ("1.0", "2.0"):
            out[f"m={m},T={t}"] = {"kT": Counter({"0.1": 2, "0.3": 2})}
        for v in ("a", "b"):
            out[f"m={m},var={v}"] = {"kv": Counter({"0.5": 4})}
    return out


def _result(kt_list=True) -> dict:
    out: dict = {}
    for m in (10, 20):
        for t in ("1.0", "2.0"):
            out[f"m={m},T={t}"] = {"kT": 0.2, **({"kT_values": [0.1, 0.1, 0.3, 0.3]} if kt_list else {})}
        for v in ("a", "b"):
            out[f"m={m},var={v}"] = {"kv": 0.5, "kv_values": [0.5] * 4}
    return out


def _problems(result: dict, record=None) -> list:
    record = record or _record()
    pooled: dict[str, Counter] = {}
    for by in record.values():
        for name, c in by.items():
            pooled.setdefault(name, Counter()).update(c)
    return rm._value_count_findings(PROTOCOL, {"successful_per_cell": {}}, result, trial_mode=True,
                                    recorded=pooled, recorded_by_cell=record)


def test_a_metric_is_owed_over_the_axes_its_record_keeps_it_under() -> None:
    assert set(rm.metric_grid(PROTOCOL, "kT", _record())) == {"m", "T"}
    assert set(rm.metric_grid(PROTOCOL, "kv", _record())) == {"m", "var"}


def test_values_listed_per_value_of_one_inside_axis_are_not_more_than_the_record() -> None:
    assert _problems(_result()) == []


def test_more_values_than_the_record_holds_at_a_value_is_still_found() -> None:
    result = _result()
    result["m=10,T=1.0"]["kT_values"] = [0.1, 0.1, 0.3, 0.3, 0.2]
    found = _problems(result)
    assert found and any(rm.kind_of(f) == rm.FABRICATED for f, _ in found)


def test_a_mean_printed_without_a_list_that_equals_the_trials_mean_is_backed() -> None:
    assert _problems(_result(kt_list=False)) == []


def test_a_mean_printed_without_a_list_that_is_not_the_trials_mean_is_still_flagged() -> None:
    result = _result(kt_list=False)
    result["m=20,T=2.0"]["kT"] = 0.25
    found = _problems(result)
    assert found and all(rm.kind_of(f) == rm.UNPRINTED for f, _ in found)


def test_no_mean_and_no_list_at_a_setting_is_still_flagged() -> None:
    result = _result(kt_list=False)
    del result["m=20,T=2.0"]["kT"]
    assert _problems(result)
