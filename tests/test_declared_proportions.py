"""The kind of interval a result gets is the plan's to declare (``protocol.metrics[].kind``), never read from the values the trials
happened to give. Made-up topic (seedlings); the trial record is written here, no model is called."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import metric_spec, paper_values, stats, trial_table

PROPORTION = {"metrics": [{"id": "sprouted", "estimand": "chance a seed sprouts", "kind": "proportion", "unit": "one seed"}]}
MEAN = {"metrics": [{"id": "sprouted", "estimand": "mean sprouting score", "kind": "mean", "unit": "one seed"}]}


def _cells(per_setting: dict[int, list[float]], name: str = "sprouted") -> list[dict[str, Any]]:
    return [{"cell": {"dose": d}, "key": f"dose={d}", "planned": len(v), "ok": len(v), "failed": 0,
             "metrics": {name: {"values": v, "trials": list(range(len(v)))}}} for d, v in per_setting.items()]


def _quest(tmp_path: Path, per_setting: dict[int, list[float]]) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": _cells(per_setting)}), encoding="utf-8")
    return tmp_path


def test_the_declared_proportions_are_the_metrics_of_kind_proportion_only() -> None:
    assert metric_spec.proportion_ids(PROPORTION) == frozenset({"sprouted"})
    assert metric_spec.proportion_ids(MEAN) == frozenset() and metric_spec.proportion_ids(None) == frozenset()
    assert metric_spec.proportion_ids({"metrics": []}) == frozenset()


def test_a_setting_where_every_trial_gave_0_or_1_does_not_switch_a_mean_to_a_binomial_interval(tmp_path: Path) -> None:
    # the original failure: of three settings of one value, those whose trials all happened to be 0 or 1 got a Wilson interval
    # and the others a t interval, so one table mixed two kinds of interval for one quantity
    quest = _quest(tmp_path, {1: [0.0, 1.0, 1.0, 0.0], 2: [0.0, 1.0, 2.0, 1.0], 3: [1.0, 1.0, 1.0, 1.0]})
    rows = {r["setting"]: r for r in trial_table.rows(quest, (), metric_spec.proportion_ids(MEAN))}
    assert {r["method"].split(" (")[0] for r in rows.values()} == {"t interval of the mean"}
    assert "does not declare" in rows["dose=1"]["method"] and "does not declare" in rows["dose=3"]["method"]
    assert rows["dose=2"]["method"] == "t interval of the mean", "0, 1 and 2 is no proportion in either reading: nothing to say"
    assert (rows["dose=1"]["ci_lower"], rows["dose=1"]["ci_upper"]) != stats.wilson_interval(2, 4)
    # a plan that says nothing about the kind (no protocol) is not guessed from the values either
    rows = {r["setting"]: r for r in trial_table.rows(quest)}
    assert all(not r["method"].startswith("Wilson") for r in rows.values())


def test_a_declared_proportion_keeps_its_wilson_interval_in_every_setting_that_gave_0_or_1(tmp_path: Path) -> None:
    quest = _quest(tmp_path, {1: [0.0, 1.0, 1.0, 0.0], 2: [1.0, 1.0, 1.0, 1.0], 3: [0.0, 0.0, 0.0, 0.0]})
    rows = {r["setting"]: r for r in trial_table.rows(quest, (), metric_spec.proportion_ids(PROPORTION))}
    assert all(r["method"] == "Wilson" for r in rows.values())
    assert (rows["dose=1"]["ci_lower"], rows["dose=1"]["ci_upper"]) == stats.wilson_interval(2, 4)
    assert (rows["dose=2"]["ci_lower"], rows["dose=2"]["ci_upper"]) == stats.wilson_interval(4, 4)


def test_the_paper_s_interval_follows_the_plan_s_kind(tmp_path: Path) -> None:
    cells = _cells({1: [0.0, 1.0, 1.0, 0.0, 1.0]})
    declared = paper_values.Records(cells=cells, protocol=PROPORTION)
    _mean, lo, hi, how, *_rest = paper_values._interval(declared, cells, "sprouted")
    assert (lo, hi) == stats.wilson_interval(3, 5) and how == "95% Wilson interval"
    for protocol in (MEAN, None):
        rec = paper_values.Records(cells=cells, protocol=protocol)
        mean, lo, hi, how, n, _symmetric, _se = paper_values._interval(rec, cells, "sprouted")
        assert how.startswith("95% t interval of the mean") and "does not declare" in how and n == 5
        assert lo < mean < hi and (lo, hi) != stats.wilson_interval(3, 5)


def test_the_precision_target_is_held_to_the_interval_the_plan_declares(tmp_path: Path) -> None:
    quest = _quest(tmp_path, {1: [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]})
    wilson = trial_table.per_setting_precision(quest, "sprouted", 0.3, metric_spec.proportion_ids(PROPORTION))
    t_based = trial_table.per_setting_precision(quest, "sprouted", 0.3)
    lo, hi = stats.wilson_interval(7, 10)
    assert wilson is not None and wilson["widest"] == pytest.approx((hi - lo) / 2.0)
    # an undeclared value that gave only 0 and 1 is still judged by Wilson (a t interval overstates the precision of few trials)
    assert t_based is not None and t_based["widest"] == pytest.approx(wilson["widest"])
