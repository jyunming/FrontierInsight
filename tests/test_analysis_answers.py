"""A quantity the plan names, that no trial returns and the analysis records one value for, gets a key (core/paper_values.py),
and a quantity the topic asks for that FI holds a value for but the paper never places is counted as missing
(core/goal_coverage.py::unplaced_answers). Neutral made-up study: a decay rate measured per trial, and a half-life limit the
analysis works out across the settings."""
from __future__ import annotations

import json
from pathlib import Path

from core import goal_coverage as gc
from core import paper_values as pv

PROTOCOL = {
    "metrics": [
        {"id": "decay_rate", "label": "Decay rate", "kind": "mean"},
        {"id": "half_life_limit", "label": "Limit of the half-life (hours)", "kind": "mean"},
        {"id": "fit_slope", "label": "Fit slope", "kind": "mean"},
    ],
    "topic_map": [
        {"item": "Q1", "by": "metric:decay_rate"},
        {"item": "Q2", "by": "metric:half_life_limit"},
        {"item": "Q3", "by": "metric:fit_slope"},
    ],
}
REQ = {"quantities": [
    {"name": "decay rate", "stated": "report the decay rate", "main": True},
    {"name": "half-life limit", "stated": "give the limit of the half-life", "main": True},
    {"name": "slope of the fit", "stated": "the slope of the fit", "main": False},
]}


def _cells() -> list[dict]:
    return [{"cell": {"dose": d}, "key": f"dose={d}", "planned": 1, "ok": 1, "failed": 0,
             "metrics": {"decay_rate": {"values": [0.1 * d], "trials": [0], "count": 1, "total": 0.1 * d, "non_finite": 0}}}
            for d in (1, 2, 3)]


def _rec(tmp_path: Path, result: dict | None) -> pv.Records:
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": _cells()}), encoding="utf-8")
    if result is not None:
        pv.write_answers(tmp_path, result, PROTOCOL)
    rec = pv.records_of(tmp_path, PROTOCOL)
    assert rec is not None
    return rec


def test_the_analysis_value_under_a_plan_name_has_a_key(tmp_path):
    rec = _rec(tmp_path, {"half_life_limit": 12.3456, "unplanned_extra": 5.0, "fit_slope": float("nan")})
    assert rec.answers == {"half_life_limit": 12.3456}  # only a plan name, only a finite number
    value = pv.resolve("{{v:half_life_limit}}", rec)
    assert value.filled and value.text == "12.35" and value.numbers == [12.3456]
    assert "{{v:half_life_limit}} -> 12.35" in pv.writer_block(rec)


def test_a_quantity_the_trials_returned_keeps_its_keys_by_setting(tmp_path):
    rec = _rec(tmp_path, {"decay_rate": 7.0, "half_life_limit": 1.5})
    assert "decay_rate" not in rec.answers
    assert pv.resolve("{{v:decay_rate}}", rec).hit is not None  # a key with no settings is still refused for a trial value
    assert pv.resolve("{{v:decay_rate @ dose=2}}", rec).filled


def test_a_key_for_an_analysis_value_takes_no_settings_form_or_comparison(tmp_path):
    rec = _rec(tmp_path, {"half_life_limit": 1.5})
    for bad in ("{{v:half_life_limit @ dose=all}}", "{{v:half_life_limit:sd}}", "{{v:half_life_limit | ref}}"):
        value = pv.resolve(bad, rec)
        assert value.text == pv.NOT_RECORDED and "one value for the whole study" in value.hit.reason


def test_nothing_is_guessed_for_a_name_the_analysis_did_not_record(tmp_path):
    rec = _rec(tmp_path, {"half_life_limit": 1.5})
    assert pv.resolve("{{v:fit_slope}}", rec).hit is not None
    assert pv.resolve("{{v:unknown_name}}", rec).hit is not None


def test_a_failed_run_does_not_erase_the_earlier_record(tmp_path):
    rec = _rec(tmp_path, {"half_life_limit": 1.5})
    assert pv.write_answers(tmp_path, {}, PROTOCOL) == [] and pv.write_answers(tmp_path, None, PROTOCOL) == []
    assert pv.records_of(tmp_path, PROTOCOL).answers == rec.answers


def test_placed_metrics_counts_filled_keys_only(tmp_path):
    rec = _rec(tmp_path, {"half_life_limit": 1.5})
    filled, _ = pv.fill_values("A {{v:half_life_limit}} and {{v:decay_rate @ dose=2}} and {{v:bogus}}.", rec)
    assert pv.placed_metrics(filled) == {"half_life_limit", "decay_rate"}


def test_the_goal_check_counts_an_unplaced_topic_answer_as_missing():
    held = {"decay_rate", "half_life_limit", "fit_slope"}
    found = gc.unplaced_answers(REQ, PROTOCOL, held, {"decay_rate"})
    assert [f["metric"] for f in found] == ["half_life_limit"]  # fit_slope is not a main quantity
    assert "half-life limit" in found[0]["sentence"] and "never gives it" in found[0]["sentence"]


def test_the_goal_check_passes_a_paper_that_places_the_value():
    assert gc.unplaced_answers(REQ, PROTOCOL, {"decay_rate", "half_life_limit"}, {"decay_rate", "half_life_limit"}) == []


def test_the_goal_check_asks_for_nothing_FI_has_no_value_for():
    assert gc.unplaced_answers(REQ, PROTOCOL, {"decay_rate"}, {"decay_rate"}) == []
    assert gc.unplaced_answers(None, PROTOCOL, {"decay_rate"}, set()) == []
    assert gc.unplaced_answers(REQ, None, {"decay_rate"}, set()) == []
