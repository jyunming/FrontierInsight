"""A quantity the plan names, that no trial returns and the analysis records one value for, gets a key (core/paper_values.py),
and a quantity the topic asks for that FI holds a value for but the paper never places is counted as missing
(core/goal_coverage.py::unplaced_answers). Neutral made-up study: a decay rate measured per trial, and a settling limit the
analysis works out across the settings."""
from __future__ import annotations

import json
from pathlib import Path

from core import goal_coverage as gc
from core import paper_values as pv

PROTOCOL = {
    "metrics": [
        {"id": "decay_rate", "label": "Decay rate", "kind": "mean"},
        {"id": "settling_limit", "label": "Limit of the settling time (hours)", "kind": "mean"},
        {"id": "fit_slope", "label": "Fit slope", "kind": "mean"},
    ],
    "topic_map": [
        {"item": "Q1", "by": "metric:decay_rate"},
        {"item": "Q2", "by": "metric:settling_limit"},
        {"item": "Q3", "by": "metric:fit_slope"},
    ],
}
REQ = {"quantities": [
    {"name": "decay rate", "stated": "report the decay rate", "main": True},
    {"name": "settling limit", "stated": "give the limit of the settling time", "main": True},
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
    rec = _rec(tmp_path, {"settling_limit": 12.3456, "unplanned_extra": 5.0, "fit_slope": float("nan")})
    assert rec.answers == {"settling_limit": 12.3456}  # only a plan name, only a finite number
    value = pv.resolve("{{v:settling_limit}}", rec)
    assert value.filled and value.text == "12.35" and value.numbers == [12.3456]
    assert "{{v:settling_limit}} -> 12.35" in pv.writer_block(rec)


def test_a_quantity_the_trials_returned_keeps_its_keys_by_setting(tmp_path):
    rec = _rec(tmp_path, {"decay_rate": 7.0, "settling_limit": 1.5})
    assert "decay_rate" not in rec.answers
    assert pv.resolve("{{v:decay_rate}}", rec).hit is not None  # a key with no settings is still refused for a trial value
    assert pv.resolve("{{v:decay_rate @ dose=2}}", rec).filled


def test_a_key_for_an_analysis_value_takes_no_settings_form_or_comparison(tmp_path):
    rec = _rec(tmp_path, {"settling_limit": 1.5})
    for bad in ("{{v:settling_limit @ dose=all}}", "{{v:settling_limit:sd}}", "{{v:settling_limit | ref}}"):
        value = pv.resolve(bad, rec)
        assert value.text == pv.NOT_RECORDED and "one value for the whole study" in value.hit.reason


def test_nothing_is_guessed_for_a_name_the_analysis_did_not_record(tmp_path):
    rec = _rec(tmp_path, {"settling_limit": 1.5})
    assert pv.resolve("{{v:fit_slope}}", rec).hit is not None
    assert pv.resolve("{{v:unknown_name}}", rec).hit is not None


def test_a_failed_run_does_not_erase_the_earlier_record(tmp_path):
    rec = _rec(tmp_path, {"settling_limit": 1.5})
    assert pv.write_answers(tmp_path, {}, PROTOCOL) == [] and pv.write_answers(tmp_path, None, PROTOCOL) == []
    assert pv.records_of(tmp_path, PROTOCOL).answers == rec.answers


def test_placed_metrics_counts_filled_keys_only(tmp_path):
    rec = _rec(tmp_path, {"settling_limit": 1.5})
    filled, _ = pv.fill_values("A {{v:settling_limit}} and {{v:decay_rate @ dose=2}} and {{v:bogus}}.", rec)
    assert pv.placed_metrics(filled) == {"settling_limit", "decay_rate"}


def test_the_goal_check_counts_an_unplaced_topic_answer_as_missing():
    held = {"decay_rate", "settling_limit", "fit_slope"}
    found = gc.unplaced_answers(REQ, PROTOCOL, held, {"decay_rate"})
    assert [f["metric"] for f in found] == ["settling_limit"]  # fit_slope is not a main quantity
    assert "settling limit" in found[0]["sentence"] and "never gives it" in found[0]["sentence"]


def test_the_goal_check_passes_a_paper_that_places_the_value():
    assert gc.unplaced_answers(REQ, PROTOCOL, {"decay_rate", "settling_limit"}, {"decay_rate", "settling_limit"}) == []


def test_the_goal_check_asks_for_nothing_FI_has_no_value_for():
    assert gc.unplaced_answers(REQ, PROTOCOL, {"decay_rate"}, {"decay_rate"}) == []
    assert gc.unplaced_answers(None, PROTOCOL, {"decay_rate"}, set()) == []
    assert gc.unplaced_answers(REQ, None, {"decay_rate"}, set()) == []


# --- replicates: no silent pick among different values ---------------------------------------------------------------


def test_replicates_with_different_values_offer_no_single_value(tmp_path):
    runs = [{"_seed": s, "settling_limit": v} for s, v in ((0, 0.30), (1, 0.40), (2, 0.50))]
    rec = _rec(tmp_path, None)
    pv.write_answers(tmp_path, runs[0], PROTOCOL, runs)
    rec = pv.records_of(tmp_path, PROTOCOL)
    assert "settling_limit" not in rec.answers and "settling_limit" not in rec.fillable()
    value = pv.resolve("{{v:settling_limit}}", rec)
    assert value.text == pv.NOT_RECORDED and "run 0: 0.3000; run 1: 0.4000; run 2: 0.5000" in value.hit.reason
    block = pv.writer_block(rec)
    assert "Worked out once per replicate" in block and "{{v:settling_limit}} ->" not in block


def test_replicates_that_all_agree_give_the_one_value(tmp_path):
    runs = [{"_seed": s, "settling_limit": 1.25} for s in (0, 1)]
    _rec(tmp_path, None)
    pv.write_answers(tmp_path, runs[0], PROTOCOL, runs)
    rec = pv.records_of(tmp_path, PROTOCOL)
    assert rec.answers == {"settling_limit": 1.25} and rec.by_run == {}


def test_a_later_run_replaces_an_earlier_record(tmp_path):
    _rec(tmp_path, {"settling_limit": 1.0})
    pv.write_answers(tmp_path, {"settling_limit": 2.0}, PROTOCOL)
    assert pv.records_of(tmp_path, PROTOCOL).answers == {"settling_limit": 2.0}


# --- the review is never asked for what the paper cannot place ----------------------------------------------------------


def test_a_left_out_quantity_is_not_held_and_gives_no_hit(tmp_path):
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": _cells()}), encoding="utf-8")
    pv.write_answers(tmp_path, {"settling_limit": 1.5}, PROTOCOL)
    rec = pv.load(tmp_path, PROTOCOL, left_out={"decay_rate": "wrong", "settling_limit": "wrong"})
    rec.answers = {"settling_limit": 1.5}  # even a record that kept it
    assert rec.fillable() == set()
    assert gc.unplaced_answers(REQ, PROTOCOL, rec.fillable(), set()) == []


def test_the_engine_check_gives_no_hit_for_a_left_out_or_per_replicate_quantity(tmp_path):
    import logging
    from types import SimpleNamespace

    from core import engine as E

    runs = [{"_seed": s, "settling_limit": v} for s, v in ((0, 0.30), (1, 0.40))]
    _rec(tmp_path, None)
    pv.write_answers(tmp_path, runs[0], PROTOCOL, runs)
    (tmp_path / ".fi").mkdir(exist_ok=True)
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(REQ), encoding="utf-8")
    me = SimpleNamespace(quest_root=tmp_path, fi_dir=tmp_path / ".fi", _log=logging.getLogger("t"))
    me._protocol_block = lambda st: PROTOCOL
    me._paper_records = lambda st: pv.records_of(tmp_path, PROTOCOL)
    paper = "The decay rate is {{v:decay_rate @ dose=2}}."
    filled, _ = pv.fill_values(paper, me._paper_records(None))
    assert E.Engine._unplaced_topic_answers(me, filled, {}) == []  # the per-replicate quantity cannot be placed


# --- a run that failed is missing; an old record is read; fillable fills; a count is not the value ----------------------


def test_a_failed_replicate_counts_as_missing_so_no_key_is_offered(tmp_path):
    _rec(tmp_path, None)
    two_agree = [{"_seed": 0, "settling_limit": 1.5}, {"_seed": 1, "settling_limit": 1.5}]
    pv.write_answers(tmp_path, two_agree[0], PROTOCOL, two_agree, scheduled=3)  # the third run failed
    rec = pv.records_of(tmp_path, PROTOCOL)
    assert "settling_limit" not in rec.answers and "settling_limit" in rec.by_run and "settling_limit" not in rec.fillable()
    pv.write_answers(tmp_path, {"settling_limit": 1.5}, PROTOCOL, None, scheduled=3)  # one of three succeeded
    assert "settling_limit" not in pv.records_of(tmp_path, PROTOCOL).answers
    pv.write_answers(tmp_path, two_agree[0], PROTOCOL, two_agree, scheduled=2)  # every scheduled run agreed
    assert pv.records_of(tmp_path, PROTOCOL).answers == {"settling_limit": 1.5}


def test_a_record_written_before_runs_were_kept_is_read_as_one_run(tmp_path):
    _rec(tmp_path, None)
    (tmp_path / ".fi").mkdir(exist_ok=True)
    (tmp_path / ".fi" / "analysis_answers.json").write_text(json.dumps({"values": {"settling_limit": 2.5}}), encoding="utf-8")
    assert pv.records_of(tmp_path, PROTOCOL).answers == {"settling_limit": 2.5}


def test_fillable_is_what_a_key_can_fill(tmp_path):
    cells = _cells()
    cells[0]["metrics"]["fit_slope"] = {"values": [float("nan")], "trials": [0], "count": 1, "total": 0.0, "non_finite": 1}
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}, allow_nan=True), encoding="utf-8")
    rec = pv.records_of(tmp_path, PROTOCOL)
    assert "decay_rate" in rec.fillable() and "fit_slope" not in rec.fillable()  # no finite value anywhere: no key fills
    req = {"quantities": [{"name": "slope of the fit", "stated": "the slope", "main": True}]}
    protocol = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:fit_slope"}]}
    assert gc.unplaced_answers(req, protocol, rec.fillable(), set()) == []  # nothing the writer could place is asked for
    assert gc.unplaced_answers(req, protocol, set(rec.metrics()), set())  # the bare names would have asked for it


def test_only_a_key_that_renders_the_value_places_the_quantity():
    for placed in ("{{v:decay_rate @ dose=2}}", "{{v:decay_rate:mean_ci @ dose=all}}", "{{v:decay_rate:pm @ dose=1}}",
                   "{{v:decay_rate @ dose=each | max}}", "{{v:settling_limit}}", "{{v:settling_limit | %}}"):
        assert pv.placed_metrics(placed.join(["", ""])) in ({"decay_rate"}, {"settling_limit"}), placed
    for not_placed in ("{{v:decay_rate:n @ dose=2}}", "{{v:decay_rate:failed @ dose=2}}", "{{v:decay_rate:sd @ dose=2}}",
                       "{{v:decay_rate:ref @ dose=2}}", "{{v:decay_rate:count @ dose=2}}",
                       "{{v:decay_rate @ dose=each | argmax}}", "{{v:decay_rate @ dose=2 | ratio ref}}"):
        assert pv.placed_metrics(not_placed) == set(), not_placed
