"""The criteria a quest's code is judged by (core/criteria.py): proposed in the plan, frozen with the protocol, computed by FI
after every run, never the study's headline finding."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from core import criteria as cr, frozen_protocol as fp, plan

ORACLE = {"name": "rk4 error", "kind": "special_case", "check": "error at t=1 of y'=-y", "expected": 0.0,
          "tolerance": 1e-5, "case": {"dt": 0.1}, "measure": "error", "order": 4}
SCRIPTED = {"name": "energy drift", "kind": "invariant", "check": "worst energy change", "expected": 0.0, "tolerance": 1e-6}
METRIC = {"id": "outbreak_probability", "estimand": "P(outbreak)", "kind": "proportion", "unit": "trajectory"}
BASE = {"oracles": [ORACLE, SCRIPTED], "metrics": [METRIC]}

ERR = {"name": "rk4 error small", "oracle": "rk4 error", "direction": "lower", "target": 1e-5, "tolerance": 1e-7}
DRIFT = {"name": "energy kept", "oracle": "Energy Drift", "use": "value", "direction": "lower is better", "tolerance": 1e-8}
CASE = {"name": "error at dt 0.05", "case": {"dt": 0.05}, "measure": "error", "direction": "lower", "tolerance": 1e-8}
SE = {"name": "error bars shrink", "trials": "final_size", "direction": "target", "target": 0.5, "tolerance": 0.15}


def _proto(*items: dict[str, Any]) -> dict[str, Any]:
    return {**BASE, "criteria": list(items)}


# --- reading what the plan wrote -------------------------------------------------------------------------------------


def test_each_kind_of_criterion_is_read_and_put_in_one_form() -> None:
    fixed, why = cr.normalize([ERR, DRIFT, CASE, SE], _proto())
    assert why is None, why
    by = {c["name"]: c for c in fixed}
    assert by["rk4 error small"]["use"] == "error" and by["rk4 error small"]["direction"] == "lower"
    # The oracle is named as the protocol names it, and a direction in words is read as the word it means.
    assert by["energy kept"]["oracle"] == "energy drift" and by["energy kept"]["direction"] == "lower"
    assert by["error at dt 0.05"]["case"] == {"dt": 0.05} and by["error at dt 0.05"]["measure"] == "error"
    assert by["error bars shrink"]["trials"] == "final_size" and by["error bars shrink"]["target"] == 0.5


@pytest.mark.parametrize("bad, why", [
    ({**ERR, "oracle": "no such check"}, "names no declared check"),
    ({k: v for k, v in ERR.items() if k != "tolerance"}, "`tolerance`"),
    ({**ERR, "tolerance": -1}, "`tolerance`"),
    ({**ERR, "direction": "sideways"}, "`direction`"),
    ({**SE, "target": None}, "`target`"),
    ({**ERR, "use": "square"}, "`use`"),
    ({"name": "x", "direction": "lower", "tolerance": 0.1}, "what FI measures"),
    ({**ERR, "trials": "y"}, "only one"),
    ({**CASE, "measure": None}, "`measure`"),
    ({**ERR, "name": ""}, "no `name`"),
    ({"name": "from results", "result": "mean_error", "direction": "lower", "tolerance": 0.1}, "the script's own results"),
])
def test_a_criterion_that_cannot_be_computed_by_fi_is_refused_with_the_reason(bad: dict[str, Any], why: str) -> None:
    fixed, reason = cr.normalize([bad], _proto())
    assert fixed is None and why in (reason or "") and "`protocol.criteria`" in reason, reason


@pytest.mark.parametrize("bad", [
    {"name": "headline", "trials": "outbreak_probability", "direction": "higher", "tolerance": 0.01},
    {"name": "headline", "case": {"dt": 0.1}, "measure": "OUTBREAK_PROBABILITY", "direction": "higher", "tolerance": 0.01},
    {"name": "outbreak_probability", "oracle": "rk4 error", "direction": "lower", "tolerance": 0.01},
])
def test_a_criterion_on_the_studys_headline_number_is_refused(bad: dict[str, Any]) -> None:
    fixed, why = cr.normalize([bad], _proto())
    assert fixed is None and "headline" in (why or ""), why


def test_the_headline_named_by_the_precision_target_is_refused_too() -> None:
    protocol = {**BASE, "precision": {"target_half_width": 0.03, "metric": "attack_rate"}}
    fixed, why = cr.normalize([{"name": "a", "trials": "attack_rate", "direction": "lower", "tolerance": 1}], protocol)
    assert fixed is None and "headline" in why


def test_more_than_five_is_refused_and_two_names_alike_are_refused() -> None:
    six = [{**ERR, "name": f"c{i}"} for i in range(6)]
    assert "at most 5" in (cr.normalize(six, _proto())[1] or "")
    assert "twice" in (cr.normalize([ERR, dict(ERR)], _proto())[1] or "")


def test_a_draft_keeps_the_usable_criteria_and_says_which_were_left_out() -> None:
    headline = {"name": "headline", "trials": "outbreak_probability", "direction": "higher", "tolerance": 0.01}
    kept, notes = cr.repair([ERR, headline, {**CASE, "measure": None}, *[{**DRIFT, "name": f"d{i}"} for i in range(5)]], _proto())
    assert [c["name"] for c in kept] == ["rk4 error small", "d0", "d1", "d2", "d3"]
    assert len(notes) == 3 and any("headline" in n for n in notes) and any("`measure`" in n for n in notes)
    assert any("at most 5" in n and "d4" in n for n in notes)


def test_the_protocol_reads_criteria_strictly_and_a_draft_is_repaired_one_entry_at_a_time() -> None:
    fixed, why = plan.normalize_protocol(_proto(ERR, SE))
    assert why is None and [c["name"] for c in fixed["criteria"]] == ["rk4 error small", "error bars shrink"]
    # A person's edit that cannot be used stops the plan with the reason.
    fixed, why = plan.normalize_protocol(_proto(ERR, {**SE, "direction": "sideways"}))
    assert fixed is None and "`protocol.criteria`" in why
    # A draft keeps the good one, and the plan says what was left out.
    fixed, notes = plan.repair_protocol(_proto(ERR, {**SE, "direction": "sideways"}))
    assert [c["name"] for c in fixed["criteria"]] == ["rk4 error small"] and fixed["oracles"]
    assert len(notes) == 1 and "error bars shrink" in notes[0]


def test_the_plan_notes_a_criterion_on_a_check_the_script_measures_itself_and_fewer_than_two() -> None:
    notes = cr.plan_notes(_proto(DRIFT))
    assert any("energy drift" in n and "measured by the script" in n for n in notes)
    assert any("only one" in n for n in notes)
    assert cr.plan_notes(_proto(ERR, SE)) == []
    assert any("no criterion" in n.lower() for n in cr.plan_notes({"oracles": [ORACLE]}))


# --- the plan shows them -------------------------------------------------------------------------------------------


def test_the_plan_has_a_section_in_plain_words_that_is_not_read_back() -> None:
    design = {"hypothesis": "h", "protocol": plan.normalize_protocol(_proto(ERR, DRIFT, SE))[0]}
    text = plan.render("topic", {}, design)
    assert f"## {plan.CRITERIA_HEADING}" in text and plan.CRITERIA_HEADING == "How we will judge whether the code got better"
    section = text.split(f"## {plan.CRITERIA_HEADING}", 1)[1].split("\n## ", 1)[0]
    assert "**rk4 error small**" in section and "lower is better" in section and "1e-05 or below" in section
    assert "a change smaller than 1e-07 counts as no change" in section
    assert "within 0.15 of 0.5" in section and "standard error" in section
    assert "measured by the script" in section  # the energy check has no case
    assert plan.parse(text).design["protocol"]["criteria"] == design["protocol"]["criteria"]


def test_a_plan_with_a_protocol_and_no_criteria_says_so() -> None:
    text = plan.render("topic", {}, {"hypothesis": "h", "protocol": {"oracles": [ORACLE]}})
    section = text.split(f"## {plan.CRITERIA_HEADING}", 1)[1].split("\n## ", 1)[0]
    assert "none" in section.lower()
    assert plan.CRITERIA_HEADING not in plan.render("topic", {}, {"hypothesis": "h"})


# --- frozen with the protocol ----------------------------------------------------------------------------------------


def test_the_frozen_hash_covers_the_criteria_and_a_change_is_an_amendment(tmp_path: Path) -> None:
    with_c = plan.normalize_protocol(_proto(ERR))[0]
    without = plan.normalize_protocol(BASE)[0]
    assert fp.sha256(with_c) != fp.sha256(without)
    record = fp.freeze(tmp_path, with_c, approved_by="human: test", source="plan.md")
    assert record["protocol"]["criteria"][0]["name"] == "rk4 error small"
    changed = {**with_c, "criteria": [{**with_c["criteria"][0], "tolerance": 1e-6}]}
    pending = fp.propose(tmp_path, changed, None, source="test", reason="looser", results_seen=False)
    assert any(line.startswith("criteria") for line in pending["changes"]), pending["changes"]
    ok, _ = fp.approve(tmp_path, "someone", via="test")
    assert ok
    fp.apply(tmp_path, fp.load_pending(tmp_path), fp.approval_for(tmp_path, fp.load_pending(tmp_path)), raw_root=None)
    assert fp.protocol_of(tmp_path)["criteria"][0]["tolerance"] == 1e-6 and fp.load(tmp_path)["version"] == 2
    # An edit of the frozen file's criteria is found by the hash.
    path = fp.frozen_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["protocol"]["criteria"][0]["tolerance"] = 1.0
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "problem" in fp.load(tmp_path)


# --- computing them ------------------------------------------------------------------------------------------------


def _judged(name: str, value: float, expected: float, by: str = "engine") -> dict[str, Any]:
    return {"name": name, "value": value, "expected": expected, "measured_by": by, "passed_by_engine": True}


def test_each_criterion_is_computed_from_what_fi_measured_and_met_is_judged_by_its_direction() -> None:
    crit = cr.normalize([ERR, DRIFT, CASE, SE], _proto())[0]
    rng = random.Random(1)
    series = {"final_size": {"R0=2": [rng.gauss(0, 1) for _ in range(64)]}}
    rows = cr.evaluate(crit, judged=[_judged("rk4 error", 3e-7, 0.0), _judged("energy drift", 2e-9, 0.0, by="script")],
                       case_values={"error at dt 0.05": 2e-8}, series=series)
    by = {r["name"]: r for r in rows}
    assert by["rk4 error small"]["value"] == pytest.approx(3e-7) and by["rk4 error small"]["met"] is True
    assert by["rk4 error small"]["counts"] is True
    # A value the script's own code measured is shown, never counted.
    assert by["energy kept"]["value"] == 2e-9 and by["energy kept"]["counts"] is False
    assert by["error at dt 0.05"]["value"] == 2e-8 and by["error at dt 0.05"]["met"] is None  # no bar was set
    assert by["error bars shrink"]["value"] == pytest.approx(0.5, abs=0.15) and by["error bars shrink"]["met"] is True


def test_trials_that_are_copies_of_each_other_do_not_shrink_the_error_bar() -> None:
    rng = random.Random(2)
    copied = [v for v in (rng.gauss(0, 1) for _ in range(16)) for _ in range(4)]  # each value four times in a row
    rate = cr.se_rate(copied)
    assert rate is not None and rate < 0.35, rate
    assert cr.se_rate([1.0] * 64) is None and cr.se_rate([1.0, 2.0, 3.0]) is None


def test_a_criterion_with_nothing_measured_says_why_and_is_not_met() -> None:
    crit = cr.normalize([ERR, CASE, SE], _proto())[0]
    rows = cr.evaluate(crit, judged=[], case_values={}, series={}, why_missing={"case": "the trials are not run by FI"})
    assert all(r["value"] is None and r["met"] is None and r["why"] for r in rows)
    assert "not run by FI" in {r["name"]: r for r in rows}["error at dt 0.05"]["why"]


def test_the_history_row_and_the_log_line(tmp_path: Path) -> None:
    crit = cr.normalize([ERR, DRIFT], _proto())[0]
    rows = cr.evaluate(crit, judged=[_judged("rk4 error", 3e-4, 0.0), _judged("energy drift", 1e-9, 0.0, by="script")],
                       case_values={}, series={})
    row = cr.record(tmp_path, run="run_1", code_commit="abc123", results=rows)
    again = cr.record(tmp_path, run="run_1", code_commit="def456", results=[])
    history = cr.history(tmp_path)
    assert [h["n"] for h in history] == [1, 2] and history[0]["code_commit"] == "abc123" and row["n"] == 1
    assert again["note"] == "no criterion" and history[1]["criteria"] == []
    line = cr.summary_line(rows)
    assert "0 of 1" in line and "not met" in line and "rk4 error small" in line and "energy kept" in line
    assert "the script" in line
    assert "no criterion" in cr.summary_line([]).lower()
    assert math.isfinite(history[0]["criteria"][0]["value"])
