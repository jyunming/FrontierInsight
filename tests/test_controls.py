"""The study's own controls (core/controls.py): typed in the plan, judged by FI from its own record after the full run,
and disclosed (QUALITY.md, the paper's closing sentence), never a route. Fake model only; the study is made up (a
cooling cup whose cooling rate is measured with and without a lid)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import contract
from core import controls as ctl
from core import plan
from core import quality_pointer
from core import quality_report
from core import rerun_from
from core.engine import Engine, _stripped
from tests.test_cross_check import _mk_cfg

GRID = {"lid": ["on", "off"], "volume_ml": [100, 200]}
METRICS = [{"id": "cooling_rate", "kind": "mean", "estimand": "the cooling rate", "unit": "a trial"},
           {"id": "fitted_rate", "kind": "mean", "estimand": "the rate fitted to all trials", "unit": "a fit"}]
POSITIVE = {"name": "positive control: the open cup", "setting": {"lid": "off", "volume_ml": 100},
            "quantity": "cooling_rate", "expected": 0.5, "tolerance": 0.05}
KIND = "results whose check against a known answer did not pass"


def _protocol(**extra: Any) -> dict[str, Any]:
    return {"grid": dict(GRID), "runs_per_setting": 20, "metrics": [dict(m) for m in METRICS], **extra}


def _cells(values_by_cell: dict[tuple[str, int], list[float]], name: str = "cooling_rate") -> list[dict[str, Any]]:
    return [{"cell": {"lid": lid, "volume_ml": vol}, "metrics": {name: list(vals)}}
            for (lid, vol), vals in values_by_cell.items()]


def _spread(centre: float, half: float, n: int = 20) -> list[float]:
    """``n`` values with mean ``centre``: half at centre - half, half at centre + half."""
    return [centre - half if i % 2 else centre + half for i in range(n)]


# ---- the plan -----------------------------------------------------------------------------------------------------------


def test_a_typed_control_is_read_with_its_tolerance_written_out() -> None:
    out, fault = plan.protocol_fault(_protocol(controls=[POSITIVE]))
    assert fault is None
    assert out["controls"] == [{**POSITIVE, "tolerance_mode": "absolute"}]
    assert ctl.of_protocol(out) == out["controls"]


def test_a_plan_with_no_control_is_valid() -> None:
    out, fault = plan.protocol_fault(_protocol())
    assert fault is None and "controls" not in out
    assert ctl.of_protocol(out) == []


@pytest.mark.parametrize("bad, said", [
    ({**POSITIVE, "quantity": "warmth"}, "`quantity` must be the id of a declared result (cooling_rate, fitted_rate)"),
    ({**POSITIVE, "setting": {"colour": "red"}}, "`colour` is not a setting of `protocol.grid`"),
    ({**POSITIVE, "setting": {"volume_ml": 300}}, "which is not one of its values in `protocol.grid` (100, 200)"),
    ({**POSITIVE, "expected": "about a half"}, "`expected` must be a number"),
    ({**POSITIVE, "tolerance": 0}, "`tolerance` must be a number above 0"),
    ({**POSITIVE, "expected": 0, "tolerance_mode": "relative"}, "an `expected` of 0 takes an absolute `tolerance`"),
    ("the cup with no lid", "must be a mapping with a `name`"),
])
def test_a_control_that_cannot_be_judged_is_left_out_with_a_plain_reason(bad: Any, said: str) -> None:
    _out, fault = plan.protocol_fault(_protocol(controls=[bad]))
    assert fault is not None and said in fault.reason and fault.path == "protocol.controls[1]"
    kept, notes = plan.repair_protocol(_protocol(controls=[POSITIVE, bad]))
    assert kept is not None and kept["controls"] == [{**POSITIVE, "tolerance_mode": "absolute"}]  # the good one stays
    assert len(notes) == 1 and "was left out of `protocol.controls` because it could not be judged" in notes[0]
    assert said in notes[0] and "FI does not judge it" in notes[0]


def test_a_plan_whose_only_control_cannot_be_judged_keeps_every_other_part() -> None:
    kept, notes = plan.repair_protocol(_protocol(controls=[{**POSITIVE, "quantity": "warmth"}]))
    assert kept is not None and "controls" not in kept and kept["grid"] == GRID and len(notes) == 1


def test_the_plan_rule_names_no_route() -> None:
    assert "back to design" not in ctl.PLAN_RULE and "quality report and the paper" in ctl.PLAN_RULE


def test_the_code_writer_never_sees_what_a_control_must_give() -> None:
    shown = _stripped(_protocol(controls=[{**POSITIVE, "expected": 0.4321, "tolerance": 0.0123}],
                                acceptance=["the open cup cools at 0.4321 per minute"]))
    assert "expected" not in shown["controls"][0] and "tolerance" not in shown["controls"][0]
    assert shown["controls"][0]["setting"] == POSITIVE["setting"]  # the setting is an input: kept
    assert "0.4321" not in json.dumps(shown) and "0.0123" not in json.dumps(shown)


# ---- the judge ----------------------------------------------------------------------------------------------------------


def _judge(control: dict[str, Any], cells: list[dict[str, Any]], derived: dict[str, Any] | None = None,
           protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = protocol or _protocol()
    control = ctl.normalize([control], protocol)[0][0]  # type: ignore[index]
    return ctl.judge([control], derived or {}, cells, protocol)[0]


def test_a_control_inside_its_tolerance_passes() -> None:
    v = _judge(POSITIVE, _cells({("off", 100): _spread(0.51, 0.02), ("on", 100): _spread(0.2, 0.02)}))
    assert v["verdict"] == ctl.PASSED and len(v["settings"]) == 1 and v["settings"][0]["on"] == "trials"


def test_a_control_outside_its_tolerance_fails() -> None:
    v = _judge(POSITIVE, _cells({("off", 100): _spread(0.30, 0.02)}))
    assert v["verdict"] == ctl.FAILED
    assert "failed: FI measured 0.3 (mean of 20 trials, standard error 0.0046)" in ctl.words(v)


def test_a_mean_whose_interval_reaches_both_sides_of_the_tolerance_does_not_fail() -> None:
    # mean 0.56, standard error about 0.022: three of them reach from 0.49 to 0.63, across the edge at 0.55
    v = _judge(POSITIVE, _cells({("off", 100): _spread(0.56, 0.1)}))
    assert v["verdict"] == ctl.NOT_DECIDED and ctl.failed([v]) == []


def test_too_few_trials_that_vary_decide_nothing() -> None:
    v = _judge(POSITIVE, _cells({("off", 100): _spread(0.0, 0.01, n=4)}))
    assert v["verdict"] == ctl.NOT_DECIDED


@pytest.mark.parametrize("values", [[1.0], [0.5], [0.3] * 5, [0.5] * 5])
def test_one_trial_or_a_few_equal_trials_of_a_random_study_decide_nothing(values: list[float]) -> None:
    # no standard error is never a standard error of 0: one value (or five that agree) of a study that runs 20
    # trials per setting neither passes nor fails a control
    v = _judge(POSITIVE, _cells({("off", 100): values}))
    assert v["verdict"] == ctl.NOT_DECIDED
    assert "fewer than the 10 FI needs to judge a mean" in v["why"]


def test_a_study_of_one_run_per_setting_is_judged_on_its_value() -> None:
    once = {**_protocol(), "runs_per_setting": 1}
    assert _judge(POSITIVE, _cells({("off", 100): [0.3]}), protocol=once)["verdict"] == ctl.FAILED
    assert _judge(POSITIVE, _cells({("off", 100): [0.52]}), protocol=once)["verdict"] == ctl.PASSED


def test_one_run_of_a_study_whose_words_say_it_is_random_decides_nothing() -> None:
    # the run counts say one run per setting, but the quest's reading (its design names a random process) says the
    # trials repeat: one draw is not a value of 0 error
    once = {**_protocol(), "runs_per_setting": 1}
    control = ctl.normalize([POSITIVE], once)[0][0]  # type: ignore[index]
    v = ctl.judge([control], {}, _cells({("off", 100): [0.3]}), once, repeated=True)[0]
    assert v["verdict"] == ctl.NOT_DECIDED


def test_the_analysis_value_is_found_when_its_setting_lists_the_study_constants_too() -> None:
    # FI's analysis names a setting by its whole cell, the constants of the study included
    derived = {"derived": {"fitted_rate": {"status": "ok", "by_setting": {
        "lid=off,volume_ml=100,room_c=20": 0.2, "lid=off,volume_ml=200,room_c=20": 0.5}}}}
    v = _judge({**POSITIVE, "quantity": "fitted_rate", "setting": {"lid": "off"}}, [], derived)
    assert v["verdict"] == ctl.FAILED and [s["verdict"] for s in v["settings"]] == [ctl.FAILED, ctl.PASSED]


def test_a_control_over_more_settings_than_fi_lists_is_never_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    # FI lists at most a capped number of the study's settings; with a cap of 3 the fourth setting (lid=off,
    # volume_ml=200) is not listed, so a control covering it cannot pass on the first one alone
    monkeypatch.setattr("core.reference_values.MAX_CHECKED_SETTINGS", 3)
    wide = {**POSITIVE, "setting": {"lid": "off"}}
    cells = _cells({("off", 100): _spread(0.5, 0.01), ("off", 200): _spread(0.5, 0.01)})
    v = _judge(wide, cells)
    assert v["verdict"] == ctl.NOT_DECIDED and "FI did not list every setting the control covers" in v["why"]
    # a control whose settings all lie past the cap is not decided either (never judged on the record's own settings)
    v = _judge({**POSITIVE, "setting": {"lid": "off", "volume_ml": 200}}, cells)
    assert v["verdict"] == ctl.NOT_DECIDED and v["settings"] == []
    # a failure at a listed setting still stands
    assert _judge(wide, _cells({("off", 100): _spread(0.1, 0.01)}))["verdict"] == ctl.FAILED
    monkeypatch.setattr("core.reference_values.MAX_CHECKED_SETTINGS", 4)
    assert _judge(wide, cells)["verdict"] == ctl.PASSED  # every setting listed


def test_records_of_one_setting_that_differ_in_a_constant_are_told_apart_by_the_plans_constants() -> None:
    one_axis = {"grid": {"x": [1, 2]}, "constants": {"temp": 20}, "runs_per_setting": 20,
                "metrics": [{"id": "rate", "kind": "mean", "estimand": "the rate", "unit": "a trial"}]}
    control = {"name": "negative control: no effect at x=1", "setting": {"x": 1}, "quantity": "rate",
               "expected": 0.0, "tolerance": 0.01}
    # listed first: a record at another temperature; the plan's constant is temp=20
    derived = {"derived": {"rate": {"status": "ok", "by_setting": {"x=1,temp=99": 1.0, "x=1,temp=20": 0.0}}}}
    v = _judge(control, [], derived, protocol=one_axis)
    assert v["verdict"] == ctl.PASSED and v["settings"][0]["value"] == 0.0
    # neither record at the plan's constant, or both: not decided, and the records are named
    for by in ({"x=1,temp=99": 1.0, "x=1,temp=50": 0.0}, {"x=1,temp=20,run=a": 1.0, "x=1,temp=20,run=b": 0.0}):
        v = _judge(control, [], {"derived": {"rate": {"status": "ok", "by_setting": by}}}, protocol=one_axis)
        assert v["verdict"] == ctl.NOT_DECIDED and "the plan's constants do not pick one" in v["why"]
    # the trials' cells follow the same rule
    cells = [{"cell": {"x": 1, "temp": 99}, "metrics": {"rate": [1.0] * 20}},
             {"cell": {"x": 1, "temp": 20}, "metrics": {"rate": [0.0] * 20}}]
    v = _judge(control, cells, protocol=one_axis)
    assert v["verdict"] == ctl.PASSED and v["settings"][0]["trials"] == 20
    cells[1]["cell"]["temp"] = 50
    assert _judge(control, cells, protocol=one_axis)["verdict"] == ctl.NOT_DECIDED


def test_a_declared_share_is_judged_on_its_binomial_interval() -> None:
    shares = {**_protocol(), "metrics": [*METRICS, {"id": "boiled", "kind": "proportion", "estimand": "the share of "
                                                    "trials in which the cup boils", "unit": "a trial"}]}
    none_boil = {**POSITIVE, "name": "negative control: no cup boils", "quantity": "boiled", "expected": 0.0,
                 "tolerance": 0.01}
    # twenty trials that never boil do not show a share within 0.01 of 0 (their interval reaches about 0.31)
    v = _judge(none_boil, _cells({("off", 100): [0.0] * 20}, name="boiled"), protocol=shares)
    assert v["verdict"] == ctl.NOT_DECIDED and "share" in v["why"] and "0 of 20 trials" in ctl.words(v)
    # two thousand do
    assert _judge(none_boil, _cells({("off", 100): [0.0] * 2000}, name="boiled"), protocol=shares)["verdict"] == ctl.PASSED


def test_a_covered_setting_with_no_value_keeps_the_control_from_passing_and_is_named() -> None:
    wide = {**POSITIVE, "setting": {"lid": "off"}}
    # no trial cell at volume_ml=200
    v = _judge(wide, _cells({("off", 100): _spread(0.5, 0.01)}))
    assert v["verdict"] == ctl.NOT_DECIDED and v["missing"] == [{"lid": "off", "volume_ml": 200}]
    assert "no value of `cooling_rate` at 1 of its settings (volume_ml=200)" in v["why"]
    # a cell at volume_ml=200 that holds no value of the quantity
    cells = _cells({("off", 100): _spread(0.5, 0.01)}) + _cells({("off", 200): [0.5] * 20}, name="other")
    assert _judge(wide, cells)["verdict"] == ctl.NOT_DECIDED
    # a failure at a setting that has a value still stands
    assert _judge(wide, _cells({("off", 100): _spread(0.1, 0.01)}))["verdict"] == ctl.FAILED


def test_the_trials_decide_even_when_the_analysis_writes_a_slightly_different_number() -> None:
    # mean 0.56 with a standard error of about 0.022: not decided against 0.5 within 0.05, whatever the analysis's
    # rounded copy of the mean (0.5599) says on its own
    derived = {"derived": {"cooling_rate": {"status": "ok", "by_setting": {"lid=off,volume_ml=100": 0.5599}}}}
    v = _judge(POSITIVE, _cells({("off", 100): _spread(0.56, 0.1)}), derived)
    assert v["verdict"] == ctl.NOT_DECIDED and v["settings"][0]["on"] == "trials"


def test_a_quantity_not_recorded_at_the_setting_is_not_judged() -> None:
    v = _judge(POSITIVE, _cells({("off", 100): [0.5] * 20}, name="other"))
    assert v["verdict"] == ctl.NOT_JUDGED and "holds no value of `cooling_rate`" in v["why"]
    v = _judge(POSITIVE, _cells({("on", 200): [0.5] * 20}))
    assert v["verdict"] == ctl.NOT_JUDGED and "holds no value of `cooling_rate` at lid=off, volume_ml=100" in v["why"]


def test_a_quantity_the_analysis_works_out_is_judged_on_its_value() -> None:
    derived = {"derived": {"fitted_rate": {"status": "ok", "by_setting": {
        "lid=off,volume_ml=100": 0.2, "lid=on,volume_ml=100": 0.1}}}}
    v = _judge({**POSITIVE, "quantity": "fitted_rate"}, _cells({("off", 100): _spread(0.5, 0.01)}), derived)
    assert v["verdict"] == ctl.FAILED and v["settings"][0]["on"] == "analysis" and "(no interval)" in ctl.words(v)


def test_a_control_over_several_settings_fails_when_one_of_them_fails() -> None:
    wide = {**POSITIVE, "setting": {"lid": "off"}}
    cells = _cells({("off", 100): _spread(0.5, 0.01), ("off", 200): _spread(0.1, 0.01), ("on", 100): _spread(0.1, 0.01)})
    v = _judge(wide, cells)
    assert v["verdict"] == ctl.FAILED and len(v["settings"]) == 2
    assert "at volume_ml=200" in ctl.words(v) and "volume_ml=100" not in ctl.words(v).split("FI measured")[1]
    both = _cells({("off", 100): _spread(0.5, 0.01), ("off", 200): _spread(0.49, 0.01)})
    assert _judge(wide, both)["verdict"] == ctl.PASSED


def test_the_quality_row_keeps_the_numbers_inside_the_length_the_report_keeps() -> None:
    long_name = "positive control: the open cup at the smallest volume, where the answer is known in advance"
    v = _judge({**POSITIVE, "name": long_name}, _cells({("off", 100): _spread(0.30, 0.02)}))
    (text, status), = ctl.quality_rows({"judgements": [{"controls": [v]}]})
    assert status == "not fixed" and "FI measured 0.3" in text[:220]


# ---- the cross-check step: judged and disclosed, never a route ---------------------------------------------------------


def _quest(tmp_path: Path, centre: float, max_iter: int = 3, **cfg: Any) -> tuple[Engine, dict[str, Any]]:
    """An engine whose quest holds the trials' record of a finished run and a design with one control."""
    eng = Engine(_mk_cfg(tmp_path, max_iter=max_iter, **cfg))
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="{}")})()
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]
    root = eng.quest_root
    (root / "raw").mkdir(parents=True, exist_ok=True)
    cells = [{"cell": c["cell"], "key": f"lid={c['cell']['lid']},volume_ml={c['cell']['volume_ml']}",
              "metrics": {"cooling_rate": {"values": c["metrics"]["cooling_rate"]}}}
             for c in _cells({("off", 100): _spread(centre, 0.01), ("on", 100): _spread(0.2, 0.01)})]
    (root / "raw" / "trials.json").write_text(json.dumps({"schema": "fi.trials/v1", "cells": cells}), encoding="utf-8")
    state = {"topic": "cooling of a cup", "iteration": 0, "design": {"protocol": _protocol(controls=[POSITIVE])},
             "analysis": {"key_findings": ["x"], "next_step": "publish"},
             "exec_result": {"returncode": 0}, "result_json": {"cooling_rate": centre}}
    return eng, state


def _closing_sentence(eng: Engine) -> str:
    eng._write_quality_pointer()
    path = eng.quest_root / quality_pointer.FILE
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


@pytest.fixture(autouse=True)
def _nothing_to_look_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.engine._cross_check_items.items", lambda **kw: [])


@pytest.mark.asyncio
async def test_a_failed_control_is_written_up_listed_and_named_by_the_closing_sentence(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.30)
    patch = await eng._node_cross_check(dict(state))
    assert patch["cross_check_route"] == "write" and "iteration" not in patch
    assert eng._route_after_cross_check({**state, **patch}) == "write"
    record = ctl.read_record(eng.quest_root)
    assert record["judgements"][0]["controls"][0]["verdict"] == ctl.FAILED
    report = quality_report.build(eng.quest_root, running=False)
    rows = [p for s in report["stages"] for p in s["problems"]["items"] if p["found_by"] == "FI's control check"]
    assert len(rows) == 1 and rows[0]["status"] == "not fixed" and "failed: FI measured 0.3" in rows[0]["text"]
    assert "positive control: the open cup" in quality_report.render(report)
    said = _closing_sentence(eng)
    assert said.lower().count(KIND) == 1 and said.endswith("are listed in the accompanying quality report.")


@pytest.mark.asyncio
async def test_a_passing_control_adds_no_closing_sentence(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.5)
    patch = await eng._node_cross_check(dict(state))
    assert patch["cross_check_route"] == "write"
    assert ctl.read_record(eng.quest_root)["judgements"][0]["controls"][0]["verdict"] == ctl.PASSED
    assert _closing_sentence(eng) == ""
    report = quality_report.build(eng.quest_root, running=False)
    assert not [p for s in report["stages"] for p in s["problems"]["items"] if p["found_by"] == "FI's control check"]


@pytest.mark.asyncio
async def test_a_control_not_decided_is_in_the_quality_report_only(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.5)
    wide = {**POSITIVE, "setting": {"lid": "off"}}  # no trials at volume_ml=200
    state["design"] = {"protocol": _protocol(controls=[wide])}
    await eng._node_cross_check(dict(state))
    assert ctl.read_record(eng.quest_root)["judgements"][0]["controls"][0]["verdict"] == ctl.NOT_DECIDED
    assert _closing_sentence(eng) == ""
    report = quality_report.build(eng.quest_root, running=False)
    rows = [p for s in report["stages"] for p in s["problems"]["items"] if p["found_by"] == "FI's control check"]
    assert len(rows) == 1 and rows[0]["status"] == "left out" and "volume_ml=200" in rows[0]["text"]


@pytest.mark.asyncio
async def test_a_failed_control_and_a_failed_known_answer_check_share_one_kind(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.30)
    await eng._node_cross_check(dict(state))
    from core import accepted_checks

    (eng.quest_root / "needs").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "needs" / accepted_checks.ORACLE_RECORD).write_text(json.dumps(
        {"status": accepted_checks.WENT_ON, "went_on": [{"name": "a known case"}]}), encoding="utf-8")
    assert _closing_sentence(eng).lower().count(KIND) == 1


@pytest.mark.asyncio
async def test_the_controls_are_judged_when_the_literature_check_is_off(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.30, per_finding_k=0)
    patch = await eng._node_cross_check(dict(state))
    assert patch == {"cross_check": [], "cross_check_route": "write"}
    assert ctl.read_record(eng.quest_root)["judgements"][0]["controls"][0]["verdict"] == ctl.FAILED


@pytest.mark.asyncio
async def test_a_run_that_did_not_finish_judges_no_control(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.30)
    patch = await eng._node_cross_check({**state, "exec_result": {"returncode": 1}})
    assert patch["cross_check_route"] == "write"
    verdict = ctl.read_record(eng.quest_root)["judgements"][0]["controls"][0]
    assert verdict["verdict"] == ctl.NOT_JUDGED and verdict["why"] == "the run gave no results to judge it on"


@pytest.mark.asyncio
async def test_a_run_whose_result_holds_no_number_judges_no_control(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.30)
    await eng._node_cross_check({**state, "result_json": {"status": "done"}})
    assert ctl.read_record(eng.quest_root)["judgements"][0]["controls"][0]["verdict"] == ctl.NOT_JUDGED


@pytest.mark.asyncio
async def test_a_study_of_collected_data_is_judged_on_fis_analysis_record(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.5)
    (eng.quest_root / "raw" / "trials.json").unlink()
    no_sim = {**state, "no_simulation_resolved": True, "exec_result": {}, "result_json": {}}
    await eng._node_cross_check(dict(no_sim))
    verdict = ctl.read_record(eng.quest_root)["judgements"][-1]["controls"][0]
    assert verdict["verdict"] == ctl.NOT_JUDGED and "analyses collected data" in verdict["why"]
    derived = eng.quest_root / "data" / "results" / "derived.json"
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_text(json.dumps({"derived": {"cooling_rate": {"status": "ok", "by_setting": {
        "lid=off,volume_ml=100": 0.3}}}}), encoding="utf-8")
    await eng._node_cross_check(dict(no_sim))
    verdict = ctl.read_record(eng.quest_root)["judgements"][-1]["controls"][0]
    assert verdict["verdict"] == ctl.FAILED and verdict["settings"][0]["on"] == "analysis"


@pytest.mark.asyncio
async def test_a_plan_with_no_control_leaves_no_old_judgement_behind(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.30)
    await eng._node_cross_check(dict(state))
    assert ctl.has_failed(eng.quest_root)
    state["design"] = {"protocol": _protocol()}  # the plan was done again, without the control
    patch = await eng._node_cross_check(dict(state))
    assert patch["cross_check_route"] == "write"
    assert not (eng.quest_root / ctl.RECORD_PATH).exists() and _closing_sentence(eng) == ""
    report = quality_report.build(eng.quest_root, running=False)
    assert not [p for s in report["stages"] for p in s["problems"]["items"] if p["found_by"] == "FI's control check"]


def test_doing_the_cross_check_or_an_earlier_step_again_moves_the_judgement_aside(tmp_path: Path) -> None:
    for step in ("crosscheck", "analysis", "run", "code", "design", "plan"):
        assert ".fi/controls.json" in rerun_from.outputs_of(step), step
    for step in ("evidence", "writing", "claims", "review"):
        assert ".fi/controls.json" not in rerun_from.outputs_of(step), step
    (tmp_path / ".fi").mkdir()
    (tmp_path / ctl.RECORD_PATH).write_text("{}", encoding="utf-8")
    dest, moved = rerun_from.back_up(tmp_path, "plan")
    assert ".fi/controls.json" in moved and not (tmp_path / ctl.RECORD_PATH).exists()
    assert dest is not None and (dest / ".fi" / "controls.json").exists()


@pytest.mark.asyncio
async def test_the_precision_route_is_unchanged(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, centre=0.30)  # a failed control does not change it either way
    eng._precision_missed = lambda s, quiet=False: ["cooling_rate"]  # type: ignore[method-assign]
    state["analysis"] = {"key_findings": ["x"], "next_step": "re_experiment"}
    patch = await eng._node_cross_check(dict(state))
    assert patch["cross_check_route"] == "redesign" and patch["iteration"] == 1
    assert "FAILED THE PLAN'S OWN CONTROLS" not in eng._design_prompt({**state, **patch})


# ---- the seams: the plan file and the contract ----------------------------------------------------------------------------


def test_a_control_survives_the_plan_file_and_the_contract_places_it_without_a_gap() -> None:
    design = {"hypothesis": "a lid slows the cooling", "method": "measure the cooling rate", "dependencies": [],
              "figures_planned": [], "protocol": plan.protocol_fault(_protocol(controls=[POSITIVE]))[0]}
    text = plan.render("cooling of a cup", {"in_short": "s"}, design)
    parsed = plan.parse(text)
    assert parsed.error is None and parsed.design["protocol"]["controls"] == design["protocol"]["controls"]
    c = contract.from_design(parsed.design)
    assert c.checks.controls == tuple(design["protocol"]["controls"])
    assert not any("controls" in g for g in c.gaps)
