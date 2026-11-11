"""Trying the plan on a small case before the study (``pauses.plan: after_trial``, core/feasibility.py).

The record's checks are read from FI's own records (the small run, the pilot, the timing); a ``fail`` is only an exact
fault left after its repair, what a small sample can explain is ``not_shown``. The engine's side: the trial runs the
checks in which no model reads the code and the timing, writes the record, the marker and stops; a resume goes on unless
the plan or the code changed; ``--approve-as`` on that resume is named at the freeze with every change made after the
trial. A study that does not ask for a trial is untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import feasibility as feas
from core import frozen_protocol, plan, run_estimate, small_run
from core.config import Config, PausesConfig
from tests import exact_timing
from tests.test_run_checks import SLOW_SIM, _sizing


@pytest.fixture(autouse=True)
def _exact_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    exact_timing.install(monkeypatch)
    from core.engine import Engine

    async def off(self: Any, state: Any) -> None:
        return None

    # the fake plan model's first reply is the plan (as in tests/test_run_checks.py): the reading of the topic is left out
    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)


PROTOCOL: dict[str, Any] = {
    "grid": {"load": [1, 2, 3], "width": [0.5, 1.0]},
    "runs_per_setting": 10,
    "metrics": [{"id": "stretch", "kind": "mean", "returned": "each_run"},
                {"id": "ratio", "kind": "mean", "returned": "each_run"}],
}


def _small(values: list[dict[str, list[Any]]], **record: Any) -> dict[str, Any]:
    settings = [{"key": f"s{i}", "planned": 3, "ok": 3, "failed": 0, "timed_out": False, "reasons": [], "values": v}
                for i, v in enumerate(values)]
    returned = sorted({n for v in values for n in v})
    return {"says": "9 trial(s) at 3 setting(s) ran", "crashed": False, "load_error": "", "returned": returned,
            "reads": {"measurable": True, "read": ["load", "width"], "unknown": [], "never": [], "why": ""},
            "settings": settings, **record}


GOOD = _small([{"stretch": [1.0, 1.1, 0.9], "ratio": [0.2, 0.3, 0.2]}] * 3)


# ---- the six checks, each from a record --------------------------------------------------------------------------------


def test_only_after_trial_asks_for_a_trial() -> None:
    assert not feas.asked(Config(topic="t", pauses=PausesConfig()))
    assert not feas.asked(Config(topic="t", pauses=PausesConfig(plan="ask")))
    assert feas.asked(Config(topic="t", pauses=PausesConfig(plan="after_trial")))


def test_runs_says_what_ran_and_fails_only_after_the_repairs() -> None:
    ok = feas.check_runs(GOOD)
    assert ok["result"] == "pass" and ok["says"] == "The simulation ran to the end at the 3 settings FI tried (9 trials)."
    assert feas.check_runs({**GOOD, "crashed": True, "says": "the simulation could not be loaded: boom"})["result"] == "fail"
    late = _small([{"stretch": [1.0]}] * 2)
    late["settings"][1]["timed_out"] = True
    assert feas.check_runs(late)["result"] == "not_shown"
    assert feas.check_runs({})["result"] == "not_shown"
    failed = feas.check_runs(GOOD, timing_failed="at load=3: ZeroDivisionError")
    assert failed["result"] == "fail" and "ZeroDivisionError" in failed["says"]


def test_numbers_counts_settings_where_a_number_is_not_finite_and_never_fails_on_a_small_sample() -> None:
    mixed = _small([{"stretch": [1.0], "ratio": [float("nan")]}, {"stretch": [2.0], "ratio": [0.4]},
                    {"stretch": [3.0], "ratio": [float("inf")]}])
    got = feas.check_numbers(PROTOCOL, mixed, {})
    assert got["result"] == "pass"
    assert got["says"].startswith("Every number the plan declares had a finite value at one or more of the 3 settings")
    assert "`ratio`: not finite at 2 of 3 settings" in got["says"]
    never = _small([{"stretch": [1.0], "ratio": [float("nan")]}] * 3)
    assert feas.check_numbers(PROTOCOL, never, {})["result"] == "not_shown", "no value at every small setting: not shown"
    # what the pilot found and its repairs left is the only fail
    pilot = {"outcome": "stopped", "problem_items": [{"metric": "ratio", "kind": "undefined",
                                                       "says": "the metric `ratio` had no finite value"}]}
    assert feas.check_numbers(PROTOCOL, GOOD, pilot)["result"] == "fail"
    assert feas.check_numbers(PROTOCOL, GOOD, {**pilot, "outcome": "repairing"})["result"] == "pass"


def test_results_fail_for_an_exact_fault_and_not_for_an_unchecked_analysis() -> None:
    assert feas.check_results(PROTOCOL, GOOD, {})["result"] == "pass"
    missing = _small([{"stretch": [1.0, 1.1, 0.9]}] * 3, sent_back={"faults": ["absent:ratio"], "count": 1})
    got = feas.check_results(PROTOCOL, missing, {})
    assert got["result"] == "fail" and "`ratio`" in got["says"] and "sent back to be repaired once" in got["says"]
    unchecked = {**GOOD, "analysis": {"ok": False, "stage": "", "kind": "not_checked", "detail": "docker"}}
    assert feas.check_results(PROTOCOL, unchecked, {}, docker=True)["result"] == "not_shown"
    partial = _small([{"stretch": [1.0]}] * 3)
    partial["settings"][0]["failed"] = 1
    assert feas.check_results(PROTOCOL, partial, {})["result"] == "not_shown", "a result no trial returned needs a whole run"


def test_moves_fails_only_for_a_setting_nothing_asks_for() -> None:
    assert feas.check_moves(PROTOCOL, GOOD, {"outcome": "passed"}, {})["says"] == (
        "Each setting the study compares was asked for by the code at the settings FI tried, and the pilot found no "
        "result that stays the same across them.")
    assert feas.check_moves(PROTOCOL, GOOD, {}, {}, unread=["width"])["result"] == "fail"
    assert feas.check_moves(PROTOCOL, GOOD, {}, {}, conditional=["width"])["result"] == "not_shown"
    flat = {"outcome": "stopped", "problem_items": [{"kind": "flat", "says": "every result is the same"}]}
    assert feas.check_moves(PROTOCOL, GOOD, flat, {})["result"] == "fail"
    assert feas.check_moves({"grid": {"load": [1]}}, GOOD, {}, {})["result"] == "not_applicable"
    unmeasured = {**GOOD, "reads": {"measurable": False, "never": [], "why": "the small run did not run every setting"}}
    assert feas.check_moves(PROTOCOL, unmeasured, {}, {})["result"] == "not_shown"


def test_time_says_the_estimate_against_the_time_allowed() -> None:
    fits = feas.check_time({"fits": True, "estimate_s": 3 * 3600 + 600, "limit_s": 4 * 3600})
    assert fits["result"] == "pass" and fits["says"] == "The whole study fits the time allowed: about 3 h 10 min of the 4 h."
    cut = feas.check_time({"fits": True, "estimate_s": 3000, "limit_s": 3600,
                           "reduced": {"from": {"runs_per_setting": 30}, "to": {"runs_per_setting": 10}}})
    assert cut["result"] == "pass" and cut["small_change"] == "fewer runs per setting: 30 -> 10, to fit 1 h"
    assert "with 10 runs per setting instead of 30" in cut["says"]
    stopped = feas.check_time({"fits": False, "estimate_s": 9e4, "limit_s": 3600, "stopped": "it would take 25 h"})
    assert stopped["result"] == "fail" and "25 h" in stopped["says"]
    truncated = feas.check_time({"fits": False, "estimate_s": 9e4, "limit_s": 3600,
                                 "truncated": {"estimate_s": 9e4, "settings": 40, "fit": {"settings_that_fit": 2,
                                                                                         "settings": 40}}})
    assert truncated["result"] == "fail" and "about 2 of its 40 settings fit" in truncated["says"]
    assert feas.check_time({"fits": True, "estimate_s": 60, "limit_s": 3600, "at_least": True})["result"] == "not_shown"
    assert feas.check_time({})["result"] == "not_shown"


def test_data_counts_the_supplied_files_but_not_the_papers(tmp_path: Path) -> None:
    assert feas.check_data(tmp_path)["result"] == "not_applicable"
    (tmp_path / "inputs" / "papers").mkdir(parents=True)
    (tmp_path / "inputs" / "papers" / "a.pdf").write_text("x", encoding="utf-8")
    assert feas.check_data(tmp_path)["result"] == "not_applicable"
    (tmp_path / "inputs" / "table.csv").write_text("a,b\n", encoding="utf-8")
    assert feas.check_data(tmp_path)["says"] == "The study's supplied files are there: 1 file in inputs/."


def test_the_verdict_fails_on_any_failed_check_and_is_feasible_only_with_every_goal_answered() -> None:
    passed = [{"id": c, "result": "pass", "says": "ok."} for c in feas.CHECKS]
    assert feas.verdict(passed, feas.goals_not_read()) == "unsure", "the goals are not read in a trial yet"
    assert feas.verdict(passed, {"state": "answered"}) == "feasible"
    failed = [*passed[:4], {"id": "time", "result": "fail", "says": "The whole study does not fit the time allowed."},
              passed[5]]
    assert feas.verdict(failed, {"state": "answered"}) == "not_feasible"
    assert feas.line(failed, {}, "not_feasible") == "Not feasible: the whole study does not fit the time allowed."
    shown = [*passed[:1], {"id": "numbers", "result": "not_shown", "says": "x."}, *passed[2:]]
    assert feas.line(shown, feas.goals_not_read(), "unsure") == (
        "Unsure: nothing FI checked failed; FI could not show that its numbers are finite; FI does not read the study's "
        "goals in a trial yet.")


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "small_run.json").write_text(json.dumps(GOOD), encoding="utf-8")
    (root / ".fi" / "pilot_check.json").write_text(json.dumps({"outcome": "passed"}), encoding="utf-8")
    (root / ".fi" / "run_estimate.json").write_text(json.dumps({"fits": True, "estimate_s": 1800, "limit_s": 3600}),
                                                    encoding="utf-8")
    return root


def test_the_record_is_built_from_the_records_and_says_one_sentence_per_check(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    record = feas.build(root, PROTOCOL, quest_id="q1", plan_sha256="abc", code_version="v1")
    assert record["schema"] == "fi.feasibility/v1" and [c["id"] for c in record["checks"]] == list(feas.CHECKS)
    assert all(c["says"].endswith(".") for c in record["checks"])
    assert record["verdict"] == "unsure" and "does not read the study's goals" in record["line"]
    assert record["goals"]["state"] == "could_not_judge" and record["changed_after_trial"] is None
    assert record["estimate"]["seconds"] == 1800 and record["estimate"]["other_studies_running"] is None
    # a timing stop that no record shows still fails the time
    stopped = feas.build(root, PROTOCOL, quest_id="q1", plan_sha256="abc", code_version="v1",
                         stopped="it would take 9 h", stopped_kind="too_long")
    assert stopped["verdict"] == "not_feasible" and "9 h" in stopped["line"]


def test_the_stop_writes_the_record_then_the_marker_and_a_resume_goes_on_unless_something_changed(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    record = feas.build(root, PROTOCOL, quest_id="q1", plan_sha256="p1", code_version="c1")
    feas.write(root, record, PROTOCOL)
    feas.write_marker(root, record)
    assert feas.read(root)["plan_sha256"] == "p1" and not list((root / "needs").glob("*.tmp"))
    assert feas.on_resume(root, plan_sha256="p1", code_version="c1") == feas.DONE
    assert feas.on_resume(root, plan_sha256="p2", code_version="c1") == feas.AGAIN, "the plan changed"
    assert feas.on_resume(root, plan_sha256="p1", code_version="c2") == feas.AGAIN, "the code changed"
    (root / "inputs").mkdir()
    (root / "inputs" / "study_inputs.json").write_text("{}", encoding="utf-8")
    assert feas.on_resume(root, plan_sha256="p1", code_version="c1") == feas.AGAIN, "a value file changed"
    (root / "inputs" / "study_inputs.json").unlink()
    (root / "needs" / "FEASIBILITY.json").write_text('{"schema": "fi.feasibility/v1"}', encoding="utf-8")
    assert feas.on_resume(root, plan_sha256="p1", code_version="c1") == feas.AGAIN, "a partial record"
    # once the study went on after the trial, a repair during the run never stops it at the trial again
    feas.went_on(root)
    assert feas.on_resume(root, plan_sha256="p9", code_version="c9") == feas.DONE
    # a second trial keeps the first record under .fi/feasibility/
    feas.write(root, record, PROTOCOL)
    assert list((root / ".fi" / "feasibility").glob("record_*.json"))


def test_an_approval_counts_only_for_the_plan_that_was_tried(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    assert not feas.note_approval(root, "A. Person"), "no trial, nothing to approve"
    record = feas.build(root, PROTOCOL, quest_id="q1", plan_sha256="p1", code_version="c1")
    feas.write(root, record, PROTOCOL)
    feas.write_marker(root, record)
    assert feas.note_approval(root, "  A.   Person ")
    assert feas.approval(root)["who"] == "A. Person"
    feas.write_marker(root, {**record, "plan_sha256": "p2"})  # tried again on a changed plan
    assert feas.approval(root) is None


def test_the_changes_after_the_trial_name_the_check_that_made_them(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    assert feas.changed_after_trial(root, PROTOCOL) is None, "nothing was tried"
    feas.write(root, feas.build(root, PROTOCOL, quest_id="q", plan_sha256="p", code_version="c"), PROTOCOL)
    frozen = {**PROTOCOL, "runs_per_setting": 8, "oracles": [{"name": "limit", "expected": 1.0}]}
    changes = feas.changed_after_trial(root, frozen)
    by = {c["change"].split(":", 1)[0]: c["by"] for c in changes}
    assert by["runs_per_setting"] == "FI's timing (fewer runs per setting)"
    assert any(v == "FI's known-answer checks" for k, v in by.items() if k.startswith("oracles"))
    text = feas.approved_by(root, "A. Person", "p" * 64, changes)
    assert text.startswith("human: A. Person approved the tried plan (sha256 pppppppppppp; pauses.plan=after_trial)")
    assert "not seen by A. Person" in text and "runs_per_setting: 10 -> 8" in text


def test_the_stop_card_gives_the_command_that_runs_the_tried_study(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    record = feas.build(root, PROTOCOL, quest_id="q1", plan_sha256="p", code_version="c")
    headline, steps, recommended, alternatives = feas.stop_card(record, root, "q1")
    assert headline.startswith("FI tried the plan on a small case")
    assert any("--resume q1 --approve-as <you>" in s for s in steps)
    assert recommended and alternatives
    lines = feas.report_lines(feas.summary(root) or {"line": record["line"], "checks": record["checks"]})
    assert lines[0].startswith("Before the study, FI tried the plan on a small case")


# ---- the engine: the trial, the stop, the resume, the freeze -----------------------------------------------------------


class Paused(Exception):
    """Stands in for LangGraph's GraphInterrupt."""


def _catch_pause(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []

    def interrupted(payload: Any) -> Any:
        seen.append(payload)
        raise Paused(payload["pause"]["kind"])

    monkeypatch.setattr("core.engine.interrupt", interrupted)
    return seen


async def _trial_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, timeout_s: int = 3600,
                        mode: str = "after_trial") -> tuple[Any, Any, dict[str, Any], list[int]]:
    """A quest with a plan and a simulation of the trial contract, its small run and its timing real; the checks before
    them in the first part of the gate (static, stub, labels) are left out (their own tests cover them)."""
    eng, _model, runner, _ex = await _sizing(tmp_path, timeout_s=timeout_s)
    eng.config.pauses.plan = mode
    eng._trial_mode, eng._trial_entries = True, {"run_trial"}
    calls = [0]
    chat = eng._client.chat

    async def counted(*a: Any, **k: Any) -> Any:
        calls[0] += 1
        return await chat(*a, **k)

    eng._client = type("Stub", (), {"chat": counted})()

    async def first(state: Any, py: Any, env: Any) -> Any:
        return await eng._small_run_gate(state, py, env)

    monkeypatch.setattr(eng, "_gate_pass_first", first)
    state: dict[str, Any] = {"topic": "Measure how a damped spring settles", "iteration": 0}
    return eng, runner, state, calls


@pytest.mark.asyncio
async def test_a_study_that_does_not_ask_for_a_trial_runs_exactly_as_before(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, state, calls = await _trial_engine(tmp_path, monkeypatch, mode="off")
    seen = _catch_pause(monkeypatch)

    async def must_not_run(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("nothing of the trial runs for a study that did not ask for one")

    monkeypatch.setattr(eng, "_gate_pass_first", must_not_run)
    monkeypatch.setattr(eng, "_size_the_run", must_not_run)
    assert await eng._try_plan_first(state, runner, sys.executable, None) is None
    assert eng._stop_not_tried(state, feas.NOT_TRIED_YET) is None
    assert seen == [] and calls[0] == 0
    assert not (eng.quest_root / "needs" / "FEASIBILITY.json").exists()
    assert not (eng.quest_root / ".fi" / "paused_at_trial.flag").exists()
    # and the full checks before the run start with the same first part, once
    count = [0]

    async def first(*_a: Any, **_k: Any) -> Any:
        count[0] += 1
        return {"exec_result": {"returncode": 1}}

    monkeypatch.setattr(eng, "_gate_pass_first", first)
    stop, _code, _rewrote = await eng._gate_pass(state, sys.executable, None, eng.quest_root / "code" / "simulate.py")
    assert stop == {"exec_result": {"returncode": 1}} and count[0] == 1


@pytest.mark.asyncio
async def test_the_trial_writes_its_record_and_stops_and_the_resume_goes_on(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, state, calls = await _trial_engine(tmp_path, monkeypatch)
    seen = _catch_pause(monkeypatch)
    with pytest.raises(Paused):
        await eng._try_plan_first(state, runner, sys.executable, None)
    record = feas.read(eng.quest_root)
    assert record["schema"] == "fi.feasibility/v1" and record["quest_id"] == eng.quest_id
    by = {c["id"]: c for c in record["checks"]}
    assert by["runs"]["result"] == "pass" and by["time"]["result"] == "pass", record
    assert by["time"]["says"].startswith("The whole study fits the time allowed: about")
    assert record["verdict"] == "unsure" and calls[0] == 0, "the trial asks no model when nothing has to be repaired"
    marker = feas.read_marker(eng.quest_root)
    assert marker and marker["plan_sha256"] == record["plan_sha256"] and marker["went_on"] is False
    pause = json.loads((eng.quest_root / ".fi" / "pause.json").read_text(encoding="utf-8"))
    assert pause["kind"] == "trial" and pause["interaction"] == "supply"
    assert seen[0]["trial_stage"] is True and "plan_stage" not in seen[0]
    assert "--approve-as <you>" in (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert not (eng.quest_root / "raw").exists(), "nothing the trial ran is a result"
    # the resume: nothing changed, so the full checks and the study go on, and the trial is not made again
    small_before = small_run.read_record(eng.quest_root)

    async def must_not_run(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the trial is not made again")

    monkeypatch.setattr(eng, "_gate_pass_first", must_not_run)
    assert await eng._try_plan_first(state, runner, sys.executable, None) is None
    assert feas.read_marker(eng.quest_root)["went_on"] is True
    assert small_run.read_record(eng.quest_root) == small_before


@pytest.mark.asyncio
async def test_a_plan_edited_during_the_stop_is_tried_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, state, _calls = await _trial_engine(tmp_path, monkeypatch)
    _catch_pause(monkeypatch)
    with pytest.raises(Paused):
        await eng._try_plan_first(state, runner, sys.executable, None)
    first = feas.read(eng.quest_root)
    path = plan.plan_path(eng.quest_root)
    edited = plan.edit_design_block(path.read_text(encoding="utf-8"), lambda d: {
        **d, "protocol": {**d["protocol"], "grid": {"n": [1, 2]}}})
    assert edited is not None
    path.write_text(edited, encoding="utf-8")
    with pytest.raises(Paused):
        await eng._try_plan_first(state, runner, sys.executable, None)
    assert feas.read(eng.quest_root)["plan_sha256"] != first["plan_sha256"]
    assert list((eng.quest_root / ".fi" / "feasibility").glob("record_*.json")), "the first record is kept"


@pytest.mark.asyncio
async def test_a_study_too_long_is_a_record_of_the_trial_not_the_quests_stop(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, state, _calls = await _trial_engine(tmp_path, monkeypatch, timeout_s=1)
    _catch_pause(monkeypatch)
    with pytest.raises(Paused):  # the trial's own pause, not a stop returned to the graph
        await eng._try_plan_first(state, runner, sys.executable, None)
    record = feas.read(eng.quest_root)
    time_check = next(c for c in record["checks"] if c["id"] == "time")
    assert time_check["result"] == "fail" and record["verdict"] == "not_feasible"
    assert record["line"].startswith("Not feasible: ")
    assert not (eng.quest_root / "needs" / "STUCK.json").exists()


@pytest.mark.asyncio
async def test_a_setting_that_fails_when_timed_goes_to_the_repair_and_after_the_last_repair_is_a_record(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, state, _calls = await _trial_engine(tmp_path, monkeypatch)
    _catch_pause(monkeypatch)
    failed = eng._sizing_failed("at n=4: ZeroDivisionError")

    async def sizing(*_a: Any, **_k: Any) -> Any:
        return failed

    monkeypatch.setattr(eng, "_size_the_run", sizing)
    assert await eng._try_plan_first(state, runner, sys.executable, None) is failed, "a repair is left: the repair"
    spent = {**state, "exec_reflect_iter": eng.config.engine.exec_reflect_max_iterations}
    with pytest.raises(Paused):
        await eng._try_plan_first(spent, runner, sys.executable, None)
    runs = next(c for c in feas.read(eng.quest_root)["checks"] if c["id"] == "runs")
    assert runs["result"] == "fail" and "ZeroDivisionError" in runs["says"]


@pytest.mark.asyncio
async def test_a_failed_run_of_the_first_checks_goes_to_the_repair_as_without_a_trial(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, state, _calls = await _trial_engine(tmp_path, monkeypatch)
    seen = _catch_pause(monkeypatch)
    back = eng._code_review_failed_run("repair this", script="simulate.py")

    async def first(*_a: Any, **_k: Any) -> Any:
        return back

    monkeypatch.setattr(eng, "_gate_pass_first", first)
    assert await eng._try_plan_first(state, runner, sys.executable, None) is back
    assert seen == [] and not (eng.quest_root / "needs" / "FEASIBILITY.json").exists()


@pytest.mark.asyncio
async def test_a_study_with_no_simulation_gets_one_record_that_says_it_was_not_tried(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, _runner, state, _calls = await _trial_engine(tmp_path, monkeypatch)
    _catch_pause(monkeypatch)
    nosim = {**state, "no_simulation_resolved": True, "auto_collected_count": 3}
    with pytest.raises(Paused):
        eng._stop_not_tried(nosim, feas.NOT_TRIED_YET)
    record = feas.read(eng.quest_root)
    assert record["verdict"] == "unsure" and all(c["result"] == "not_applicable" for c in record["checks"])
    assert "FI collected 3 file(s)" in next(c for c in record["checks"] if c["id"] == "data")["says"]
    assert eng._stop_not_tried(nosim, feas.NOT_TRIED_YET) is None, "once per quest: the resume goes on"


@pytest.mark.asyncio
async def test_the_freeze_names_who_approved_the_tried_plan_and_every_change_after_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, state, _calls = await _trial_engine(tmp_path, monkeypatch)
    _catch_pause(monkeypatch)
    with pytest.raises(Paused):
        await eng._try_plan_first(state, runner, sys.executable, None)
    assert feas.note_approval(eng.quest_root, "A. Person")
    # after the trial, one of FI's checks changes the protocol (here the runs per setting)
    path = plan.plan_path(eng.quest_root)
    edited = plan.edit_design_block(path.read_text(encoding="utf-8"), lambda d: {
        **d, "protocol": {**d["protocol"], "runs_per_setting": 7}})
    path.write_text(edited, encoding="utf-8")
    eng._freeze_protocol_if_due(state)
    frozen = frozen_protocol.load(eng.quest_root)
    assert frozen["approved_by"].startswith("human: A. Person approved the tried plan (sha256 ")
    assert "not seen by A. Person" in frozen["approved_by"] and "runs_per_setting" in frozen["approved_by"]
    changed = feas.read(eng.quest_root)["changed_after_trial"]
    assert [c["by"] for c in changed] == ["FI's timing (fewer runs per setting)"]


@pytest.mark.asyncio
async def test_without_approval_the_freeze_says_nobody_approved_it_as_before(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, state, _calls = await _trial_engine(tmp_path, monkeypatch)
    _catch_pause(monkeypatch)
    with pytest.raises(Paused):
        await eng._try_plan_first(state, runner, sys.executable, None)
    eng._freeze_protocol_if_due(state)
    frozen = frozen_protocol.load(eng.quest_root)
    assert frozen["approved_by"].startswith("auto: nobody approved this protocol before the run (pauses.plan=after_trial)")
    assert feas.read(eng.quest_root)["changed_after_trial"] == []


@pytest.mark.asyncio
async def test_a_tried_plan_is_not_rewritten_to_fit_the_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, _state, _calls = await _trial_engine(tmp_path, monkeypatch, timeout_s=1, mode="off")
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and patch["exec_result"].get("too_long")
    state = {"exec_result": patch["exec_result"], "result_json": {}, "exec_give_up_reason": "x", "iteration": 0}
    assert eng._no_results_verdict(state)["redesign"] is True, "without a trial: the plan is asked once for a fit"
    eng.config.pauses.plan = "after_trial"
    verdict = eng._no_results_verdict(state)
    assert verdict["stuck"] is True and not verdict.get("redesign")
    assert run_estimate.read_record(eng.quest_root).get("stopped")
