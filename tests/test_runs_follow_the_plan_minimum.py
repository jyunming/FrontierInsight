"""A plan's run counts are read one way: ``mc_error.repeats_trials`` is the one reading of "does this plan repeat trials", so a
plan with one run per setting beside a minimum of 200 is a random study in the frame, the raw-data check and the checks of a
random quantity alike, and before the protocol is frozen FI makes the planned runs the plan's own minimum (never undoing a cut
FI made itself for the time allowed; a minimum that does not fit the time runs the most runs that fit, disclosed and marked)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from core import mc_error, plan, skeleton
from core.engine import Engine

@pytest.mark.parametrize("protocol, words, expected", [
    ({"runs_per_setting": 1, "min_runs_per_setting": 200}, "", True),   # the plan that contradicts itself
    ({"runs_per_setting": 1}, "", False),
    ({"runs_per_setting": 300}, "", True),
    ({"min_runs_per_setting": 20}, "", True),
    ({}, "", False),
    ({"runs_per_setting": 1}, "a stochastic simulation of a colony", True),  # the design's own words still count
])
def test_one_function_reads_whether_a_design_repeats_trials(protocol: dict[str, Any], words: str, expected: bool) -> None:
    design = {"hypothesis": words or "h", "protocol": protocol}
    assert mc_error.design_repeats_trials(design) is expected
    # the frame is written from that one answer: a random study gets the entry FI calls once per trial
    frame = skeleton.frame_source(design, stochastic=mc_error.design_repeats_trials(design))
    assert ("def run_trial" in frame) is expected


def test_the_two_old_readings_are_one_function_now() -> None:
    import core.split_run as split_run

    assert not hasattr(split_run, "design_is_stochastic")
    protocol = {"runs_per_setting": 1, "min_runs_per_setting": 200}
    assert mc_error.repeats_trials(protocol) is True and mc_error.design_repeats_trials({"protocol": protocol}) is True


@pytest.mark.asyncio
async def test_a_plan_whose_runs_are_below_its_own_minimum_runs_the_minimum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_run_checks import _sizing

    async def off(self: Engine, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)
    eng, _model, runner, _ex = await _sizing(tmp_path, timeout_s=3600, runs=1, min_runs_per_setting=40)
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 1
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    protocol = plan.load_design(eng.quest_root)[0]["protocol"]
    assert protocol["runs_per_setting"] == 40 and protocol["min_runs_per_setting"] == 40
    assert eng._protocol_block({"iteration": 0})["runs_per_setting"] == 40
    assert "FI runs 40 per setting" in (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert eng._design_after_sizing({"design": {"protocol": {"runs_per_setting": 1}}})["protocol"]["runs_per_setting"] == 40


@pytest.mark.asyncio
async def test_a_minimum_that_does_not_fit_the_time_runs_the_most_runs_that_fit_disclosed_and_marked(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan's minimum does not fit: FI does not stop. It runs the most runs per setting that fit (at least two), records its
    own reduction (the paper's fewer-runs item) and marks every result of the plan ``(unconfirmed)``."""
    from core import paper_facts, run_estimate
    from tests.test_run_checks import _sizing

    async def off(self: Engine, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)
    eng, _model, runner, _ex = await _sizing(tmp_path, timeout_s=8, runs=1, min_runs_per_setting=200)
    out = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert out is None, out
    runs = plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"]
    assert 2 <= runs < 200
    record = run_estimate.read_record(eng.quest_root)
    reduced = record["reduced"]
    assert reduced["marked_unconfirmed"] and reduced["below_plan_minimum"] and reduced["plan_minimum"] == 200
    assert reduced["from"]["runs_per_setting"] == 200 and reduced["to"]["runs_per_setting"] == runs
    assert "the time allowed asked for it" in reduced["note"] and "marked (unconfirmed)" in reduced["note"]
    assert "below the plan's minimum of 200" in paper_facts.fewer_runs_text(reduced)
    # a resume does not raise them again or cut again
    assert eng._raise_runs_to_the_minimum({"iteration": 0}) is None
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == runs


@pytest.mark.asyncio
async def test_a_minimum_that_fits_is_not_cut_or_marked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import result_checks, run_estimate
    from tests.test_run_checks import _sizing

    async def off(self: Engine, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)
    eng, _model, runner, _ex = await _sizing(tmp_path, timeout_s=3600, runs=1, min_runs_per_setting=40)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 40
    assert not run_estimate.read_record(eng.quest_root).get("reduced") and result_checks.fewer_runs_marks(eng.quest_root) == {}


@pytest.mark.asyncio
async def test_a_plan_at_or_above_its_minimum_is_left_as_it_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_run_checks import _sizing

    async def off(self: Engine, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)
    for runs, fewest in ((40, 10), (40, 40), (5, None)):
        extra = {} if fewest is None else {"min_runs_per_setting": fewest}
        eng, _model, runner, _ex = await _sizing(tmp_path / f"q{runs}_{fewest}", timeout_s=3600, runs=runs, **extra)
        assert eng._raise_runs_to_the_minimum({"iteration": 0}) is None
        assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == runs


@pytest.mark.asyncio
async def test_runs_FI_lowered_for_the_time_allowed_are_not_raised_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import run_estimate
    from tests.test_run_checks import _sizing

    async def off(self: Engine, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)
    eng, _model, _runner, _ex = await _sizing(tmp_path, timeout_s=3600, runs=12, min_runs_per_setting=40)
    run_estimate.write_record(eng.quest_root, {"fits": False, "reduced": {"note": "cut", "to": {"runs_per_setting": 12}}})
    assert eng._raise_runs_to_the_minimum({"iteration": 0}) is None
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 12




def test_every_result_of_the_plan_is_marked_when_the_run_has_fewer_runs_than_the_minimum(tmp_path: Path) -> None:
    import json

    from core import frozen_protocol, result_checks, run_estimate

    protocol = {"metrics": [{"id": "y", "kind": "mean"}, {"id": "gap", "kind": "mean", "role": "diagnostic"}],
                "derived": [{"id": "slope", "role": "result", "inputs": ["y"]}], "runs_per_setting": 12}
    frozen_protocol.freeze(tmp_path, protocol, approved_by="auto: test", source="test")
    assert result_checks.marked_quantities(tmp_path) == {}
    run_estimate.write_record(tmp_path, {"reduced": {"note": "n", "to": {"runs_per_setting": 12}, "marked_unconfirmed": True}})
    marked = result_checks.marked_quantities(tmp_path)
    assert set(marked) == {"y", "slope"}, json.dumps(marked)
    assert "12 runs per setting, fewer than the plan's minimum" in marked["y"]["note"]
    assert "`y`" in result_checks.marked_note(tmp_path)


# --- the first external review of the one-trial-per-cell change ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_resumed_study_with_a_sizing_record_of_the_old_rule_is_sized_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import reuse_key, run_estimate
    from tests.test_run_checks import _fdr_sizing

    async def off(self: Engine, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)
    eng, _model, runner = await _fdr_sizing(tmp_path, min_runs_per_setting=1537)
    # the key of one study is fixed here (a sizing changes the runs it keys on); only the rule's number is the real one
    monkeypatch.setattr(eng, "_sizing_key", lambda *_a, **_k: f"study|rule{reuse_key.RULES['sizing']}")
    # the record a run made before the rule changed: the same key under the old rule's number, the study cut to some settings
    run_estimate.write_record(eng.quest_root, {"key": "study|rule2", "truncated": {"run_id": "old", "planned_settings": [{"n": 1}]},
                                               "says": "the experiment is run up to the time limit"})
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    again = run_estimate.read_record(eng.quest_root)
    assert "truncated" not in again and again["key"] == f"study|rule{reuse_key.RULES['sizing']}"


def _plan_with_words(tmp_path: Path, words: str) -> dict[str, Any]:
    """A plan whose own words say ``words`` and whose protocol says one run per setting."""
    design = {"hypothesis": "h", "method": words, "expected_outcome": "e", "variables": {},
              "protocol": {"runs_per_setting": 1}}
    folder = tmp_path / "quest"
    folder.mkdir(parents=True)
    (folder / "plan.md").write_text(plan.render("topic", None, design), encoding="utf-8")
    return {"root": folder, "design": plan.load_design(folder)[0]}


def test_every_check_of_the_trials_reads_the_design_words_with_the_counts(tmp_path: Path) -> None:
    made = _plan_with_words(tmp_path, "a stochastic simulation of a colony")
    protocol = made["design"]["protocol"]
    assert mc_error.repeats_trials(protocol) is False and mc_error.design_repeats_trials(made["design"]) is True
    assert mc_error.quest_repeats_trials(made["root"], protocol) is True
    checked = {**protocol, "oracles": [{"name": "p", "case": {"n": 1}, "expected": 0.5, "measure": "hit"}]}
    assert mc_error.unsaid_reading(checked) == [] and mc_error.unsaid_reading(checked, made["design"]) == ["p"]
    plain = _plan_with_words(tmp_path / "plain", "an ordinary calculation")
    assert mc_error.quest_repeats_trials(plain["root"], plain["design"]["protocol"]) is False


@pytest.mark.asyncio
async def test_the_known_answer_cases_of_a_stochastic_design_with_one_run_per_setting_are_judged_as_repeated(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import code_checks, trial_runner

    made = _plan_with_words(tmp_path, "a stochastic simulation of a colony")
    root = made["root"]
    (root / "code").mkdir()
    (root / "code" / "simulate.py").write_text("def run_trial(**kw):\n    return {'hit': 1.0}\n", encoding="utf-8")
    protocol = {**made["design"]["protocol"], "oracles": [{"name": "p", "case": {"n": 1}, "expected": 0.5, "tolerance": 0.1,
                                                           "measure": "hit"}]}
    seen: list[Any] = []

    async def fake(*_a: Any, **kw: Any) -> tuple[list[Any], list[str], bool]:
        seen.append(kw.get("repeated"))
        return [], [], False

    monkeypatch.setattr(trial_runner, "measure_oracles", fake)
    await code_checks.known_answer_values(root, protocol, executor=None, python=sys.executable, env={}, timeout_s=10)
    await code_checks._oracle_verdict(root, protocol, executor=None, python=sys.executable, env={}, log=None, timeout_s=10)
    assert seen == [True, True]


def _derived_record(root: Path, derived: dict[str, Any]) -> None:
    import json

    path = root / "data" / "results" / "derived.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"derived": derived}), encoding="utf-8")


def test_only_the_results_that_rest_on_the_reduced_runs_are_marked(tmp_path: Path) -> None:
    from core import frozen_protocol, result_checks, run_estimate

    protocol = {"metrics": [{"id": "y", "kind": "mean"}],
                "derived": [{"id": "constant", "role": "result", "formula": "1"},
                            {"id": "body_constant", "role": "result"},
                            {"id": "slope", "role": "result"},
                            {"id": "scaled", "role": "result", "formula": "2 * slope"},
                            {"id": "no_record", "role": "result"}],
                "runs_per_setting": 12}
    frozen_protocol.freeze(tmp_path, protocol, approved_by="auto: test", source="test")
    run_estimate.write_record(tmp_path, {"reduced": {"note": "n", "to": {"runs_per_setting": 12}, "marked_unconfirmed": True}})
    _derived_record(tmp_path, {
        "constant": {"by": "formula", "status": "ok", "inputs": []},
        "body_constant": {"by": "derive_body_constant", "status": "ok", "asked": []},
        "slope": {"by": "derive_slope", "status": "ok", "asked": ["y", "n"]},
        "scaled": {"by": "formula", "status": "ok", "inputs": ["slope"]},
    })
    assert set(result_checks.marked_quantities(tmp_path)) == {"y", "slope", "scaled", "no_record"}


@pytest.mark.asyncio
async def test_after_the_freeze_the_trials_are_judged_by_the_reading_frozen_with_the_protocol(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import code_checks, frozen_protocol, trial_runner

    made = _plan_with_words(tmp_path, "a stochastic simulation of a colony")
    root, protocol = made["root"], made["design"]["protocol"]
    (root / "code").mkdir()
    (root / "code" / "simulate.py").write_text("def run_trial(**kw):\n    return {'hit': 1.0}\n", encoding="utf-8")
    protocol = {**protocol, "oracles": [{"name": "p", "case": {"n": 1}, "expected": 0.5, "tolerance": 0.1, "measure": "hit"}]}
    frozen_protocol.freeze(root, protocol, approved_by="auto: test", source="test",
                           repeats_trials=mc_error.planned_repeats_trials(root, protocol))
    # only the plan's prose changes afterwards: the protocol, and so the frozen record, stays
    design = made["design"]
    (root / "plan.md").write_text(plan.render("topic", None, {**design, "method": "an ordinary calculation"}), encoding="utf-8")
    assert mc_error.repeats_trials(protocol, plan.load_design(root)[0]) is False
    seen: list[Any] = []

    async def fake(*_a: Any, **kw: Any) -> tuple[list[Any], list[str], bool]:
        seen.append(kw.get("repeated"))
        return [], [], False

    monkeypatch.setattr(trial_runner, "measure_oracles", fake)
    await code_checks.known_answer_values(root, protocol, executor=None, python=sys.executable, env={}, timeout_s=10)
    assert seen == [True] and frozen_protocol.repeats_trials_of(root) is True


def test_a_frozen_record_without_the_reading_or_with_an_edited_one_falls_back_to_the_plan(tmp_path: Path) -> None:
    import json

    from core import frozen_protocol

    made = _plan_with_words(tmp_path, "a stochastic simulation of a colony")
    root, protocol = made["root"], made["design"]["protocol"]
    frozen_protocol.freeze(root, protocol, approved_by="auto: test", source="test")  # a record of an older FI: no reading
    assert frozen_protocol.repeats_trials_of(root) is None and mc_error.quest_repeats_trials(root, protocol) is True
    path = frozen_protocol.frozen_path(root)
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, "repeats_trials": False, "repeats_trials_sha256": "0" * 64}), encoding="utf-8")
    assert frozen_protocol.repeats_trials_of(root) is None and mc_error.quest_repeats_trials(root, protocol) is True
