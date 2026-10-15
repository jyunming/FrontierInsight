"""The topic's own run-time limit, when shorter than FI's (execution.timeout_s), is what the run is sized to.

The real case (rounds s1 and r13, first-order kinetics): the topic says "under 60 seconds of runtime" and the config's
execution.timeout_s is 900. FI sized both runs to 900 s and told the plan the topic's figure was only a wish; s1 ran a
study of about 4 minutes. FI's own limit stays the point where a run is stopped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from core import disclosure, plan, run_estimate
from core.engine import Engine
from tests.test_design_prompt_inputs import _cfg
from tests.test_run_checks import _log, _sizing

FIX = Path(__file__).parent / "fixtures" / "plan_fidelity"
SHORT = "Measure how a damped spring settles. CPU only; under 8 seconds of runtime."


@pytest.fixture(autouse=True)
def _first_reply_is_the_plan(monkeypatch):
    """tests/test_run_checks.py's fake model takes its first call as the plan's draft (as tests/conftest.py arranges for
    that file): the reading of the topic before the plan is left out here too."""
    async def off(self, state):  # noqa: ANN001
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)


def _topic(quest: str) -> tuple[str, list[str]]:
    config = yaml.safe_load((FIX / quest / "config.yaml").read_text(encoding="utf-8"))
    required = json.loads((FIX / quest / "topic_requirements.json").read_text(encoding="utf-8"))
    return config["topic"], [c if isinstance(c, str) else c.get("stated", "") for c in required.get("constraints") or []]


@pytest.mark.parametrize("quest", ["s1_kinetics", "r13_kinetics"])
def test_the_kinetics_topics_state_sixty_seconds(quest):
    topic, constraints = _topic(quest)
    assert disclosure._topic_limit(constraints, topic) == 60.0


@pytest.mark.parametrize("quest", ["s1_kinetics", "r13_kinetics"])
def test_replay_the_kinetics_runs_are_sized_to_the_topics_sixty_seconds(quest):
    """Arithmetic on the recorded estimates (run_estimate.json, 125 settings at 900 s), scaled to the 25 settings left once
    the unread temperature leaves the grid: s1 (about 46 s) fits 60 s; r13 (about 247 s, with runs_per_setting equal to
    the plan's fewest, 400) cannot be cut and is run within 900 s with the shortfall said."""
    record = json.loads((FIX / quest / "run_estimate.json").read_text(encoding="utf-8"))
    protocol = plan.parse((FIX / quest / "plan.md").read_text(encoding="utf-8")).design["protocol"]
    est = run_estimate.Estimate(seconds=record["estimate_s"] / run_estimate.SAFETY * 25 / 125, at_least=False,
                                complete=True, cells=25, trials=record["trials"] // 5)
    runs = int(protocol["runs_per_setting"])
    if quest == "s1_kinetics":
        assert not run_estimate.known_too_long(est, 60.0)
    else:
        assert run_estimate.known_too_long(est, 60.0) and not run_estimate.known_too_long(est, 900.0)
        assert run_estimate.runs_that_fit(est, 60.0, runs, False, protocol) is None


def test_the_plan_and_the_code_are_told_the_topics_limit(tmp_path):
    eng = Engine(_cfg(tmp_path))
    eng.config.execution.timeout_s = 900
    state = {"topic": "CPU only; under 60 seconds of runtime."}
    note = eng._time_note(state)
    assert "at most 60 s: that is the limit to plan for, not FI's own 900 s" in note
    assert eng._planning_limit(state) == "60 (the topic's own limit; FI stops a run only at 900 s, its execution.timeout_s)"
    assert eng._planning_limit({"topic": "SIR final outbreak size"}) == "900"
    assert "FI's own limit of 900 s is shorter" in eng._time_note({"topic": "Run in under 2 hours of runtime."})


@pytest.mark.asyncio
async def test_a_run_over_the_topics_limit_has_its_runs_lowered_to_it(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600, min_runs_per_setting=8)
    assert await eng._size_the_run({"iteration": 0, "topic": SHORT}, runner, sys.executable, None) is None
    protocol = plan.load_design(eng.quest_root)[0]["protocol"]
    assert protocol["grid"] == {"n": [1, 2, 4]} and 8 <= protocol["runs_per_setting"] < 40
    log = _log(eng)
    assert "the topic asks for a run of at most 8 seconds; FI sizes the experiment to that, not to its own 1 h" in log
    assert "fits the topic's limit of 8 seconds" in log
    record = run_estimate.read_record(eng.quest_root)
    assert record["fits"] is True and record["topic_limit_s"] == 8.0 and not record.get("over_topic")


@pytest.mark.asyncio
async def test_a_run_that_cannot_meet_the_topics_limit_runs_within_fis_and_says_so(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600)  # the plan states no fewest runs: no cut
    assert await eng._size_the_run({"iteration": 0, "topic": SHORT}, runner, sys.executable, None) is None
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 40, "nothing was cut"
    record = run_estimate.read_record(eng.quest_root)
    assert record["fits"] is True and "the topic's time limit is not met" in record["over_topic"]
    assert "the topic's time limit is not met" in _log(eng)
    assert "the topic's time limit is not met" in plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "do not describe the run as within the topic's time limit" in eng._run_checks_note()
    # A resume with the same plan and code does not time it again, nor add the note twice.
    calls = ex.calls
    assert await eng._size_the_run({"iteration": 0, "topic": SHORT}, runner, sys.executable, None) is None
    assert ex.calls == calls
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8").count("the topic's time limit is not met") == 1


@pytest.mark.asyncio
async def test_fis_own_limit_still_stops_a_run(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=6)
    patch = await eng._size_the_run({"iteration": 0, "topic": SHORT.replace("8 seconds", "4 seconds")}, runner,
                                    sys.executable, None)
    assert patch is not None and "more than the 6 seconds allowed" in patch["exec_result"]["too_long"]
