"""What each run compares inside it (``protocol.in_each_run``) is never dropped in silence when it cannot be read: the plan's
model is asked again once with the plain reason, and what is still unusable is a requirement of the study not done, which
the paper must state.

The real case (round s1, first-order kinetics, plan.md line 106 and run.log 17:39:45): the plan wrote
``in_each_run.temperature_1_over_T`` with ONE value; FI wrote "it is not used" and went on, and the five temperatures stayed
as a grid setting the simulation never read. Fake model only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import plan
from core import topic_requirements as tr
from tests.test_plan_parts import RawModel, _design, _text
from tests.test_plan_optimisation import _engine
from tests.test_protocol_part_asked import GOOD_GRID, TOPIC, _draft

FIX = Path(__file__).parent / "fixtures" / "plan_fidelity"
ONE_VALUE = {"wall_material": ["aluminium"]}
TWO_VALUES = {"wall_material": ["aluminium", "copper"]}


def _covered(eng) -> dict:
    path = eng.fi_dir / tr.COVERED_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def test_the_s1_protocol_with_one_value_is_marked_to_be_asked_not_dropped():
    design = plan.parse((FIX / "s1_kinetics" / "plan.v1.md").read_text(encoding="utf-8")).design
    protocol = {**design["protocol"], "in_each_run": {"temperature_1_over_T": [0.003125]}}
    fixed, notes = plan.repair_protocol(protocol)
    assert fixed is not None and "in_each_run" not in fixed
    note = next(n for n in notes if "`protocol.in_each_run`" in n)
    assert "has one value" in note and plan.ASK_MARK in note and "it is not used" not in note


@pytest.mark.asyncio
async def test_an_in_each_run_with_one_value_is_asked_again_once_and_the_answer_is_kept(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = RawModel(eng, [json.dumps({"protocol": {"in_each_run": TWO_VALUES}})],
                     _draft(grid=GOOD_GRID, in_each_run=ONE_VALUE))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) == 1
    asked = model.asked[0]
    assert "`protocol.in_each_run`" in asked and "has one value" in asked, "the plain reason goes with the request"
    after = _design(eng)["protocol"]
    assert after["in_each_run"] == TWO_VALUES and after["grid"] == GOOD_GRID
    assert "`protocol.in_each_run` again, and it is now in the plan" in _text(eng)
    assert not _covered(eng).get("in_each_run_left")


@pytest.mark.asyncio
async def test_a_factor_the_answer_puts_in_each_run_leaves_the_grid(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    grid = {**GOOD_GRID, "wall_material": ["aluminium", "copper"]}
    RawModel(eng, [json.dumps({"protocol": {"in_each_run": TWO_VALUES}})], _draft(grid=grid, in_each_run=ONE_VALUE))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    after = _design(eng)["protocol"]
    assert after["in_each_run"] == TWO_VALUES
    assert after["grid"] == GOOD_GRID, "a factor every run goes through together is not also run on its own"


@pytest.mark.asyncio
async def test_still_unusable_after_one_ask_is_a_requirement_not_done_and_the_paper_says_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    still_bad = json.dumps({"protocol": {"in_each_run": ONE_VALUE}})
    model = RawModel(eng, [still_bad, still_bad, still_bad], _draft(grid=GOOD_GRID, in_each_run=ONE_VALUE))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    # Asked once; an answer refused for its form earns the one retry every part gets, with the reader's exact reason.
    assert 1 <= len(model.asked) <= 2
    if len(model.asked) == 2:
        assert "Your last answer was not used" in model.asked[1] and "has one value" in model.asked[1]
    asked = len(model.asked)
    assert "in_each_run" not in _design(eng)["protocol"]
    left = _covered(eng).get("in_each_run_left")
    assert left and "has one value" in left and "asked the plan's model to write it again once" in left
    assert "still not readable (the experiment goes on without it)" in _text(eng)
    note = tr.write_note(eng.quest_root)
    assert "planned and not done" in note and "has one value" in note
    # A resume asks no more.
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_parts({"topic": TOPIC, "iteration": 0}, design, sha)
    assert len(model.asked) == asked
