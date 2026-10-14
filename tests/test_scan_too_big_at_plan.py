"""A search whose coarse scan cannot run (a real source-and-focus study: thirteen axes, 19,245,600,000 settings, twelve of
them conditions the plan also holds fixed) is told so by FI at the plan, with the plan's own arithmetic, and asked once to
rewrite the scan; a rewrite that fits goes on, one that does not ends in the honest stop at design, minutes after the plan
and not hours into the run. Fake model only; the block is built here from the real plan's grid."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import optimisation_plan as op
from core import plan
from tests.plan_parts_helpers import as_parts
from tests.test_plan_optimisation import BLOCK, EXTRA, HEAT_SINK, _NO_OBJECTIONS, _engine
from tests.test_search_block_asked import _record, _with_block

TOPIC = "Find the source shape that gives the widest common focus range for the pitches of a 193 nm layer"
MARK = "the coarse scan in the plan cannot run as written"
FAMILIES = ["conventional", "annular", "x_dipole", "quasar", "freeform"]
#: The axes of the real plan's `protocol.optimisation.grid` (r9 duv), as the plan wrote them (counts: 5 33 10 5 6 6 3 4 5 3 18 10 2).
GRID: dict[str, list[Any]] = {
    "source_family": FAMILIES,
    "pitch_nm": list(range(33)),
    "conventional_sigma": [round(0.1 * i, 2) for i in range(1, 11)],
    "annular_inner_sigma": [0.2, 0.3, 0.4, 0.5, 0.6],
    "annular_outer_sigma": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "x_dipole_pole_centre_sigma": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "x_dipole_pole_radius_sigma": [0.1, 0.15, 0.2],
    "quasar_inner_sigma": [0.2, 0.3, 0.4, 0.5],
    "quasar_outer_sigma": [0.6, 0.7, 0.8, 0.9, 0.95],
    "quasar_pole_angle_deg": [20, 30, 45],
    "neighbour_distance_nm": list(range(18)),
    "ambit_conventional_sigma": [round(0.1 * i, 2) for i in range(1, 11)],
    "imaging_method": ["Abbe", "Hopkins"],
}


def _block(grid: dict[str, list[Any]]) -> dict[str, Any]:
    block = copy.deepcopy(BLOCK)
    block["fixed"] = {**block["fixed"], **{name: values[0] for name, values in GRID.items()}}
    block["grid"] = copy.deepcopy(grid)
    return block


def _draft(grid: dict[str, list[Any]]) -> dict[str, Any]:
    draft = copy.deepcopy(HEAT_SINK)
    draft["protocol"]["optimisation"] = _block(grid)
    return draft


class Model:
    """A fake model: the plan draft first, an answer to each request to rewrite the scan, no objections otherwise."""

    def __init__(self, eng: Any, answers: list[Any], draft: dict[str, Any]) -> None:
        self.eng, self.answers, self.draft = eng, list(answers), draft
        self.asked: list[str] = []
        self.drafted = False
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if MARK in prompt:
            self.asked.append(prompt)
            answer = self.answers.pop(0) if self.answers else None
            if answer is None:
                raise RuntimeError("no answer")
            return as_parts(answer(plan.plan_path(self.eng.quest_root).read_text(encoding="utf-8")))
        if not self.drafted and "Also write the plan" in prompt:
            self.drafted = True
            return json.dumps({**self.draft, "plan": EXTRA})
        return _NO_OBJECTIONS

    @property
    def requests(self) -> int:
        return len([p for p in self.asked if "Your previous reply could not be used" not in p])


def _fits(text: str) -> str:
    return _with_block(text, _block({"source_family": FAMILIES, "conventional_sigma": GRID["conventional_sigma"]}))


def _same(text: str) -> str:
    return text


def test_the_real_plans_scan_is_counted_and_named_axis_by_axis() -> None:
    problem = op.scan_problem(_draft(GRID), 3600.0)
    assert "19,245,600,000 settings" in problem
    assert "`source_family` 5 values" in problem and "`pitch_nm` 33 values" in problem
    assert "`imaging_method` 2 values" in problem and "(a condition the plan holds fixed)" in problem
    assert "does not list or run more than 500,000" in problem
    assert op.scan_problem(_draft({"source_family": FAMILIES, "conventional_sigma": GRID["conventional_sigma"]}), 3600.0) == ""
    # By time alone: 50 settings at the plan's own 2 s each is 100 s, over a 60 s limit.
    assert "about 100 s, and one run may take 60 s" in op.scan_problem(_draft({"source_family": FAMILIES, "conventional_sigma": GRID["conventional_sigma"]}), 60.0)
    assert op.missing_parts(_draft(GRID), 3600.0)[0].startswith("the coarse scan cannot run")
    assert op.missing_parts(_draft(GRID)) and "coarse scan" in op.missing_parts(_draft(GRID), 3600.0)[0]


@pytest.mark.asyncio
async def test_the_plan_is_asked_once_to_rewrite_the_scan_and_a_scan_that_fits_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    eng.config.execution.timeout_s = 3600
    model = Model(eng, [_fits], _draft(GRID))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert model.requests == 1 and _record(eng)["count"] == 1
    request = model.asked[0]
    assert "19,245,600,000 settings" in request and "`source_family` 5 values" in request
    assert "`pitch_nm` 33 values" in request and "500,000" in request
    assert "multiplies every other" in request and "only its held value can start the search" in request
    assert "a parameter of just one of the alternatives" in request
    state = {"topic": TOPIC, "iteration": 0}
    patch = await eng._node_design(state)
    grid = patch["design"]["protocol"]["optimisation"]["grid"]
    assert list(grid) == ["source_family", "conventional_sigma"]
    assert eng._search_cannot_start_stop(state, patch["design"]) is None
    assert model.requests == 1  # the design step asks nothing more
    # the rest of the block is the plan's own
    assert patch["design"]["protocol"]["optimisation"]["evaluation_budget"]["per_start"] == 60


@pytest.mark.asyncio
async def test_a_scan_that_still_does_not_fit_stops_at_design_and_the_plan_is_asked_only_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    eng.config.execution.timeout_s = 3600
    model = Model(eng, [_same, _same, _same], _draft(GRID))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    state = {"topic": TOPIC, "iteration": 0}
    patch = await eng._node_design(state)
    assert model.requests == 1
    stop = eng._search_cannot_start_stop(state, patch["design"])
    assert stop is not None
    shown = stop["exec_result"]["no_search"] + " " + " ".join(stop["exec_result"]["no_search_tried"])
    assert "a first scan of the settings small enough to run" in shown and "19,245,600,000" in shown
    assert "asked the plan's model 1 time(s)" in shown
    # nothing was made up: the plan's own scan is as it was
    assert len(patch["design"]["protocol"]["optimisation"]["grid"]) == len(GRID)
    # a second pass (a resume) asks nothing more
    patch = await eng._node_design(state)
    assert model.requests == 1 and eng._search_cannot_start_stop(state, patch["design"]) is not None


@pytest.mark.asyncio
async def test_a_scan_that_fits_is_not_asked_anything(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    eng.config.execution.timeout_s = 3600
    model = Model(eng, [], _draft({"source_family": FAMILIES, "conventional_sigma": GRID["conventional_sigma"]}))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.requests == 0
    assert eng._search_cannot_start_stop({"topic": TOPIC, "iteration": 0}, patch["design"]) is None


def test_the_plans_runs_per_evaluation_never_make_the_plan_time_check_stricter_than_the_run() -> None:
    # 50 settings at 2 s is 100 s, under a 150 s limit; three runs each would be 300 s, but whether the simulation is
    # deterministic (one run per setting) is known only when the code exists, so the run's own timed check decides.
    draft = _draft({"source_family": FAMILIES, "conventional_sigma": GRID["conventional_sigma"]})
    draft["protocol"]["optimisation"]["runs_per_evaluation"] = 3
    assert op.scan_problem(draft, 150.0) == ""
