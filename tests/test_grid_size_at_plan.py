"""A plan whose `protocol.grid` holds the axes of several separate sub-studies (a real source-and-focus study: fourteen axes,
147,604,654,080,000 settings, FI lists at most 500,000) is told so by FI at the plan, with the plan's own arithmetic, and
asked once to rewrite the grid; a rewrite that fits goes on, one that does not ends in the honest stop at design, before any
code is written and not hours into the run. A grid that fits is not asked anything. Fake model only."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import optimise_search as search
from core import plan
from tests.plan_parts_helpers import as_parts
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, _engine, _NO_OBJECTIONS

TOPIC = "Compare light-source shapes for the focus range of a 193 nm layer at many pitches"
MARK = "The experiment in this plan cannot run as written"
#: The axes of the real plan's `protocol.grid` (d3 duv), as the plan wrote them: name -> number of values.
AXES = {"source": 5, "pitch": 135, "conventional_sigma": 13, "annular_sigma_inner_outer": 52, "xdipole_pole_centre": 10,
        "xdipole_pole_radius": 3, "neighbour_distance_nm": 39, "ambit_conventional_sigma": 18, "ambit_source_type": 4,
        "source_design_class": 2, "pitch_nm": 32, "dipole_pole_centre": 10, "dipole_pole_radius": 3,
        "source_parametrisation": 2}
BIG = {name: list(range(n)) for name, n in AXES.items()}
SMALL = {"source": [0, 1, 2, 3, 4], "pitch": list(range(20))}


def test_the_real_grid_is_counted_and_named_axis_by_axis() -> None:
    said = search.grid_too_large(BIG)
    assert "147,604,654,080,000" in said and "500,000" in said and "`protocol.grid`" in said
    assert math.prod(AXES.values()) == 147_604_654_080_000
    assert said.index("`pitch` 135 values") < said.index("`annular_sigma_inner_outer` 52 values") < said.index("`source` 5 values")
    # the same count the run stops on
    with pytest.raises(search.TooManySettings) as caught:
        search.check_listable(BIG)
    assert "147,604,654,080,000" in str(caught.value)


def test_a_grid_that_fits_says_nothing_and_a_fixed_value_or_a_run_inside_does_not_multiply() -> None:
    assert search.grid_too_large(SMALL) == "" and search.grid_too_large({}) == "" and search.grid_too_large(None) == ""
    assert search.grid_too_large({"a": [1] * 500, "b": [1] * 1000}) == ""  # exactly 500,000
    assert "500,001" in search.grid_too_large({"a": [1] * 500_001, "b": [1]})
    # `in_each_run` and `constants` are not in `grid`, so nothing in them is counted (only the grid is read)
    protocol = {"grid": SMALL, "in_each_run": {"method": list(range(10**6))}, "constants": {"k": 1}}
    assert search.grid_too_large(protocol["grid"]) == ""


def test_the_request_states_the_arithmetic_and_what_the_plan_can_do() -> None:
    ask = plan.grid_request(TOPIC, search.grid_too_large(BIG))
    assert "147,604,654,080,000" in ask and "every combination" in ask
    assert "`protocol.constants`" in ask and "`protocol.in_each_run`" in ask and "fewer values" in ask
    assert "must not be crossed with one another" in ask


def _draft(grid: dict[str, list[Any]], **protocol: Any) -> dict[str, Any]:
    draft = copy.deepcopy(HEAT_SINK)
    draft["study_type"] = "measure"
    draft["protocol"] = {"oracles": copy.deepcopy(HEAT_SINK["protocol"]["oracles"]), "grid": grid, **protocol}
    return draft


def _set(**parts: Any) -> Any:
    def answer(text: str) -> str:
        out = plan.edit_design_block(text, lambda d: {**d, "protocol": {**d["protocol"], **parts}})
        assert out is not None
        return out
    return answer


class Model:
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


def _record(eng: Any) -> dict[str, Any]:
    return json.loads((eng.fi_dir / "grid_asked.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_the_plan_is_asked_once_and_a_grid_that_fits_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(grid=BIG, sub_studies=PARTS, tolerance_changed=True)], _draft(BIG))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) == 1 and _record(eng)["count"] == 1
    assert "147,604,654,080,000" in model.asked[0] and "`pitch` 135 values" in model.asked[0]
    state = {"topic": TOPIC, "iteration": 0}
    patch = await eng._node_design(state)
    protocol = patch["design"]["protocol"]
    assert protocol["grid"] == BIG and len(protocol["sub_studies"]) == len(PARTS)
    assert len(model.asked) == 1, "the design step asks nothing more"
    assert eng._grid_too_large_stop(state) is None
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "tolerance_changed" not in text and "tolerance: 0.01" in text, "only the asked parts are kept"


@pytest.mark.asyncio
async def test_a_grid_still_too_large_stops_at_design_and_is_asked_only_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(grid=BIG)] * 3, _draft(BIG))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    state = {"topic": TOPIC, "iteration": 0}
    patch = await eng._node_design(state)
    patch2 = await eng._node_design(state)  # a second pass (a resume) asks nothing more
    assert len(model.asked) == 1
    assert patch["design"]["protocol"]["grid"] == BIG == patch2["design"]["protocol"]["grid"]
    stop = await eng._node_implement({**state, "design": patch["design"]})
    result = stop["exec_result"]
    assert "147,604,654,080,000" in result["too_long"] and "500,000" in result["too_long"]
    tried = " ".join(result["too_long_tried"])
    assert "asked the plan's model" in tried and "does not choose" in tried
    assert "fewer settings" in result["too_long_advice"] and "timeout_s" not in result["too_long_advice"]
    assert not (eng.quest_root / "code").exists() or not any((eng.quest_root / "code").iterdir())
    assert _record(eng)["count"] == 1


@pytest.mark.asyncio
async def test_a_grid_that_fits_is_not_asked_anything_and_logs_nothing(tmp_path: Path, caplog: Any) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft(SMALL))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    caplog.clear()
    state = {"topic": TOPIC, "iteration": 0}
    patch = await eng._node_design(state)
    assert model.asked == [] and not (eng.fi_dir / "grid_asked.json").exists()
    assert eng._grid_too_large_stop({**state, "design": patch["design"]}) is None
    assert "settings limit" not in caplog.text and "too large" not in caplog.text


@pytest.mark.asyncio
async def test_a_note_FI_adds_to_plan_md_does_not_make_it_ask_a_second_time(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(grid=BIG)] * 3, _draft(BIG))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) == 1
    path = plan.plan_path(eng.quest_root)
    path.write_text(path.read_text(encoding="utf-8") + "\n- a note FI added after the plan was asked about\n", encoding="utf-8")
    await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) == 1
    # ... and the stop still says the plan's model was asked (read by the model asked, not by plan.md's hash).
    stop = eng._grid_too_large_stop({"topic": TOPIC, "design": {"protocol": {"grid": BIG}}})
    assert stop is not None and "asked the plan's model" in " ".join(stop["exec_result"]["too_long_tried"])


@pytest.mark.asyncio
async def test_a_redesign_drafted_at_design_is_asked_once_per_round_and_a_grid_that_fits_is_kept(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    sent: list[str] = []

    async def chat(prompt: str, **kw: Any) -> str:
        sent.append(prompt)
        return json.dumps({"grid": SMALL, "in_each_run": {"method": ["abbe", "hopkins"]}, "constants": {"k": 1}})

    eng._chat = chat  # type: ignore[method-assign]
    eng._client = object()
    state = {"topic": TOPIC, "iteration": 1}
    design = {"protocol": {"grid": {**BIG, "method": ["abbe", "hopkins"]}, "runs_per_setting": 3}}
    out = await eng._ask_design_for_the_grid(state, design, iteration=1)
    assert len(sent) == 1 and "must not be crossed with one another" in sent[0] and "settings in `protocol.grid` are" in sent[0]
    protocol = out["protocol"]
    assert protocol["grid"] == SMALL and protocol["in_each_run"] == {"method": ["abbe", "hopkins"]}
    assert protocol["constants"] == {"k": 1} and protocol["runs_per_setting"] == 3
    again = await eng._ask_design_for_the_grid(state, design, iteration=1)
    assert len(sent) == 1 and again is design  # the same round asks nothing more
    stop = eng._grid_too_large_stop({**state, "design": design})
    assert stop is not None and "asked the plan's model" in " ".join(stop["exec_result"]["too_long_tried"])
    await eng._ask_design_for_the_grid(state, {"protocol": {"grid": SMALL}}, iteration=2)
    assert len(sent) == 1  # a grid that fits is never asked about


PARTS = [
    {"name": "sources", "vary": ["source", "pitch", "conventional_sigma"]},
    {"name": "annular", "vary": ["annular_sigma_inner_outer", "source_design_class", "pitch_nm"]},
    {"name": "dipoles", "vary": ["xdipole_pole_centre", "xdipole_pole_radius", "dipole_pole_centre", "dipole_pole_radius"]},
    {"name": "ambit", "vary": ["neighbour_distance_nm", "ambit_conventional_sigma", "ambit_source_type"]},
    {"name": "parametrisation", "vary": ["source_parametrisation"]},
]


@pytest.mark.asyncio
async def test_the_plan_that_divides_its_grid_into_sub_studies_fits_and_nothing_is_lost(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(grid=BIG, sub_studies=PARTS)], _draft(BIG))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) == 1 and "sub_studies" in model.asked[0]
    state = {"topic": TOPIC, "iteration": 0}
    patch = await eng._node_design(state)
    protocol = patch["design"]["protocol"]
    assert protocol["grid"] == BIG and [s["name"] for s in protocol["sub_studies"]] == [p["name"] for p in PARTS]
    study = search.study_grid(protocol)
    assert search.grid_too_large(study) == "" and search.grid_size(study) < 20_000 < math.prod(AXES.values())
    assert {a for c in search.grid_cells(study) for a in c} == set(AXES)  # every setting carries every axis
    for axis, values in BIG.items():
        assert {c[axis] for c in search.grid_cells(study)} == set(values), axis  # no comparison lost
    assert len(model.asked) == 1 and eng._grid_too_large_stop(state) is None


@pytest.mark.asyncio
async def test_a_grid_written_again_replaces_the_division_into_sub_studies_it_no_longer_matches(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft(BIG, sub_studies=PARTS))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "sub_studies" in text  # a plan that already divides its grid is not asked anything
    wider = {**BIG, "extra": [0, 1]}
    kept = eng._keep_only_the_parts(text, {"protocol": {"grid": wider}}, ["grid", "sub_studies", "in_each_run", "constants"])
    protocol = plan.parse(kept).design["protocol"]
    assert protocol["grid"] == wider and "sub_studies" not in protocol  # the old parts do not cover the new grid
    with pytest.raises(ValueError, match="no entry of `protocol.sub_studies` varies"):  # parts that leave a setting out
        eng._keep_only_the_parts(text, {"protocol": {"grid": BIG, "sub_studies": PARTS[:2]}}, ["grid", "sub_studies"])
    assert plan.parse(text).design["protocol"]["grid"] == BIG  # the grid is not taken without its division
    moved = eng._keep_only_the_parts(text, {"protocol": {"in_each_run": {"pitch": [0, 1]}}}, ["in_each_run"])
    inside = plan.parse(moved).design["protocol"]
    assert "pitch" not in inside["grid"] and all("pitch" not in p["vary"] for p in inside["sub_studies"])
    assert search.grid_size(search.study_grid(inside)) > 0
