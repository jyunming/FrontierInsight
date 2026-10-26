"""A first scan the plan's model wrote for a search, and that FI's reading of the block left out (a numeric condition of
the scan that the block does not hold fixed, say), is asked back of the plan's model with the reader's exact reason,
once, instead of being lost: the search would start without it, and what it was to compare would not be run. A scan
that cannot be written still never stops the quest. No model is called: a stub answers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import optimisation_plan as optim
from core import plan
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, _engine

# A numeric condition of the scan that the block does not hold under `fixed`: FI cannot know the value the search holds it at.
SCAN = {"fin_spacing": [2.0, 4.0], "inlet_temperature_C": [20.0, 40.0]}


def _with_scan(fixed_too: bool) -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    block = design["protocol"]["optimisation"]
    block["grid"] = dict(SCAN)
    block["search_method"] = "global_then_local"
    if fixed_too:
        block["fixed"] = {**block["fixed"], "inlet_temperature_C": 20.0}
    return design


async def _plan_step(tmp_path: Path, answers: list[str], design: dict[str, Any] | None = None) -> tuple[Any, list[str]]:
    """The plan step on a stub model: the plan's draft is ``design`` (the scan with its condition not held fixed), and
    each request FI makes to complete the plan is answered from ``answers`` in turn."""
    asked: list[str] = []
    draft = json.dumps({**(design or _with_scan(fixed_too=False)), "plan": EXTRA})
    pending = list(answers)

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        text = messages[-1]["content"]
        asked.append(text)
        if "Experiment Design** stage" in text:
            return draft
        if "Self-Critique" in text:
            return json.dumps({"objections_addressed": []})
        if "first scan" in text or "coarse scan" in text:
            if not pending:
                raise RuntimeError("no answer")
            return pending.pop(0)
        return "{}"

    eng = _engine(tmp_path, [])
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    await eng._node_plan({"topic": "heat sink", "iteration": 0})
    return eng, asked


def test_the_reader_leaves_out_a_scan_whose_condition_is_not_held_and_says_why() -> None:
    block, notes = optim.repair(_with_scan(fixed_too=False)["protocol"]["optimisation"])
    # only the setting that cannot be read is left out; the rest of the scan stays
    assert block is not None and block["grid"] == {"fin_spacing": SCAN["fin_spacing"]}
    assert any(optim.scan_setting_left_out(n) == "inlet_temperature_C" for n in notes), notes


@pytest.mark.asyncio
async def test_a_scan_left_out_is_asked_back_with_the_exact_reason_and_kept_when_the_answer_can_be_read(
        tmp_path: Path) -> None:
    answer = json.dumps({"protocol": {"optimisation": _with_scan(fixed_too=True)["protocol"]["optimisation"]}})
    eng, asked = await _plan_step(tmp_path, [answer] * 3)
    ask = next((a for a in asked if "a part of it in the plan could not be read" in a), None)
    assert ask is not None, "the plan's model is asked for the scan it wrote"
    assert "inlet_temperature_C" in ask and "neither a design variable nor a condition held fixed" in ask
    design = plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design
    block = design["protocol"]["optimisation"]
    assert block["grid"] == SCAN and block["fixed"]["inlet_temperature_C"] == 20.0
    assert eng._scan_left_out(design) == [], "a scan that is back is not asked for again"


@pytest.mark.asyncio
async def test_an_answer_that_drops_a_setting_the_scan_already_had_gets_it_back(tmp_path: Path) -> None:
    block = copy.deepcopy(_with_scan(fixed_too=True)["protocol"]["optimisation"])
    block["grid"] = {"inlet_temperature_C": SCAN["inlet_temperature_C"]}  # the readable `fin_spacing` left out
    eng, _asked = await _plan_step(tmp_path, [json.dumps({"protocol": {"optimisation": block}})] * 3)
    design = plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design
    assert design["protocol"]["optimisation"]["grid"] == SCAN, "the plan's own setting is put back beside the one asked for"


@pytest.mark.asyncio
async def test_an_answer_whose_parts_do_not_fit_the_settings_put_back_is_refused_and_asked_again_bounded(
        tmp_path: Path) -> None:
    block = copy.deepcopy(_with_scan(fixed_too=True)["protocol"]["optimisation"])
    block["grid"] = {"inlet_temperature_C": SCAN["inlet_temperature_C"]}
    block["sub_studies"] = [{"name": "inlet", "vary": ["inlet_temperature_C"], "hold": {}}]  # leaves `fin_spacing` unrun
    eng, asked = await _plan_step(tmp_path, [json.dumps({"protocol": {"optimisation": block}})] * 6)
    scan_asks = [a for a in asked if "a part of it in the plan could not be read" in a or "was not used" in a]
    assert 1 <= len(scan_asks) <= 4, len(scan_asks)
    assert any("Your last answer was not used" in a for a in asked), "the reason goes back to the plan's model"
    design = plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design
    kept = design["protocol"]["optimisation"]
    assert kept["grid"] == {"fin_spacing": SCAN["fin_spacing"]} and "sub_studies" not in kept, "the answer is not used"


@pytest.mark.asyncio
async def test_a_scan_still_unreadable_is_asked_once_and_the_search_goes_on_without_it(tmp_path: Path) -> None:
    still = json.dumps({"protocol": {"optimisation": _with_scan(fixed_too=False)["protocol"]["optimisation"]}})
    eng, asked = await _plan_step(tmp_path, [still] * 6)
    scan_asks = [a for a in asked if "a part of it in the plan could not be read" in a]
    # asked once, and once more with the reason its answer was refused; a left-out scan never stops the quest
    assert 1 <= len(scan_asks) <= 2, len(scan_asks)
    design, _sha = eng._design_from_plan()
    assert design is not None and design["protocol"]["optimisation"]["grid"] == {"fin_spacing": SCAN["fin_spacing"]}
    assert eng._stop_if_the_search_cannot_start({"topic": "heat sink", "iteration": 0}, design) is None
    assert eng._search_cannot_start_stop({"topic": "heat sink", "iteration": 0}, design) is None
    # The design step reads the same plan: what the plan step asked is not asked again there.
    sha = plan.sha256(plan.plan_path(eng.quest_root).read_text(encoding="utf-8"))
    before = len(asked)
    await eng._ask_plan_for_the_search({"topic": "heat sink", "iteration": 0}, design, sha)
    assert len(asked) == before, "a scan asked for at the plan step is not asked for again at the design step"


@pytest.mark.asyncio
async def test_a_search_with_no_scan_written_is_not_asked_for_one(tmp_path: Path) -> None:
    _eng, asked = await _plan_step(tmp_path, [], design=copy.deepcopy(HEAT_SINK))
    assert not any("a part of it in the plan could not be read" in a for a in asked)
