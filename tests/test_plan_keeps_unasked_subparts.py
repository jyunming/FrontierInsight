"""A rewrite of one key of a plan's block never costs the block a key it had, and a search that a later rewrite left unable
to start is asked of the plan's model again instead of stopping with nobody asked. Fake model only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan
from core import plan_scope
from tests.plan_parts_helpers import as_parts
from tests.test_plan_optimisation import BLOCK, EXTRA, HEAT_SINK, _engine

TOPIC = "Find the fin spacing and thickness that minimise the base temperature of a heat sink"


def _design(**changes: Any) -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    block = design["protocol"]["optimisation"]
    for key, value in changes.items():
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    return design


#: A budget whose time for one evaluation is words where a number is asked: the reader leaves the whole budget out.
UNREADABLE_BUDGET = {"starts": 4, "per_start": 60, "seconds_per_evaluation": "measured on a few evaluations first"}
#: A search method that is a sentence, not one of the methods the reader knows.
UNREADABLE_METHOD = "a local search from the best point of a coarse look, stopped when it stops improving"
NEW_BASELINE = {"values": {"fin_spacing": 2.5, "fin_thickness": 1.2, "fin_count": 18}, "source": "[4] Table 1"}


# -- the methodology audit's change (Engine._hold_audit_repairs_to_the_schema) ------------------------------------------

def test_an_audit_rewrite_of_the_search_block_keeps_the_plans_readable_budget_and_method(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    before = _design()
    amended = _design(evaluation_budget=UNREADABLE_BUDGET, search_method=UNREADABLE_METHOD, baseline=NEW_BASELINE)
    design, not_applied = eng._hold_audit_repairs_to_the_schema(before, amended)
    block = design["protocol"]["optimisation"]
    assert block["evaluation_budget"] == BLOCK["evaluation_budget"]
    assert block["search_method"] == BLOCK["search_method"]
    assert block["baseline"] == NEW_BASELINE, "the audit's readable change is made"
    assert [n["key"] for n in not_applied] == ["protocol.optimisation"] and not_applied[0].get("applied")
    for handler in eng._log.handlers:
        handler.flush()
    said = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "`protocol.optimisation.evaluation_budget` could not be read, so that change was not applied" in said
    assert "`protocol.optimisation.search_method` could not be read" in said


def test_an_audit_change_to_the_budget_that_reads_is_still_made(tmp_path: Path) -> None:
    """Known positive: the audit may change the budget; only a budget that cannot be read keeps the plan's own."""
    eng = _engine(tmp_path, [])
    new_budget = {"starts": 6, "per_start": 80, "seconds_per_evaluation": 1.5}
    amended = _design(evaluation_budget=new_budget, search_method=UNREADABLE_METHOD)
    design, _not_applied = eng._hold_audit_repairs_to_the_schema(_design(), amended)
    block = design["protocol"]["optimisation"]
    assert block["evaluation_budget"] == new_budget
    assert block["search_method"] == BLOCK["search_method"]


# -- a scoped request for some keys of a block (core.plan_scope.translate) --------------------------------------------

_BUDGET_ONLY = plan_scope.PartScope(inside={"optimisation": ("evaluation_budget",)})


def test_a_reply_that_writes_the_whole_block_back_is_read_for_the_asked_keys_alone() -> None:
    old = copy.deepcopy(HEAT_SINK["protocol"])
    new_budget = {"starts": 2, "per_start": 40, "seconds_per_evaluation": 2.0}
    reply = {"protocol": {"optimisation": {**copy.deepcopy(BLOCK), "evaluation_budget": new_budget,
                                           "fixed": {"heat_load_W": 55, "air_speed_m_s": 2.0}}}}
    not_applied: list[str] = []
    out, problems = plan_scope.translate(_BUDGET_ONLY, old, reply, not_applied)
    assert problems == []
    block = out["protocol"]["optimisation"]
    assert block["evaluation_budget"] == new_budget, "the asked key is changed"
    assert block["fixed"] == BLOCK["fixed"], "an unasked key stays exactly as the plan has it"
    assert {k: v for k, v in block.items() if k != "evaluation_budget"} == {
        k: v for k, v in BLOCK.items() if k != "evaluation_budget"}
    assert len(not_applied) == 1 and "`fixed`" in not_applied[0] and "not applied" in not_applied[0]


def test_a_reply_with_only_unasked_changes_is_refused() -> None:
    old = copy.deepcopy(HEAT_SINK["protocol"])
    reply = {"protocol": {"optimisation": {"fixed": {"heat_load_W": 55, "air_speed_m_s": 2.0}}}}
    out, problems = plan_scope.translate(_BUDGET_ONLY, old, reply, [])
    assert problems and "`fixed`" in problems[0]
    assert "optimisation" not in out["protocol"]


# -- a part merged into the plan (Engine._keep_only_the_parts) ---------------------------------------------------------

def test_a_part_asked_for_that_cannot_be_read_is_refused_whole_and_the_plan_keeps_its_own(tmp_path: Path) -> None:
    """The scoped repairs read a part strictly: a part the answer wrote with a key that cannot be read is not taken at all
    (its reason goes back to the model), so the plan's own key is never lost there."""
    eng = _engine(tmp_path, [])
    before_text = plan.render(TOPIC, EXTRA, _design())
    answer = {"protocol": {"optimisation": {**copy.deepcopy(BLOCK), "evaluation_budget": UNREADABLE_BUDGET,
                                            "baseline": NEW_BASELINE}}}
    with pytest.raises(ValueError, match="seconds_per_evaluation"):
        eng._keep_only_the_parts(before_text, answer, ["optimisation"])
    good = {"protocol": {"optimisation": {**copy.deepcopy(BLOCK), "baseline": NEW_BASELINE}}}
    kept = eng._keep_only_the_parts(before_text, good, ["optimisation"])
    block = plan.raw_design_block(kept)["protocol"]["optimisation"]
    assert block["baseline"] == NEW_BASELINE and block["evaluation_budget"] == BLOCK["evaluation_budget"]


# -- a search a later rewrite left unable to start (Engine._ask_again_for_a_search_lost_since) --------------------------

def _plan_without_budget(eng: Any) -> tuple[dict[str, Any], str]:
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render(TOPIC, EXTRA, _design(evaluation_budget=None)), encoding="utf-8")
    design, sha = eng._design_from_plan()
    assert design is not None and sha
    return design, sha


def _answers_with_the_whole_block(eng: Any) -> list[str]:
    asked: list[str] = []

    async def chat(messages: list[dict[str, str]], **_kw: Any) -> str:
        asked.append(messages[-1]["content"])
        text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
        whole = plan.edit_design_block(text, lambda d: {**d, "protocol": {**d["protocol"], "optimisation": copy.deepcopy(BLOCK)}})
        return as_parts(whole)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    return asked


@pytest.mark.asyncio
async def test_a_plan_changed_since_its_search_was_completed_is_asked_for_the_search_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design, sha = _plan_without_budget(eng)
    # The record of the asks is for an earlier plan (before a later rewrite): this plan was never asked.
    eng._record_search_asked(2, 4, "an-earlier-plan", "kept")
    asked = _answers_with_the_whole_block(eng)
    state = {"topic": TOPIC, "iteration": 0}
    assert eng._search_problems(state, design)[1], "the search cannot start"
    design2, sha2 = await eng._ask_again_for_a_search_lost_since(state, design, sha)
    assert asked, "the plan's model was asked"
    assert sha2 != sha and not eng._search_problems(state, design2)[1], "the search can start now"
    assert design2["protocol"]["optimisation"]["evaluation_budget"] == BLOCK["evaluation_budget"]
    assert eng._search_cannot_start_stop(state, design2) is None


@pytest.mark.asyncio
async def test_a_plan_already_asked_for_its_search_is_not_asked_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design, sha = _plan_without_budget(eng)
    eng._record_search_asked(2, 4, sha, "not usable")
    asked = _answers_with_the_whole_block(eng)
    state = {"topic": TOPIC, "iteration": 0}
    design2, sha2 = await eng._ask_again_for_a_search_lost_since(state, design, sha)
    assert asked == [] and (design2, sha2) == (design, sha)
    assert eng._search_cannot_start_stop(state, design2) is not None, "the honest stop still follows"


@pytest.mark.asyncio
async def test_a_search_that_can_start_is_not_asked_for(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render(TOPIC, EXTRA, _design()), encoding="utf-8")
    design, sha = eng._design_from_plan()
    asked = _answers_with_the_whole_block(eng)
    out = await eng._ask_again_for_a_search_lost_since({"topic": TOPIC, "iteration": 0}, design, sha)
    assert asked == [] and out == (design, sha)
    assert json.dumps(out[0], sort_keys=True, default=str) == json.dumps(design, sort_keys=True, default=str)
