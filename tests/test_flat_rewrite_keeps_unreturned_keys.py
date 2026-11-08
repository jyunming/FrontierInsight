"""A rewrite for a flat objective that replaces the search block is read as the keys it changes (maintainer decision
2026-10-25: a redesign rewrites only the changed parts). A key the plan's block has that the answer leaves out is kept
from the plan, and FI says so in run.log and in the plan's change note; a key the answer sets to empty is a removal.
A live answer replaced the block without the plan's list of separate sweeps, so the whole grid became one scan too big to
run and the answer was refused. Neutral toy topic only (a heat sink)."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from core import plan
from core.engine import _flat_request
from tests.test_flat_objective_redesign import FACT, _flat_state, _planned


def _reply(eng: Any, drop: tuple[str, ...] = (), **set_keys: Any) -> str:
    block = copy.deepcopy(plan.raw_design_block(plan.plan_path(eng.quest_root).read_text(encoding="utf-8"))
                          ["protocol"]["optimisation"])
    block["design_variables"][0]["high"] = 9.5
    for key in drop:
        block.pop(key, None)
    block.update(set_keys)
    return json.dumps({"protocol": {"optimisation": block}})


def _block_now(eng: Any) -> dict[str, Any]:
    return plan.raw_design_block(plan.plan_path(eng.quest_root).read_text(encoding="utf-8"))["protocol"]["optimisation"]


@pytest.mark.asyncio
async def test_a_key_the_answer_leaves_out_is_kept_from_the_plan_and_said(tmp_path: Path, caplog: Any) -> None:
    eng, model = await _planned(tmp_path)
    kept_value = copy.deepcopy(_block_now(eng)["fixed"])
    model.revise = _reply(eng, drop=("fixed",))
    with caplog.at_level(logging.WARNING):
        done = await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    block = _block_now(eng)
    assert block["design_variables"][0]["high"] == 9.5, "what the answer changed is taken"
    assert block["fixed"] == kept_value, "what it left out stays as the plan had it"
    assert "kept from the plan: fixed" in plan.history(eng.quest_root)[done["version"] - 1]["note"]


@pytest.mark.asyncio
async def test_a_key_the_answer_sets_to_empty_is_a_removal(tmp_path: Path) -> None:
    eng, model = await _planned(tmp_path)
    model.revise = _reply(eng, fixed={})
    await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    assert not _block_now(eng).get("fixed"), "an empty answer is not replaced by the plan's value"


@pytest.mark.asyncio
async def test_an_answer_that_leaves_out_nothing_is_noted_as_before(tmp_path: Path) -> None:
    eng, model = await _planned(tmp_path)
    model.revise = _reply(eng)
    done = await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    assert "kept from the plan" not in plan.history(eng.quest_root)[done["version"] - 1]["note"]
