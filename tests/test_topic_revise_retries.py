"""The topic-coverage revise: an answer the reader refused for its form goes back with the reader's exact reason and is
asked again within a bound (the plan step and the design step); the three leftovers of the design-step ask. Fake models only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.test_design_critique_record import DRAFT, _engine as _critique_engine
from tests.test_topic_requirements import LID_AXIS, _redesign_setup


@pytest.mark.asyncio
async def test_a_design_step_answer_refused_for_its_form_is_asked_again_up_to_four_times(tmp_path: Path) -> None:
    eng, model, redesign, state = await _redesign_setup(tmp_path)
    original = model.chat
    prompts: list[str] = []
    def bad_with(value: Any) -> dict[str, Any]:
        return {**redesign, "protocol": {**redesign["protocol"], "grid": {**redesign["protocol"]["grid"], "lid_material": value}}}
    # three DIFFERENT unreadable forms (the same refusal twice would end the asking early)
    bads = [bad_with([]), bad_with(["paper", [1, 2]]), bad_with([[1.0], [1.0, 2.0]])]
    good = {**redesign, "protocol": {**redesign["protocol"], "grid": {**redesign["protocol"]["grid"], **LID_AXIS}}}

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if "The topic asks for things this plan does not yet include" in prompt:
            prompts.append(prompt)
            return json.dumps(bads[len(prompts) - 1] if len(prompts) < 4 else good)  # the fourth try is the first that can be read
        return await original(messages, **kw)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    design, _sha = await eng._hold_the_design_to_the_topic(state, redesign, "")
    assert design["protocol"]["grid"]["lid_material"] == ["paper", "foam", "steel"]
    assert len(prompts) == 4
    reasons = ("non-empty list", "mixes lists with single values", "the same number of numbers")
    assert all(r in p for r, p in zip(reasons, prompts[1:])), "each retry quotes the reader's exact reason"
    record = json.loads((eng.fi_dir / "topic_design_asked.json").read_text(encoding="utf-8"))
    assert record["asks"] == 4 and not record["problem"] and not record["schema"], "a used answer clears the refusal"
    await eng._hold_the_design_to_the_topic(state, redesign, "")  # a resume never asks past the bound
    assert len(prompts) == 4


@pytest.mark.asyncio
async def test_a_failed_call_does_not_earn_the_extra_tries(tmp_path: Path) -> None:
    eng, model, redesign, state = await _redesign_setup(tmp_path)
    original = model.chat
    prompts: list[str] = []

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if "The topic asks for things this plan does not yet include" in prompt:
            prompts.append(prompt)
            raise RuntimeError("connection reset")
        return await original(messages, **kw)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    await eng._hold_the_design_to_the_topic(state, redesign, "")
    assert len(prompts) == 2


@pytest.mark.asyncio
async def test_what_the_redesign_still_lacks_never_stops_the_quest(tmp_path: Path) -> None:
    eng, _model, redesign, state = await _redesign_setup(tmp_path)  # the model never adds the axis
    (eng.fi_dir / "topic_asked.json").write_text(json.dumps({"problem": "an old plan-step reason", "count": 2}), encoding="utf-8")
    await eng._hold_the_design_to_the_topic(state, redesign, "")
    assert not (eng.fi_dir / "topic_unmet.json").exists()
    covered = json.loads((eng.fi_dir / "topic_covered.json").read_text(encoding="utf-8"))
    assert any("lid" in str(g.get("name")) for g in covered["left"]), "it is kept for the paper"


@pytest.mark.asyncio
async def test_an_amended_design_whose_protocol_is_not_a_mapping_is_reverted_not_passed_with_the_protocol_dropped(tmp_path: Path) -> None:
    """A protocol the plan step cannot read at all used to be dropped before the design was checked, so the amended design
    looked readable and the audit's change was applied; now it is put back whole."""
    eng = _critique_engine(tmp_path, [])
    amended = {**DRAFT, "protocol": "the same as before, but with more runs"}
    out, not_applied = eng._hold_audit_repairs_to_the_schema(copy.deepcopy(DRAFT), amended)
    assert out == DRAFT
    assert not_applied and not_applied[0]["reason"]
    unrepairable = {**DRAFT, "protocol": {"grid": "no values here"}}
    out, not_applied = eng._hold_audit_repairs_to_the_schema(copy.deepcopy(DRAFT), unrepairable)
    assert out["protocol"] == DRAFT["protocol"] and not_applied
