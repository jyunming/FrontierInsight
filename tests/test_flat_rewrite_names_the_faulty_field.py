"""A field of the search block that FI cannot read is a named fault of that field, and the model is asked again with it.

A live answer to the flat-objective request wrote the search method as a name followed by a sentence. FI logged the
reader's reason, took only the answer's readable other part, and then refused the answer with "it changed nothing of the
protocol": the second ask did not say which field was wrong and the model repeated it. Now the refusal carries the
reader's own reason (the field, what was given and the allowed values), and the first ask shows the allowed values of the
closed-choice fields and where an explanation goes. Neutral toy topic only (a heat sink).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import optimisation_plan as optim, plan
from core.engine import _flat_request
from tests.test_flat_objective_redesign import FACT, _flat_state, _planned



def _answer(eng: Any, method: Any, direction: str | None = None, kind: str | None = None) -> str:
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    block = copy.deepcopy(plan.raw_design_block(text)["protocol"]["optimisation"])
    block["design_variables"][0]["high"] = 9.5
    block["search_method"] = method
    if direction:
        block["objective"]["direction"] = direction
    if kind:
        block["design_variables"][0]["kind"] = kind
    return json.dumps({"protocol": {"optimisation": block, "simplifications": [
        {"id": "S9", "replaces": "the first design", "because": "its objective did not move", "effect": "wider range",
         "probe": "none"}]}})


class Asked:
    """The plan's model: the replies in order; every prompt kept."""

    def __init__(self, eng: Any, replies: list[str]) -> None:
        from unittest.mock import AsyncMock

        self.prompts: list[str] = []
        self.replies = list(replies)
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


WORDINGS = {
    "prose after the value": "global_then_local: sample the whole range first, then refine around the best point",
    "value, dash, note": "global_then_local — a wide first pass",
    "a sentence containing the value": "I would use global_then_local here, then refine",
    "a misspelt value": "global_then_locl",
    "a list instead of one value": ["global_then_local", "bounded_local"],
}


async def _refused(tmp_path: Path, **answer: Any) -> tuple[str, str, str]:
    """``(refusal, plan.md before, plan.md after)`` for a model that gives the same answer twice."""
    eng, _model = await _planned(tmp_path)
    Asked(eng, [_answer(eng, **answer)] * 2)
    path = plan.plan_path(eng.quest_root)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError) as refused:
        await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    return str(refused.value), before, path.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("wording", list(WORDINGS))
async def test_every_wrong_wording_of_a_closed_choice_is_a_named_fault_with_the_allowed_values(tmp_path: Path,
                                                                                             wording: str) -> None:
    said, before, after = await _refused(tmp_path, method=WORDINGS[wording])
    assert "search_method" in said, said
    assert all(v in said for v in optim.BUILT_IN_METHODS), "the allowed values are quoted"
    assert after == before, "an unreadable value is never read leniently"


@pytest.mark.asyncio
@pytest.mark.parametrize("field,given,allowed", [("direction", "downwards", ("minimise", "maximise")),
                                                 ("kind", "continous", optim.KINDS)])
async def test_a_wrong_value_in_another_closed_choice_field_is_named_the_same_way(tmp_path: Path, field: str, given: str,
                                                                                allowed: tuple[str, ...]) -> None:
    said, before, after = await _refused(tmp_path, method="bounded_local", **{field: given})
    assert field in said and all(v in said for v in allowed), said
    assert after == before


@pytest.mark.asyncio
async def test_the_second_ask_gets_the_fault_and_a_bare_value_is_then_taken(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    asked = Asked(eng, [_answer(eng, WORDINGS["prose after the value"]), _answer(eng, "global_then_local")])
    done = await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    assert done["version"] >= 2 and len(asked.prompts) == 2
    assert "search_method" in asked.prompts[1].split("Your last answer was not used.")[1]
    now = plan.raw_design_block(plan.plan_path(eng.quest_root).read_text(encoding="utf-8"))["protocol"]["optimisation"]
    assert now["search_method"] == "global_then_local"


@pytest.mark.asyncio
async def test_a_correct_answer_is_taken_with_no_note_and_no_fault(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    asked = Asked(eng, [_answer(eng, "bounded_local")])
    done = await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    assert len(asked.prompts) == 1
    assert plan.history(eng.quest_root)[done["version"] - 1]["note"] == "a design whose objective can move, asked by FI"


@pytest.mark.asyncio
async def test_the_first_ask_shows_every_closed_choice_field_of_the_asked_parts_from_the_validators_lists(
        tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    asked = Asked(eng, [_answer(eng, "bounded_local")])
    await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    first = asked.prompts[0]
    for name, values in optim.CLOSED_CHOICES:
        assert f"`{name}` is one of {', '.join(values)}" in first
    assert "`simplifications`" in first
    assert optim.closed_choices_note(("grid",)) == "", "a request that does not ask for the block shows nothing"
