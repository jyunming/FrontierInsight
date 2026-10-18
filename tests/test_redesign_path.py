"""A redesign (the design step run again after a run) goes through the same checks as the first design: its grid is made
small enough by the plan's model also when what the topic asks for was added to it after the first look at its size; its
code outline belongs to the design it was written for; its hypothesis is not changed after a run that produced nothing.
Fake model only; the studies are made up."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_grid_size_at_plan import BIG, SMALL, TOPIC
from tests.test_plan_optimisation import HEAT_SINK, _engine


def _redesign(grid: dict[str, list[Any]]) -> dict[str, Any]:
    draft = copy.deepcopy(HEAT_SINK)
    draft["study_type"] = "measure"
    draft["protocol"] = {"oracles": copy.deepcopy(HEAT_SINK["protocol"]["oracles"]), "grid": grid}
    return draft


@pytest.mark.asyncio
async def test_a_redesign_made_larger_by_the_topic_is_asked_to_make_its_grid_small_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    sent: list[str] = []

    async def chat(prompt: str, **kw: Any) -> str:
        sent.append(prompt)
        return json.dumps({"grid": SMALL, "sub_studies": [{"name": "all", "vary": list(SMALL), "hold": {}}]})

    async def draft(prompt: str, node: str = "") -> dict[str, Any]:
        return _redesign(SMALL)

    async def audit(state: Any, design: Any) -> tuple[Any, Any]:
        return design, None

    async def topic(state: Any, design: Any, plan_sha: str) -> tuple[Any, str]:
        # what the topic asks for is added here, and the study becomes too large to run
        return {**design, "protocol": {**design["protocol"], "grid": BIG}}, plan_sha

    async def nothing(state: Any, design: Any, **kw: Any) -> Any:
        return design

    eng._chat = chat  # type: ignore[method-assign]
    eng._client = object()
    eng._design_reply = draft  # type: ignore[method-assign]
    eng._audit_design = audit  # type: ignore[method-assign]
    eng._hold_the_design_to_the_topic = topic  # type: ignore[method-assign]
    eng._read_topic_before_the_prompt = lambda state: _none()  # type: ignore[method-assign]
    eng._design_prompt = lambda state: ""  # type: ignore[method-assign]
    eng._complete_a_drafted_search = nothing  # type: ignore[method-assign]
    eng._complete_drafted_parts = nothing  # type: ignore[method-assign]
    eng._audit_the_design_that_runs = lambda state, design: _same(design)  # type: ignore[method-assign]
    eng._check_the_ledger_of_a_redesign = lambda state, design: None  # type: ignore[method-assign]
    state = {"topic": TOPIC, "iteration": 1, "design": _redesign(SMALL)}
    patch = await eng._node_design(state)
    assert len(sent) == 1 and "settings in `protocol.grid` are" in sent[0]
    assert patch["design"]["protocol"]["grid"] == SMALL and patch["design"]["protocol"]["sub_studies"]
    assert eng._grid_too_large_stop({**state, "design": patch["design"]}) is None
    assert (eng.fi_dir / "drafted_grid_asked_again.json").is_file()


async def _none() -> None:
    return None


async def _same(design: Any) -> Any:
    return design
