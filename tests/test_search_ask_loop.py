"""The plan step's request for a search block stops asking when the plan was refused for something other than the block,
quotes the reader's reason in the next request when it was the block, and does not say a search was completed when it
never was. Fake model only; neutral made-up names (see tests/test_search_block_converges.py for the plan used)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import plan
from core.engine_base import _PlanBeyondTheBlock
from tests.test_replay_misplaced_part import STATE, _setup
from tests.test_search_block_converges import BASE, _reply
from tests.test_search_controls import _plan_without_a_block

BEYOND = "the plan with your block in it cannot be read: `protocol.thresholds` entry 1: not a number"


def _raiser(error: Exception, seen: list[str]) -> Any:
    def keep(before_text: str, answer: dict[str, Any]) -> str | None:
        seen.append("kept")
        raise error

    return keep


@pytest.mark.asyncio
async def test_a_plan_refused_for_something_other_than_the_block_is_not_asked_again(tmp_path: Path) -> None:
    """Before: the same refusal cost a second request (and, as FI's own notes changed the plan's hash, a fresh pair of
    requests at each later step). Now: one request, and no more for as long as the plan's design block is the same."""
    eng, asked, _paused = _setup(tmp_path, [_reply(BASE), _reply(BASE), _reply(BASE)])
    seen: list[str] = []
    eng._keep_only_the_search = _raiser(_PlanBeyondTheBlock(BEYOND), seen)  # type: ignore[method-assign]
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    assert len(asked) == 1, "one request, not two"
    record = json.loads((eng.fi_dir / "search_block_asked.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "refused beyond the block" and "not a number" in record["problem"]
    # FI's own note in plan.md changes the plan's hash but not its design block: still not asked
    path = plan.plan_path(eng.quest_root)
    path.write_text(path.read_text(encoding="utf-8") + "\n\nA note FI wrote beside the plan.\n", encoding="utf-8")
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    assert len(asked) == 1
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "FI does not ask the plan's model for the search again" in log and "not a number" in log


@pytest.mark.asyncio
async def test_a_plan_whose_design_changed_since_that_refusal_is_asked_again(tmp_path: Path) -> None:
    eng, asked, _paused = _setup(tmp_path, [_reply(BASE), _reply(BASE)])
    eng._keep_only_the_search = _raiser(_PlanBeyondTheBlock(BEYOND), [])  # type: ignore[method-assign]
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    path = plan.plan_path(eng.quest_root)
    changed = path.read_text(encoding="utf-8").replace("hypothesis:", "hypothesis: (reworded)", 1)
    assert changed != path.read_text(encoding="utf-8")
    path.write_text(changed, encoding="utf-8")
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    assert len(asked) == 2


@pytest.mark.asyncio
async def test_known_positive_a_refused_block_is_still_asked_again_with_the_readers_reason(tmp_path: Path) -> None:
    """The refusal of the block itself keeps its second request, and that request quotes the reader's reason."""
    bad = copy.deepcopy(BASE)
    bad["constraints"] = [{"quantity": "loss", "max": 3}]
    eng, asked, _paused = _setup(tmp_path, [_reply(bad), _reply(bad)])
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    assert len(asked) >= 2 and "you wrote `max`" in asked[1]


@pytest.mark.asyncio
async def test_the_retry_after_a_refusal_of_the_plan_quotes_the_reason(tmp_path: Path) -> None:
    eng, asked, _paused = _setup(tmp_path, [_reply(BASE), _reply(BASE)])
    # a plan with no block at all (the fixture's own block is unreadable, which would ask for its faulted paths instead)
    plan.plan_path(eng.quest_root).write_text(_plan_without_a_block([]), encoding="utf-8")
    (eng.fi_dir / "search_refused.json").unlink(missing_ok=True)
    eng._keep_only_the_search = _raiser(ValueError("the plan with your block in it cannot be read: marker_reason"), [])  # type: ignore[method-assign]
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    assert len(asked) == 2 and "marker_reason" not in asked[0] and "marker_reason" in asked[1]


@pytest.mark.asyncio
async def test_a_search_that_was_never_completed_is_not_logged_as_completed(tmp_path: Path) -> None:
    eng, asked, _paused = _setup(tmp_path, [])
    design, sha = eng._design_from_plan()
    await eng._ask_again_for_a_search_lost_since(STATE, design, sha)
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "the plan still has no search FI can run" in log
    assert "after its search was completed" not in log
