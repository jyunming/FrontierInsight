"""Replay of a real stop: a plan's search block sat at the top level of the design block (beside ``protocol:``), where the
readers do not look, and its one design variable had a sentence for its ``kind`` that describes an array of values. The
ask to the plan's model listed all four parts as missing instead of the reason, the model wrote the same block again, and
the quest paused asking a person to try another model. The evidence is the real plan.md of that quest
(tests/fixtures/replay/plan_part_misplaced/). No model is called: a stub answers."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import optimisation_plan as optim
from core import plan
from tests.test_plan_optimisation import Paused, _engine

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "plan_part_misplaced"
STATE = {"topic": "find the best design", "iteration": 0}
_WRITTEN = plan.raw_design_block((FIXTURE / "plan.md").read_text(encoding="utf-8"), as_written=True) or {}
_BLOCK = _WRITTEN.get("optimisation") or {}
# Names from the evidence file itself: the design variable the plan described as an array, its shape, and the objective.
VAR = str(_BLOCK["design_variables"][0]["name"])
SHAPE = re.search(r"\d+\s*x\s*\d+", str(_BLOCK["design_variables"][0]["kind"])).group(0)  # type: ignore[union-attr]
QUANTITY = str(_BLOCK["objective"]["quantity"])


def _text() -> str:
    return (FIXTURE / "plan.md").read_text(encoding="utf-8")


def _setup(tmp_path: Path, replies: list[str]) -> tuple[Any, list[str], list[dict[str, Any]]]:
    eng = _engine(tmp_path, [])
    plan.plan_path(eng.quest_root).write_text(_text(), encoding="utf-8")
    asked: list[str] = []
    pending = list(replies)

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        asked.append(messages[-1]["content"])
        if not pending:
            raise RuntimeError("no answer")
        return pending.pop(0)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    paused: list[dict[str, Any]] = []

    def pause(**kw: Any) -> None:
        paused.append(kw)
        raise Paused(kw["kind"])

    eng._pause_for_human = pause  # type: ignore[method-assign]
    return eng, asked, paused


def _with_scalar_search(text: str) -> str:
    """The plan as a model that took the instruction would return it: a few scalar variables under ``protocol``."""
    block = {
        "objective": {"quantity": QUANTITY, "direction": "maximise"},
        "design_variables": [{"name": "shape_a", "low": 0.1, "high": 0.9, "kind": "continuous"},
                             {"name": "shape_b", "low": 0.1, "high": 0.9, "kind": "continuous"}],
        "baseline": {"values": {"shape_a": 0.5, "shape_b": 0.5}, "source": "the topic"},
        "evaluation_budget": {"starts": 2, "per_start": 10},
    }

    def change(design: dict[str, Any]) -> dict[str, Any]:
        design.pop("optimisation", None)
        design["protocol"] = {**{k: v for k, v in design["protocol"].items() if k != "grid"}, "optimisation": block}
        return design

    out = plan.edit_design_block(text, change)
    assert out is not None
    return out


def test_the_evidence_is_what_the_run_log_says() -> None:
    log = (FIXTURE / "run_log_excerpt.txt").read_text(encoding="utf-8")
    assert "FI asked the plan to complete it (2 of 2)" in log and "no search block that can be used" in log
    assert "Try another model for the plan step" in (FIXTURE / "NEXT_STEP.md").read_text(encoding="utf-8")
    written = plan.raw_design_block(_text(), as_written=True)
    assert written is not None and isinstance(written.get("optimisation"), dict)
    assert "optimisation" not in written["protocol"], "the block sat beside protocol, where nothing reads it"


def test_the_top_level_part_is_found_and_the_kind_sentence_is_reported_exactly() -> None:
    assert not optim.has_block(plan.raw_design_block(_text(), as_written=True))  # the old reading: all four parts "missing"
    block = plan.raw_design_block(_text())["protocol"]["optimisation"]  # found under protocol now
    fixed, why = optim.normalize(block)
    assert fixed is None
    assert f"`{VAR}`" in why and "array of values" in why and SHAPE in why
    assert "ONE number per design variable" in why  # the search takes scalar variables, said plainly
    entry = plan.offending_entry({"optimisation": block}, why)
    assert json.loads(entry)["name"] == VAR and SHAPE in entry


def test_a_kind_written_as_a_sentence() -> None:
    assert optim.read_kind("continuous") == ("continuous", None)
    assert optim.read_kind("Continuous, in micrometres") == ("continuous", None)
    assert optim.read_kind("one of the integer levels") == ("integer", None)
    kind, why = optim.read_kind("integer or choice")
    assert kind is None and "more than one" in why
    kind, why = optim.read_kind("a free text")
    assert kind is None and "must be exactly" in why
    for array in ("continuous 8x8 grid of intensities", "integer vector of levels", "choice per pixel",
                  "continuous array"):
        kind, why = optim.read_kind(array)
        assert kind is None and "array of values" in why, array  # never reduced to a single number


def test_other_protocol_parts_written_beside_protocol_are_moved_too() -> None:
    design = {"hypothesis": "h", "grid": {"a": [1, 2]}, "metrics": [{"id": "m"}], "runs_per_setting": 5,
              "protocol": {"grid": {"a": [9]}}}
    out, moved = plan.hoist_protocol_parts(design)
    assert sorted(moved) == ["metrics", "runs_per_setting"], "a part the protocol already has stays as the protocol has it"
    assert out["protocol"]["grid"] == {"a": [9]} and out["protocol"]["runs_per_setting"] == 5
    assert "metrics" not in out and design["metrics"] == [{"id": "m"}], "the input is not changed"
    assert plan.hoist_protocol_parts({"hypothesis": "h"}) == ({"hypothesis": "h"}, [])


@pytest.mark.asyncio
async def test_fi_moves_the_part_and_asks_with_the_exact_reason_and_ends_in_an_honest_stop(tmp_path: Path) -> None:
    # the model writes the same top-level block with the same kind sentence, as it did six times (it took out the grid)
    same = plan.edit_design_block(_text(), lambda d: {**d, "protocol": {k: v for k, v in d["protocol"].items() if k != "grid"}})
    assert same is not None
    eng, asked, paused = _setup(tmp_path, [same, same, same, same])
    design, sha = eng._design_from_plan()
    assert design is not None and sha
    after = plan.raw_design_block(plan.plan_path(eng.quest_root).read_text(encoding="utf-8"), as_written=True)
    assert "optimisation" not in after, "FI moved the part under protocol in plan.md"
    versions = plan.history(eng.quest_root)
    assert any("beside `protocol`" in str(v.get("note")) for v in versions), versions
    # what the move left out is said in plan.md with the exact reason and the entry as written
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "optimisation block was left out" in text and VAR in text and SHAPE in text

    design, sha = await eng._ask_plan_for_the_search(STATE, design, sha)
    record = json.loads((eng.fi_dir / "search_block_asked.json").read_text(encoding="utf-8"))
    assert record["count"] == 2 and record["calls"] <= 6, "bounded: two asks, at most six chat calls"
    assert len(asked) <= 6
    first, second = asked[0], next(a for a in asked[1:] if "Your last answer was not used" in a)
    assert "ONE number per design variable" in first, "the ask says plainly that the search takes scalar variables"
    assert VAR in first and "array of values" in first, "the first ask quotes the reason the block was left out"
    assert "not at the top level of the design block" in first
    assert "Your last answer was not used. The reason, exactly:" in second
    assert f"`{VAR}`" in second and "array of values" in second and SHAPE in second
    assert not paused, "nobody is asked to change the model"

    # the end: an honest stop, no pause
    assert eng._stop_if_the_search_cannot_start(STATE, design) is None
    stop = eng._search_cannot_start_stop(STATE, design)
    assert stop is not None and not paused
    result = stop["exec_result"]
    assert result["no_search"].startswith("the plan is a search for the best design, but FI could not make a search")
    assert "asked the plan's model 2 time(s)" in " ".join(result["no_search_tried"])
    # the same stop is what the run step gives on a resume, and the stuck node writes the plain record
    verdict = eng._no_results_verdict({**STATE, "exec_result": result, "result_json": {}})
    assert verdict is not None and verdict["stuck_reason"] == "no_search"
    out = await eng._node_stuck_no_findings({**STATE, "exec_result": result,
                                             "evidence_assessment": {"stuck_reason": "no_search"}})
    record = json.loads((eng.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert out["stuck"]["kind"] == "no_findings" and "FI could not make a search" in record["problem"]
    assert "another model" not in json.dumps(record).lower()
    assert not paused


@pytest.mark.asyncio
async def test_the_model_that_describes_the_design_with_scalar_parameters_gets_a_search(tmp_path: Path) -> None:
    eng, asked, paused = _setup(tmp_path, [_with_scalar_search(_text())])
    design, sha = eng._design_from_plan()
    design, sha = await eng._ask_plan_for_the_search(STATE, design, sha)
    assert len(asked) == 1 and optim.has_block(design)
    names = [v["name"] for v in design["protocol"]["optimisation"]["design_variables"]]
    assert names == ["shape_a", "shape_b"]
    assert eng._search_cannot_start_stop(STATE, design) is None and not paused


@pytest.mark.asyncio
async def test_the_reason_of_a_rejected_answer_is_kept_on_disk_and_a_resume_asks_no_more(tmp_path: Path) -> None:
    same = plan.edit_design_block(_text(), lambda d: {**d, "protocol": {k: v for k, v in d["protocol"].items() if k != "grid"}})
    assert same is not None
    eng, asked, _paused = _setup(tmp_path, [same] * 6)
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, design, sha)
    record = json.loads((eng.fi_dir / "search_block_asked.json").read_text(encoding="utf-8"))
    assert record["count"] == 2 and record["calls"] <= 6
    assert VAR in record["problem"] and "array of values" in record["problem"]
    made = len(asked)
    again, sha2 = eng._design_from_plan()
    await eng._ask_plan_for_the_search(STATE, again, sha2)
    assert len(asked) == made, "a resume does not ask again once the bound is reached"
