"""Replay of a real stop: a study that both compares fixed settings and searches for the best design.

In tests/fixtures/replay/search_with_sweep/ the plan is a search for the best design (``protocol.optimisation``). The topic
names ten settings to vary. The reader of the topic asked for them in ``protocol.grid``; the search rule refused a plan with
both a search and a top-level ``protocol.grid``; the plan's model was asked again and again, every answer was refused for
that conflict, and the quest paused asking a person to try another model, which cannot fix a contradiction in FI's own
rules.

The rule now: the settings a search compares go in the block's own scan (``protocol.optimisation.grid``): a design
variable, or a condition listed in ``protocol.optimisation.fixed`` with the value the search holds it at. The reader of the
topic counts what the block compares as covered, the request says where to write it, and when the bounded asks still leave a
gap the outcome is the honest stop (no paper), never a pause. No real model is called: a stub answers."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan
from core import topic_requirements as tr
from tests.plan_parts_helpers import as_parts
from tests.test_plan_optimisation import Paused, _engine

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "search_with_sweep"
REQ = tr.parse(json.loads((FIXTURE / "topic_requirements.json").read_text(encoding="utf-8")))
TEXT = (FIXTURE / "plan.md").read_text(encoding="utf-8")
DESIGN = plan.parse(TEXT).design
JUDGED = next(iter(json.loads((FIXTURE / "topic_covered.json").read_text(encoding="utf-8"))["judged"].values()))


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _comparison() -> dict[str, list[Any]]:
    """The settings to compare the way a plan's model writes them: one key per thing the topic asks to vary, with the
    levels or the range the topic states. The broadest name first, so an axis whose name another factor's contains is
    never the first match of that other factor (a name-matching quirk that has nothing to do with this rule)."""
    grid: dict[str, list[Any]] = {}
    for factor in sorted(REQ["factors"], key=lambda f: -len(str(f["name"]).split())):
        values = [v for v in (*factor["levels"], *factor["range"])]
        numbers = [v for v in values if isinstance(v, (int, float))]
        grid[_slug(factor["name"])] = (values if len(numbers) in (0, len(values)) else numbers) or [0.5, 1.0]  # no value stated
    return grid


def test_the_evidence_is_what_the_real_quest_held() -> None:
    assert DESIGN is not None and isinstance(DESIGN["protocol"].get("optimisation"), dict)
    assert "grid" not in DESIGN["protocol"], "a search for the best design: no top-level sweep"
    assert len([f for f in REQ["factors"] if f.get("main")]) == 10
    asked = json.loads((FIXTURE / "topic_asked.json").read_text(encoding="utf-8"))
    assert "cannot both be given" in asked["problem"] and asked["outcome"] == "not usable"
    assert "Try another model for the plan step" in (FIXTURE / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "paused" in (FIXTURE / "run_log_excerpt.txt").read_text(encoding="utf-8")


def test_a_sweep_beside_a_search_is_refused_with_the_place_to_write_it_instead() -> None:
    protocol = copy.deepcopy(DESIGN["protocol"])
    protocol["grid"] = _comparison()
    fixed, why = plan.normalize_protocol(protocol)
    assert fixed is None
    assert "cannot both be given" in why and "optimisation.grid" in why and "optimisation.fixed" in why


def test_a_condition_the_search_compares_must_say_the_value_it_is_held_at() -> None:
    protocol = copy.deepcopy(DESIGN["protocol"])
    protocol["optimisation"]["grid"] = {"setting_not_held": [1.0, 2.0]}
    fixed, why = plan.normalize_protocol(protocol)
    assert fixed is None and "setting_not_held" in why and "optimisation.fixed" in why
    protocol["optimisation"]["fixed"] = {**protocol["optimisation"]["fixed"], "setting_not_held": 1.0}
    fixed, why = plan.normalize_protocol(protocol)
    assert fixed is not None, why
    protocol["optimisation"]["grid"] = {"setting_not_held": [1.0, "wide"]}
    assert plan.normalize_protocol(protocol)[0] is None, "all numbers or all names, as every other sweep"


def test_the_answer_written_the_old_way_is_read_into_the_blocks_own_scan_and_covers_what_the_topic_asked() -> None:
    gaps, _unresolved = tr.match(REQ, tr.view(DESIGN))
    comparison = _comparison()
    answer = {"protocol": {"grid": comparison,
                           "optimisation": {"fixed": {k: v[0] for k, v in comparison.items()}}}}
    merged = tr.merge(DESIGN, answer, gaps)
    assert merged is not None and "grid" not in merged["protocol"], "no sweep beside the search"
    assert set(comparison) <= set(merged["protocol"]["optimisation"]["grid"])
    fixed, why = plan.normalize_protocol(merged["protocol"])
    assert fixed is not None, why  # the search rule accepts it
    gaps_after, unresolved_after = tr.match(REQ, tr.view({**merged, "protocol": fixed}))
    assert gaps_after == [], "the topic's reader agrees: nothing the plan states is short of what the topic asks"
    still = {i for i in unresolved_after if i.startswith("F")}
    assert len(still) < len(REQ["factors"]), "a setting named in the block's scan is found by name: " + ", ".join(sorted(still))


def test_the_block_counts_for_what_the_search_itself_finds() -> None:
    view = tr.view(DESIGN)
    assert view["search"] is True
    item = REQ["factors"][0]
    plan_map = {**view, "topic_map": [{"item": item["name"], "by": "optimisation"}]}
    assert tr.mapping_of(REQ, plan_map, "factors", item) == ("judge", "optimisation"), "a reader confirms it, never a name"
    assert tr.mapping_of(REQ, {**plan_map, "search": False}, "factors", item) == ("bad", "nothing it names exists")


def test_the_request_to_the_plans_model_says_where_a_search_keeps_the_settings_to_compare() -> None:
    gap = tr.gap("factor", REQ["factors"][0])
    text = tr.request("t", [gap], search=True)
    assert "protocol.optimisation.grid" in text and "protocol.optimisation.fixed" in text and "do NOT write a `protocol.grid`" in text
    assert "protocol.grid" in tr.request("t", [gap]) and "optimisation.fixed" not in tr.request("t", [gap])


def _setup(tmp_path: Path, plan_replies: list[str]) -> tuple[Any, list[str], list[dict[str, Any]]]:
    eng = _engine(tmp_path, [])
    plan.plan_path(eng.quest_root).write_text(TEXT, encoding="utf-8")
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / tr.FILE).write_text((FIXTURE / "topic_requirements.json").read_text(encoding="utf-8"), encoding="utf-8")
    eng.config.topic = "a study that compares settings and searches for the best design"
    asked: list[str] = []
    pending = list(plan_replies)

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        content = messages[-1]["content"]
        if "The topic asks for things this plan does not yet include" in content:
            asked.append(content)
            if not pending:
                raise RuntimeError("no answer")
            return as_parts(pending.pop(0))
        # the reader of the topic, as it answered in the real quest
        return json.dumps({"items": [{"id": i, "verdict": v} for i, v in JUDGED.items()]})

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    paused: list[dict[str, Any]] = []

    def pause(**kw: Any) -> None:
        paused.append(kw)
        raise Paused(kw["kind"])

    eng._pause_for_human = pause  # type: ignore[method-assign]
    return eng, asked, paused


def _plan_with(grid_place: str) -> str:
    """The plan as the model returns it, with the settings to compare written at ``grid_place``."""
    comparison = _comparison()

    def change(design: dict[str, Any]) -> dict[str, Any]:
        protocol = design["protocol"]
        if grid_place == "top":
            protocol["grid"] = comparison
        else:
            protocol["optimisation"]["grid"] = comparison
            protocol["optimisation"]["fixed"] = {**protocol["optimisation"].get("fixed", {}),
                                                 **{k: v[0] for k, v in comparison.items()}}
        return design

    out = plan.edit_design_block(TEXT, change)
    assert out is not None
    return out


@pytest.mark.asyncio
async def test_an_answer_with_the_sweep_beside_the_search_is_refused_once_then_the_block_is_asked_for(tmp_path: Path) -> None:
    eng, asked, paused = _setup(tmp_path, [_plan_with("top"), _plan_with("block")])
    design, sha = eng._design_from_plan()
    out, _sha = await eng._ask_plan_for_the_topic({"topic": "t"}, design, sha, REQ)
    assert paused == []
    assert len(asked) >= 2 and "protocol.optimisation.grid" in asked[0], "the ask names the place the settings go"
    # The sweep written beside the search is read as the search's own scan (so a plan never holds both); that scan names a
    # setting the search neither varies nor holds fixed, and the reader's exact reason goes back to the model.
    assert "neither a design variable nor a condition held fixed" in asked[1], "the first answer was refused, and the model was told exactly why"
    assert set(_comparison()) <= set(out["protocol"]["optimisation"]["grid"]), "the plan covers what the topic asked"
    assert "grid" not in out["protocol"]


@pytest.mark.asyncio
async def test_when_the_asks_leave_a_gap_the_outcome_is_the_honest_stop_and_never_a_pause(tmp_path: Path) -> None:
    eng, asked, paused = _setup(tmp_path, [TEXT, TEXT])  # the plan's model returns the plan unchanged, twice
    design, _sha = eng._design_from_plan()
    await eng._node_design({"topic": eng.config.topic, "iteration": 0})
    assert paused == [], "nobody is asked to try another model"
    assert 1 <= len(asked) <= 4, "bounded"
    stop = eng._topic_unmet_stop()
    assert stop is not None
    result = stop["exec_result"]
    assert result["returncode"] == 1 and result["topic_unmet"] and result["topic_unmet_tried"]
    assert any("asked the plan's model" in line for line in result["topic_unmet_tried"])
    # the code step stops the same way and writes no code
    assert (await eng._node_implement({"topic": eng.config.topic, "design": design}))["exec_result"]["topic_unmet"]
    again = len(asked)
    await eng._node_design({"topic": eng.config.topic, "iteration": 0})
    assert len(asked) == again, "a resume asks no more"
