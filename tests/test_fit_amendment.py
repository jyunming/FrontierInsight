"""A study too long to run, rewritten to fit the time FI measured, after its protocol was frozen (core/engine.py,
``Engine._amend_to_fit``).

A live search was frozen and started, stopped part-way (the machine ran low on memory), resumed, timed, and found too long;
the plan was rewritten to fit (6 starts of 60 evaluations instead of 3000), but the frozen protocol still held the old
search, so FI timed the old one again and stopped the quest with no paper. Now FI approves the rewritten plan as the frozen
protocol itself: made before results when FI's record holds no result of the study. Neutral toy topic only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import frozen_protocol, plan
from tests.test_flat_objective_redesign import TOPIC, _no_pause, _planned
from tests.test_plan_optimisation import EXTRA, HEAT_SINK


def _fitted() -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    design["protocol"]["optimisation"]["evaluation_budget"] = {"starts": 2, "per_start": 7}
    return design


def _after_fit(eng: Any) -> dict[str, Any]:
    plan.plan_path(eng.quest_root).write_text(plan.render(TOPIC, EXTRA, plan.normalize_design(_fitted())[0]),
                                              encoding="utf-8")
    return {"topic": TOPIC, "iteration": 0, "design": copy.deepcopy(HEAT_SINK), "design_from_plan": True,
            "evidence_fit_redesigns": 1, "result_json": {},
            "exec_result": {"returncode": 1, "too_long": "the study would take 29 h, more than the 8 h allowed"}}


@pytest.mark.asyncio
async def test_a_plan_rewritten_to_fit_becomes_the_frozen_protocol_before_results(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    eng._freeze_protocol_if_due({"topic": TOPIC, "iteration": 0})
    before = frozen_protocol.load(eng.quest_root)
    assert before is not None
    assert before["protocol"]["optimisation"]["evaluation_budget"] != {"starts": 2, "per_start": 7}
    raw = eng._raw_root()
    raw.mkdir(parents=True, exist_ok=True)
    # A search started and stopped before any evaluation finished: its header and planned lines are not results.
    (raw / "optimisation_ledger.jsonl").write_text('{"event": "search"}\n', encoding="utf-8")
    _no_pause(eng)
    patch = await eng._node_design(_after_fit(eng))
    now = frozen_protocol.load(eng.quest_root)
    assert now["protocol"]["optimisation"]["evaluation_budget"]["per_start"] == 7, "the smaller study is what runs"
    assert patch["design"]["protocol"]["optimisation"]["evaluation_budget"]["per_start"] == 7
    record = json.loads((eng.quest_root / "needs" / "PROTOCOL_AMENDMENT_1.json").read_text(encoding="utf-8"))
    assert record["prespecified"] is True and record["results_seen_before_change"] is False
    assert record["approved_by"].startswith("FI:") and "fit the time" in record["reason"]
    assert not (eng.quest_root / "archive" / "run_1" / "raw").exists()
    assert not frozen_protocol.post_hoc(eng.quest_root)


@pytest.mark.asyncio
async def test_a_plan_rewritten_to_fit_after_results_were_recorded_says_so_and_keeps_them(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    eng._freeze_protocol_if_due({"topic": TOPIC, "iteration": 0})
    raw = eng._raw_root()
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "optimisation_ledger.jsonl").write_text(
        '{"event": "search"}\n{"event": "evaluation", "status": "ok", "objective": 1.0}\n', encoding="utf-8")
    _no_pause(eng)
    await eng._node_design(_after_fit(eng))
    record = json.loads((eng.quest_root / "needs" / "PROTOCOL_AMENDMENT_1.json").read_text(encoding="utf-8"))
    assert record["prespecified"] is False and record["results_seen_before_change"] is True
    assert (eng.quest_root / "archive" / "run_1" / "raw" / "optimisation_ledger.jsonl").is_file()
    assert frozen_protocol.post_hoc(eng.quest_root)


@pytest.mark.asyncio
async def test_with_nothing_frozen_a_plan_rewritten_to_fit_needs_no_amendment(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    assert frozen_protocol.load(eng.quest_root) is None
    _no_pause(eng)
    patch = await eng._node_design(_after_fit(eng))
    assert patch["design"]["protocol"]["optimisation"]["evaluation_budget"]["per_start"] == 7
    assert not (eng.quest_root / "needs" / "PROTOCOL_AMENDMENT_1.json").exists()
