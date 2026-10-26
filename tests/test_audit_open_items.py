"""A change the methodology audit proposed that the design does not hold is kept open, never dropped."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core import audit_open_items as ao
from core import design_parts as dp
from core import code_review, plan, quality_report
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

DRAFT = {
    "hypothesis": "the integrator converges at fourth order",
    "variables": {"independent": ["dt"], "dependent": ["period_error"], "controls": ["seed"]},
    "method": "measure the period from zero-crossings",
    "expected_outcome": "error falls as dt^4",
    "figures_planned": ["c.png"],
    "dependencies": ["numpy"],
    "protocol": {"runs_per_setting": 5, "thresholds": {"tol": 0.1}},
}
ANCHOR = "Measure every method against the finest step that all of them can reach, and say which ones cannot reach it."


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="pendulum period", title="pend", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=[])})()
    return eng


def _objection(*changes: tuple[str, object]) -> dict:
    return {"check": "weak_baseline", "where": "method", "objection": "the methods are anchored to different steps",
            "fix": "one shared anchor", "changes": [{"path": p, "value": v} for p, v in changes]}


def _items(eng: Engine) -> dict:
    return json.loads((eng.quest_root / ".fi" / "audit_open_items.json").read_text(encoding="utf-8"))


def _write_plan(eng: Engine) -> dict:
    design, _why = plan.normalize_design(DRAFT)
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("pendulum period", {}, design, []), encoding="utf-8")
    plan.record_version(eng.quest_root, path.read_text(encoding="utf-8"), by="engine", note="test")
    return plan.parse(path.read_text(encoding="utf-8")).design


def test_a_proposal_the_design_lacks_is_missing_and_one_it_has_is_held() -> None:
    props = ao.proposals([_objection(("protocol.anchor_rule", ANCHOR), ("method", "measure from interpolated crossings"))])
    design = {**DRAFT, "method": "measure from interpolated crossings"}
    held, missing = ao.split(props, design)
    assert [p["path"] for p in held] == ["method"] and [p["path"] for p in missing] == ["protocol.anchor_rule"]
    assert missing[0]["value"] == ANCHOR and missing[0]["objection"].startswith("the methods are anchored")


def test_a_path_the_protocol_has_no_home_for_is_kept_open_with_the_audits_full_text(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    objs = [_objection(("protocol.anchor_rule", ANCHOR))]
    amended, refused, _ = dp.amended_from_audit(DRAFT, objs)
    assert refused and "no home" in refused[0]["reason"]  # no longer guessed to be a misspelling
    eng._keep_audit_open_items(objs, amended, reasons={r["key"]: r["reason"] for r in refused}, default_reason="x")
    item = _items(eng)["open"][0]
    assert item["path"] == "protocol.anchor_rule" and item["value"] == ANCHOR and "no home" in item["reason"]
    block = json.loads(eng._code_design_block({"design": DRAFT}))
    shown = block["proposed_by_the_methodology_audit_not_applied"]["items"]
    assert shown[0]["proposed"] == ANCHOR and shown[0]["audit_said"].startswith("the methods are anchored")
    assert ANCHOR in ao.writer_note(eng.quest_root) and "limitations" in ao.writer_note(eng.quest_root)


def test_a_misspelt_name_next_to_a_real_part_is_still_reported_and_not_applied() -> None:
    why = dp.invalid_reason("protocol.orracles", {**DRAFT, "protocol": {"oracles": []}})
    assert "misspelt" in why and "`protocol.oracles`" in why
    amended, refused, _ = dp.amended_from_audit(DRAFT, [_objection(("protocol.orracles", [1]))])
    assert "orracles" not in amended["protocol"] and refused[0]["key"] == "protocol.orracles"


def test_an_open_item_leaves_the_list_once_a_later_design_holds_it(tmp_path: Path) -> None:
    props = ao.proposals([_objection(("protocol.anchor_rule", ANCHOR))])
    ao.record(tmp_path, [], props, reasons={}, default_reason="FI did not apply it")
    assert [x["path"] for x in ao.read(tmp_path)["open"]] == ["protocol.anchor_rule"]
    ao.record(tmp_path, props, [], reasons={}, default_reason="")
    assert ao.read(tmp_path)["open"] == [] and ao.read(tmp_path)["applied"][0]["path"] == "protocol.anchor_rule"


def test_a_change_reported_as_made_that_the_frozen_protocol_lacks_becomes_open(tmp_path: Path) -> None:
    props = ao.proposals([_objection(("protocol.runs_per_setting", 50))])
    ao.record(tmp_path, props, [], reasons={}, default_reason="")
    assert ao.check_against(tmp_path, {"runs_per_setting": 50}, why="w") == []
    lost = ao.check_against(tmp_path, {"runs_per_setting": 5}, why="the frozen protocol lacks it")
    assert [x["path"] for x in lost] == ["protocol.runs_per_setting"]
    assert ao.read(tmp_path)["open"][0]["reason"] == "the frozen protocol lacks it"


@pytest.mark.asyncio
async def test_a_change_held_back_after_the_audit_reported_it_is_kept_open(tmp_path: Path) -> None:
    """The audit changed `protocol.thresholds` (a check the results are judged by, which FI puts back): the design that
    runs has the old value, and the proposal is an open item, not only a log line."""
    eng = _engine(tmp_path)
    design = _write_plan(eng)
    change = _objection(("protocol.thresholds", {"tol": 0.01}), ("method", "measure from interpolated crossings"),
                        ("protocol.anchor_rule", ANCHOR))
    eng._client.chat.side_effect = [
        json.dumps({"objections_addressed": [change]}),
        json.dumps({"objections_addressed": []}),
    ]
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs["protocol"]["thresholds"] == {"tol": 0.1} and "interpolated" in runs["method"]
    open_paths = {x["path"]: x for x in _items(eng)["open"]}
    assert set(open_paths) == {"protocol.thresholds", "protocol.anchor_rule"}
    assert "judged by" in open_paths["protocol.thresholds"]["reason"]
    assert "no home" in open_paths["protocol.anchor_rule"]["reason"]
    assert [x["path"] for x in _items(eng)["applied"]] == ["method"]


def test_quality_lists_what_the_audit_proposed_and_the_plan_lacks() -> None:
    q = SimpleNamespace(critique=[], calls=[], receipts={}, drafted_asked={}, design_acted=[],
                        audit_open=[{"path": "protocol.anchor_rule", "origins": [{"objection": "different anchors"}],
                                     "reason": "no home"}])
    got = quality_report._design_problems(q)
    assert any("proposed by the audit, not applied" in p["text"] and "protocol.anchor_rule" in p["text"] for p in got)


# ---- the faults an outside review found -------------------------------------------------------------------------------

def test_a_later_proposal_on_the_same_path_does_not_close_an_earlier_one(tmp_path: Path) -> None:
    first = ao.proposals([_objection(("method", "M1"))])
    ao.record(tmp_path, [], first, reasons={}, default_reason="x", iteration=0)
    later = ao.proposals([_objection(("method", "M2"))])
    ao.record(tmp_path, later, [], reasons={}, default_reason="", iteration=1)
    assert [x["value"] for x in ao.read(tmp_path)["open"]] == ["M1"]
    both = ao.proposals([_objection(("method", "A"), ("method", "B"))])
    assert [p["value"] for p in both] == ["A", "B"]


def test_the_same_proposal_in_two_passes_is_one_item_with_two_origins(tmp_path: Path) -> None:
    for n in (0, 1):
        ao.record(tmp_path, [], ao.proposals([_objection(("protocol.anchor_rule", ANCHOR))]), reasons={},
                  default_reason="x", iteration=n)
    (item,) = ao.read(tmp_path)["open"]
    assert [o["iteration"] for o in item["origins"]] == [0, 1]


def test_an_applied_scalar_the_plan_reads_as_a_list_is_not_lost_at_the_freeze(tmp_path: Path) -> None:
    props = ao.proposals([_objection(("protocol.grid.dt", 0.1))])
    design, _why = plan.normalize_design({**DRAFT, "protocol": {**DRAFT["protocol"], "grid": {"dt": [0.1]}}})
    held, missing = ao.split(props, design)
    assert held and not missing
    ao.record(tmp_path, held, [], reasons={}, default_reason="")
    assert ao.read(tmp_path)["applied"][0]["value"] == [0.1]
    assert ao.check_against(tmp_path, design["protocol"], why="w") == [] and ao.read(tmp_path)["open"] == []


@pytest.mark.asyncio
async def test_an_open_item_the_design_now_holds_is_closed_without_the_audit_repeating_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    objs = [_objection(("method", "measure from interpolated crossings"))]
    eng._keep_audit_open_items(objs, DRAFT, reasons={}, default_reason="the design was already settled")
    assert len(_items(eng)["open"]) == 1
    edited = {**DRAFT, "method": "measure from interpolated crossings"}
    eng._keep_audit_open_items([], edited, reasons={}, default_reason="")  # a re-audit that proposes nothing
    assert _items(eng)["open"] == [] and _items(eng)["applied"][0]["path"] == "method"


def test_a_damaged_record_is_kept_aside_and_a_failed_read_is_never_written_over(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    (fi / "audit_open_items.json").write_text('{"open": [{"path": "me', encoding="utf-8")
    props = ao.proposals([_objection(("method", "M1"))])
    ao.record(tmp_path, [], props, reasons={}, default_reason="x")
    kept = list(fi.glob("audit_open_items.corrupt-*.json"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8").startswith('{"open": [{"path": "me')
    assert [x["path"] for x in ao.read(tmp_path)["open"]] == ["method"] and not list(fi.glob("*.tmp"))
    (fi / "audit_open_items.json").unlink()
    (fi / "audit_open_items.json").mkdir()  # cannot be opened as a file: nothing may be written over it
    assert ao.record(tmp_path, [], props, reasons={}, default_reason="x") == []


def test_the_open_items_are_one_key_of_the_single_plan_json_and_come_before_the_plans_cut(tmp_path: Path) -> None:
    ao.record(tmp_path, [], ao.proposals([_objection(("protocol.anchor_rule", ANCHOR))]), reasons={}, default_reason="x")
    view = code_review.plan_view(DRAFT, DRAFT["protocol"], open_items=ao.for_code(tmp_path))
    assert json.loads(view)["proposed_by_the_methodology_audit_not_applied"]["items"][0]["proposed"] == ANCHOR
    big = "x" * 8000
    ao.record(tmp_path, [], ao.proposals([{**_objection(("protocol.other_rule", "y")), "objection": big}]),
              reasons={}, default_reason="x")
    view = code_review.plan_view(DRAFT, {**DRAFT["protocol"], "grid": {"a": list(range(3000))}},
                                 open_items=ao.for_code(tmp_path))
    assert "(the rest of the plan is not shown)" in view  # the plan was cut, after the whole audit section
    assert big in view and view.index(big) < view.index("(the rest of the plan is not shown)")


def test_two_objections_with_the_same_text_keep_each_fix_as_an_origin(tmp_path: Path) -> None:
    objs = [{**_objection(("method", "M1")), "fix": "F1"}, {**_objection(("method", "M1")), "fix": "F2"}]
    props = ao.proposals(objs)
    assert [p["fix"] for p in props] == ["F1", "F2"]
    ao.record(tmp_path, [], props, reasons={}, default_reason="x")
    (item,) = ao.read(tmp_path)["open"]
    assert [o["fix"] for o in item["origins"]] == ["F1", "F2"]


def test_held_is_decided_in_the_context_of_the_actual_design() -> None:
    """A numerical setting is folded into the grid by the plan: the design holds the grid as the proposal gives it."""
    design, why = plan.normalize_design({
        **DRAFT, "protocol": {"grid": {"x": [1, 2]}, "numerical_settings": {"n": {"search": 8, "check": [4]}}}})
    assert design is not None, why
    assert design["protocol"]["grid"] == {"x": [1, 2], "n": [8]}
    held, missing = ao.split(ao.proposals([_objection(("protocol.grid", {"x": [1, 2]}))]), design)
    assert held and not missing and held[0]["norm"] == {"x": [1, 2], "n": [8]}  # the value in this design's context
    held, missing = ao.split(ao.proposals([_objection(("protocol.grid", {"x": [1, 2, 3]}))]), design)
    assert missing and not held


def test_an_item_keeps_its_id_when_it_is_recorded_again_in_another_context(tmp_path: Path) -> None:
    props = ao.proposals([_objection(("method", "M1"))])
    ao.record(tmp_path, [], ao.split(props, DRAFT)[1], reasons={}, default_reason="x")
    (first,) = ao.read(tmp_path)["open"]
    ao.record(tmp_path, [], [{**first, "objection": "again", "fix": "", "check": ""}], reasons={}, default_reason="x",
              iteration=1)
    (again,) = ao.read(tmp_path)["open"]
    assert again["id"] == first["id"] and len(again["origins"]) == 2


@pytest.mark.asyncio
async def test_the_design_handed_to_implementation_closes_the_items_it_holds(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._keep_audit_open_items([_objection(("method", "measure from interpolated crossings"))], DRAFT, reasons={},
                               default_reason="x")
    design = _write_plan(eng)
    edited = {**design, "method": "measure from interpolated crossings"}
    state = {"topic": "t", "iteration": 0, "design": edited, "design_from_plan": True}
    eng._stop_if_no_settings_to_vary = lambda *a, **k: None
    eng._stop_if_the_search_cannot_start = lambda *a, **k: None
    eng._audit_the_design_that_runs = AsyncMock(return_value=edited)
    eng._hold_the_design_to_the_topic = AsyncMock(return_value=(edited, "sha"))
    eng._amend_to_fit = lambda d: d
    try:
        await eng._node_design(state)
    except Exception:  # noqa: BLE001 -- the node's later steps are not under test; the close comes before them
        pass
    assert _items(eng)["open"] == []


def test_a_record_with_a_byte_that_is_not_text_is_kept_aside_not_a_crash(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    (fi / "audit_open_items.json").write_bytes(b'{"open": [\xff]}')
    assert ao.read(tmp_path) == {"open": [], "applied": []}
    assert len(list(fi.glob("audit_open_items.corrupt-*.json"))) == 1
    assert ao.record(tmp_path, [], ao.proposals([_objection(("method", "M1"))]), reasons={}, default_reason="x")
