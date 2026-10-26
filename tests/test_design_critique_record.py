"""The methodology audit of a design leaves a record (needs/DESIGN_CRITIQUE.json) and asks the questions the audit found missing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import plan, receipts
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

AGENTS = Path(__file__).resolve().parent.parent / "agents"

DRAFT = {
    "hypothesis": "model-based OPC reduces EPE more than rule-based",
    "variables": {"independent": ["strategy"], "dependent": ["epe"], "controls": ["seed"]},
    "method": "compare three strategies on synthetic clips",
    "expected_outcome": "model-based wins",
    "figures_planned": ["c.png"],
    "dependencies": ["numpy"],
    "protocol": {"runs_per_setting": 300},
}
OBJECTION = {"check": "precision", "where": "method, protocol.runs_per_setting, protocol.optimisation", "objection": "300 runs give about +-0.057", "fix": "raised the runs to 1000"}


def _engine(tmp_path: Path, replies: list[str]) -> Engine:
    cfg = Config(
        topic="OPC under low k1", title="opc", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=replies)})()
    return eng


def _record(eng: Engine) -> list[dict]:
    return json.loads((eng.quest_root / "needs" / "DESIGN_CRITIQUE.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_what_the_audit_objected_to_and_changed_is_kept_in_full(tmp_path: Path) -> None:
    amended = {**DRAFT, "protocol": {"runs_per_setting": 1000}, "method": "compare three strategies with 1000 runs each"}
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended})])
    design, objections = await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert design == amended and objections == [OBJECTION]
    (entry,) = _record(eng)
    assert entry["status"] == "ok" and entry["failure"] == "" and entry["iteration"] == 0
    assert entry["objections"] == [OBJECTION]
    assert entry["changed_keys"] == ["method", "protocol"]
    assert entry["before"]["protocol"] == {"runs_per_setting": 300} and entry["after"]["protocol"] == {"runs_per_setting": 1000}


@pytest.mark.asyncio
async def test_an_audit_that_found_nothing_is_recorded_as_such(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [], "amended_design": DRAFT})])
    await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    (entry,) = _record(eng)
    assert entry["status"] == "ok" and entry["objections"] == [] and entry["changed_keys"] == []


@pytest.mark.asyncio
async def test_an_audit_that_failed_says_so_instead_of_looking_like_one_that_found_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=RuntimeError("provider down"))})()
    design, objections = await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert design == DRAFT and objections is None
    (entry,) = _record(eng)
    assert entry["status"] == "failed" and "provider down" in entry["failure"] and entry["changed_keys"] == []


@pytest.mark.asyncio
async def test_an_amended_design_that_dropped_a_key_is_recorded_as_failed_and_the_draft_is_kept(tmp_path: Path) -> None:
    dropped = {k: v for k, v in DRAFT.items() if k != "protocol"}
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": dropped})])
    design, _ = await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert design == DRAFT
    (entry,) = _record(eng)
    assert entry["status"] == "failed" and "protocol" in entry["failure"]


@pytest.mark.asyncio
async def test_every_pass_adds_an_entry(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [], "amended_design": DRAFT})] * 2)
    await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    await eng._audit_design({"topic": "OPC", "iteration": 1}, dict(DRAFT))
    assert [e["iteration"] for e in _record(eng)] == [0, 1]


@pytest.mark.asyncio
async def test_the_plan_says_what_the_audit_changed_or_that_it_did_not_finish(tmp_path: Path) -> None:
    amended = {**DRAFT, "method": "a different method"}
    eng = _engine(tmp_path, [json.dumps(DRAFT), json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended})])
    await eng._node_plan({"topic": "OPC", "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "The methodology audit amended the design: method" in text and "needs/DESIGN_CRITIQUE.json" in text

    failing = _engine(tmp_path / "f", [json.dumps(DRAFT)])
    failing._client = type("Stub", (), {"chat": AsyncMock(side_effect=[json.dumps(DRAFT), RuntimeError("down")])})()
    await failing._node_plan({"topic": "OPC", "iteration": 0})
    assert "The methodology audit did not complete" in plan.plan_path(failing.quest_root).read_text(encoding="utf-8")


def test_the_audit_asks_about_precision_the_estimand_thresholds_streams_convergence_oracles_and_failed_runs() -> None:
    text = (AGENTS / "design_self_critique.md").read_text(encoding="utf-8")
    for check in ("precision", "estimand", "threshold_sensitivity", "rng_independence", "numerical_convergence", "oracle", "failed_runs"):
        assert check in text, check
    for words in ("0.98/sqrt(n)", "not a sample size", "common-random-numbers", "protocol.oracles", "Do not repeat parts you do not change"):
        assert words in text, words


@pytest.mark.asyncio
async def test_each_audit_leaves_a_receipt_and_a_research_quest_stops_once_when_it_could_not_judge(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [], "amended_design": DRAFT})])
    await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    receipt = json.loads((eng.quest_root / "needs" / "receipts" / "design_audit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "pass" and receipt["input_hashes"]["design"]

    eng.config = eng.config.model_copy(update={"rigor_profile": "research"})
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=RuntimeError("provider down"))})()
    stops: list[dict] = []

    def pause(**kwargs):  # noqa: ANN003
        stops.append(kwargs)
        raise RuntimeError("stopped")

    eng._pause_for_human = pause  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="stopped"):
        await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert stops[0]["kind"] == "design_audit_unknown"
    receipt = json.loads((eng.quest_root / "needs" / "receipts" / "design_audit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "unknown" and "provider down" in receipt["error"]
    await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))  # the retry fails too: no second stop
    assert len(stops) == 1


SEARCH = {
    "objective": {"quantity": "y", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "w", "low": 0.0, "high": 1.0, "kind": "continuous"}],
    "baseline": {"values": {"w": 0.0}, "source": "the topic"},
    "evaluation_budget": {"starts": 2, "per_start": 20},
}
SEARCH_DRAFT = {**DRAFT, "study_type": "find_best_design", "protocol": {"optimisation": SEARCH}}


def _with_baseline(value: object) -> dict:
    return {**SEARCH_DRAFT, "protocol": {"optimisation": {**SEARCH, "baseline": {"values": {"w": value}, "source": "the topic"}}}}


@pytest.mark.asyncio
async def test_a_repair_that_breaks_the_search_block_is_not_applied_and_is_recorded(tmp_path: Path) -> None:
    assert plan.normalize_design(SEARCH_DRAFT)[0] is not None
    broken = _with_baseline([[0.1, 0.1], [0.0, 0.0]])  # a baseline outside what the variable takes
    assert plan.normalize_design(broken)[0] is None  # the plan step would drop the whole block over it
    objection = {"check": "baseline", "where": "protocol.optimisation.baseline", "objection": "the baseline is a single number", "fix": "made it a grid"}
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [objection], "amended_design": broken})])
    design, objections = await eng._audit_design({"topic": "a topic", "iteration": 0}, json.loads(json.dumps(SEARCH_DRAFT)))
    assert design["protocol"]["optimisation"]["baseline"]["values"] == {"w": 0.0}  # the block and its baseline stay
    assert objections == [objection]  # the objection is still reported
    (entry,) = _record(eng)
    assert entry["changed_keys"] == []
    assert [n["key"] for n in entry["not_applied"]] == ["protocol.optimisation"]
    assert entry["not_applied"][0]["reason"]
    assert "did not apply it" in " ".join(eng._critique_summary())
    status, record, _problem = receipts.read(eng.quest_root, "design_audit")
    assert status == "fail" and "protocol.optimisation" in json.dumps(record)


@pytest.mark.asyncio
async def test_a_repair_that_passes_the_same_checks_is_still_applied(tmp_path: Path) -> None:
    fixed = _with_baseline(0.5)
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": fixed})])
    design, _ = await eng._audit_design({"topic": "a topic", "iteration": 0}, json.loads(json.dumps(SEARCH_DRAFT)))
    assert design["protocol"]["optimisation"]["baseline"]["values"] == {"w": 0.5}
    (entry,) = _record(eng)
    assert entry["changed_keys"] == ["protocol"] and "not_applied" not in entry


def _thresholded(thresholds: dict) -> dict:
    return {**DRAFT, "protocol": {"runs_per_setting": 300, "thresholds": thresholds}}


@pytest.mark.asyncio
async def test_a_part_the_audit_left_out_is_put_back_and_recorded(tmp_path: Path) -> None:
    left_out = {**DRAFT, "protocol": {}}
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": left_out})])
    design, _ = await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert design["protocol"] == {"runs_per_setting": 300}
    (entry,) = _record(eng)
    assert [n["key"] for n in entry["not_applied"]] == ["protocol.runs_per_setting"]


@pytest.mark.asyncio
async def test_a_repair_that_cannot_be_applied_fails_the_receipt_even_when_no_objection_was_named(tmp_path: Path) -> None:
    broken = _with_baseline([[0.1, 0.1], [0.0, 0.0]])
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [], "amended_design": broken})])
    await eng._audit_design({"topic": "a topic", "iteration": 0}, json.loads(json.dumps(SEARCH_DRAFT)))
    status, record, _problem = receipts.read(eng.quest_root, "design_audit")
    assert status == "fail" and "could not be applied" in json.dumps(record)


@pytest.mark.asyncio
async def test_a_change_to_the_top_level_is_caught_even_when_the_draft_needed_its_protocol_repaired(tmp_path: Path) -> None:
    before = _thresholded({"a": 1, "rule": "prose that is not a number"})
    assert plan.normalize_design(before)[0] is None  # the plan step repairs this protocol before it reads the design
    amended = {**before, "hypothesis": ""}
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended})])
    design, _ = await eng._audit_design({"topic": "OPC", "iteration": 0}, json.loads(json.dumps(before)))
    assert design["hypothesis"] == DRAFT["hypothesis"]
    (entry,) = _record(eng)
    assert any("hypothesis" in n["key"] for n in entry["not_applied"])


@pytest.mark.asyncio
async def test_swapping_one_unreadable_part_for_another_is_not_a_repair(tmp_path: Path) -> None:
    before = _thresholded({"a": 1, "rule": "prose one"})
    amended = _thresholded({"a": 1, "rule": "prose two"})
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended})])
    design, _ = await eng._audit_design({"topic": "OPC", "iteration": 0}, json.loads(json.dumps(before)))
    assert design["protocol"]["thresholds"]["rule"] == "prose one"
    (entry,) = _record(eng)
    assert [n["key"] for n in entry["not_applied"]] == ["protocol.thresholds"]


SCANNED = {**SEARCH, "fixed": {"a": 1, "b": 3}, "grid": {"a": [1, 2]}, "sub_studies": [{"name": "scan a", "vary": ["a"], "hold": {}}]}
SCANNED_DRAFT = {**DRAFT, "study_type": "find_best_design", "protocol": {"optimisation": SCANNED}}


def _scan_amended(entry_vary: str) -> dict:
    """The audit's change: a new baseline, and a scan entry that varies ``entry_vary`` (not a setting of the grid unless "a")."""
    opt = {**SCANNED, "baseline": {"values": {"w": 0.5}, "source": "the audit"},
           "sub_studies": [{"name": "scan a", "vary": ["a"], "hold": {}}, {"name": "scan b", "vary": [entry_vary], "hold": {"a": 1}}]}
    return {**SCANNED_DRAFT, "protocol": {"optimisation": opt}}


@pytest.mark.asyncio
async def test_one_unreadable_part_of_a_change_does_not_cost_the_rest_and_is_asked_for_again_with_the_reason(tmp_path: Path) -> None:
    amended = _scan_amended("b")
    assert plan.normalize_design(SCANNED_DRAFT)[0] is not None
    fixed_opt = {**amended["protocol"]["optimisation"], "grid": {"a": [1, 2], "b": [3, 4]},
                 "sub_studies": [{"name": "scan a", "vary": ["a"], "hold": {"b": 3}}, {"name": "scan b", "vary": ["b"], "hold": {"a": 1}}]}
    replies = [json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended}),
               json.dumps({"protocol": {"optimisation": fixed_opt}})]
    eng = _engine(tmp_path, replies)
    design, _ = await eng._audit_design({"topic": "a topic", "iteration": 0}, json.loads(json.dumps(SCANNED_DRAFT)))
    asked = eng._client.chat.call_args_list[1].args[0][-1]["content"]
    assert "varies `b`" in asked and "protocol.optimisation" in asked, "the second call carries FI's exact reason"
    assert eng._client.chat.call_count == 2, "asked once"
    opt = design["protocol"]["optimisation"]
    assert opt["baseline"]["values"] == {"w": 0.5} and [s["name"] for s in opt["sub_studies"]] == ["scan a", "scan b"]
    (entry,) = _record(eng)
    assert "not_applied" not in entry, "the corrected part is read in full: nothing is left out"


@pytest.mark.asyncio
async def test_a_part_still_unreadable_after_the_ask_is_left_out_alone_and_named(tmp_path: Path) -> None:
    amended = _scan_amended("b")
    replies = [json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended}),
               json.dumps({"protocol": {"optimisation": amended["protocol"]["optimisation"]}})]  # answers with no change
    eng = _engine(tmp_path, replies)
    design, _ = await eng._audit_design({"topic": "a topic", "iteration": 0}, json.loads(json.dumps(SCANNED_DRAFT)))
    opt = design["protocol"]["optimisation"]
    assert opt["baseline"]["values"] == {"w": 0.5}, "the readable part of the change is made"
    assert [x["name"] for x in opt["sub_studies"]] == ["scan a"], "only the entry that cannot be read is left out"
    (entry,) = _record(eng)
    (item,) = entry["not_applied"]
    assert item["key"] == "protocol.optimisation" and "sub_studies" in item["reason"] and item["applied"]
    summary = " ".join(eng._critique_summary())
    assert "applied except a part FI could not read" in summary and "did not apply it" not in summary
    assert eng._client.chat.call_count == 2


@pytest.mark.asyncio
async def test_a_change_nothing_of_which_can_be_read_is_still_not_applied_and_not_asked_again(tmp_path: Path) -> None:
    broken = _with_baseline([[0.1, 0.1], [0.0, 0.0]])
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": broken})])
    design, _ = await eng._audit_design({"topic": "a topic", "iteration": 0}, json.loads(json.dumps(SEARCH_DRAFT)))
    assert design["protocol"]["optimisation"]["baseline"]["values"] == {"w": 0.0}
    assert eng._client.chat.call_count == 1
    (entry,) = _record(eng)
    assert [n["key"] for n in entry["not_applied"]] == ["protocol.optimisation"] and "applied" not in entry["not_applied"][0]


@pytest.mark.asyncio
async def test_the_audit_answers_are_kept_and_the_plan_says_beside_the_design_what_was_not_applied(tmp_path: Path) -> None:
    amended = _scan_amended("b")
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended}),
                             json.dumps({"protocol": {"optimisation": amended["protocol"]["optimisation"]}})])
    await eng._audit_design({"topic": "a topic", "iteration": 0}, json.loads(json.dumps(SCANNED_DRAFT)))
    kept = sorted((eng.fi_dir / "plan_answers").glob("*.txt"))
    assert len(kept) == 2 and "amended_design" in kept[0].read_text(encoding="utf-8"), "the audit's reply and the correction"
    (sentence,) = eng._audit_changes_held_back()
    assert "except one part" in sentence and "`protocol.optimisation`" in sentence
    text = plan.render("a topic", {}, SCANNED_DRAFT, held_back=[sentence])
    head = text.index("## " + plan.DESIGN_HEADING)
    assert sentence in text[:head] and sentence in text[head:], "beside the design and under the summary"
    assert plan.parse(text).design is not None, "the added sentence does not disturb reading the block"
    assert "FI applied" not in plan.render("a topic", {}, SCANNED_DRAFT)
