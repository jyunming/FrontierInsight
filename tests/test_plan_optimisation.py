"""A study that looks for the best design: the plan's `study_type`, its `optimisation` block, the section of plan.md,
the question asked when the topic is ambiguous, and the stop before anything runs (the search itself is not built yet)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import frozen_protocol, optimisation_plan as op, plan
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine
from core.protocol_check import plan_notes

BLOCK: dict[str, Any] = {
    "objective": {"quantity": "max_base_temperature", "direction": "minimise", "unit": "K",
                  "meaning": "the highest temperature of the heat-sink base"},
    "design_variables": [
        {"name": "fin_spacing", "low": 1.0, "high": 6.0, "unit": "mm", "kind": "continuous"},
        {"name": "fin_thickness", "low": 0.4, "high": 2.0, "unit": "mm", "kind": "continuous"},
        {"name": "fin_count", "low": 8, "high": 30, "kind": "integer"},
    ],
    "fixed": {"heat_load_W": 40, "air_speed_m_s": 2.0},
    "constraints": [{"quantity": "mass_g", "limit": "<= 120", "unit": "g"},
                    {"quantity": "pressure_drop", "limit": "<= 25", "unit": "Pa"}],
    "baseline": {"values": {"fin_spacing": 3.0, "fin_thickness": 1.0, "fin_count": 15}, "source": "[3] Table 2"},
    "numerical_settings": {"mesh_size_mm": {"search": 0.5, "check": [0.25, 0.125]}},
    "evaluation_budget": {"starts": 4, "per_start": 60, "seconds_per_evaluation": 2.0},
    "search_method": "bounded_local",
}

HEAT_SINK: dict[str, Any] = {
    "hypothesis": "a fin spacing and thickness exist that lower the base temperature below the commercial design's",
    "study_type": "find_best_design",
    "variables": {"independent": ["fin_spacing", "fin_thickness", "fin_count"], "dependent": ["max_base_temperature"],
                  "controls": ["heat_load_W"]},
    "method": "search the fin geometry in a finite-volume conduction-convection model",
    "expected_outcome": "a design a few kelvin cooler than the baseline",
    "figures_planned": ["search_progress.png"],
    "dependencies": ["numpy"],
    "protocol": {
        "optimisation": BLOCK,
        "oracles": [{"name": "baseline_energy_balance", "kind": "invariant", "check": "heat in = heat out at the baseline",
                     "expected": 0, "tolerance": 0.01, "reference": "derivation: steady state, so the net flux is 0"}],
    },
}

EXTRA = {"in_short": "We look for the coolest fin geometry.", "literature": [], "gap": "no search in this range"}


def _design(**block_changes: Any) -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    design["protocol"]["optimisation"].update(block_changes)
    return design


# --- the block ------------------------------------------------------------------------------------------------------


def test_a_full_block_is_accepted_and_reading_it_twice_changes_nothing() -> None:
    fixed, why = plan.normalize_design(copy.deepcopy(HEAT_SINK))
    assert why is None
    again, _ = plan.normalize_design(copy.deepcopy(fixed))
    assert again == fixed
    assert fixed["study_type"] == "find_best_design"
    assert fixed["protocol"]["optimisation"]["constraints"][0]["limit"] == "<= 120"


def test_the_plan_file_round_trips_the_block_and_the_study_type() -> None:
    text = plan.render("heat sink", EXTRA, plan.normalize_design(copy.deepcopy(HEAT_SINK))[0])
    parsed = plan.parse(text)
    assert parsed.error is None
    assert parsed.design["study_type"] == "find_best_design"
    assert parsed.design["protocol"]["optimisation"]["evaluation_budget"]["per_start"] == 60


@pytest.mark.parametrize("change, words", [
    ({"baseline": {"values": {"fin_spacing": 9.0, "fin_thickness": 1.0, "fin_count": 15}}}, "outside the range"),
    ({"baseline": {"values": {"fin_spacing": 3.0, "fin_thickness": 1.0}}}, "no value for `fin_count`"),
    ({"design_variables": [{"name": "fin_spacing", "low": 6.0, "high": 1.0}]}, "must be below `high`"),
    ({"constraints": [{"quantity": "mass_g", "limit": "about 120"}]}, "cannot be read"),
    ({"numerical_settings": {"mesh_size_mm": {"search": 0.5, "check": [0.5]}}}, "is not finer than 0.5"),
    ({"numerical_settings": {"mesh_size_mm": {"search": 0.5, "check": [0.48]}}}, "at least 1.1 times"),
    ({"numerical_settings": {"elements": {"search": 100, "check": [50], "finer": "larger"}}}, "is not finer than 100"),
    ({"evaluation_budget": {"starts": 0, "per_start": 60}}, "whole number of at least 1"),
    ({"search_method": "simulated_annealing"}, "scipy:<function>"),
    ({"objective": {"quantity": "max_base_temperature", "direction": "lower"}}, "`minimise` or `maximise`"),
    ({"fixed": {"fin_count": 12}}, "also a design variable"),
    ({"grid": {"fin_spacing": [0.5, 2.0]}}, "inside that variable's range"),
])
def test_a_block_that_cannot_drive_a_search_says_why(change: dict[str, Any], words: str) -> None:
    design = _design(**change)
    if "design_variables" in change:
        design["protocol"]["optimisation"]["baseline"] = {"values": {"fin_spacing": 3.0}}
    fixed, why = plan.normalize_design(design)
    assert fixed is None
    assert words in why and "`protocol.optimisation" in why


def test_a_library_method_is_accepted_by_name() -> None:
    for method in ("scipy:differential_evolution", "optuna:tpe", "global_then_local", "exhaustive"):
        assert plan.normalize_design(_design(search_method=method))[1] is None, method


def test_a_sweep_and_a_search_cannot_both_be_the_protocol_and_measure_cannot_carry_a_block() -> None:
    both = _design()
    both["protocol"]["grid"] = {"fin_spacing": [2.0, 3.0]}
    assert "optimisation.grid" in plan.normalize_design(both)[1]
    contradicted = _design()
    contradicted["study_type"] = "measure"
    assert "`study_type` is measure" in plan.normalize_design(contradicted)[1]
    assert "`study_type` must be" in plan.normalize_design({**HEAT_SINK, "study_type": "explore"})[1]


def test_find_best_design_without_its_block_is_readable_and_never_counts_as_a_measurement() -> None:
    sweep = {**copy.deepcopy(HEAT_SINK), "protocol": {"grid": {"fin_spacing": [2.0, 3.0, 4.0]}}}
    fixed, why = plan.normalize_design(sweep)
    assert why is None
    assert op.study_type_of(fixed) == "find_best_design"
    assert "no `optimisation` block" in op.missing_parts(fixed)[0]
    # A block with no study_type is a search too: the safe reading.
    assert op.study_type_of({"hypothesis": "h", "protocol": {"optimisation": BLOCK}}) == "find_best_design"


# --- repairing a model's draft --------------------------------------------------------------------------------------


def test_a_draft_keeps_the_usable_parts_and_says_what_it_left_out() -> None:
    draft = copy.deepcopy(HEAT_SINK["protocol"])
    draft["optimisation"]["constraints"].append({"quantity": "stress", "limit": "small"})
    draft["optimisation"]["numerical_settings"] = {"mesh_size_mm": {"search": 0.5, "check": [0.5]}}
    fixed, notes = plan.repair_protocol(draft)
    block = fixed["optimisation"]
    assert [c["quantity"] for c in block["constraints"]] == ["mass_g", "pressure_drop"]
    assert "check" not in block["numerical_settings"]["mesh_size_mm"]
    text = " ".join(notes)
    assert "stress" in text and "FI's fixed rule" in text


def test_a_top_level_grid_beside_the_block_becomes_its_coarse_scan() -> None:
    draft = copy.deepcopy(HEAT_SINK["protocol"])
    draft["grid"] = {"fin_spacing": [2.0, 3.0, 4.0]}
    fixed, notes = plan.repair_protocol(draft)
    assert "grid" not in fixed
    assert fixed["optimisation"]["grid"] == {"fin_spacing": [2.0, 3.0, 4.0]}
    assert "coarse scan" in " ".join(notes)


def test_a_variable_that_cannot_be_read_is_left_out_with_its_baseline_value() -> None:
    draft = copy.deepcopy(HEAT_SINK["protocol"])
    draft["optimisation"]["design_variables"][2] = {"name": "fin_count", "low": 30, "high": 8}
    fixed, notes = plan.repair_protocol(draft)
    block = fixed["optimisation"]
    assert [v["name"] for v in block["design_variables"]] == ["fin_spacing", "fin_thickness"]
    assert "fin_count" not in block["baseline"]["values"]
    assert "fin_count" in " ".join(notes)


@pytest.mark.asyncio
async def test_a_block_with_no_objective_is_left_out_and_the_plan_still_says_it_was_a_search(tmp_path: Path) -> None:
    broken = copy.deepcopy(HEAT_SINK)
    del broken["study_type"]
    broken["protocol"]["optimisation"]["objective"] = "cooler"
    eng = _engine(tmp_path, [json.dumps({**broken, "plan": EXTRA}), json.dumps({"objections_addressed": []})])
    await eng._node_plan({"topic": "heat sink", "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    design = plan.parse(text).design
    assert design["study_type"] == "find_best_design" and "optimisation" not in design["protocol"]
    assert "not run as a plain sweep" in text


# --- what FI does with what the plan leaves out ----------------------------------------------------------------------


def test_finer_check_settings_come_from_the_plan_or_from_the_fixed_rule() -> None:
    assert op.check_levels({"search": 0.5, "check": [0.3, 0.2]}) == ([0.3, 0.2], "plan")
    assert op.check_levels({"search": 0.5}) == ([0.25, 0.125], "rule")
    assert op.check_levels({"search": 40, "finer": "larger"}) == ([80, 160], "rule")


def test_the_method_and_the_improvement_rule_have_defaults_that_are_said() -> None:
    block = op.normalize({k: v for k, v in BLOCK.items() if k != "search_method"})[0]
    assert op.effective_method(block) == ("bounded_local", "rule")
    assert op.effective_method({**block, "grid": {"fin_spacing": [2.0, 4.0]}}) == ("global_then_local", "rule")
    small = {**block, "design_variables": [{"name": "fin_count", "low": 8, "high": 12, "kind": "integer"}]}
    assert op.effective_method(small) == ("exhaustive", "rule")
    assert op.improvement_rule(block) == ("better than the baseline by more than the numerical error of the two designs", "rule")
    assert op.improvement_rule({**block, "improvement_tolerance": 0.5})[0].endswith("0.5 K")
    # A threshold and a target are optional: neither is needed for the block to be read.
    assert op.normalize({**block, "target": 60})[1] is None


# --- plan.md ---------------------------------------------------------------------------------------------------------


def test_plan_md_says_what_is_optimised_and_the_budget_before_anything_runs() -> None:
    text = plan.render("heat sink", EXTRA, plan.normalize_design(copy.deepcopy(HEAT_SINK))[0])
    assert f"## {op.HEADING}" in text
    assert "make max_base_temperature (the highest temperature of the heat-sink base) as low as possible" in text
    assert "- fin_count: 8 to 30 (whole numbers)" in text
    assert "mass_g ≤ 120 g" in text
    assert "4 starting points × 60 = 240 evaluations" in text
    # (3 best + the baseline) × 2 finer levels + 2 × 2 continuous nudges = 12
    assert "= 12 evaluations" in text
    assert "about 8 min" in text and "not measured yet" in text
    assert "numerical error of the two designs" in text
    assert "cannot yet run the search" in text


def test_plan_md_names_the_fixed_rule_when_the_plan_gives_no_finer_values() -> None:
    design = _design(numerical_settings={"mesh_size_mm": {"search": 0.5}}, evaluation_budget=None)
    del design["protocol"]["optimisation"]["evaluation_budget"]
    text = plan.render("heat sink", EXTRA, plan.normalize_design(design)[0])
    assert "the check uses 0.25, then 0.125 (FI's fixed rule" in text
    assert "no evaluation budget is written" in text and "time per evaluation: not measured yet" in text


def test_the_topics_numbers_are_found_in_the_block() -> None:
    notes = plan_notes("the fins must keep the mass under 120 g", plan.normalize_design(copy.deepcopy(HEAT_SINK))[0]["protocol"])
    assert notes == []


# --- the frozen protocol covers the block ----------------------------------------------------------------------------


def test_the_freeze_covers_the_block_and_a_bigger_budget_shows_as_a_change(tmp_path: Path) -> None:
    protocol = plan.normalize_design(copy.deepcopy(HEAT_SINK))[0]["protocol"]
    record = frozen_protocol.freeze(tmp_path, protocol, approved_by="test", source="plan.md")
    assert frozen_protocol.load(tmp_path)["protocol"]["optimisation"] == protocol["optimisation"]
    bigger = copy.deepcopy(protocol)
    bigger["optimisation"]["evaluation_budget"]["per_start"] = 90
    assert frozen_protocol.sha256(bigger) != record["sha256"]
    assert frozen_protocol.diff(protocol, bigger) == ["optimisation.evaluation_budget.per_start: 60 -> 90"]


# --- the question when the topic is ambiguous ------------------------------------------------------------------------


@pytest.mark.parametrize("topic, kind", [
    ("Find the fin spacing and thickness that minimise the base temperature of a heat sink", "find_best_design"),
    ("Choose the lightest truss cross-section that keeps the stress under 250 MPa", "find_best_design"),
    ("How does fin spacing change the base temperature of a heat sink?", "measure"),
    ("Effect of controller gain on overshoot in a second-order system", "measure"),
    ("Best controller gain for low overshoot", "ambiguous"),
    ("How does the optimal fin spacing depend on air speed?", "ambiguous"),
    ("找出讓散熱片溫度最低的鰭片間距", "find_best_design"),
    ("控制器增益如何影響超調量，最佳值在哪裡", "ambiguous"),
])
def test_the_topic_decides_whether_the_person_is_asked(topic: str, kind: str) -> None:
    assert op.classify_topic(topic) == kind
    questions: dict[str, Any] = {}
    assert op.add_study_type_question(questions, topic) is (kind == "ambiguous")
    if kind == "ambiguous":
        assert questions["study_type"]["default"] == op.LET_THE_PLAN_DECIDE
        assert "Answer 1 or 2" in questions["study_type"]["question"]


@pytest.mark.parametrize("answer, kind", [
    ("2", "find_best_design"), ("1", "measure"), (op.LET_THE_PLAN_DECIDE, None), ("", None),
    ("find the best design", "find_best_design"), ("measure", "measure"), ("I want the best one", "find_best_design"),
])
def test_an_answer_is_read_as_a_number_or_in_words(answer: str, kind: str | None) -> None:
    assert op.resolve_answer(answer) == kind


# --- the engine ------------------------------------------------------------------------------------------------------


def _engine(tmp_path: Path, replies: list[str], clarify: str = "off", **pauses: Any) -> Engine:
    eng = Engine(Config(
        topic="heat sink fin geometry", title="fins", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode=clarify),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"), pauses=PausesConfig(**pauses),
    ))
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=replies)})()
    return eng


class Paused(Exception):
    """Stands in for LangGraph's GraphInterrupt."""


def _stop_at_pause(eng: Engine) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        seen.append(kwargs)
        raise Paused(kwargs["kind"])

    eng._pause_for_human = fake  # type: ignore[method-assign]
    return seen


_NO_OBJECTIONS = json.dumps({"objections_addressed": []})


@pytest.mark.asyncio
async def test_a_search_plan_is_written_and_the_quest_stops_before_anything_runs_every_time(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({**HEAT_SINK, "plan": EXTRA}), _NO_OBJECTIONS])
    await eng._node_plan({"topic": "heat sink", "iteration": 0})
    prompt = eng._client.chat.await_args_list[0].args[0][0]["content"]
    assert "A measurement, or a search for the best design?" in prompt
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert plan.parse(text).design["protocol"]["optimisation"]["objective"]["quantity"] == "max_base_temperature"

    for _ in range(2):  # no once-only marker: every resume checks the plan again
        seen = _stop_at_pause(eng)
        with pytest.raises(Paused):
            await eng._node_design({"topic": "heat sink", "iteration": 0})
        assert seen[0]["headline"] == "the search for the best design is not available yet"
        steps = " ".join(seen[0]["steps"])
        assert "Nothing was run" in steps and "study_type: measure" in steps and "--revise-plan" in steps
    assert not list(eng.quest_root.glob("**/*.py"))


@pytest.mark.asyncio
async def test_a_plan_changed_to_a_measurement_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [_NO_OBJECTIONS])
    measure = {**{k: v for k, v in HEAT_SINK.items() if k != "protocol"}, "study_type": "measure",
               "protocol": {"grid": {"fin_spacing": [2.0, 3.0, 4.0]}, "oracles": HEAT_SINK["protocol"]["oracles"]}}
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("heat sink", EXTRA, measure), encoding="utf-8")
    seen = _stop_at_pause(eng)
    patch = await eng._node_design({"topic": "heat sink", "iteration": 0})
    assert seen == [] and patch["design"]["study_type"] == "measure"


@pytest.mark.asyncio
async def test_the_persons_answer_reaches_the_plan_and_is_kept(tmp_path: Path) -> None:
    measure_draft = {**{k: v for k, v in HEAT_SINK.items() if k not in ("protocol", "study_type")},
                     "protocol": {"grid": {"fin_spacing": [2.0, 3.0]}}}
    eng = _engine(tmp_path, [json.dumps({**measure_draft, "plan": EXTRA}), _NO_OBJECTIONS])
    state = {"topic": "best fin spacing", "iteration": 0, "clarify_answers": {"study_type": "2"}}
    await eng._node_plan(state)
    prompt = eng._client.chat.await_args_list[0].args[0][0]["content"]
    assert "The person answered that this study should find the best design" in prompt
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert plan.parse(text).design["study_type"] == "find_best_design"
    assert "You answered that this study should find the best design" in text
    seen = _stop_at_pause(eng)
    with pytest.raises(Paused):
        await eng._node_design(state)
    assert "no `optimisation` block" in " ".join(seen[0]["steps"])


@pytest.mark.asyncio
async def test_an_ambiguous_topic_adds_one_question_to_the_clarify_step(tmp_path: Path) -> None:
    reply = json.dumps({"success_metric": {"question": "What counts?", "default": "overshoot"}})
    eng = _engine(tmp_path, [reply], clarify="auto")
    patch = await eng._node_clarify_questions({"topic": "Best controller gain for low overshoot"})
    assert patch["clarify_questions"]["study_type"]["default"] == op.LET_THE_PLAN_DECIDE
    assert op.resolve_answer(patch["clarify_answers"]["study_type"]) is None  # nobody answered: the plan decides

    clear = _engine(tmp_path / "clear", [reply], clarify="auto")
    patch = await clear._node_clarify_questions({"topic": "Effect of controller gain on overshoot"})
    assert "study_type" not in patch["clarify_questions"]


@pytest.mark.asyncio
async def test_a_whole_run_with_a_search_plan_stops_at_the_design_and_writes_no_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the real graph: the plan is written, the quest stops with a plain card, and nothing is implemented or run
    (the plan is never run as a plain sweep)."""
    from tests.test_engine_smoke import _classify, _fake_response_for

    cfg = Config(
        topic="Find the fin spacing and thickness that minimise the base temperature of a heat sink", title="fins",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design":
            return json.dumps({**HEAT_SINK, "plan": EXTRA})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    await engine.run()

    root = engine.quest_root
    pause = json.loads((root / ".fi" / "pause.json").read_text(encoding="utf-8"))
    assert pause["headline"] == "the search for the best design is not available yet"
    assert "Nothing was run" in (root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert f"## {op.HEADING}" in (root / "plan.md").read_text(encoding="utf-8")
    assert not list(root.glob("**/simulate.py")) and not list(root.glob("**/experiment.py"))
    assert "[implement]" not in (root / ".fi" / "run.log").read_text(encoding="utf-8")
