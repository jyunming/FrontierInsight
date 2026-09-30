"""A second opinion on the plan's checks against known answers (core/oracle_review.py): one model call at plan time,
read strictly, its findings sent to the plan once together with the checks not in their kind's form, and plan.md saying
in plain words what the reviewer found and what changed. Under rigor_profile: research the reviewer is the model named
for `oracle_review`, not the planner's."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import oracle_forms as of, oracle_review as orv, plan
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

MODEL = {"summary": "a lossless two-port network", "equations": [
    {"id": "E1", "formula": "P_out = P_in", "role": "generates", "source": "derivation", "derivation": "energy balance: P_out = P_in"},
    {"id": "E2", "formula": "V = I R", "role": "generates", "source": "derivation", "derivation": "Ohm's law: V = I R"},
]}
CHECK = {"name": "power_conservation", "kind": "invariant", "check": "output power equals input power", "expected": 0,
         "tolerance": 1e-6, "case": {"n": 4}, "measure": "abs(P_out - P_in) / P_in",
         "reference": "derivation: a lossless network gives P_out = P_in, so the violation is 0"}
PROTOCOL = {"grid": {"n": [4, 8]}, "model": MODEL, "oracles": [CHECK],
            "criteria": [{"name": "conserved", "oracle": "power_conservation", "direction": "lower", "target": 1e-6,
                          "tolerance": 1e-7}]}

REVIEW = {
    "checks": [{"name": "power_conservation", "appropriate": "yes", "why": "it tests E1", "discriminating": "yes",
                "bug_it_would_catch": "a dropped loss term fails it", "well_defined": "yes", "definition_note": "",
                "better": ""},
               {"name": "not a check of this plan", "appropriate": "no", "why": "invented"}],
    "equations_tested": ["E1"], "equations_not_tested": ["e2", "E9"],
    "add": [{"name": "ohm", "kind": "special_case", "expected": 2.0, "tolerance": 1e-9, "case": {"n": 4},
             "measure": "V / I", "reference": "derivation: R = 2 gives V / I = 2"}],
    "summary": "One check, and it tests energy balance; nothing tests Ohm's law.",
}


# --- reading the answer ------------------------------------------------------------------------------------------------


def test_the_answer_is_read_strictly_and_only_about_the_plans_own_checks_and_equations() -> None:
    review = orv.parse(REVIEW, PROTOCOL)
    assert [c["name"] for c in review.checks] == ["power_conservation"]
    assert review.tested == ["E1"] and review.untested == ["E2"], "an equation the model does not list is dropped"
    found = orv.findings(review)
    assert any("no check tests equation E2" in f for f in found) and any("a check to add" in f for f in found)
    assert not any("may not test the model" in f for f in found)


@pytest.mark.parametrize("reply", [None, "text", {}, {"checks": "all fine"}, {"checks": [{"name": "unknown"}]},
                                   json.loads(_FAKE_RESPONSES["review"]) if _FAKE_RESPONSES.get("review") else {}])
def test_an_answer_that_names_none_of_the_checks_is_not_used(reply: Any) -> None:
    assert orv.parse(reply, PROTOCOL) is None


def test_a_check_found_wanting_is_named_with_the_reason() -> None:
    bad = {"checks": [{"name": "power_conservation", "appropriate": "no", "why": "P_out is set equal to P_in by the code",
                       "discriminating": False, "bug_it_would_catch": "none: it holds whatever the code does",
                       "well_defined": "no", "definition_note": "a ratio and a violation are mixed",
                       "better": "compare with an independent circuit solver"}]}
    found = orv.findings(orv.parse(bad, PROTOCOL))
    assert len(found) == 4 and "may not test the model: P_out is set equal" in found[0]
    assert "would not fail on a plausible bug" in found[1] and "not well defined" in found[2] and "better" in found[3]
    assert "leave it as it is if not" in orv.request(found)


def test_the_plan_says_which_model_read_the_checks() -> None:
    review = orv.parse(REVIEW, PROTOCOL)
    other = orv.plan_lines(review, reviewer="model-b", planner="model-a", same_model=False, research=True)
    assert "second model (model-b), not the one that wrote the plan (model-a)" in other[1]
    same = orv.plan_lines(review, reviewer="model-a", planner="model-a", same_model=True, research=True)
    assert "a second look, not a second opinion" in same[1] and "node_models: oracle_review" in same[1]
    assert "node_models" not in orv.plan_lines(review, reviewer="m", planner="m", same_model=True, research=False)[1]
    unusable = orv.plan_lines(None, reviewer="m", planner="m", same_model=True, research=False, error="not JSON")
    assert "could not be used (not JSON)" in unusable[1]


def test_the_prompt_is_a_template_with_its_parts_and_a_first_line_no_other_node_uses() -> None:
    import string
    text = (Path(__file__).resolve().parents[1] / "agents" / "oracle_review.md").read_text(encoding="utf-8")
    model, oracles = orv.prompt_parts(PROTOCOL)
    filled = string.Template(text).substitute(topic="t", model=model, oracles=oracles)
    assert "power_conservation" in filled and "E2" in filled
    assert _classify(filled) == "(unknown)", "the fake models of other tests must not mistake it for another step"


# --- the engine --------------------------------------------------------------------------------------------------------


def _config(tmp_path: Path, *, research: bool = False, node_models: dict[str, str] | None = None) -> Config:
    cfg = Config(
        topic="power through a lossless network", title="review", provider=ProviderConfig(
            name="openai", model="planner-model", node_models=node_models or {}),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(plan="off", papers=False),
    )
    # The profile alone (its plan stop is another test's): what it changes here is which model must read the checks.
    return cfg.model_copy(update={"rigor_profile": "research"}) if research else cfg


class _Model:
    def __init__(self, protocol: dict[str, Any], review: Any, revise: Any = None) -> None:
        self.protocol, self.review, self.revise = protocol, review, revise
        self.reviews: list[dict[str, Any]] = []
        self.revisions: list[str] = []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if prompt.lstrip().startswith("# Second Opinion on the Checks"):
            self.reviews.append({"prompt": prompt, "model": kw.get("model")})
            return self.review if isinstance(self.review, str) else json.dumps(self.review)
        if "You are revising the plan" in prompt:
            self.revisions.append(prompt)
            current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
            if self.revise is None:
                return current
            return plan.edit_design_block(current, self.revise) or current
        if _classify(prompt) == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = self.protocol
            body["plan"] = {"in_short": "x", "literature": [], "gap": "g", "success_criteria": ["s"], "risks": ["r"],
                            "out_of_scope": ["o"]}
            return json.dumps(body)
        return _fake_response_for(prompt)

    async def aclose(self) -> None:
        return None


def _section(engine: Engine) -> str:
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    return text.split(f"## {of.HEADING}", 1)[1].split("\n## ", 1)[0]


def _add_ohm(block: dict[str, Any]) -> dict[str, Any]:
    oracles = [*block["protocol"]["oracles"], REVIEW["add"][0]]
    return {**block, "protocol": {**block["protocol"], "oracles": oracles}}


@pytest.mark.asyncio
async def test_the_review_is_asked_once_its_findings_go_back_once_and_plan_md_says_what_changed(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(PROTOCOL, REVIEW, revise=_add_ohm)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.reviews) == 1 and len(model.revisions) == 1
    assert "power_conservation" in model.reviews[0]["prompt"] and "E2" in model.reviews[0]["prompt"]
    assert "No check tests equation E2" in model.revisions[0]
    names = [o["name"] for o in plan.load_design(engine.quest_root)[0]["protocol"]["oracles"]]
    assert names == ["power_conservation", "ohm"]
    section = _section(engine)
    assert "nothing tests Ohm's law" in section and "**power_conservation**: tests the model" in section
    assert "the check 'ohm' was added" in section
    cost = [json.loads(line) for line in (engine.fi_dir / "cost.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()] if (engine.fi_dir / "cost.jsonl").is_file() else []
    assert not cost or any(row.get("node") == orv.NODE for row in cost)
    # Once per quest: another pass through the plan step asks nothing.
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.reviews) == 1 and len(model.revisions) == 1


@pytest.mark.asyncio
async def test_forms_and_the_review_share_one_revision(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    bare = {**CHECK, "measure": "ratio", "expected": 1.0}
    model = _Model({**PROTOCOL, "oracles": [bare]}, REVIEW)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.revisions) == 1, "one request to the plan, not one per source of findings"
    assert "does not say whether that number is the quantity" in model.revisions[0]
    assert "No check tests equation E2" in model.revisions[0]


@pytest.mark.asyncio
async def test_a_review_that_finds_nothing_changes_nothing_and_an_unreadable_one_is_said(tmp_path: Path) -> None:
    fine = {**REVIEW, "equations_not_tested": [], "add": []}
    engine = Engine(_config(tmp_path / "a"))
    model = _Model(PROTOCOL, fine)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.reviews) == 1 and model.revisions == []
    assert "**power_conservation**: tests the model" in _section(engine)

    engine = Engine(_config(tmp_path / "b"))
    model = _Model(PROTOCOL, "I think these checks look fine.")
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert model.revisions == [] and "Its answer could not be used" in _section(engine)
    assert [r["by"] for r in plan.history(engine.quest_root)] == ["model", "engine"]


@pytest.mark.asyncio
async def test_under_research_the_reviewer_is_the_model_named_for_it(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path / "a", research=True, node_models={"oracle_review": "reviewer-model"}))
    model = _Model(PROTOCOL, {**REVIEW, "equations_not_tested": [], "add": []})
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert model.reviews[0]["model"] == "reviewer-model"
    assert "second model (reviewer-model), not the one that wrote the plan (planner-model)" in _section(engine)

    engine = Engine(_config(tmp_path / "b", research=True))
    model = _Model(PROTOCOL, {**REVIEW, "equations_not_tested": [], "add": []})
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    section = _section(engine)
    assert "the model that wrote the plan (planner-model)" in section and "node_models: oracle_review" in section
    assert "oracle_review" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_no_review_without_checks_or_with_the_checks_off(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path / "a"))
    model = _Model({"grid": {"n": [4]}}, REVIEW)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert model.reviews == []
