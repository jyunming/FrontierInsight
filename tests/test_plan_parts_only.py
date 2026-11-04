"""Every repair of the plan asks its model for the faulty parts only, never for the whole file.

A live study that was too long to run asked the plan's model to rewrite the whole plan (about 55,000 tokens); the YAML of
the long reply broke at one line and the quest stopped. Now each of FI's repairs (the study's size, a search whose
objective did not move, the checks' sources, a missing check, a request about the checks) shows the plan read-only and
asks for ONE small JSON object of the parts that repair concerns; FI merges those parts into the plan, writes it and
shows again the sections it renders from the block. A reply that names any other part is refused with FI's reason and
asked once more. Fake model only; neutral toy topics.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan, plan_scope
from core.engine import Engine, _fit_request
from tests.plan_parts_helpers import parts_of
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, _engine

TOPIC = "Find the fin spacing that minimises the base temperature"
ENTIRE = "Return the **entire** file"


def _design(**protocol: Any) -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    design["protocol"] = {**design["protocol"], "thresholds": {"max_error": 0.05},
                          "oracles": [
                              {"name": "energy_balance", "kind": "invariant", "check": "heat in = heat out", "expected": 0,
                               "tolerance": 0.01, "reference": "derivation: steady state, so the net flux is 0"},
                              {"name": "no_gain", "kind": "invariant", "check": "no fin adds heat", "expected": 0,
                               "tolerance": 0.02, "reference": ""}], **protocol}
    return design


class Scripted:
    """A plan model that answers each request with the next of ``replies`` and keeps what it was asked."""

    def __init__(self, eng: Engine, replies: list[Any]) -> None:
        self.prompts: list[str] = []
        self.replies = list(replies)
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.prompts.append(messages[-1]["content"])
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def _planned(tmp_path: Path, design: dict[str, Any] | None = None) -> tuple[Engine, Path, str]:
    eng = _engine(tmp_path, [])
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = plan.render(TOPIC, EXTRA, plan.normalize_design(design or _design())[0])
    path.write_text(text, encoding="utf-8")
    return eng, path, text


def _only_these_changed(before: str, after: str, *parts: str, prose: bool = True) -> None:
    """Every protocol part of the design block but ``parts``, and every byte of the prose outside the sections FI renders
    from the block, is as it was."""
    old, new = plan.raw_design_block(before) or {}, plan.raw_design_block(after) or {}
    assert {k: v for k, v in old.items() if k != "protocol"} == {k: v for k, v in new.items() if k != "protocol"}
    left = {k for k in {*old["protocol"], *new["protocol"]} if old["protocol"].get(k) != new["protocol"].get(k)}
    assert left and left <= set(parts), left
    if not prose:  # the section FI shows from the block appears with the checks added to a plan that had none
        def sections(text: str) -> dict[str, str]:
            return {p.split("\n", 1)[0]: p for p in text.split("\n## ")[1:]}

        was, now = sections(before), sections(after)
        for title, body in was.items():
            if title not in (plan.MODEL_HEADING, plan.DESIGN_HEADING):
                assert now.get(title) == body, title
        return
    block = plan.design_block(before)[2]
    again = plan.replace_design_block(after, block)
    fresh = plan.refresh_optimisation_section(plan.refresh_model_section(before))
    assert plan.refresh_optimisation_section(plan.refresh_model_section(again)) == fresh, "the prose is as it was"


def _asked_for_parts(prompt: str, *parts: str) -> None:
    assert ENTIRE not in prompt and "You are revising the plan" not in prompt
    assert "# What is asked" in prompt and "Change only these parts of the plan" in prompt
    asked = prompt.split("Change only these parts of the plan", 1)[1].split("\n", 1)[0]
    for part in parts:
        assert f"`{part}`" in asked, (part, asked)


# --- the scope itself ----------------------------------------------------------------------------------------------


def test_a_scope_merges_only_what_it_names_into_the_plans_own() -> None:
    scope = plan_scope.PartScope(
        parts=("criteria",), inside={"optimisation": ("evaluation_budget",)},
        entries={"oracles": plan_scope.EntryScope("name", frozenset({"reference"})),
                 "simplifications": plan_scope.EntryScope("id", frozenset(), True)})
    old = {"optimisation": {"objective": {"x": 1}, "evaluation_budget": {"starts": 4}},
           "oracles": [{"name": "a", "expected": 0, "reference": ""}, {"name": "b", "expected": 1}],
           "simplifications": [{"id": "X1", "because": "b"}]}
    answer = {"protocol": {"optimisation": {"evaluation_budget": {"starts": 2}},
                           "oracles": [{"name": "A", "reference": "[1]"}],
                           "simplifications": [{"id": "X1", "because": "b"}, {"id": "X2", "because": "new"}]}}
    out, problems = plan_scope.translate(scope, old, answer)
    assert problems == []
    assert out["protocol"]["optimisation"] == {"objective": {"x": 1}, "evaluation_budget": {"starts": 2}}
    assert out["protocol"]["oracles"] == [{"name": "a", "expected": 0, "reference": "[1]"}, {"name": "b", "expected": 1}]
    assert [s["id"] for s in out["protocol"]["simplifications"]] == ["X1", "X2"]


@pytest.mark.parametrize("answer, words", [
    ({"protocol": {"thresholds": {"max_error": 1}}}, "`protocol.thresholds`"),
    ({"hypothesis": "another claim"}, "`hypothesis`"),
    ({"protocol": {"optimisation": {"constraints": []}}}, "`constraints`"),
    ({"protocol": {"oracles": [{"name": "a", "expected": 5}]}}, "changes `expected`"),
    ({"protocol": {"oracles": [{"name": "new", "reference": "[1]"}]}}, "does not have"),
    ({"protocol": {"simplifications": [{"id": "X1", "because": "changed"}]}}, "changes `because`"),
])
def test_a_reply_that_names_more_than_the_scope_is_refused_with_the_reason(answer: dict[str, Any], words: str) -> None:
    scope = plan_scope.PartScope(
        inside={"optimisation": ("evaluation_budget",)},
        entries={"oracles": plan_scope.EntryScope("name", frozenset({"reference"})),
                 "simplifications": plan_scope.EntryScope("id", frozenset(), True)})
    old = {"optimisation": {"evaluation_budget": {"starts": 4}}, "oracles": [{"name": "a", "expected": 0}],
           "simplifications": [{"id": "X1", "because": "b"}]}
    _out, problems = plan_scope.translate(scope, old, answer)
    assert problems and words in " ".join(problems), problems


# --- the size of a study that does not fit the time (the evidence gate) -----------------------------------------------

FIT = {"seconds_per_setting": 2.0, "at_least": False, "settings": 9000, "settings_that_fit": 400, "limit_s": 3600.0,
       "search": True}
SMALLER = {"starts": 2, "per_start": 20, "seconds_per_evaluation": 2.0}
FIT_STATE = {"topic": TOPIC, "iteration": 0, "result_json": {}, "exec_result": {
    "returncode": 1, "duration_s": 1.0, "timed_out": False, "stdout_tail": "", "stderr_tail": "", "failed_script": None,
    "too_long": "the study's 9,000 settings would take about 5 h, more than the 1 h allowed", "too_long_fit": FIT}}


@pytest.mark.asyncio
async def test_the_sizing_fit_asks_for_the_parts_that_set_the_size_and_changes_only_those(tmp_path: Path) -> None:
    eng, path, before = _planned(tmp_path)
    model = Scripted(eng, [{"protocol": {"optimisation": {"evaluation_budget": SMALLER}}}])
    patch = await eng._node_evidence_gate(dict(FIT_STATE))
    assert len(model.prompts) == 1
    _asked_for_parts(model.prompts[0], "optimisation")
    assert patch["evidence_assessment"]["route"] == "redesign", patch["evidence_assessment"]
    assert "about 400 settings fit" in model.prompts[0] and "`thresholds`" not in model.prompts[0].split("Change only")[1]
    after = path.read_text(encoding="utf-8")
    assert plan.raw_design_block(after)["protocol"]["optimisation"]["evaluation_budget"] == SMALLER
    _only_these_changed(before, after, "optimisation")
    assert (eng.fi_dir / "plan_answers" / "001.txt").is_file(), "the raw reply is kept"
    assert plan.history(eng.quest_root)[-1]["by"] == "engine"


@pytest.mark.asyncio
async def test_a_sizing_reply_that_names_a_part_outside_the_size_is_refused_and_asked_once_more(tmp_path: Path) -> None:
    eng, path, before = _planned(tmp_path)
    wide = {"protocol": {"optimisation": {"evaluation_budget": SMALLER}, "thresholds": {"max_error": 9.0}}}
    model = Scripted(eng, [wide, {"protocol": {"optimisation": {"evaluation_budget": SMALLER}}}])
    patch = await eng._node_evidence_gate(dict(FIT_STATE))
    assert patch["evidence_assessment"]["route"] == "redesign"
    assert len(model.prompts) == 2
    assert "Your last answer was not used" in model.prompts[1] and "protocol.thresholds" in model.prompts[1]
    after = path.read_text(encoding="utf-8")
    assert plan.raw_design_block(after)["protocol"]["thresholds"] == {"max_error": 0.05}, "the threshold is the plan's own"
    _only_these_changed(before, after, "optimisation")
    assert len(list((eng.fi_dir / "plan_answers").glob("*.txt"))) == 2, "both raw replies are kept"


@pytest.mark.asyncio
async def test_a_sizing_reply_refused_twice_leaves_plan_md_untouched_and_the_stop_stands(tmp_path: Path) -> None:
    eng, path, before = _planned(tmp_path)
    wide = {"protocol": {"thresholds": {"max_error": 9.0}}}
    Scripted(eng, [wide, wide])
    patch = await eng._node_evidence_gate(dict(FIT_STATE))
    assert patch["evidence_assessment"]["route"] == "stuck"
    assert path.read_text(encoding="utf-8") == before


def test_the_parts_a_sizing_fit_may_change_come_from_what_the_request_asks_and_the_plans_own_keys() -> None:
    from core.engine import _fit_scope

    sweep = {"grid": {"a": [1, 2]}, "sub_studies": [], "runs_per_setting": 3, "oracles": [], "thresholds": {}}
    assert set(_fit_scope({"search": False}, sweep).names()) == {"grid", "sub_studies", "runs_per_setting", "simplifications"}
    searched = _fit_scope(FIT, {"optimisation": {"evaluation_budget": {}, "grid": {"a": [1]}, "constraints": []}})
    assert searched.inside == {"optimisation": ("evaluation_budget", "grid")}
    assert "evaluation_budget" in _fit_request("too long", FIT)


# --- a search whose objective did not move ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_flat_objective_repair_asks_for_how_the_design_is_described_and_nothing_else(tmp_path: Path) -> None:
    from tests.test_flat_objective_redesign import FACT, _flat_state, _planned as planned_flat

    eng, _model = await planned_flat(tmp_path)
    path = plan.plan_path(eng.quest_root)
    before = path.read_text(encoding="utf-8")
    block = copy.deepcopy(plan.raw_design_block(before)["protocol"]["optimisation"])
    block["design_variables"][0]["high"] = 9.5
    model = Scripted(eng, [{"protocol": {"optimisation": block}}])
    from core.engine import _flat_request

    done = await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    assert done["version"] >= 2
    _asked_for_parts(model.prompts[0], "optimisation", "grid", "in_each_run", "sub_studies", "numerical_settings")
    after = path.read_text(encoding="utf-8")
    assert plan.raw_design_block(after)["protocol"]["optimisation"]["design_variables"][0]["high"] == 9.5
    _only_these_changed(before, after, "optimisation")


@pytest.mark.asyncio
async def test_the_flat_objective_repair_refuses_a_reply_that_eases_a_check(tmp_path: Path) -> None:
    from tests.test_flat_objective_redesign import FACT, _flat_state, _planned as planned_flat
    from core.engine import _flat_request

    eng, _model = await planned_flat(tmp_path)
    path = plan.plan_path(eng.quest_root)
    before = path.read_text(encoding="utf-8")
    eased = [{**o, "tolerance": 0.5} for o in plan.raw_design_block(before)["protocol"]["oracles"]]
    model = Scripted(eng, [{"protocol": {"oracles": eased}}] * 2)
    with pytest.raises(ValueError, match="protocol.oracles"):
        await eng._flat_rewrite(_flat_state(), _flat_request(FACT))
    assert len(model.prompts) == 2
    assert path.read_text(encoding="utf-8") == before


# --- where a check's expected value comes from ------------------------------------------------------------------------


def _research_plan(tmp_path: Path) -> tuple[Engine, Path, str]:
    """A plan under rigor_profile research whose checks do not say where their expected values come from."""
    from tests.test_plan_sources_way_forward import GOOD_MODEL, SOURCES, _cfg, _oracles

    eng = Engine(_cfg(tmp_path))
    eng.config = eng.config.model_copy(update={"rigor_profile": "research"})
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    design = {"hypothesis": "a wider source prints the line best", "method": "sweep sigma",
              "protocol": {"grid": {"sigma": [0.3, 0.5]}, "oracles": _oracles(), "model": GOOD_MODEL}}
    path.write_text(plan.render("source", {}, design, sources=SOURCES), encoding="utf-8")
    return eng, path, path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_the_sources_fill_asks_for_the_checks_sources_and_changes_only_those(tmp_path: Path) -> None:
    from tests.test_plan_sources_way_forward import STATE

    eng, path, before = _research_plan(tmp_path)
    names = [o["name"] for o in plan.raw_design_block(before)["protocol"]["oracles"]]
    reply = {"protocol": {"oracles": [{"name": names[0], "reference": "derivation: sum of S = 1 by definition",
                                       "kind": "invariant"}]}}
    model = Scripted(eng, [reply])
    await eng._fill_plan_sources(dict(STATE))
    _asked_for_parts(model.prompts[0], "oracles")
    after = path.read_text(encoding="utf-8")
    old, new = plan.raw_design_block(before)["protocol"]["oracles"], plan.raw_design_block(after)["protocol"]["oracles"]
    assert new[0]["reference"] == "derivation: sum of S = 1 by definition" and new[1] == old[1]
    _only_these_changed(before, after, "oracles")


@pytest.mark.asyncio
async def test_the_sources_fill_refuses_a_reply_that_changes_a_checks_number(tmp_path: Path) -> None:
    from tests.test_plan_sources_way_forward import STATE

    eng, path, before = _research_plan(tmp_path)
    name = plan.raw_design_block(before)["protocol"]["oracles"][0]["name"]
    bad = {"protocol": {"oracles": [{"name": name, "reference": "derivation: x = 1 by definition", "expected": 7.0}]}}
    model = Scripted(eng, [bad, bad])
    await eng._fill_plan_sources(dict(STATE))
    assert len(model.prompts) == 2 and "changes `expected`" in model.prompts[1]
    assert path.read_text(encoding="utf-8") == before


# --- a protocol that declares no check, and the checks' numbers -----------------------------------------------------------

ORACLE = {"name": "closed_form", "kind": "special_case", "check": "the limit with no fins", "expected": 0.5,
          "tolerance": 0.01, "reference": "derivation: the no-fin limit gives 0.5 by definition", "case": {"n": 1},
          "measure": "ratio"}


@pytest.mark.asyncio
async def test_declaring_a_check_asks_for_the_checks_alone(tmp_path: Path) -> None:
    design = _design()
    design["protocol"].pop("oracles")
    eng, path, before = _planned(tmp_path, design)
    model = Scripted(eng, [{"protocol": {"oracles": [ORACLE]}}])
    assert await eng._declare_oracles() is True
    _asked_for_parts(model.prompts[0], "oracles", "criteria")
    after = path.read_text(encoding="utf-8")
    assert [o["name"] for o in plan.raw_design_block(after)["protocol"]["oracles"]] == ["closed_form"]
    _only_these_changed(before, after, "oracles", prose=False)


@pytest.mark.asyncio
async def test_declaring_a_check_refuses_a_reply_that_also_moves_the_thresholds(tmp_path: Path) -> None:
    design = _design()
    design["protocol"].pop("oracles")
    eng, path, before = _planned(tmp_path, design)
    wide = {"protocol": {"oracles": [ORACLE], "thresholds": {"max_error": 9.0}}}
    model = Scripted(eng, [wide, {"protocol": {"oracles": [ORACLE]}}])
    assert await eng._declare_oracles() is True
    assert len(model.prompts) == 2 and "protocol.thresholds" in model.prompts[1]
    _only_these_changed(before, path.read_text(encoding="utf-8"), "oracles", prose=False)


@pytest.mark.asyncio
async def test_completing_a_checks_numbers_may_change_those_numbers_alone(tmp_path: Path) -> None:
    eng, path, before = _planned(tmp_path)
    bad = {"protocol": {"oracles": [{"name": "no_gain", "expected": 0, "tolerance": 0.02, "check": "no fin removes heat"}]}}
    good = {"protocol": {"oracles": [{"name": "no_gain", "expected": 0, "tolerance": 0.03}]}}
    model = Scripted(eng, [bad, good])
    assert await eng._declare_oracles(["The check 'no_gain' has no numeric expected value."]) is True
    assert len(model.prompts) == 2 and "changes `check`" in model.prompts[1]
    after = path.read_text(encoding="utf-8")
    assert plan.raw_design_block(after)["protocol"]["oracles"][1]["tolerance"] == 0.03
    assert plan.raw_design_block(after)["protocol"]["oracles"][1]["check"] == "no fin adds heat"
    _only_these_changed(before, after, "oracles")


# --- a request about the checks (a reader's objection, a test run, a missing worked example) ---------------------------


@pytest.mark.asyncio
async def test_a_request_about_the_checks_asks_for_the_checks_and_keeps_every_other_part(tmp_path: Path) -> None:
    eng, path, before = _planned(tmp_path)
    old = plan.raw_design_block(before)["protocol"]["oracles"]
    new = [{**old[0], "tolerance": 0.005}, old[1]]
    model = Scripted(eng, [{"protocol": {"oracles": new}}])
    failed = await eng._revise_checks_only("The check 'energy_balance' is too loose; tighten it.")
    assert failed == ""
    _asked_for_parts(model.prompts[0], "oracles", "criteria")
    assert "`model`" not in model.prompts[0].split("Change only")[1]
    after = path.read_text(encoding="utf-8")
    assert plan.raw_design_block(after)["protocol"]["oracles"][0]["tolerance"] == 0.005
    _only_these_changed(before, after, "oracles")


@pytest.mark.asyncio
async def test_a_request_about_the_checks_refuses_a_reply_that_also_names_the_grid(tmp_path: Path) -> None:
    eng, path, before = _planned(tmp_path)
    old = plan.raw_design_block(before)["protocol"]["oracles"]
    wide = {"protocol": {"oracles": [{**old[0], "tolerance": 0.005}, old[1]], "grid": {"fin_spacing": [1, 2]}}}
    model = Scripted(eng, [wide, wide])
    failed = await eng._revise_checks_only("Tighten the check 'energy_balance'.")
    assert "could not be rewritten" in failed and "protocol.grid" in failed
    assert len(model.prompts) == 2
    assert path.read_text(encoding="utf-8") == before


# --- a reply with no JSON, and the sections shown from the block ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reply_with_no_json_is_asked_once_more_and_then_ends_as_a_failed_repair(tmp_path: Path) -> None:
    from tests.test_plan_sources_way_forward import STATE

    eng, path, before = _research_plan(tmp_path)
    name = plan.raw_design_block(before)["protocol"]["oracles"][0]["name"]
    prose = "I filled the sources in, as asked."
    good = {"protocol": {"oracles": [{"name": name, "reference": "derivation: x = 1 by definition"}]}}
    # no JSON twice (the shorter re-ask inside one request), then the repair's own re-ask gets a usable reply
    model = Scripted(eng, [prose, prose, good])
    await eng._fill_plan_sources(dict(STATE))
    assert len(model.prompts) == 3 and "Your last answer was not used" in model.prompts[2]
    assert "not one JSON object" in model.prompts[2]
    assert plan.raw_design_block(path.read_text(encoding="utf-8"))["protocol"]["oracles"][0]["reference"].startswith("derivation")


@pytest.mark.asyncio
async def test_a_reply_with_no_json_at_all_leaves_the_plan_and_raises_the_error_callers_catch(tmp_path: Path) -> None:
    eng, path, before = _planned(tmp_path)
    Scripted(eng, ["no json here"] * 4)
    with pytest.raises(ValueError, match="plan.md is unchanged"):
        await eng._revise_plan_parts("change it", path, scope=plan_scope.PartScope(parts=("thresholds",)), note="n")
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_the_sections_shown_from_the_block_follow_a_change_without_a_finish(tmp_path: Path) -> None:
    eng, path, before = _planned(tmp_path)
    Scripted(eng, [{"protocol": {"optimisation": {"evaluation_budget": SMALLER}}}])
    await eng._revise_plan_parts("smaller", path, scope=plan_scope.PartScope(inside={"optimisation": ("evaluation_budget",)}),
                                 note="n")
    after = path.read_text(encoding="utf-8")
    assert plan.refresh_optimisation_section(plan.refresh_model_section(after)) == after
    assert "2 starting points × 20" in after
