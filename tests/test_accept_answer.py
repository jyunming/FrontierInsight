"""An answer to a request to complete the plan is taken part by part and entry by entry: what can be read is kept, what
cannot goes back to the model with the reader's own reason. A real quest (fdr, r10b) lost a whole answer to one metric
whose `kind` the reader refuses; the deep-UV quests lost 14 answers in one run that way. Nothing is loosened: what
cannot be read is still not taken."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core import plan
from tests.test_code_review import _engine

UNIT = {"unit": "one simulated experiment"}
FDR = {"id": "realized_fdr", "kind": "proportion", "estimand": "share of false discoveries", **UNIT}
POWER = {"id": "power", "kind": "mean", "estimand": "mean power", **UNIT}
REFUSED_KIND = {"id": "realized_fdr_uncertainty", "kind": "standard_error"}  # the r10b answer's entry
PLAN = {"metrics": [FDR], "runs_per_setting": 100, "thresholds": {"alpha": 0.05}}


def test_what_can_be_read_is_taken_and_what_cannot_comes_back_with_its_reason() -> None:
    answer = {**PLAN, "metrics": [FDR, REFUSED_KIND, POWER], "thresholds": {"alpha": 0.05, "q": "about five percent"}}
    got, refused = plan.accept_answer(PLAN, answer)
    assert got is not None and [m["id"] for m in got["metrics"]] == ["realized_fdr", "power"]
    assert got["thresholds"] == {"alpha": 0.05}
    assert any("realized_fdr_uncertainty" in r and "`kind` must be" in r for r in refused)
    assert any("`q`" in r and "not one number" in r for r in refused)


def test_an_answer_that_breaks_what_the_plan_had_keeps_the_plan_s_own() -> None:
    answer = {**PLAN, "metrics": [{"id": "realized_fdr", "kind": "share"}, POWER]}
    got, refused = plan.accept_answer(PLAN, answer)
    assert got is not None and [(m["id"], m["kind"]) for m in got["metrics"]] == [("power", "mean"), ("realized_fdr", "proportion")]
    assert refused


def test_an_answer_with_nothing_readable_or_that_costs_the_plan_a_part_is_not_taken() -> None:
    assert plan.accept_answer(PLAN, "junk")[0] is None
    assert plan.accept_answer(PLAN, {**PLAN, "metrics": [FDR, REFUSED_KIND]})[0] is None  # adds nothing readable
    whole, refused = plan.accept_answer(PLAN, {**PLAN, "metrics": [FDR, POWER]})
    assert whole is not None and refused == []  # read whole: as before


def test_the_plan_step_sends_the_part_not_used_back_with_the_next_request(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    plan.plan_path(engine.quest_root).parent.mkdir(parents=True, exist_ok=True)
    plan.plan_path(engine.quest_root).write_text("# plan\n", encoding="utf-8")
    asked: list[str] = []

    async def complete(ask, path, **kwargs):
        asked.append(ask)
        engine._answer_refused = ["`protocol.metrics` entry `realized_fdr_uncertainty`: `kind` must be `proportion`"] \
            if len(asked) == 1 else []

    lacks = iter([["uncertainty is missing"], ["uncertainty is missing"], []])
    engine._complete_plan_parts = complete  # type: ignore[method-assign]
    engine._design_from_plan = lambda: ({"a": 1}, "sha2")  # type: ignore[method-assign]
    engine._progress = lambda *a, **k: None  # type: ignore[method-assign]
    asyncio.run(engine._ask_plan_to_complete(
        {}, "sha", lacks=lambda design: next(lacks), request=lambda design, text: "ask",
        keep=lambda before, revised: revised, file="part_asked.json", what="settings", short="it", note="n"))
    assert len(asked) == 2
    assert "Part of your last answer was used. This part was not:" in asked[1] and "realized_fdr_uncertainty" in asked[1]
    assert "was used in part" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


def test_the_design_step_keeps_what_it_can_read_and_asks_again_for_the_rest(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    prompts: list[str] = []

    async def chat(prompt, **kwargs):
        prompts.append(prompt)
        metrics = [FDR, REFUSED_KIND, POWER] if len(prompts) == 1 else [FDR, POWER, {**REFUSED_KIND, **UNIT, "kind": "mean",
                                                                                      "estimand": "spread of the fdr"}]
        return json.dumps({"protocol": {**PLAN, "metrics": metrics}})

    gaps_left = iter([([{"sentence": "the topic asks for its uncertainty"}], []), ([], [])])

    async def topic_gaps(req, design, protocol=None):
        return next(gaps_left)

    engine._chat = chat  # type: ignore[method-assign]
    engine._progress = lambda *a, **k: None  # type: ignore[method-assign]
    engine._topic_gaps = topic_gaps  # type: ignore[method-assign]
    monkeypatch.setattr("core.engine._topic_req.merge", lambda old, new, gaps: {**old, "protocol": new["protocol"]})
    gaps = [{"sentence": "the topic asks for power and its uncertainty"}]
    result = asyncio.run(engine._ask_design_for_the_topic({}, {"protocol": PLAN}, {}, gaps, iteration=1))
    assert len(prompts) == 2
    assert "Part of your last answer was used. This part was not:" in prompts[1] and "`kind` must be" in prompts[1]
    assert [m["id"] for m in result["protocol"]["metrics"]] == ["realized_fdr", "power", "realized_fdr_uncertainty"]


ORACLE = {"name": "tail_agreement", "kind": "second_implementation", "check": "p-values agree with erfc",
          "expected": 0, "tolerance": 1e-12, "tolerance_mode": "absolute",
          "reference": "independent second implementation: shares no code", "case": {"m": 20}, "measure": "max_abs_diff"}
CRITERION = {"name": "tail-agreement", "oracle": "tail_agreement", "use": "value", "direction": "lower", "target": 0,
             "tolerance": 1e-12}


def test_a_part_that_rests_on_another_comes_back_whatever_the_order_of_the_plan() -> None:
    # The criterion names the check: put back before the check, it cannot be read; it must still come back.
    before = {"criteria": [CRITERION], "oracles": [ORACLE], **PLAN}
    assert plan.normalize_protocol(before)[0] is not None
    got, refused = plan.accept_answer(before, {"metrics": [FDR, REFUSED_KIND, POWER]})
    assert got is not None and refused
    assert [c["name"] for c in got["criteria"]] == ["tail-agreement"]
    assert [o["name"] for o in got["oracles"]] == ["tail_agreement"]
    assert {m["id"] for m in got["metrics"]} == {"realized_fdr", "power"}


def test_an_equation_the_answer_lost_inside_the_model_comes_back_beside_the_answer_s_own() -> None:
    model = {"summary": "independent tests", "equations": [
        {"id": "E1", "formula": "p = 0.5*erfc(z/sqrt(2))", "role": "generates", "source": "derivation", "derivation": "x = y"},
        {"id": "E2", "formula": "q = p*m/k", "role": "analyses", "source": "derivation", "derivation": "x = y"}]}
    before = {**PLAN, "model": model}
    assert plan.normalize_protocol(before)[0] is not None
    answer = {**PLAN, "metrics": [FDR, POWER], "model": {"summary": "independent tests", "equations": [
        model["equations"][0], {"id": "E3", "role": "generates"},  # E3 has no formula: not taken
        {"id": "E4", "formula": "power = s/n1", "role": "analyses", "source": "derivation", "derivation": "x = y"}]}}
    got, refused = plan.accept_answer(before, answer)
    assert got is not None and refused
    assert [e["id"] for e in got["model"]["equations"]] == ["E1", "E4", "E2"]
    assert {m["id"] for m in got["metrics"]} == {"realized_fdr", "power"}


def test_the_same_part_refused_twice_is_not_asked_for_a_third_time(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    plan.plan_path(engine.quest_root).parent.mkdir(parents=True, exist_ok=True)
    plan.plan_path(engine.quest_root).write_text("# plan\n", encoding="utf-8")
    asked: list[str] = []

    async def complete(ask, path, **kwargs):
        asked.append(ask)
        engine._answer_refused = ["`protocol.metrics` entry `realized_fdr_uncertainty`: `kind` must be `proportion`"]

    engine._complete_plan_parts = complete  # type: ignore[method-assign]
    engine._design_from_plan = lambda: ({"a": 1}, "sha2")  # type: ignore[method-assign]
    engine._progress = lambda *a, **k: None  # type: ignore[method-assign]
    asyncio.run(engine._ask_plan_to_complete(
        {}, "sha", lacks=lambda design: ["uncertainty is missing"], request=lambda design, text: "ask",
        keep=lambda before, revised: revised, file="part_asked.json", what="settings", short="it", note="n"))
    assert len(asked) == 2
    assert "same part that could not be used" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


def test_a_part_refused_again_while_other_gaps_are_filled_is_asked_for_again(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    plan.plan_path(engine.quest_root).parent.mkdir(parents=True, exist_ok=True)
    plan.plan_path(engine.quest_root).write_text("# plan\n", encoding="utf-8")
    asked: list[str] = []

    async def complete(ask, path, **kwargs):
        asked.append(ask)
        engine._answer_refused = ["`protocol.metrics` entry `realized_fdr_uncertainty`: `kind` must be `proportion`"]

    lacks = iter([["a", "b", "c"], ["a", "b"], ["a"], ["a"], ["a"], ["a"]])
    engine._complete_plan_parts = complete  # type: ignore[method-assign]
    engine._design_from_plan = lambda: ({"a": 1}, "sha2")  # type: ignore[method-assign]
    engine._progress = lambda *a, **k: None  # type: ignore[method-assign]
    asyncio.run(engine._ask_plan_to_complete(
        {}, "sha", lacks=lambda design: next(lacks), request=lambda design, text: "ask",
        keep=lambda before, revised: revised, file="part_asked.json", what="settings", short="it", note="n"))
    assert len(asked) == 3  # asked again while the gaps shrank; stopped once the same refusal left them as they were
    record = json.loads((engine.fi_dir / "part_asked.json").read_text(encoding="utf-8"))
    assert "same refusal again" in json.dumps(record)
