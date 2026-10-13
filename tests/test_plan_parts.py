"""When FI asks the plan's model to complete a part of the plan, it asks for ONLY the missing parts, as one JSON object
(``agents/plan_parts.md``), and FI is the only writer of the plan's YAML. Fake model only.

The real stop this replaces (a long plan of 900 lines, rewritten whole by the model on every ask): "the design block is not
valid YAML (line 903 of the block): mapping values are not allowed here", "no `## The design (used as written)` section was
found", three to ten minutes per ask, and the quest stopped at the plan after two hours with nothing run.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan
from core import topic_requirements as tr
from tests.test_plan_optimisation import EXTRA, Paused, _engine, _NO_OBJECTIONS
from tests.test_protocol_part_asked import GOOD_GRID, MARK, TOPIC, _draft, _record

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "plan_parts_not_whole"
MIXED = {"fin_spacing": [2.0, [1, 2]]}  # a list among single values: cannot be read


class Killed(BaseException):
    """The quest process dying in the middle of a call (not an error the engine can catch)."""


class RawModel:
    """The plan step's model, answering each request for parts with ``answers`` as they are (text, or a function of the
    plan as it stands); a ``None`` is a failed call."""

    def __init__(self, eng: Any, answers: list[Any], draft: dict[str, Any]) -> None:
        self.eng, self.answers, self.draft = eng, list(answers), draft
        self.asked: list[str] = []
        self.drafted = False
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if MARK in prompt:
            self.asked.append(prompt)
            answer = self.answers.pop(0) if self.answers else None
            if answer is None:
                raise RuntimeError("no answer")
            if isinstance(answer, type) and issubclass(answer, BaseException):
                raise answer()
            return answer(plan.plan_path(self.eng.quest_root).read_text(encoding="utf-8")) if callable(answer) else answer
        if prompt.lstrip().startswith("# What a Research Topic Asks For"):
            return "{}"  # the topic names nothing to look for: this file is about the parts asked for
        if not self.drafted:
            self.drafted = True
            return json.dumps({**self.draft, "plan": EXTRA})
        return _NO_OBJECTIONS


def _stop(eng: Any) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    return seen


def _text(eng: Any) -> str:
    return plan.plan_path(eng.quest_root).read_text(encoding="utf-8")


def _written_by_the_model(eng: Any) -> list[dict[str, Any]]:
    """The versions of plan.md that record a completion by the plan's model."""
    return [h for h in plan.history(eng.quest_root) if "written by the plan's model" in str(h.get("note"))]


def _design(eng: Any) -> dict[str, Any]:
    design = plan.parse(_text(eng)).design
    assert design is not None
    return design


# (a) a valid answer is merged, and nothing else of the plan moves ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_parts_answer_is_merged_and_nothing_else_of_the_plan_changes(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    draft = _draft(grid=MIXED)
    model = RawModel(eng, [json.dumps({"protocol": {"grid": GOOD_GRID}, "hypothesis": "something else"})], draft)
    # The plan as FI first writes it (the grid is unreadable there); then the plan step completes it.
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    after = _design(eng)
    assert after["protocol"]["grid"] == GOOD_GRID
    assert len(model.asked) == 1, "one call, not two whole-file tries"
    assert _record(eng)["count"] == 1 and _record(eng)["outcome"] in ("kept", "told in the plan")
    # The request asks for the parts as JSON, shows the plan as context, and does not ask for the file back.
    assert "ONE JSON object" in model.asked[0] and TOPIC in model.asked[0]
    assert "Return the **entire** file" not in model.asked[0]
    # Every other part of the design is the plan's own as it was; the YAML block re-parses.
    first_protocol = plan.normalize_protocol({k: v for k, v in draft["protocol"].items() if k != "grid"})[0]
    assert {k: v for k, v in after["protocol"].items() if k != "grid"} == first_protocol
    assert after["hypothesis"] == draft["hypothesis"], "a top-level key the request did not ask for is not taken"
    assert plan.raw_design_block(_text(eng)) is not None
    assert plan.history(eng.quest_root)[-1]["by"] == "engine"


@pytest.mark.asyncio
async def test_a_part_written_beside_protocol_or_without_it_is_read_where_the_readers_look(tmp_path: Path) -> None:
    for n, reply in enumerate((json.dumps({"grid": GOOD_GRID}), "Here it is:\n```json\n" + json.dumps({"protocol": {"grid": GOOD_GRID}}) + "\n```")):
        eng = _engine(tmp_path / str(n), [])
        RawModel(eng, [reply], _draft(grid=MIXED))
        await eng._node_plan({"topic": TOPIC, "iteration": 0})
        assert _design(eng)["protocol"]["grid"] == GOOD_GRID


# (b) an answer that breaks a rule is refused with the exact reason ----------------------------------------------------------


@pytest.mark.asyncio
async def test_an_answer_that_breaks_a_rule_is_refused_with_the_reason_and_the_plan_is_unchanged(tmp_path: Path) -> None:
    bad = [{"id": "base_temperature", "kind": "median", "estimand": "x", "unit": "y"}]
    eng = _engine(tmp_path, [])
    # Ask 1: a metric of a kind the plan does not know. Ask 2: the same plan, a different (also unusable) answer.
    other = [{"id": "base_temperature", "kind": "mode", "estimand": "x", "unit": "y"}]
    model = RawModel(eng, [json.dumps({"protocol": {"metrics": bad}}), json.dumps({"protocol": {"metrics": other}})],
                     _draft(grid=GOOD_GRID, metrics=bad))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert "Your last answer was not used" not in model.asked[0]
    before = _text(eng)
    # The refusal is the checker's own reason, and it went back with the next ask (the request after the first refusal).
    assert 2 <= len(model.asked) <= 4
    assert "Your last answer was not used" in model.asked[1]
    assert "`protocol.metrics` as you wrote it cannot be read" in model.asked[1] and "median" in model.asked[1]
    # The plan was not changed by either answer: no version of it was written by the plan's model, and it holds neither kind.
    assert "kind: mode" not in before and "kind: median" not in before
    assert _written_by_the_model(eng) == [], "no version was written for an answer that was refused"


# (c) unreadable JSON: one more ask, then refused, bounded ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reply_with_no_json_is_asked_once_more_then_counted_as_refused_and_the_calls_stay_bounded(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = RawModel(eng, ["no object here"] * 20, _draft(grid=MIXED))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen, "a sweep that cannot be read still stops the quest, as before"
    record = _record(eng)
    # two chat calls make one ask (the reply, then one short re-ask); the same refusal twice ends the asking
    assert record["count"] == 2 and len(model.asked) == 4 and record["calls"] <= 8
    assert "Your previous reply could not be used" in model.asked[1], "the second call is the short JSON re-answer"
    assert "grid" not in plan.raw_design_block(_text(eng))["protocol"], "plan.md is as it was: the unreadable grid was left out"
    assert _written_by_the_model(eng) == []
    again = len(model.asked)
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) == again, "a resume asks no more"


# (d) the count is written before the call --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_ask_is_counted_before_the_call_so_a_killed_run_does_not_ask_again_on_re_entry(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = RawModel(eng, [Killed, None, None, None], _draft(grid=MIXED))
    with pytest.raises(Killed):
        await eng._node_plan({"topic": TOPIC, "iteration": 0})
    record = _record(eng)
    assert record["count"] == 1 and record["outcome"] == "asking" and record["calls"] == 2, record
    assert record["parts"] == ["grid"], "the part asked about is written down too"
    # Re-entering (a resume) asks no second time for the part it already asked about.
    design, sha = eng._design_from_plan()
    state = {"topic": TOPIC, "iteration": 0}
    await eng._ask_plan_for_the_parts(state, design, sha)
    assert len(model.asked) == 1 and _record(eng)["count"] == 1


# (e) a person's edit made during the call is kept ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_plan_a_person_saved_during_the_call_is_not_overwritten(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    path = plan.plan_path(eng.quest_root)
    mine = {"text": ""}

    def edited_meanwhile(text: str) -> str:
        mine["text"] = text + "\n<!-- my note -->\n"
        path.write_text(mine["text"], encoding="utf-8")
        return json.dumps({"protocol": {"grid": GOOD_GRID}})

    RawModel(eng, [edited_meanwhile], _draft(grid=MIXED))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    text = path.read_text(encoding="utf-8")
    assert "my note" in text, "the person's version is on disk, not the model's"
    assert "grid" not in plan.raw_design_block(text)["protocol"], "the model's grid was not written over it"
    assert _written_by_the_model(eng) == []


# (f) the real round-8 plan ------------------------------------------------------------------------------------------------------


def test_the_real_long_plan_takes_a_small_answer_and_its_topic_gaps_shrink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan of the real stop (a search for the best design, 1,300 lines). The old request had the model write all of it
    back and it came back as invalid YAML; the answer is now a few lines, and FI writes the YAML."""
    eng = _engine(tmp_path, [])
    shutil.copy(FIXTURE / "plan.md", plan.plan_path(eng.quest_root).parent.mkdir(parents=True, exist_ok=True) or plan.plan_path(eng.quest_root))
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    req = tr.parse(json.loads((FIXTURE / "topic_requirements.json").read_text(encoding="utf-8")))
    eng.config.topic = "a study that searches for the best design and compares illumination settings"
    before_text = _text(eng)
    design, sha = eng._design_from_plan()
    assert design is not None

    async def no_judge(self: Any, requirements: Any, view: Any, unresolved: Any) -> dict[str, Any]:
        return {ident: "missing" for ident in unresolved}  # the reader of the topic is not asked: what a name cannot find is missing

    monkeypatch.setattr(type(eng), "_judge_topic_items", no_judge)
    gaps_before, _u = asyncio.run(eng._topic_gaps(req, design))
    assert gaps_before, "the real plan lacked what the topic asked for"
    pitches = next(f for f in req["factors"] if f["name"] == "pitches")  # a numeric setting: its value is held fixed in the search
    scan = {"pitches": [float(v) for v in pitches["range"]]}
    answer = {"protocol": {"optimisation": {"grid": scan, "fixed": {"pitches": scan["pitches"][0]}}}}
    calls: list[str] = []

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        calls.append(messages[-1]["content"])
        return json.dumps(answer)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    eng._progress = lambda *a, **k: None  # type: ignore[method-assign]
    out, new_sha = asyncio.run(eng._ask_plan_for_the_topic({"topic": eng.config.topic}, design, sha, req))
    # The first answer is taken; the asks that follow (other gaps remain, and the same answer adds nothing) are refused and bounded.
    assert 1 <= len(calls) <= 3 and "ONE JSON object" in calls[0] and "Return the **entire** file" not in calls[0]
    assert len(_written_by_the_model(eng)) == 1
    assert new_sha != sha and out["protocol"]["optimisation"]["grid"] == scan
    after_text = _text(eng)
    assert after_text != before_text and plan.raw_design_block(after_text) is not None
    # everything else of the protocol is the plan's own
    for key in ("oracles", "metrics", "model", "thresholds", "definitions", "simplifications"):
        assert out["protocol"].get(key) == design["protocol"].get(key), key
    gaps_after, _u = asyncio.run(eng._topic_gaps(req, out))
    assert len(gaps_after) < len(gaps_before), (len(gaps_before), len(gaps_after))


# LaTeX in a string of the answer ----------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latex_with_single_backslashes_in_the_answer_is_read_as_written(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    reply = r'{"protocol": {"model": {"equations": [{"id": "E1", "formula": "$\text{SNR} = \frac{a}{b}$", "role": "analyses"}]}}}'
    assert r"\text{SNR}" in reply  # one backslash, as a model often writes it

    async def chat(prompt: str, **kw: Any) -> str:
        return reply

    eng._chat = chat  # type: ignore[method-assign]
    parts = await eng._ask_for_plan_parts("ask")
    assert parts["protocol"]["model"]["equations"][0]["formula"] == r"$\text{SNR} = \frac{a}{b}$"
