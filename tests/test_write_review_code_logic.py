"""The code reviewer checks LOGIC against the plan, not only coverage (core/code_review.py): a wrong normalisation is found,
goes to the existing repair as a plain directive with no expected value, and is fixed; a reviewer's word alone (even a
high-severity one) never stops a quest that the run's results show to be sound. Fake models, neutral topics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import code_review as cr
from core import review_pairs as pairs
from tests.test_code_review import DESIGN, PROTOCOL, REPAIR_HEAD, STATE, _engine

WITH_PLAN = {**STATE, "design": {**DESIGN, "protocol": PROTOCOL}}
WRONG = "# E1\ndef run_cell(cell):\n    return {'quality': cell['m'] * cell['w'] / cell['length']}\n"
RIGHT = "# E1\ndef run_cell(cell):\n    return {'quality': cell['m'] * cell['w'] / cell['c']}\n"


class _Logic:
    """A reader that marks every requirement implemented (with the line) and, for code that divides by the length, finds the
    wrong normalisation (with a value in its fix, as a careless reviewer might)."""

    def __init__(self, finding: dict[str, Any] | None = None, *, always: bool = False, checked: bool = True) -> None:
        self.finding, self.always, self.checked, self.prompts = finding, always, checked, []
        self.repair_prompts: list[str] = []
        self.repaired = ""

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if prompt.lstrip().startswith(REPAIR_HEAD):
            self.repair_prompts.append(prompt)
            return f"```python\n# file: simulate.py\n{self.repaired}```" if self.repaired else ""
        assert prompt.lstrip().startswith("# Code Against the Plan"), prompt[:80]
        self.prompts.append(prompt)
        reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        wrong = "cell['length']" in prompt
        findings = []
        if self.finding and (wrong or self.always):
            findings = [self.finding]
        return json.dumps({
            "requirements": [{"id": r["id"], "status": "implemented", "file": "simulate.py", "line": 3, "note": ""} for r in reqs],
            "findings": findings,
            "checked": ["E1: the factors and signs"] if self.checked and not findings else [],
            "summary": "s"})

    async def aclose(self) -> None:
        return None


NORMALISATION = {"where": "simulate.py:3", "what": "E1 divides by the length where the plan divides by the damping",
                 "why": "the plan's equation E1 normalises by c", "fix": "divide by cell['c']. The value should come out at 0.123456789.",
                 "severity": "high"}


def test_the_reviewer_is_shown_the_plans_equations_and_cases_but_never_an_expected_value() -> None:
    view = cr.plan_view(DESIGN, PROTOCOL)
    data = json.loads(view)
    assert data["equations"][0]["formula"] == "Q = m * w / c" and data["settings_varied"] == ["c"]
    assert data["checks"][0]["measure"] == "quality" and data["checks"][0]["case"] == {"c": 0.5}
    assert "0.123456789" not in view and "tolerance" not in view and "expected" not in view
    # free text the reader sees is scrubbed too, so it cannot echo a value of the plan's checks
    leaky = {**PROTOCOL, "model": {"summary": "a spring whose result should be 0.123456789",
                                   "equations": [{"id": "E1", "formula": "Q = m * w / c", "source": "gives 0.123456789 at c=0.5"}]}}
    shown = cr.plan_view({**DESIGN, "method": "check against 0.123456789 first"}, leaky)
    assert "0.123456789" not in shown and pairs.PLACEHOLDER in shown and "Q = m * w / c" in shown


def test_findings_are_read_as_problems_of_logic_without_any_value() -> None:
    reqs = cr.requirements(DESIGN, PROTOCOL)
    reply = {"findings": [NORMALISATION | {"quantity": "quality_factor", "constant": True},
                          {"what": "a step is skipped", "where": "experiment.py:10", "severity": "low"}]}
    got = cr.parse_logic(reply, reqs)
    assert [(v["id"], v["kind"], v["file"], v["line"], v["severity"]) for v in got] == [
        ("L1", "logic", "simulate.py", 3, "high"), ("L2", "logic", "experiment.py", 10, "low")]
    assert cr.parse_logic(reply, reqs, pairs.plan_values(PROTOCOL))[0]["fix"].endswith(pairs.PLACEHOLDER + ".")
    got = cr.parse_logic(reply, reqs, pairs.plan_values(PROTOCOL))
    assert got[0]["headline"] is True and got[0]["constant"] is True and "0.123456789" not in got[0]["fix"]
    text = cr.directive(got)
    assert "E1 divides by the length" in text and "divide by cell['c']" in text and "0.123456789" not in text
    assert "simulate.py, line 3" in text
    assert cr.merge_logic(got, cr.parse_logic(reply, reqs)) == got   # the same finding twice is one


@pytest.mark.asyncio
async def test_a_wrong_normalisation_is_sent_to_a_repair_request_without_a_value_and_is_fixed(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    (engine.quest_root / "code" / "simulate.py").write_text(WRONG, encoding="utf-8")
    reader = _Logic(NORMALISATION)
    reader.repaired = RIGHT
    engine._client = reader
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert "0.123456789" not in reader.prompts[0], "the plan's expected value is never shown to the reviewer"
    assert len(reader.repair_prompts) == 1
    said = reader.repair_prompts[0]
    assert "E1 divides by the length" in said and "divide by cell['c']" in said and "### simulate.py" in said
    assert "0.123456789" not in said and pairs.PLACEHOLDER in said, "no value of the plan's reaches the repair"
    assert "has NOT crashed" in said
    # The repaired code is read again and nothing is left.
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == RIGHT
    record = cr.load(engine.quest_root)
    assert record["problems"] == [] and record["final"] is True and record["send_backs"] == 1
    assert cr.not_computed(engine.quest_root, engine.quest_root / "code") == []
    assert cr.write_note(engine.quest_root, engine.quest_root / "code") == ""
    assert len(reader.prompts) == 2 and "Findings still open in the version before this one" in reader.prompts[1]
    assert "E1 divides by the length" in reader.prompts[1].split("Findings still open", 1)[1]


@pytest.mark.asyncio
async def test_a_high_finding_still_there_after_the_repairs_is_said_in_the_limitations_and_never_stops_the_run(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    (engine.quest_root / "code" / "simulate.py").write_text(WRONG, encoding="utf-8")
    engine._client = _Logic(NORMALISATION)
    assert await engine._code_review_gate(STATE, final=True) is None   # no repair is made of the version that runs
    code = engine.quest_root / "code"
    note = cr.write_note(engine.quest_root, code)
    assert "still flags" in note and "E1 divides by the length" in note and "[high]" in note and "not resolved" in note
    # The results cannot decide a normalisation: the quest goes on whatever they hold.
    for result in ({"by_c": [{"quality": 1.0}, {"quality": 2.0}]}, {"quality": 1.0}):
        assert engine._no_results_verdict({**STATE, "result_json": result}) is None
    assert cr.headline_missing(engine.quest_root, code) == [] and cr.headline_constant(engine.quest_root, code) == []


CONSTANT = {"where": "simulate.py:3", "what": "quality_factor does not depend on the damping", "why": "it is rescaled away",
            "fix": "do not divide by the damping twice", "severity": "high", "quantity": "quality_factor", "constant": True}


@pytest.mark.asyncio
async def test_a_false_high_severity_constant_does_not_stop_a_correct_quest(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Logic(CONSTANT, always=True)
    assert await engine._code_review_gate(STATE, final=True) is None
    code = engine.quest_root / "code"
    varies = {**STATE, "result_json": {"cells": [{"c": 0.2, "quality_factor": 2.5}, {"c": 0.5, "quality_factor": 1.0}]}}
    assert engine._no_results_verdict(varies) is None, "the results show the quantity varies: the reviewer was wrong"
    record = cr.load(engine.quest_root)
    assert record["disagreement"] and "quality_factor does not depend" in record["disagreement"][0]
    assert cr.write_note(engine.quest_root, code) == "", "the paper is not told a quantity is a constant when it is not"


@pytest.mark.asyncio
async def test_a_high_severity_constant_the_results_confirm_is_the_honest_stop(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Logic(CONSTANT, always=True)
    assert await engine._code_review_gate(STATE, final=True) is None
    flat = {**STATE, "result_json": {"cells": [{"c": 0.2, "quality_factor": 1.0}, {"c": 0.5, "quality_factor": 1.0}]}}
    verdict = engine._no_results_verdict(flat)
    assert verdict and verdict["stuck"] and verdict["stuck_reason"] == "headline_missing"
    stuck = await engine._node_stuck_no_findings({**flat, "evidence_assessment": verdict})
    assert "is a constant in the run's results" in stuck["stuck"]["problem"]
    # A result that cannot tell (the quantity appears once) leaves the reviewer's word alone: no stop.
    alone = _engine(tmp_path / "b")
    alone._client = _Logic(CONSTANT, always=True)
    await alone._code_review_gate(STATE, final=True)
    assert alone._no_results_verdict({**STATE, "result_json": {"quality_factor": 1.0}}) is None


def test_a_quantity_is_constant_only_when_the_results_hold_it_at_one_value_in_more_than_one_place() -> None:
    assert cr.constant_in("quality_factor", {"a": [{"Quality_Factor": 1.0}, {"quality_factor": 1.0}]}) is True
    assert cr.constant_in("quality_factor", {"a": [{"quality_factor": 1.0}, {"quality_factor": 2.0}]}) is False
    assert cr.constant_in("quality_factor", {"quality_factor": [3, 3, 3]}) is True
    assert cr.constant_in("quality_factor", {"quality_factor": 3}) is None and cr.constant_in("x", {"y": [1, 2]}) is None
    assert cr.constant_in("", {"a": 1}) is None


@pytest.mark.asyncio
async def test_an_agree_only_code_review_is_asked_again_and_then_recorded_as_not_read(tmp_path: Path) -> None:
    class _Agrees(_Logic):
        async def chat(self, messages, **kw):  # noqa: ANN001
            prompt = messages[-1]["content"]
            self.prompts.append(prompt)
            reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
            return json.dumps({"requirements": [{"id": r["id"], "status": "implemented"} for r in reqs]})

    engine = _engine(tmp_path)
    reader = _Agrees()
    engine._client = reader
    assert await engine._code_review_gate(STATE) is None
    assert len(reader.prompts) == 2 and "Your previous reply was no review" in reader.prompts[1]
    record = cr.load(engine.quest_root)
    assert record["error"] == pairs.NOT_A_REVIEW and not record.get("read_version")
    assert "not compared with the plan by another model" in cr.write_note(engine.quest_root, engine.quest_root / "code")
    # Asked again, it says what it checked: a review with nothing found.
    class _Then(_Logic):
        async def chat(self, messages, **kw):  # noqa: ANN001
            self.prompts.append(messages[-1]["content"])
            reqs = json.loads(messages[-1]["content"].split("```json", 1)[1].split("```", 1)[0])
            out: dict[str, Any] = {"requirements": [{"id": r["id"], "status": "implemented"} for r in reqs]}
            if len(self.prompts) > 1:
                out["checked"] = ["E1: the factors and the signs agree with the plan"]
            return json.dumps(out)

    second = _engine(tmp_path / "b")
    reader = _Then()
    second._client = reader
    assert await second._code_review_gate(STATE) is None and len(reader.prompts) == 2
    record = cr.load(second.quest_root)
    assert record["final"] is True and record["checked"] == ["E1: the factors and the signs agree with the plan"]
