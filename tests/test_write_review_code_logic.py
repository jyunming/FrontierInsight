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
from tests.test_code_review import DESIGN, PROTOCOL, STATE, _engine, code_model_answer

pytestmark = pytest.mark.usefixtures("this_python")

WITH_PLAN = {**STATE, "design": {**DESIGN, "protocol": PROTOCOL}}
WRONG = "# E1\ndef run_cell(cell):\n    return {'quality': cell['m'] * cell['w'] / cell['length']}\n"
RIGHT = "# E1\ndef run_cell(cell):\n    return {'quality': cell['m'] * cell['w'] / cell['c']}\n"


class _Logic:
    """A reader that marks every requirement implemented (with the line) and, for code that divides by the length, finds the
    wrong normalisation (with a value in its fix, as a careless reviewer might)."""

    def __init__(self, finding: dict[str, Any] | None = None, *, always: bool = False, checked: bool = True) -> None:
        self.finding, self.always, self.checked, self.prompts = finding, always, checked, []
        self.repair_prompts: list[str] = []
        self.test_prompts: list[str] = []
        self.repaired: dict[str, str] = {}

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        answered = code_model_answer(self, prompt)
        if answered is not None:
            return answered
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
    # a check's answer is hidden by its field: the expected value, tolerance and reference of every check, and the worked
    # examples of the model's steps; the plan's own text is shown as written
    hidden = {**PROTOCOL, "model": {"summary": "a spring", "equations": [{"id": "E1", "formula": "Q = m * w / c"}],
                                    "steps": [{"id": "S1", "name": "load", "does": "scales the load",
                                               "example": {"inputs": {"m": 2.0}, "expected_formula": "0.123456789"}}]}}
    shown = cr.plan_view(DESIGN, hidden)
    assert "0.123456789" not in shown and "Q = m * w / c" in shown and "scales the load" in shown


def test_the_whole_plan_is_shown_with_what_it_holds_fixed_and_is_never_cut_mid_text() -> None:
    protocol = {**PROTOCOL, "constants": {"mass_kg": 2.5}, "optimisation": {"fixed": {"stiffness": 40.0},
                "grid": {"c": [0.1, 0.5]}, "in_each_run": {"loads": [1, 2, 3]}},
                "definitions": [{"metric": "quality", "defined_as": "the peak over the width", "source": "[1]"}]}
    data = json.loads(cr.plan_view(DESIGN, protocol))
    assert data["held_fixed"]["constants"] == {"mass_kg": 2.5} and "not a hardcoding fault" in data["held_fixed"]["note"]
    assert data["held_fixed"]["fixed_values_of_the_settings"] == {"stiffness": 40.0}
    assert data["in_each_run"] == {"loads": [1, 2, 3]} and data["grid"] == PROTOCOL["grid"]
    assert data["definitions"][0]["defined_as"] == "the peak over the width"
    text, dropped = cr.plan_view_fit(DESIGN, {**protocol, "failure_policy": "x" * 3000, "seed_policy": "y" * 300}, limit=1200)
    assert dropped == ["failure_policy"] and "failure_policy" not in text.split("\n(these sections")[0]
    assert "not shown: failure_policy" in text and json.loads(text.split("\n(")[0])


def test_a_value_the_plan_shows_as_a_setting_is_not_replaced_in_what_the_reader_writes() -> None:
    protocol = {**PROTOCOL, "constants": {"mass_kg": 2.5}}
    assert 2.5 not in cr.reportable_values(DESIGN, protocol)
    assert set(cr.reportable_values(DESIGN, PROTOCOL)) <= set(pairs.plan_values(PROTOCOL))


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
    reader.repaired = {"simulate.py": RIGHT}
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
    # "Implemented" with no file and line is not a review: nothing is counted, the requirements are asked for once more
    assert len(reader.prompts) == 2
    record = cr.load(engine.quest_root)
    assert record["error"] == "its answers cited no place in the code it was shown" and not record.get("read_version")
    assert record["cited_outside_what_was_shown"]
    assert cr.write_note(engine.quest_root, engine.quest_root / "code") == ""
    assert "not compared with the plan by another model" in cr.findings_block(engine.quest_root, engine.quest_root / "code")
    # Asked again, it cites the places and says what it checked: a review with nothing found.
    class _Then(_Logic):
        async def chat(self, messages, **kw):  # noqa: ANN001
            self.prompts.append(messages[-1]["content"])
            reqs = json.loads(messages[-1]["content"].split("```json", 1)[1].split("```", 1)[0])
            cite = {"file": "simulate.py", "line": 1} if len(self.prompts) > 1 else {}
            out: dict[str, Any] = {"requirements": [{"id": r["id"], "status": "implemented", **cite} for r in reqs]}
            if len(self.prompts) > 1:
                out["checked"] = ["E1: the factors and the signs agree with the plan"]
            return json.dumps(out)

    second = _engine(tmp_path / "b")
    reader = _Then()
    second._client = reader
    assert await second._code_review_gate(STATE) is None and len(reader.prompts) == 2
    record = cr.load(second.quest_root)
    assert record["final"] is True and record["checked"] == ["E1: the factors and the signs agree with the plan"]


def test_a_value_is_hidden_in_the_entry_that_owns_it_and_nowhere_else() -> None:
    check = {**PROTOCOL["oracles"][0], "check": "Q at c=0.5; the result must be 0.123456789 within 1e-9",
             "expected_formula": "0.123456789"}
    protocol = {**PROTOCOL, "oracles": [check, {"name": "other", "kind": "invariant", "check": "a second check",
                                               "case": {"c": 1.0}, "measure": "quality", "expected": 10, "tolerance": 0.5}],
                "model": {"summary": "the scan covers the interval [10, p-10] nm", "equations": [
                    {"id": "E1", "formula": "Q = m * w / c", "source": "a plain derivation"}]}}
    data = json.loads(cr.plan_view({**DESIGN, "method": "a window of [10, p-10] nm, then c = 0.5"}, protocol))
    # a check's own text no longer states its expected value, tolerance or formula result
    assert "0.123456789" not in data["checks"][0]["check"] and pairs.PLACEHOLDER in data["checks"][0]["check"]
    assert "c=0.5" in data["checks"][0]["check"] and data["checks"][0]["case"] == {"c": 0.5}
    # the same number in another section is the plan's own prose and stays as written
    assert data["model"] == "the scan covers the interval [10, p-10] nm"
    assert data["method"] == "a window of [10, p-10] nm, then c = 0.5"
    assert "[10, p-10]" in json.dumps(data) and "0.123456789" not in json.dumps(data)


def test_a_plan_that_does_not_fit_is_split_by_whole_sections_and_no_request_is_over_the_cap() -> None:
    oracles = [{"name": f"c{i}", "kind": "invariant", "check": "x" * 600, "case": {"c": 0.5}, "measure": "quality",
                "expected": 0, "tolerance": 1e-9} for i in range(100)]
    views = cr.plan_views(DESIGN, {**PROTOCOL, "oracles": oracles}, limit=20000)
    assert len(views) > 1 and all(len(v["text"]) <= 20000 for v in views)
    got = sum(len(json.loads(v["text"].split("\n(")[0]).get("checks", [])) for v in views)
    assert got == 100, "every check is in some request"
    assert any("other requests of this reading" in v["text"] for v in views)


def test_the_note_of_what_a_request_leaves_out_is_one_entry_per_section_and_the_request_stays_under_the_cap() -> None:
    equations = [{"id": f"E{i}", "formula": "x" * 30000} for i in range(400)]
    protocol = {**PROTOCOL, "model": {"summary": "a spring", "equations": equations}}
    views = cr.plan_views(DESIGN, protocol, limit=20000)
    assert views and all(len(v["text"]) <= 20000 for v in views)
    notes = [x for v in views for x in v["left_out"]]
    assert "equations: 400 items left out" in set(notes), notes[:3]
    assert len(set(notes)) < 10 and all(len(v["left_out"]) < 10 for v in views)
    assert "spring" in json.dumps([json.loads(v["text"].split("\n(")[0]) for v in views])


def _sections_of(views: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Every request's sections read back from its text: ``{section: [value in each request that holds it]}``."""
    out: dict[str, list[Any]] = {}
    for v in views:
        for name, value in json.loads(v["text"].split("\n(")[0]).items():
            out.setdefault(name, []).append(value)
    return out


def _long_audit() -> dict[str, Any]:
    return {"note": "proposed, not applied", "items": [{"path": f"p{i}", "proposed": "z" * 1500} for i in range(30)]}


def test_a_plan_over_the_cap_shows_every_section_in_some_request_and_drops_none() -> None:
    protocol = {**PROTOCOL, "constants": {"mass_kg": 2.5}, "optimisation": {"grid": {"c": [0.1, 0.5]},
                "in_each_run": {"loads": [1, 2, 3]}},
                "definitions": [{"metric": "quality", "defined_as": "the peak over the width", "source": "[1]"}],
                "ci_method": "a percentile interval", "seed_policy": "one seed per cell", "failure_policy": "retry once"}
    limit = 20000
    whole = cr.plan_sections(DESIGN, protocol, _long_audit())
    assert len(json.dumps(whole)) > limit
    views = cr.plan_views(DESIGN, protocol, limit=limit, open_items=_long_audit())
    assert len(views) > 1 and all(len(v["text"]) <= limit for v in views)
    assert all(v["left_out"] == [] for v in views)
    shown = _sections_of(views)
    assert set(shown) == set(whole), set(whole) - set(shown)
    for name in ("definitions", "grid", "in_each_run", "ci_method", "seed_policy", "failure_policy"):
        assert shown[name] == [whole[name]]
    # a list section is split by its items: each item is shown exactly once
    items = [x for part in shown[cr._AUDIT_SECTION] for x in part]
    assert sorted(i["path"] for i in items if "path" in i) == sorted(f"p{i}" for i in range(30))


def test_a_section_too_big_alone_is_still_left_out_and_named() -> None:
    protocol = {**PROTOCOL, "failure_policy": "q" * 3000, "seed_policy": "one seed per cell"}
    views = cr.plan_views(DESIGN, protocol, limit=1200)
    assert all(len(v["text"]) <= 1200 for v in views)
    assert {n for v in views for n in v["left_out"]} == {"failure_policy"}
    shown = _sections_of(views)
    assert "failure_policy" not in shown and shown["seed_policy"] == ["one seed per cell"]
    assert all("not shown: failure_policy" in v["text"] for v in views)


def test_every_plan_part_reaches_a_request_and_requests_are_the_larger_of_code_and_plan_parts(tmp_path: Path) -> None:
    (tmp_path / "simulate.py").write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    protocol = {**PROTOCOL, "definitions": [{"metric": "m", "defined_as": "d" * 1000, "source": "[1]"}] * 60}
    plans = cr.plan_views(DESIGN, protocol, open_items=_long_audit())
    parts = cr.source_parts_detail(tmp_path)
    calls = cr.reading_calls(tmp_path, DESIGN, protocol, open_items=_long_audit())
    assert len(plans) > 1 and len(calls) == max(len(parts), len(plans))
    assert {c["plan"] for c in calls} == {p["text"] for p in plans}


def test_a_section_that_fits_a_request_alone_is_never_left_out_and_no_text_is_over_the_cap() -> None:
    cap = cr._MAX_PLAN_CHARS
    protocol = {**PROTOCOL, "grid": {"a": ["x" * (cap - 2000)]}}
    design = {**DESIGN, "method": "m" * 6000}
    views = cr.plan_views(design, protocol)
    assert all(v["left_out"] == [] for v in views), views[0]["left_out"]
    assert "grid" in _sections_of(views) and "method" in _sections_of(views)
    assert all(len(v["text"]) <= cap for v in views)
    # a section that fits a request only with the shortest note is still shown
    protocol = {**PROTOCOL, "grid": {"a": ["x" * (cap - 600)]}}
    views = cr.plan_views(design, protocol)
    assert "grid" in _sections_of(views) and all(len(v["text"]) <= cap for v in views)
    # the cap holds for the composed text, whatever is left out, also when nothing fits
    for limit in (40, 300):
        small = cr.plan_views({**DESIGN, "method": "x" * 40}, PROTOCOL, limit=limit)
        assert small and all(len(v["text"]) <= limit for v in small)


def test_a_list_section_split_across_requests_says_the_other_items_are_elsewhere() -> None:
    protocol = {**PROTOCOL, "definitions": [{"metric": f"m{i}", "defined_as": "d" * 1500, "source": "[1]"} for i in range(4)]}
    views = cr.plan_views(DESIGN, protocol, limit=4000)
    holding = [v for v in views if "definitions" in v["sections"]]
    assert len(holding) > 1
    for v in holding:
        assert "the other items of these sections are in the plan given with the other requests" in v["text"]
        assert "definitions" in v["text"].split("the other items of these sections")[1]
