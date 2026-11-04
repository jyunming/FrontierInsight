"""Replay of a run that stopped for a worked example no code could meet (an EUV resist study).

The plan's equation E7 was the probability of a defect, `1-exp(-exp(beta0 + beta1*dose))`, and the worked example it gave
for E7 was the one of another equation (`3*line_width_sd_nm`). The code computed E7 correctly, the test demanded the
other number, and the quest went round three times without anyone telling the plan its example was the problem. Now the
plan is asked, once, to work the example out again from the formula; the request never holds the code's number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.plan_parts_helpers import checks_reply, is_scoped_ask, parts_of, plan_shown

from core import equation_tests as eqt, function_steps as fs, plan
from tests.test_equation_gate import _engine

FORMULA = "p_defect(dose) = 1-exp(-exp(beta0 + beta1*dose_mj_cm2))"
BAD_EXAMPLE = {"inputs": {"line_width_sd_nm": 0.8}, "expected_formula": "3*line_width_sd_nm"}
GOOD_EXAMPLE = {"inputs": {"beta0": -1.0, "beta1": 0.5, "dose_mj_cm2": 2.0},
                "expected_formula": "1-exp(-exp(beta0 + beta1*dose_mj_cm2))"}
STILL_WRONG = {"inputs": {"beta0": -1.0, "beta1": 0.5, "dose_mj_cm2": 2.0}, "expected_formula": "3*dose_mj_cm2"}
THE_CODES_NUMBER = ("0.632", "0.63212")  # 1 - exp(-1): what the code computes for GOOD_EXAMPLE's inputs
CODE = ("import math\n\n\n# E7\ndef cloglog_defect_probability(beta0, beta1, dose_mj_cm2):\n"
        "    return 1 - math.exp(-math.exp(beta0 + beta1 * dose_mj_cm2))\n")
SIMULATE = "from defect import model\n\n\ndef run_cell(cell):\n    return {'p': model.cloglog_defect_probability(-1.0, 0.5, 2.0)}\n"


def _protocol(example: dict[str, Any]) -> dict[str, Any]:
    return {"grid": {"dose_mj_cm2": [10.0, 20.0]}, "model": {"summary": "a defect probability", "equations": [
        {"id": "E7", "formula": FORMULA, "role": "generates", "source": "derivation", "derivation": "cloglog link",
         "example": example}]}}


class _Plan:
    """A fake model: answers a request to the plan with a rewrite that gives E7 the example it is set to; answers a
    request to repair a function with the function as it is. Keeps the requests."""

    def __init__(self, reply_example: dict[str, Any] | None) -> None:
        self.reply_example, self.revisions, self.other = reply_example, [], []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if is_scoped_ask(prompt):
            self.revisions.append(prompt)
            current = plan_shown(prompt)
            if self.reply_example is None:
                return checks_reply(current, prompt)

            def change(block: dict[str, Any]) -> dict[str, Any]:
                model = block["protocol"]["model"]
                eqs = [{**e, "example": self.reply_example} if e["id"] == "E7" else e for e in model["equations"]]
                return {**block, "protocol": {**block["protocol"], "model": {**model, "equations": eqs}}}

            return checks_reply(plan.edit_design_block(current, change) or current, prompt)
        self.other.append(prompt)
        return "```python\n" + CODE + "```"

    async def aclose(self) -> None:
        return None


def _setup(tmp_path: Path, example: dict[str, Any], reply: dict[str, Any] | None):
    engine = _engine(tmp_path)
    protocol = _protocol(example)
    text = plan.render(engine.config.topic, {}, {"hypothesis": "h", "protocol": protocol})
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="model")
    code = engine.quest_root / "code"
    (code / "defect").mkdir(parents=True, exist_ok=True)
    (code / "defect" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (code / "defect" / "model.py").write_text(CODE, encoding="utf-8")
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    model = _Plan(reply)
    engine._client = model
    return engine, model, {"design": {"protocol": protocol}}


def _plan_example(engine) -> Any:  # noqa: ANN001
    design = plan.load_design(engine.quest_root)[0]
    return {e["id"]: e for e in design["protocol"]["model"]["equations"]}["E7"]["example"]


@pytest.mark.asyncio
async def test_a_wrong_example_is_given_back_to_the_plan_without_the_codes_number_and_the_new_one_passes(tmp_path: Path) -> None:
    engine, model, state = _setup(tmp_path, BAD_EXAMPLE, GOOD_EXAMPLE)
    assert await engine._equation_gate(state, None, None) is None
    assert len(model.revisions) == 1, "one request to the plan"
    asked = model.revisions[0].split("# What is asked", 1)[1]
    assert "E7" in asked and "1-exp(-exp(beta0 + beta1*dose_mj_cm2))" in asked
    assert "3*line_width_sd_nm" in asked and "line_width_sd_nm" in asked, "the example as the plan wrote it"
    for number in THE_CODES_NUMBER:
        assert number not in asked, "the plan is never shown the number the code computed"
    assert "cloglog_defect_probability" not in asked and "math.exp" not in asked, "nor the code"
    assert _plan_example(engine) == GOOD_EXAMPLE
    record = json.loads((engine.fi_dir / "equation_tests.json").read_text(encoding="utf-8"))
    assert record["failed"] == [] and record["example_rederived"] == ["E7"]
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "the plan's worked example for E7 could not be met by code that follows its formula" in log
    assert "FI asks the plan to work the example out again from the formula" in log
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    assert "FI asked the plan to work the example of equation E7 out again" in text
    assert len(model.other) <= 4, "the repairs of the function were asked as before (at most two per equation)"
    # A later design (a redesign that brings the old example back) is held to the example that was worked out.
    redesign = {"iteration": 1, "design": {"protocol": _protocol(BAD_EXAMPLE)}}
    kept = engine._draft_protocol(redesign)
    assert kept["model"]["equations"][0]["example"] == GOOD_EXAMPLE


@pytest.mark.asyncio
async def test_an_example_the_plan_keeps_wrong_goes_the_old_way_and_is_asked_only_once(tmp_path: Path) -> None:
    engine, model, state = _setup(tmp_path, BAD_EXAMPLE, STILL_WRONG)
    patch = await engine._equation_gate(state, None, None)
    assert patch is not None and patch["exec_result"]["returncode"] == 1
    assert "equation E7" in patch["exec_result"]["stderr_tail"]
    for number in THE_CODES_NUMBER + ("3.0", "6.0"):
        assert number not in patch["exec_result"]["stderr_tail"]
    assert len(model.revisions) == 1
    # Asked once per quest and equation: the same failure on another pass asks nothing more.
    again = await engine._equation_gate(state, None, None)
    assert again is not None and len(model.revisions) == 1


@pytest.mark.asyncio
async def test_a_redesign_that_gives_the_equation_a_new_formula_may_be_asked_again(tmp_path: Path) -> None:
    engine, model, state = _setup(tmp_path, BAD_EXAMPLE, STILL_WRONG)
    assert await engine._equation_gate(state, None, None) is not None and len(model.revisions) == 1
    new = engine._rederive_examples(state, _protocol(BAD_EXAMPLE), [
        {"id": "E7", "formula": "p = 1-exp(-exp(beta0 + beta1*dose_mj_cm2 + 0))", "example": BAD_EXAMPLE,
         "why": eqt.MISMATCH, "source": "tests"}])
    await new
    assert len(model.revisions) == 2


@pytest.mark.asyncio
async def test_a_plan_that_changes_nothing_or_gives_no_testable_example_falls_back_to_the_old_path(tmp_path: Path) -> None:
    engine, model, state = _setup(tmp_path, BAD_EXAMPLE, None)
    assert await engine._equation_gate(state, None, None) is not None and len(model.revisions) == 1
    engine2, model2, state2 = _setup(tmp_path / "b", BAD_EXAMPLE, {"untestable": "no closed form"})
    patch = await engine2._equation_gate(state2, None, None)
    assert patch is not None, "untestable is not a way out of a failing example"
    assert not (engine2.fi_dir / "examples_rederived.json").read_text(encoding="utf-8").count('"examples"')
    # The reply is not kept in plan.md either: a later pass still tests the equation with the example as it was.
    assert _plan_example(engine2) == BAD_EXAMPLE
    assert eqt.example_rows(engine2._draft_protocol({"iteration": 1, "design": {"protocol": _protocol(BAD_EXAMPLE)}}))[0][
        "state"] == "ok"


@pytest.mark.asyncio
async def test_a_frozen_protocol_is_never_changed_by_the_request(tmp_path: Path) -> None:
    from core import frozen_protocol as frozen

    engine, model, state = _setup(tmp_path, BAD_EXAMPLE, GOOD_EXAMPLE)
    frozen.freeze(engine.quest_root, _protocol(BAD_EXAMPLE), approved_by="auto: test", source="plan.md")
    patch = await engine._equation_gate(state, None, None)
    assert patch is not None and model.revisions == []


def test_the_request_holds_the_equation_the_example_and_the_kind_of_failure_only() -> None:
    text = eqt.rederive_request([{"id": "E7", "formula": FORMULA, "example": BAD_EXAMPLE, "why": eqt.MISMATCH}])
    assert FORMULA in text and "3*line_width_sd_nm" in text and "computes another number" in text
    assert "got" not in text.split("E7 (", 1)[1]


def test_parameters_that_are_not_the_examples_inputs_are_found_before_any_code_is_written() -> None:
    rows = eqt.example_rows(_protocol(BAD_EXAMPLE))
    three = ["beta0", "beta1", "line_width_sd_nm"]
    found = eqt.parameter_problems(rows, {"E7": (three, three)})
    assert list(found) == ["E7"] and "beta0, beta1, line_width_sd_nm" in found["E7"] and "line_width_sd_nm" in found["E7"]
    assert eqt.parameter_problems(rows, {"E7": (["line_width_sd_nm"], ["line_width_sd_nm"])}) == {}
    assert eqt.parameter_problems(rows, {"E7": (["line_width_sd_nm", "**kw"], ["line_width_sd_nm"])}) == {}
    good = eqt.example_rows(_protocol(GOOD_EXAMPLE))
    names = ["beta0", "beta1", "dose_mj_cm2"]
    assert eqt.parameter_problems(good, {"E7": (names, names)}) == {}
    assert eqt.parameter_problems(good, {"E7": ([*names, "rng"], [*names, "rng"])}) != {}, "rng only for a random equation"
    # A parameter with a default need not be an input (the test calls the function with the inputs only) ...
    assert eqt.parameter_problems(good, {"E7": ([*names, "clip"], names)}) == {}
    # ... but every input must be a parameter.
    assert eqt.parameter_problems(good, {"E7": (["beta0", "beta1"], ["beta0", "beta1"])}) != {}


@pytest.mark.asyncio
async def test_an_outline_whose_parameters_are_not_the_examples_inputs_is_given_the_examples_not_asked_of_the_plan(
        tmp_path: Path) -> None:
    engine, model, state = _setup(tmp_path, BAD_EXAMPLE, GOOD_EXAMPLE)
    outline = {"scaffold": "x = 1", "functions": [
        {"name": "cloglog_defect_probability", "signature": "def cloglog_defect_probability(beta0, beta1, dose_mj_cm2) -> float:",
         "one_line_purpose": "defect probability", "implements": "E7"}]}
    import shutil

    shutil.rmtree(engine.quest_root / "code")  # nothing is written yet: the outline has just been made
    specs, _notes = fs.model_functions(outline, engine._protocol_block(state))
    assert [s.params() for s in specs] == [["line_width_sd_nm"]], "FI sets the function's parameters from the example's inputs"
    aligned = engine._outline_with_example_names(state, outline)
    assert aligned["functions"][0]["signature"] == "def cloglog_defect_probability(*, line_width_sd_nm) -> float:"
    await engine._fill_model_functions(state, outline)
    assert model.revisions == [], "the plan is not asked to rename its example to the outline's words"
    assert _plan_example(engine) == BAD_EXAMPLE
