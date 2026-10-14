"""Replay, in the engine, of suite r12 first_order_kinetics (quest 1791991314): equation E4 defines two outputs, the
plan's worked example gave one value, and the correct function returned both by name. The run's own record (the
function's return) now puts the example at fault: no repair of the function is spent, the plan is asked once for a value
of each output, and the equation passes with the example it gives (tests/test_replay_multi_output_equation.py has the
test file's side)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import plan
from tests.test_equation_gate import _engine

FORMULA = "ln(k_i) = alpha + beta*(1/T_i - mean(1/T)); Ea_hat=-R*beta; ln_A_hat=alpha+Ea_hat*mean(1/T)/R"
ONE_VALUE = {"inputs": {"alpha": 4.0, "beta": -1000.0, "R": 8.314462618, "mean_invT": 0.002},
             "expected_formula": "alpha+(-R*beta)*mean_invT/R"}
PER_OUTPUT = {"inputs": ONE_VALUE["inputs"],
              "expected_formula": {"Ea_hat": "-R*beta", "ln_A_hat": "alpha+(-R*beta)*mean_invT/R"}}
CODE = ("# E4\ndef arrhenius_parameters(alpha, beta, R, mean_invT):\n"
        "    return {'Ea_hat': -R * beta, 'ln_A_hat': alpha - beta * mean_invT}\n")
WRONG = CODE.replace("alpha - beta * mean_invT", "alpha + beta * mean_invT")
SIMULATE = "from kinetics import model\n\n\ndef run_cell(cell):\n    return model.arrhenius_parameters(4.0, -1.0, 8.3, 0.002)\n"


def _protocol(example: dict[str, Any]) -> dict[str, Any]:
    return {"grid": {"T_low_K": [450.0]}, "model": {"summary": "an Arrhenius fit", "equations": [
        {"id": "E4", "formula": FORMULA, "role": "analyses", "source": "derivation", "derivation": "ln of E2",
         "example": example}]}}


class _Model:
    """A fake model: a request to the plan gives E4 the example it is set to; a request to repair a function gives the
    function back as written. Keeps the requests."""

    def __init__(self, reply_example: dict[str, Any] | None, code: str = CODE) -> None:
        self.reply_example, self.code, self.revisions, self.other = reply_example, code, [], []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if "You are revising the plan" in prompt:
            self.revisions.append(prompt)
            current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
            if self.reply_example is None:
                return current

            def change(block: dict[str, Any]) -> dict[str, Any]:
                model = block["protocol"]["model"]
                eqs = [{**e, "example": self.reply_example} if e["id"] == "E4" else e for e in model["equations"]]
                return {**block, "protocol": {**block["protocol"], "model": {**model, "equations": eqs}}}

            return plan.edit_design_block(current, change) or current
        self.other.append(prompt)
        return "```python\n" + self.code + "```"

    async def aclose(self) -> None:
        return None


def _setup(tmp_path: Path, example: dict[str, Any], reply: dict[str, Any] | None, code: str = CODE):
    engine = _engine(tmp_path)
    protocol = _protocol(example)
    text = plan.render(engine.config.topic, {}, {"hypothesis": "h", "protocol": protocol})
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="model")
    pkg = engine.quest_root / "code" / "kinetics"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (pkg / "model.py").write_text(code, encoding="utf-8")
    (engine.quest_root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    model = _Model(reply, code)
    engine._client = model
    return engine, model, {"design": {"protocol": protocol}}


def _plan_example(engine) -> Any:  # noqa: ANN001
    design = plan.load_design(engine.quest_root)[0]
    return {e["id"]: e for e in design["protocol"]["model"]["equations"]}["E4"]["example"]


@pytest.mark.asyncio
async def test_a_one_value_example_for_outputs_returned_by_name_is_not_tested_and_the_study_runs(tmp_path: Path) -> None:
    # The person's decision (2026-10-14): when FI's own test cannot judge the function, the equation is recorded as not
    # tested before the run (said in the limitations) and the study goes on; nothing is repaired or asked.
    engine, model, state = _setup(tmp_path, ONE_VALUE, PER_OUTPUT)
    assert await engine._equation_gate(state, None, None) is None
    assert model.other == [] and model.revisions == [], "no repair of the function, no request to the plan"
    assert _plan_example(engine) == ONE_VALUE
    record = json.loads((engine.fi_dir / "equation_tests.json").read_text(encoding="utf-8"))
    assert record["failed"] == [] and record["tested"] == []
    assert any("equation E4 is not tested before the run" in n and "by name" in n for n in record["not_tested"])
    assert "equation E4 is not tested before the run" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_with_a_value_for_each_output_a_wrong_output_is_still_caught_and_repaired(tmp_path: Path) -> None:
    engine, model, state = _setup(tmp_path, PER_OUTPUT, None, code=WRONG)
    model.code = CODE  # the repair gives the right function
    assert await engine._equation_gate(state, None, None) is None
    assert len(model.other) == 1 and model.revisions == []
    assert "ln_A_hat" in model.other[0] and "8314" not in model.other[0] and "6.0" not in model.other[0]
    assert "alpha - beta * mean_invT" in (engine.quest_root / "code" / "kinetics" / "model.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_wrong_function_returning_one_number_is_still_caught_with_a_one_value_example(tmp_path: Path) -> None:
    # Not testing applies only when the test cannot judge: a function that returns one number is held to the value as before.
    wrong_one = ("# E4\ndef arrhenius_parameters(alpha, beta, R, mean_invT):\n"
                 "    return alpha + beta * mean_invT\n")
    engine, model, state = _setup(tmp_path, ONE_VALUE, None, code=wrong_one)
    patch = await engine._equation_gate(state, None, None)
    assert patch is not None and patch["exec_result"]["returncode"] == 1
    assert len(model.other) == 2, "the function's repairs as before"
