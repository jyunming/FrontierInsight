"""The code writer sees the topic: every request that writes or repairs code is shown the topic text (minus a result it
states) and the formulas it states, alongside the plan's equations, and none of a check's expected values. Fake models only,
a neutral topic (a cup of coffee cooling under different lids)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import topic_requirements as tr
from tests.test_topic_requirements import EXTRACTION, TOPIC


def test_the_topic_a_code_request_sees_has_no_result_in_it_and_keeps_the_formulas() -> None:
    shown = tr.code_block(TOPIC, EXTRACTION)
    assert "dT/dt = -k*(T - Tr)" in shown and "three lid materials" in shown
    assert "0.05" not in shown, "the topic's stated result is not shown to a request that writes code"
    assert "expected value" not in shown
    assert "Formulas the topic states" in shown and "Newton's law of cooling" in shown
    assert tr.code_block("", None) == "(the topic is not available)"
    # Without the reader's list, a sentence that says what a result should be is still left out.
    plain = tr.topic_for_code("Fit the curve. The reference value is 3.7. The result should equal 12.")
    assert plain == "Fit the curve."


def test_the_function_steps_prompts_carry_the_topic(tmp_path: Path) -> None:
    import string

    from core.engine import Engine  # noqa: F401

    agents = Path(__file__).resolve().parent.parent / "agents"
    for name, kwargs in {
        "implement_function": dict(model_block="m", equation_block="E1: y", function_block="def f(a):", related_block="(none)",
                                  constants_block="(none)", package="p"),
        "implement_function_repair": dict(equation_block="E1: y", function_block="def f(a): pass", problem="p",
                                          related_block="(none)", package="p"),
        "implement_module_repair": dict(passing_block="(none)", failing_block="f", module="x = 1", package="p"),
        "execute_reflect": None, "implement": None, "implement_outline": None, "implement_body": None, "improve": None,
    }.items():
        template = string.Template((agents / f"{name}.md").read_text(encoding="utf-8"))
        assert "$topic_block" in template.template, f"{name}.md does not show the topic"
        if kwargs is not None:
            assert "MARKER-TOPIC" in template.substitute(topic_block="MARKER-TOPIC", **kwargs)


@pytest.mark.asyncio
async def test_a_whole_fake_quests_code_prompts_all_show_the_topic_and_none_a_check_s_expected_value(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _FAKE_EXPERIMENT_CODE, _FAKE_RESPONSES, _classify, _fake_response_for
    from tests.test_run_manifest import _cfg

    secret_check, secret_example = "0.31416", "2.71828"
    protocol = {
        "grid": {"room_temperature_C": [10, 20, 30], "lid_material": ["paper", "foam", "steel"]}, "runs_per_setting": 1,
        "oracles": [{"name": "cooling_limit", "kind": "special_case", "check": "the cup ends at the room temperature",
                     "expected": float(secret_check), "tolerance": 0.01, "reference": "derivation: the long-time limit"}],
        "model": {"summary": "a cup cooling", "equations": [
            {"id": "E1", "formula": "T(t) = Tr + (T0 - Tr)*exp(-k*t)", "role": "generates",
             "source": "derivation: the solution of the cooling law",
             "example": {"inputs": {"Tr": 20, "T0": 90, "k": 0.05, "t": 10},
                         "expected_formula": f"{secret_example} + 0*exp(0)"}}]},
    }
    crash_first = ("import os, sys\nif not os.path.exists('crashed_once'):\n    open('crashed_once', 'w').write('1')\n"
                   "    sys.exit(3)\n" + _FAKE_EXPERIMENT_CODE)
    seen: list[tuple[str, str]] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        seen.append((kind, prompt))
        if prompt.lstrip().startswith("# What a Research Topic Asks For"):
            return json.dumps(EXTRACTION)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return json.dumps({"code": crash_first, "deps": ["matplotlib"]})
        if kind == "ExecuteReflect":
            return json.dumps({"code": _FAKE_EXPERIMENT_CODE, "deps": ["matplotlib"], "patch_summary": "fixed"})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = _cfg(tmp_path, split=False)
    cfg.topic = TOPIC
    engine = Engine(cfg)
    await engine.run()
    code_prompts = [(k, p) for k, p in seen if k in ("Implementation", "ExecuteReflect")
                    or p.lstrip().startswith(("You are the **Function Body**", "You are repairing", "# Code That Does Not Do"))]
    kinds = {k for k, _p in code_prompts}
    assert "Implementation" in kinds and "ExecuteReflect" in kinds, f"the quest should have written and repaired code: {kinds}"
    for kind, prompt in code_prompts:
        assert "three lid materials" in prompt, f"a {kind} request did not show the topic"
        assert "dT/dt = -k*(T - Tr)" in prompt, f"a {kind} request did not show the topic's formula"
        assert secret_check not in prompt and secret_example not in prompt, f"a {kind} request showed a check's expected value"
        assert "0.05 per minute" not in prompt, f"a {kind} request showed the topic's stated result"
