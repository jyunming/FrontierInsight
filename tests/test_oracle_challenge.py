"""The code questions the plan's own working of a known answer, the plan's model answers, a third model judges
(core/oracle_challenge.py), and FI computes every formula itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import accepted_checks, oracle_challenge as ch
from core.engine import Engine
from tests.test_oracle_gate import ORACLE, _BENT, _HEAD, _PROTOCOL, _TAIL, _cfg, _fake, _record

# The plan's working gives 1.0 (FI computes it from the plan's formula); the simulation measures 0.5.
_CHECK = {**ORACLE, "expected_formula": "0.6 + 0.4", "case": {"n": 4}}
_MEASURES_HALF = _HEAD + """\
if os.environ.get("FI_ORACLE") == "1":
    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "final size closed form", "value": 0.5}]}))
    raise SystemExit(0)
""" + _TAIL
# The repair says the plan added a term twice and gives the value as a formula FI can compute.
_QUESTION = [{"name": ORACLE["name"], "expected": 0.5, "tolerance": 0.01, "reason": "the second term is counted twice",
              "disputed_step": "the plan adds the second term twice", "expected_formula": "n / 8"}]


# --- the exchange itself ------------------------------------------------------------------------------------------------


def test_a_question_needs_the_step_and_a_formula_fi_computes_to_the_value_claimed() -> None:
    taken, why = ch.challenge_of(_CHECK, _QUESTION[0])
    assert taken is not None and taken["value"] == 0.5 and not why
    bare = {k: v for k, v in _QUESTION[0].items() if k not in ("disputed_step", "expected_formula")}
    assert ch.challenge_of(_CHECK, bare)[0] is None, "a number alone cannot be told apart from the measurement"
    assert ch.challenge_of(_CHECK, {**_QUESTION[0], "expected_formula": "0.5"})[0] is None, "a number restated"
    off, why = ch.challenge_of(_CHECK, {**_QUESTION[0], "expected_formula": "0.3 + 0.3"})
    assert off is None and "not the 0.5 it claims" in why


def test_the_plan_conceding_with_a_formula_that_gives_the_code_s_value_corrects_the_check() -> None:
    question = ch.challenge_of(_CHECK, _QUESTION[0])[0]
    assert question is not None
    conceded = ch.parse_plan({"answer": "concede", "expected_formula": "1 - 0.5", "why": "counted twice"})
    assert ch.settle(_CHECK, question, conceded, None)["outcome"] == "corrected"
    # A concession whose own formula FI computes to another value settles nothing by itself.
    wrong = ch.parse_plan({"answer": "concede", "expected_formula": "0.2 + 0.2"})
    assert ch.settle(_CHECK, question, wrong, None)["outcome"] == "unsettled"


def test_the_judge_decides_only_with_a_formula_fi_computes() -> None:
    question = ch.challenge_of(_CHECK, _QUESTION[0])[0]
    assert question is not None
    stands = ch.parse_plan({"answer": "stand", "code_fault": "the code halves the result", "why": "1.0 is right"})
    for_code = ch.parse_judge({"side": "code", "expected_formula": "0.25 * 2", "why": "counted twice"})
    assert ch.settle(_CHECK, question, stands, for_code)["outcome"] == "corrected"
    word_only = ch.parse_judge({"side": "code", "why": "the code is right"})
    assert ch.settle(_CHECK, question, stands, word_only)["outcome"] == "unsettled"
    for_plan = ch.parse_judge({"side": "plan", "expected_formula": "0.6 + 0.4", "why": "the code halves it"})
    assert ch.settle(_CHECK, question, stands, for_plan)["outcome"] == "code"
    assert ch.settle(_CHECK, question, ch.parse_plan("not json"), ch.parse_judge(None))["outcome"] == "unsettled"
    # Both may be wrong: the judge's own working gives a third value, so nothing is corrected and the code is not
    # sent back on the plan's word.
    third = ch.parse_judge({"side": "plan", "expected_formula": "0.3 + 0.4", "why": "both are off"})
    settled = ch.settle(_CHECK, question, stands, third)
    assert settled["outcome"] == "unsettled" and "may both be wrong" in settled["text"]


def test_a_disagreement_is_said_as_one_and_never_as_a_failure_of_the_code() -> None:
    entry = {"name": ORACLE["name"], "expected": 1.0, "limit": 0.05, "measured": 0.5, "by": accepted_checks.AUTOMATIC,
             "disagreement": "the simulation and the plan's own working disagree on the check 'x' and neither is "
                             "confirmed: the code: counted twice; the plan: it is right"}
    said = accepted_checks.failing_disclosure({"status": accepted_checks.WENT_ON, "went_on": [entry]})
    assert "disagree, and neither is confirmed" in said and "FAILED" not in said
    assert "not a failure of the code" in said


# --- through the real gate ----------------------------------------------------------------------------------------------


def _keep(prompts: dict[str, list[str]] | None) -> dict[str, list[str]]:
    return prompts if prompts is not None else {}


def _exchange_fake(calls: list[str], *, plan: dict[str, Any], judge: dict[str, Any] | None = None,
                   second_repair: str | None = None, prompts: dict[str, list[str]] | None = None):
    base = _fake(calls, implement=_MEASURES_HALF, protocol={**_PROTOCOL, "oracles": [_CHECK]})

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if prompt.startswith("# Answer an Objection to Your Known Answer"):
            calls.append("PlanAnswers")
            _keep(prompts).setdefault("plan", []).append(prompt)
            return json.dumps(plan)
        if prompt.startswith("# Judge Between Two Workings of a Known Answer"):
            calls.append("Judge")
            _keep(prompts).setdefault("judge", []).append(prompt)
            return json.dumps(judge or {"side": "undecided"})
        if "FI_ORACLE=1 to check its oracles" in prompt:
            calls.append("OracleRepair")
            _keep(prompts).setdefault("repair", []).append(prompt)
            if calls.count("OracleRepair") == 1:
                return json.dumps({"code": "", "deps": [], "patch_summary": "the check is wrong",
                                   "oracle_change": _QUESTION})
            return json.dumps({"code": second_repair or "", "deps": [], "patch_summary": "fixed the halving"})
        return await base(self, messages, **kw)

    return fake_chat


@pytest.mark.asyncio
async def test_the_plan_that_concedes_corrects_its_check_and_no_second_repair_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    prompts: dict[str, list[str]] = {}
    monkeypatch.setattr("core.engine.LLMClient.chat", _exchange_fake(
        calls, plan={"answer": "concede", "expected_formula": "1 - 0.5", "why": "the term was counted twice"},
        prompts=prompts))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=2))
    await engine.run()
    record = _record(engine)
    assert record["status"] == "ok", record.get("status")
    assert calls.count("OracleRepair") == 1 and calls.count("PlanAnswers") == 1 and "Judge" not in calls
    assert "0.5" not in prompts["plan"][0].split("# The objection", 1)[0], "the plan is never shown the measurement"
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == _MEASURES_HALF


@pytest.mark.asyncio
async def test_the_plan_and_a_judge_that_name_a_fault_in_the_code_send_it_back_to_be_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    prompts: dict[str, list[str]] = {}
    monkeypatch.setattr("core.engine.LLMClient.chat", _exchange_fake(
        calls, plan={"answer": "stand", "code_fault": "the code halves the result", "why": "the plan's 1.0 is right"},
        judge={"side": "plan", "expected_formula": "0.6 + 0.4", "why": "the code halves it"},
        second_repair=_BENT, prompts=prompts))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=2))
    await engine.run()
    record = _record(engine)
    assert record["status"] == "ok", record.get("status")
    assert calls.count("OracleRepair") == 2 and calls.count("Judge") == 1
    assert "the code halves the result" in prompts["repair"][1], "the second repair is told why"


@pytest.mark.asyncio
async def test_an_exchange_nobody_settles_goes_on_unconfirmed_and_is_said_as_a_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _exchange_fake(
        calls, plan={"answer": "stand", "why": "the plan's 1.0 is right"},
        judge={"side": "undecided", "why": "cannot tell"}))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=2, oracle_check="warn"))
    await engine.run()
    record = _record(engine)
    assert calls.count("OracleRepair") == 1, "no second repair once the exchange is set aside"
    assert record.get("set_aside", {}).get(ORACLE["name"], "").startswith("the simulation and the plan's own working disagree")
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == _MEASURES_HALF


# --- the numerical referee ----------------------------------------------------------------------------------------------


def test_the_numerical_script_must_compute_from_the_definitions_alone() -> None:
    script, why = ch.numeric_script("```python\nimport numpy as np\nprint('NUMERIC_JSON: {\"value\": 1}')\n```")
    assert script and not why
    for bad in ("import simulate", "from code.model import f", "import sys; sys.path.insert(0, 'code')"):
        assert ch.numeric_script(f"```python\n{bad}\n```")[0] == "", bad
    assert ch.numeric_value('noise\nNUMERIC_JSON: {"value": 0.75, "how": "a sum of 10^6 terms"}') == (0.75, "a sum of 10^6 terms")
    assert ch.numeric_value("nothing")[0] is None


def test_the_number_computed_numerically_decides_and_can_find_both_wrong() -> None:
    question = ch.challenge_of(_CHECK, _QUESTION[0])[0]
    assert question is not None
    judge = ch.parse_judge({"side": "undecided", "expected_formula": "0.5 + 0.25"})
    before = {"outcome": "unsettled", "by": "", "text": "nobody settled it"}
    assert ch.settle_numeric(_CHECK, question, judge, 0.5, "grid", before)["outcome"] == "corrected"
    assert ch.settle_numeric(_CHECK, question, judge, 1.0, "grid", before)["outcome"] == "code"
    both = ch.settle_numeric(_CHECK, question, judge, 0.75, "grid", before)
    assert both["outcome"] == "both" and both["expected"] == 0.75 and "both wrong" in both["text"]
    assert ch.settle_numeric(_CHECK, question, judge, 0.9, "grid", before)["outcome"] == "unsettled"
    assert ch.settle_numeric(_CHECK, question, judge, None, "it failed", before)["outcome"] == "unsettled"


_MEASURES_THREE_QUARTERS = _MEASURES_HALF.replace('"value": 0.5', '"value": 0.75')


@pytest.mark.asyncio
async def test_when_the_plan_and_the_code_are_both_wrong_the_number_computed_numerically_corrects_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    prompts: dict[str, list[str]] = {}
    base = _exchange_fake(calls, plan={"answer": "stand", "why": "the plan's 1.0 is right"},
                          judge={"side": "undecided", "expected_formula": "0.5 + 0.25", "why": "I get 0.75"},
                          second_repair=_MEASURES_THREE_QUARTERS, prompts=prompts)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if prompt.startswith("# Compute a Known Answer Numerically"):
            calls.append("Numeric")
            prompts.setdefault("numeric", []).append(prompt)
            return "```python\ns = sum(0.75 / 1000 for _ in range(1000))\nprint('NUMERIC_JSON: {\"value\": %r, \"how\": \"a sum\"}' % s)\n```"
        return await base(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=2))
    await engine.run()
    record = _record(engine)
    assert calls.count("Numeric") == 1 and calls.count("OracleRepair") == 2
    assert record["status"] == "ok", record.get("status")
    shown = prompts["numeric"][0]
    assert "0.6 + 0.4" not in shown and "n / 8" not in shown, "the numerical script sees neither working"
    assert "the plan's working and the code were both wrong" in prompts["repair"][1]


def test_a_formula_that_writes_the_measurement_down_is_no_working_and_the_step_never_shows_it() -> None:
    restated = {**_QUESTION[0], "expected_formula": "0.5 * 1"}
    assert ch.challenge_of(_CHECK, restated, measured=0.5)[0] is None, "numbers alone, the measured one among them"
    assert ch.challenge_of(_CHECK, {**_QUESTION[0], "expected_formula": "n * 0.125"}, measured=0.5)[0] is not None
    assert ch.challenge_of(_CHECK, {**_QUESTION[0], "expected_formula": "n * 0 + 0.5"}, measured=0.5)[0] is None
    taken, _ = ch.challenge_of(_CHECK, {**_QUESTION[0], "disputed_step": "the simulation gives 0.500, the plan 1.0"},
                               measured=0.5)
    assert taken is not None and "0.500" not in taken["step"] and "1.0" in taken["step"]


def test_a_concession_or_a_verdict_that_copies_the_codes_formula_settles_nothing_by_itself() -> None:
    question, _ = ch.challenge_of(_CHECK, _QUESTION[0])
    copied = ch.parse_plan({"answer": "concede", "expected_formula": "n/8", "why": "yes"})
    assert ch.settle(_CHECK, question, copied, None)["outcome"] == "unsettled"
    stand = ch.parse_plan({"answer": "stand", "code_fault": "", "why": "mine is right"})
    judge = ch.parse_judge({"side": "code", "expected_formula": " n / 8 ", "why": "the code is right"})
    result = ch.settle(_CHECK, question, stand, judge)
    assert result["outcome"] == "unsettled" and "copied" in result["text"]


def test_a_restated_number_is_found_by_whether_it_moves_with_the_case_not_by_its_digits() -> None:
    for restated in ("n*0 + 1/2", "pi*0 + 0.5", "n/n*0.5"):
        assert ch.computes(_CHECK, restated, measured=0.5) is None, restated
    # a working whose numbers happen to be near the measured value is still a working
    assert ch.computes(_CHECK, "1 - exp(-n*2)", measured=0.9997) is not None
    assert ch.numeric_script("```python\nglobal_x = 1\nprint(global_x)\n```")[0], "a name that holds a forbidden word"
