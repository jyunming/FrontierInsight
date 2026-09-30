"""The checks against known answers held to one numeric form per kind, their numbers computed by the engine from a formula
of what the simulation returns, and a test run of them before the study (core/oracle_forms.py).

Two live quests: a power-conservation check that expected 1 (the ratio) while the simulation measured 0 (the violation),
and a Kimi quest whose expected values were a truncated 0.367879 within 1e-12 and step-size limits used as the answers at
a finite step."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import oracle_check as oc, oracle_forms as of, plan, trial_runner
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from core.execution import SharedInterpreterExecutor
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

# --- the formula language ------------------------------------------------------------------------------------------------


def test_a_formula_is_computed_from_the_returned_names_and_the_whitelisted_functions() -> None:
    values = {"P_out": 9.0, "P_in": 10.0, "a": -2.0}
    assert of.evaluate("abs(P_out - P_in) / P_in", values).value == pytest.approx(0.1)
    assert of.evaluate("sqrt(abs(a)) ** 2 + max(P_out, P_in, 3) - pi * 0", values).value == pytest.approx(12.0)
    assert of.evaluate("sum(P_out, P_in) / 2", values).value == pytest.approx(9.5)
    assert of.names("abs(P_out - P_in) / P_in + pi") == ["P_out", "P_in"]
    assert of.is_name("error") and not of.is_name("abs(error)")


@pytest.mark.parametrize("text, why", [
    ("__import__('os').system('x')", "calls something"),
    ("__import__('os')", "__import__()"),
    ("x.__class__", "a dot"),
    ("values[0]", "an index"),
    ("open('f')", "open()"),
    ("(lambda: 1)()", "calls something"),
    ("a if b else c", "if/else"),
    ("a < b", "a comparison"),
    ("'text'", "not a number"),
    ("True + 1", "not a number"),
    ("abs", "without calling it"),
    ("max(x, key=y)", "named argument"),
    ("[a, b]", "a list"),
    ("a & b", "BitAnd"),
    ("", "empty"),
    ("x" * 500, "longer than"),
])
def test_a_formula_outside_the_small_language_is_refused_before_anything_runs(text: str, why: str) -> None:
    assert why in (of.problem(text) or ""), (text, of.problem(text))
    got = of.evaluate(text, {"x": 1.0, "a": 1.0, "b": 2.0, "c": 3.0, "values": 1.0, "y": 1.0})
    assert got.value is None and got.unreadable


def test_arithmetic_that_gives_no_finite_number_is_named_and_a_huge_power_does_not_hang() -> None:
    assert of.evaluate("a / b", {"a": 1.0, "b": 0.0}).problem == "it divides by zero"
    assert "cannot be computed" in of.evaluate("log(a)", {"a": -1.0}).problem
    assert of.evaluate("9 ** 9 ** 9", {}).value is None  # a float power overflows at once, it never grows an integer
    missing = of.evaluate("abs(P_out - P_in)", {"P_out": 1.0})
    assert missing.missing == ["P_in"] and missing.value is None


# --- one numeric form per kind -------------------------------------------------------------------------------------------

CONSERVATION = {"name": "power_conservation", "kind": "invariant", "check": "output power equals input power",
                "expected": 1.0, "tolerance": 1e-6, "case": {"n": 4}, "measure": "P_out / P_in",
                "reference": "derivation: a lossless network gives P_out = P_in, so P_out / P_in = 1"}


def test_a_conservation_check_expecting_1_for_a_formula_is_rewritten_to_its_violation_expecting_0() -> None:
    got = of.enforce({"oracles": [CONSERVATION]})
    assert got.requests == [] and len(got.rewrites) == 1
    new = got.oracles[0]
    assert new["measure"] == "abs((P_out / P_in) - 1)" and new["expected"] == 0 and new["tolerance"] == 1e-6
    assert "verdict is the same as before" in got.rewrites[0]
    # The verdict is the same before and after for any measured ratio.
    for ratio in (1.0, 1.0 + 5e-7, 1.0 + 2e-6, 0.0):
        values = {"P_out": ratio, "P_in": 1.0}
        before = abs(of.evaluate(CONSERVATION["measure"], values).value - 1.0) <= 1e-6
        after = abs(of.evaluate(new["measure"], values).value - 0.0) <= 1e-6
        assert before == after, ratio


def test_a_relative_tolerance_becomes_the_relative_violation_and_a_negative_expected_value_keeps_its_sign() -> None:
    rel = of.enforce({"oracles": [{**CONSERVATION, "expected": 2.0, "tolerance": 0.01, "tolerance_mode": "relative"}]})
    assert rel.oracles[0]["measure"] == "abs((P_out / P_in) - 2) / 2" and rel.oracles[0]["tolerance_mode"] == "absolute"
    neg = of.enforce({"oracles": [{**CONSERVATION, "kind": "symmetry", "expected": -3.5}]})
    assert neg.oracles[0]["measure"] == "abs((P_out / P_in) + 3.5)"


@pytest.mark.parametrize("change, word", [
    ({"measure": "ratio"}, "does not say whether that number is the quantity"),  # a bare name: the live failure
    ({"case": None, "measure": None}, "script's own oracle() decides"),
])
def test_a_violation_check_whose_representation_is_not_stated_goes_back_to_the_plan(change: dict[str, Any], word: str) -> None:
    oracle = {k: v for k, v in {**CONSERVATION, **change}.items() if v is not None}
    got = of.enforce({"oracles": [oracle]})
    assert got.rewrites == [] and len(got.requests) == 1 and word in got.requests[0]
    assert "expects 1" in got.requests[0] and "`expected: 0`" in got.requests[0]
    assert got.oracles[0]["expected"] == 1.0, "nothing is changed without the plan"


def test_a_check_whose_value_a_criterion_uses_is_not_rewritten() -> None:
    protocol = {"oracles": [CONSERVATION], "criteria": [{"name": "kept", "oracle": "power_conservation", "use": "value"}]}
    got = of.enforce(protocol)
    assert got.rewrites == [] and "criteria 'kept' are judged on its number" in got.requests[0]
    # A relative tolerance would change the scale of any criterion on the check, whatever it uses.
    relative = {**CONSERVATION, "expected": 2.0, "tolerance_mode": "relative"}
    got = of.enforce({"oracles": [relative], "criteria": [{"name": "err", "oracle": "power_conservation"}]})
    assert got.rewrites == [] and "'err'" in got.requests[0]


@pytest.mark.parametrize("oracle, why", [
    ({**CONSERVATION, "measure": "l2-ratio"}, "written with hyphens"),  # a returned key, not a subtraction
    ({**CONSERVATION, "measure": "P_out" + " " * 385 + "/ P_in"}, "too long"),
])
def test_a_rewrite_that_could_change_what_is_read_is_sent_back_instead(oracle: dict[str, Any], why: str) -> None:
    got = of.enforce({"oracles": [oracle]})
    assert got.rewrites == [] and why in got.requests[0], got.requests


def test_a_quest_where_the_script_reports_its_checks_is_never_rewritten() -> None:
    got = of.enforce({"oracles": [CONSERVATION]}, fi_runs=False)
    assert got.rewrites == [] and "does not run this quest's checks itself" in got.requests[0]


def test_whole_numbers_and_short_binary_fractions_are_exact_as_written() -> None:
    for exact in (101325.0, 12345.0, 299792458.0, 0.015625):
        oracle = {"name": "x", "kind": "published_value", "expected": exact, "tolerance": 1e-12, "case": {}, "measure": "v"}
        assert of.enforce({"oracles": [oracle]}).requests == [], exact


def test_a_formula_can_never_give_a_complex_number_or_span_lines() -> None:
    assert of.evaluate("a ** 0.5", {"a": -4.0}).value is None
    assert of.evaluate("abs(a ** 0.5)", {"a": -4.0}).value is None, "a negative under a root is not its magnitude"
    assert of.evaluate("(a ** 0.5) * 0 + 1", {"a": -4.0}).problem
    assert of.evaluate("a ** 0.5", {"a": 4.0}).value == 2.0
    assert "more than one line" in of.problem("(P_out # c\n - P_in)")


def test_the_kimi_truncated_expected_value_with_a_tolerance_below_its_precision_goes_back() -> None:
    kimi = {"name": "decay at t=1", "kind": "special_case", "expected": 0.367879, "tolerance": 1e-12,
            "case": {"dt": 0.1}, "measure": "y_end"}
    got = of.enforce({"oracles": [kimi]})
    assert len(got.requests) == 1 and "6 significant digits" in got.requests[0] and "full precision" in got.requests[0]
    # An exact value, a tolerance the written precision allows, or a value given in full is left alone.
    for fine in ({**kimi, "expected": 0.5}, {**kimi, "tolerance": 1e-6}, {**kimi, "expected": 0.36787944117144233}):
        assert of.enforce({"oracles": [fine]}).requests == [], fine


def test_the_quantity_kinds_keep_their_form_and_an_old_error_measure_expecting_0_still_passes() -> None:
    old = {"name": "rk4 error", "kind": "closed_form", "expected": 0.0, "tolerance": 1e-5, "case": {"dt": 0.1},
           "measure": "error"}
    assert of.enforce({"oracles": [old]}).requests == [] and of.enforce({"oracles": [old]}).rewrites == []
    assert "observed order itself" in of.enforce({"oracles": [{**old, "kind": "convergence_rate"}]}).requests[0]
    assert of.enforce({"oracles": [{**old, "kind": "convergence_rate", "expected": 4.0, "tolerance": 0.2}]}).requests == []


def test_a_formula_that_cannot_be_read_goes_back_to_the_plan() -> None:
    got = of.enforce({"oracles": [{**CONSERVATION, "expected": 0.0, "measure": "abs(P.out - P.inp)"}]})
    assert len(got.requests) == 1 and "cannot be read" in got.requests[0] and "a dot" in got.requests[0]


# --- the engine computes the formula on a toy run_cell -------------------------------------------------------------------

TOY = '''
def run_cell(cell):
    n = cell["n"]
    p_in = float(n)
    return {"P_in": p_in, "P_out": p_in * (1.0 - 1e-9), "violation": 0.0, "zero": 0.0}
'''


def _measure(tmp_path: Path, oracles: list[dict[str, Any]], source: str = TOY) -> tuple[list[dict[str, Any]], list[str]]:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    checks, problems, _ = asyncio.run(trial_runner.measure_oracles(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", oracles, timeout_s=60))
    return checks, problems


def test_the_engine_applies_the_formula_to_what_the_simulation_returned_and_old_names_still_work(tmp_path: Path) -> None:
    rewritten = of.enforce({"oracles": [CONSERVATION]}).oracles[0]
    old = {"name": "old style", "kind": "invariant", "expected": 0.0, "tolerance": 1e-6, "case": {"n": 4},
           "measure": "violation"}
    checks, problems = _measure(tmp_path, [rewritten, old])
    assert problems == []
    by = {c["name"]: c for c in checks}
    assert by["power_conservation"]["value"] == pytest.approx(1e-9) and by["power_conservation"]["measured_by"] == "engine"
    assert by["power_conservation"]["computed_as"] == rewritten["measure"]
    assert set(by["power_conservation"]["returned"]) == {"P_out", "P_in", "violation", "zero"}  # all it returned
    assert by["old style"]["value"] == 0.0 and "computed_as" not in by["old style"]
    judged = oc.judged([rewritten, old], {"checks": checks, "engine_measured": True})
    assert oc.engine_passed(judged) == ["power_conservation", "old style"]


def test_a_formula_the_simulation_cannot_answer_is_named_and_one_that_divides_by_zero_is_a_definition_problem(
    tmp_path: Path,
) -> None:
    _, problems = _measure(tmp_path / "a", [{**CONSERVATION, "measure": "abs(P_out - P_lost)", "expected": 0.0}])
    assert len(problems) == 1 and "'P_lost', which the simulation did not return" in problems[0]
    checks, problems = _measure(tmp_path / "b", [{**CONSERVATION, "measure": "violation / zero", "expected": 0.0}])
    assert "divides by zero" in problems[0] and checks[0]["formula_problem"] == "it divides by zero"
    assert "divides by zero" in (of.mismatches([CONSERVATION], checks)[0])


# --- a test run of the checks before the study ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle, value, word", [
    ({**CONSERVATION, "measure": "violation"}, 0.0, "reads as the violation"),  # the live failure
    ({"name": "x", "kind": "invariant", "expected": 0.0, "tolerance": 1e-6, "measure": "ratio", "case": {}}, 1.0,
     "reads as a ratio"),
    ({"name": "x", "kind": "special_case", "expected": 0.3679, "tolerance": 1e-3}, 36.79, "orders of magnitude"),
    ({"name": "x", "kind": "published_value", "expected": 2.5, "tolerance": 0.1}, 0.0, "exactly 0"),
    ({"name": "x", "kind": "special_case", "expected": -1.2, "tolerance": 0.1}, 1.2, "the sign the other way"),
])
def test_a_test_run_number_whose_size_says_the_definitions_differ_is_named(oracle: dict[str, Any], value: float, word: str) -> None:
    assert word in (of.mismatch(oracle, value) or ""), of.mismatch(oracle, value)


@pytest.mark.parametrize("oracle, value", [
    ({**CONSERVATION, "measure": "violation"}, 1.0 + 1e-7),  # it passes
    ({"name": "x", "kind": "special_case", "expected": 0.3679, "tolerance": 1e-3}, 0.30),  # simply wrong: a repair's to explain
    ({"name": "x", "kind": "invariant", "expected": 0.0, "tolerance": 1e-6, "measure": "abs(a / b - 1)", "case": {}}, 1.0),
    ({"name": "x", "kind": "special_case", "expected": 0.0, "tolerance": 1e-6}, 5.0),  # no scale to compare with
])
def test_a_plain_failure_is_not_read_as_a_definition_mismatch(oracle: dict[str, Any], value: float) -> None:
    assert of.mismatch(oracle, value) is None


def test_the_request_after_a_test_run_never_asks_for_the_measured_number() -> None:
    text = of.dry_run_request(["the check 'x' expects 1, but its test run measured 0"])
    assert "Never set an expected value to the number measured here" in text and "leave the check exactly" in text


# --- the engine, at plan time and at the test run -----------------------------------------------------------------------


def _config(tmp_path: Path, plan_pause: str = "off", **engine: Any) -> Config:
    return Config(
        topic="power through a lossless network", title="forms", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False, **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(plan=plan_pause, papers=False),
    )


CRITERION = {"name": "conserved", "oracle": "power_conservation", "direction": "lower", "target": 1e-6, "tolerance": 1e-7}


class _Model:
    """The plan's design carries ``protocol``; a plan revision applies ``revise`` to the oracles in the plan it is given."""

    def __init__(self, protocol: dict[str, Any], revise: Any = None) -> None:
        self.protocol, self.revise, self.revisions = protocol, revise, []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if "You are revising the plan" in prompt:
            self.revisions.append(prompt)
            current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
            if self.revise is None:
                return current
            edited = plan.edit_design_block(current, lambda block: {**block, "protocol": {
                **block["protocol"], "oracles": [self.revise(o) for o in block["protocol"]["oracles"]]}})
            return edited or current
        if _classify(prompt) == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = self.protocol
            body["plan"] = {"in_short": "x", "literature": [], "gap": "g", "success_criteria": ["s"], "risks": ["r"],
                            "out_of_scope": ["o"]}
            return json.dumps(body)
        return _fake_response_for(prompt)

    async def aclose(self) -> None:
        return None


def _plan_text(engine: Engine) -> str:
    return plan.plan_path(engine.quest_root).read_text(encoding="utf-8")


def _oracle_in_plan(engine: Engine) -> dict[str, Any]:
    return plan.parse(_plan_text(engine)).design["protocol"]["oracles"][0]


@pytest.mark.asyncio
async def test_the_plan_step_rewrites_an_unambiguous_check_and_says_so_in_plain_words(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model({"grid": {"n": [4, 8]}, "oracles": [CONSERVATION], "criteria": [CRITERION]})
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    oracle = _oracle_in_plan(engine)
    assert oracle["measure"] == "abs((P_out / P_in) - 1)" and oracle["expected"] == 0
    assert model.revisions == [], "a rewrite that cannot change the verdict needs no model call"
    text = _plan_text(engine)
    section = text.split(f"## {of.HEADING}", 1)[1].split("\n## ", 1)[0]
    assert "worst violation and it expects 0" in section and "verdict is the same as before" in section
    assert text.index(f"## {of.HEADING}") < text.index(f"## {plan.DESIGN_HEADING}")
    assert "computed as `abs((P_out / P_in) - 1)`" in text  # how the number is computed, above the block
    assert [r["by"] for r in plan.history(engine.quest_root)] == ["model", "engine"]
    # Once: a later pass through the plan step does not look again.
    before = _plan_text(engine)
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert _plan_text(engine) == before


@pytest.mark.asyncio
async def test_the_plan_step_sends_a_bare_name_conservation_check_back_once_with_a_precise_request(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    bare = {**CONSERVATION, "measure": "ratio"}

    def fix(oracle: dict[str, Any]) -> dict[str, Any]:
        return {**oracle, "measure": "abs(ratio - 1)", "expected": 0}

    model = _Model({"grid": {"n": [4, 8]}, "oracles": [bare], "criteria": [CRITERION]}, revise=fix)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.revisions) == 1
    asked = model.revisions[0]
    assert "'power_conservation'" in asked and "expects 1" in asked and "`expected: 0`" in asked
    assert _oracle_in_plan(engine)["measure"] == "abs(ratio - 1)" and _oracle_in_plan(engine)["expected"] == 0
    section = _plan_text(engine).split(f"## {of.HEADING}", 1)[1].split("\n## ", 1)[0]
    assert "FI asked the plan to change a check" in section and "expected 1.0 → 0" in section
    assert "Still not in its kind's form" not in section


@pytest.mark.asyncio
async def test_a_check_the_plan_does_not_fix_is_left_for_the_person_with_a_plain_note(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    kimi = {"name": "decay at t=1", "kind": "special_case", "expected": 0.367879, "tolerance": 1e-12,
            "case": {"n": 4}, "measure": "P_out", "reference": "derivation: y(1) = exp(-1) = 0.367879"}
    model = _Model({"grid": {"n": [4, 8]}, "oracles": [kimi],
                    "criteria": [{**CRITERION, "oracle": "decay at t=1"}]})  # the revision changes nothing
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.revisions) == 1, "asked once, never in a loop"
    section = _plan_text(engine).split(f"## {of.HEADING}", 1)[1].split("\n## ", 1)[0]
    assert "The plan did not change the checks." in section and "Still not in its kind's form" in section
    assert _oracle_in_plan(engine)["expected"] == 0.367879, "no expected value is changed by FI itself"
    assert "still not in its kind's form" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


TOY_VIOLATION = '''
def run_cell(cell):
    return {"violation": 0.0, "P_in": 1.0, "P_out": 1.0}
'''


async def _gate(engine: Engine, protocol: dict[str, Any], source: str) -> None:
    design = {"hypothesis": "power is conserved", "protocol": protocol}
    text = plan.render(engine.config.topic, {}, design)
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="user")
    seed = engine.quest_root / "code" / "simulate.py"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(source, encoding="utf-8")
    engine.executor = SharedInterpreterExecutor(python_version="3.11")
    engine._trial_mode = True
    await engine._oracle_gate({"topic": engine.config.topic, "iteration": 0, "design": design}, sys.executable, None, seed)


@pytest.mark.asyncio
async def test_the_test_run_flags_expected_1_against_measured_0_and_sends_it_back_before_the_freeze(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path, plan_pause="ask", oracle_check="warn"))
    person_edited = {**CONSERVATION, "measure": "violation"}  # written after the plan step looked (a person's edit)

    def fix(oracle: dict[str, Any]) -> dict[str, Any]:
        return {**oracle, "expected": 0}

    model = _Model({}, revise=fix)
    engine._client = model
    await _gate(engine, {"grid": {"n": [4]}, "oracles": [person_edited]}, TOY_VIOLATION)
    assert len(model.revisions) == 1
    asked = model.revisions[0]
    assert "expects 1, but its test run measured 0" in asked and "Never set an expected value" in asked
    record = json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok" and len(record["attempts"]) == 2
    assert record["attempts"][0]["test_run_mismatches"] and not record["attempts"][1]["problems"]
    assert any(r.get("repair") for r in record["attempts"]) is False, "no repair of the script was spent on it"
    assert (engine.fi_dir / "oracle_dry_run.json").is_file()
    assert "A test run of the checks before the study" in _plan_text(engine)
    added = engine._oracles_added_read()
    assert added["oracles"] == ["power_conservation"] and added["shown"] is False and "test run" in added["reason"]
    # The change makes the test run's own number pass: said as exactly that, so the reason is checked, not the result.
    assert "pass on the test run's own numbers" in added["reason"]
    assert "passes on the test run's own numbers" in _plan_text(engine)
    # The person reads the change before the freeze (pauses.plan: ask), in plain words.
    with pytest.raises(Exception):
        engine._hold_added_oracles()
    card = (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "FI changed checks against known answers" in card and "test run of the checks" in card
    assert engine._oracles_added_read()["shown"] is True


@pytest.mark.asyncio
async def test_a_plain_failure_goes_to_the_usual_repair_and_the_test_run_asks_nothing(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path, oracle_check="warn"))
    model = _Model({})
    engine._client = model
    wrong = {"name": "violation small", "kind": "invariant", "expected": 0.0, "tolerance": 1e-6, "case": {"n": 4},
             "measure": "abs(P_out - P_in) + 0.5"}
    await _gate(engine, {"grid": {"n": [4]}, "oracles": [wrong]}, TOY_VIOLATION)
    assert model.revisions == [] and not (engine.fi_dir / "oracle_dry_run.json").exists()


@pytest.mark.asyncio
async def test_a_check_the_plan_removes_after_the_test_run_is_named_to_the_person(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path, plan_pause="ask", oracle_check="warn"))
    keep = {"name": "kept", "kind": "invariant", "expected": 0.0, "tolerance": 1e-6, "case": {"n": 4},
            "measure": "violation"}

    class _Removing(_Model):
        async def chat(self, messages, **kw):  # noqa: ANN001
            prompt = messages[-1]["content"]
            if "You are revising the plan" in prompt:
                self.revisions.append(prompt)
                current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
                return plan.edit_design_block(current, lambda b: {**b, "protocol": {
                    **b["protocol"], "oracles": [keep]}})
            return await super().chat(messages, **kw)

    model = _Removing({})
    engine._client = model
    await _gate(engine, {"grid": {"n": [4]}, "oracles": [{**CONSERVATION, "measure": "violation"}, keep]}, TOY_VIOLATION)
    added = engine._oracles_added_read()
    assert added["removed"] == ["power_conservation"] and added["shown"] is False
    record = json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert record["attempts"][-1]["oracles"] == ["kept"], "measured again as the plan now states them"
    with pytest.raises(Exception):
        engine._hold_added_oracles()
    assert "power_conservation (removed)" in (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_revision_for_the_checks_can_change_only_the_checks(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path, oracle_check="warn"))

    class _Wandering(_Model):
        async def chat(self, messages, **kw):  # noqa: ANN001
            prompt = messages[-1]["content"]
            if "You are revising the plan" in prompt:
                self.revisions.append(prompt)
                current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
                return plan.edit_design_block(current, lambda b: {**b, "hypothesis": "a different claim", "protocol": {
                    **b["protocol"], "grid": {"n": [99]},
                    "oracles": [{**o, "expected": 0} for o in b["protocol"]["oracles"]]}})
            return await super().chat(messages, **kw)

    model = _Wandering({})
    engine._client = model
    await _gate(engine, {"grid": {"n": [4]}, "oracles": [{**CONSERVATION, "measure": "violation"}]}, TOY_VIOLATION)
    design = plan.load_design(engine.quest_root)[0]
    assert design["protocol"]["grid"] == {"n": [4]} and design["hypothesis"] == "power is conserved"
    assert design["protocol"]["oracles"][0]["expected"] == 0
    assert "only the checks were kept" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_revision_call_that_fails_is_said_and_never_crashes_the_step(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))

    class _Failing(_Model):
        async def chat(self, messages, **kw):  # noqa: ANN001
            if "You are revising the plan" in messages[-1]["content"]:
                raise TimeoutError("the provider did not answer")
            return await super().chat(messages, **kw)

    engine._client = _Failing({"grid": {"n": [4, 8]}, "oracles": [{**CONSERVATION, "measure": "ratio"}],
                               "criteria": [CRITERION]})
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    section = _plan_text(engine).split(f"## {of.HEADING}", 1)[1].split("\n## ", 1)[0]
    assert "could not be rewritten" in section and "Still not in its kind's form" in section
    assert json.loads((engine.fi_dir / "oracle_guidance.json").read_text(encoding="utf-8"))["forms"]["asked"]


@pytest.mark.asyncio
async def test_a_one_script_quest_is_asked_not_rewritten_at_the_plan_step(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.execution.split_analysis = False
    engine = Engine(cfg)
    model = _Model({"grid": {"n": [4, 8]}, "oracles": [CONSERVATION]})
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert _oracle_in_plan(engine)["measure"] == "P_out / P_in" and len(model.revisions) == 1


@pytest.mark.asyncio
async def test_a_plan_read_at_the_old_plan_stop_shows_what_the_look_changed_before_the_freeze(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path, plan_pause="ask"))
    design = {"hypothesis": "power is conserved", "protocol": {"grid": {"n": [4]}, "oracles": [CONSERVATION],
                                                               "criteria": [CRITERION]}}
    text = plan.render(engine.config.topic, {}, design)
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="model")
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    (engine.fi_dir / "paused_at_plan.flag").write_text("plan", encoding="utf-8")  # read before this version of FI
    engine._client = _Model({})
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert _oracle_in_plan(engine)["expected"] == 0
    added = engine._oracles_added_read()
    assert added["oracles"] == ["power_conservation"] and added["shown"] is False and "after you read" in added["reason"]


def test_a_formula_that_takes_nothing_from_the_simulation_is_not_a_measurement(tmp_path: Path) -> None:
    for constant in ("0", "abs(pi - pi)"):
        got = of.enforce({"oracles": [{**CONSERVATION, "expected": 0.0, "measure": constant}]})
        assert "takes nothing the simulation returns" in got.requests[0], constant
    checks, problems = _measure(tmp_path, [{**CONSERVATION, "expected": 0.0, "measure": "0"}])
    assert checks == [] and "not a measurement of the simulation" in problems[0]


def test_every_record_of_an_engine_change_keeps_what_is_not_yet_shown(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    engine._note_engine_change(["a"], ["gone"], reason="first reason.")
    engine._note_engine_change(["b"])  # the gate adding a check afterwards, as _declare_oracles does
    added = engine._oracles_added_read()
    assert added["oracles"] == ["a", "b"] and added["removed"] == ["gone"] and added["reason"] == "first reason."
    engine._oracles_added_write({**added, "shown": True})
    engine._note_engine_change(["c"])
    fresh = engine._oracles_added_read()
    assert fresh["oracles"] == ["c"] and fresh["removed"] == [] and "reason" not in fresh


@pytest.mark.asyncio
async def test_a_criterion_that_reads_a_changed_check_may_change_with_it_and_no_other(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    reads_value = {"name": "kept", "oracle": "power_conservation", "use": "value", "direction": "target", "target": 1.0,
                   "tolerance": 1e-6}
    other = {"name": "other", "oracle": "rk", "direction": "lower", "target": 2e-6, "tolerance": 1e-7}
    rk = {"name": "rk", "kind": "special_case", "expected": 0.0, "tolerance": 1e-6, "case": {"n": 4}, "measure": "err"}

    class _Both(_Model):
        async def chat(self, messages, **kw):  # noqa: ANN001
            prompt = messages[-1]["content"]
            if "You are revising the plan" in prompt:
                self.revisions.append(prompt)
                current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()

                def change(b: dict[str, Any]) -> dict[str, Any]:
                    oracles = [{**o, "measure": "abs(ratio - 1)", "expected": 0} if o["name"] == "power_conservation" else o
                               for o in b["protocol"]["oracles"]]
                    criteria = [{**c, "direction": "lower", "target": 1e-6} for c in b["protocol"]["criteria"]]
                    return {**b, "protocol": {**b["protocol"], "oracles": oracles, "criteria": criteria}}

                return plan.edit_design_block(current, change)
            return await super().chat(messages, **kw)

    model = _Both({"grid": {"n": [4, 8]}, "oracles": [{**CONSERVATION, "measure": "ratio"}, rk],
                   "criteria": [reads_value, other]})
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    criteria = {c["name"]: c for c in plan.load_design(engine.quest_root)[0]["protocol"]["criteria"]}
    assert criteria["kept"]["direction"] == "lower", "the criterion on the changed check changed with it"
    assert criteria["other"]["target"] == 2e-6, "a criterion on an unchanged check is put back"
    assert "change those criteria too" in model.revisions[0]
