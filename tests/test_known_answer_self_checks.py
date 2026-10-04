"""Before a failing known-answer check stops a quest, FI finds out for itself whether the check or the script is wrong
(core/oracle_triage.py), and the repair budget is not spent rewriting a correct script.

The cases are the real stops of the investigation of 2026-10-01: an RK4 error expected at 1.637e-08 where the true one is
3.33241e-07 (about 20 times, below the gate's 100-times test-run threshold), Verlet's energy expected conserved to 1e-12
where the scheme's own error at h = 0.01 is h**2/8 = 1.25e-05, a percent written where a fraction was expected, a short
M/M/1 queue judged on one random trial, the same KeyError after a repair, a dispute forgotten on resume, and the
protocol check that counted the oracle's own 20000 runs as the main experiment's runs per setting. No real model is
called: the model is a fake that answers by the prompt's heading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import oracle_forms as of, oracle_triage as ot, plan, protocol_check as pc
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from core.execution import SharedInterpreterExecutor
from tests.test_engine_smoke import _fake_response_for

RECOMPUTE_HEAD = "Work Out a Known Answer Again"


class _Model:
    """Answers the recompute prompt with ``recompute`` (a dict, or text), a repair request with ``repair_code`` (no code
    when ``None``), and everything else as the smoke test's fake does."""

    def __init__(self, recompute: Any = None, repair_code: str | None = None) -> None:
        self.recompute, self.repair_code = recompute, repair_code
        self.recompute_prompts: list[str] = []
        self.repairs: list[str] = []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if RECOMPUTE_HEAD in prompt[:200]:
            self.recompute_prompts.append(prompt)
            return self.recompute if isinstance(self.recompute, str) else json.dumps(self.recompute or {})
        if "has NOT run its experiment yet" in prompt:
            self.repairs.append(prompt)
            return json.dumps({"code": self.repair_code or "", "patch_summary": "changed the integrator"})
        return _fake_response_for(prompt)

    async def aclose(self) -> None:
        return None


def _config(tmp_path: Path, **engine: Any) -> Config:
    return Config(
        topic="numerical methods checked against known answers", title="self-checks", provider=ProviderConfig(name="openai"),
        # A quest whose result is for a decision stops at a failed check; an exploration goes on by itself.
        result_use="decision",
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False, **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(plan="off", papers=False),
    )


def _write_plan(engine: Engine, oracles: list[dict[str, Any]], grid: dict[str, Any] | None = None) -> dict[str, Any]:
    design = {"hypothesis": "the method is right", "protocol": {"grid": grid or {"dt": [0.1]}, "oracles": oracles}}
    text = plan.render(engine.config.topic, {}, design)
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="user")
    return design


async def _gate(engine: Engine, oracles: list[dict[str, Any]], source: str) -> None:
    """One run of the oracle gate. It never stops to ask: it repairs, corrects or goes on (the record says which)."""
    design = _write_plan(engine, oracles)
    seed = engine.quest_root / "code" / "simulate.py"
    seed.parent.mkdir(parents=True, exist_ok=True)
    if not seed.is_file() or seed.read_text(encoding="utf-8") != source:
        seed.write_text(source, encoding="utf-8")
    engine.executor = SharedInterpreterExecutor(python_version="3.11")
    engine._trial_mode = True
    await engine._oracle_gate({"topic": engine.config.topic, "iteration": 0, "design": design}, sys.executable, None,
                              seed)


def _record(engine: Engine) -> dict[str, Any]:
    return json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))


def _explained(engine: Engine) -> dict[str, Any]:
    """Why FI went on, in plain words, as the record keeps it (``leaning``, ``why``)."""
    return _record(engine).get("explained") or {}


def _triage(record: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [t for a in record["attempts"] for t in a.get("triage") or [] if t["kind"] == kind]


RK4_SOURCE = '''
import math

def run_cell(cell):
    dt, t_end = float(cell["dt"]), float(cell.get("t_end", 1.0))
    y = 1.0
    for _ in range(int(round(t_end / dt))):
        k1 = -y; k2 = -(y + dt / 2 * k1); k3 = -(y + dt / 2 * k2); k4 = -(y + dt * k3)
        y += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return {"error": abs(y - math.exp(-t_end))}
'''
# The plan's expected value is wrong by about 20 times (the true RK4 error at h=0.1 on y' = -y to t=1 is 3.33241e-07).
RK4 = {"name": "rk4_closed_form_h01", "kind": "special_case", "check": "RK4 error on y' = -y at t = 1 with h = 0.1",
       "expected": 1.637e-08, "tolerance": 1e-09, "case": {"dt": 0.1, "t_end": 1.0}, "measure": "error",
       "reference": "derivation: the global error of RK4 is about h^4/120 times y(1), so at h = 0.1 it is 1.637e-08"}
# A repair that would "fix" the correct script towards the wrong number.
BENT = RK4_SOURCE.replace('return {"error": abs(y - math.exp(-t_end))}', 'return {"error": 1.637e-08}')


@pytest.mark.asyncio
async def test_rk4_expected_value_off_20x_is_disputed_by_a_blind_recompute_and_the_script_is_not_rewritten(
        tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(recompute={"expected": 3.3324e-07, "how": "the RK4 amplification factor at h=0.1 is "
                                                            "0.9048375; 0.9048375^10 - exp(-1) = 3.3324e-07"},
                   repair_code=BENT)
    engine._client = model
    await _gate(engine, [RK4], RK4_SOURCE)
    assert model.repairs == [], "no repair was spent on a correct script"
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == RK4_SOURCE
    assert len(model.recompute_prompts) == 1, "one call per failing check"
    asked = model.recompute_prompts[0]
    assert "3.33241" not in asked and "3.3324e-07" not in asked, "the measured value is never shown"
    record = _record(engine)
    assert record["status"] == "went_on_failing" and record["disputed"] == ["rk4_closed_form_h01"]
    (entry,) = _triage(record, "recompute")
    # The model that wrote the plan, asked again: said, never counted as independent evidence.
    assert entry["verdict"] == "disputed" and entry["points_to"] == "" and entry["same_model"] is True
    assert "same model that wrote the plan" in entry["tried"]
    proposal = record["proposed_changes"][0]
    assert proposal["source"] == "recompute" and proposal["expected"] == pytest.approx(3.3324e-07)
    assert proposal["tolerance"] == 1e-09, "the check's own tolerance: nothing is loosened"
    assert record["self_checks"]["model_calls"] == 1
    assert not (engine.fi_dir / "oracle_corrections.json").exists(), "the same model's value never corrects a check"
    assert plan.load_design(engine.quest_root)[0]["protocol"]["oracles"][0]["expected"] == 1.637e-08
    assert _explained(engine)["why"], "why FI went on, in plain words"
    assert "this run spent 1 model call(s)" in plan.plan_path(engine.quest_root).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_rk4_expected_value_off_20x_is_corrected_from_another_models_value_and_then_passes(
        tmp_path: Path) -> None:
    """With another model named for checking the checks, its value (worked out without seeing the result) is
    independent: FI corrects the plan's expected value to it, never to the measured value, keeps the tolerance, and
    measures again; the corrected check passes on the correct script."""
    engine = Engine(_config(tmp_path).model_copy(update={"provider": ProviderConfig(
        name="openai", model="planner", node_models={"oracle_review": "another-model"})}))
    model = _Model(recompute={"expected": 3.3324e-07, "how": "the RK4 amplification factor at h=0.1 is "
                                                            "0.9048375; 0.9048375^10 - exp(-1) = 3.3324e-07"},
                   repair_code=BENT)
    engine._client = model
    await _gate(engine, [RK4], RK4_SOURCE)
    assert model.repairs == [] and (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == RK4_SOURCE
    record = _record(engine)
    assert record["status"] == "ok", "the corrected check passes on the correct script"
    after = plan.load_design(engine.quest_root)[0]["protocol"]["oracles"][0]
    assert after["expected"] == pytest.approx(3.3324e-07) and after["expected"] != 3.33241e-07
    assert after["tolerance"] == 1e-09, "never loosened"
    corrected = json.loads((engine.fi_dir / "oracle_corrections.json").read_text(encoding="utf-8"))
    assert corrected["rk4_closed_form_h01"]["from"] == 1.637e-08 and corrected["rk4_closed_form_h01"]["source"] == "recompute"
    assert plan.history(engine.quest_root)[-1]["by"] == "engine"


@pytest.mark.asyncio
async def test_a_dispute_survives_resume_until_the_check_is_changed(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    engine._client = _Model(recompute={"expected": 3.3324e-07, "how": "h^4 term"}, repair_code=BENT)
    await _gate(engine, [RK4], RK4_SOURCE)
    # The gate runs again (a redo from the run): the dispute is read back from needs/ORACLE_CHECK.json, so the model is
    # not asked again and the correct script is not rewritten towards the disputed number.
    again = _Model(recompute={"expected": 1.637e-08, "how": "would agree with the plan"}, repair_code=BENT)
    engine._client = again
    await _gate(engine, [RK4], RK4_SOURCE)
    assert again.repairs == [] and again.recompute_prompts == []
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == RK4_SOURCE
    record = _record(engine)
    assert record["disputed"] == ["rk4_closed_form_h01"] and record["proposed_changes"][0]["source"] == "recompute"
    # A changed check is not the one the dispute was about, and it now passes.
    changed = {**RK4, "expected": 3.3324e-07}
    engine._client = _Model(recompute={"expected": 3.3324e-07}, repair_code=BENT)
    await _gate(engine, [changed], RK4_SOURCE)
    record = _record(engine)
    assert record["status"] == "ok" and "disputed" not in record and "proposed_changes" not in record


VERLET_SOURCE = '''
def run_cell(cell):
    dt, t_end = float(cell["dt"]), float(cell.get("t_end", 10.0))
    x, v = 1.0, 0.0
    e0 = 0.5 * (x * x + v * v)
    worst = 0.0
    for _ in range(int(round(t_end / dt))):
        a = -x
        x += v * dt + 0.5 * a * dt * dt
        v += 0.5 * (a - x) * dt
        worst = max(worst, abs(0.5 * (x * x + v * v) - e0))
    return {"energy_drift": worst}
'''
VERLET = {"name": "verlet_energy_conservation_undamped", "kind": "invariant",
          "check": "velocity Verlet conserves the energy of an undamped oscillator", "expected": 0, "tolerance": 1e-12,
          "case": {"dt": 0.01, "t_end": 10.0}, "measure": "energy_drift",
          "reference": "derivation: the oscillator conserves E = (x^2 + v^2) / 2, and Verlet is symplectic, so it "
                       "conserves energy to machine precision"}


@pytest.mark.asyncio
async def test_verlet_energy_at_1e_12_is_the_methods_own_error_at_its_step_and_the_script_is_left_alone(
        tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(recompute={"expected": 0, "how": "symplectic, so energy is conserved"},
                   repair_code=VERLET_SOURCE.replace("worst = max", "worst = 0 * max"))
    engine._client = model
    await _gate(engine, [VERLET], VERLET_SOURCE)
    assert model.repairs == [], "a tolerance tighter than the method's own error is not the script's to fix"
    record = _record(engine)
    (step,) = _triage(record, "half_step")
    assert step["verdict"] == "method_error" and step["points_to"] == "tolerance"
    assert step["steps"] == [0.01, 0.005, 0.0025]
    assert step["values"][0] == pytest.approx(1.25e-05, rel=1e-3) and step["order"] == pytest.approx(2, abs=0.05)
    assert abs(step["extrapolated"]) < 1e-8 and step["method_error"] == pytest.approx(1.25e-05, rel=1e-2)
    assert "verlet_energy_conservation_undamped" in record["set_aside"]
    assert record["self_checks"] == {"model_calls": 1, "extra_runs": 2}
    (recomputed,) = _triage(record, "recompute")
    assert recomputed["verdict"] == "agrees"
    assert "tighter than the method's own error at this step" in (step.get("cause") or {}).get("text", "")
    assert record["status"] == "went_on_failing", "the verdict stands: the check stays failed, marked unconfirmed"
    assert _explained(engine)["leaning"] == "check" and "step error" in _explained(engine)["why"]
    assert "2 extra run(s)" in plan.plan_path(engine.quest_root).read_text(encoding="utf-8")


def test_a_check_whose_number_is_an_error_at_a_finite_step_is_not_called_method_error() -> None:
    # The RK4 check measures the error itself at h = 0.1: at smaller steps it shrinks towards 0, not towards the plan's
    # 1.637e-08, so the gap is not the step's error.
    got = ot.step_verdict(RK4, [3.33241e-07, 1.99761e-08, 1.22274e-09], order=None)
    assert got is not None and got["verdict"] == "converges_elsewhere"
    with_order = ot.step_verdict({**RK4, "order": 4}, [3.33241e-07, 1.99761e-08, 1.22274e-09], order=4)
    assert with_order is not None and with_order["verdict"] == "converges_elsewhere"
    # A wrong constant does not move with the step at all.
    flat = ot.step_verdict({**VERLET, "tolerance": 1e-6}, [0.25, 0.25, 0.25], order=None)
    assert flat is not None and flat["verdict"] == "not_converging"


def _euler(dt: float) -> float:
    y = 1.0
    for _ in range(int(round(1.0 / dt))):
        y += dt * -y
    return y


def test_a_first_order_scheme_where_a_higher_one_is_meant_is_never_set_aside() -> None:
    # Explicit Euler where RK4 is meant: it does converge to exp(-1), but at first order.
    decay = {"name": "decay", "kind": "special_case", "expected": 0.36787944117144233, "tolerance": 1e-6,
             "case": {"dt": 0.02}, "measure": "y"}
    values = [_euler(0.02), _euler(0.01), _euler(0.005)]
    said = ot.step_verdict(decay, values, order=None)
    assert said is not None and said["verdict"] == "low_order" and said["order"] == pytest.approx(1, abs=0.1)
    assert ot.step_entry(decay, "dt", [0.02, 0.01, 0.005], said)["points_to"] != "tolerance"
    declared = ot.step_verdict({**decay, "order": 4}, values, order=4)
    assert declared is not None and declared["verdict"] == "wrong_order"
    # RK2 where RK4 is meant converges too, at order 2: with no order declared that is said, never set aside.
    rk2 = [_rk2(0.1), _rk2(0.05), _rk2(0.025)]
    unsure = ot.step_verdict({**decay, "tolerance": 1e-9}, rk2, order=None)
    assert unsure is not None and unsure["verdict"] == "converges_unconfirmed"
    assert ot.step_entry(decay, "dt", [0.1, 0.05, 0.025], unsure)["points_to"] != "tolerance"
    assert ot.step_verdict({**decay, "tolerance": 1e-9, "order": 4}, rk2, order=4)["verdict"] == "wrong_order"
    # A scheme that converges at its declared order to a value 500 tolerances off is not the method's error.
    off = [1.0005 + 0.01 * h ** 2 for h in (0.1, 0.05, 0.025)]
    biased = ot.step_verdict({"name": "y", "expected": 1.0, "tolerance": 1e-6, "order": 2}, off, order=2)
    assert biased is not None and biased["verdict"] == "converges_elsewhere"


def _rk2(dt: float) -> float:
    y = 1.0
    for _ in range(int(round(1.0 / dt))):
        k1 = -y
        y += dt * -(y + dt / 2 * k1)
    return y


def test_trials_of_a_rule_every_trial_must_keep_that_differ_are_the_fault_not_noise() -> None:
    sums = {"name": "probabilities_sum_to_one", "kind": "invariant", "expected": 0.0, "tolerance": 1e-12}
    said = ot.seeds_verdict(sums, [1e-3, 2.5e-3, 2e-4, 1.8e-3])
    assert said is not None and said["verdict"] == "varies"
    assert ot.seeds_entry(sums, said)["points_to"] == "script"
    # A biased mean is not noise either: four trials whose mean is far (in standard errors) from the expected value.
    biased = ot.seeds_verdict(MM1, [0.80, 0.86, 0.83, 0.90])
    assert biased is not None and biased["verdict"] == "beyond_noise"


def test_a_recompute_never_disputes_the_worst_violation_of_a_rule_expecting_0() -> None:
    assert ot.recompute_verdict(VERLET, 1e-5, 2.9e-5) == "differs"
    # Near means within 1.5 times where the plan is ten times or more away.
    assert ot.recompute_verdict(RK4, 3.0e-07, 3.33241e-07) == "disputed"
    assert ot.recompute_verdict(RK4, 1.5e-07, 3.33241e-07) == "differs"


MM1_SOURCE = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    lam, mu, n = float(cell["lam"]), float(cell["mu"]), int(cell["customers"])
    w, total = 0.0, 0.0
    for _ in range(n):
        total += w
        w = max(0.0, w + rng.expovariate(mu) - rng.expovariate(lam))
    return {"mean_wait": total / n}
'''
MM1 = {"name": "mm1_mean_wait", "kind": "special_case", "check": "the M/M/1 mean wait in queue at rho = 0.5",
       "expected": 1.0, "tolerance": 0.01, "case": {"lam": 0.5, "mu": 1.0, "customers": 1000}, "measure": "mean_wait",
       "reference": "derivation: Wq = rho / (mu - lambda) = 0.5 / 0.5 = 1"}


@pytest.mark.asyncio
async def test_a_single_random_trial_judged_tighter_than_its_noise_is_said_and_not_repaired(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(recompute={"expected": 1.0, "how": "Wq = rho/(mu-lambda)"}, repair_code=MM1_SOURCE)
    engine._client = model
    await _gate(engine, [MM1], MM1_SOURCE)
    record = _record(engine)
    (seeds,) = _triage(record, "seeds")
    assert seeds["verdict"] == "noise" and len(seeds["values"]) == 4 and seeds["sd"] > 0.01
    assert len(set(seeds["values"])) == 4, "each extra trial has its own seed"
    assert model.repairs == [] and record["self_checks"]["extra_runs"] == 3
    assert "smaller than one trial's noise" in (seeds.get("cause") or {}).get("text", "")
    assert record["status"] == "went_on_failing" and _explained(engine)["leaning"] == "check"


SAME_ERROR = '''
import os

def run_cell(cell):
    scale = float(os.environ["MY_SETTING"])
    return {"error": scale}
'''


@pytest.mark.asyncio
async def test_the_same_exception_after_a_repair_stops_the_repairs_early(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(repair_code=SAME_ERROR.replace("import os", "import os  # the repair looked elsewhere"))
    engine._client = model
    await _gate(engine, [RK4], SAME_ERROR)
    assert len(model.repairs) == 1, "the second repair of the budget is kept"
    assert "The run of the checks stopped with: KeyError: 'MY_SETTING'" in model.repairs[0]
    record = _record(engine)
    assert [a.get("exception") for a in record["attempts"]] == ["KeyError: 'MY_SETTING'"] * 2
    (stopped,) = _triage(record, "same_exception")
    assert stopped["repairs_left"] == 1
    assert record["status"] == "went_on_failing" and record["went_on"][0]["measured"] is None
    assert "KeyError: 'MY_SETTING'" in record["went_on"][0]["unmeasured"]
    assert _explained(engine)["leaning"] == "script" and "stopped with an error" in _explained(engine)["why"]


@pytest.mark.asyncio
async def test_a_run_that_runs_out_of_time_gets_one_retry_at_twice_the_time(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Engine(_config(tmp_path))
    engine._client = _Model()
    limits: list[int] = []
    from core import trial_runner

    async def measure(executor, py, root, module, oracles, *, timeout_s, **kw):  # noqa: ANN001
        limits.append(timeout_s)
        if len(limits) == 1:
            return [], [f"the oracle {RK4['name']!r}: the simulation could not be run on its case (run_cell() ran out "
                        "of time)"], True
        return [{"name": RK4["name"], "value": 1.6e-08, "measured_by": "engine"}], [], False

    monkeypatch.setattr(trial_runner, "measure_oracles", measure)
    await _gate(engine, [RK4], RK4_SOURCE)
    assert limits == [limits[0], limits[0] * 2]
    record = _record(engine)
    assert record["status"] == "ok"
    (retry,) = _triage(record, "timeout_retry")
    assert retry["verdict"] == "finished" and record["self_checks"]["extra_runs"] == 1


# --- a unit or a representation: pure ----------------------------------------------------------------------------------


@pytest.mark.parametrize("oracle, value, words", [
    ({"name": "attack_rate", "expected": 0.25, "tolerance": 1e-3}, 25.0, "a per cent and a fraction"),
    ({"name": "attack_rate", "expected": 25.0, "tolerance": 0.1}, 0.25, "a per cent and a fraction"),
    ({"name": "mean_degree", "expected": 0.5, "tolerance": 1e-3, "case": {"n_agents": 10}}, 5.0,
     "a total and a mean over the case's n_agents=10"),
    ({"name": "frequency", "expected": 1.0, "tolerance": 1e-3}, 6.28319, "an angular frequency and a frequency"),
    ({"name": "rate", "expected": 4.0, "tolerance": 1e-3}, 0.25, "a rate and a time"),
    ({"name": "length", "expected": 2.5e-3, "tolerance": 1e-6}, 2.5, "a thousand apart"),
])
def test_a_unit_or_representation_multiple_is_named(oracle: dict[str, Any], value: float, words: str) -> None:
    said = of.mismatch(oracle, value)
    assert said is not None and words in said, said


@pytest.mark.parametrize("oracle, value", [
    ({"name": "violation", "kind": "invariant", "expected": 0.0, "tolerance": 1e-6}, 0.5),  # expected 0: no factor
    ({"name": "attack_rate", "expected": 0.25, "tolerance": 1e-3}, 25.4),  # near 100x, but not within the tolerance
    ({"name": "attack_rate", "expected": 0.25, "tolerance": 1e-3}, 0.30),  # simply outside the tolerance
    ({"name": "x", "expected": 1.0, "tolerance": 0.01, "case": {"L": 2, "seed": 42}}, 2.0),  # L and seed are no counts
    ({"name": "x", "expected": 0.9, "tolerance": 0.05}, 1.1),  # a reciprocal the tolerance cannot tell apart
])
def test_no_multiple_is_named_on_a_ratio_that_is_merely_close(oracle: dict[str, Any], value: float) -> None:
    assert of.unit_multiple(oracle, value) is None


def test_the_test_run_asks_the_plan_only_about_a_fixed_unit_factor_never_a_count_or_a_reciprocal() -> None:
    mean = {"name": "mean_degree", "expected": 0.5, "tolerance": 1e-3, "case": {"n_agents": 10}, "measure": "deg"}
    assert of.mismatches([mean], [{"name": "mean_degree", "value": 5.0, "measured_by": "engine"}]) == []
    assert "a total and a mean" in (of.mismatch(mean, 5.0) or ""), "the card still says it"
    percent = {"name": "attack_rate", "expected": 0.25, "tolerance": 1e-3}
    assert "a per cent" in of.mismatches([percent], [{"name": "attack_rate", "value": 25.0}])[0]


# --- the protocol check and the oracle's own runs ----------------------------------------------------------------------

# From the two real quests the protocol card stopped: the oracle's own run counts, read as runs per setting.
LUNA1_SIMULATE = '''
ORACLE_RUNS = 20000
ORACLE_ABS_TOL = 0.008

def run_trial(cell, trial_id, seed):
    return {"attack": 0.5}

def oracle():
    hits = 0
    for run_index in range(ORACLE_RUNS):
        hits += 1
    zero_runs = 1000
    return {"two_person": hits / float(ORACLE_RUNS), "zero": zero_runs}
'''
LUNA1_EXPERIMENT = '''
PRIMARY_R0_VALUES = [1.5, 3]
'''
LUNA2_SIMULATE = '''
def run_trial(cell, trial_id, seed):
    return {"final_size": 0.8}

def oracle() -> dict:
    n2_runs = 5000
    n2 = 0
    for _ in range(n2_runs):
        n2 += 1
    return {"n2_probability": n2 / n2_runs}
'''
ONE_SCRIPT = '''
import os
if os.environ.get("FI_ORACLE") == "1":
    check_runs = 20000
    print("ORACLE_JSON: {}")
else:
    n_runs = 900
'''


def test_the_protocol_check_does_not_count_the_oracles_own_runs_as_runs_per_setting() -> None:
    luna1 = pc.check({"grid": {"R0": [0.9, 1, 1.1, 1.5, 3]}, "runs_per_setting": 900},
                     {"simulate.py": LUNA1_SIMULATE, "experiment.py": LUNA1_EXPERIMENT})
    assert [m.kind for m in luna1] == ["grid"], [m.message() for m in luna1]  # the true positive stays
    luna2 = pc.check({"grid": {"N": [100, 250]}, "runs_per_setting": 300}, {"simulate.py": LUNA2_SIMULATE})
    assert not [m for m in luna2 if m.kind == "runs"], [m.message() for m in luna2]
    assert pc.check({"runs_per_setting": 900}, {"experiment.py": ONE_SCRIPT}) == []
    flag_off = '''
import os
if os.environ.get("FI_ORACLE", "0") == "0":
    n_runs = 900
else:
    check_runs = 20000
'''
    assert [m.found for m in pc.check({"runs_per_setting": 300}, {"experiment.py": flag_off})] == [[900.0]]
    # The main experiment's own count is still read.
    wrong = pc.check({"runs_per_setting": 300}, {"experiment.py": ONE_SCRIPT})
    assert [m.kind for m in wrong] == ["runs"] and wrong[0].found == [900.0]
