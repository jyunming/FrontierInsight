"""A stop at the known-answer checks asks a person one plain question, never to judge a tolerance or debug a script.

FI's own look at a failing check (core/oracle_triage.py) already says which side is most likely wrong: the check, the
simulation, or nobody can tell. The card (core/oracle_card.py) turns that into ONE yes/no question with FI's
recommendation, and keeps the numbers and the script lines as details. Nothing here relaxes a check: a failed check
stays failed; going on marks it unconfirmed, which keeps the result exploratory.

The arithmetic case is a real one (a gemma4 quest on the pendulum, 2026-10-04): the plan derived the exact period at
90 degrees as ``2*pi*sqrt(1/9.81) * (2/pi) * 1.85407 => 1.18034``, which multiplies out to 2.368 (and 2.368 was
measured, correctly). The recheck by the same model repeated the slip, the card said the simulation was the likely
cause, and FI's repair then "fixed" the correct code towards 1.180.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import accepted_checks, oracle_card, oracle_triage, todo
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

PENDULUM_REFERENCE = ("derivation: T = T0 * (2/pi) * K(sin(45deg)) = 2*pi*sqrt(1/9.81) * (2/pi) * 1.85407 => 1.18034")
PENDULUM_RECHECK = ("T0 = 2 * pi * sqrt(1 / 9.81) = 2.006066; k = sin(45 deg) = 0.70710678; K(k) = ellipk(0.5) = "
                    "1.85407466. T = T0 * (2 / pi) * K(k) = 2.006066 * 0.63661977 * 1.85407466 = 1.1803415.")
EXACT_90 = {"name": "exact_90_deg", "kind": "special_case", "check": "T_exact at 90 degrees", "expected": 1.18034,
            "tolerance": 1e-05, "tolerance_mode": "absolute", "reference": PENDULUM_REFERENCE,
            "case": {"amplitude_deg": 90}, "measure": "T_exact"}
MEASURED = 2.3678419475762373


# --- the arithmetic a derivation writes out ------------------------------------------------------------------------------


def test_the_plans_own_arithmetic_is_worked_out_and_the_slip_is_found() -> None:
    slip = oracle_triage.arithmetic_slip(PENDULUM_REFERENCE)
    assert slip is not None and slip["written"] == 1.18034 and abs(slip["computes"] - MEASURED) < 1e-4
    # The recheck repeated the same slip: its own working does not add up either.
    again = oracle_triage.arithmetic_slip(PENDULUM_RECHECK)
    assert again is not None and abs(again["computes"] - 2.3678) < 1e-3 and again["written"] == 1.1803415


@pytest.mark.parametrize("text", [
    "derivation: T = 2*pi*sqrt(1/9.81) = 2.00607",  # right
    "h^4/120 = 0.1^4/120 = 8.33e-07",  # right, with a power written as ^
    "derivation: sum of the weights = 1 by definition",  # no arithmetic
    "derivation: T = T0 * (2/pi) * K(k)",  # symbols only
    "[1], eq. 3",
    "y(10) = exp(-10) = 4.54e-05 s",  # right, with a unit after the number
])
def test_working_that_adds_up_or_is_not_arithmetic_is_left_alone(text: str) -> None:
    assert oracle_triage.arithmetic_slip(text) is None


def test_the_calculator_reads_only_arithmetic_never_code() -> None:
    assert abs(oracle_triage.calculate("sin(45deg)") - 0.70710678) < 1e-8
    assert oracle_triage.calculate("2^3") == 8.0
    for hostile in ('__import__("os").system("echo x")', "(1).__class__", "open('x')", "2**10000", "lambda: 1",
                    "x + 1", "[1, 2][0]"):
        assert oracle_triage.calculate(hostile) is None, hostile


# --- the recheck: the same model is not independent, and a recheck that does not add up is not evidence ------------------


def test_a_recheck_by_the_model_that_wrote_the_plan_is_said_and_never_counted() -> None:
    entry = oracle_triage.recompute_entry(EXACT_90, "agrees", 2.0, MEASURED, "", model="gemma", same_model=True)
    assert entry["points_to"] == "" and "same model that wrote the plan" in entry["tried"]
    other = oracle_triage.recompute_entry(EXACT_90, "agrees", 1.18034, MEASURED, "", model="m2", same_model=False)
    assert other["points_to"] == "script", "another model agreeing with the plan still points to the simulation"


def test_a_recheck_whose_own_working_does_not_add_up_is_not_used() -> None:
    entry = oracle_triage.recompute_entry(EXACT_90, "agrees", 1.1803415, MEASURED, PENDULUM_RECHECK, model="m2",
                                          same_model=False)
    assert entry["points_to"] == "" and entry["how_slip"] and "does not add up" in entry["tried"]


# --- the card's plain explanation (recorded with FI's automatic decision; never a question) --------------------------------


def _card(tmp_path: Path, oracles: list[dict[str, Any]], judged: list[dict[str, Any]], triage: list[dict[str, Any]],
          *, kept: str | None = None, proposals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    script = tmp_path / "q-1" / "code" / "simulate.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    found = [f"the oracle '{j['name']}' failed: FI measured {j.get('value')}" for j in judged]
    return oracle_card.build(quest_id="q-1", quest_root=tmp_path / "q-1", script=script, found=found, oracles=oracles,
                             judged=judged, attempts=[{"checks": judged, "triage": triage, "repair": "applied"}],
                             proposals=proposals, trial=True, kept=kept)


def _judged(name: str, value: float | None) -> dict[str, Any]:
    return {"name": name, "value": value, "passed_by_engine": False, "measured_by": "engine"}


def test_the_real_pendulum_stop_is_explained_as_the_plans_arithmetic_not_the_simulation(tmp_path: Path) -> None:
    same_model = oracle_triage.recompute_entry(EXACT_90, "agrees", 1.1803415, MEASURED, PENDULUM_RECHECK,
                                               model="gemma4:31b-cloud", same_model=True)
    card = _card(tmp_path, [EXACT_90], [_judged("exact_90_deg", MEASURED)], [same_model])
    assert card["leaning"] == "check", "the slip in the plan's arithmetic, not the simulation"
    assert "gives 2.36784" in card["why"] and "1.18034" in card["why"]
    assert card["checks"][0]["status"] == "failed", "explaining changes no verdict"
    assert not any("simulation is the likely cause" in c["text"] for c in card["causes"])


def test_with_a_second_check_not_measured_the_explanation_still_names_the_slip(tmp_path: Path) -> None:
    numerical = {"name": "numerical_period_90_deg", "kind": "special_case", "expected": 1.18034, "tolerance": 0.001,
                 "reference": "Numerical integration should match T_exact", "case": {"amplitude_deg": 90, "dt": 0.001},
                 "measure": "T_num"}
    card = _card(tmp_path, [EXACT_90, numerical], [_judged("exact_90_deg", MEASURED),
                                                    _judged("numerical_period_90_deg", None)], [])
    assert card["leaning"] == "unclear" and "gives 2.36784" in card["why"]


ORACLE = {"name": "final size", "kind": "closed_form", "check": "final size", "expected": 1.0, "tolerance": 0.05,
          "reference": "[1]"}


@pytest.mark.parametrize("look, side, said", [
    ({"kind": "seeds", "verdict": "noise", "points_to": "tolerance"}, "check", "differ by more than the check allows"),
    ({"kind": "half_step", "verdict": "converges_elsewhere", "points_to": "script"}, "script",
     "settles on a different value"),
    (None, "unclear", "do not agree"),
])
def test_the_explanation_follows_fis_own_look(tmp_path: Path, look: dict[str, Any] | None, side: str,
                                              said: str) -> None:
    triage = [{"check": "final size", **look, "cause": {"text": "x", "evidence": "y"}}] if look else []
    card = _card(tmp_path, [ORACLE], [_judged("final size", 0.97 if side == "check" else 0.5)], triage)
    assert card["leaning"] == side and said in card["why"]


def test_a_recheck_by_the_same_model_is_said_to_be_not_independent(tmp_path: Path) -> None:
    same = oracle_triage.recompute_entry(ORACLE, "agrees", 1.0, 0.5, "", model="gemma", same_model=True)
    card = _card(tmp_path, [ORACLE], [_judged("final size", 0.5)], [same])
    assert card["leaning"] == "unclear" and "same model that wrote the plan" in card["why"]


def test_a_check_with_no_number_is_explained_as_the_plan(tmp_path: Path) -> None:
    card = _card(tmp_path, [{"name": "final size", "check": "final size"}], [], [])
    assert card["leaning"] == "plan" and "no number to compare" in card["why"]


# --- nothing is relaxed ---------------------------------------------------------------------------------------------------


def test_a_check_that_measured_nothing_is_gone_on_with_marked_and_named(tmp_path: Path) -> None:
    found = ["the oracle 'final size': the simulation could not be run on its case (TypeError: ...)"]
    entries = accepted_checks.went_on_entries(found, [ORACLE], [{"name": "final size", "value": None}])
    assert len(entries) == 1 and entries[0]["measured"] is None and "TypeError" in entries[0]["unmeasured"]
    sentence = accepted_checks.gap({**entries[0], "by": accepted_checks.AUTOMATIC})
    assert "could not be judged" in sentence and accepted_checks.UNCONFIRMED in sentence
    # A run that names no check still leaves a gap.
    (only,) = accepted_checks.went_on_entries(["the plan has no known-answer check"], [], [])
    assert only["name"] == "" and "no known-answer check" in only["unmeasured"]


def test_the_went_on_item_says_what_it_means_and_asks_no_comparison(tmp_path: Path) -> None:
    quest = tmp_path / "q-1"
    (quest / "needs").mkdir(parents=True)
    record = {"status": accepted_checks.WENT_ON, "went_on": [{"name": "final size", "by": "jun", "expected": 1.0,
                                                               "limit": 0.05, "measured": 0.5}]}
    (quest / "needs" / accepted_checks.ORACLE_RECORD).write_text(json.dumps(record), encoding="utf-8")
    item = next(i for i in todo.waiting(quest) if i.kind == "went_on")
    assert "Nothing you have to fix" in item.recommended and "exploratory" in item.recommended
    assert "Compare the measured value" not in item.recommended


# --- the gate: a slip in the plan's arithmetic is never "fixed" in the simulation -------------------------------------


@pytest.mark.asyncio
async def test_the_gate_sets_a_check_with_a_slip_aside_from_the_repairs_and_offers_the_corrected_value(
        tmp_path: Path) -> None:
    eng = Engine(Config(topic="pendulum", title="slip", provider=ProviderConfig(name="openai"),
                        engine=EngineConfig(max_iterations=1, review_loop=False),
                        execution=ExecutionConfig(sandbox="venv"), knowledge=KnowledgeConfig(enabled=False),
                        output=OutputConfig(output_dir=tmp_path / "out")))
    eng._oracle_proposals, eng._oracle_disputed, eng._oracle_set_aside, eng._oracle_noisy = {}, set(), {}, set()
    same = oracle_triage.recompute_entry(EXACT_90, "agrees", 1.1803415, MEASURED, PENDULUM_RECHECK, model="gemma",
                                         same_model=True)
    eng._recompute_expected = AsyncMock(return_value=(same, True))  # type: ignore[method-assign]
    seed = eng.quest_root / "code" / "simulate.py"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text("x = 1\n", encoding="utf-8")
    record: dict[str, Any] = {"judged": [_judged("exact_90_deg", MEASURED)]}
    await eng._look_at_failing_checks({}, None, seed, [EXACT_90], [EXACT_90], record, set(), protocol={},
                                      timeout=10, case_env={})
    assert "exact_90_deg" in eng._oracle_set_aside, "the script is not rewritten towards the plan's slip"
    proposal = eng._oracle_proposals["exact_90_deg"]
    assert proposal["source"] == "arithmetic" and abs(proposal["expected"] - MEASURED) < 1e-4
    kinds = [t["kind"] for t in record["triage"]]
    assert "arithmetic" in kinds and record["triage"][kinds.index("arithmetic")]["points_to"] == "check"


# --- FI corrects a check itself, only from an independent value ----------------------------------------------------------


def _engine_with_plan(tmp_path: Path, oracle: dict[str, Any]) -> Engine:
    from core import plan

    eng = Engine(Config(topic="pendulum", title="fix", provider=ProviderConfig(name="openai"),
                        engine=EngineConfig(max_iterations=1, review_loop=False),
                        execution=ExecutionConfig(sandbox="venv"), knowledge=KnowledgeConfig(enabled=False),
                        output=OutputConfig(output_dir=tmp_path / "out")))
    eng._oracle_proposals, eng._oracle_disputed, eng._oracle_set_aside, eng._oracle_noisy = {}, set(), {}, set()
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    design = {"hypothesis": "h", "method": "m", "protocol": {"grid": {"amplitude_deg": [5, 90]}, "oracles": [oracle]}}
    text = plan.render("pendulum", {}, design)
    plan.plan_path(eng.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(eng.quest_root, text, by="model", note="written")
    return eng


def _planned(eng: Engine) -> dict[str, Any]:
    from core import plan

    return plan.load_design(eng.quest_root)[0]["protocol"]["oracles"][0]


def _confirmed(proposal: dict[str, Any], value: float = 2.367842, *, same_model: bool = False) -> dict[str, Any]:
    """``proposal`` (the plan's own arithmetic) with another model's blind recheck attached when it agrees."""
    entry = {**oracle_triage.recompute_entry(EXACT_90, "disputed", value, MEASURED, "x", model="m2",
                                             same_model=same_model), "blind": True}
    second = Engine._second_source(EXACT_90, proposal, entry)  # noqa: SLF001
    return {**proposal, **({"confirmed_by": second} if second else {})}


def test_the_plans_arithmetic_alone_never_rewrites_a_check(tmp_path: Path) -> None:
    """A slip in the plan's own working only sets the check aside: the arithmetic can misread a correct derivation, so
    on its own it may lower a result (unconfirmed, exploratory) and never rewrite the check."""
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng._oracle_proposals["exact_90_deg"] = oracle_triage.arithmetic_proposal(
        EXACT_90, oracle_triage.arithmetic_slip(PENDULUM_REFERENCE))
    assert eng._correct_expected_values([_judged("exact_90_deg", MEASURED)], [EXACT_90]) == []
    assert _planned(eng)["expected"] == 1.18034
    assert not (eng.fi_dir / "oracle_corrections.json").exists()


def test_a_second_model_that_disagrees_with_the_arithmetic_or_is_the_same_model_confirms_nothing() -> None:
    proposal = oracle_triage.arithmetic_proposal(EXACT_90, oracle_triage.arithmetic_slip(PENDULUM_REFERENCE))
    assert "confirmed_by" not in _confirmed(proposal, 2.3678), "off by more than the check's own tolerance (1e-5)"
    assert "confirmed_by" not in _confirmed(proposal, same_model=True)
    assert Engine._may_correct(_confirmed(proposal)) is True  # noqa: SLF001


def test_fi_corrects_a_slip_to_the_value_the_plans_own_working_gives_never_to_the_measured_one(tmp_path: Path) -> None:
    from core import audit_log, plan

    # A slip whose working gives 2.36784, confirmed by another model, while the simulation measured something else
    # (3.0): the correction must be the derived value, not the measurement.
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng._oracle_proposals["exact_90_deg"] = _confirmed(oracle_triage.arithmetic_proposal(
        EXACT_90, oracle_triage.arithmetic_slip(PENDULUM_REFERENCE)))
    fixed = eng._correct_expected_values([_judged("exact_90_deg", 3.0)], [EXACT_90])
    assert fixed == ["exact_90_deg"]
    after = _planned(eng)
    assert abs(after["expected"] - 2.36784) < 1e-5 and after["expected"] != 3.0, "derived, never the measured value"
    assert after["tolerance"] == EXACT_90["tolerance"] and after.get("tolerance_mode") == "absolute", "tolerance kept"
    assert after["case"] == EXACT_90["case"] and after["measure"] == EXACT_90["measure"]
    assert plan.history(eng.quest_root)[-1]["by"] == "engine"
    record = json.loads((eng.fi_dir / "oracle_corrections.json").read_text(encoding="utf-8"))
    assert record["exact_90_deg"]["from"] == 1.18034 and record["exact_90_deg"]["source"] == "arithmetic"
    assert record["exact_90_deg"]["confirmed_by"]["source"] == "recompute"
    assert any(e.get("check") == "oracle_correction" for e in audit_log.read(eng.audit.path))
    # Never twice: the same check is not corrected again in this quest.
    eng._oracle_proposals["exact_90_deg"] = {**eng._oracle_proposals.get("exact_90_deg", {}), "name": "exact_90_deg",
                                             "expected": 9.0, "source": "arithmetic", "tolerance": 1e-5}
    assert eng._correct_expected_values([_judged("exact_90_deg", 3.0)], [_planned(eng)]) == []


@pytest.mark.parametrize("proposal", [
    {"name": "exact_90_deg", "expected": 2.36784, "tolerance": 1e-5, "reason": "the repair says so"},  # saw the run
    {"name": "exact_90_deg", "expected": 2.36784, "tolerance": 1e-5, "source": "recompute", "same_model": True},
    {"name": "exact_90_deg", "expected": 2.36784, "tolerance": 1e-5, "source": "recompute", "how_slip": True},
])
def test_a_value_that_is_not_independent_is_never_used_to_correct_a_check(tmp_path: Path,
                                                                         proposal: dict[str, Any]) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng._oracle_proposals["exact_90_deg"] = proposal
    assert eng._correct_expected_values([_judged("exact_90_deg", MEASURED)], [EXACT_90]) == []
    assert _planned(eng)["expected"] == 1.18034


def test_one_other_model_alone_never_corrects_a_check(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    entry = oracle_triage.recompute_entry(EXACT_90, "disputed", 2.3678, MEASURED, "", model="m2", same_model=False)
    eng._oracle_proposals["exact_90_deg"] = oracle_triage.recompute_proposal(EXACT_90, entry)
    assert eng._correct_expected_values([_judged("exact_90_deg", MEASURED)], [EXACT_90]) == []
    assert _planned(eng)["expected"] == 1.18034


def test_after_the_freeze_fi_corrects_nothing(tmp_path: Path) -> None:
    from core import frozen_protocol

    eng = _engine_with_plan(tmp_path, EXACT_90)
    frozen_protocol.freeze(eng.quest_root, {"oracles": [EXACT_90]}, approved_by="t", source="plan.md")
    eng._oracle_proposals["exact_90_deg"] = oracle_triage.arithmetic_proposal(
        EXACT_90, oracle_triage.arithmetic_slip(PENDULUM_REFERENCE))
    assert eng._correct_expected_values([_judged("exact_90_deg", MEASURED)], [EXACT_90]) == []
    assert _planned(eng)["expected"] == 1.18034


def test_every_quest_goes_on_by_itself_research_included(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng.config = eng.config.model_copy(update={"rigor_profile": "research"})
    assert eng._goes_on_by_itself() is True
    eng.config.engine.oracle_check = "warn"
    assert eng._goes_on_by_itself() is False, "warn records and goes on its own way"


# --- after the first external review ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("T0 = 2*pi*sqrt(L/g) with L = 1, g = 9.81 = 2.0061", 2.0061),  # a parameter, then the result
    ("final error at t = 1 = 3.3e-7", 3.3e-7),
    ("y(t) = exp(-t), so y at t = 1 = 0.3679", 0.3679),
    ("final size with R0 = 2 => 0.7968", 0.7968),
    ("K(m) at m = 0.5 = 1.8541", 1.8541),
    ("area = pi*r^2 with r=2 = 12.566", 12.566),
    ("= 0.25 + 0.25 + 0.5 = 1 = 100%", 1.0),
    ("omega = sqrt(9.81/1) = 3.31; T = 2*pi/3.132 = 2.006", 2.006),  # a slip in an intermediate step only
    ("2*pi*sqrt(1/9.81) = 2.000", 2.0),  # rounded, not a slip
    ("2*pi*16 = 100", 100.0),
])
def test_a_parameter_an_intermediate_step_or_a_rounding_is_never_a_slip(text: str, expected: float) -> None:
    assert oracle_triage.arithmetic_slip(text, expected) is None


def test_a_slip_counts_only_in_the_step_that_writes_the_checks_expected_value() -> None:
    assert oracle_triage.arithmetic_slip(PENDULUM_REFERENCE, 1.18034) is not None
    assert oracle_triage.arithmetic_slip(PENDULUM_REFERENCE, 9.0) is None, "another expected value: not this step"
    assert oracle_triage.plan_slip(EXACT_90) is not None
    corrected = {**EXACT_90, "expected": 2.3678359741866406}
    assert oracle_triage.plan_slip(corrected) is None, "after the correction the same working is no slip any more"


def test_a_check_measured_as_a_violation_or_a_formula_is_never_corrected() -> None:
    assert oracle_triage.correctable(EXACT_90)
    assert not oracle_triage.correctable({**EXACT_90, "kind": "invariant"})
    assert not oracle_triage.correctable({**EXACT_90, "measure": "abs((T_exact) - 1.18034)"})


def test_an_old_recheck_record_that_does_not_say_which_model_answered_is_not_independent(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng._oracle_proposals["exact_90_deg"] = {"name": "exact_90_deg", "expected": 2.3678, "tolerance": 1e-5,
                                             "source": "recompute", "reason": "an earlier FI"}
    assert eng._correct_expected_values([_judged("exact_90_deg", MEASURED)], [EXACT_90]) == []


def test_the_corrected_plan_says_what_fi_changed_next_to_the_derivation(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng._oracle_proposals["exact_90_deg"] = _confirmed(oracle_triage.arithmetic_proposal(
        EXACT_90, oracle_triage.plan_slip(EXACT_90)))
    assert eng._correct_expected_values([_judged("exact_90_deg", MEASURED)], [EXACT_90]) == ["exact_90_deg"]
    after = _planned(eng)
    assert "FI corrected the expected value from 1.18034 to 2.36784" in after["reference"]
    assert oracle_triage.plan_slip(after) is None, "a second look finds no slip, so a wrong simulation is still repaired"


@pytest.mark.asyncio
async def test_improve_never_optimises_against_a_check_fi_could_not_confirm(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from core import criteria as crit

    eng = _engine_with_plan(tmp_path, EXACT_90)
    criterion = {"name": "exact error", "oracle": "exact_90_deg", "use": "error", "direction": "lower",
                 "target": 0.0, "tolerance": 1.0e-6}
    protocol = {"grid": {"amplitude_deg": [90]}, "oracles": [EXACT_90], "criteria": [criterion]}
    monkeypatch.setattr(eng, "_draft_protocol", lambda state: protocol)
    monkeypatch.setattr(eng, "_improve_skip", lambda state: None)
    monkeypatch.setattr(crit, "history", lambda root: [{"n": 1, "criteria": [{"name": "exact error", "value": 1.19,
                                                                               "met": False, "counts": True}]}])
    loops: list[Any] = []

    async def loop(*a: Any, **k: Any) -> dict[str, Any]:
        loops.append(a)
        return {}

    monkeypatch.setattr(eng, "_improve_loop", loop)
    (eng.quest_root / "needs").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(
        {"status": accepted_checks.WENT_ON, "went_on": [{"name": "exact_90_deg", "by": accepted_checks.AUTOMATIC}]}),
        encoding="utf-8")
    await eng._node_improve({"topic": "pendulum"})
    assert loops == [], "no change to the simulation is made against the unconfirmed check"
    (eng.quest_root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    await eng._node_improve({"topic": "pendulum"})
    assert len(loops) == 1, "a confirmed check is optimised against as before"


def test_a_check_fitted_to_the_test_run_is_a_gap_in_the_evidence(tmp_path: Path) -> None:
    from core import evidence

    from core import frozen_protocol

    quest = tmp_path / "q"
    (quest / "needs").mkdir(parents=True)
    frozen_protocol.freeze(quest, {"oracles": [EXACT_90]}, approved_by="t", source="plan.md")
    (quest / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(
        {"status": "ok", "judged_by": "engine", "fitted_to_test_run": ["exact_90_deg"]}), encoding="utf-8")
    record = evidence.assess(quest, {}, settings={})
    assert "was changed after a test run measured it" in json.dumps(record)
    assert record.get("level") not in ("independently_validated", "publication_ready")


def test_the_paper_is_told_when_no_check_could_be_judged() -> None:
    record = {"status": accepted_checks.WENT_ON,
              "went_on": [{"name": "", "measured": None, "unmeasured": "the plan declares no oracle"}]}
    assert "no known-answer check could be judged" in accepted_checks.failing_disclosure(record)


# --- after the second review -------------------------------------------------------------------------------------------


def test_a_check_corrected_to_another_models_value_is_never_independent_evidence(tmp_path: Path) -> None:
    from core import evidence, frozen_protocol

    quest = tmp_path / "q"
    (quest / "needs").mkdir(parents=True)
    frozen_protocol.freeze(quest, {"oracles": [EXACT_90]}, approved_by="t", source="plan.md")

    def level_with(source: str) -> str:
        (quest / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(
            {"status": "ok", "judged_by": "engine",
             "corrected": {"exact_90_deg": {"from": 1.18034, "to": 2.3678, "source": source}}}), encoding="utf-8")
        return json.dumps(evidence.assess(quest, {}, settings={}))

    assert "corrected to another model's value" in level_with("recompute")
    assert "corrected to another model's value" not in level_with("arithmetic"), "the plan's own arithmetic is no gap"


def test_the_fitted_mark_lives_with_the_record_of_the_engines_plan_changes(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng._oracles_added_write({"oracles": ["exact_90_deg"], "shown": False, "fitted": ["exact_90_deg"]})
    assert eng._fitted_to_test_run([EXACT_90]) == ["exact_90_deg"]
    from core import rerun_from

    assert ".fi/oracles_added.json" not in rerun_from._FROM_DESIGN, "a redesign keeps it, as it keeps plan.md"


def test_a_recheck_by_any_model_that_writes_the_checks_is_not_independent(tmp_path: Path) -> None:
    from core.config import ProviderConfig as PC

    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng.config = eng.config.model_copy(update={"provider": PC(name="openai", model="B",
                                                              node_models={"plan": "A", "oracle_review": "B"})})
    eng._client = type("M", (), {"chat": AsyncMock(return_value='{"expected": 2.3678, "how": "x"}')})()
    eng._prompts = {**getattr(eng, "_prompts", {}), "oracle_recompute": __import__("string").Template("$topic $check")}
    import core.oracle_triage as ot

    original = ot.recompute_prompt
    ot.recompute_prompt = lambda template, **kw: "prompt"
    try:
        import asyncio

        entry, _ = asyncio.run(eng._recompute_expected({"topic": "t"}, EXACT_90, MEASURED))
    finally:
        ot.recompute_prompt = original
    assert entry["same_model"] is True, "plan_revise answers with B, the same model that rechecks"


# --- after the combined review of the plain check stops and the plain failure card -----------------------------------


def test_a_later_engine_change_never_drops_the_fitted_mark(tmp_path: Path) -> None:
    """A check fitted to the test run stays marked when FI later changes another check (a correction, a rewrite):
    losing the mark would let it count as independently validated."""
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng._oracles_added_write({"oracles": ["exact_90_deg"], "shown": False, "fitted": ["exact_90_deg"]})
    eng._note_engine_change(["other_check"], reason="FI corrected an expected value")
    assert eng._fitted_to_test_run([EXACT_90]) == ["exact_90_deg"]
    eng._oracles_added_write({**eng._oracles_added_read(), "shown": True})
    eng._note_engine_change(["third_check"])
    assert eng._fitted_to_test_run([EXACT_90]) == ["exact_90_deg"]


@pytest.mark.asyncio
@pytest.mark.parametrize("same_model, how_slip, corrects", [(True, False, False), (False, True, False),
                                                             (False, False, True)])
async def test_a_disputing_recheck_keeps_the_script_as_it_is_and_only_an_independent_one_corrects(
        tmp_path: Path, same_model: bool, how_slip: bool, corrects: bool) -> None:
    """Any recheck that disputes a check keeps the script from being bent towards a value it disputes (bending a correct
    script to a wrong check is the worse failure). Only an independent one may correct the check; the others leave it
    to go on unconfirmed."""
    eng = _engine_with_plan(tmp_path, EXACT_90)
    plain = {**EXACT_90, "reference": "derivation: the exact period from the elliptic integral (E2)"}  # no arithmetic
    entry = oracle_triage.recompute_entry(plain, "disputed", 2.3678, MEASURED, "x", model="m", same_model=same_model)
    if how_slip:
        entry["how_slip"] = True
    eng._recompute_expected = AsyncMock(return_value=(entry, True))  # type: ignore[method-assign]
    seed = eng.quest_root / "code" / "simulate.py"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text("x = 1\n", encoding="utf-8")
    record: dict[str, Any] = {"judged": [_judged("exact_90_deg", MEASURED)]}
    await eng._look_at_failing_checks({}, None, seed, [plain], [plain], record, set(), protocol={},
                                      timeout=10, case_env={})
    assert "exact_90_deg" in eng._oracle_disputed, "the script is not bent towards a disputed value"
    assert eng._independent_value(eng._oracle_proposals["exact_90_deg"]) is corrects
    assert eng._may_correct(eng._oracle_proposals["exact_90_deg"]) is False, "one recheck alone never corrects"


@pytest.mark.parametrize("text, expected", [
    ("derivation: level = log(1000) = 3", 3.0),  # log base 10 meant; the calculator's log is natural
    ("derivation: y = sin(30) = 0.5", 0.5),      # degrees meant; the calculator's sin is in radians
    ("derivation: y = cos(60) = 0.50", 0.5),
])
def test_a_bare_log_or_a_trig_function_of_a_bare_number_is_never_called_a_slip(text: str, expected: float) -> None:
    assert oracle_triage.arithmetic_slip(text, expected) is None
    assert oracle_triage.ambiguous(text.split("=")[1])


def test_unambiguous_forms_are_still_worked_out() -> None:
    assert oracle_triage.arithmetic_slip("derivation: level = log10(1000) = 4.00", 4.0)["computes"] == 3.0
    assert not oracle_triage.ambiguous("sin(30 deg)") and not oracle_triage.ambiguous("sin(pi/6)")


def test_a_correction_that_matches_the_measurement_says_so(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng._oracle_proposals["exact_90_deg"] = _confirmed(oracle_triage.arithmetic_proposal(
        EXACT_90, oracle_triage.arithmetic_slip(PENDULUM_REFERENCE)))
    assert eng._correct_expected_values([_judged("exact_90_deg", 2.367836)], [EXACT_90]) == ["exact_90_deg"]
    record = json.loads((eng.fi_dir / "oracle_corrections.json").read_text(encoding="utf-8"))
    assert record["exact_90_deg"].get("agrees_with_measurement") is True


def test_a_failing_check_under_warn_and_a_check_corrected_by_another_model_are_unconfirmed(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    (eng.quest_root / "needs").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({
        "status": "warned", "attempts": [{"judged": [{"name": "warned_check", "passed_by_engine": False},
                                                     {"name": "fine", "passed_by_engine": True}]}],
        "corrected": {"by_model": {"source": "recompute"}, "by_arithmetic": {"source": "arithmetic"},
                      "by_both": {"source": "arithmetic", "confirmed_by": {"source": "recompute", "value": 1.0}}}}),
        encoding="utf-8")
    out = eng._unconfirmed_checks()
    assert {"warned_check", "by_model", "by_arithmetic"} <= out and not ({"fine", "by_both"} & out)


def test_the_fill_report_asks_the_person_for_nothing() -> None:
    import launch

    source = Path(launch.__file__).read_text(encoding="utf-8")
    assert "ask again, or edit plan.md" not in source
    assert "fix the script and resume" not in json.dumps(todo._RESEARCH_INSTEAD)


# --- after the second, targeted review --------------------------------------------------------------------------------


def test_a_kept_recheck_is_judged_by_the_model_that_answered_not_by_todays_settings(tmp_path: Path) -> None:
    """Run 1 had no `oracle_review`: the plan's own model answered and the answer was kept. A person then names
    another reviewer and resumes: the kept answer is still the plan's own model's, never an independent one."""
    import asyncio

    from core import oracle_triage as ot
    from core.config import ProviderConfig as PC

    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    ot.write(eng.fi_dir, {"recompute": {ot.recompute_key(EXACT_90): {
        "name": "exact_90_deg", "recomputed": 2.3678, "how": "x", "model": "A"}}})
    eng.config = eng.config.model_copy(update={"provider": PC(name="openai", model="A",
                                                              node_models={"oracle_review": "B"})})
    eng._client = type("M", (), {"chat": AsyncMock(side_effect=AssertionError("a kept answer is not asked again"))})()
    entry, called = asyncio.run(eng._recompute_expected({"topic": "t"}, EXACT_90, MEASURED))
    assert not called and entry["same_model"] is True


def test_a_kept_recheck_by_an_unknown_model_is_the_same_model(tmp_path: Path) -> None:
    import asyncio

    from core import oracle_triage as ot
    from core.config import ProviderConfig as PC

    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    ot.write(eng.fi_dir, {"recompute": {ot.recompute_key(EXACT_90): {
        "name": "exact_90_deg", "recomputed": 2.3678, "how": "x", "model": "the provider's default model"}}})
    eng.config = eng.config.model_copy(update={"provider": PC(name="openai", model="A",
                                                              node_models={"oracle_review": "B"})})
    entry, _ = asyncio.run(eng._recompute_expected({"topic": "t"}, EXACT_90, MEASURED))
    assert entry["same_model"] is True


@pytest.mark.parametrize("expression", ["cos(2*30)", "sin(90-60)", "atan2(1, 1)"])
def test_a_trig_function_whose_argument_names_no_angle_unit_is_ambiguous(expression: str) -> None:
    assert oracle_triage.ambiguous(expression)


def test_the_card_never_changes_the_proposals_it_is_given(tmp_path: Path) -> None:
    proposals: list[dict[str, Any]] = []
    _card(tmp_path, [EXACT_90], [_judged("exact_90_deg", MEASURED)], [], proposals=proposals)
    assert proposals == [], "the card works on its own copy"


def test_a_kept_recheck_by_another_model_stays_independent(tmp_path: Path) -> None:
    import asyncio

    from core import oracle_triage as ot
    from core.config import ProviderConfig as PC

    eng = _engine_with_plan(tmp_path, EXACT_90)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    ot.write(eng.fi_dir, {"recompute": {ot.recompute_key(EXACT_90): {
        "name": "exact_90_deg", "recomputed": 2.3678, "how": "x", "model": "C", "same_model": False}}})
    eng.config = eng.config.model_copy(update={"provider": PC(name="openai", model="A",
                                                              node_models={"oracle_review": "C"})})
    entry, _ = asyncio.run(eng._recompute_expected({"topic": "t"}, EXACT_90, MEASURED))
    assert entry["same_model"] is False


def test_the_analysis_never_calls_the_plans_own_model_another_model() -> None:
    from core import oracle_check

    base = {"name": "exact_90_deg", "value": MEASURED, "expected": 1.18034, "limit": 1e-5, "passed_by_engine": False,
            "measured_by": "engine", "disputed_expected": 2.3678}
    same = oracle_check.analysis_note([{**base, "disputed_by": "recompute_same_model"}])
    other = oracle_check.analysis_note([{**base, "disputed_by": "recompute"}])
    assert "another model" not in same and "not an independent check" in same
    assert "another model" in other


# --- the plan's arithmetic alone only lowers a result; a correction needs a second, independent working ------------------


@pytest.mark.parametrize("reference, written", [
    ("derivation: y = sin(0.5*pi - 0.01) = 1.000000", 1.0),   # a peak inside the range of the rounded constants
    ("derivation: T/T0 = 1 + (pi/4)**2/16 = 1.0400", 1.04),   # a truncated series written with "="
])
def test_a_derivation_the_calculator_may_misread_can_only_set_a_check_aside(tmp_path: Path, reference: str,
                                                                            written: float) -> None:
    oracle = {**EXACT_90, "expected": written, "reference": reference}
    eng = _engine_with_plan(tmp_path, oracle)
    slip = oracle_triage.plan_slip(oracle)
    if slip is not None:
        eng._oracle_proposals["exact_90_deg"] = oracle_triage.arithmetic_proposal(oracle, slip)
    assert eng._correct_expected_values([_judged("exact_90_deg", 0.5)], [oracle]) == []
    assert _planned(eng)["expected"] == written, "never rewritten on the arithmetic alone"


async def _look(eng: Engine, oracle: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    eng._recompute_expected = AsyncMock(return_value=(entry, True))  # type: ignore[method-assign]
    seed = eng.quest_root / "code" / "simulate.py"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text("x = 1\n", encoding="utf-8")
    record: dict[str, Any] = {"judged": [_judged(str(oracle["name"]), MEASURED)]}
    await eng._look_at_failing_checks({}, None, seed, [oracle], [oracle], record, set(), protocol={},
                                      timeout=10, case_env={})
    return record


@pytest.mark.asyncio
async def test_the_real_pendulum_slip_with_only_the_plans_own_model_goes_on_unconfirmed(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    same = oracle_triage.recompute_entry(EXACT_90, "agrees", 1.1803415, MEASURED, PENDULUM_RECHECK,
                                         model="gemma4:31b-cloud", same_model=True)
    record = await _look(eng, EXACT_90, same)
    assert "exact_90_deg" in eng._oracle_set_aside, "not sent to the repair"
    assert not eng._may_correct(eng._oracle_proposals["exact_90_deg"])
    assert eng._correct_expected_values(record["judged"], [EXACT_90]) == []
    assert _planned(eng)["expected"] == 1.18034
    card = _card(tmp_path, [EXACT_90], record["judged"], record["triage"])
    assert "gives 2.36784, not the 1.18034" in card["why"] and "does not rely on this check" in card["why"]


@pytest.mark.asyncio
async def test_the_real_pendulum_slip_confirmed_by_another_model_is_corrected(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    other = oracle_triage.recompute_entry(EXACT_90, "disputed", 2.367842, MEASURED, "T0 * (2/pi) * K(0.5)",
                                          model="another-model", same_model=False)
    record = await _look(eng, EXACT_90, other)
    assert eng._may_correct(eng._oracle_proposals["exact_90_deg"])
    assert eng._correct_expected_values(record["judged"], [EXACT_90]) == ["exact_90_deg"]
    after = _planned(eng)
    assert abs(after["expected"] - 2.36784) < 1e-5 and after["tolerance"] == EXACT_90["tolerance"]


def test_the_approx_45_degree_check_is_never_rewritten_its_measure_is_a_formula(tmp_path: Path) -> None:
    check = {"name": "large_amplitude_period", "kind": "published_value", "expected": 1.031, "tolerance": 0.001,
             "tolerance_mode": "relative", "case": {"amplitude_deg": 45, "dt": 0.001}, "measure": "T / T0",
             "reference": "derivation: For A=45 deg, T/T0 = (2/pi)*ellipk(sin(pi/8)**2) approx 1.031"}
    assert oracle_triage.plan_slip(check) is not None, "the slip is seen (it sets the check aside)"
    assert not oracle_triage.correctable(check), "a formula measure is never corrected"



# --- the second external review of the two-source rule --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_second_source_never_sees_the_plans_working(tmp_path: Path) -> None:
    eng = _engine_with_plan(tmp_path, EXACT_90)
    other = oracle_triage.recompute_entry(EXACT_90, "disputed", 2.367842, MEASURED, "x", model="m2", same_model=False)
    await _look(eng, EXACT_90, other)
    asked = eng._recompute_expected.call_args.args[1]  # type: ignore[attr-defined]
    assert asked["reference"] == oracle_triage.BLIND_REFERENCE and "1.85407" not in json.dumps(asked)


def test_a_recheck_shown_the_working_or_agreeing_with_the_plan_confirms_nothing() -> None:
    proposal = oracle_triage.arithmetic_proposal(EXACT_90, oracle_triage.arithmetic_slip(PENDULUM_REFERENCE))
    shown = oracle_triage.recompute_entry(EXACT_90, "disputed", 2.367842, MEASURED, "x", model="m2", same_model=False)
    assert Engine._second_source(EXACT_90, proposal, shown) is None, "not blind to the plan's working"  # noqa: SLF001
    agrees = {**shown, "verdict": "agrees", "blind": True}
    assert Engine._second_source(EXACT_90, proposal, agrees) is None  # noqa: SLF001


def test_a_confirmation_is_never_taken_from_a_file_or_kept_for_a_changed_check(tmp_path: Path) -> None:
    # A forged record: "corrected to the measured value, confirmed".
    eng = _engine_with_plan(tmp_path, EXACT_90)
    forged = {"name": "exact_90_deg", "expected": MEASURED, "tolerance": 1e-5, "source": "arithmetic",
              "confirmed_by": {"source": "recompute", "value": MEASURED}}
    (eng.quest_root / "needs").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({
        "proposed_changes": [forged], "disputed": ["exact_90_deg"],
        "fingerprints": {"exact_90_deg": oracle_triage.fingerprint(EXACT_90)}}), encoding="utf-8")
    eng._restore_oracle_disputes([EXACT_90])
    assert "confirmed_by" not in eng._oracle_proposals["exact_90_deg"]
    # Even placed in memory, a "confirmation" the check's current derivation does not give is refused.
    plain = {**EXACT_90, "reference": "derivation: the exact period from the elliptic integral (E2)"}
    eng2 = _engine_with_plan(tmp_path / "b", plain)
    eng2._oracle_proposals["exact_90_deg"] = dict(forged)
    assert eng2._correct_expected_values([_judged("exact_90_deg", MEASURED)], [plain]) == []
    assert _planned(eng2)["expected"] == 1.18034


def test_a_name_at_the_start_of_a_reason_is_not_lowered() -> None:
    from core import engine as engine_module

    assert engine_module._lower_first("FI worked it out") == "FI worked it out"
    assert engine_module._lower_first("The plan says") == "the plan says"


def test_the_card_drops_an_arithmetic_finding_about_a_value_the_check_no_longer_has() -> None:
    entry = oracle_triage.arithmetic_entry(EXACT_90, oracle_triage.arithmetic_slip(PENDULUM_REFERENCE))
    now = [{"id": "exact_90_deg", "status": "failed", "expected": 2.36784}]
    assert oracle_card._current([entry], now) == []  # noqa: SLF001
    assert oracle_card._current([entry], [{**now[0], "expected": 1.18034}]) == [entry]  # noqa: SLF001



def test_evidence_counts_an_arithmetic_only_correction_as_a_gap(tmp_path: Path) -> None:
    from core import evidence, frozen_protocol

    quest = tmp_path / "q"
    (quest / "needs").mkdir(parents=True)
    frozen_protocol.freeze(quest, {"oracles": [EXACT_90]}, approved_by="t", source="plan.md")

    def said(fix: dict[str, Any]) -> str:
        (quest / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(
            {"status": "ok", "judged_by": "engine",
             "corrected": {"exact_90_deg": {"from": 1.18034, "to": 2.36784, **fix}}}), encoding="utf-8")
        return json.dumps(evidence.assess(quest, {}, settings={}))

    assert "from the plan's own arithmetic alone" in said({"source": "arithmetic"})
    assert "from the plan's own arithmetic alone" not in said(
        {"source": "arithmetic", "confirmed_by": {"source": "recompute", "value": 2.36784}})
