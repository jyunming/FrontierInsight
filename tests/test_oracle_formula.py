"""FI computes each expected value itself, before anything runs (core/oracle_forms.py::formula_findings, applied by
Engine._apply_expected_formulas inside the one request that already goes back to the plan about its checks), and the
arithmetic-slip detector holds a written number to half a unit in its last place (core/oracle_triage.py::_close).

Fake models only: nothing here calls a real model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import frozen_protocol, oracle_forms as of, oracle_triage as tri, plan
from core.engine import Engine
from tests.test_oracle_review import CHECK, MODEL, _Model, _config

# The two real checks of a quest (verbatim): both expected values are wrong, and the simulation was right.
PERIOD = {"name": "exact_period_check", "kind": "published_value", "check": "T_num", "expected": 2.112,
          "tolerance": 0.01, "tolerance_mode": "relative",
          "reference": "derivation: L=1, g=9.81, T0=2.006. For A=30 deg, k=sin(15 deg)=0.2588, K(0.2588^2)=1.654. "
                       "T = 2.006 * (2/pi) * 1.654 = 2.1118",
          "case": {"amplitude_deg": 30, "dt": 0.001}, "measure": "T_num"}
LARGE = {"name": "exact_formula_large_angle", "kind": "published_value", "check": "T_exact", "expected": 2.366,
         "tolerance": 0.001, "tolerance_mode": "absolute",
         "reference": "derivation: L=1, g=9.81, T0=2.006. For A=90 deg, k=sin(45 deg)=0.7071, K(0.7071^2)=1.8541. "
                      "T = 2.006 * (2/pi) * 1.8541 = 2.366",
         "case": {"amplitude_deg": 90}, "measure": "T_exact"}
PERIOD_FORMULA = "2*pi*sqrt(1/9.81)*(2/pi)*ellipk(sin(15*pi/180)**2)"
LARGE_FORMULA = "2*pi*sqrt(1/9.81)*(2/pi)*ellipk(sin(amplitude_deg*pi/360)**2)"
NO_REVIEW = "I think these checks look fine."  # a reply that is no review: the second reading adds nothing here


def _protocol(*oracles: dict[str, Any]) -> dict[str, Any]:
    return {"grid": {"dt": [0.01]}, "model": MODEL, "oracles": list(oracles)}


def _planned(engine: Engine) -> dict[str, dict[str, Any]]:
    return engine._planned_oracles()


def _log(engine: Engine) -> str:
    return (engine.fi_dir / "run.log").read_text(encoding="utf-8")


async def _plan_step(engine: Engine) -> None:
    await engine._node_plan({"topic": engine.config.topic, "literature": []})


def _spy_changes(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], str]]:
    seen: list[tuple[list[str], str]] = []
    real = engine._note_engine_change

    def spy(changed, removed=None, reason=""):  # noqa: ANN001
        seen.append((list(changed), reason))
        return real(changed, removed, reason)

    monkeypatch.setattr(engine, "_note_engine_change", spy)
    return seen


@pytest.mark.asyncio
async def test_two_real_wrong_expected_values_are_computed_by_fi_before_anything_runs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Engine(_config(tmp_path))
    seen = _spy_changes(engine, monkeypatch)
    model = _Model(_protocol({**PERIOD, "expected_formula": PERIOD_FORMULA},
                             {**LARGE, "expected_formula": LARGE_FORMULA}), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)

    # One request, naming both numbers of each check FI computed (no measured value exists, so nothing leaks).
    assert len(model.revisions) == 1
    assert "FI computed your `expected_formula`" in model.revisions[0] and "2.04098988952" in model.revisions[0]
    checks = _planned(engine)
    assert checks["exact_period_check"]["expected"] == pytest.approx(2.04095, abs=1e-4)
    assert checks["exact_formula_large_angle"]["expected"] == pytest.approx(2.36780, abs=1e-4)
    for name, original in (("exact_period_check", PERIOD), ("exact_formula_large_angle", LARGE)):
        for key in ("tolerance", "tolerance_mode", "case", "measure"):
            assert checks[name][key] == original[key], key
    assert plan.history(engine.quest_root)[-1]["by"] == "engine"
    assert any(set(names) >= {"exact_period_check", "exact_formula_large_angle"} for names, _ in seen)
    assert all("measured" not in why for _, why in seen)
    log = _log(engine)
    assert "before anything ran, FI computed the expected value of 'exact_period_check' from the plan's own formula" in log
    assert "(the plan had written 2.112)" in log
    # The frozen protocol carries the corrected values, and does not say a person approved them.
    engine._freeze_protocol_if_due({"iteration": 0})
    frozen = frozen_protocol.load(engine.quest_root)
    by_name = {o["name"]: o for o in frozen["protocol"]["oracles"]}
    assert by_name["exact_period_check"]["expected"] == pytest.approx(2.04095, abs=1e-4)
    assert by_name["exact_formula_large_angle"]["expected"] == pytest.approx(2.36780, abs=1e-4)
    assert by_name["exact_period_check"]["expected_formula"] == PERIOD_FORMULA
    assert "nobody approved" in frozen["approved_by"]


@pytest.mark.asyncio
async def test_a_plan_that_corrects_its_own_number_when_shown_the_two_is_not_overruled(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))

    def fix(block: dict[str, Any]) -> dict[str, Any]:
        oracles = [{**o, "expected": 2.04099} for o in block["protocol"]["oracles"]]
        return {**block, "protocol": {**block["protocol"], "oracles": oracles}}

    model = _Model(_protocol({**PERIOD, "expected_formula": PERIOD_FORMULA}), NO_REVIEW, revise=fix)
    engine._client = model
    await _plan_step(engine)
    assert len(model.revisions) == 1
    assert _planned(engine)["exact_period_check"]["expected"] == 2.04099, "the plan's own corrected number stands"
    assert "FI computed the expected value" not in _log(engine)


@pytest.mark.asyncio
async def test_a_formula_that_agrees_is_left_alone_and_never_asked_about(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    good = {**PERIOD, "expected": 2.04099, "expected_formula": PERIOD_FORMULA}
    model = _Model(_protocol(good), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)
    assert model.revisions == []
    assert _planned(engine)["exact_period_check"]["expected"] == 2.04099
    assert not (engine.fi_dir / "oracles_added.json").is_file()


@pytest.mark.asyncio
async def test_a_missing_formula_is_asked_once_and_if_still_missing_the_check_stays_as_written(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(_protocol(dict(PERIOD)), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)
    assert len(model.revisions) == 1
    assert "give `expected_formula`" in model.revisions[0] and "ellipk(m)" in model.revisions[0]
    assert _planned(engine)["exact_period_check"]["expected"] == 2.112
    assert "FI could not compute the expected value of 'exact_period_check' itself" in _log(engine)
    assert not (engine.fi_dir / "oracles_added.json").is_file(), "nothing changed, so nothing is recorded as changed"


@pytest.mark.asyncio
async def test_a_formula_the_calculator_cannot_read_is_asked_about_and_then_left(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(_protocol({**PERIOD, "expected_formula": "T0 * ellipk(m)"}), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)
    assert len(model.revisions) == 1 and "cannot be computed" in model.revisions[0]
    assert _planned(engine)["exact_period_check"]["expected"] == 2.112


@pytest.mark.asyncio
async def test_a_second_pass_neither_asks_again_nor_applies_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(_protocol({**PERIOD, "expected_formula": PERIOD_FORMULA}), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)
    versions = len(plan.history(engine.quest_root))
    progress = json.loads((engine.fi_dir / "oracle_review.json").read_text(encoding="utf-8"))
    assert progress["asked"] and progress["answered"] and progress["formulas"][0]["name"] == "exact_period_check"

    resumed = Engine(_config(tmp_path), resume_quest_id=engine.quest_id)
    resumed._client = model
    seen = _spy_changes(resumed, monkeypatch)
    (resumed.fi_dir / "oracle_guidance.json").unlink()  # as if the plan step were entered again
    await _plan_step(resumed)
    assert len(model.revisions) == 1
    assert len(plan.history(resumed.quest_root)) == versions and seen == []
    assert _planned(resumed)["exact_period_check"]["expected"] == pytest.approx(2.04099, abs=1e-4)


@pytest.mark.asyncio
async def test_once_the_protocol_is_frozen_nothing_new_happens(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    design = {"hypothesis": "h", "protocol": _protocol({**PERIOD, "expected_formula": PERIOD_FORMULA})}
    text = plan.render(engine.config.topic, {}, design)
    path = plan.plan_path(engine.quest_root)
    path.write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="model")
    engine._freeze_protocol_if_due({"iteration": 0})
    assert frozen_protocol.load(engine.quest_root) is not None
    model = _Model({}, NO_REVIEW)
    engine._client = model
    await engine._guide_oracles({"topic": engine.config.topic})
    assert model.revisions == [] and model.reviews == []
    assert engine._apply_expected_formulas(path) == []
    assert path.read_text(encoding="utf-8") == text
    assert _planned(engine)["exact_period_check"]["expected"] == 2.112


@pytest.mark.asyncio
async def test_a_check_expecting_zero_by_its_form_is_never_asked_for_a_formula(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    model = _Model(_protocol(dict(CHECK)), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)
    assert model.revisions == []
    assert of.formula_findings(_protocol(dict(CHECK))) == []
    assert "expected_formula" not in _planned(engine)["power_conservation"]


def test_blind_withholds_the_plans_own_working() -> None:
    shown = tri.blind({**PERIOD, "expected_formula": PERIOD_FORMULA})
    assert "expected_formula" not in shown and PERIOD_FORMULA not in json.dumps(shown)
    assert tri.recompute_key({**PERIOD, "expected_formula": PERIOD_FORMULA}) == tri.recompute_key(PERIOD), (
        "what the blind prompt shows is all the recompute depends on")


def test_the_prompt_language_is_generated_from_the_calculators_own_tables() -> None:
    for name in list(of.FUNCTIONS) + list(of.SPECIAL_FUNCTIONS):
        assert name in of.EXPECTED_FORMULA_LANGUAGE, name
    from core.engine import _PLAN_DIRECTIVE
    assert "@@FORMULA@@" not in _PLAN_DIRECTIVE and "expected_formula" in _PLAN_DIRECTIVE
    assert "ellipk(m) with m = k**2" in _PLAN_DIRECTIVE and "radians" in _PLAN_DIRECTIVE


# --- the arithmetic-slip detector --------------------------------------------------------------------------------------


def test_the_large_angle_derivation_is_a_slip_the_check_can_tell_apart() -> None:
    slip = tri.plan_slip(LARGE)
    assert slip is not None and slip["computes"] == pytest.approx(2.3678, abs=1e-4) and slip["written"] == 2.366


def test_a_correct_derivation_with_rounded_constants_inside_its_tolerance_is_not_a_slip() -> None:
    right = {**LARGE, "expected": 2.3678, "reference": LARGE["reference"].replace("= 2.366", "= 2.3678")}
    assert tri.plan_slip(right) is None
    rounded = {**LARGE, "expected": 2.368, "reference": LARGE["reference"].replace("= 2.366", "= 2.368")}
    assert tri.plan_slip(rounded) is None, "a last digit rounded the other way is no slip"


def test_the_slack_of_a_written_number_is_half_a_unit_in_its_last_place() -> None:
    assert tri._close(2.3678, 2.366, "2.366", limit=0.001) is False
    assert tri._close(2.3678, 2.3675, "2.3675", limit=0.0001) is False
    assert tri._close(1.854073, 1.85407, "1.85407") is True
    assert tri._close(1.85408, 1.85407, "1.85407", limit=1e-9) is False, "0.00001 apart, half a unit is 0.000005"
    assert tri._close(0.0021, 0.0020, "2.0e-3", limit=1e-9) is False
    assert tri._close(0.00204, 0.0020, "2.0e-3", limit=1e-9) is True
    assert tri._close(2.3678, 2.3676, "2.3676", approximate=True, limit=1e-9) is True, "approximate: five times looser"
    assert tri._close(2.3678, 2.366, "2.366", approximate=True, limit=0.001) is True, "approximate: the floor holds"


@pytest.mark.asyncio
async def test_after_the_simulation_ran_nothing_is_computed_or_asked(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    engine._audit("node_started", node="execute")  # the trace outlives a rerun from the plan
    model = _Model(_protocol({**PERIOD, "expected_formula": PERIOD_FORMULA}), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)
    assert model.revisions == []
    assert _planned(engine)["exact_period_check"]["expected"] == 2.112


@pytest.mark.asyncio
async def test_a_formula_that_stays_missing_is_said_in_plan_md(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    engine._client = _Model(_protocol(dict(PERIOD)), NO_REVIEW)
    await _plan_step(engine)
    assert "FI could not compute the expected value of 'exact_period_check' itself" in plan.plan_path(
        engine.quest_root).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_formula_that_can_be_read_two_ways_is_asked_about_and_never_applied(tmp_path: Path) -> None:
    case_30 = {"name": "s30", "kind": "special_case", "check": "sin of the angle", "expected": 0.5, "tolerance": 1e-6,
               "case": {"angle_deg": 30}, "measure": "s", "reference": "derivation: sin(30 deg) = 0.5"}
    engine = Engine(_config(tmp_path))
    model = _Model(_protocol({**case_30, "expected_formula": "sin(30)"}), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)
    assert len(model.revisions) == 1 and "can only be read one way" in model.revisions[0]
    assert _planned(engine)["s30"]["expected"] == 0.5, "sin(30) read in radians must not replace the expected value"
    assert "can be read two ways" in _log(engine)

    engine = Engine(_config(tmp_path / "ok"))
    model = _Model(_protocol({**case_30, "expected": 0.51, "expected_formula": "sin(30*pi/180)"}), NO_REVIEW)
    engine._client = model
    await _plan_step(engine)
    assert _planned(engine)["s30"]["expected"] == pytest.approx(0.5, abs=1e-12)


def test_after_an_approximate_sign_a_small_gap_is_never_a_slip() -> None:
    """The number written after an approximate sign may be better than the expression before it (a truncated series
    written "≈ 1.0400"), so the approximate floor holds even with a tolerance."""
    base = "derivation: (2/pi)*ellipk(sin(pi/8)**2) ≈ "
    exact = 1.0400  # about; the written number is judged against the expression's own value
    value = tri.calculate("(2/pi)*ellipk(sin(pi/8)**2)")
    assert tri.arithmetic_slip(base + f"{value * 1.0005:.5f}", value * 1.0005, 0.001 * exact) is None
    assert tri.arithmetic_slip(base + f"{value * 1.004:.5f}", value * 1.004) is None
