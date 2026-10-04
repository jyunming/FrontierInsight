"""Five things a real gemma4 quest on the pendulum (2026-10-05, after the known-answer checks stopped asking people)
showed FI still got wrong, each pinned with that quest's own strings and numbers.

1. The plan wrote "derivation: For A=45 deg, T/T0 = (2/pi)*ellipk(sin(pi/8)**2) approx 1.031". That multiplies out to
   1.0400 (and 1.0400 is right), but FI did not work it out: "approx" was not read as the result sign, ``ellipk`` was
   not on the calculator's list, and a 0.9% gap sat under the 1% floor although it is nine times the check's tolerance.
2. The simulation measured 0.519923, half of 1.0400: half a period. The repair was never told.
3. A repair whose code could not be used ("must parse and define oracle()") still used up one of the two repairs.
4. The console gave the reason "at a smaller step the result does not settle" for 0.519923 / 0.519923 / 0.520048 -- a
   result that stays the same -- and printed the sentence twice.
5. needs/ENVIRONMENT.json held a whole traceback: `pip freeze` crashed (NotADirectoryError) on an editable install
   whose folder had been deleted, and no package list was kept.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import oracle_card, oracle_forms, oracle_triage
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, _last_error_line

REFERENCE_45 = "derivation: For A=45 deg, T/T0 = (2/pi)*ellipk(sin(pi/8)**2) approx 1.031"
CHECK_45 = {"name": "large_amplitude_period", "kind": "published_value", "check": "T/T0 at 45 degrees",
            "expected": 1.031, "tolerance": 0.001, "tolerance_mode": "relative", "reference": REFERENCE_45,
            "case": {"amplitude_deg": 45, "dt": 0.001}, "measure": "T / T0"}
EXACT_45 = 1.039973343196804  # (2/pi) * scipy.special.ellipk(sin(pi/8)**2)
HALF_STEPS = [0.5199228968951338, 0.5199228968951529, 0.5200475188744478]  # dt = 0.001, 0.0005, 0.00025


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(topic="pendulum", title="rerun", provider=ProviderConfig(name="openai"),
                        engine=EngineConfig(max_iterations=1, review_loop=False),
                        execution=ExecutionConfig(sandbox="venv"), knowledge=KnowledgeConfig(enabled=False),
                        output=OutputConfig(output_dir=tmp_path / "out")))
    eng._oracle_proposals, eng._oracle_disputed, eng._oracle_set_aside, eng._oracle_noisy = {}, set(), {}, set()
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    return eng


# --- 1. a derivation written with "approx" and ellipk is worked out ------------------------------------------------------


def test_the_calculator_knows_ellipk_in_scipys_convention_only_for_fis_own_working() -> None:
    assert abs(oracle_forms.evaluate("ellipk(0.5)", {}, special=True).value - 1.8540746773013719) < 1e-12
    assert abs(oracle_forms.evaluate("(2/pi)*ellipk(sin(pi/8)**2)", {}, special=True).value - EXACT_45) < 1e-12
    # A check's own formula (what the simulation returns, worked out) never gets it.
    assert oracle_forms.evaluate("ellipk(0.5)", {}).value is None


def test_the_real_approx_derivation_is_a_slip_and_its_value_is_the_one_fi_works_out() -> None:
    slip = oracle_triage.plan_slip(CHECK_45)
    assert slip is not None and slip["written"] == 1.031 and abs(slip["computes"] - EXACT_45) < 1e-9
    proposal = oracle_triage.arithmetic_proposal(CHECK_45, slip)
    assert proposal["source"] == "arithmetic" and abs(proposal["expected"] - EXACT_45) < 1e-9
    assert proposal["tolerance"] == 0.001 and proposal["tolerance_mode"] == "relative", "the tolerance is kept"


@pytest.mark.parametrize("sign", ["approx", "approximately", "≈", "~", "about"])
def test_every_approximate_sign_is_read_and_a_right_answer_is_left_alone(sign: str) -> None:
    wrong = f"derivation: T/T0 = (2/pi)*ellipk(sin(pi/8)**2) {sign} 1.031"
    right = f"derivation: T/T0 = (2/pi)*ellipk(sin(pi/8)**2) {sign} 1.040"
    assert oracle_triage.arithmetic_slip(wrong, 1.031, 0.001031) is not None
    assert oracle_triage.arithmetic_slip(right, 1.040, 0.00104) is None


@pytest.mark.parametrize("text, expected, limit", [
    # Rounded to the digits it writes: never a slip, however tight the check.
    ("derivation: T/T0 = (2/pi)*ellipk(sin(pi/8)**2) approx 1.04", 1.04, 1e-8),
    # ellipk given the modulus k (the other convention): not worked out, so never "corrected".
    ("derivation: T/T0 = (2/pi)*ellipk(sin(pi/8)) = 1.040", 1.040, 1e-3),
    # The bare log and the trig of a bare number stay ambiguous.
    ("derivation: level = log(1000) approx 3", 3.0, 1e-3),
    ("derivation: y = sin(30) approx 0.5", 0.5, 1e-3),
    # Prose with "about" and no arithmetic before it.
    ("derivation: the period is about 2.0 s", 2.0, 1e-3),
])
def test_rounding_the_other_convention_and_ambiguous_forms_are_never_slips(text: str, expected: float,
                                                                           limit: float) -> None:
    assert oracle_triage.arithmetic_slip(text, expected, limit) is None


def test_without_a_tolerance_the_one_percent_floor_still_holds() -> None:
    # 0.9% off: within the floor when the check's tolerance is not known.
    assert oracle_triage.arithmetic_slip("derivation: (2/pi)*ellipk(sin(pi/8)**2) approx 1.031", 1.031) is None


# --- 2. a measured value that is a simple multiple of FI's own value -----------------------------------------------------


def test_half_the_value_fi_worked_out_is_named_as_half_a_quantity() -> None:
    found = oracle_triage.multiple_of(HALF_STEPS[0], EXACT_45, 0.001 * EXACT_45)
    assert found is not None and found[0] == 0.5 and "half a period" in found[2]
    assert oracle_triage.multiple_of(EXACT_45 * 1.0001, EXACT_45, 0.001 * EXACT_45) is None, "agreeing is not a multiple"
    assert oracle_triage.multiple_of(0.7, EXACT_45, 0.001 * EXACT_45) is None


def test_the_hint_reaches_the_repair_and_points_to_the_script(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    corrected = {**CHECK_45, "expected": EXACT_45}
    (eng.fi_dir).mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / "oracle_corrections.json").write_text(json.dumps(
        {"large_amplitude_period": {"from": 1.031, "to": EXACT_45, "source": "arithmetic"}}), encoding="utf-8")
    attempt: dict[str, Any] = {}
    judged = [{"name": "large_amplitude_period", "value": HALF_STEPS[0], "passed_by_engine": False}]
    hints = eng._multiple_hints([corrected], judged, attempt, set())
    assert len(hints) == 1 and "half" in hints[0] and "0.519923" in hints[0]
    entry = attempt["triage"][0]
    assert entry["kind"] == "multiple" and entry["points_to"] == "script"
    # Only against a value FI worked out itself: the plan's number alone (no derivation FI can work out) gives none.
    plain = {**CHECK_45, "expected": EXACT_45, "reference": "the exact period (E2)"}
    assert _engine(tmp_path / "other")._multiple_hints([plain], judged, {}, set()) == []


def test_the_card_names_the_multiple_before_the_step_runs(tmp_path: Path) -> None:
    triage = [oracle_triage.step_entry(CHECK_45, "dt", [0.001, 0.0005, 0.00025],
                                       oracle_triage.step_verdict(CHECK_45, HALF_STEPS, order=None)),
              oracle_triage.multiple_entry(CHECK_45, HALF_STEPS[0], EXACT_45, (0.5, "half", "half of the quantity"),
                                           "its correction of the plan's arithmetic")]
    checks = [{"id": "large_amplitude_period", "status": "failed"}]
    assert oracle_card.leaning(checks, triage, []) == "script"
    why = oracle_card._why("script", checks, triage, [])  # noqa: SLF001
    assert why.startswith("what was measured is half the value FI worked out itself"), why


# --- 3. a repair whose code cannot be used is not spent (once) ----------------------------------------------------------


def test_an_unusable_repair_is_not_counted_once_and_the_loop_stays_bounded() -> None:
    import inspect

    src = inspect.getsource(Engine._oracle_gate)
    assert "unusable_left = 1" in src and 'outcome == "not_usable" and unusable_left > 0' in src
    # The exemption comes before the line that spends a repair, and is used up once.
    assert src.index('outcome == "not_usable" and unusable_left > 0') < src.index("repairs_left -= 1")
    assert "unusable_left -= 1" in src


# --- 4. the reason is the strongest evidence, a steady result is said as such, said once --------------------------------


def test_a_result_that_stays_the_same_at_smaller_steps_is_steady_not_unsettled() -> None:
    verdict = oracle_triage.step_verdict(CHECK_45, HALF_STEPS, order=None)
    assert verdict is not None and verdict["verdict"] == "steady_elsewhere"
    entry = oracle_triage.step_entry(CHECK_45, "dt", [0.001, 0.0005, 0.00025], verdict)
    assert "stays at the same value" in entry["cause"]["text"]
    why = oracle_card._why("script", [{"id": "large_amplitude_period", "status": "failed"}], [entry], [])  # noqa: SLF001
    assert "does not settle" not in why and "stays the same" in why


def test_a_slip_in_the_plans_arithmetic_is_the_reason_given_first() -> None:
    slip = oracle_triage.plan_slip(CHECK_45)
    step = oracle_triage.step_entry(CHECK_45, "dt", [0.001, 0.0005, 0.00025],
                                    oracle_triage.step_verdict(CHECK_45, HALF_STEPS, order=None))
    triage = [step, oracle_triage.arithmetic_entry(CHECK_45, slip)]
    why = oracle_card._why("unclear", [{"id": "large_amplitude_period", "status": "failed"}], triage, [])  # noqa: SLF001
    assert why.startswith("the plan's own working for the expected value does not add up"), why


def test_the_went_on_sentence_is_printed_once_per_quest_run() -> None:
    import inspect

    src = inspect.getsource(Engine._oracle_gate)
    assert "_unconfirmed_said" in src and "if line not in said:" in src


# --- 5. a pip freeze that crashes keeps a package list, and the record keeps one line ------------------------------------


PIP_TRACEBACK = ("ERROR: Exception:\nTraceback (most recent call last):\n  File \"...subprocess.py\", line 1538, in "
                 "_execute_child\n    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,\n"
                 "                       ^^^^^^^^^^^^^^^^^^^^^^^\n"
                 "NotADirectoryError: [WinError 267] The directory name is invalid\n")


def test_the_record_keeps_the_error_line_not_the_traceback() -> None:
    assert _last_error_line(PIP_TRACEBACK) == "NotADirectoryError: [WinError 267] The directory name is invalid"
    assert _last_error_line("") == "" and _last_error_line("just one line") == "just one line"


def test_the_installed_distributions_are_listed_without_pip() -> None:
    listed = asyncio.run(Engine._installed_distributions(sys.executable))
    assert listed and all("==" in line for line in listed) and listed == sorted(listed)
    assert any(line.lower().startswith("pytest==") for line in listed)


@pytest.mark.asyncio
async def test_a_crashing_pip_freeze_falls_back_and_says_why_in_one_line(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path)

    class _Proc:
        returncode = 2

        async def communicate(self):  # noqa: ANN202
            return b"", PIP_TRACEBACK.encode()

    class _Tree:
        proc = _Proc()

        @classmethod
        async def start(cls, *args, **kw):  # noqa: ANN002, ANN003, ANN206
            return cls()

        async def aclose(self, aborted: bool = False) -> None:
            return None

    monkeypatch.setattr("core.engine.AsyncProcessTree", _Tree)

    async def listed(py: Any) -> list[str]:
        return ["numpy==2.0.0", "scipy==1.14.0"]

    monkeypatch.setattr(Engine, "_installed_distributions", staticmethod(listed))
    await eng._record_environment("after installing the experiment's packages")
    record = json.loads((eng.quest_root / "needs" / "ENVIRONMENT.json").read_text(encoding="utf-8"))
    assert record["packages"] == ["numpy==2.0.0", "scipy==1.14.0"] and "packages_error" not in record
    assert record["packages_source"].endswith("(pip freeze failed: NotADirectoryError: [WinError 267] The directory "
                                              "name is invalid)")
    assert "Traceback" not in json.dumps(record)
