"""FI measures the reported uncertainty against the true value the PLAN declares, from the trials, and never against
anything the code says about itself.

Real code (r6t kinetics, tests/fixtures/replay/r6t_code/, copied from the quest's ``code/``: a correct weighted Arrhenius
fit whose own 0/1 coverage flags were right) is run through FI's own trial harness. With the true activation energy typed
wrong inside the code (52000 where the plan says 50000), the code's data, its estimates and its own coverage flags all
agree with each other, and all are wrong: the old check read those flags and passed it. The code reports a standard
error, not an interval, so FI makes no coverage claim of its own: it measures how far the estimates lie from the plan's
true value in units of their own spread, and the quest stops."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from core import trial_checks, trial_runner
from core.execution import SharedInterpreterExecutor

CODE = Path(__file__).parent / "fixtures" / "replay" / "r6t_code"
GRID = {"noise_sigma_rel": [0.002, 0.01], "n_time": [24]}
THRESHOLDS = {"Ea_true_J_mol": 50000.0, "A_true_s_inv": 3000000.0, "R_J_mol_K": 8.314462618, "n_temperatures": 5,
              "true_k_span_ratio": 10.0, "time_start_over_k": 0.0, "time_end_over_k": 3.0, "c0_mol_L": 1.0,
              "T_ref_K": 350.0, "k_fit_lower_s_inv": 0.0001, "k_fit_upper_s_inv": 1.0, "solver_ftol": 1e-09,
              "solver_xtol": 1e-09, "solver_gtol": 1e-09}
PROTOCOL = {"grid": GRID, "thresholds": THRESHOLDS, "runs_per_setting": 40, "metrics": [
    {"id": "weighted_ea", "estimand": "the activation energy", "kind": "mean", "unit": "one trial",
     "known_value": {"true_value": "Ea_true_J_mol", "estimate": "weighted_ea", "standard_error": "weighted_ea_se",
                     "level": 0.95}}]}


def _judge(tmp_path: Path, plant: tuple[str, str] | None) -> trial_checks.Verdict:
    root = tmp_path / "quest"
    shutil.copytree(CODE, root / "code")
    if plant:
        sim = root / "code" / "simulate.py"
        text = sim.read_text(encoding="utf-8")
        assert text.count(plant[0]) == 1
        sim.write_text(text.replace(*plant), encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), sys.executable, root,
        "code/simulate.py", GRID, runs_per_setting=40, base_seed=0, deterministic=False, timeout_s=600,
        thresholds=THRESHOLDS, paired=True))
    data = json.loads(run.summary_path.read_text(encoding="utf-8"))
    return trial_checks.judge(data["cells"], PROTOCOL, data.get("thresholds"), {}, repairs_left=False)


def test_real_code_with_the_true_value_typed_wrong_inside_it_stops(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    verdict = _judge(tmp_path, ("EA_TRUE_J_MOL = 50000.0", "EA_TRUE_J_MOL = 52000.0"))
    assert verdict.action == trial_checks.STOP
    bias = next(q for q in verdict.calibration.faults if q.kind == "bias")
    assert min(s.value for s in bias.settings) > 1.0
    assert "Ea_true_J_mol = 50000" in bias.truth and "grossly wrong" in verdict.say


def test_the_same_real_code_as_delivered_passes(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    verdict = _judge(tmp_path, None)
    assert verdict.action == trial_checks.OK and verdict.calibration.checked
    bias = next(q for q in verdict.calibration.quantities if q.kind == "bias")
    assert max(abs(s.value) for s in bias.settings) < 0.5
