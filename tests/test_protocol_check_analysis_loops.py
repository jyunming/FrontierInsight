"""A ``for`` loop over literals in the ANALYSIS script is not the grid (core/protocol_check.py).

The live round s5 (false discovery) stopped without a paper because its acceptance check

    for pi1 in (0.05, 0.2):
        ok4 &= cells[key_of(2000, pi1)]["power_gap"]["mean"] > cells[key_of(20, pi1)]["power_gap"]["mean"]

("the gap grows with m", undefined at pi1 = 0) was read as the analysis sweeping pi1 over [0.05, 0.2]. FI runs every
setting itself under the trial contract; whether the analysis reports every setting is counted from FI's own record
after the run (``protocol_check.analysis_suspects`` and ``run_manifest.confirmed``). The scripts are the real s5 ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import protocol_check as pc, run_manifest as rm

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "protocol_runs"
PROTOCOL: dict[str, Any] = {
    "grid": {"m": [20, 200, 2000], "pi1": [0.0, 0.05, 0.2]},
    "runs_per_setting": 2000,
    "thresholds": {"alpha": 0.05, "delta": 2.5},
    "seed_policy": "Each setting draws from its own independent SeedSequence child stream.",
}
LOOP = '    for pi1 in (0.05, 0.2):\n        ok4 &= cells[key_of(2000, pi1)]["power_gap"]["mean"] > cells[key_of(20, pi1)]["power_gap"]["mean"]\n'


def _scripts() -> dict[str, str]:
    return {"simulate.py": (FIXTURES / "fdr_s5_simulate.py").read_text(encoding="utf-8"),
            "experiment.py": (FIXTURES / "fdr_s5_experiment.py").read_text(encoding="utf-8")}


def test_the_s5_acceptance_check_loop_is_not_a_difference() -> None:
    scripts = _scripts()
    assert LOOP in scripts["experiment.py"]  # the fixture is the real script, the loop is the one that stopped the round
    assert pc.check(PROTOCOL, scripts) == []


def test_the_loop_is_still_read_when_asked_for_and_is_a_suspect_for_the_run_to_settle() -> None:
    scripts = _scripts()
    read = pc.check(PROTOCOL, scripts, analysis_loops=True)
    assert [(m.kind, m.name, m.left_out()) for m in read] == [("grid", "pi1", [0.0])]
    assert "line 117" in read[0].where
    assert [(m.name, m.left_out()) for m in pc.analysis_suspects(PROTOCOL, scripts)] == [("pi1", [0.0])]


def test_a_loop_in_the_simulation_is_still_a_difference() -> None:
    """run_trial that snaps the cell's pi1 to a literal loop runs another setting than the one FI planned: no count of the
    analysis's output can show it, so it is read where it is written."""
    scripts = _scripts()
    old = '    pi1 = float(cell["pi1"])\n'
    assert old in scripts["simulate.py"]
    scripts["simulate.py"] = scripts["simulate.py"].replace(
        old, '    for pi1 in (0.05, 0.2):\n        if pi1 >= float(cell["pi1"]):\n            break\n')
    found = pc.check(PROTOCOL, scripts)
    assert [(m.kind, m.name) for m in found] == [("grid", "pi1")] and "simulate.py" in found[0].where


def test_a_list_the_analysis_names_after_the_axis_is_still_a_difference() -> None:
    scripts = _scripts()
    scripts["experiment.py"] += "\nPI1_VALUES = [0.05, 0.2]\n"
    found = pc.check(PROTOCOL, scripts)
    assert [(m.kind, m.name) for m in found] == [("grid", "pi1")] and "PI1_VALUES" in found[0].where


def test_a_script_outside_the_trial_contract_keeps_its_loops_read() -> None:
    """With no run_trial/run_cell the script runs its own grid, and a loop over a subset is the grid it runs."""
    scripts = {"experiment.py": _scripts()["experiment.py"]}
    assert [(m.kind, m.name) for m in pc.check(PROTOCOL, scripts)] == [("grid", "pi1")]
    assert pc.analysis_suspects(PROTOCOL, scripts) == []


def _differences(short_settings: list[str]) -> list[dict[str, Any]]:
    return [{"metric": "fdr_bh", "measured": (6000, 0), "listed": (4000, 0), "settings": [],
             "short": [{"setting": s, "listed": (0, 0), "recorded": (2000, 0)} for s in short_settings]}]


def test_the_suspect_is_confirmed_only_by_fis_own_count_of_the_settings_left_out() -> None:
    suspects = pc.analysis_suspects(PROTOCOL, _scripts())
    assert rm.confirmed(suspects, []) == []  # the analysis reports every setting: cleared
    assert rm.confirmed(suspects, _differences(["m=20,pi1=0.05"])) == []  # trials left out elsewhere: not this axis value
    said = rm.confirmed(suspects, _differences(["m=20,pi1=0.0", "m=200,pi1=0.0"]))
    assert len(said) == 1 and "pi1 = [0]" in said[0] and "[0.05, 0.2]" in said[0]

