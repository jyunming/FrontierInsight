"""Regression contracts from an external re-audit's reproductions (each began as a probe asserting the defect; every one
now asserts the correct behaviour). Neutral names throughout: nothing here depends on a topic."""
from __future__ import annotations

from types import SimpleNamespace

from core import flat_output


def test_the_same_numbers_in_other_units_get_the_same_flat_decision() -> None:
    def cells(scale: float) -> list[SimpleNamespace]:
        return [SimpleNamespace(cell={"setting": i}, rows=[{"status": "ok", "values": {"size": i * scale}}])
                for i in (1, 2, 3)]
    # 1, 2 and 3 of a small unit written in the base unit, and in the small unit: one decision for both.
    assert flat_output.flat_sweep(cells(1e-12)) is None
    assert flat_output.flat_sweep(cells(1.0)) is None
    assert flat_output.flat_sweep(cells(1e12)) is None


# ---- an output that does not change is an observation, not a broken simulation ------------------------------------------

import asyncio
import json
import sys
from pathlib import Path

import pytest

from core import run_estimate as est
from core import typed_results
from tests.test_flat_output import SWEEP_ANALYSIS, _quest, _sweep_runner

CONSERVING = ('def run_cell(cell):\n'
              '    a = cell["flow"]\n'
              '    b = 10.0 - a\n'
              '    return {"total": a + b}\n')
IGNORING = 'def run_cell(cell):\n    return {"total": 10.0}\n'
DEAD_READ = 'def run_cell(cell):\n    unused = cell["flow"] * 2.0\n    return {"total": 10.0}\n'


def _run(tmp_path: Path, source: str):
    root = _quest(tmp_path, source)
    (root / "code" / "experiment.py").write_text(SWEEP_ANALYSIS, encoding="utf-8")
    runner = _sweep_runner(root)
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60,
                                        env={"FI_REPLICATE_SEED": "0"}))
    return root, runner, result


def test_a_conserved_total_keeps_its_result_and_is_recorded_as_observed(tmp_path: Path) -> None:
    assert not typed_results.typed_results(CONSERVING, "simulate.py")
    root, runner, result = _run(tmp_path, CONSERVING)
    assert result.returncode == 0, result.stderr[-1500:]
    assert runner.failed_script is None and runner.flat is None
    assert runner.no_difference and "cannot show that there is no effect" in runner.no_difference
    assert "does not use" not in (result.stderr or "")


@pytest.mark.parametrize("source", [IGNORING, DEAD_READ])
def test_a_program_that_lets_no_setting_reach_its_result_is_still_sent_back(tmp_path: Path, source: str) -> None:
    root, runner, result = _run(tmp_path, source)
    assert result.returncode == 1 and runner.failed_script == "simulate.py"
    assert "does not use" in result.stderr and runner.flat and runner.no_difference is None


def test_the_plan_declaring_the_quantity_invariant_is_never_a_fault(tmp_path: Path) -> None:
    root = _quest(tmp_path, IGNORING)
    (root / "code" / "experiment.py").write_text(SWEEP_ANALYSIS, encoding="utf-8")
    from core import trial_runner
    from core.execution import SharedInterpreterExecutor
    protocol = {"grid": {"flow": [1.0, 2.0, 3.0]},
                "oracles": [{"name": "mass balance", "kind": "invariant", "measure": "total", "expected": 0.0,
                             "tolerance": 1e-9, "reference": "derivation: what flows in flows out"}]}
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), quest_root=root,
        protocol=protocol, deterministic=True, simulate=root / "code" / "simulate.py",
        analysis=root / "code" / "experiment.py")
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60,
                                        env={"FI_REPLICATE_SEED": "0"}))
    assert result.returncode == 0 and runner.flat is None and runner.no_difference


def test_the_run_record_says_observed_not_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, ["{}"])
    eng._record_no_difference("`total` was the same at all 3 settings tried")
    note = eng._run_checks_note()
    assert "Observed:" in note and "do not call the simulation broken" in note
    eng._record_no_difference(None)
    assert "Observed:" not in eng._run_checks_note()


@pytest.mark.parametrize("source,keys,expected", [
    (CONSERVING, ["flow"], []),
    (IGNORING, ["flow"], ["flow"]),
    (DEAD_READ, ["flow"], ["flow"]),
    ('def run_cell(cell):\n    if cell["mode"] == "a":\n        v = 1.0\n    else:\n        v = 2.0\n    return {"t": v}\n',
     ["mode"], []),
    ('def run_cell(cell):\n    if cell["mode"] == "a":\n        return {"t": 1.0}\n    return {"t": 2.0}\n', ["mode"], []),
    ('def run_cell(cell):\n    return {"t": cell.get("flow", 1.0)}\n', ["flow", "other"], ["other"]),
    ('def run_cell(cell):\n    return {"t": helper(cell)}\n', ["flow"], None),
    ('def run_cell(cell):\n    return {"t": 1.0}\n\ndef run_trial(cell, t, s):\n    return {"t": 1.0}\n', ["flow"], None),
    ("def run_cell(cell):\n    return {\"t\": eval(\"cell['flow']\")}\n", ["flow"], None),
    ("not python (", ["flow"], None),
])
def test_the_code_evidence_names_only_what_it_can_show(source: str, keys: list[str], expected: list[str] | None) -> None:
    from core import flat_output
    assert flat_output.unread_settings(source, keys) == expected


def test_a_quantity_built_from_no_setting_is_named_and_one_that_is_built_from_one_is_not() -> None:
    from core import flat_output
    source = ('def run_trial(cell, trial, seed):\n    out = {}\n    out["fixed"] = 1.0\n    out["moves"] = cell["x"] * 2\n'
              '    return out\n')
    assert flat_output.unused_outputs(source, ["fixed", "moves"]) == ["fixed"]
    assert flat_output.unused_outputs("def run_cell(cell):\n    return other(cell)\n", ["a"]) is None


def test_the_pilot_does_not_send_back_the_conserved_total() -> None:
    cells = [{"flow": f} for f in (1.0, 2.0, 3.0)]
    obs = [(dict(c), {"total": 10.0}) for c in cells for _ in range(2)]
    assert est.pilot_problems(obs, {"total": "mean"}, source=CONSERVING).problems == []
    assert [p.kind for p in est.pilot_problems(obs, {"total": "mean"}, source=IGNORING).problems] == ["flat"]
