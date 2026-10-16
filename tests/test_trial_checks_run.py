"""FI's checks of the trials inside the run (core/trial_checks.py through core/trial_runner.TrialsRunner): a fault is sent
back to the simulation's repair with the numbers while a repair is left; once the repairs are spent, settings where most
trials failed are left out of what the analysis reads, and a grossly wrong interval stops the quest.

The live part runs the code a real quest delivered (r12: a noise level typed into the fitting function, and a solver
tolerance that fails most noisy trials; tests/fixtures/replay/r12_code/, copied from study 2's ``orig/``) through FI's own
trial harness at two of its settings, 40 trials each. Its run_trial is made to return each fit's interval ends as well
(``weighted_Ea_lower`` / ``_upper``), the names the plan's ``known_value`` declares: FI measures the coverage of the true
value itself and never reads the code's own 0/1 coverage flag."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import paper_facts, quality_report, trial_checks, trial_runner
from core.execution import SharedInterpreterExecutor

R12 = Path(__file__).parent / "fixtures" / "replay" / "r12_code"

THRESHOLDS = {"Ea_true_J_mol": 50000.0, "ln_A_true": 9.451858239, "nominal_coverage": 0.95}

DECLARED = {"id": "weighted_Ea", "estimand": "the activation energy", "kind": "mean", "unit": "one trial", "paired": True,
            "known_value": {"true_value": "Ea_true_J_mol", "estimate": "weighted_Ea_hat",
                            "interval": ["weighted_Ea_lower", "weighted_Ea_upper"], "level": 0.95}}

ANALYSIS = """
import json, os
data = json.load(open(os.environ["FI_TRIALS"]))
print("RESULT_JSON: " + json.dumps({"settings": [c["key"] for c in data["cells"]]}))
"""


def _quest(tmp_path: Path, *, relaxed: bool = False) -> Path:
    root = tmp_path / "quest"
    shutil.copytree(R12, root / "code")
    if relaxed:  # what a repair of the crashes alone leaves: the typed-in noise level stays
        model = root / "code" / "kinetics_parameter_recovery" / "model.py"
        text = model.read_text(encoding="utf-8")
        assert "abs(float(gradient[0])) >= solver_gtol" in text
        model.write_text(text.replace("abs(float(gradient[0])) >= solver_gtol", "abs(float(gradient[0])) >= 1e-6"),
                         encoding="utf-8")
    sim = root / "code" / "simulate.py"
    text = sim.read_text(encoding="utf-8")
    assert text.count('f"{label}_Ea_hat": ea_hat,') == 1
    sim.write_text(text.replace('f"{label}_Ea_hat": ea_hat,', 'f"{label}_Ea_hat": ea_hat, f"{label}_Ea_lower": ea_low, '
                                'f"{label}_Ea_upper": ea_high,'), encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    return root


def _run(root: Path, noise: list[float], *, repairs_left: bool, trials: int = 40):
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), quest_root=root,
        protocol={"grid": {"noise_std_rel": noise, "n_timepoints": [12]}, "runs_per_setting": trials,
                  "thresholds": THRESHOLDS, "metrics": [DECLARED]},
        deterministic=False, simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
        repairs_left=lambda: repairs_left,
    )
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300,
                                        env={"FI_REPLICATE_SEED": "0"}))
    return runner, result


def test_replay_r12_most_noisy_trials_fail_sent_back_then_left_out(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    root = _quest(tmp_path)
    runner, result = _run(root, [0.005, 0.08], repairs_left=True)
    assert result.returncode == 1 and runner.failed_script == "simulate.py"
    assert runner.trial_fault and runner.trial_fault["action"] == trial_checks.REPAIR and runner.trial_fault["partial"]
    # the repair is told which setting, how many failed, why, and where in the code
    assert "noise_std_rel=0.08,n_timepoints=12" in result.stderr and "more than half of the trials failed" in result.stderr
    assert "did not satisfy required optimality" in result.stderr and "model.py line" in result.stderr
    assert "noise_std_rel=0.005,n_timepoints=12:" not in result.stderr.split("more than half")[-1]
    record = trial_checks.read_record(root)
    assert record["action"] == trial_checks.REPAIR

    # the repairs are spent: the setting is left out of what the analysis reads, FI's own records keep it
    runner, result = _run(root, [0.005, 0.08], repairs_left=False)
    assert result.returncode == 0, result.stderr[-800:]
    assert json.loads(result.stdout.split("RESULT_JSON: ")[-1]) == {"settings": ["noise_std_rel=0.005,n_timepoints=12"]}
    assert runner.left_out_settings == ["noise_std_rel=0.08,n_timepoints=12"]
    kept = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    assert len(kept["cells"]) == 2
    # the paper must name the setting left out
    items = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.SETTING_LEFT_OUT]
    assert [i.name for i in items] == ["noise_std_rel=0.08,n_timepoints=12"]
    assert items[0].id.startswith("setting-left-out-") and "is left out of the results" in items[0].sentence
    # the paper must place it in its Limitations; placed there, nothing is missing
    hits = paper_facts.placement_hits("# T\n\n## Limitations\n\nNone.\n", root)
    assert any("{{not_done:" + items[0].id + "}}" in h for h in hits)
    placed = paper_facts.placement_hits(
        paper_facts.render("# T\n\n## Limitations\n\n{{not_done:" + items[0].id + "}}\n", root).text, root)
    assert not any(items[0].id in h for h in placed), placed
    report = quality_report.build(root, running=False)
    assert report["trial_checks"]["left_out"] == ["noise_std_rel=0.08,n_timepoints=12"]


def test_replay_r12_with_the_crashes_repaired_a_grossly_wrong_interval_is_sent_back_then_stops(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    root = _quest(tmp_path, relaxed=True)
    runner, result = _run(root, [0.04, 0.08], repairs_left=True)
    assert result.returncode == 1 and runner.trial_fault["action"] == trial_checks.REPAIR
    assert not runner.trial_fault["partial"]
    assert "weighted_Ea_lower .. weighted_Ea_upper" in result.stderr and "grossly wrong" in result.stderr
    runner, result = _run(root, [0.04, 0.08], repairs_left=False)
    assert result.returncode == 1 and runner.trial_fault["action"] == trial_checks.STOP
    assert "grossly wrong" in runner.trial_fault["say"]


# ---- the engine's side: the stop, or a run once more with the settings left out -----------------------------------------


class _Engine:
    from core.run_checks import RunChecksMixin as _M

    _trial_checks_stop = _M._trial_checks_stop
    _stop_before_run = _M._stop_before_run

    def __init__(self) -> None:
        import logging

        self._log = logging.getLogger("test")
        self.config = SimpleNamespace(engine=SimpleNamespace(exec_reflect_max_iterations=3))


def test_a_standing_fault_stops_with_its_record_and_partial_failures_run_once_more():
    eng = _Engine()
    fault = {"action": "stop", "say": "FI measured ... grossly wrong", "tried": ["ran every trial"], "partial": False}
    out = eng._trial_checks_stop({"exec_result": {"returncode": 1, "trial_checks": fault}, "exec_reflect_iter": 3})
    assert out["exec_result"]["cannot_repair"] == fault["say"]
    assert out["exec_result"]["cannot_repair_tried"][-1].startswith("sent the simulation back to be repaired")
    partial = {**fault, "action": "repair", "partial": True}
    out = eng._trial_checks_stop({"exec_result": {"returncode": 1, "trial_checks": partial}, "exec_reflect_iter": 1})
    assert out == {"exec_patch_pending": True, "exec_give_up_reason": "", "exec_reflect_iter": 3}
    assert eng._trial_checks_stop({"exec_result": {"returncode": 1}}) is None
    assert eng._trial_checks_stop({"exec_result": {"returncode": 0, "trial_checks": fault}}) is None


def test_judge_follows_the_repairs():
    cells = [{"key": "a=1", "planned": 100, "failed": 80, "metrics": {}},
             {"key": "a=2", "planned": 100, "failed": 0, "metrics": {}}]
    assert trial_checks.judge(cells, repairs_left=True).action == trial_checks.REPAIR
    v = trial_checks.judge(cells, repairs_left=False)
    assert v.action == trial_checks.LEAVE_OUT and v.left_out == ["a=1"]
    every = [{**c, "failed": 80} for c in cells]
    v = trial_checks.judge(every, repairs_left=False)
    assert v.action == trial_checks.STOP and "every setting" in v.say
    few = [{**c, "failed": 10} for c in cells]
    v = trial_checks.judge(few, repairs_left=False)
    assert v.action == trial_checks.OK and all(f.stated for f in v.fails)


def test_more_than_half_of_the_settings_left_out_stops_with_a_plain_reason():
    # what would be left is too little to report: the quest stops with no paper, once the repairs are spent
    cells = [{"key": f"a={n}", "planned": 100, "failed": 80 if n < 3 else 0, "metrics": {}} for n in range(1, 4)]
    v = trial_checks.judge(cells, repairs_left=True)
    assert v.action == trial_checks.REPAIR and not v.partial
    v = trial_checks.judge(cells, repairs_left=False)
    assert v.action == trial_checks.STOP and not v.left_out
    assert "2 of 3 settings" in v.say and "more than half of the settings" in v.say


def test_replay_r12_thirteen_of_twenty_five_settings_left_out_is_a_stop_not_a_paper():
    import gzip

    with gzip.open(Path(__file__).parent / "fixtures" / "replay" / "trial_checks" / "r12_kinetics.json.gz", "rt",
                   encoding="utf-8") as fh:
        data = json.load(fh)
    v = trial_checks.judge(data["cells"], data.get("protocol"), data.get("thresholds"), repairs_left=False)
    assert v.action == trial_checks.STOP and not v.partial and not v.left_out
    assert "13 of 25 settings" in v.say
