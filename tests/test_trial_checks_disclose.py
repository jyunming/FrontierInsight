"""A reported standard error or interval FI measured not to match the trials, with the repairs spent (core/trial_checks.py
``DISCLOSE``): the quest goes on, the paper and the quality report say at which settings it was measured, and the settings
where it matched are left as they are. A fault that is not about a reported uncertainty (the multiplier the plan fixes,
failed trials) still stops. Neutral made-up study: a mean of noisy draws with a declared true value."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from core import frozen_protocol, paper_facts, quality_report, trial_checks as tc

from tests.test_trial_checks import _proto, known

N = 200
KEYS = [f"noise={s},n={n}" for s in (0.1, 0.2) for n in (5, 10, 20)]


def _cells(factor: dict[str, float] | None = None, interval: bool = False) -> list[dict[str, Any]]:
    """Trials of a mean of ``n`` draws (true value 1.0): the reported standard error is the usual one, times ``factor[key]``
    where a fault is planted."""
    factor = factor or {}
    rng = random.Random(7)
    out = []
    for key in KEYS:
        sigma = float(key.split(",")[0].split("=")[1])
        n = int(key.split("n=")[1])
        est, se, lo, hi = [], [], [], []
        for _ in range(N):
            x = [rng.gauss(1.0, sigma) for _ in range(n)]
            m = sum(x) / n
            s = (sum((v - m) ** 2 for v in x) / (n - 1)) ** 0.5 / n ** 0.5 * factor.get(key, 1.0)
            est.append(m)
            se.append(s)
            lo.append(m - 1.96 * s)
            hi.append(m + 1.96 * s)
        metrics = {"x_hat": est, "x_se": se, **({"x_lo": lo, "x_hi": hi} if interval else {})}
        out.append({"key": key, "cell": {}, "planned": N, "failed": 0,
                    "metrics": {k: {"values": v, "trials": list(range(N))} for k, v in metrics.items()}})
    return out


def _plan() -> dict[str, Any]:
    return _proto(known("x_hat", 1.0, se="x_se"))


def _judge(cells: list[dict[str, Any]], *, left: bool) -> tc.Verdict:
    return tc.judge(cells, _plan(), None, None, repairs_left=left)


def _quest(tmp_path: Path, verdict: tc.Verdict) -> Path:
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    frozen_protocol.freeze(root, {"grid": {"n": [5, 10]}, "runs_per_setting": N}, approved_by="test", source="test")
    tc.write_record(root, verdict.record())
    return root


def _disclosed(root: Path) -> list[paper_facts.Item]:
    return [i for i in paper_facts.required_items(root) if i.kind == paper_facts.CALIBRATION
            and "measured not to match" in i.name]


def test_honest_one_setting_off_is_kept_and_scoped_once_the_repairs_are_spent(tmp_path: Path) -> None:
    cells = _cells({KEYS[0]: 3.0})  # one setting's reported error is too large; the others are right
    first = _judge(cells, left=True)
    assert first.action == tc.REPAIR  # FI still sends the code back first
    v = _judge(cells, left=False)
    assert v.action == tc.DISCLOSE, v.say
    [q] = v.calibration.faults
    assert q.kind == "se_ratio" and [s.key for s in q.settings if s.gross] == [KEYS[0]]
    root = _quest(tmp_path, v)
    [item] = _disclosed(root)
    assert KEYS[0] in item.sentence and "1 of 6 settings" in item.sentence and "from 200 trials" in item.sentence
    assert "In the other 5 settings the two were measured to agree" in item.sentence
    assert KEYS[1] not in item.sentence
    assert "not presented as" in item.sentence
    lines = quality_report._trial_checks_summary(quality_report._Quest(root))["lines"]
    assert any("measured not to match at: " + KEYS[0] in ln and "the other 5 of 6 settings matched" in ln for ln in lines)


def test_planted_error_times_three_everywhere_is_sent_back_then_still_stops() -> None:
    cells = _cells({k: 3.0 for k in KEYS})
    assert _judge(cells, left=True).action == tc.REPAIR  # caught and sent back for repair
    v = _judge(cells, left=False)
    [q] = v.calibration.faults
    assert q.gross == len(KEYS) and q.pooled is not None and q.pooled.gross  # wrong over all settings together
    assert v.action == tc.STOP  # a fault everywhere is never kept: nothing it can vouch for


def test_a_standard_error_on_the_wrong_scale_everywhere_stops() -> None:
    v = _judge(_cells({k: 16.0 for k in KEYS}), left=False)
    assert v.action == tc.STOP and v.calibration.faults[0].pooled.gross


def test_planted_error_times_three_at_half_the_settings_names_exactly_those(tmp_path: Path) -> None:
    wrong = KEYS[::2]
    cells = _cells({k: 3.0 for k in wrong})
    assert _judge(cells, left=True).action == tc.REPAIR
    v = _judge(cells, left=False)
    assert v.action == tc.DISCLOSE
    [q] = v.calibration.faults
    assert sorted(s.key for s in q.settings if s.gross) == sorted(wrong)
    [item] = _disclosed(_quest(tmp_path, v))
    assert "3 of 6 settings" in item.sentence and all(k in item.sentence for k in wrong)
    assert not any(k in item.sentence for k in KEYS if k not in wrong)


def test_an_interval_measured_to_miss_is_disclosed_as_not_valid_at_its_level(tmp_path: Path) -> None:
    cells = _cells({KEYS[0]: 0.3}, interval=True)  # intervals far too narrow at one setting
    plan = _proto(known("x_hat", 1.0, interval=("x_lo", "x_hi")))
    assert tc.judge(cells, plan, None, None, repairs_left=True).action == tc.REPAIR
    v = tc.judge(cells, plan, None, None, repairs_left=False)
    assert v.action == tc.DISCLOSE
    [item] = _disclosed(_quest(tmp_path, v))
    assert "95% intervals" in item.sentence and "not presented as valid 95% intervals" in item.sentence
    assert KEYS[0] in item.sentence and KEYS[1] not in item.sentence


def test_failed_trials_with_a_standing_calibration_fault_still_stop() -> None:
    cells = _cells({KEYS[0]: 3.0})
    cells[1]["failed"] = N  # most trials of another setting failed: not a reported-uncertainty result alone
    cells[1]["planned"] = N
    v = _judge(cells, left=False)
    assert v.action == tc.STOP


def test_a_multiplier_the_plan_does_not_fix_still_stops() -> None:
    cells = _cells(interval=True)
    plan = _proto(known("x_hat", 1.0, se="x_se", interval=("x_lo", "x_hi"), df=3))  # the code used 1.96, the plan fixes t(3)
    v = tc.judge(cells, plan, None, None, repairs_left=False)
    assert v.action == tc.STOP and any(q.kind == "multiplier" for q in v.calibration.faults)


def test_the_runner_goes_on_when_a_disclosed_fault_stands(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from core.trial_runner import TrialsRunner

    root = tmp_path / "q"
    (root / "raw").mkdir(parents=True)
    (root / "code").mkdir()
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": _cells({KEYS[0]: 3.0}), "thresholds": {}}),
                                              encoding="utf-8")
    plan = _plan()
    runner = TrialsRunner(object(), quest_root=root, protocol=lambda: plan, deterministic=False,
                          simulate=root / "code" / "simulate.py", analysis=root / "code" / "analysis.py",
                          repairs_left=lambda: False)
    run = SimpleNamespace(summary_path=root / "raw" / "trials.json", cells=[], stderr=lambda: "")
    assert runner._check_trials(run, plan, 0.0) is None and not getattr(runner, "trial_fault", None)
    assert tc.read_record(root)["action"] == tc.DISCLOSE
    runner = TrialsRunner(object(), quest_root=root, protocol=lambda: plan, deterministic=False,
                          simulate=root / "code" / "simulate.py", analysis=root / "code" / "analysis.py",
                          repairs_left=lambda: True)
    assert runner._check_trials(run, plan, 0.0).returncode == 1 and runner.trial_fault["action"] == tc.REPAIR
