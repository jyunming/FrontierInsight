"""What FI's checks of the trials could not show becomes a required item of the paper's Limitations
(core/paper_facts.py:required_items, placed as ``{{not_done:<id>}}``): trials the time limit cut off are not called failed,
reported intervals and standard errors not measured against the true values (and why), one measured below its level, and
trials FI did not run itself. Only for a plan with more than one trial a setting. Real files, no model."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core import frozen_protocol, optimise, paper_facts, quality_report, trial_checks


def _root(tmp_path: Path, runs: int | None = 20) -> Path:
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    if runs is not None:
        frozen_protocol.freeze(root, {"grid": {"n": [5, 10]}, "runs_per_setting": runs}, approved_by="test", source="test")
    return root


def _checks(root: Path, record: dict[str, Any]) -> None:
    trial_checks.write_record(root, record)


def _items(root: Path, kind: str) -> list[paper_facts.Item]:
    return [i for i in paper_facts.required_items(root) if i.kind == kind]


def _ledger(root: Path, statuses: dict[str, list[str]]) -> None:
    rows = []
    for cell, sts in statuses.items():
        rows.append({"event": "planned", "cell": cell, "trials": len(sts)})
        rows += [{"event": "trial", "cell": cell, "trial": t, "seed": 100 + t, "status": s} for t, s in enumerate(sts)]
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_trials_the_time_limit_cut_off_are_not_called_failed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    # FI's ledger records a trial the time limit cut off as failed; the check keeps them apart.
    _ledger(root, {"n=5": ["ok", "ok", "ok", "failed"], "n=10": ["ok", "ok", "failed", "failed"]})
    _checks(root, {"out_of_time": [{"key": "n=10", "planned": 4, "trials": 2}]})
    [item] = _items(root, paper_facts.FAILED_TRIALS)
    assert item.sentence == ("1 of 8 trials failed; their results are not in the reported numbers; 2 trials were not run "
                             "because the study's time limit ran out.")
    assert paper_facts._trial_counts(root)[2] == "1"
    # every one of them cut off by the time limit: none is called failed
    _ledger(root, {"n=5": ["ok", "ok", "ok", "ok"], "n=10": ["ok", "ok", "failed", "failed"]})
    [item] = _items(root, paper_facts.FAILED_TRIALS)
    assert item.sentence == "2 trials were not run because the study's time limit ran out."
    assert paper_facts._trial_counts(root)[2] == "0"


def test_intervals_not_measured_against_the_truth_are_an_item_with_the_reason(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _checks(root, {"settings": 2, "calibration": {"quantities": [], "checked": False,
                                                   "not_checked": [trial_checks.NO_DECLARATION]}})
    [item] = _items(root, paper_facts.CALIBRATION)
    assert item.id == "calibration-all"
    assert item.sentence == ("The coverage of the reported intervals, and the accuracy of their standard errors, were not "
                             "checked against known true values.")
    assert "known_value" not in item.sentence and "run_trial" not in item.sentence
    # a search says why in its own words
    _checks(root, {"settings": 2, "calibration": {"quantities": [], "checked": False,
                                                   "not_checked": [optimise.NO_CALIBRATION]}})
    [item] = _items(root, paper_facts.CALIBRATION)
    assert "searched for the best design" in item.sentence
    # too few trials: the record's own reason
    why = "`coverage` (ea): no setting had 10 or more trials with these values (and some spread of the estimate), too few to judge"
    _checks(root, {"settings": 2, "calibration": {"quantities": [], "checked": False, "not_checked": [why]}})
    [item] = _items(root, paper_facts.CALIBRATION)
    assert "too few" not in item.sentence and "`" not in item.sentence  # the reason is in the quality report
    section = paper_facts.writer_section(root, topic="t")
    assert "{{not_done:calibration-all}}" in section


def test_an_interval_below_its_level_is_an_item_and_one_at_its_level_is_not(tmp_path: Path) -> None:
    root = _root(tmp_path)

    def q(below: int, fault: bool = False, kind: str = "coverage") -> dict[str, Any]:
        return {"kind": kind, "label": "Ea" if kind == "coverage" else "Ea_se", "names": ["a", "b"], "level": 0.95,
                "judged": 5, "below": below, "fault": fault, "gross": 1 if fault else 0,
                "settings": [{"key": f"n={i}", "value": v} for i, v in enumerate([0.95, 0.94, 0.88, 0.87, 0.96])]}

    _checks(root, {"settings": 5, "calibration": {"quantities": [q(2)], "checked": True, "not_checked": []}})
    [item] = _items(root, paper_facts.CALIBRATION)
    assert item.sentence == ("The 95% intervals reported for `Ea` contain the true value less often than 95% in 2 of 5 "
                             "settings, by more than chance explains (lowest coverage 0.87).")
    _checks(root, {"settings": 5, "calibration": {"quantities": [q(0)], "checked": True, "not_checked": []}})
    assert _items(root, paper_facts.CALIBRATION) == []
    _checks(root, {"settings": 5, "calibration": {"quantities": [q(1, kind="se_ratio")], "checked": True,
                                                   "not_checked": []}})
    [item] = _items(root, paper_facts.CALIBRATION)
    assert item.sentence.startswith("The standard error reported for `Ea_se` differs from the spread of the estimates")


def test_trials_fi_did_not_run_itself_are_an_item(tmp_path: Path) -> None:
    root = _root(tmp_path)
    trial_checks.record_not_run(root, "one_script")
    [item] = _items(root, paper_facts.TRIALS_UNCHECKED)
    assert item.id == "trials-unchecked-all"
    assert "failed trials were not counted" in item.sentence
    assert _items(root, paper_facts.CALIBRATION) == []
    q = SimpleNamespace(trial_checks=trial_checks.read_record(root))
    summary = quality_report._trial_checks_summary(q)  # type: ignore[arg-type]
    assert summary and summary["not_run"] and "runs its own trials" in summary["lines"][0]


def test_one_trial_a_setting_or_no_plan_gives_no_trial_item(tmp_path: Path) -> None:
    for runs in (1, None):
        root = _root(tmp_path / str(runs), runs=runs)
        trial_checks.record_not_run(root, "own_loop")
        assert _items(root, paper_facts.TRIALS_UNCHECKED) == []
        _checks(root, {"settings": 2, "calibration": {"quantities": [], "checked": False,
                                                       "not_checked": [trial_checks.NO_DECLARATION]}})
        assert _items(root, paper_facts.CALIBRATION) == []
    # no record at all (a quest that predates it): nothing
    assert _items(_root(tmp_path / "no_record"), paper_facts.TRIALS_UNCHECKED) == []


def test_estimates_off_the_truth_beyond_chance_are_an_item(tmp_path: Path) -> None:
    root = _root(tmp_path)
    q = {"kind": "bias", "label": "Ea_hat", "names": ["Ea_hat"], "judged": 4, "below": 2, "fault": False, "gross": 0,
         "settings": [{"key": f"n={i}", "value": v} for i, v in enumerate([0.1, -0.2, 0.62, 0.7])]}
    _checks(root, {"settings": 4, "calibration": {"quantities": [q], "checked": True, "not_checked": []}})
    [item] = _items(root, paper_facts.CALIBRATION)
    assert item.sentence == ("The estimates of `Ea_hat` lie off the true value in 2 of 4 settings, by more than chance "
                             "explains (by -0.20 to +0.70 of their own spread).")


def test_the_failed_trials_item_names_the_settings_where_they_failed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _ledger(root, {"n=5": ["ok", "ok", "ok", "failed"], "n=10": ["ok", "failed", "failed", "failed"],
                   "n=20": ["ok"] * 4})
    _checks(root, {"failures": [{"key": "n=5", "planned": 4, "failed": 1}, {"key": "n=10", "planned": 4, "failed": 3}]})
    [item] = _items(root, paper_facts.FAILED_TRIALS)
    assert item.sentence == ("4 of 12 trials failed; their results are not in the reported numbers (n=10: 3 of 4; "
                             "n=5: 1 of 4).")
