"""The run time and the topic's time limit are judged against what a reader's run of the delivered code takes: its
settings run one after another. When FI ran them at the same time (it recorded how many), the paper says both times, the
one-after-another time is a lower bound, and a limit is never "within" on a lower bound
(``core/run_facts.py``: ``reader_run``, read by ``core/paper_facts.py`` and ``core/disclosure.py``)."""

from __future__ import annotations

import json
from pathlib import Path

from core import disclosure, paper_facts, run_facts

TOPIC = "Estimate a rate. The whole study must run in under 60 seconds."


def _quest(root: Path, *, wall: float, total: float, at_once: int | None) -> Path:
    run_facts.record_trials(root, key="k", wall_s=wall, trials=100, cells=25, trial_sum_s=total, at_once=at_once)
    return root


def test_parallel_wall_under_the_limit_but_serial_over_it_is_not_within(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=35.4, total=112.0, at_once=16)
    known = paper_facts.facts(root, topic=TOPIC)
    text = known["run_time"].text
    assert "35.4" in text and "16 settings run at the same time" in text and "at least 112" in text
    assert known["time_limit_met"].text == "over"
    sentence = disclosure._time_sentence(root, TOPIC, 35.4)
    assert "within" not in sentence and "over" in sentence and "112" in sentence and "35.4" in sentence


def test_a_lower_bound_under_the_limit_is_never_within(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=12.0, total=40.0, at_once=4)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["time_limit_met"].text == "not shown to be within" and "at least 40" in known["run_time"].text
    assert "not shown to be within" in disclosure._time_sentence(root, TOPIC, 12.0)


def test_overlap_is_what_was_recorded_not_a_ratio(tmp_path: Path) -> None:
    # two settings at once, wall 100 s, trials add to 105 s: a ratio would call it one at a time and say "within" 102 s
    topic = "The whole study must run in under 102 seconds."
    root = _quest(tmp_path / "q", wall=100.0, total=105.0, at_once=2)
    assert paper_facts.facts(root, topic=topic)["time_limit_met"].text == "over"
    assert "within" not in disclosure._time_sentence(root, topic, 100.0).replace("not shown to be within", "")


def test_a_record_without_at_once_never_claims_overlap(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=35.0, total=112.0, at_once=None)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["run_time"].text == "35 s" and known["time_limit_met"].text == "within"
    assert "one after another" not in disclosure._time_sentence(root, TOPIC, 35.0)


def test_a_run_of_one_setting_at_a_time_is_unchanged(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=45.0, total=44.0, at_once=1)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["run_time"].text == "45 s" and known["time_limit_met"].text == "within"
    over = _quest(tmp_path / "r", wall=75.0, total=74.0, at_once=None)
    assert paper_facts.facts(over, topic=TOPIC)["time_limit_met"].text == "over"


def test_paper_and_run_record_read_the_same_time_with_the_search(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=20.0, total=50.0, at_once=8)
    (root / "results").mkdir()
    (root / "results" / "best_design.json").write_text(json.dumps({"time": {"seconds": 30.0}}), encoding="utf-8")
    known = paper_facts.facts(root, topic=TOPIC)
    assert "at least 80 s" in known["run_time"].text and known["time_limit_met"].text == "over"
    assert "at least 80 s" in disclosure._time_sentence(root, TOPIC, 20.0)


def test_at_once_comes_from_fis_other_records_of_the_same_run(tmp_path: Path) -> None:
    # the run record has no at_once (written before it was kept): the estimate, then the line FI logged, say it
    root = _quest(tmp_path / "q", wall=35.4, total=82.0, at_once=None)
    assert paper_facts.facts(root, topic=TOPIC)["time_limit_met"].text == "within"
    (root / ".fi").mkdir(exist_ok=True)
    (root / ".fi" / "run.log").write_text(
        "2026-10-25 20:49:06,419 [INFO] [execute] 16 settings of the study run at the same time (execution.settings_at_once)\n",
        encoding="utf-8")
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["time_limit_met"].text == "over" and "16 settings run at the same time" in known["run_time"].text
    other = _quest(tmp_path / "r", wall=12.0, total=40.0, at_once=None)
    (other / ".fi").mkdir(exist_ok=True)
    (other / ".fi" / "run_estimate.json").write_text(json.dumps({"at_once": 4}), encoding="utf-8")
    assert "4 settings run at the same time" in paper_facts.facts(other, topic=TOPIC)["run_time"].text


def test_every_settings_own_wall_time_makes_the_time_firm(tmp_path: Path) -> None:
    root = tmp_path / "q"
    run_facts.record_trials(root, key="k", wall_s=12.0, trials=10, cells=5, trial_sum_s=40.0, at_once=4, settings_s=50.0)
    known = paper_facts.facts(root, topic=TOPIC)
    assert known["time_limit_met"].text == "within" and "take 50 s" in known["run_time"].text
    assert "at least" not in known["run_time"].text
    sentence = disclosure._time_sentence(root, TOPIC, 12.0)
    assert "at least" not in sentence and "is within" in sentence
    run_facts.record_trials(root, key="k", wall_s=12.0, trials=10, cells=5, trial_sum_s=40.0, at_once=4, settings_s=75.0)
    assert paper_facts.facts(root, topic=TOPIC)["time_limit_met"].text == "over"


def test_the_runner_keeps_each_settings_wall_time(tmp_path: Path) -> None:
    from core import trial_runner

    cells = [trial_runner.CellRun(key=str(i), cell={}, planned=1, rows=[{"duration_s": 1.0, "status": "ok"}], wall_s=w)
             for i, w in enumerate((2.0, 3.5))]
    run = trial_runner.TrialRun(cells=cells, ledger_path=tmp_path / "l", summary_path=tmp_path / "s")
    trial_runner._keep_run_time(tmp_path, "k", run, 3.0, 2)
    kept = run_facts.read(tmp_path)["trials_run"]
    assert kept["settings_s"] == 5.5 and kept["at_once"] == 2
    cells[1].wall_s = None  # a setting that was not run: no firm time
    trial_runner._keep_run_time(tmp_path, "k", run, 3.0, 2)
    assert "settings_s" not in run_facts.read(tmp_path)["trials_run"]


def test_the_writer_is_told_the_third_wording() -> None:
    assert "not shown to be within" in paper_facts._FACT_WORDS["time_limit_met"]
