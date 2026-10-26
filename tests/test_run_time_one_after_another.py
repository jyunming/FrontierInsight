"""The run time and the topic's time limit are judged against what a reader's run of the delivered code takes: its
settings run one after another. FI's wall time decides neither "within" nor "over" when the settings ran at the same time
(or FI does not know that they did not): "over" only when the least a one-after-another run takes is over the limit,
"within" only when the most it takes is at or under it (``core/run_facts.py``: ``reader_run``, ``reader_verdict``, read by
``core/paper_facts.py`` and ``core/disclosure.py``)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from core import disclosure, paper_facts, run_facts, trial_runner
from core.execution import SharedInterpreterExecutor

TOPIC = "Estimate a rate. The whole study must run in under 60 seconds."
NOT_SHOWN = "not shown to be within or over"


def _quest(root: Path, *, wall: float, total: float, at_once: int | None, key: str = "k",
           most: float | None = None) -> Path:
    run_facts.record_trials(root, key=key, wall_s=wall, trials=100, cells=25, trial_sum_s=total, at_once=at_once,
                            settings_s=most)
    return root


def _met(root: Path, wall: float, topic: str = TOPIC) -> tuple[str, str, str]:
    known = paper_facts.facts(root, topic=topic)
    return known["time_limit_met"].text, known["run_time"].text, disclosure._time_sentence(root, topic, wall)


def test_parallel_wall_under_the_limit_but_the_least_over_it_is_over(tmp_path: Path) -> None:
    met, said, sentence = _met(_quest(tmp_path / "q", wall=35.4, total=112.0, at_once=16), 35.4)
    assert met == "over" and "35.4" in said and "16 settings run at the same time" in said and "at least 112" in said
    assert "within" not in sentence and "is over" in sentence and "112" in sentence


def test_a_lower_bound_under_the_limit_is_never_within(tmp_path: Path) -> None:
    met, said, sentence = _met(_quest(tmp_path / "q", wall=12.0, total=40.0, at_once=4), 12.0)
    assert met == NOT_SHOWN and "at least 40" in said and NOT_SHOWN in sentence


def test_the_wall_time_alone_never_makes_it_over(tmp_path: Path) -> None:
    # four at once, wall 75 s, the trials add to 40 s, limit 60 s: the wall time is not what a reader's run takes
    met, _said, sentence = _met(_quest(tmp_path / "q", wall=75.0, total=40.0, at_once=4), 75.0)
    assert met == NOT_SHOWN and "is over" not in sentence


def test_overlap_is_what_was_counted_not_a_ratio(tmp_path: Path) -> None:
    # two settings at once, wall 100 s, trials add to 105 s, limit 102 s: the least is over
    topic = "The whole study must run in under 102 seconds."
    met, _said, sentence = _met(_quest(tmp_path / "q", wall=100.0, total=105.0, at_once=2), 100.0, topic)
    assert met == "over" and "is over the topic" in sentence


def test_a_record_without_a_count_never_defaults_to_one_setting_at_a_time(tmp_path: Path) -> None:
    # an older parallel run: no count, wall 35.4 s, trials add to 112 s. They cannot add to more than the wall time
    # one at a time, so the settings overlapped, and the least is over the limit.
    met, said, sentence = _met(_quest(tmp_path / "q", wall=35.4, total=112.0, at_once=None), 35.4)
    assert met == "over" and "uncounted number of settings run at the same time" in said and "within" not in sentence
    # no count and no proof of overlap: not known whether they ran together, so neither within nor over
    met, said, sentence = _met(_quest(tmp_path / "r", wall=35.0, total=30.0, at_once=None), 35.0)
    assert met == NOT_SHOWN and "did not record whether the settings ran at the same time" in said
    assert NOT_SHOWN in sentence


def test_a_run_counted_as_one_setting_at_a_time_is_judged_by_its_wall_time(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=45.0, total=44.0, at_once=1)
    assert _met(root, 45.0)[:2] == ("within", "45 s")
    assert _met(_quest(tmp_path / "r", wall=75.0, total=74.0, at_once=1), 75.0)[0] == "over"


def test_paper_and_run_record_read_the_same_time_with_the_search(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=20.0, total=50.0, at_once=8)
    (root / "results").mkdir()
    (root / "results" / "best_design.json").write_text(json.dumps({"time": {"seconds": 30.0}}), encoding="utf-8")
    met, said, sentence = _met(root, 20.0)
    assert "at least 80 s" in said and met == "over" and "at least 80 s" in sentence


def test_the_estimates_count_is_never_used(tmp_path: Path) -> None:
    # the estimate's number is how many FI let run, not how many did; a malformed estimate is not read at all
    root = _quest(tmp_path / "q", wall=12.0, total=40.0, at_once=None, key="confirm")
    (root / ".fi").mkdir(exist_ok=True)
    for text in (json.dumps({"key": "confirm", "at_once": 8}), "[1, 2]", "not json"):
        (root / ".fi" / "run_estimate.json").write_text(text, encoding="utf-8")
        said = paper_facts.facts(root, topic=TOPIC)["run_time"].text
        assert "8 settings" not in said and "uncounted number" in said
    assert run_facts.settings_at_once(root) is None


def test_the_log_line_is_not_a_count_of_this_run(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q", wall=35.0, total=30.0, at_once=None)
    (root / ".fi").mkdir(exist_ok=True)
    (root / ".fi" / "run.log").write_text("[execute] 16 settings of the study run at the same time\n", encoding="utf-8")
    assert run_facts.settings_at_once(root) is None


def _both(root: Path, *, least: float, most: float | None, topic: str = TOPIC) -> tuple[str, str, str]:
    _quest(root, wall=12.0, total=least, at_once=4, most=most)
    return _met(root, 12.0, topic)


def test_within_only_when_the_most_it_can_take_is_at_or_under_the_limit(tmp_path: Path) -> None:
    met, said, sentence = _both(tmp_path / "q", least=40.0, most=50.0)
    assert met == "within" and "between 40 s and 50 s" in said and "measured while the settings ran at the same time" in said
    assert "is within the topic" in sentence and "between 40 s and 50 s" in sentence
    assert _both(tmp_path / "e", least=40.0, most=60.0)[0] == "within"


def test_over_when_the_least_it_can_take_is_over_the_limit(tmp_path: Path) -> None:
    met, _said, sentence = _both(tmp_path / "q", least=70.0, most=150.0)
    assert met == "over" and "is over the topic" in sentence


def test_between_the_two_is_neither_within_nor_over(tmp_path: Path) -> None:
    met, said, sentence = _both(tmp_path / "q", least=40.0, most=75.0)
    assert met == NOT_SHOWN and "between 40 s and 75 s" in said
    assert f"{NOT_SHOWN} the topic's limit of 60 s" in sentence


def test_the_runner_counts_the_settings_that_ran_together(tmp_path: Path) -> None:
    source = "import time\n\ndef run_trial(cell, trial_id, seed):\n    time.sleep(0.6)\n    return {'y': cell['k'] * 2.0}\n"

    def run(name: str, at_once: int, grid: dict):
        root = tmp_path / name
        (root / "code").mkdir(parents=True)
        (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
        return asyncio.run(trial_runner.run_trials(
            SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"),
            sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=0, deterministic=False,
            timeout_s=120, run_id="r1", at_once=at_once, env={**os.environ}))

    assert run("one", 1, {"k": [1, 2, 3]}).most_at_once == 1
    assert run("two", 2, {"k": [1, 2, 3, 4]}).most_at_once == 2
    # room for eight, but the plan has one setting: one ran, whatever the capacity was
    assert run("single", 8, {"k": [1]}).most_at_once == 1


def test_the_runner_keeps_the_counted_number_and_each_settings_wall_time(tmp_path: Path) -> None:
    cells = [trial_runner.CellRun(key=str(i), cell={}, planned=1, rows=[{"duration_s": 1.0, "status": "ok"}], wall_s=w)
             for i, w in enumerate((2.0, 3.5))]
    run = trial_runner.TrialRun(cells=cells, ledger_path=tmp_path / "l", summary_path=tmp_path / "s", most_at_once=2)
    trial_runner._keep_run_time(tmp_path, "k", run, 3.0)
    kept = run_facts.read(tmp_path)["trials_run"]
    assert kept["settings_s"] == 5.5 and kept["at_once"] == 2
    cells[1].wall_s = None  # a setting that was not run: no sum
    run.most_at_once = 0  # not counted: no count is kept
    trial_runner._keep_run_time(tmp_path, "k", run, 3.0)
    kept = run_facts.read(tmp_path)["trials_run"]
    assert "settings_s" not in kept and "at_once" not in kept


def test_the_writer_is_told_when_the_third_wording_applies() -> None:
    words = paper_facts._FACT_WORDS["time_limit_met"]
    assert NOT_SHOWN in words and "at or under the limit or not known while the most it takes is over it or not known" in words


def test_the_least_decides_over_whatever_the_count(tmp_path: Path) -> None:
    # no count, no proof of overlap (the trials add to less than the wall time): the trials' own times, 121 s, are still
    # the least a one-after-another run takes, and the limit is over
    met, said, sentence = _met(_quest(tmp_path / "q", wall=130.0, total=121.0, at_once=None), 130.0)
    assert met == "over" and "at least 121 s" in said and "is over the topic" in sentence


def test_a_run_counted_as_serial_still_counts_the_search(tmp_path: Path) -> None:
    # wall 45 s, trials 44 s, search 20 s, limit 60 s: a reader's run takes at least 64 s
    root = _quest(tmp_path / "q", wall=45.0, total=44.0, at_once=1)
    (root / "results").mkdir()
    (root / "results" / "best_design.json").write_text(json.dumps({"time": {"seconds": 20.0}}), encoding="utf-8")
    met, said, sentence = _met(root, 45.0)
    assert met == "over" and "one at a time" in said and "is over the topic" in sentence


def test_without_a_wall_time_the_bounds_still_give_the_same_verdict(tmp_path: Path) -> None:
    root = tmp_path / "q"
    run_facts.record_trials(root, key="k", wall_s=None, trials=10, cells=5, trial_sum_s=70.0, at_once=4)
    known = paper_facts.facts(root, topic=TOPIC)
    sentence = disclosure._time_sentence(root, TOPIC, None)
    assert known["time_limit_met"].text == "over" and "is over the topic" in sentence
    ok = tmp_path / "r"
    run_facts.record_trials(ok, key="k", wall_s=None, trials=10, cells=5, trial_sum_s=30.0, at_once=4, settings_s=50.0)
    assert paper_facts.facts(ok, topic=TOPIC)["time_limit_met"].text == "within"
    assert "is within the topic" in disclosure._time_sentence(ok, TOPIC, None)


def test_the_range_says_a_one_after_another_run_takes_at_most_the_upper_time(tmp_path: Path) -> None:
    _met_, said, _s = _both(tmp_path / "q", least=40.0, most=50.0)
    assert "takes at most that" in said
