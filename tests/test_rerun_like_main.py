"""A rerun of the study with a repair runs the way the delivered run ran (core/rerun_check.py).

The evidence gate reruns the study on a copy of the code to see whether a repair moves a reported number. It used to run the
study's own run.py, one setting after another, while the delivered run had many settings at once; it ran the settings the
delivered run's time limit had cut off, and counted them as changed failure counts; it estimated the time without the
delivered run's own; and it ran a repair that had already crashed again at every check. These tests hold the new rules.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import rerun_check, run_facts, trial_runner
from core.execution import SharedInterpreterExecutor
from core.rerun_check import Verdict


def _record(root: Path, **run: Any) -> None:
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "run_facts.json").write_text(json.dumps({"trials_run": run}), encoding="utf-8")


# ---- how long a rerun may take ------------------------------------------------------------------------------------------
def test_a_rerun_like_the_delivered_run_is_allowed_one_and_a_half_times_its_own_time(tmp_path: Path) -> None:
    """Old code: 1.5 x (3 x the settings' own time added up), as for one setting after another (2160 s here)."""
    _record(tmp_path, wall_s=100.0, at_once=16, settings_s=480.0)
    assert rerun_check.time_allowed(tmp_path, 900, at_once=16) == 150.0
    assert rerun_check.time_allowed(tmp_path, 900, at_once=8) == 300.0  # half as many at once: twice as long
    assert rerun_check.time_allowed(tmp_path, 120, at_once=16) == 120.0  # never above the quest's own limit
    assert rerun_check.time_allowed(tmp_path, 900, at_once=64) == 150.0  # more at once than it had: no faster than it was
    _record(tmp_path, wall_s=10.0, at_once=16)
    assert rerun_check.time_allowed(tmp_path, 900, at_once=16) == rerun_check.FLOOR_S
    # a study run again one setting after another keeps its own, serial limit
    _record(tmp_path, wall_s=100.0, at_once=16, settings_s=480.0)
    assert rerun_check.time_allowed(tmp_path, 28800) == 1.5 * 3 * 480.0


def test_the_whole_check_may_take_twice_one_rerun_or_the_quests_own_limit() -> None:
    assert rerun_check.total_allowed(150.0, 900) == 900.0
    assert rerun_check.total_allowed(600.0, 900) == 1200.0
    assert rerun_check.total_allowed(None, 900) == 900.0


class _Calls:
    """An executor that only counts what it is asked to run."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def execute(self, cmd: list[str], **_k: Any) -> Any:
        self.calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)


def test_a_rerun_the_delivered_run_says_will_not_fit_is_not_started(tmp_path: Path) -> None:
    """The delivered run took longer than the time left: "could not rerun" at 0 s, with the reason (old code ran it for the
    whole time and came back with the same answer)."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "run.py").write_text("print('ran')\n", encoding="utf-8")
    _record(root, wall_s=29_073.0, at_once=16)
    copy = rerun_check.copy_code(root, tmp_path / "c")
    calls = _Calls()
    mode, out, why, secs, crashed = asyncio.run(rerun_check.run_copy(
        calls, sys.executable, root, copy, ["simulate.py"], 28_800, None, tmp_path / "w0"))
    assert calls.calls == [] and (mode, out, secs, crashed) == ("whole study", None, 0.0, False)
    assert "not started" in why and "8 h" in why
    mode, out, why, secs, crashed = asyncio.run(rerun_check.run_copy(
        calls, sys.executable, root, copy, ["simulate.py"], 28_800, None, tmp_path / "w1", at_once=16, protocol={"grid": {}}))
    assert calls.calls == [] and (out, secs) == (None, 0.0) and "not started" in why
    # known positive: a rerun that fits is started
    _record(root, wall_s=10.0, at_once=16)
    mode, out, why, _secs, _crashed = asyncio.run(rerun_check.run_copy(
        calls, sys.executable, root, copy, ["simulate.py"], 600, None, tmp_path / "w2"))
    assert len(calls.calls) == 1


# ---- the settings the delivered run's time limit cut off ----------------------------------------------------------------
def _cell(key: str, ok: int, failed: int, values: list[float]) -> dict[str, Any]:
    metrics = {"y": {"values": values, "trials": list(range(len(values))), "count": len(values), "total": sum(values),
                     "non_finite": 0}} if values else {}
    return {"cell": {"a": key[2:]}, "key": key, "planned": ok + failed, "ok": ok, "failed": failed, "metrics": metrics}


def _delivered(tmp_path: Path, *, reason_c: str) -> tuple[Path, Path]:
    """A delivered run of three settings whose third has no number: its one trial failed for ``reason_c``."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {'y': 1.0}\n", encoding="utf-8")
    (root / "raw").mkdir()
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": [
        _cell("a=1", 1, 0, [1.0]), _cell("a=2", 1, 0, [2.0]), _cell("a=3", 0, 1, [])]}), encoding="utf-8")
    rows = [{"event": "planned", "cell": k, "trials": 1} for k in ("a=1", "a=2", "a=3")]
    rows += [{"event": "trial", "cell": "a=1", "trial": 0, "seed": None, "status": "ok"},
             {"event": "trial", "cell": "a=2", "trial": 0, "seed": None, "status": "ok"},
             {"event": "trial", "cell": "a=3", "trial": 0, "seed": None, "status": "failed", "reason": reason_c}]
    (root / "raw" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "data" / "results").mkdir(parents=True)
    (root / "data" / "results" / "derived.json").write_text(json.dumps(
        {"derived": {"y": {"status": "ok", "by_setting": {"a=1": 1.0, "a=2": 2.0, "a=3": None}}}}), encoding="utf-8")
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "simulate.py").write_text("def run_cell(cell):\n    return {'y': 1.0}  # repaired\n", encoding="utf-8")
    return root, copy


def _judged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: Path, copy: Path, *, a2: float = 2.0) -> Verdict:
    """The verdict when the repaired run ran every setting (the third one too) and gave ``a2`` at the second."""
    out = tmp_path / "out"
    (out / "raw").mkdir(parents=True)
    (out / "raw" / "trials.json").write_text(json.dumps({"cells": [
        _cell("a=1", 1, 0, [1.0]), _cell("a=2", 1, 0, [a2]), _cell("a=3", 1, 0, [3.0])]}), encoding="utf-8")
    (out / "data" / "results").mkdir(parents=True)
    (out / "data" / "results" / "derived.json").write_text(json.dumps(
        {"derived": {"y": {"status": "ok", "by_setting": {"a=1": 1.0, "a=2": a2, "a=3": 3.0}}}}), encoding="utf-8")

    async def fake_run_copy(*a: Any, **k: Any) -> tuple[Any, ...]:
        return "whole study", out, "", 1.0, False

    monkeypatch.setattr(rerun_check, "run_copy", fake_run_copy)
    return asyncio.run(rerun_check.judge(None, "py", root, copy, {"metrics": [{"id": "y"}]}, 60, None))


def test_a_setting_the_delivered_runs_time_limit_cut_off_is_no_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: "changed" (its failed trials 1 -> 0, "failed trials"), so a study cut by its time limit could never come back
    unchanged."""
    root, copy = _delivered(tmp_path, reason_c=trial_runner.OUT_OF_TIME)
    v = _judged(tmp_path, monkeypatch, root, copy)
    assert v.status == "unchanged" and v.keys == []
    assert rerun_check.cut_settings(root) == {"a=3"}
    assert any("time ran out" in n for n in v.notes)


def test_a_cut_setting_does_not_hide_a_move_at_a_setting_that_ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Known positive: a repair that moves a delivered number is still "changed"."""
    root, copy = _delivered(tmp_path, reason_c=trial_runner.OUT_OF_TIME)
    v = _judged(tmp_path, monkeypatch, root, copy, a2=2.5)
    assert v.status == "changed" and "y" in v.metrics
    assert all("a=3" not in k for k in v.keys)


def test_a_setting_that_failed_for_another_reason_is_still_compared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the time limit's cut is left out: a trial that raised and now runs is a change."""
    root, copy = _delivered(tmp_path, reason_c="ValueError: bad")
    v = _judged(tmp_path, monkeypatch, root, copy)
    assert v.status == "changed" and any(k.startswith("trials/a=3/") for k in v.keys)
    assert rerun_check.cut_settings(root) == set()


# ---- the whole study, run again through FI's trial runner -------------------------------------------------------------
SIM = "import time\n\ndef run_trial(cell, trial_id, seed):\n    return {'y': cell['a'] * 1.0 + (seed % 97) / 1000.0}\n"
EXPERIMENT = '''import json, os
t = json.load(open(os.environ["FI_TRIALS"], encoding="utf-8"))
by = {c["key"]: sum(c["metrics"]["y"]["values"]) / len(c["metrics"]["y"]["values"]) for c in t["cells"] if c["ok"]}
os.makedirs("data/results", exist_ok=True)
json.dump({"derived": {"y": {"status": "ok", "by_setting": by}}}, open("data/results/derived.json", "w"))
'''
PROTOCOL = {"grid": {"a": [1, 2, 3]}, "runs_per_setting": 3, "metrics": [{"id": "y"}]}


def _executor() -> SharedInterpreterExecutor:
    return SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")


def _delivered_by_fi(tmp_path: Path, *, cut_last: bool) -> Path:
    """A quest whose study FI's trial runner ran (its ledger, record of the trials and run record), then its analysis. With
    ``cut_last`` the third setting is recorded as its time limit cut it off, as the runner records it."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(EXPERIMENT, encoding="utf-8")
    cells = trial_runner.cells(PROTOCOL["grid"])
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", PROTOCOL["grid"], runs_per_setting=3, base_seed=0,
        deterministic=False, timeout_s=120, at_once=2, cell_list=cells[:2] if cut_last else None))
    if cut_last:
        cut = trial_runner.CellRun(key="a=3", cell=cells[2], planned=3, rows=[
            {"trial": t, "seed": trial_runner.trial_seed(0, "a=3", t), "status": "failed", "reason": trial_runner.OUT_OF_TIME}
            for t in range(3)])
        every = [*run.cells, cut]
        (root / "raw" / "trials.json").write_text(json.dumps(trial_runner.summary_of(every)), encoding="utf-8")
        with open(root / "raw" / "ledger.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "planned", "cell": "a=3", "trials": 3}) + "\n")
            for r in cut.rows:
                fh.write(json.dumps({"event": "trial", "cell": "a=3", **r}) + "\n")
    import os
    import subprocess

    subprocess.run([sys.executable, "-B", "code/experiment.py"], cwd=root, check=True,
                   env={**os.environ, "FI_TRIALS": "raw/trials.json"})
    run_facts.record_trials(root, key="k", wall_s=5.0, trials=9, cells=3, trial_sum_s=None, at_once=2)
    return root


def _repaired(root: Path, tmp_path: Path, name: str, text: str) -> Path:
    copy = rerun_check.copy_code(root, tmp_path / name)
    (copy / "simulate.py").write_text(text, encoding="utf-8")
    return copy


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for folder in ("raw", "data", ".fi")
            for p in sorted((root / folder).rglob("*")) if p.is_file()}


def test_the_whole_study_is_run_again_through_fis_trial_runner_as_the_delivered_run_ran(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: the study's own run.py, one setting after another, every setting of the grid (judge took no ``at_once``)."""
    root = _delivered_by_fi(tmp_path, cut_last=True)
    assert rerun_check.parallel(root, PROTOCOL, 4) == 2  # as many at once as the delivered run had
    asked: list[dict[str, Any]] = []
    real = trial_runner.run_trials

    async def spy(*a: Any, **k: Any) -> Any:
        asked.append(k)
        return await real(*a, **k)

    monkeypatch.setattr(trial_runner, "run_trials", spy)
    before = _snapshot(root)
    same = _repaired(root, tmp_path, "c0", SIM + "# repaired, numbers unchanged\n")
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, same, PROTOCOL, 120, None, at_once=2))
    assert (v.status, v.mode) == ("unchanged", "whole study"), v
    assert [trial_runner.cell_key(c) for c in asked[0]["cell_list"]] == ["a=1", "a=2"]  # not the setting cut off
    assert asked[0]["at_once"] == 2 and asked[0]["base_seed"] == 0 and asked[0]["per_process"] == 1
    assert _snapshot(root) == before  # the quest's own record of the trials is untouched
    # known positive: a repair that moves the numbers is "changed"
    moved = _repaired(root, tmp_path, "c1", SIM.replace("* 1.0", "* 1.5"))
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, moved, PROTOCOL, 120, None, at_once=2))
    assert v.status == "changed" and v.metrics == ["y"] and all("a=3" not in k for k in v.keys), v


def test_a_study_whose_numbers_are_only_in_fis_record_of_its_trials_is_run_again(tmp_path: Path) -> None:
    """A delivered run that left no result file of its own: its numbers are read from FI's record of the trials, so that
    record is what the rerun must write (old code asked for the analysis record's name and answered "wrote no results")."""
    root = _delivered_by_fi(tmp_path, cut_last=False)
    (root / "data" / "results" / "derived.json").unlink()
    (root / "code" / "experiment.py").write_text("print('the analysis writes no result file')\n", encoding="utf-8")
    assert rerun_check.results_of(root) and not rerun_check.result_files(root, "data/results/derived.json")
    same = _repaired(root, tmp_path, "c0", SIM + "# repaired, numbers unchanged\n")
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, same, PROTOCOL, 120, None, at_once=2))
    assert (v.status, v.mode) == ("unchanged", "whole study"), v
    moved = _repaired(root, tmp_path, "c1", SIM.replace("* 1.0", "* 1.5"))  # known positive
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, moved, PROTOCOL, 120, None, at_once=2))
    assert v.status == "changed", v
    # planted: an analysis that crashes is no run, though the trial runner wrote its record of the trials again
    crash = _repaired(root, tmp_path, "c2", SIM + "# repaired\n")
    (crash / "experiment.py").write_text("raise RuntimeError('the analysis fails')\n", encoding="utf-8")
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, crash, PROTOCOL, 120, None, at_once=2))
    assert v.status == "could_not" and v.crashed, v
    # planted: a repair of the analysis alone compares nothing here (it runs on the delivered record of the trials, and the
    # analysis writes no result file FI reads), so it is never "unchanged"
    analysis = rerun_check.copy_code(root, tmp_path / "c3")
    (analysis / "experiment.py").write_text("print('a repaired analysis')\n", encoding="utf-8")
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, analysis, PROTOCOL, 120, None, at_once=2))
    assert v.mode == "analysis only" and v.status == "could_not", v


def test_settings_that_shared_a_process_in_the_delivered_run_share_one_in_the_rerun(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The delivered run put several settings in one process (after FI's check that this gives the same numbers): the rerun
    does too, in the same groups, so it takes the time the delivered run took (old code: one process per setting)."""
    from core import run_estimate

    root = _delivered_by_fi(tmp_path, cut_last=False)
    run_facts.record_trials(root, key="k", wall_s=5.0, trials=9, cells=3, trial_sum_s=None, at_once=2, per_process=2)
    run_estimate.write_record(root, {"fits": True, "per_process": 2, "shared_groups": [["a=1", "a=3"], ["a=2"]],
                                     "group_allowance_s": 60.0})
    asked: list[dict[str, Any]] = []
    real = trial_runner.run_trials

    async def spy(*a: Any, **k: Any) -> Any:
        asked.append(k)
        return await real(*a, **k)

    monkeypatch.setattr(trial_runner, "run_trials", spy)
    same = _repaired(root, tmp_path, "c0", SIM + "# repaired, numbers unchanged\n")
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, same, PROTOCOL, 120, None, at_once=2))
    assert v.status == "unchanged", v
    assert (asked[0]["per_process"], asked[0]["groups"], asked[0]["group_allowance_s"]) == (2, [["a=1", "a=3"], ["a=2"]], 60.0)


def test_a_rerun_through_the_trial_runner_that_crashes_or_runs_out_of_time_is_could_not(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.INFO)
    root = _delivered_by_fi(tmp_path, cut_last=False)
    broken = _repaired(root, tmp_path, "c0", "def run_trial(cell, trial_id, seed):\n    raise ValueError('broken')\n")
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, broken, PROTOCOL, 120, None, at_once=2))
    assert v.status == "could_not" and v.crashed and "ValueError" in v.why and not v.timed_out
    # The study's time runs out before its last setting: the trial runner records that setting's trials as cut off. The
    # runner's own answer is stood in for (no real wait on a clock): the real run of the repair, with the last setting's
    # trials then marked as the runner marks the ones its time limit cut off, handed back only once the rerun has said
    # where it is, as the main run says it.
    monkeypatch.setattr(rerun_check, "PROGRESS_EVERY_S", 0.01)
    real = trial_runner.run_trials

    def said() -> list[str]:
        return [r.getMessage() for r in caplog.records if "still running" in r.getMessage()]

    async def cut_short(*a: Any, **k: Any) -> Any:
        run = await real(*a, **k)
        for c in run.cells:
            if c.key == "a=3":
                c.rows = [{"trial": r["trial"], "seed": r.get("seed"), "status": "failed", "reason": trial_runner.OUT_OF_TIME}
                          for r in c.rows]
        for _ in range(500):  # at most 5 s; the first line comes after 0.01 s
            if said():
                break
            await asyncio.sleep(0.01)
        return run

    monkeypatch.setattr(trial_runner, "run_trials", cut_short)
    slow = _repaired(root, tmp_path, "c1", SIM + "# repaired\n")
    v = asyncio.run(rerun_check.judge(_executor(), sys.executable, root, slow, PROTOCOL, 120, None, at_once=2))
    # a setting the delivered run ran and the rerun did not reach is neither a change nor "unchanged"
    assert v.status == "could_not" and v.timed_out and not v.crashed and "did not finish" in v.why, v
    assert said() and "settings done" in said()[-1], said()


def test_other_seeds_than_the_delivered_runs_are_refused_before_anything_runs(tmp_path: Path) -> None:
    root = _delivered_by_fi(tmp_path, cut_last=False)
    copy = _repaired(root, tmp_path, "c0", SIM + "# repaired\n")
    rows = [json.loads(line) for line in (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r.get("event") == "trial":
            r["seed"] = int(r["seed"]) + 1
    (root / "raw" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    calls = _Calls()
    v = asyncio.run(rerun_check.judge(calls, sys.executable, root, copy, PROTOCOL, 120, None, at_once=2))
    assert v.status == "could_not" and "seeds" in v.why and calls.calls == []


# ---- an answer "could not rerun" of the same repaired code is not run again ---------------------------------------------
def test_a_repaired_code_that_crashed_is_not_run_again_at_the_next_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: a repair that crashed was run twice in each check (once more after asking again) and again at every check."""
    from tests.test_rerun_check import _engine_with_code

    engine, root, issue, record = _engine_with_code(tmp_path)
    judged: list[int] = []
    asked: list[str] = []

    async def judge(*a: Any, **k: Any) -> Verdict:
        judged.append(1)
        return Verdict("could_not", "the repaired code did not run to the end (KeyError: x)", mode="whole study", crashed=True)

    async def ask(issue: Any, rel: str, copy_dir: Path, protocol: Any, state: Any, problem: str = "") -> dict[str, str]:
        asked.append(problem)
        return {"derive.py": "def derive_cover(table):\n    return 0.9  # repaired\n"}

    async def passes(issue: Any, copy_dir: Path) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(rerun_check, "judge", judge)
    engine._ask_repair_on_copy = ask  # type: ignore[method-assign,assignment]
    engine._repair_passes_own_test = passes  # type: ignore[method-assign,assignment]
    record["rerun_calls"] = 0
    got = asyncio.run(engine._rerun_one(issue, record, {}, {}))
    assert got.status == "could_not" and got.crashed and judged == [1] and len(asked) == 2
    assert len(record["rerun_could_not"]) == 1
    record["rerun_calls"] = 0
    got = asyncio.run(engine._rerun_one(issue, record, {}, {}))  # the next check: the same repair, the same code
    assert got.status == "could_not" and got.crashed and "KeyError" in got.why and judged == [1]

    # known positive: another repaired code is run
    async def ask_other(issue: Any, rel: str, copy_dir: Path, protocol: Any, state: Any, problem: str = "") -> dict[str, str]:
        return {"derive.py": "def derive_cover(table):\n    return 0.8\n"}

    engine._ask_repair_on_copy = ask_other  # type: ignore[method-assign,assignment]
    record["rerun_calls"] = 0
    asyncio.run(engine._rerun_one(issue, record, {}, {}))
    assert judged == [1, 1]


def test_only_a_lasting_could_not_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A timeout in the whole time a rerun is allowed is kept; one cut short by the time the check had left is not (a later
    check may have the time), and neither is an estimate made before anything ran."""
    from tests.test_rerun_check import _engine_with_code

    engine, root, issue, record = _engine_with_code(tmp_path)
    engine.config.execution.timeout_s = 900  # a rerun is allowed 1.5 x 3 x 30 s = 135 s
    answers = [Verdict("could_not", "not started", mode="whole study"),
               Verdict("could_not", "did not finish", mode="whole study", timed_out=True)]
    judged: list[float] = []

    async def judge(*a: Any, **k: Any) -> Verdict:
        judged.append(a[5])
        return answers[min(len(judged), len(answers)) - 1]

    async def ask(*a: Any, **k: Any) -> dict[str, str]:
        return {"simulate.py": "def run_cell(cell):\n    return {'q': 1.0}  # repaired\n"}

    async def passes(issue: Any, copy_dir: Path) -> tuple[bool, str]:
        return True, ""

    async def no_refusal(*a: Any, **k: Any) -> str:
        return ""

    issue["file"] = record["issues"]["F-1"]["file"] = "simulate.py"
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {'q': 1.0}\n", encoding="utf-8")
    monkeypatch.setattr(rerun_check, "judge", judge)
    engine._ask_repair_on_copy = ask  # type: ignore[method-assign,assignment]
    engine._repair_passes_own_test = passes  # type: ignore[method-assign,assignment]
    engine._repair_small_run_refusal = no_refusal  # type: ignore[method-assign,assignment]
    record["rerun_calls"] = 0
    asyncio.run(engine._rerun_one(issue, record, {}, {}))  # the estimate: nothing kept
    assert not record.get("rerun_could_not")
    record["rerun_calls"] = 0
    asyncio.run(engine._rerun_one(issue, record, {}, {}, [70.0]))  # timed out in less than the whole allowance: not kept
    assert not record.get("rerun_could_not") and judged[-1] <= 70.0 < judged[0]
    record["rerun_calls"] = 0
    asyncio.run(engine._rerun_one(issue, record, {}, {}))  # timed out in the whole allowance: kept
    assert len(record["rerun_could_not"]) == 1
    record["rerun_calls"] = 0
    asyncio.run(engine._rerun_one(issue, record, {}, {}))
    assert len(judged) == 3


def test_a_kept_could_not_is_not_reused_once_another_file_the_rerun_reads_changed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same repaired code with another frozen protocol (what each run is given) or other files under ``inputs/`` would
    run other trials: the answer kept for it is not reused (old key: the code, the delivered results and the time only)."""
    from core import frozen_protocol
    from tests.test_rerun_check import _engine_with_code

    engine, root, issue, record = _engine_with_code(tmp_path)
    engine.config.execution.timeout_s = 900
    judged: list[int] = []

    async def judge(*a: Any, **k: Any) -> Verdict:
        judged.append(1)
        return Verdict("could_not", "the repaired code did not run to the end (KeyError: x)", mode="whole study", crashed=True)

    async def ask(*a: Any, **k: Any) -> dict[str, str]:
        return {"simulate.py": "def run_cell(cell):\n    return {'q': 1.0}  # repaired\n"}

    async def passes(issue: Any, copy_dir: Path) -> tuple[bool, str]:
        return True, ""

    async def no_refusal(*a: Any, **k: Any) -> str:
        return ""

    issue["file"] = record["issues"]["F-1"]["file"] = "simulate.py"
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {'q': 1.0}\n", encoding="utf-8")
    monkeypatch.setattr(rerun_check, "judge", judge)
    engine._ask_repair_on_copy = ask  # type: ignore[method-assign,assignment]
    engine._repair_passes_own_test = passes  # type: ignore[method-assign,assignment]
    engine._repair_small_run_refusal = no_refusal  # type: ignore[method-assign,assignment]

    def check() -> None:
        record["rerun_calls"] = 0
        asyncio.run(engine._rerun_one(issue, record, {}, {}))

    check()
    assert judged == [1] and len(record["rerun_could_not"]) == 1  # crashed: kept (asked once more, the same code)
    check()
    assert judged == [1]  # nothing else changed: the kept answer is used
    frozen = frozen_protocol.frozen_path(root)
    data = json.loads(frozen.read_text(encoding="utf-8"))
    data["protocol"] = {**data["protocol"], "in_each_run": {"steps": 200}}
    frozen.write_text(json.dumps(data), encoding="utf-8")
    check()
    assert len(judged) == 2  # another frozen protocol: run again
    check()
    assert len(judged) == 2
    (root / "inputs").mkdir(exist_ok=True)
    (root / "inputs" / "table.csv").write_text("x\n1\n", encoding="utf-8")
    check()
    assert len(judged) == 3  # another file under inputs/: run again


def test_a_rerun_that_will_not_fit_is_not_started_by_the_gate_nor_its_small_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The gate's own path: the delivered run took longer than a rerun may take, so neither FI's small run of the repair
    nor the study is run, and run.log says why (old code ran the small run, then the study for its whole limit)."""
    import logging

    from core import code_checks
    from core.engine import Engine
    from tests.test_rerun_check import SIM_KEEPS_CONTRACT, _engine_with_code

    engine, root, issue, record = _engine_with_code(tmp_path)
    engine.config.execution.timeout_s = 28_800
    _record(root, wall_s=29_073.0, at_once=16, settings_s=30.0)
    issue["file"] = record["issues"]["F-1"]["file"] = "simulate.py"
    (root / "code" / "simulate.py").write_text(SIM_KEEPS_CONTRACT, encoding="utf-8")
    ran: list[str] = []

    async def judge(*a: Any, **k: Any) -> Verdict:
        ran.append("study")
        return Verdict("unchanged")

    async def refusal(*a: Any, **k: Any) -> str:
        ran.append("small run")
        return ""

    async def ask(*a: Any, **k: Any) -> dict[str, str]:
        return {"simulate.py": SIM_KEEPS_CONTRACT + "# repaired" + chr(10)}

    async def passes(issue: Any, copy_dir: Path) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(rerun_check, "judge", judge)
    monkeypatch.setattr(code_checks, "small_run_refusal", refusal)
    engine._ask_repair_on_copy = ask  # type: ignore[method-assign,assignment]
    engine._repair_passes_own_test = passes  # type: ignore[method-assign,assignment]
    engine._log = logging.getLogger("test_rerun_like_main.gate")  # type: ignore[assignment]
    caplog.set_level(logging.INFO)
    record["rerun_calls"] = 0
    got = asyncio.run(Engine._rerun_shown(engine, [(issue, "its test fails")], record, {}))["F-1"]
    assert got.status == "could_not" and "not started" in got.why and got.seconds == 0.0 and ran == [], (got, ran)
    said = [r.getMessage() for r in caplog.records if "[evidence_gate] F-1" in r.getMessage()]
    assert any("could not be run again" in m and "8 h 5 min" in m for m in said), said
    assert not record.get("rerun_could_not")  # an estimate is no lasting answer: a later check with more time may run it



def test_a_rerun_input_that_is_there_but_cannot_be_read_keeps_no_answer(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    # A frozen protocol that cannot be read gives the same key whatever it holds: no kept answer may be reused then.
    from core import frozen_protocol, reuse_key

    root = tmp_path / "quest"
    proto = frozen_protocol.frozen_path(root)
    proto.parent.mkdir(parents=True, exist_ok=True)
    proto.write_text("{}", encoding="utf-8")
    assert rerun_check.inputs_of(root)["complete"] == "yes"
    real = reuse_key.digest

    def unreadable(path: Any) -> str:
        if Path(path) == proto:
            raise PermissionError("locked")
        return real(path)

    monkeypatch.setattr(reuse_key, "digest", unreadable)
    assert rerun_check.inputs_of(root)["complete"] == "no"
    proto.unlink()
    assert rerun_check.inputs_of(root)["complete"] == "yes", "a file that is not there is a fact the key holds"
