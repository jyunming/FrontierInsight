"""One key per kind of work FI reuses a record for (core/reuse_key.py): the trials, the search for the best design, the
timing of a run. A neutral toy simulation; no model call.

What must hold: a change to any file the run can read (the simulation, a module, the person's ``inputs/``, a file of
``data/``, a comment) is another key; the analysis FI writes and what a run writes under ``data/results/`` are not; an
unreadable file makes the key one no other read can match; a record an older FI wrote is not reused; and a frozen
study's record of an older FI is still kept (no new stop)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import optimise, reuse_key, trial_runner
from tests.test_optimise_runner import _count, _protocol, _quest, _run

SIM = "def run_cell(cell):\n    return {'f': cell['x'] ** 2}\n"


def _toy(tmp_path: Path) -> Path:
    root = tmp_path / "toy"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "inputs").mkdir()
    (root / "inputs" / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "readings.csv").write_text("t,v\n0,1\n", encoding="utf-8")
    return root


def _trial_key(root: Path) -> str:
    return trial_runner._run_key(root / "code" / "simulate.py", {"grid": {"x": [1, 2]}}, 3, 0, True)


def test_the_same_files_give_the_same_key(tmp_path: Path) -> None:
    root = _toy(tmp_path)
    assert _trial_key(root) == _trial_key(root)


@pytest.mark.parametrize("path,text", [
    ("code/simulate.py", SIM + "# a comment\n"),   # bytes, not tokens: a comment may hold a value the run reads
    ("code/model_helper.py", "K = 1\n"),
    ("inputs/table.csv", "a,b\n1,3\n"),
    ("data/readings.csv", "t,v\n0,2\n"),            # in the run's copy, in no key before
    ("data/more/extra.csv", "x\n1\n"),
])
def test_a_change_to_a_file_the_run_can_read_is_another_key(tmp_path: Path, path: str, text: str) -> None:
    root = _toy(tmp_path)
    before = _trial_key(root)
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    assert _trial_key(root) != before


def test_what_a_run_writes_and_the_analysis_are_not_the_simulation(tmp_path: Path) -> None:
    root = _toy(tmp_path)
    before = _trial_key(root)
    (root / "data" / "results").mkdir()
    (root / "data" / "results" / "cache.bin").write_bytes(b"\x00\x01")
    (root / "code" / "fi_table.py").write_text("X = 1\n", encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print(1)\n", encoding="utf-8")
    assert _trial_key(root) == before


def test_the_harness_is_part_of_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _toy(tmp_path)
    before = _trial_key(root)
    monkeypatch.setattr(trial_runner, "HARNESS_SOURCE", trial_runner.HARNESS_SOURCE + "\n# another harness\n")
    assert _trial_key(root) != before


def test_an_unreadable_file_is_never_reused_and_never_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _toy(tmp_path)
    real = reuse_key.digest

    def digest(path: Any) -> str:
        if Path(path).name == "table.csv":
            raise PermissionError("locked")
        return real(path)

    monkeypatch.setattr(reuse_key, "digest", digest)
    first = reuse_key.computed("trials", root, {"grid": {}}, seed=0, config={"runs": 1, "deterministic": True})
    second = reuse_key.computed("trials", root, {"grid": {}}, seed=0, config={"runs": 1, "deterministic": True})
    assert not first.complete and first.unreadable == ["inputs/table.csv"]
    assert first.key != second.key, "two unreadable reads must not match"
    assert not reuse_key.matches({"reuse": reuse_key.stamp(first)}, second)
    assert "inputs/table.csv" in reuse_key.why_not({"reuse": reuse_key.stamp(first)}, second)


def test_changed_names_what_moved(tmp_path: Path) -> None:
    root = _toy(tmp_path)
    old = reuse_key.computed("trials", root, {"grid": {}}, seed=0, config={"runs": 1, "deterministic": True})
    (root / "data" / "readings.csv").write_text("t,v\n0,9\n", encoding="utf-8")
    new = reuse_key.computed("trials", root, {"grid": {}}, seed=1, config={"runs": 1, "deterministic": True})
    assert new.changed(old) == ["data/readings.csv", "the seed"]
    assert "data/readings.csv" in reuse_key.why_not({"reuse": reuse_key.stamp(old)}, new)


def _run_record(root: Path) -> trial_runner.TrialRun:
    ledger, summary = root / "raw" / "ledger.jsonl", root / "raw" / "trials.json"
    ledger.parent.mkdir(exist_ok=True)
    ledger.write_text("{}\n", encoding="utf-8")
    summary.write_text("{}", encoding="utf-8")
    return trial_runner.TrialRun(cells=[], ledger_path=ledger, summary_path=summary)


def test_a_trial_record_of_an_older_fi_is_not_used_and_a_current_one_is(tmp_path: Path) -> None:
    root = _toy(tmp_path)
    (root / ".fi" / "trials").mkdir(parents=True)
    inputs = trial_runner._run_inputs(root / "code" / "simulate.py", {"grid": {}}, 1, 0, True)
    run = _run_record(root)
    trial_runner._save_run(root, inputs.key, run)  # no reuse block: what an older FI wrote
    assert trial_runner._load_run(root, inputs.key, inputs) is None
    assert "older version" in trial_runner._why_not_reused(root, inputs)
    trial_runner._save_run(root, inputs.key, run, inputs)
    assert trial_runner._load_run(root, inputs.key, inputs) is not None
    (root / "data" / "readings.csv").write_text("t,v\n0,7\n", encoding="utf-8")
    moved = trial_runner._run_inputs(root / "code" / "simulate.py", {"grid": {}}, 1, 0, True)
    assert trial_runner._load_run(root, moved.key, moved) is None
    assert "data/readings.csv" in trial_runner._why_not_reused(root, moved)


def test_the_sizing_key_names_the_rule_it_was_made_by() -> None:
    assert reuse_key.RULES["sizing"] >= 2


# --- the search --------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_search_is_kept_until_a_file_it_can_read_changes(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6})
    first = await _run(root, protocol)
    assert first.reused is False and reuse_key.is_current(json.loads((root / optimise.RECORD).read_text(encoding="utf-8")),
                                                         "search")
    spent = _count(root)
    again = await _run(root, protocol)
    assert again.reused is True and _count(root) == spent
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "readings.csv").write_text("t,v\n0,1\n", encoding="utf-8")  # a file the run can read, new
    after = await _run(root, protocol)
    assert after.reused is False and _count(root) > spent, "a changed data/ file is another search"


@pytest.mark.asyncio
async def test_a_search_does_not_depend_on_the_file_others_write(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    run = await _run(root, protocol)
    assert run.scan, "this search has a coarse scan"
    spent = _count(root)
    # another run of the trials (the improve step's measurement) rewrites the shared scan file
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": [], "other": "run"}), encoding="utf-8")
    again = await _run(root, protocol)
    assert again.reused is True and _count(root) == spent, "the search keeps its own copy of its scan"
    # its own copy changed: the record no longer stands for it
    (root / optimise.SCAN_DIR / "trials.json").write_text("{}", encoding="utf-8")
    third = await _run(root, protocol)
    assert third.reused is False


@pytest.mark.asyncio
async def test_a_search_record_of_an_older_fi_is_run_again(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6})
    await _run(root, protocol)
    spent = _count(root)
    path = root / optimise.RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("reuse")
    path.write_text(json.dumps(record), encoding="utf-8")
    again = await _run(root, protocol)
    assert again.reused is False and _count(root) > spent


def test_the_stop_variants_and_the_rounds_come_from_the_one_builder(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    block = _protocol(evaluation_budget={"starts": 1, "per_start": 6})["optimisation"]
    rounds = [{"added": 5, "target": None}, {"added": 7, "target": 1.0}]
    inputs = optimise._run_inputs(root, block, 0, "run_cell", None, rounds, False, True, True)
    assert inputs.prefix(1).key == optimise._run_key(root, root / "code" / "simulate.py", block, 0, "run_cell", None,
                                                    rounds[:1], False, True, True)
    assert inputs.prefix(0).key == optimise._run_key(root, root / "code" / "simulate.py", block, 0, "run_cell", None, [],
                                                    False, True, True)
    variants = inputs.stop_variants()
    assert inputs.key in variants and len(variants) == 2
    assert variants[inputs.key] == frozenset()


@pytest.mark.asyncio
async def test_a_frozen_studys_record_of_an_older_fi_is_kept_and_stays_kept(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    run = await _run(root, protocol)
    spent = _count(root)
    # the record as an FI before this check wrote it: its own study key, no reuse block
    block = optimise._plan.normalize(protocol["optimisation"])[0]
    path = root / optimise.RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("reuse")
    record["study_key"] = optimise._legacy_study_key(root, root / "code" / "simulate.py", block, "run_cell", None)
    path.write_text(json.dumps(record), encoding="utf-8")
    kept = await _run(root, protocol, frozen=True)
    assert kept.reused is True and not kept.problems and _count(root) == spent, kept.problems
    assert reuse_key.is_current(json.loads(path.read_text(encoding="utf-8")), "search"), "adopted: its scan is bound now"
    # the confirm run's own trials rewrite the shared scan file: the confirm still completes
    assert run.scan
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": [], "other": "run"}), encoding="utf-8")
    again = await _run(root, protocol, frozen=True)
    assert again.reused is True and not again.problems and _count(root) == spent


@pytest.mark.asyncio
async def test_a_frozen_study_whose_comments_changed_after_its_search_keeps_it_and_says_so(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    await _run(root, protocol)
    spent = _count(root)
    sim = root / "code" / "simulate.py"
    sim.write_text(sim.read_text(encoding="utf-8") + "\n# a note added after the search\n", encoding="utf-8")
    kept = await _run(root, protocol, frozen=True)
    assert kept.reused is True and not kept.problems and _count(root) == spent
    note = json.loads((root / optimise.RECORD).read_text(encoding="utf-8"))["reuse"]["note"]
    assert "code/simulate.py" in note and "was not repeated" in note
    # a change that is more than a comment is not the same study: no search record to keep
    sim.write_text(sim.read_text(encoding="utf-8").replace("** 2", "** 3"), encoding="utf-8")
    moved = await _run(root, protocol, frozen=True)
    assert moved.reused is False and moved.problems


def test_the_quality_report_says_a_kept_search_met_changed_comments(tmp_path: Path) -> None:
    from core import quality_report

    root = tmp_path / "quest"
    (root / ".fi" / "optimisation").mkdir(parents=True)
    note = "the simulation's comments or line endings changed after its search (code/simulate.py); the search was not repeated"
    (root / ".fi" / "optimisation" / "run.json").write_text(json.dumps({"reuse": {"schema": 2, "note": note}}),
                                                            encoding="utf-8")
    rows = quality_report._run_problems(quality_report._Quest(root))
    assert any("comments or line endings changed" in str(r) for r in rows), rows


@pytest.mark.asyncio
async def test_a_frozen_study_is_not_stopped_by_a_missing_scan_copy_or_an_unreadable_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    await _run(root, protocol)
    spent = _count(root)
    # the search's own copy of its scan is gone: main kept the search while the scan file in raw/ was there
    shutil_rmtree = __import__("shutil").rmtree
    shutil_rmtree(root / optimise.SCAN_DIR)
    kept = await _run(root, protocol, frozen=True)
    assert kept.reused is True and not kept.problems and _count(root) == spent
    # a file of the study that cannot be read: main skipped it in the key, so the frozen study is not refused for it
    real = reuse_key.digest

    def digest(path: Any) -> str:
        if Path(path).name == "simulate.py":
            raise PermissionError("locked")
        return real(path)

    monkeypatch.setattr(reuse_key, "digest", digest)
    again = await _run(root, protocol, frozen=True)
    assert again.reused is True and not again.problems and _count(root) == spent


@pytest.mark.asyncio
async def test_a_frozen_record_from_before_study_keys_is_kept_by_its_exploration_seed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import phased

    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    await _run(root, protocol)
    spent = _count(root)
    block = optimise._plan.normalize(protocol["optimisation"])[0]
    path = root / optimise.RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("reuse")
    record.pop("study_key")
    record["key"] = optimise._legacy_run_key(root, root / "code" / "simulate.py", block, 0, "run_cell", None)
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(phased, "load", lambda _root: {"explore_seed_bases": [0]})
    kept = await _run(root, protocol, frozen=True)
    assert kept.reused is True and not kept.problems and _count(root) == spent, kept.problems
