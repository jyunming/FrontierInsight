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


@pytest.mark.asyncio
async def test_a_frozen_study_keeps_its_search_after_fi_is_updated_and_says_so(tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """FI's own harness and search code are not what a frozen study's search was made from: after an update of FI the
    confirm stage keeps the search (it may not search again) and says what changed; a change to the simulation still stops."""
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    await _run(root, protocol)
    spent = _count(root)
    monkeypatch.setattr(trial_runner, "HARNESS_SOURCE", trial_runner.HARNESS_SOURCE + "\n# an updated harness\n")
    kept = await _run(root, protocol, frozen=True)
    assert kept.reused is True and not kept.problems and _count(root) == spent, kept.problems
    note = json.loads((root / optimise.RECORD).read_text(encoding="utf-8"))["reuse"]["note"]
    assert "FI's own trial harness" in note and "was not repeated" in note
    sim = root / "code" / "simulate.py"
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


@pytest.mark.asyncio
async def test_a_file_that_becomes_unreadable_after_the_search_does_not_stop_a_frozen_study(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    (root / "code" / "params.txt").write_text("k = 1\n", encoding="utf-8")
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6})
    await _run(root, protocol)
    spent = _count(root)
    real = reuse_key.version_digest

    def version_digest(path: Any) -> str:
        if Path(path).name == "params.txt":
            raise PermissionError("locked")
        return real(path)

    monkeypatch.setattr(reuse_key, "version_digest", version_digest)
    kept = await _run(root, protocol, frozen=True)
    assert kept.reused is True and not kept.problems and _count(root) == spent, kept.problems
    # a readable file that really changed is still another study
    (root / "code" / "simulate.py").write_text(SIM.replace("** 2", "** 3"), encoding="utf-8")
    assert (await _run(root, protocol, frozen=True)).problems


def test_a_scan_copy_that_cannot_be_made_leaves_no_copy_of_an_earlier_search(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    root = _toy(tmp_path)
    (root / "raw").mkdir()
    (root / "raw" / "trials.json").write_text("{}", encoding="utf-8")
    (root / optimise.SCAN_DIR).mkdir(parents=True)
    (root / optimise.SCAN_DIR / "trials.json").write_text('{"older": true}', encoding="utf-8")
    monkeypatch.setattr(shutil, "copyfile", lambda *a, **k: (_ for _ in ()).throw(PermissionError("no")))
    optimise.keep_scan_copy(root)
    assert optimise.scan_hashes(root) == {}


@pytest.mark.asyncio
async def test_a_data_file_changed_after_the_freeze_does_not_stop_the_confirm_stage(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    await _run(root, protocol)
    spent = _count(root)
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "readings.csv").write_text("t,v\n0,5\n", encoding="utf-8")
    kept = await _run(root, protocol, frozen=True)
    assert kept.reused is True and not kept.problems and _count(root) == spent, kept.problems


def test_the_small_records_follow_the_same_list_including_data(tmp_path: Path) -> None:
    from core import cost_model, outside_run

    root = _toy(tmp_path)
    sim = root / "code" / "simulate.py"
    before = (outside_run.code_version(root), trial_runner.code_inputs_sha(root, SIM), cost_model.design_key({"g": 1}, sim))
    (root / "data" / "readings.csv").write_text("t,v\n0,3\n", encoding="utf-8")
    after = (outside_run.code_version(root), trial_runner.code_inputs_sha(root, SIM), cost_model.design_key({"g": 1}, sim))
    assert all(a != b for a, b in zip(before, after))
    (root / "data" / "results").mkdir()
    (root / "data" / "results" / "x.bin").write_bytes(b"1")
    assert (outside_run.code_version(root), trial_runner.code_inputs_sha(root, SIM), cost_model.design_key({"g": 1}, sim)) == after


# --- what a run actually opened (the files the key leaves out) ---------------------------------------------------------

READS_OTHER = '''
import os

def run_cell(cell):
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "notes_for_run.txt"), encoding="utf-8") as fh:
        return {"f": float(len(fh.read()))}
'''

READS_DATA = '''
import os

def run_cell(cell):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "data", "readings.csv"), encoding="utf-8") as fh:
        n = len(fh.read())
    os.makedirs(os.path.join(root, "data", "results", "cache"), exist_ok=True)
    with open(os.path.join(root, "data", "results", "cache", "tmp.txt"), "w", encoding="utf-8") as fh:
        fh.write("x")
    return {"f": float(n)}
'''


def _opened_run(tmp_path: Path, source: str, extra: dict[str, str]) -> tuple[Path, Any]:
    import asyncio

    from core.execution import SharedInterpreterExecutor

    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "readings.csv").write_text("t,v\n0,1\n", encoding="utf-8")
    for rel, text in extra.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), sys.executable, root, "code/simulate.py", {"x": [1]},
                                              runs_per_setting=1, base_seed=0, deterministic=True, timeout_s=120))
    return root, run


def test_the_files_a_run_opened_are_recorded_and_the_run_written_folders_are_left_out(tmp_path: Path) -> None:
    root, run = _opened_run(tmp_path, READS_DATA, {})
    opened = run.opened()
    assert "data/readings.csv" in [o.lower() for o in opened] or any(o.endswith("data/readings.csv") for o in opened), opened
    files = reuse_key.files(root, "simulation").digests
    assert reuse_key.left_out(opened, files) == [], "the data file is in the key; the cache folder is written by the run"


def test_a_run_that_read_a_file_the_key_leaves_out_is_not_reused_and_the_file_is_named(tmp_path: Path) -> None:
    # a note beside the simulation: FI writes notes like README.md itself, so the key leaves them out
    root, run = _opened_run(tmp_path, READS_OTHER.replace("notes_for_run.txt", "README.md"), {"code/README.md": "hello"})
    inputs = trial_runner._run_inputs(root / "code" / "simulate.py", {"grid": {"x": [1]}}, 1, 0, True)
    left = reuse_key.left_out(run.opened(), inputs.files)
    assert left == ["code/README.md"], (left, run.opened())
    (root / ".fi" / "trials").mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    trial_runner._save_run(root, inputs.key, run, inputs)
    assert trial_runner._load_run(root, inputs.key, inputs) is None
    assert "code/README.md" in trial_runner._why_not_reused(root, inputs)


# ---- slice 4: the checks and the numeric-warnings resume read the same files ------------------------------------------


def test_the_hidden_check_version_follows_the_simulation_files_and_not_the_analysis(tmp_path: Path) -> None:
    from core import hidden_check

    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "inputs").mkdir()
    (root / "code" / "simulate.py").write_text("def run_cell(c):\n    return 1\n", encoding="utf-8")
    (root / "inputs" / "p.csv").write_text("a\n1\n", encoding="utf-8")
    base = hidden_check.code_sha(root)
    assert base
    (root / "code" / "experiment.py").write_text("x = 1\n", encoding="utf-8")  # the analysis: no new check
    assert hidden_check.code_sha(root) == base
    (root / "inputs" / "p.csv").write_text("a\n2\n", encoding="utf-8")  # the person's file: the simulation may read it
    assert hidden_check.code_sha(root) != base


def test_a_repair_in_the_package_is_not_accepted_as_unchanged_after_a_numeric_stop(tmp_path: Path) -> None:
    from core.engine import Engine
    from tests.test_numeric_warnings import _cfg

    engine = Engine(_cfg(tmp_path))
    code = engine.quest_root / "code"
    (code / "pkg").mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text("import pkg\n", encoding="utf-8")
    (code / "experiment.py").write_text("print(1)\n", encoding="utf-8")
    (code / "pkg" / "__init__.py").write_text("K = 1\n", encoding="utf-8")
    marker = engine.fi_dir / "numeric_stop.json"
    marker.parent.mkdir(parents=True, exist_ok=True)

    def stop(rule: int) -> None:
        marker.write_text(json.dumps({"scripts_sha": engine._scripts_sha(rule), "scripts_rule": rule}), encoding="utf-8")

    stop(2)
    assert engine._resume_after_numeric_stop() == {"numeric_warnings_accepted": True}
    stop(2)
    (code / "pkg" / "__init__.py").write_text("K = 2\n", encoding="utf-8")
    assert engine._resume_after_numeric_stop().get("exec_patch_pending") is True
    # a marker written by the first rule (two scripts only) is read by it: the package is not looked at
    marker.write_text(json.dumps({"scripts_sha": engine._scripts_sha(1)}), encoding="utf-8")
    (code / "pkg" / "__init__.py").write_text("K = 3\n", encoding="utf-8")
    assert engine._resume_after_numeric_stop() == {"numeric_warnings_accepted": True}


def test_a_frozen_studys_old_search_key_is_made_with_the_harness_as_it_was_before_the_recording_of_opened_files() -> None:
    """The confirm stage may not search again: the old rule's key holds the harness text of the FI that wrote the record, so
    a change to the harness (the recording of opened files) must not make that record a stranger. The hash is that of the harness
    text of the FI before the recording (on this line of work: the harness with core/in_run.py as the fixes before the recording
    left it, since the old rule hashed the harness text whole)."""
    import hashlib

    assert "addaudithook" in trial_runner.HARNESS_SOURCE and "addaudithook" not in trial_runner.LEGACY_HARNESS_SOURCE
    assert hashlib.sha256(trial_runner.LEGACY_HARNESS_SOURCE.encode()).hexdigest() == (
        "ce17183f5369dcf349c270de63a8c05a82efa53a9f0114cdf9c90703d90a4bb9")
