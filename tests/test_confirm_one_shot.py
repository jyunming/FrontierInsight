"""A confirmation is one try (core/phased.py, core/confirmations.py): each frozen version of the study (a candidate) is
confirmed once on data or seeds exploration never saw; running it again keeps both verdicts and the failure counts; a
version changed after its confirmation is a new candidate with a confirm run of its own; the evidence reads the current
candidate only; the paper's methods paragraph counts the candidates; the record is appended to, never rewritten, and
each line is named in the hash-chained decision trace.

No real model is called."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core import audit_log, confirmations, disclosure, phased, todo
from core import evidence as _evidence


def _env(seed: int, index: int) -> dict[str, str]:
    return {"PATH": "/bin", "FI_REPLICATE_SEED": str(seed), "FI_REPLICATE_INDEX": str(index)}


def _code(root: Path, text: str = "def run_cell(cell):\n    return {'y': 1.0}\n") -> None:
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(text, encoding="utf-8")


def _freeze_and_confirm(root: Path, result: dict | None, *, stride: int = 10) -> dict:
    """Exploration ran once, the study is frozen, its confirm run gives ``result``."""
    if phased.load(root) is None:
        phased.prepare(root, "q1")
    phased.seed_env(root, _env(0, 0))
    phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=stride, replicates=1,
                         explore_runs=1)
    phased.seed_env(root, _env(0, 0))  # the confirm run
    record, _ = phased.record_confirm(root, result)
    return record


def _ready_gaps(root: Path) -> list[str]:
    record = _evidence.assess(root, {"result_json": {"a": 1}}, settings=phased.evidence_settings(root))
    return next(level["gaps"] for level in record["ladder"] if level["level"] == "publication_ready")


# ---- the same candidate is confirmed once ---------------------------------------------------------------------------


def test_a_confirm_run_again_for_the_same_version_keeps_both_verdicts_and_the_failure_counts(tmp_path: Path) -> None:
    _code(tmp_path)
    record = _freeze_and_confirm(tmp_path, None)  # the one confirm run gave nothing
    assert phased.status(record) == "confirm_failed"
    # A retry of the same version on the confirm seeds (nothing in it changed): not a new version.
    assert phased.new_candidate(tmp_path) == []
    phased.seed_env(tmp_path, _env(0, 0))
    record, lines = phased.record_confirm(tmp_path, {"a": 2})
    lines_kept = confirmations.read(tmp_path)
    assert [e["verdict"] for e in lines_kept] == ["confirm_failed", "confirm_reused"], lines_kept
    assert all(e["candidate"] == 1 for e in lines_kept)
    assert phased.status(record) != phased.CONFIRMED
    assert confirmations.worst(lines_kept) == "confirm_failed", "a failure is never replaced by a later run"
    assert phased.evidence_settings(tmp_path)["phased"] != phased.CONFIRMED
    assert any("run again" in x for x in lines)


def test_a_failed_confirmation_says_it_cannot_be_tried_again(tmp_path: Path) -> None:
    _code(tmp_path)
    phased.prepare(tmp_path, "q1")
    phased.seed_env(tmp_path, _env(0, 0))
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    phased.seed_env(tmp_path, _env(0, 0))
    _record, lines = phased.record_confirm(tmp_path, None)
    assert phased.ONE_SHOT in lines
    item = todo.confirm_item(tmp_path)
    assert item is not None and todo.CONFIRM_FAILED in item.why
    assert "confirmation on unseen data did not agree" in item.why and "new candidate" in item.why


# ---- a changed version is a new candidate ---------------------------------------------------------------------------


def test_a_version_changed_after_its_confirmation_is_a_new_candidate_with_its_own_confirm_run(tmp_path: Path) -> None:
    _code(tmp_path)
    first = _freeze_and_confirm(tmp_path, {"a": 2}, stride=10)
    assert phased.status(first) == phased.CONFIRMED and phased.candidate(first) == 1
    first_base = first["confirm_seed_base"]
    # The review sent it back and the code was written again (review -> implement), or the improve loop kept a change.
    _code(tmp_path, "def run_cell(cell):\n    return {'y': 2.0}\n")
    lines = phased.new_candidate(tmp_path)
    assert lines and "version 2" in lines[0] and "code" in lines[0]
    record = phased.load(tmp_path)
    assert record["stage"] == phased.EXPLORE and phased.candidate(record) == 2
    assert record["earlier_candidates"] == [{"candidate": 1, "verdict": phased.CONFIRMED}]
    assert phased.status(record) == phased.EXPLORE, "the new version is not confirmed by the earlier one"
    assert any("version 2 of the study" in g for g in _ready_gaps(tmp_path))
    assert phased.new_candidate(tmp_path) == [], "once is enough: nothing happens again on a resume"
    # Its exploration runs on exploration seeds, then it is frozen and confirmed on seeds no earlier run used.
    assert phased.seed_env(tmp_path, _env(0, 0))["FI_REPLICATE_SEED"] == "0"
    record, out = phased.enter_confirm(tmp_path, explore_result={"a": 3}, frozen_sha256=None, stride=10, replicates=1,
                                       explore_runs=2)
    assert record["confirm_seed_base"] not in (0, first_base)
    assert any("version 2 of the study is frozen" in x for x in out)
    phased.seed_env(tmp_path, _env(0, 0))
    record, _ = phased.record_confirm(tmp_path, {"a": 4})
    assert phased.status(record) == phased.CONFIRMED
    kept = confirmations.read(tmp_path)
    assert [(e["candidate"], e["verdict"]) for e in kept] == [(1, "confirmed"), (2, "confirmed")]
    assert kept[0]["frozen"]["code_sha256"] != kept[1]["frozen"]["code_sha256"]


def test_a_repair_of_a_crashed_confirm_run_is_a_new_candidate_and_the_crash_is_kept(tmp_path: Path) -> None:
    _code(tmp_path)
    phased.prepare(tmp_path, "q1")
    phased.seed_env(tmp_path, _env(0, 0))
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    phased.seed_env(tmp_path, _env(0, 0))  # the confirm run crashed; the repair changes the code
    _code(tmp_path, "def run_cell(cell):\n    return {'y': 3.0}\n")
    lines = phased.new_candidate(tmp_path)
    assert lines and phased.load(tmp_path)["stage"] == phased.EXPLORE
    (only,) = confirmations.read(tmp_path)
    assert only["candidate"] == 1 and only["verdict"] == "confirm_failed" and "during its confirm run" in only["why"]


def test_a_redraw_helper_written_into_code_is_not_a_new_version(tmp_path: Path) -> None:
    _code(tmp_path)
    _freeze_and_confirm(tmp_path, {"a": 2})
    (tmp_path / "code" / "replot_figures.py").write_text("print('redraw')\n", encoding="utf-8")
    (tmp_path / "code" / "NOTES.md").write_text("notes\n", encoding="utf-8")
    assert phased.new_candidate(tmp_path) == [], "a re-run would otherwise get a fresh confirm run for nothing"


def test_a_new_version_of_a_quest_that_held_rows_back_cannot_be_confirmed_on_them(tmp_path: Path) -> None:
    data = tmp_path / "inputs" / "data" / "d.csv"
    data.parent.mkdir(parents=True)
    original = ("x,y\n" + "".join(f"{i},{i * i}\n" for i in range(60))).encode()
    data.write_bytes(original)
    _code(tmp_path)
    phased.prepare(tmp_path, "q1")
    record = phased.load(tmp_path)
    record["split_decided"] = True
    phased._save(tmp_path, record)
    _freeze_and_confirm(tmp_path, {"a": 2})
    _code(tmp_path, "def run_cell(cell):\n    return {'y': 5.0}\n")
    lines = phased.new_candidate(tmp_path)
    record = phased.load(tmp_path)
    assert phased.status(record) == "not_confirmable" and record["held_back_used"]
    assert any("cannot be confirmed" in x for x in lines)
    assert data.read_bytes() == original, "every row was read by now: the whole file is back"
    gaps = _ready_gaps(tmp_path)
    assert any("no rows remain that no version has seen" in g for g in gaps), gaps
    assert "no data remains that no version has seen" in phased.summary(record)
    # The data is not split again at the next start, and the next run is not a confirm run.
    again, _ = phased.prepare(tmp_path, "q1")
    assert again["files"] == [] and data.read_bytes() == original


def test_a_data_quest_redesigned_after_its_confirmation_is_a_new_version_that_cannot_be_confirmed(tmp_path: Path) -> None:
    phased.prepare(tmp_path, "q1", data_quest=True)
    record = phased.load(tmp_path)
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=1, replicates=1, explore_runs=1,
                         design_sha256="design-1")
    phased.data_quest_gate(tmp_path, "q1", design_sha256="design-1")  # the confirm reading
    phased.note_confirm_result(tmp_path, {"a": 2})
    phased.record_confirm(tmp_path, {"a": 2}, design_sha256="design-1")
    assert phased.status(phased.load(tmp_path)) == phased.CONFIRMED
    lines, question = phased.data_quest_gate(tmp_path, "q1", design_sha256="design-2")
    assert question == "" and lines and "design" in lines[0]
    record = phased.load(tmp_path)
    assert phased.candidate(record) == 2 and phased.status(record) == "not_confirmable"
    assert [e["verdict"] for e in confirmations.read(tmp_path)] == ["confirmed"]


# ---- the evidence reads the current candidate ------------------------------------------------------------------------


def test_the_evidence_reads_only_the_current_versions_confirmation(tmp_path: Path) -> None:
    _code(tmp_path)
    _freeze_and_confirm(tmp_path, None)  # version 1 did not hold
    assert any("produced no result" in g for g in _ready_gaps(tmp_path))
    _code(tmp_path, "def run_cell(cell):\n    return {'y': 9.0}\n")
    phased.new_candidate(tmp_path)
    phased.seed_env(tmp_path, _env(0, 0))
    phased.enter_confirm(tmp_path, explore_result={"a": 3}, frozen_sha256=None, stride=10, replicates=1, explore_runs=2)
    phased.seed_env(tmp_path, _env(0, 0))
    record, _ = phased.record_confirm(tmp_path, {"a": 4})
    assert phased.status(record) == phased.CONFIRMED
    settings = phased.evidence_settings(tmp_path)
    assert settings["phased"] == phased.CONFIRMED and settings["phased_candidate"] == "2"
    assert "version 1: confirm failed" in settings["phased_earlier"]
    gaps = _ready_gaps(tmp_path)
    assert not any("confirm" in g or "exploration" in g for g in gaps), gaps
    assert todo.confirm_item(tmp_path) is None, "the current version's confirmation held"
    assessed = _evidence.assess(tmp_path, {"result_json": {"a": 1}}, settings=settings)
    assert assessed["phased"]["version"] == 2 and "confirm failed" in assessed["phased"]["earlier_versions"]


# ---- the paper says how many versions were confirmed -----------------------------------------------------------------


def _attempt(root: Path, n: int) -> None:
    fi = root / ".fi"
    fi.mkdir(parents=True, exist_ok=True)
    with (fi / "attempts.jsonl").open("a", encoding="utf-8") as f:
        for _ in range(n):
            f.write(json.dumps({"kind": "run", "outcome": "ok"}) + "\n")


def test_the_methods_paragraph_counts_the_versions_and_the_confirmations_that_did_not_hold(tmp_path: Path) -> None:
    _code(tmp_path)
    _attempt(tmp_path, 3)
    _freeze_and_confirm(tmp_path, {"a": 2})
    once = disclosure.paragraph(tmp_path)
    assert "versions of the study" not in once and "version of the study" not in once, "confirmed once, as planned"
    phased.seed_env(tmp_path, _env(0, 0))  # run again on the confirm seeds: the second verdict does not hold
    phased.record_confirm(tmp_path, {"a": 5})
    _code(tmp_path, "def run_cell(cell):\n    return {'y': 7.0}\n")
    phased.new_candidate(tmp_path)
    text = disclosure.paragraph(tmp_path)
    assert "1 version of the study was confirmed once on data or seeds that exploration never saw" in text
    assert "1 confirmation did not hold" in text
    assert disclosure._LAST_UNCONFIRMED in text
    assert disclosure.is_engine_paragraph(text), text
    phased.seed_env(tmp_path, _env(0, 0))
    phased.enter_confirm(tmp_path, explore_result={"a": 3}, frozen_sha256=None, stride=10, replicates=1, explore_runs=2)
    phased.seed_env(tmp_path, _env(0, 0))
    phased.record_confirm(tmp_path, {"a": 4})
    text = disclosure.paragraph(tmp_path)
    assert "2 versions of the study were confirmed" in text and "1 confirmation did not hold" in text
    assert disclosure._LAST_UNCONFIRMED not in text
    assert disclosure.is_engine_paragraph(text), text
    assert disclosure.strip_for_checks(disclosure.mark_paper("# T\n\n## Methods\n\nx\n", text)).count("versions") == 0


# ---- the record is appended to and sealed -----------------------------------------------------------------------------


def test_the_record_is_append_only_and_a_changed_line_is_found(tmp_path: Path) -> None:
    _code(tmp_path)
    _freeze_and_confirm(tmp_path, None)
    phased.seed_env(tmp_path, _env(0, 0))
    phased.record_confirm(tmp_path, {"a": 2})
    path = confirmations.path(tmp_path)
    first = path.read_text(encoding="utf-8").splitlines()
    assert len(first) == 2 and json.loads(first[1])["prev_sha256"] == confirmations._sha(first[0])
    assert confirmations.problems(tmp_path, None) == []
    # Rewriting the failed verdict as confirmed is found.
    edited = json.loads(first[0])
    edited["verdict"] = "confirmed"
    path.write_text(json.dumps(edited, sort_keys=True) + "\n" + first[1] + "\n", encoding="utf-8")
    assert confirmations.problems(tmp_path, None)
    assert phased.evidence_settings(tmp_path).get("phased_record_gap")


def test_the_record_says_confirmed_but_a_verdict_line_does_not_is_a_gap(tmp_path: Path) -> None:
    _code(tmp_path)
    _freeze_and_confirm(tmp_path, None)
    record = phased.load(tmp_path)
    record.update(confirm_result_sha256="x", results_seen_in_confirm=1, confirm_executions=1)  # hand-edited
    phased._save(tmp_path, record)
    assert phased.status(phased.load(tmp_path)) == phased.CONFIRMED
    gap = phased.evidence_settings(tmp_path).get("phased_record_gap") or ""
    assert "did not hold" in gap
    assert any("did not hold" in g for g in _ready_gaps(tmp_path))


def test_the_engine_names_every_verdict_line_in_the_decision_trace(tmp_path: Path) -> None:
    from tests.test_phased import _config
    from core.engine import Engine

    engine = Engine(_config(tmp_path / "out", phased_on=True))
    root = engine.quest_root
    _code(root)
    engine._audit("quest_started")
    _freeze_and_confirm(root, {"a": 2})
    engine._phased_log([])
    events = audit_log.read(engine.audit.path)
    named = [e["line_sha256"] for e in events if e.get("kind") == confirmations.EVENT]
    assert named == confirmations.line_hashes(root)
    engine._phased_log([])
    assert len([e for e in audit_log.read(engine.audit.path) if e.get("kind") == confirmations.EVENT]) == 1, "once"
    assert audit_log.verify(engine.audit.path).ok
    # A line the trace named, removed afterwards, is found.
    confirmations.path(root).write_text("", encoding="utf-8")
    assert confirmations.problems(root, audit_log.read(engine.audit.path))
    assert phased.evidence_settings(root).get("phased_record_gap")


# ---- the engine: improve, review -> implement, and the search, after the freeze --------------------------------------


def test_after_the_freeze_improve_does_not_run_and_a_changed_version_needs_its_own_confirm_run(tmp_path: Path) -> None:
    from tests.test_phased import _config
    from core.engine import Engine

    engine = Engine(_config(tmp_path / "out", phased_on=True).model_copy(
        update={"engine": _config(tmp_path / "out", phased_on=True).engine.model_copy(update={"improve_rounds": 2})}))
    root = engine.quest_root
    _code(root, _TRIAL)  # takes the seed it is given, so new seeds can confirm it
    phased.prepare(root, engine.quest_id)
    gate = {"evidence_assessment": {"route": "write"}, "result_json": {"a": 1}, "exec_result": {"returncode": 0}}
    phased.seed_env(root, _env(0, 0))
    assert engine._route_after_evidence_gate(gate) == "confirm"
    assert "confirm run" in (engine._improve_skip({}) or ""), "the improve loop never changes a frozen version"
    phased.seed_env(root, _env(0, 0))
    assert engine._route_after_evidence_gate({**gate, "result_json": {"a": 2}}) == "write"
    assert phased.status(phased.load(root)) == phased.CONFIRMED
    assert "confirm run" in (engine._improve_skip({}) or "")
    # The review sends the experiment back (review -> implement): the code is written again, then it runs.
    _code(root, _TRIAL.replace("% 7", "% 11"))  # the review asked for a change
    engine._phased_before_first_run({})
    record = phased.load(root)
    assert record["stage"] == phased.EXPLORE and phased.candidate(record) == 2
    assert engine._improve_skip({}) != "this is the confirm run, which runs the frozen design as it is"
    # Its accepted result is not written up as confirmed: it goes to a confirm run of its own.
    phased.seed_env(root, _env(0, 0))
    assert engine._route_after_evidence_gate({**gate, "result_json": {"a": 3}}) == "confirm"
    assert phased.load(root)["stage"] == phased.CONFIRM and phased.candidate(phased.load(root)) == 2
    run_log = (root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "version 2 of the study" in run_log and "confirmed on its own" in run_log


_TRIAL = "def run_trial(cell, trial_id, seed):\n    return {'y': float(seed % 7)}\n"


class _NoRun:
    """An executor that must not be called: a frozen study's search is not run again."""

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, *a, **k):  # noqa: ANN002, ANN003
        self.calls += 1
        raise AssertionError("the search ran after the freeze")


def test_a_frozen_search_is_not_run_again_on_the_confirm_seeds(tmp_path: Path) -> None:
    from core import optimise

    root = tmp_path
    _code(root)
    protocol = {"optimisation": {
        "objective": {"quantity": "y", "direction": "minimise", "unit": "J"},
        "design_variables": [{"name": "x", "low": 0.0, "high": 1.0, "kind": "continuous"}],
        "baseline": {"values": {"x": 0.0}, "source": "the design in use"},
        "evaluation_budget": {"starts": 1, "per_start": 4}, "search_method": "bounded_local"}}
    block, why = optimise._plan.normalize(protocol["optimisation"])
    assert block is not None, why
    executor = _NoRun()
    # No search record from exploration: nothing is searched; the confirm run gets no result.
    run = asyncio.run(optimise.run_search(executor, "python", root, "code/simulate.py", protocol, base_seed=777,
                                          timeout_s=5, frozen=True))
    assert run.problems and "searching again" in run.problems[0] and executor.calls == 0
    # Exploration's search record for this simulation and plan is kept, whatever the seed.
    study_key = optimise._study_key(root, root / "code" / "simulate.py", block, "run_cell", None)
    ledger = json.dumps({"event": "evaluation", "status": "ok", "design": {"x": 0.5}, "objective": 1.0}) + "\n"
    best = json.dumps({"best": {"design": {"x": 0.5}, "objective": 1.0}}) + "\n"
    optimise._save(root, "exploration-key", ledger, best, "", False, study_key)
    run = asyncio.run(optimise.run_search(executor, "python", root, "code/simulate.py", protocol, base_seed=777,
                                          timeout_s=5, frozen=True))
    assert run.reused and not run.problems and executor.calls == 0
    assert run.record["best"]["design"] == {"x": 0.5}


# ---- what the review of this change found ----------------------------------------------------------------------------


def test_line_endings_and_comments_are_not_a_new_version(tmp_path: Path) -> None:
    _code(tmp_path)
    _freeze_and_confirm(tmp_path, {"a": 2})
    text = (tmp_path / "code" / "simulate.py").read_text(encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_bytes(("# a note\n" + text).replace("\n", "\r\n").encode("utf-8"))
    assert phased.new_candidate(tmp_path) == [], "a checkout's line endings or a comment would buy a fresh confirm run"


def test_a_change_before_the_confirm_run_began_is_not_a_failed_confirmation(tmp_path: Path) -> None:
    data = tmp_path / "inputs" / "data" / "d.csv"
    data.parent.mkdir(parents=True)
    data.write_bytes(("x,y\n" + "".join(f"{i},{i * i}\n" for i in range(60))).encode())
    _code(tmp_path)
    phased.prepare(tmp_path, "q1")
    record = phased.load(tmp_path)
    record["split_decided"] = True
    phased._save(tmp_path, record)
    explore_part = data.read_bytes()
    phased.seed_env(tmp_path, _env(0, 0))
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    assert data.read_bytes() != explore_part, "the held-back part is in place for the confirm run"
    _code(tmp_path, "def run_cell(cell):\n    return {'y': 8.0}\n")  # edited before the confirm run ran
    lines = phased.new_candidate(tmp_path, quest_id="q1")
    record = phased.load(tmp_path)
    assert record["stage"] == phased.EXPLORE and phased.candidate(record) == 1 and not record.get("not_confirmable")
    assert any("before the confirm run began" in x for x in lines)
    assert confirmations.read(tmp_path) == [], "nothing ran on the confirm data: there is no verdict"
    assert data.read_bytes() == explore_part, "exploration's part is back, never the held-back rows"


def test_held_back_rows_given_to_the_code_are_read_even_before_the_confirm_run_is_counted(tmp_path: Path) -> None:
    data = tmp_path / "inputs" / "data" / "d.csv"
    data.parent.mkdir(parents=True)
    data.write_bytes(("x,y\n" + "".join(f"{i},{i * i}\n" for i in range(60))).encode())
    _code(tmp_path)
    phased.prepare(tmp_path, "q1")
    record = phased.load(tmp_path)
    record["split_decided"] = True
    phased._save(tmp_path, record)
    phased.seed_env(tmp_path, _env(0, 0))
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    phased.note_confirm_handed(tmp_path)  # the step began: a known-answer check may read the held-back part
    _code(tmp_path, "def run_cell(cell):\n    return {'y': 12.0}\n")  # its repair changed the code
    phased.new_candidate(tmp_path, quest_id="q1")
    record = phased.load(tmp_path)
    assert phased.candidate(record) == 2 and record.get("held_back_used"), "the rows were given to the code"
    assert [e["verdict"] for e in confirmations.read(tmp_path)] == ["confirm_failed"]


def test_the_search_and_the_version_agree_on_what_is_unchanged(tmp_path: Path) -> None:
    from core import optimise

    _code(tmp_path)
    block = {"evaluation_budget": {"starts": 1, "per_start": 2}}
    before = optimise._study_key(tmp_path, tmp_path / "code" / "simulate.py", block, "run_cell", None)
    fp = phased.fingerprint(tmp_path)["code_sha256"]
    text = (tmp_path / "code" / "simulate.py").read_text(encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_bytes(("# note\n" + text).replace("\n", "\r\n").encode("utf-8"))
    assert phased.fingerprint(tmp_path)["code_sha256"] == fp
    assert optimise._study_key(tmp_path, tmp_path / "code" / "simulate.py", block, "run_cell", None) == before


def test_code_changed_during_the_confirm_run_confirms_nothing(tmp_path: Path) -> None:
    _code(tmp_path)
    phased.prepare(tmp_path, "q1")
    phased.seed_env(tmp_path, _env(0, 0))
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    phased.seed_env(tmp_path, _env(0, 0))
    _code(tmp_path, "def run_cell(cell):\n    return {'y': 6.0}\n")  # a repair inside the confirm run's own step
    record, lines = phased.record_confirm(tmp_path, {"a": 2})
    assert phased.status(record) == "not_confirmable" and any("changed during the confirm run" in x for x in lines)
    assert [e["verdict"] for e in confirmations.read(tmp_path)] == ["not_confirmable"]


def test_a_line_edited_by_hand_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    _code(tmp_path)
    _freeze_and_confirm(tmp_path, {"a": 2})
    with confirmations.path(tmp_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"candidate": "x", "verdict": "confirmed"}) + "\n")
    assert confirmations.candidates(tmp_path) == [{"candidate": 1, "verdict": "confirmed", "lines": 1}]
    assert confirmations.problems(tmp_path, None)
    assert phased.evidence_settings(tmp_path).get("phased_record_gap")


def test_the_confirm_run_of_a_frozen_search_checks_the_kept_design_again_on_its_new_seeds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import optimise, optimum_check
    from core.execution import ExecutionResult

    root = tmp_path
    _code(root)
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    protocol = {"optimisation": {
        "objective": {"quantity": "y", "direction": "minimise", "unit": "J"},
        "design_variables": [{"name": "x", "low": 0.0, "high": 1.0, "kind": "continuous"}],
        "baseline": {"values": {"x": 0.0}, "source": "the design in use"},
        "evaluation_budget": {"starts": 1, "per_start": 4}, "search_method": "bounded_local"}}
    block, _ = optimise._plan.normalize(protocol["optimisation"])
    study_key = optimise._study_key(root, root / "code" / "simulate.py", block, "run_cell", None)
    ledger = json.dumps({"event": "evaluation", "status": "ok", "design": {"x": 0.5}, "objective": 1.0}) + "\n"
    optimise._save(root, "exploration-key", ledger, json.dumps({"best": {"design": {"x": 0.5}}}) + "\n", "", False,
                   study_key)
    seen: dict = {}

    async def fake_check(*a, **k):  # noqa: ANN002, ANN003
        seen.update(k)
        return {"verdict": "verified", "finished": True}, "{}", "k"

    monkeypatch.setattr(optimum_check, "run_check", fake_check)
    monkeypatch.setattr(optimise, "attach_check", lambda *a, **k: None)

    class _Analysis:
        async def execute(self, cmd, **k):  # noqa: ANN001, ANN003
            return ExecutionResult(returncode=0, stdout="RESULT_JSON: {}", stderr="", duration_s=0.0)

    runner = optimise.OptimisationRunner(_Analysis(), quest_root=root, protocol=protocol,
                                         simulate=root / "code" / "simulate.py",
                                         analysis=root / "code" / "experiment.py", frozen=lambda: True)
    asyncio.run(runner.execute(["python", str(root / "code" / "experiment.py")], cwd=root, timeout_s=5,
                               env={"FI_REPLICATE_SEED": "424242"}))
    assert seen.get("fresh") is True and seen.get("base_seed") == 424242, seen
