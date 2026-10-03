"""A data quest's confirm run, and what a held-back confirm may claim (``core/phased.py``, ``core/phased_data.py``,
``core/phased_isolation.py``).

A no-simulation quest that analyses one table holds part of it back before exploration first reads it, decided by which
rows belong together (the plan's ``protocol.split``, the answer in plan.md, or the table's columns; never rows one by
one unless they are said to be independent), reads the held-back rows once more with the frozen design when exploration
ends, records that run as the confirm run and sets its numbers beside exploration's. Without a container the kept files
are encrypted with a key only the running FI holds, and the quest's code is read for paths out of the quest folder;
held back is not called unseen otherwise.

No real model is called.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import evidence as _evidence
from core import phased, phased_data, phased_isolation
from core.config import Config

QUEST = "q-data"


def _table(path: Path, header: str, rows: list[str]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (header + "\n" + "".join(r + "\n" for r in rows)).encode()
    path.write_bytes(data)
    return data


def _visits(path: Path, subjects: int = 20, visits: int = 4) -> bytes:
    """Repeated measures: several rows per subject (the case a row split gets wrong)."""
    rows = [f"s{s:03d},{v},{s * 10 + v}.5" for s in range(subjects) for v in range(visits)]
    return _table(path, "subject_id,visit,score", rows)


def _independent(path: Path, n: int = 80) -> bytes:
    return _table(path, "x,y", [f"{i},{i * i}" for i in range(n)])


def _series(path: Path, months: int = 48) -> bytes:
    rows = [f"{2020 + m // 12}-{m % 12 + 1:02d},{100 + m}" for m in range(months)]
    return _table(path, "month,sales", rows)


def _parts(root: Path, rel: str = "data/d.csv") -> tuple[list[str], list[str]]:
    """Exploration's rows (on disk) and the held-back rows (from the kept part, read with ``key``)."""
    record = phased.load(root)
    (info,) = record["files"]
    explore = (root / rel).read_text().splitlines()[1:]
    return explore, info


def _held_rows(root: Path, info: dict[str, Any], key: bytes | None) -> list[str]:
    raw = phased._read_kept(phased._kept(root, info, "held_back"), key)
    assert raw is not None
    return raw.decode().splitlines()[1:]


def _cells(rows: list[str], col: int) -> set[str]:
    return {r.split(",")[col] for r in rows}


# ---- the rules ------------------------------------------------------------------------------------------------------


def test_repeated_measures_are_held_back_by_subject_never_row_by_row(tmp_path: Path) -> None:
    _visits(tmp_path / "data" / "d.csv")
    key = phased_isolation.new_key()
    phased.prepare(tmp_path, QUEST, data_quest=True, key=key)
    record, lines, question = phased.decide_split(tmp_path, QUEST, key=key)
    assert question == ""
    explore, info = _parts(tmp_path)
    held = _held_rows(tmp_path, info, key)
    assert info["rule"]["strategy"] == "group" and info["rule"]["unit"] == "subject_id"
    assert info["rule"]["source"] == "columns"
    assert _cells(explore, 0).isdisjoint(_cells(held, 0)), "a subject is on one side only"
    assert info["manifest"]["overlap"] == 0 and info["manifest"]["units_held_back"] >= phased_data.MIN_HELD_UNITS
    assert record["split_decided"] is True
    assert any("whole subject_id values" in line for line in lines)


def test_the_split_is_the_same_every_time(tmp_path: Path) -> None:
    for name in ("a", "b"):
        _visits(tmp_path / name / "data" / "d.csv")
        phased.prepare(tmp_path / name, QUEST, data_quest=True)
        phased.decide_split(tmp_path / name, QUEST)
    assert (tmp_path / "a" / "data" / "d.csv").read_bytes() == (tmp_path / "b" / "data" / "d.csv").read_bytes()
    # Applied again at a later start (after the whole file was put back), it gives the same parts.
    first = (tmp_path / "a" / "data" / "d.csv").read_bytes()
    phased.restore_inputs(tmp_path / "a")
    phased.prepare(tmp_path / "a", QUEST, data_quest=True)
    assert (tmp_path / "a" / "data" / "d.csv").read_bytes() == first


def test_a_time_series_holds_back_its_latest_period_with_an_embargo(tmp_path: Path) -> None:
    _series(tmp_path / "data" / "d.csv")
    phased.prepare(tmp_path, QUEST, data_quest=True)
    _record, _lines, question = phased.decide_split(
        tmp_path, QUEST, declared={"strategy": "time", "time_column": "month", "embargo": 2})
    assert question == ""
    explore, info = _parts(tmp_path)
    held = _held_rows(tmp_path, info, None)
    latest_explored = max(r.split(",")[0] for r in explore)
    assert min(r.split(",")[0] for r in held) > latest_explored, "only later periods are held back"
    assert info["manifest"]["embargo_rows"] == 2 and info["manifest"]["rows_left_out"] == 2
    assert info["manifest"]["overlap"] == 0
    # Inferred from the column name alone, too (a single time column, no unit column).
    _series(tmp_path / "b" / "data" / "d.csv")
    phased.prepare(tmp_path / "b", QUEST, data_quest=True)
    phased.decide_split(tmp_path / "b", QUEST)
    assert phased.load(tmp_path / "b")["files"][0]["rule"]["strategy"] == "time"


def test_a_second_reading_keeps_the_embargo(tmp_path: Path) -> None:
    _series(tmp_path / "data" / "d.csv")
    phased.prepare(tmp_path, QUEST, data_quest=True)
    declared = {"strategy": "time", "time_column": "month", "embargo": 2}
    phased.data_quest_gate(tmp_path, QUEST, declared=declared)
    first = (tmp_path / "data" / "d.csv").read_bytes()
    phased.data_quest_gate(tmp_path, QUEST, declared=declared)  # the next reading (a redesign in exploration)
    assert (tmp_path / "data" / "d.csv").read_bytes() == first
    info = phased.load(tmp_path)["files"][0]
    assert info["manifest"]["rows_left_out"] == 2 and info["manifest"]["embargo_rows"] == 2


def test_later_months_added_after_exploration_read_the_data_never_reach_exploration(tmp_path: Path) -> None:
    path = tmp_path / "data" / "d.csv"
    _series(path)
    phased.prepare(tmp_path, QUEST, data_quest=True)
    declared = {"strategy": "time", "time_column": "month", "embargo": 1}
    phased.data_quest_gate(tmp_path, QUEST, declared=declared)
    explored = set((tmp_path / "data" / "d.csv").read_text().splitlines()[1:])
    latest = max(r.split(",")[0] for r in explored)
    # The person puts the whole file back with two newer months, and the quest reads the data again.
    phased.restore_inputs(tmp_path)
    _table(path, "month,sales", [*[f"{2020 + m // 12}-{m % 12 + 1:02d},{100 + m}" for m in range(48)],
                                 "2024-01,999", "2024-02,998"])
    phased.prepare(tmp_path, QUEST, data_quest=True)
    phased.data_quest_gate(tmp_path, QUEST, declared=declared)
    now = (tmp_path / "data" / "d.csv").read_text().splitlines()[1:]
    assert max(r.split(",")[0] for r in now) == latest, "exploration still ends where it ended"


def test_a_column_that_numbers_the_rows_is_not_taken_for_a_unit() -> None:
    """``id`` is unique on every row: it says nothing about which rows belong to one patient."""
    rows = [f"{i},p{i // 3},{i}" for i in range(60)]
    decision = phased_data.decide("t.csv", "id,patient_name,score", rows, ",", QUEST)
    assert decision.ask and "numbers the rows" in decision.why and not decision.held


def test_a_header_with_a_byte_order_mark_is_read_by_its_name() -> None:
    rows = [f"{2020 + m // 12}-{m % 12 + 1:02d},{m}" for m in range(48)]
    decision = phased_data.decide("t.csv", "﻿month,sales", rows, ",", QUEST)
    assert not decision.ask and decision.rule["strategy"] == "time" and decision.rule["time_column"] == "month"


def test_a_table_outside_data_is_not_held_back_for_a_data_quest(tmp_path: Path) -> None:
    """A data quest's numbers come from what its data-reading step reads (data/): a table in inputs/data/ is not it."""
    original = _visits(tmp_path / "inputs" / "data" / "d.csv")
    (tmp_path / "data" / "literature").mkdir(parents=True)
    (tmp_path / "data" / "literature" / "paper.md").write_text("# a paper\n")
    phased.prepare(tmp_path, QUEST, data_quest=True)
    lines, question = phased.data_quest_gate(tmp_path, QUEST)
    assert question == "" and phased.status(phased.load(tmp_path)) == phased.NOT_APPLICABLE
    assert any("put it in data/" in line for line in lines)
    assert (tmp_path / "inputs" / "data" / "d.csv").read_bytes() == original


def test_a_confirm_reading_cut_short_is_not_a_second_look(tmp_path: Path) -> None:
    _visits(tmp_path / "data" / "d.csv")
    phased.prepare(tmp_path, QUEST, data_quest=True)
    phased.data_quest_gate(tmp_path, QUEST)
    phased.enter_confirm(tmp_path, explore_result={"m": 1.0}, frozen_sha256=None, stride=1, replicates=1,
                         explore_runs=1, isolation=(phased_isolation.ENCRYPTED, ""))
    phased.data_quest_gate(tmp_path, QUEST)  # the confirm reading starts, and the model's connection fails
    phased.data_quest_gate(tmp_path, QUEST)  # resumed: the same reading again
    assert phased.load(tmp_path)["confirm_executions"] == 1
    phased.note_confirm_result(tmp_path, {"m": 1.1})
    record, _lines = phased.record_confirm(tmp_path, {"m": 1.1})
    assert phased.status(record) == phased.CONFIRMED


def test_a_confirm_reading_that_found_nothing_still_counts(tmp_path: Path) -> None:
    _visits(tmp_path / "data" / "d.csv")
    phased.prepare(tmp_path, QUEST, data_quest=True)
    phased.data_quest_gate(tmp_path, QUEST)
    phased.enter_confirm(tmp_path, explore_result={"m": 1.0}, frozen_sha256=None, stride=1, replicates=1,
                         explore_runs=1, isolation=(phased_isolation.ENCRYPTED, ""))
    phased.data_quest_gate(tmp_path, QUEST)
    phased.note_confirm_result(tmp_path, {})  # the model's answer could not be read: the reading still happened
    phased.data_quest_gate(tmp_path, QUEST)  # a second look
    assert phased.load(tmp_path)["confirm_executions"] == 2


def test_a_waiting_background_job_without_a_container_gets_the_whole_files_back(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    original = _visits(engine.quest_root / "data" / "d.csv")
    engine._phased_prepare()
    phased.data_quest_gate(engine.quest_root, engine.quest_id, key=engine._phased_key())
    engine._phased_job_pending = True
    engine._phased_restore_inputs()
    assert (engine.quest_root / "data" / "d.csv").read_bytes() == original
    assert phased.status(phased.load(engine.quest_root)) == "compromised"
    # After the confirm result (files whole already) a waiting job changes nothing about the result.
    other = _engine(tmp_path / "done")
    _visits(other.quest_root / "data" / "d.csv")
    other._phased_prepare()
    key = other._phased_key()
    phased.data_quest_gate(other.quest_root, other.quest_id, key=key)
    phased.enter_confirm(other.quest_root, explore_result={"m": 1.0}, frozen_sha256=None, stride=1, replicates=1,
                         explore_runs=1, key=key, isolation=(phased_isolation.ENCRYPTED, ""))
    phased.data_quest_gate(other.quest_root, other.quest_id, key=key)
    phased.note_confirm_result(other.quest_root, {"m": 1.1})
    phased.record_confirm(other.quest_root, {"m": 1.1}, key=key)
    other._phased_job_pending = True
    other._phased_restore_inputs()
    assert phased.status(phased.load(other.quest_root)) == phased.CONFIRMED


def test_a_switch_to_a_container_rewrites_the_kept_files_plain_and_clears_stray_copies(tmp_path: Path) -> None:
    root = tmp_path / QUEST
    original = _visits(root / "data" / "d.csv")
    key = phased_isolation.new_key()
    phased.prepare(root, QUEST, data_quest=True, key=key)
    phased.decide_split(root, QUEST, key=key)
    phased.restore_inputs(root, key)
    stray = phased.store_dir(root) / "tmp" / ".d.csv.fi-tmp"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(original)
    phased.prepare(root, QUEST, data_quest=True, mode="docker")  # no key: a container now
    info = phased.load(root)["files"][0]
    assert phased._read_kept(phased._kept(root, info, "original"), None) == original
    assert not stray.exists()


def test_without_the_encryption_package_rows_are_kept_plain_and_not_called_isolated(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _visits(engine.quest_root / "data" / "d.csv")

    def missing() -> bytes:
        raise ImportError("No module named 'cryptography'")

    monkeypatch.setattr(phased_isolation, "new_key", missing)
    engine._phased_prepare()
    assert engine._phased_key() is None and engine._phased_mode() == "plain"
    assert phased.load(engine.quest_root)["starts"] == ["plain"]
    state, why = engine._phased_isolation()
    assert state == phased_isolation.UNVERIFIED and "cryptography" in why


def test_a_record_from_before_isolation_was_noted_is_not_counted_as_isolated(tmp_path: Path) -> None:
    _visits(tmp_path / "data" / "d.csv")
    phased.prepare(tmp_path, QUEST, data_quest=True)  # no mode: as a record written before starts were noted
    record = phased.load(tmp_path)
    record.pop("starts", None)
    phased._save(tmp_path, record)
    phased.prepare(tmp_path, QUEST, data_quest=True, mode="encrypted")
    assert phased.load(tmp_path)["starts"] == ["unrecorded", "encrypted"]


def test_a_time_column_that_cannot_be_ordered_never_falls_back_to_a_random_split() -> None:
    rows = [f"day {i},{i}" for i in range(60)]
    decision = phased_data.decide("t.csv", "date,v", rows, ",", QUEST)
    assert decision.why and "cannot all be put in order" in decision.why and not decision.held


def test_rows_are_split_one_by_one_only_when_said_to_be_independent(tmp_path: Path) -> None:
    _independent(tmp_path / "data" / "d.csv")
    phased.prepare(tmp_path, QUEST, data_quest=True)
    # Nothing says which rows belong together: a research quest asks, once, before anything reads the data.
    _record, lines, question = phased.decide_split(tmp_path, QUEST, research=True)
    assert question and "Rows that belong together" in question and lines == []
    assert not phased.load(tmp_path).get("split_decided")
    # The answer in plan.md decides it.
    plan = "## How the result will be confirmed\n\nRows that belong together: independent\n"
    assert phased_data.plan_answer(plan) == "independent"
    _record, _lines, question = phased.decide_split(tmp_path, QUEST, answer=phased_data.plan_answer(plan), research=True)
    assert question == ""
    info = phased.load(tmp_path)["files"][0]
    assert info["rule"] == {"strategy": "row", "source": "answer"}


def test_outside_research_an_undecidable_table_holds_nothing_back_and_says_why(tmp_path: Path) -> None:
    original = _independent(tmp_path / "data" / "d.csv")
    phased.prepare(tmp_path, QUEST, data_quest=True)
    record, lines, question = phased.decide_split(tmp_path, QUEST, research=False)
    assert question == "" and record["strategy"] == phased.FRESH_SEEDS
    assert (tmp_path / "data" / "d.csv").read_bytes() == original, "the whole file is back"
    assert any("could put one subject in both parts" in line for line in lines)


def test_the_plan_line_asks_the_question_in_plain_words(tmp_path: Path) -> None:
    _independent(tmp_path / "data" / "d.csv")
    phased.prepare(tmp_path, QUEST, data_quest=True)
    note = phased.split_preview(tmp_path, QUEST)
    assert phased_data.QUESTION_LINE in note
    assert phased_data.plan_answer(note) is None, "an unanswered line is not an answer"
    text = "\n".join(phased.plan_lines(phased.load(tmp_path), on=True, research=True, runs_code=False,
                                       data_quest=True, split_note=note))
    assert "Rows that belong together: ?" in text and "encrypted" in text


# ---- the data quest through the engine ------------------------------------------------------------------------------


def _engine(root: Path, **engine: Any):  # noqa: ANN202
    from core.engine import Engine

    cfg = Config.model_validate({
        "topic": "does the score rise with visits", "title": "data-confirm", "rigor_profile": "research",
        "provider": {"name": "openai", "node_models": {"review_panel.statistician": "m-other"}},
        "engine": {"max_iterations": 1, "no_simulation": True, "web_derived_plots": False, **engine},
        "knowledge": {"enabled": False}, "output": {"output_dir": str(root)},
    })
    eng = Engine(cfg)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    return eng


def _read_by_data_load(engine, monkeypatch: pytest.MonkeyPatch, design: dict[str, Any]) -> tuple[dict[str, Any], str]:  # noqa: ANN001
    seen: dict[str, str] = {}

    async def fake_chat(prompt, *, node="", **kw):  # noqa: ANN001, ANN003
        seen["prompt"] = prompt
        n = prompt.count("\ns0") + prompt.count("\ns1")  # rows of the table the model was shown
        return json.dumps({"summary": "ok", "measurements": {"mean_score": 100.0 + n, "slope": 1.0}})

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = asyncio.run(engine._node_data_load({"topic": "t", "design": design}))
    return out, seen.get("prompt", "")


def test_exploration_never_reads_a_held_back_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _visits(engine.quest_root / "data" / "d.csv")
    engine._phased_prepare()
    key = engine._phased_key()
    _out, prompt = _read_by_data_load(engine, monkeypatch, {"hypothesis": "h"})
    info = phased.load(engine.quest_root)["files"][0]
    held = _held_rows(engine.quest_root, info, key)
    assert held and not any(row in prompt for row in held), "the data-reading step saw only exploration's rows"
    # The analysis step and the figures step read the same part.
    text = engine._gather_collected_text({"literature": []})
    assert not any(row in text for row in held)
    # The kept files are encrypted: no held-back row is readable in the store.
    for path in phased.store_dir(engine.quest_root).rglob("*"):
        if path.is_file():
            assert path.read_bytes().startswith(phased_isolation.MAGIC)
            assert not any(row.encode() in path.read_bytes() for row in held)
    assert phased.load(engine.quest_root)["explore_runs"] == 1


def test_the_confirm_run_reads_only_the_held_back_rows_and_clears_the_revision_gap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    original = _visits(engine.quest_root / "data" / "d.csv")
    engine._phased_prepare()
    key = engine._phased_key()
    design = {"hypothesis": "h"}
    out, _ = _read_by_data_load(engine, monkeypatch, design)
    explore_result = out["result_json"]
    # The design was revised after the data had been analysed (the gap a data quest could never clear before).
    (engine.quest_root / "needs").mkdir(exist_ok=True)
    (engine.quest_root / "needs" / "DESIGN_HISTORY.json").write_text(json.dumps([
        {"revision": 0, "post_hoc": False},
        {"revision": 1, "post_hoc": True, "after_results": True, "reason": "review verdict=revise"}]))
    state = {"no_simulation_resolved": True, "result_json": explore_result, "design": design,
             "exec_result": {"returncode": 0}}
    assert engine._phased_route(state, "write") == "confirm_data"
    record = phased.load(engine.quest_root)
    assert record["stage"] == phased.CONFIRM and record["isolation"] == phased_isolation.ENCRYPTED
    info = record["files"][0]
    held = _held_rows(engine.quest_root, info, key)
    confirm_out, prompt = _read_by_data_load(engine, monkeypatch, design)
    assert all(row in prompt for row in held), "the confirm run reads the held-back rows"
    explored = (b"\n".join(original.splitlines()[1:])).decode().splitlines()
    assert not any(row in prompt for row in explored if row not in held)
    state = {**state, "result_json": confirm_out["result_json"]}
    assert engine._phased_route(state, "write") == "write"
    record = phased.load(engine.quest_root)
    assert phased.status(record) == phased.CONFIRMED and record["confirm_executions"] == 1
    assert (engine.quest_root / "data" / "d.csv").read_bytes() == original, "the whole file is back"
    settings = phased.evidence_settings(engine.quest_root)
    assert settings["phased_isolation"] == phased_isolation.ENCRYPTED
    ready = _ready(engine.quest_root, {"no_simulation_resolved": True}, settings)
    assert not any("the design was revised" in g for g in ready), ready
    assert not any("out of exploration's reach" in g or "one by one" in g for g in ready), ready
    # A revision after the confirm run is not confirmed by it: the gap comes back.
    history = json.loads((engine.quest_root / "needs" / "DESIGN_HISTORY.json").read_text())
    history.append({"revision": 2, "post_hoc": True, "after_results": True, "reason": "review verdict=revise"})
    (engine.quest_root / "needs" / "DESIGN_HISTORY.json").write_text(json.dumps(history))
    ready = _ready(engine.quest_root, {"no_simulation_resolved": True}, settings)
    assert any("the design was revised" in g for g in ready), ready


def _ready(root: Path, state: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    return next(level["gaps"] for level in _evidence.assess(
        root, {"result_json": {"a": 1}, **state}, settings=settings)["ladder"] if level["level"] == "publication_ready")


def test_a_disagreement_is_reported_plainly_not_hidden(tmp_path: Path) -> None:
    compared = phased_data.compare({"measurements": {"mean": 10.0, "slope": 2.0, "n_rows": 50}},
                                   {"measurements": {"mean": 10.5, "slope": -1.0, "n_rows": 20}})
    assert compared["compared"] == 2 and [d["name"] for d in compared["differs"]] == ["measurements.slope"]
    (line,) = phased_data.compare_lines(compared)
    assert "measurements.slope: exploration 2, confirm -1" in line
    # In the record, the writer's note (names only, never exploration's values) and the evidence record.
    root = tmp_path
    _visits(root / "data" / "d.csv")
    phased.prepare(root, QUEST, data_quest=True)
    phased.decide_split(root, QUEST)
    phased.data_quest_gate(root, QUEST)
    phased.enter_confirm(root, explore_result={"measurements": {"slope": 2.0}}, frozen_sha256=None, stride=1,
                         replicates=1, explore_runs=1, isolation=(phased_isolation.ENCRYPTED, ""))
    phased.data_quest_gate(root, QUEST)
    phased.note_confirm_result(root, {"measurements": {"slope": -1.0}})
    record, lines = phased.record_confirm(root, {"measurements": {"slope": -1.0}})
    assert any("differ from exploration's" in x for x in lines)
    note = phased.write_note(root)
    assert "measurements.slope" in note and "2.0" not in note
    assert "differ on the held-back rows" in phased.summary(record)
    settings = phased.evidence_settings(root)
    assert settings["phased_differs"] == "measurements.slope"
    rec = _evidence.assess(root, {"result_json": {"a": 1}}, settings=settings)
    assert rec["phased"]["confirm_differs"] == "measurements.slope"
    assert rec["phased"]["isolation"] == phased_isolation.ENCRYPTED


def test_too_few_rows_or_only_gathered_material_say_so_and_keep_the_gap(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "few")
    _visits(engine.quest_root / "data" / "d.csv", subjects=5, visits=4)  # 20 rows: fewer than 40
    engine._phased_prepare()
    lines, question = phased.data_quest_gate(engine.quest_root, engine.quest_id, key=engine._phased_key())
    assert question == ""
    assert phased.status(phased.load(engine.quest_root)) == phased.NOT_APPLICABLE
    assert any(line.startswith("no part of the data can be held back for a confirm run") for line in lines)
    (engine.quest_root / "needs").mkdir(exist_ok=True)
    (engine.quest_root / "needs" / "DESIGN_HISTORY.json").write_text(json.dumps([
        {"revision": 0}, {"revision": 1, "post_hoc": True, "after_results": True, "reason": "review verdict=revise"}]))
    ready = _ready(engine.quest_root, {"no_simulation_resolved": True}, phased.evidence_settings(engine.quest_root))
    assert any("the design was revised" in g for g in ready), "today's strict gap stays"
    # Only what the quest gathered (literature, collected pages): nothing of the person's to hold back.
    gathered = _engine(tmp_path / "gathered")
    (gathered.quest_root / "data" / "auto_collected").mkdir(parents=True)
    (gathered.quest_root / "data" / "auto_collected" / "page.md").write_text("# a page\n")
    gathered._phased_prepare()
    lines, _q = phased.data_quest_gate(gathered.quest_root, gathered.quest_id, key=gathered._phased_key())
    assert any(phased.GATHERED_ONLY in line for line in lines)


def test_a_survey_holds_nothing_back(tmp_path: Path) -> None:
    engine = _engine(tmp_path, survey_mode=True)
    data = _visits(engine.quest_root / "data" / "d.csv")
    engine._phased_prepare()
    assert phased.status(phased.load(engine.quest_root)) == phased.NOT_APPLICABLE
    assert (engine.quest_root / "data" / "d.csv").read_bytes() == data


def test_a_research_data_quest_that_cannot_tell_stops_before_reading(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _independent(engine.quest_root / "data" / "d.csv")
    engine._phased_prepare()
    asked: list[str] = []
    monkeypatch.setattr(engine, "_phased_ask_split", lambda q: (asked.append(q), (_ for _ in ()).throw(SystemExit)))

    async def no_chat(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("the data was read before the question was answered")

    monkeypatch.setattr(engine, "_chat", no_chat)
    with pytest.raises(SystemExit):
        asyncio.run(engine._node_data_load({"topic": "t", "design": {}}))
    assert asked and "Rows that belong together" in asked[0]


# ---- isolation ------------------------------------------------------------------------------------------------------


def _probe(quest_root: Path, code: str) -> str:
    """Run ``code`` as a quest script would (venv: a plain subprocess, the quest folder as its working folder)."""
    done = subprocess.run([sys.executable, "-c", code], cwd=quest_root, capture_output=True, text=True, timeout=60)
    return done.stdout + done.stderr


def test_hostile_reads_without_a_container_find_only_encrypted_bytes(tmp_path: Path) -> None:
    root = tmp_path / "out" / QUEST
    _visits(root / "data" / "d.csv")
    key = phased_isolation.new_key()
    phased.prepare(root, QUEST, data_quest=True, key=key, mode="encrypted")
    phased.decide_split(root, QUEST, key=key)
    info = phased.load(root)["files"][0]
    held = _held_rows(root, info, key)
    store = phased.store_dir(root)
    rel = os.path.relpath(store, root)
    # A link to the store made inside the quest folder (a symbolic link where the system allows one).
    link = root / "shortcut"
    try:
        link.symlink_to(store, target_is_directory=True)
        linked = True
    except OSError:
        linked = False
    probes = {
        "relative path": f"import pathlib;print(open(pathlib.Path({rel!r})/'data'/'held_back'/'d.csv','rb').read())",
        "absolute path": f"print(open({str(store / 'data' / 'held_back' / 'd.csv')!r},'rb').read())",
        "parent walk": ("import os,pathlib\nfor d,_s,fs in os.walk(pathlib.Path.cwd().parent):\n"
                        "  [print(open(os.path.join(d,f),'rb').read()) for f in fs if f.endswith('.csv')]"),
        "subprocess": (f"import subprocess,sys;subprocess.run([sys.executable,'-c',"
                       f"\"print(open({(store / 'data' / 'original' / 'd.csv').as_posix()!r},'rb').read())\"])"),
    }
    if linked:
        probes["link"] = "print(open('shortcut/data/held_back/d.csv','rb').read())"
    for name, code in probes.items():
        out = _probe(root, code)
        assert "FI-HELD-BACK-ENCRYPTED" in out or "b'" in out, (name, out[:300])
        assert not any(row in out for row in held), f"{name} read a held-back row"


def test_the_scan_finds_reads_outside_the_quest_folder_and_leaves_ordinary_code_alone(tmp_path: Path) -> None:
    root = tmp_path / QUEST
    code = root / "code"
    code.mkdir(parents=True)
    (code / "experiment.py").write_text(
        '"""Reads ../data in the docstring: not a read."""\nimport pandas as pd\n'
        "df = pd.read_csv('data/d.csv')\nHERE = __import__('pathlib').Path(__file__).resolve().parent\n"
        "QUEST = HERE.parent\nprint(df.mean(), 'm/s')\n", encoding="utf-8")
    from core import code_project

    (code / "run.py").write_text(code_project.RUN_SOURCE, encoding="utf-8")
    assert phased_isolation.scan(root) == []
    bad = {
        "a.py": "open('../_held_back/q/held_back/d.csv')",
        "b.py": "import os\nopen(os.path.join('..', 'x.csv'))",
        "c.py": "import pathlib\nprint(list(pathlib.Path.cwd().parent.glob('*')))",
        "d.py": "import pathlib\nprint(pathlib.Path(__file__).resolve().parent.parent.parent)",
        "e.py": "open('/home/someone/data.csv')",
        "f.py": "import os\nprint(os.path.expanduser('~'))",
        "g.sh": "cat ../_held_back/q/held_back/d.csv\n",
        "h.py": "import os\nopen(os.path.join(os.pardir, 'x.csv'))",
        "i.py": "from pathlib import Path\nprint(list(Path().resolve().parent.iterdir()))",
        "j.py": "import ctypes\nctypes.windll.kernel32.ReadProcessMemory(0, 0, 0, 0, 0)",
    }
    for name, text in bad.items():
        (code / name).write_text(text, encoding="utf-8")
    hits = phased_isolation.scan(root)
    for name in bad:
        assert any(h.startswith(f"code/{name}") for h in hits), (name, hits)


def test_a_scan_hit_keeps_a_held_back_confirm_below_publication_ready(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _visits(engine.quest_root / "data" / "d.csv")
    engine._phased_prepare()
    _read_by_data_load(engine, monkeypatch, {})
    (engine.quest_root / "code").mkdir(exist_ok=True)
    (engine.quest_root / "code" / "web_plots.py").write_text("open('../_held_back/x.csv')\n")
    state, why = engine._phased_isolation()
    assert state == phased_isolation.UNVERIFIED and "code/web_plots.py line 1" in why
    # A record from before isolation was recorded is not shown to be isolated either.
    assert phased.isolation({})[0] == phased_isolation.UNVERIFIED


def test_a_killed_run_cannot_put_back_rows_it_kept_encrypted_and_says_so(tmp_path: Path) -> None:
    root = tmp_path / QUEST
    original = _visits(root / "data" / "d.csv")
    key = phased_isolation.new_key()
    phased.prepare(root, QUEST, data_quest=True, key=key)
    phased.decide_split(root, QUEST, key=key)
    # A normal stop and a new run (a new key): the kept files are written again with the new key.
    phased.restore_inputs(root, key)
    assert (root / "data" / "d.csv").read_bytes() == original
    key2 = phased_isolation.new_key()
    phased.prepare(root, QUEST, data_quest=True, key=key2)
    assert phased._read_kept(phased._kept(root, phased.load(root)["files"][0], "original"), key2) == original
    # Killed: exploration's part stays on disk and the key is gone with the run.
    record, lines = phased.prepare(root, QUEST, data_quest=True, key=phased_isolation.new_key())
    assert record.get("compromised") and any("put your own copy of the whole file there" in x for x in lines)


def test_a_container_is_isolated_only_when_nothing_it_mounts_holds_the_kept_files(tmp_path: Path) -> None:
    from core.execution import DockerExecutor

    root = tmp_path / "out" / QUEST
    root.mkdir(parents=True)
    store = phased.store_dir(root)
    executor = DockerExecutor()
    assert phased_isolation.docker_mounts_clear(executor, root, store) == ""
    mounts = executor._volumes(root.resolve())
    # the quest, and FI's own records read-only over it: nothing else
    assert {h for h, v in mounts.items() if not v["bind"].startswith(("/work/.fi", "/work/needs"))} == {str(root.resolve())}

    class _Mount:  # a skill folder that happens to be the output folder
        name, declared, volume = "s", str(tmp_path / "out"), {"bind": "/fi-skills/s", "mode": "ro"}
        bind_host = str(tmp_path / "out")

        def still_safe(self) -> bool:
            return True

    executor.set_skill_mounts([_Mount()])
    assert "holds the kept files" in phased_isolation.docker_mounts_clear(executor, root, store)


def _docker_ready() -> bool:
    try:
        import docker  # type: ignore[import-not-found]

        return docker.from_env().info().get("OSType") == "linux"
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _docker_ready(), reason="needs a running Docker daemon with Linux containers")
def test_in_a_container_no_hostile_read_reaches_the_held_back_rows(tmp_path: Path) -> None:
    from core.execution import DockerExecutor

    root = tmp_path / "out" / QUEST
    _visits(root / "data" / "d.csv")
    phased.prepare(root, QUEST, data_quest=True, mode="docker")
    phased.decide_split(root, QUEST)
    executor = DockerExecutor()
    asyncio.run(executor.setup(root))
    for code in ("print(open('../_held_back/q-data/data/held_back/d.csv').read())",
                 f"print(open({str(phased.store_dir(root) / 'data' / 'held_back' / 'd.csv')!r}).read())",
                 "import os\nfor d,_s,fs in os.walk('/'):\n  [print(d,f) for f in fs if f=='d.csv']"):
        result = asyncio.run(executor.execute(["python", "-c", code], cwd=root, timeout_s=60))
        assert "held_back" not in result.stdout, result.stdout[:300]


# ---- the whole quest, through the real graph --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_data_quest_is_confirmed_on_its_held_back_rows_end_to_end(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from core import audit_log
    from core.config import EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine
    from tests.test_engine_smoke import _fake_response_for

    reads: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001, ANN003
        prompt = messages[-1]["content"]
        if "collected evidence" in prompt[:400]:
            rows = [line for line in prompt.splitlines() if line[:2] in ("s0", "s1")]
            reads.append(prompt)
            return json.dumps({"summary": "scores", "measurements": {"mean_score": float(len(rows)), "slope": 1.0},
                               "key_findings": [{"finding": "scores rise", "evidence": "[1]", "confidence": "high"}]})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = Config(topic="does the score rise with visits", title="data-confirm-e2e", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, phased=True,
                                     no_simulation=True, web_derived_plots=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120),
                 knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path))
    engine = Engine(cfg)
    original = _visits(engine.quest_root / "data" / "d.csv")
    artifacts = await engine.run()
    record = phased.load(engine.quest_root)
    assert record["data_quest"] and record["strategy"] == phased.HELD_BACK, record
    assert phased.status(record) == phased.CONFIRMED, record
    assert record["isolation"] == phased_isolation.ENCRYPTED, record.get("isolation_why")
    (info,) = record["files"]
    assert info["rule"]["strategy"] == "group" and info["manifest"]["overlap"] == 0
    assert len(reads) == 2, "exploration's reading, then the confirm run's"
    explore_subjects = {line.split(",")[0] for line in reads[0].splitlines() if line[:2] in ("s0", "s1")}
    confirm_subjects = {line.split(",")[0] for line in reads[1].splitlines() if line[:2] in ("s0", "s1")}
    assert explore_subjects and confirm_subjects and explore_subjects.isdisjoint(confirm_subjects)
    assert (engine.quest_root / "data" / "d.csv").read_bytes() == original, "the whole file is back"
    routes = [(e.get("node"), e.get("chosen")) for e in audit_log.read(engine.fi_dir / "audit.jsonl")
              if e.get("kind") == "route_decision"]
    assert ("evidence_gate", "confirm_data") in routes
    ev = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert ev["phased"]["status"] == "confirmed" and ev["phased"]["isolation"] == "encrypted+scanned"
    paper = artifacts.paper_md.read_text(encoding="utf-8")
    assert "whole units picked at random" in paper
    assert "differ on the held-back rows" in paper, "the mean differs between the two parts and the paper says so"
