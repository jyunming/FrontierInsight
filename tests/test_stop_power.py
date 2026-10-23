"""What may end a quest: a problem FI measured (or two sources agree on) stops the run step; one that rests on FI's
reading of the code is recorded, said in QUALITY.md and the paper's limitations, and the quest goes on. Fake model only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import stop_power as sp
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.quality_report import build


def _engine(tmp_path: Path, **engine: Any) -> Engine:
    cfg = Config(
        topic="a neutral topic about a cooling cup", title="stop-power", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", oracle_check="off", **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    return eng


def test_only_a_measured_or_agreed_finding_may_stop() -> None:
    assert sp.may_stop(sp.MEASURED) and sp.may_stop(sp.AGREED)
    assert not sp.may_stop(sp.READ) and not sp.may_stop(None) and not sp.may_stop("model")
    assert "reading of the code" in sp.source_words(sp.READ)


def test_a_reading_of_the_code_is_recorded_and_the_run_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._record_cannot_repair("the analysis counts 50 runs, not the 200 the plan fixes", ["read the code"], source=sp.READ, check="a")
    assert eng._cannot_repair_stop() is None
    record = json.loads((eng.fi_dir / "cannot_repair.json").read_text(encoding="utf-8"))
    assert "text" not in record and record["problems"][0]["source"] == sp.READ
    note = eng._run_checks_note()
    assert "50 runs" in note and "limitations" in note and "not as a fact of the results" in note


def test_a_measured_problem_still_stops_the_run_step(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._record_cannot_repair("the reply did not hold both scripts", ["asked twice"], source=sp.MEASURED, check="b")
    stop = eng._cannot_repair_stop()
    assert stop is not None and "both scripts" in stop["exec_result"]["cannot_repair"]


@pytest.mark.parametrize("order", ["read_first", "measured_first"])
def test_a_reading_never_hides_a_measured_problem(tmp_path: Path, order: str) -> None:
    eng = _engine(tmp_path)
    found = [("a reading of the code", sp.READ), ("a measured fault", sp.MEASURED)]
    for text, source in (found if order == "read_first" else found[::-1]):
        eng._record_cannot_repair(text, ["tried"], source=source, check=text)
    stop = eng._cannot_repair_stop()
    assert stop is not None and stop["exec_result"]["cannot_repair"] == "a measured fault"
    assert "a reading of the code" in eng._run_checks_note()


def test_a_record_written_before_sources_were_kept_still_stops(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / "cannot_repair.json").write_text(json.dumps({"text": "an older problem", "tried": ["x"]}), encoding="utf-8")
    stop = eng._cannot_repair_stop()
    assert stop is not None and stop["exec_result"]["cannot_repair"] == "an older problem"


def test_quality_report_lists_a_reading_that_went_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._record_cannot_repair("a run count typed into the analysis", ["read the code"], source=sp.READ, check="a")
    report = build(eng.quest_root)
    items = [p for stage in report["stages"] for p in stage["problems"]["items"]]
    rows = [p for p in items if "run count typed" in p["text"]]
    assert rows and rows[0]["status"] == "not fixed" and "reading of the code" in rows[0]["found_by"]


def test_every_check_that_hands_a_problem_to_the_record_says_its_source() -> None:
    from core import grid_keys, independent_check, protocol_check, typed_constants
    for module in (grid_keys, independent_check, protocol_check, typed_constants):
        assert module.SOURCE in sp.SOURCES


def test_a_reading_of_the_same_text_never_replaces_a_measured_problem(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._record_cannot_repair("the same problem", ["measured"], source=sp.MEASURED, check="one")
    eng._record_cannot_repair("the same problem", ["read"], source=sp.READ, check="two")
    stop = eng._cannot_repair_stop()
    assert stop is not None and stop["exec_result"]["cannot_repair"] == "the same problem"


def test_a_check_that_finds_nothing_again_takes_its_own_problems_out(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._record_cannot_repair("a layout problem", ["read"], source=sp.READ, check="layout")
    eng._record_cannot_repair("a seed problem", ["read"], source=sp.READ, check="seed")
    eng._clear_cannot_repair("layout")
    assert [p["text"] for p in eng._cannot_repair_problems()] == ["a seed problem"]
    assert "a layout problem" not in eng._run_checks_note()
    eng._clear_cannot_repair("seed")
    assert eng._cannot_repair_problems() == [] and not (eng.fi_dir / "cannot_repair.json").exists()


def test_what_the_claim_check_still_flags_is_recorded_and_a_clean_pass_takes_it_out(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    assert eng._record_claims_left(["record_claims: a sentence"]) == ["record_claims: a sentence"]
    left = [p for p in eng._cannot_repair_problems() if p["check"] == "record_claims"]
    assert left and left[0]["source"] == sp.READ and eng._cannot_repair_stop() is None
    eng._record_claims_left([])
    assert eng._cannot_repair_problems() == []


def test_a_problem_found_on_an_earlier_version_of_the_code_is_said_as_not_checked_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    code = eng.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text("x = 1", encoding="utf-8")
    eng._record_cannot_repair("a seed problem", ["read"], source=sp.READ, check="seed")
    assert "repairs did not remove it: a seed problem" in eng._run_checks_note()
    (code / "simulate.py").write_text("x = 2", encoding="utf-8")  # repaired again; no check looked again
    note = eng._run_checks_note()
    assert "earlier version of the code found this: a seed problem" in note and "not known to be fixed" in note
    items = [p for stage in build(eng.quest_root)["stages"] for p in stage["problems"]["items"]]
    assert any(p["text"] == "a seed problem" and p["status"] == "not checked again after a later repair" for p in items)
