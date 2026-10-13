"""The evidence gate stops a quest whose code holds a fault that changes a reported number, that a failing test showed, and
whose test still fails on the code that produced the results. A reader's word alone, a test that now passes, a test that
cannot be run, and a quest without a simulation never stop it. Real tiny test files and the real test runner; no model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from core import code_findings as cf
from core import code_review
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

pytestmark = pytest.mark.usefixtures("this_python")

TYPED_IN = "def dose():\n    return 60.0 if True else 40.0\n"
COMPUTED = "def dose():\n    return 59.0\n"
FAILING = ("import experiment\n\n\ndef test_dose_is_not_typed_in():\n"
           "    assert experiment.dose() != 60.0, 'dose is the typed-in 60.0'\n")
PASSING = "import experiment\n\n\ndef test_dose_is_not_typed_in():\n    assert experiment.dose() >= 0\n"
STATE: dict[str, Any] = {"exec_result": {"returncode": 0}, "result_json": {"dose": 60.0}}


class _Sink(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(topic="a dose to size", title="dose", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code" / "experiment.py").write_text(TYPED_IN, encoding="utf-8")
    return engine


def _issue(fid: str = "F-1", **over: Any) -> dict[str, Any]:
    base = {"id": fid, "changes_result": True, "demonstrated": True, "outcome": cf.GAVE_UP, "plan_conflict": False,
            "readings": [{"n": 1, "status": "open"}], "severity": "high", "first_seen": 1, "file": "experiment.py",
            "function": "main", "text": "dose_to_size is set to a typed-in value", "test": f".fi/finding_tests/{fid}_1.py"}
    return {**base, **over}


def _record(engine: Engine, *issues: dict[str, Any], test: str | None = FAILING) -> None:
    if test is not None:
        folder = engine.quest_root / ".fi" / "finding_tests"
        folder.mkdir(parents=True, exist_ok=True)
        for i in issues:
            (engine.quest_root / i["test"]).write_text(test, encoding="utf-8")
    code_review.save(engine.quest_root, {"issues": {i["id"]: i for i in issues}})


@pytest.mark.asyncio
async def test_a_shown_open_fault_whose_test_still_fails_stops_the_quest_with_no_paper(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and verdict["stuck"] is True and verdict["stuck_reason"] == "shown_fault_open"
    assert verdict["verdict"] == "insufficient" and verdict["gaps"] == [verdict["rationale"]]
    assert "shown by a failing test" in verdict["rationale"] and "experiment.py main" in verdict["rationale"]
    assert "typed-in value" in verdict["rationale"]
    stuck = await engine._node_stuck_no_findings({**STATE, "evidence_assessment": verdict})
    assert "failing test" in stuck["stuck"]["problem"]
    assert (engine.quest_root / "needs" / "STUCK.json").is_file()
    assert not (engine.quest_root / "paper.md").exists()


@pytest.mark.asyncio
async def test_the_same_files_give_the_same_answer_again(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())
    assert await engine._shown_fault_verdict(STATE) == await engine._shown_fault_verdict(STATE)


@pytest.mark.asyncio
async def test_a_test_that_now_passes_on_the_final_code_stops_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())
    (engine.quest_root / "code" / "experiment.py").write_text(COMPUTED, encoding="utf-8")
    assert await engine._shown_fault_verdict(STATE) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("over", [
    {"demonstrated": False}, {"demonstrated": None}, {"changes_result": False}, {"plan_conflict": True},
    {"outcome": cf.FIXED}, {"outcome": cf.GONE}, {"readings": [{"n": 1, "status": "open"}, {"n": 2, "status": "not flagged"}]},
])
async def test_an_issue_that_is_not_shown_open_and_result_changing_is_never_run(
        tmp_path: Path, over: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(**over))
    ran: list[Any] = []

    async def spy(argv, **kw):  # noqa: ANN001
        ran.append(argv)
        return 0, "", ""

    monkeypatch.setattr(Engine, "_run_in_env", spy)
    assert await engine._shown_fault_verdict(STATE) is None
    assert ran == [], "no test is run for it"


@pytest.mark.asyncio
async def test_a_test_that_cannot_run_stops_nothing_and_says_so(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    broken = "import module_that_is_not_there\n\n\ndef test_x():\n    assert False\n"
    _record(engine, _issue(), test=broken)
    sink = _Sink()
    engine._log.addHandler(sink)
    try:
        assert await engine._shown_fault_verdict(STATE) is None
    finally:
        engine._log.removeHandler(sink)
    assert any("could not be run again" in m for m in sink.messages)


@pytest.mark.asyncio
async def test_a_missing_test_file_stops_nothing_and_is_warned_about(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=None)
    sink = _Sink()
    engine._log.addHandler(sink)
    try:
        assert await engine._shown_fault_verdict(STATE) is None
    finally:
        engine._log.removeHandler(sink)
    assert any("not on disk" in m for m in sink.messages)


@pytest.mark.asyncio
async def test_a_quest_without_a_simulation_or_without_a_run_is_not_judged(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())
    assert await engine._shown_fault_verdict({**STATE, "no_simulation_resolved": True}) is None
    assert await engine._shown_fault_verdict({"result_json": {}}) is None
    assert await engine._shown_fault_verdict({**STATE, "exec_result": {}}) is None


@pytest.mark.asyncio
async def test_only_the_most_urgent_few_are_run_and_a_later_failing_one_still_stops(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    many = [_issue(f"F-{n}", first_seen=n) for n in range(1, 8)]
    _record(engine, *many)
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and verdict["stuck_reason"] == "shown_fault_open"
    assert len(cf.shown_open({"issues": {i["id"]: i for i in many}})[: cf.MAX_SHOWN_RERUNS]) == cf.MAX_SHOWN_RERUNS


def test_shown_open_ranks_result_changing_high_severity_first() -> None:
    a = _issue("F-a", severity="low", first_seen=1)
    b = _issue("F-b", severity="high", first_seen=2)
    assert [i["id"] for i in cf.shown_open({"issues": {"F-a": a, "F-b": b}})] == ["F-b", "F-a"]
